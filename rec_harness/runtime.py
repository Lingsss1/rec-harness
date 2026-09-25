"""Sequential research controller. Agents propose; the controller commits facts."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .agents import CodexAgent, run_process
from .core import Artifacts, ContractError, digest, file_hash, identifier, number, public_only, read_json, required, safe_file, validate_card, write_json
from .store import Store
from .evolution import validate_policy
from .measurement_gaps import MeasurementDeferred


def edge(edge_id, source, target, relation, evidence, task, revision):
    return {"id": edge_id, "source": source, "target": target, "relation": relation,
            "attrs": {"scope": task, "evidence_refs": evidence, "origin": "local_evidence", "status": "provisional",
                      "created_at": str(time.time()), "revision_id": str(revision)}}


def node(node_id, kind, attrs, task, protocol, origin="local_evidence"):
    return {"id": node_id, "type": kind, "attrs": {**attrs, "task_id": task, "protocol_hash": protocol, "origin": origin, "visibility": "public"}}


class Research:
    def __init__(self, root, agent=None):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / "campaign.json")
        self.protocol = self.config["protocol"]
        self.protocol_hash = digest(self.protocol)
        if self.protocol_hash != self.config["protocol_hash"]:
            raise ContractError("frozen protocol was changed")
        self.task = self.config["task_id"]
        self.store = Store(self.root / "memory")
        self.artifacts = Artifacts(self.root / "artifacts" / "objects")
        if agent is not None:
            self.agent=agent
        elif self.config.get('agent_backend',{}).get('kind')=='jailed_codex':
            from .jailed_agent import JailedCodexAgent
            backend=self.config['agent_backend'];options=self.config.get('codex',{})
            self.agent=JailedCodexAgent(backend['template'],backend['runtime_root'],
                model=options.get('model'),timeout=options.get('timeout',300),web_search=options.get('web_search'),prompts=self.config.get('research_policy',{}).get('prompts'))
        else:
            self.agent=CodexAgent(**self.config.get("codex", {}), prompts=self.config.get("research_policy", {}).get("prompts"))
        self.store.configure_budget(self.config["budget"])
        manifest = read_json(self.root / "research_archive" / "manifest.json")
        if manifest["config_hash"] != digest(self.config):
            raise ContractError("campaign config differs from frozen manifest")
        self.state_path = self.root / "state.json"

    @classmethod
    def init(cls, root, configuration, *, _validation_tool=None):
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise ContractError("campaign directory must be empty")
        config = json.loads(json.dumps(configuration))
        required(config, ["task_id", "source", "model_files", "protocol", "budget"])
        identifier(config["task_id"])
        public_only(config)
        from .measurement_gaps import validate_requirements
        validate_requirements(config.get('measurement_requirements',[]))
        from .measurement_discovery import validate_config
        validate_config(config.get('measurement_discovery'))
        if 'diagnostic_tool' in config:raise ContractError('diagnostic_tool is controller-owned; supply diagnostic_registry')
        if config.get('execution',{}).get('kind') not in {None,'isolated-toy-v1'}:
            raise ContractError('unsupported execution backend; refusing implicit fallback')
        if config.get('agent_backend',{}).get('kind') not in {None,'codex','jailed_codex'}:
            raise ContractError('unsupported agent backend; refusing implicit fallback')
        if 'harness_registry' in config:
            from .harness_activation import load_active_policy
            registry = config.pop('harness_registry')
            policy, provenance = load_active_policy(registry, config.pop('harness_identity', None))
            if config.get('research_policy'):
                raise ContractError('promoted harness cannot be mixed with local policy overrides')
            config['research_policy'] = policy
            config['harness_activation'] = provenance
        if config.get('memory_presentation', 'graph') not in {'graph', 'flat'}:
            raise ContractError('memory_presentation must be graph or flat')
        protocol = config["protocol"]
        required(protocol, ["seeds", "metric", "direction", "minimum_effect", "evaluate_command", "check_command", "timeout_seconds", "evaluation_cost", "agent_cost", "data_version", "split", "guardrails"])
        if protocol["direction"] not in {"maximize", "minimize"}:
            raise ContractError("invalid metric direction")
        if not protocol["seeds"] or len(protocol["seeds"]) != len(set(protocol["seeds"])) or any(type(s) is not int for s in protocol["seeds"]):
            raise ContractError("seeds must be distinct integers")
        for field in ("minimum_effect", "timeout_seconds", "evaluation_cost", "agent_cost"):
            number(protocol[field], field, 0)
        if protocol["timeout_seconds"] <= 0:
            raise ContractError("timeout must be positive")
        for field in ("evaluate_command", "check_command"):
            command = protocol[field]
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ContractError("commands must be argv arrays")
        for name in config["model_files"]:
            if any(part in {"trusted", ".git", ".env"} for part in Path(name).parts):
                raise ContractError("non-public model path")
        root.mkdir(parents=True, exist_ok=True)
        artifacts = Artifacts(root / "artifacts" / "objects")
        from .memory_policy import validate_package
        if 'initial_memory' in config:raise ContractError('initial_memory is controller-owned; supply initial_memory_files')
        config['initial_memory']=[]
        for path in config.pop('initial_memory_files',[]):
            validate_package(read_json(path));config['initial_memory'].append(artifacts.put(path))
        if 'diagnostic_registry' in config:
            from .diagnostics import freeze_tool
            config['diagnostic_tool']=freeze_tool(config.pop('diagnostic_registry'),artifacts)
        if _validation_tool is not None:
            if 'diagnostic_tool' in config:raise ContractError('validation tool cannot override an active registry')
            from .diagnostics import freeze_validation_tool
            config['diagnostic_tool']=freeze_validation_tool(_validation_tool,artifacts)
            config['validation_only']=True
        snapshot = artifacts.snapshot(config["source"], config["model_files"])
        trusted = config.pop("evaluator_files", [])
        evaluator = artifacts.snapshot(config["source"], trusted)
        artifacts.restore(evaluator, root / "evaluator")
        config["protocol"]["evaluator_snapshot"] = evaluator
        config["protocol_hash"] = digest(config["protocol"])
        config["source"] = "archived"
        validate_policy(config.get("research_policy", {}))
        config["harness_version"] = "0.1.0:" + digest(config.get("research_policy", {}))[:12]
        config.setdefault("max_rounds", 1)
        if type(config['max_rounds']) is not int or config['max_rounds']<1:
            raise ContractError('max_rounds must be a positive lifetime limit')
        if type(config.get('max_role_retries',0)) is not int or not 0<=config.get('max_role_retries',0)<=3:
            raise ContractError('max_role_retries must be an integer from0 to3')
        if config.get('execution',{}).get('kind')=='isolated-toy-v1':
            if protocol['metric']!='quality' or set(protocol['guardrails'])!={'serialized_coefficients'}:
                raise ContractError('isolated toy uses host quality and serialized_coefficients guardrail')
            config['execution']['host_scorer_hash']=file_hash(Path(__file__).with_name('toy_isolated.py'))
            config['execution']['worker_hash']=file_hash(Path(__file__).with_name('toy_worker.py'))
            config['single_incumbent_control']=True
        config.setdefault("codex", {})
        write_json(root / "campaign.json", config)
        write_json(root / "research_archive" / "manifest.json", {
            "config_hash": digest(config), "task_id": config["task_id"], "protocol_hash": config["protocol_hash"],
            "root_workspace": snapshot, "harness_version": "0.1.0", "concurrency": "synchronous-sequential", "worker_limit": 1,
            "base_llm": config["codex"].get("model", "local_codex_configuration"), "retention": "lossless-content-addressed", "qualification": "audit_only",
        })
        write_json(root / "state.json", {"round": 0, "incumbent": "root", "workspace": snapshot, "baseline": None, "status": "ready"})
        instance = cls(root)
        instance.store.event("campaign_created", {"config_hash": digest(config), "protocol_hash": config["protocol_hash"]})
        instance.store.patch({"base_version": 0, "nodes": [node("root", "ModelVersion", {"code_hash": digest(snapshot), "weight_source": "independent_initialization"}, config["task_id"], config["protocol_hash"])], "edges": []})
        instance.store.close()
        return root

    @contextmanager
    def lock(self):
        from .process_identity import assert_stopped
        with (self.root / ".controller.lock").open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ContractError("campaign already running") from error
            try:
                assert_stopped(self.root / 'tasks')
                assert_stopped(self.root / 'research_archive')
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _archive_directory(self, path):
        path=Path(path);files=[]
        for parent,dirs,names in os.walk(path,followlinks=False):
            relative=Path(parent).relative_to(path)
            # CPU-job receipts retain declared inputs and collected outputs. Runtime staging
            # is infrastructure, not a second copy of the candidate's research artifacts.
            parts=relative.parts
            if len(parts)>=2 and parts[-2]=='isolated':dirs[:]=[d for d in dirs if d!='jail']
            files.extend(str((Path(parent)/name).relative_to(path)) for name in names
                         if (Path(parent)/name).is_file() and not (Path(parent)/name).is_symlink())
        return self.artifacts.snapshot(path,files)

    def _agent(self, role, context, attempt, snapshot=None):
        from .memory_view import present
        if role=='knowledge':
            bundle=context['evidence']
            context={**context,'graph_patch_contract':{
                'base_version':bundle['graph_snapshot_id'],'exact_fields':['base_version','nodes','edges'],
                'allowed_node_types':['Claim','GapHypothesis','MeasurementGap'],
                'required_attrs':{'task_id':self.task,'protocol_hash':self.protocol_hash,'origin':'agent_hypothesis'},
                'claim_attrs':{'status':'provisional','scope':'this task and frozen protocol only','evidence_refs':'IDs from allowed_evidence_refs only'},
                'allowed_evidence_refs':[n['id'] for n in bundle['nodes']],
                'instructions':'Append new IDs only. No fact edits. All evidence_refs are existing graph node IDs, never file paths. Empty nodes and edges are valid.'}}
        if 'evidence' in context:
            context = {**context, 'evidence': present(context['evidence'],
                       self.config.get('memory_presentation', 'graph'))}
        count = len(list((self.root / "tasks").glob("*"))) if (self.root / "tasks").exists() else 0
        task_id = f"{attempt}-{role}-{count}"
        directory = self.root / "tasks" / task_id
        self.store.reserve(task_id, self.protocol["agent_cost"])
        started = time.time()
        status = "crash"
        try:
            directory.mkdir(parents=True)
            if snapshot:
                self.artifacts.restore(snapshot, directory / "model")
            public_only(context)
            frozen = digest(context)
            write_json(directory/'visible-state.json',context)
            self.store.event("agent_submitted", {"task_id": task_id, "role": role, "input_hash": frozen, "model_version": context.get("model_version"), "graph_version": self.store.version,
                'visible_state_ref':self.artifacts.put(directory/'visible-state.json'),'source_snapshot':snapshot or {}})
            response = self.agent.run(role, context, directory)
            public_only(response)
            status = "success"
            return response, directory
        finally:
            # Agent calls use predeclared BCU; token and wall costs are separate measured fields.
            self.store.settle(task_id, self.protocol["agent_cost"], status)
            self.store.event("agent_completed", {"task_id": task_id, "status": status, "wall_seconds": time.time() - started,
                                                   "archive": self._archive_directory(directory) if directory.exists() else {}})

    def context(self, role, state):
        from .diagnostics import measure
        from .measurement_gaps import triage
        diagnostic=measure(self,state) if role=='evidence' else None
        measurement_gaps=[];measurement_failure=None
        if role=='evidence':
            try:measurement_gaps=triage(self,state,diagnostic)
            except MeasurementDeferred as error:measurement_failure=str(error)
        return {"task_id": self.task, "model_version": state["incumbent"], "protocol_hash": self.protocol_hash,
                "protocol": self.protocol, "budget": self.store.budget(), "harness_version": self.config["harness_version"],
                "evidence": self.store.bundle(self.task, self.protocol_hash, role, limit=self.config.get("research_policy", {}).get("retrieval_limit", 24)), "baseline": state["baseline"],
                'execution_contract':self.config.get('task_contract',{}),
                'diagnostic':diagnostic,
                'measurement_gaps':measurement_gaps,
                'measurement_failure':measurement_failure,
                'supported_controls':[state['incumbent']] if self.config.get('single_incumbent_control') else None}

    def _command(self, command, model, output, seed):
        mapping = {"python": sys.executable, "model": str(model), "output": str(output), "seed": str(seed), "evaluator": str(self.root / "evaluator")}
        try:
            return [arg.format_map(mapping) for arg in command]
        except (KeyError, ValueError) as error:
            raise ContractError(f"invalid command placeholder: {error}") from error

    def evaluate(self, attempt, snapshot, baseline=None):
        runs = []
        existing = {n["id"] for n in self.store.snapshot()["nodes"]}
        if attempt not in existing:
            self.store.patch({"base_version": self.store.version, "nodes": [node(attempt, "ModelVersion", {
                "code_hash": digest(snapshot), "weight_source": "independent_initialization", "evidence_dependency_ids": [attempt]}, self.task, self.protocol_hash)], "edges": []})
        for seed in self.protocol["seeds"]:
            run_id = f"{attempt}-seed-{seed}"
            identifier(run_id)
            directory = self.root / "research_archive" / "attempts" / attempt / "runs" / run_id
            directory.mkdir(parents=True, exist_ok=False)
            model = directory / "model"
            self.artifacts.restore(snapshot, model)
            task_id = f"eval-{run_id}"
            self.store.reserve(task_id, self.protocol["evaluation_cost"])
            output = directory / "metrics.json"
            result = {"run_id": run_id, "seed": seed, "valid": False, "score": None, "status": "crash", "resources": {"gpu_seconds": None}}
            try:
                # The configured evaluator is frozen and never copied into the agent's writable model.
                for name, ref in self.protocol["evaluator_snapshot"].items():
                    if file_hash(safe_file(self.root / "evaluator", name)) != ref["sha256"]:
                        raise ContractError("evaluator modified")
                isolated=self.config.get('execution',{}).get('kind')=='isolated-toy-v1'
                if isolated:
                    from .toy_isolated import evaluate as isolated_evaluate
                    backend=self.config['execution']
                    if any(file_hash(Path(__file__).with_name(name+'.py'))!=backend[field] for name,field in
                           [('toy_isolated','host_scorer_hash'),('toy_worker','worker_hash')]):
                        raise ContractError('frozen isolated evaluator changed')
                    execution=isolated_evaluate(model,output,seed,backend,self.artifacts,self.protocol['timeout_seconds'])
                    checked={'status':execution['status']}
                else:
                    checked = run_process(self._command(self.protocol["check_command"], model, output, seed), model, directory / "check", timeout=self.protocol["timeout_seconds"])
                if checked["status"] != "success":
                    result.update(status="invalid", failure_kind="correctness_" + checked["status"])
                else:
                    if not isolated:
                        execution = run_process(self._command(self.protocol["evaluate_command"], model, output, seed), model, directory / "evaluation", timeout=self.protocol["timeout_seconds"])
                    result["resources"] = execution
                    if execution["status"] != "success":
                        result.update(status="crash", failure_kind=execution["status"])
                    else:
                        for name, ref in self.protocol["evaluator_snapshot"].items():
                            if file_hash(safe_file(self.root / "evaluator", name)) != ref["sha256"]:
                                raise ContractError("evaluator modified during execution")
                        metrics = read_json(output)
                        public_only(metrics)
                        required(metrics, [self.protocol["metric"]])
                        score = number(metrics[self.protocol["metric"]], "public score")
                        guardrails = all(name in metrics and number(metrics[name], name) <= number(maximum, name) for name, maximum in self.protocol["guardrails"].items())
                        result.update(status="success" if guardrails else "invalid", valid=guardrails, score=score, metrics=metrics,
                                      failure_kind=None if guardrails else "guardrail")
            except (ContractError, OSError, ValueError) as error:
                result.update(status="invalid", failure_kind="contract", error=str(error))
            finally:
                self.store.settle(task_id, self.protocol["evaluation_cost"], result["status"])
                result["artifact_refs"] = self._archive_directory(directory)
                write_json(directory / "result.json", result)
                self.store.event("evaluation_completed", result)
            runs.append(result)
            paired=next((r for r in baseline['runs'] if r['seed']==seed),None) if baseline else None
            sign=1 if self.protocol['direction']=='maximize' else -1
            effect_passed=bool(paired and result['valid'] and paired['valid'] and sign*(result['score']-paired['score'])>self.protocol['minimum_effect'])
            attrs = {**result, "independent_run_id": run_id, "code_hash": digest(snapshot), "effect_gate_passed": effect_passed,
                     "evidence_dependency_ids": [attempt], "weight_source": "independent_initialization"}
            observation_id = run_id + "-observation"
            nodes = [node(run_id, "Experiment", attrs, self.task, self.protocol_hash),
                     node(observation_id, "Observation", {"metrics": result.get("metrics", {}), "valid": result["valid"],
                          "artifact_refs": result["artifact_refs"], "evidence_dependency_ids": [attempt]}, self.task, self.protocol_hash)]
            edges = [edge(run_id + "-evaluates", run_id, attempt, "EVALUATES", [run_id], self.task, self.store.version + 1),
                     edge(run_id + "-produces", run_id, observation_id, "PRODUCES", [run_id], self.task, self.store.version + 1)]
            if not result["valid"]:
                failure_id = run_id + "-failure"
                nodes.append(node(failure_id, "Failure", {"failure_kind": result.get("failure_kind"), "evidence_dependency_ids": [attempt]}, self.task, self.protocol_hash))
                edges.append(edge(run_id + "-has-failure", run_id, failure_id, "HAS_FAILURE", [run_id], self.task, self.store.version + 1))
            self.store.patch({"base_version": self.store.version, "nodes": nodes, "edges": edges})
        valid = all(r["valid"] for r in runs)
        return {"attempt": attempt, "valid": valid, "runs": runs, "score": statistics.mean(r["score"] for r in runs) if valid else None}

    def _record_attempt(self, attempt, state, before, after, evaluation, decision, dependencies):
        directory = self.root / "research_archive" / "attempts" / attempt
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "node.json", {"node_id": attempt, "primary_parent_id": state["incumbent"], "branch_id": attempt if state["incumbent"] == "root" else state["incumbent"],
                   "creation_seq": state["round"] + 1, "dependencies": dependencies, "workspace_before": before, "workspace_after": after,
                   "weight_source": "independent_initialization", "protocol_hash": self.protocol_hash})
        write_json(directory / "observations.json", evaluation)
        write_json(directory / "decision.json", decision)
        write_json(directory / "artifacts.json", self._archive_directory(directory))
        self.store.event("attempt_archived", {"node_id": attempt, "archive": self._archive_directory(directory)})
        from .decision_traces import project_traces
        project_traces(self)

    def consolidate(self, attempt):
        identifier(attempt)
        with self.lock():
            directory = self.root / "research_archive" / "attempts" / attempt
            evaluation = read_json(directory / "observations.json")
            decision = read_json(directory / "decision.json")
            state = read_json(self.state_path)
            context = {**self.context("knowledge", state), "evaluation": evaluation, "decision": decision,
                       "proposal": read_json(self.root / "research_archive" / "decisions" / attempt / "action.json")}
            if not self._knowledge(attempt,context):raise ContractError('knowledge retry exhausted')
            return {"attempt": attempt, "graph_version": self.store.version, "budget": self.store.budget()}

    def _knowledge(self,attempt,context):
        from .knowledge import validate_knowledge_patch
        for retry in range(self.config.get('max_role_retries',0)+1):
            if self.store.budget()['remaining']<self.protocol['agent_cost']:break
            try:
                response,_=self._agent('knowledge',context,attempt)
                patch=json.loads(response['patch_json']);bundle=context['evidence']
                validate_knowledge_patch(patch,{'graph_version':bundle['graph_snapshot_id'],
                    'task_id':self.task,'protocol_hash':self.protocol_hash,'nodes':bundle['nodes']})
                self.store.patch(patch)
                self.store.event('knowledge_consolidated',{'attempt':attempt,'graph_version':self.store.version,'retry':retry})
                return True
            except (ContractError,OSError,ValueError,KeyError,TypeError) as error:
                self.store.event('knowledge_rejected',{'attempt':attempt,'retry':retry,'reason':str(error)})
                context={**context,'validation_error':str(error),
                         'repair_instruction':'Correct only the rejected graph patch using the supplied graph contract. Do not rerun training.'}
        self.store.event('knowledge_deferred',{'attempt':attempt,'reason':'retry or budget limit'})
        return False

    def verify(self):
        result = self.store.verify()
        checked = set()
        def visit(value):
            if isinstance(value, dict):
                if "sha256" in value and "size" in value:
                    self.artifacts.verify(value)
                    checked.add(value["sha256"])
                else:
                    for child in value.values():
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(self.config)
        visit(read_json(self.root / "research_archive" / "manifest.json"))
        visit(read_json(self.state_path))
        for event in self.store.events():
            visit(event["payload"])
        result["verified_artifacts"] = len(checked)
        return result

    def run(self, rounds=None):
        with self.lock():
            interrupted = self.store.recover()
            from .memory_policy import _import_memory
            for ref in self.config.get('initial_memory',[]):
                _import_memory(self.store,self.artifacts.verify(ref),self.task,self.protocol_hash)
            if interrupted:
                self.store.event("recovery", {"interrupted_task_ids": interrupted, "cost_policy": "retain reservation when actual unknown"})
            state = read_json(self.state_path)
            if state.get('status')=='stopped' and state.get('stop_reason') in {'budget','max_rounds','measurement_required'}:
                return self._finish(state,state['stop_reason'])
            events = self.store.events()
            completed = {e["payload"]["attempt"] for e in events if e["kind"] == "round_completed"}
            for event in events:
                if event["kind"] == "round_completed" and "state" in event["payload"]:
                    saved = event["payload"]["state"]
                    if saved["round"] > state["round"]:
                        state = saved
                        write_json(self.state_path, state)
                if event["kind"] == "round_started" and event["payload"]["attempt"] not in completed:
                    old = event["payload"]["state"]
                    attempt = event["payload"]["attempt"]
                    archived = self.root / "research_archive" / "attempts" / attempt
                    if all((archived / name).exists() for name in ("node.json", "decision.json", "observations.json")):
                        decision = read_json(archived / "decision.json")
                        if decision["selection"] == "keep":
                            state.update(incumbent=attempt, workspace=read_json(archived / "node.json")["workspace_after"], baseline=read_json(archived / "observations.json"))
                    else:
                        decision = {"selection": "crash", "conclusion_status": "unknown", "rationale": "controller interrupted; raw tasks and consumed costs retained"}
                        self._record_attempt(attempt, old, old["workspace"], old["workspace"], {"attempt": attempt, "valid": False, "score": None, "runs": []}, decision, [])
                    state["round"] = max(state["round"], old["round"] + 1)
                    self.store.event("round_completed", {"attempt": attempt, "decision": decision, "state": state, "state_hash": digest(state)})
                    write_json(self.state_path, state)
            if state.get('stop_reason')=='measurement_required':return self._finish(state,'measurement_required')
            if state["baseline"] is None:
                if self.store.budget()['remaining']<len(self.protocol['seeds'])*self.protocol['evaluation_cost']:
                    return self._finish(state,'budget')
                # Each retry is a new immutable attempt, never overwriting a partial baseline.
                suffix = len(list((self.root / "research_archive" / "attempts").glob("baseline-*")))
                baseline_id = f"baseline-{suffix}"
                state["baseline"] = self.evaluate(baseline_id, state["workspace"])
                if not state["baseline"]["valid"]:
                    raise ContractError("baseline failed frozen checks; see archived runs")
                write_json(self.state_path, state)
            rounds = max(0,self.config["max_rounds"]-state['round']) if rounds is None else rounds
            if type(rounds) is not int or rounds < 0:
                raise ContractError("rounds must be a nonnegative integer")
            rounds=min(rounds,max(0,self.config['max_rounds']-state['round']))
            stop_reason='round_limit'
            for _ in range(rounds):
                needed = (6+int('measurement_discovery' in self.config)+self.config.get('max_role_retries',0)) * self.protocol["agent_cost"] + len(self.protocol["seeds"]) * self.protocol["evaluation_cost"]+self.config.get('diagnostic_tool',{}).get('cost',0)
                if self.store.budget()["remaining"] < needed:
                    self.store.event("stopped", {"reason": "budget", "minimum_round_cost": needed})
                    stop_reason='budget'
                    break
                attempt = f"attempt-{len([e for e in self.store.events() if e['kind'] == 'round_started']) + 1}"
                self.store.event("round_started", {"attempt": attempt, "state": state})
                context = self.context("evidence", state)
                before = state["workspace"]
                dependencies = [n["id"] for n in context["evidence"]["nodes"] if n["type"] == "ModelVersion" and n["id"] != "root"]
                decision_dir = self.root / "research_archive" / "decisions" / attempt
                write_json(decision_dir / "state.json", {"incumbent": state["incumbent"], "revealed_node_ids": dependencies + ["root"], "pending": [], "budget": self.store.budget()})
                write_json(decision_dir / "context.json", context)
                evaluation = {"attempt": attempt, "valid": False, "score": None, "runs": []}
                decision = {"selection": "crash", "conclusion_status": "unknown", "rationale": "incomplete", "evidence_refs": [], "unresolved_explanations": []}
                after = before
                try:
                    if context['measurement_failure']:raise MeasurementDeferred(context['measurement_failure'])
                    gap, _ = self._agent("evidence", context, attempt, before)
                    validate_card("gap", gap)
                    if gap["model_version"] != state["incumbent"] or gap["protocol_hash"] != self.protocol_hash:
                        raise ContractError("stale gap input identity")
                    from .measurement_gaps import block_model_proposal
                    if not context['measurement_gaps']:
                        from .measurement_discovery import discover
                        context['measurement_gaps']=discover(self,state,context,gap,attempt,
                            lambda request:self._agent('measurement',request,attempt,before)[0])
                    block_model_proposal(self,attempt,gap,context['measurement_gaps'])
                    generated, _ = self._agent("hypothesis", {**self.context('hypothesis', state), "gap": gap}, attempt, before)
                    method, proposal = generated["method"], generated["proposal"]
                    validate_card("method", method)
                    validate_card("proposal", proposal)
                    if proposal["base_model_id"] != state["incumbent"] or proposal["train_protocol_hash"] != self.protocol_hash or proposal["seeds"] != self.protocol["seeds"]:
                        raise ContractError("proposal changed frozen model/protocol/seeds")
                    if gap["gap_id"] not in proposal["gap_ids"] or method["method_id"] not in proposal["method_ids"]:
                        raise ContractError("proposal references unknown gap/method")
                    if proposal["budget"] > self.store.budget()["remaining"]:
                        raise ContractError("proposal over budget")
                    if self.config.get('single_incumbent_control') and proposal['controls']!=[state['incumbent']]:
                        raise ContractError('this adapter requires the existing incumbent as its sole control')
                    write_json(decision_dir / "action.json", proposal)
                    self.store.patch({"base_version": self.store.version, "nodes": [
                        node(attempt + "-gap", "GapHypothesis", gap, self.task, self.protocol_hash, "agent_hypothesis"),
                        node(attempt + "-method", "Method", method, self.task, self.protocol_hash, method["origin"]),
                        node(attempt + "-proposal", "Proposal", proposal, self.task, self.protocol_hash, "agent_hypothesis")], "edges": [
                        edge(attempt + "-targets", attempt + "-proposal", attempt + "-gap", "TARGETS", [], self.task, self.store.version + 1),
                        edge(attempt + "-uses", attempt + "-proposal", attempt + "-method", "USES", [], self.task, self.store.version + 1)]})
                    implementation, directory = self._agent("experiment", {**self.context('experiment', state), "gap": gap, "method": method, "proposal": proposal}, attempt, before)
                    # All files, including newly added source, are retained; only allowlisted source
                    # or explicitly reported changes enter the next model snapshot.
                    paths = set(before) | set(implementation["changed_files"])
                    if any(name.startswith("model/") for name in implementation["changed_files"]):
                        raise ContractError("changed_files must be relative to model/")
                    after = self.artifacts.snapshot(directory / "model", paths)
                    evaluation = self.evaluate(attempt, after, state["baseline"])
                    sign = 1 if self.protocol["direction"] == "maximize" else -1
                    delta = sign * (evaluation["score"] - state["baseline"]["score"]) if evaluation["valid"] else None
                    eligible = delta is not None and delta > self.protocol["minimum_effect"]
                    comparison={'public_delta':delta,'effect_gate_passed':eligible,'minimum_effect':self.protocol['minimum_effect'],
                                'scope':'host-computed development selection; valid runs have passed correctness and resource guards'}
                    eval_context = {**self.context('evaluation', state), "gap": gap, "proposal": proposal, "evaluation": evaluation,'comparison':comparison}
                    recommendation, _ = self._agent("evaluation", eval_context, attempt)
                    validate_card("decision", recommendation)
                    lead, _ = self._agent("lead", {**eval_context, **self.context('lead', state), "recommendation": recommendation}, attempt, after)
                    validate_card("decision", lead)
                    sign = 1 if self.protocol["direction"] == "maximize" else -1
                    delta = sign * (evaluation["score"] - state["baseline"]["score"]) if evaluation["valid"] else None
                    eligible = delta is not None and delta > self.protocol["minimum_effect"]
                    selection = "keep" if eligible and lead["selection"] == "keep" else "discard" if evaluation["valid"] else "invalid"
                    if lead["selection"] == "provisional" and evaluation["valid"]:
                        selection = "provisional"
                    decision = {**lead, "selection": selection, "conclusion_status": "provisional" if evaluation["valid"] else "unknown",
                                "public_delta": delta, "effect_gate_passed": eligible, "agent_recommendation": lead["selection"]}
                    self.store.patch({"base_version": self.store.version, "nodes": [node(attempt, "ModelVersion", {
                        "code_hash": digest(after), "design_parent": state["incumbent"], "weight_source": "independent_initialization",
                        "decision": selection, "evidence_dependency_ids": [attempt]}, self.task, self.protocol_hash)], "edges": [
                        edge(attempt + "-derived", attempt, state["incumbent"], "DERIVED_FROM", [], self.task, self.store.version + 1)]})
                    knowledge_context = {**eval_context, **self.context('knowledge', state), "decision": decision}
                    self._knowledge(attempt,knowledge_context)
                except MeasurementDeferred as error:
                    decision={'selection':'provisional','conclusion_status':'unknown','rationale':str(error),
                              'evidence_refs':[g['measurement_gap_id'] for g in context['measurement_gaps']],
                              'unresolved_explanations':['measurement or execution prerequisite unresolved'],
                              'measurement_required':True}
                    self.store.event('model_research_deferred',{'attempt':attempt,'decision':decision})
                except (ContractError, OSError, ValueError, KeyError) as error:
                    decision = {"selection": "crash", "conclusion_status": "unknown", "rationale": str(error), "evidence_refs": [], "unresolved_explanations": ["execution failure; no mechanism refutation"]}
                    self.store.event("round_failed", {"attempt": attempt, "error": str(error)})
                    self.store.patch({"base_version": self.store.version, "nodes": [node(attempt + "-failure", "Failure", {"error": str(error), "evidence_dependency_ids": [attempt]}, self.task, self.protocol_hash)], "edges": []})
                self._record_attempt(attempt, state, before, after, evaluation, decision, dependencies)
                if decision["selection"] == "keep":
                    state.update(incumbent=attempt, workspace=after, baseline=evaluation)
                    self.store.event("gaps_need_recheck", {"new_model": attempt, "previous_model": context["model_version"]})
                state.update(round=state["round"] + 1, status="ready")
                if decision.get('measurement_required'):state.update(status='stopped',stop_reason='measurement_required')
                self.store.event("round_completed", {"attempt": attempt, "decision": decision, "state_hash": digest(state), "state": state})
                write_json(self.state_path, state)
                if decision.get('measurement_required'):return self._finish(state,'measurement_required')
            return self._finish(state,'max_rounds' if state['round']>=self.config['max_rounds'] else stop_reason)

    def _finish(self,state,reason):
        state={**state,'status':'stopped','stop_reason':reason}
        if read_json(self.state_path)!=state:
            self.store.event('controller_stopped',{'state':state,'reason':reason})
            write_json(self.state_path,state)
        self.store.export_compat(self.root/'memory'/'exports')
        report={'state':state,'budget':self.store.budget(),'integrity':self.store.verify()}
        write_json(self.root/'run-report.json',report)
        return report
