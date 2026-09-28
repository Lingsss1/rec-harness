"""Precommitted validation and promotion for tools and research-layer policies."""
from __future__ import annotations

import json
import sys
import fcntl
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from .process_identity import assert_stopped

from .agents import CodexAgent, run_process, schema_for, validate_schema
from .core import Artifacts, ContractError, digest, identifier, number, file_hash, public_only, read_json, required, validate_card, write_json
from .store import Store
from .validation_stage import ValidationRecoveryRequired

STAGES = {
    "tool": ["correctness", "synthetic", "robustness", "shadow", "utility", "cost_regression"],
    "harness": ["prefix_audit", "development", "confirmation"],
}
EDITABLE = {"prompts", "retrieval_limit", "routing"}


class ValidationGateRejected(ContractError):
    """A well-formed independent report failed its frozen numerical gates."""
    def __init__(self,message,failed_gates=()):
        super().__init__(message)
        self.failed_gates=list(failed_gates)


def numerical_gate_failures(report,rule):
    gain=number(report['gain'],'gain')
    regression=number(report['regression'],'regression',0)
    cost=number(report['total_cost'],'total_cost',0)
    return [name for name,failed in [
        ('minimum_gain',gain<rule['minimum_gain']),
        ('maximum_regression',regression>rule['maximum_regression']),
        ('maximum_total_cost',cost>rule['maximum_total_cost'])] if failed]


def serialized(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.lock():return method(self,*args,**kwargs)
    return call


def validate_policy(policy):
    if not isinstance(policy, dict) or not set(policy).issubset(EDITABLE):
        raise ContractError("patch attempts to modify stable control kernel")
    if "retrieval_limit" in policy and (type(policy["retrieval_limit"]) is not int or not 1 <= policy["retrieval_limit"] <= 100):
        raise ContractError("retrieval limit must be 1..100")
    if "prompts" in policy and (not isinstance(policy["prompts"], dict) or not all(isinstance(v, str) for v in policy["prompts"].values())):
        raise ContractError("prompts must map roles to text")
    from .agents import ROLES
    if not set(policy.get('prompts',{})).issubset(ROLES):
        raise ContractError('unknown role in policy prompts')
    if "routing" in policy:
        # First release exposes no configurable stage skipping.
        if policy["routing"] != "evidence-hypothesis-experiment-evaluation-lead-knowledge":
            raise ContractError("unsupported route; cannot skip validation roles")
    public_only(policy)


def validate_patch_scope(scope,policy):
    """Permit precise prompt paths without allowing undeclared or kernel edits."""
    from .agents import ROLES
    validate_policy(policy)
    allowed=EDITABLE|{'prompts.'+role for role in ROLES}
    if not isinstance(scope,list) or any(not isinstance(item,str) or item not in allowed for item in scope):
        raise ContractError('optimizer attempted kernel changes')
    declared=set(scope)
    for key,value in policy.items():
        if key in declared:continue
        if key=='prompts' and all('prompts.'+role in declared for role in value):continue
        raise ContractError('optimizer patch exceeds declared editable scope')


def freeze_validator_files(stages):
    """Bind built-in module code as well as argv files and declared dependencies."""
    hashes={};package=Path(__file__).resolve().parent
    for rule in stages.values():
        command=rule['command']
        if not all(isinstance(arg,str) for arg in command):raise ContractError('validator arguments must be strings')
        if len(command)>1 and command[1].endswith('.py') and '{' not in command[1] and not Path(command[1]).is_file():
            raise ContractError('validator script entrypoint missing: '+command[1])
        declared=rule.get('validator_files',[])
        if not isinstance(declared,list) or not all(isinstance(p,str) for p in declared):
            raise ContractError('validator_files must be a list of paths')
        for name in declared:
            path=Path(name)
            if not path.is_file():raise ContractError('declared validator dependency missing: '+name)
            hashes[str(path.resolve())]=file_hash(path)
        for argument in command:
            path=Path(argument)
            if '{' not in argument and path.is_file():hashes[str(path.resolve())]=file_hash(path)
        if '-m' in command:
            index=command.index('-m')
            if index+1>=len(command):raise ContractError('validator module missing')
            module=command[index+1]
            if module.startswith('rec_harness.'):
                parts=module.split('.')[1:]
                if not all(part.isidentifier() for part in parts) or not package.joinpath(*parts).with_suffix('.py').is_file():
                    raise ContractError('unknown built-in validator module')
                # Freeze the package rather than guessing its dynamic import
                # graph. This covers scoring, process boundaries and recovery.
                for path in sorted(package.rglob('*.py')):hashes[str(path.resolve())]=file_hash(path)
            elif not declared:
                raise ContractError('external module validators require explicit validator_files')
    if not hashes:raise ContractError('validation requires at least one frozen local validator file')
    return hashes


class Evolution:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.root / "memory")
        self.artifacts = Artifacts(self.root / "objects")
        self._mutex=threading.RLock();self._lock_depth=0

    @contextmanager
    def lock(self):
        with self._mutex:
            if self._lock_depth:
                yield;return
            with (self.root/'controller.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as error:raise ContractError('evolution controller already active') from error
                assert_stopped(self.root/'candidates');self.store.recover()
                self._lock_depth+=1
                try:yield
                finally:self._lock_depth-=1

    @serialized
    def propose(self, candidate_id, kind, context, plan, *, agent=None, source=None):
        identifier(candidate_id)
        if kind not in STAGES:
            raise ContractError("invalid candidate kind")
        if 'tool_method_contract' in plan and kind!='tool':raise ContractError('tool method contract requires a tool candidate')
        required(plan, ["budget", "agent_cost", "development_ids", "stages", "identity", "rollback_version"])
        if set(plan["stages"]) != set(STAGES[kind]):
            raise ContractError("all frozen validation stages are required")
        required(plan["identity"], ["base_llm", "tools", "start", "graph", "budget", "protocol"])
        for stage, rule in plan["stages"].items():
            required(rule, ["command", "timeout", "cost", "minimum_gain", "maximum_regression", "maximum_total_cost"])
            if not isinstance(rule["command"], list) or not rule["command"]:
                raise ContractError("validator command must be argv")
            for key in ("timeout", "cost", "minimum_gain", "maximum_regression", "maximum_total_cost"):
                number(rule[key], key, 0)
        validator_hashes = freeze_validator_files(plan['stages'])
        directory = self.root / "candidates" / candidate_id
        frozen = {"candidate_id": candidate_id, "kind": kind, "plan_hash": digest(plan), "context_hash": digest(context), "source_hash": digest(source), "status": "proposed", "validator_hashes": validator_hashes}
        self.store.configure_budget(plan["budget"])
        committed=[e['payload'] for e in self.store.events()
                   if e['kind'] in {'candidate_implemented','measurement_handoff_recorded'} and e['payload']['candidate_id']==candidate_id]
        if committed:
            saved=committed[-1]
            if any(saved[key]!=frozen[key] for key in ['candidate_id','kind','plan_hash','context_hash','source_hash','validator_hashes']):
                raise ContractError('committed evolution inputs differ')
            if saved.get('status')=='measurement_required':
                self.artifacts.verify(saved['proposal_ref'])
                marker=directory/'candidate.json'
                if marker.exists():
                    current=read_json(marker)
                    if current!=frozen and current!=saved:raise ContractError('measurement handoff projection changed')
                write_json(directory/'candidate.json',saved)
                return saved
            if digest(saved['implementation'])!=saved['implementation_hash']:
                raise ContractError('committed implementation identity changed')
            for ref in saved['implementation'].values():self.artifacts.verify(ref)
            if saved.get('proposal_ref'):self.artifacts.verify(saved['proposal_ref'])
            marker=directory/'candidate.json'
            if marker.exists():
                current=read_json(marker)
                if any(current[key]!=saved[key] for key in ['candidate_id','kind','plan_hash','context_hash','source_hash','validator_hashes']):
                    raise ContractError('candidate projection inputs differ')
                if 'implementation' in current:
                    if (current['implementation']!=saved['implementation'] or current['implementation_hash']!=saved['implementation_hash']):
                        raise ContractError('candidate projection implementation differs')
                    self._verify_proposal(current)
                    return current
            write_json(marker,saved)
            return saved
        if (directory/'candidate.json').exists():
            old=read_json(directory/'candidate.json')
            if any(old[key]!=frozen[key] for key in ['candidate_id','kind','plan_hash','context_hash','source_hash','validator_hashes']):
                raise ContractError('frozen evolution proposal changed')
            if 'implementation' in old:
                if digest(old['implementation'])!=old['implementation_hash']:raise ContractError('candidate implementation identity changed')
                for ref in old['implementation'].values():self.artifacts.verify(ref)
                # Compatibility with earlier projection-before-event commits.
                self.store.event('candidate_implemented',old)
                return old
            frozen=old
        else:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "plan.json", plan);write_json(directory / "context.json", context)
            write_json(directory / "candidate.json", frozen);self.store.event("candidate_proposed", frozen)
        role = "tool" if kind == "tool" else "optimizer"
        if kind=='harness' and 'measurement_handoff_request' in context:role='optimizer_router'
        task_dir = directory / "development"
        request_context={"context":context,"acceptance_plan":plan}
        if kind=='tool':
            from .tool_method import contract
            method_contract=contract(plan)
            if method_contract is not None:
                request_context['tool_method_contract']={**method_contract,
                    'inherited_implementation_hash':digest(source) if source else None}
        submissions=[e['payload'] for e in self.store.events()
                     if e['kind']=='candidate_generation_submitted' and e['payload']['candidate_id']==candidate_id]
        recovered=bool(submissions)
        if recovered:
            submitted=submissions[-1];task_id=submitted['task_id']
            if submitted['role']!=role or submitted['input_hash']!=digest(request_context):
                raise ContractError('generation request changed during recovery')
            needed=['resources.json','context.json','request.json','response.json','schema.json']
            if any(not (task_dir/name).is_file() for name in needed):
                raise ContractError('generation outcome unresolved; retained attempt requires review, not automatic resubmission')
            receipt=read_json(task_dir/'resources.json');request=read_json(task_dir/'request.json')
            if (receipt.get('status')!='success' or request.get('role')!=role
                or request.get('context')!=request_context or read_json(task_dir/'context.json')!=request_context
                or read_json(task_dir/'schema.json')!=schema_for(role)):
                raise ContractError('generation receipts do not establish a successful matching request')
            response=read_json(task_dir/'response.json');validate_schema(response,schema_for(role));public_only(response)
            files=needed+(['model/'+name for name in response['changed_files']] if kind=='tool' else [])
            archived=self.artifacts.snapshot(task_dir,files)
            recovery={'candidate_id':candidate_id,'task_id':task_id,'receipts':archived,
                      'scope':'Successful matching local terminal receipts; no new generation or charge.'}
            previous=[e['payload'] for e in self.store.events() if e['kind']=='candidate_generation_recovered'
                      and e['payload']['task_id']==task_id]
            if previous:
                if previous[-1]!=recovery:raise ContractError('recovered generation artifacts changed')
            else:self.store.event('candidate_generation_recovered',recovery)
        else:
            attempts=directory/'development-attempts';attempts.mkdir(exist_ok=True)
            if task_dir.exists():task_dir.rename(attempts/str(len(list(attempts.iterdir()))+1))
            if source:self.artifacts.restore(source,task_dir/'model')
            task_id=candidate_id+'-generation-'+str(len(list(attempts.iterdir()))+1)
            self.store.reserve(task_id,plan['agent_cost'])
            self.store.event('candidate_generation_submitted',{'candidate_id':candidate_id,'task_id':task_id,
                'role':role,'input_hash':digest(request_context)})
        status = "crash"
        def record_generation():
            from .generation_workspace import archive_files
            names=['context.json','request.json','schema.json','response.json','resources.json',
                   'stdout.jsonl','stderr.txt','literature-search.json','prompt.txt','launch.json','process.json']
            receipts=self.artifacts.snapshot(task_dir,[name for name in names if (task_dir/name).is_file()]+archive_files(task_dir))
            record={'candidate_id':candidate_id,'task_id':task_id,'role':role,'receipts':receipts,
                    'scope':'Original available generation files; missing files are not inferred and transport output is not validation evidence.'}
            prior=[e['payload'] for e in self.store.events() if e['kind']=='candidate_generation_archived'
                   and e['payload']['task_id']==task_id]
            if prior:
                if prior[-1]!=record:raise ContractError('generation archive differs from committed receipts')
            else:self.store.event('candidate_generation_archived',record)
        try:
            if not recovered:response=(agent or CodexAgent()).run(role,request_context,task_dir)
            record_generation()
            if role=='optimizer_router':
                validate_schema(response,schema_for(role));choice=response['kind'];response=json.loads(response['payload_json'])
                if choice=='measurement_gap':
                    from .optimizer_handoff import validate
                    gap=validate(context,response,candidate_id)
                    write_json(task_dir/'accepted-measurement-review.json',response)
                    frozen.update(status='measurement_required',measurement_gap=gap,
                        proposal_ref=self.artifacts.put(task_dir/'accepted-measurement-review.json'))
                    status='success';self.store.event('measurement_handoff_recorded',frozen)
                    write_json(directory/'candidate.json',frozen)
                    return frozen
                validate_schema(response,schema_for('optimizer'))
            if kind == "harness":
                validate_card("harness_patch", response)
                if 'decision_traces' in context:
                    known={trace['trace_id'] for trace in context['decision_traces']}
                    if not response['trace_refs'] or not set(response['trace_refs']).issubset(known):raise ContractError('optimizer cites traces outside its frozen prefix')
                policy = json.loads(response["patch_diff"])
                validate_patch_scope(response["editable_scope"],policy)
                if 'base_policy' in context:
                    baseline=context['base_policy'];validate_policy(baseline)
                    policy={**baseline,**policy,**({'prompts':{**baseline.get('prompts',{}),**policy['prompts']}} if 'prompts' in policy else {})}
                    validate_policy(policy)
                # Preserve the rationale independently of the executable policy.
                # Scripted transports need not write their own response receipt.
                public_only(response)
                write_json(task_dir / 'accepted-proposal.json',response)
                frozen['proposal_ref']=self.artifacts.put(task_dir / 'accepted-proposal.json')
                write_json(task_dir / "policy.json", policy)
                files = ["policy.json"]
                snapshot = self.artifacts.snapshot(task_dir, files)
            else:
                snapshot = self.artifacts.snapshot(task_dir / "model", response["changed_files"])
                if plan.get('tool_method_contract'):
                    from .tool_method import FILENAME,validate
                    if FILENAME not in snapshot:raise ContractError('tool method card must be declared in changed_files')
                    method_ref=snapshot.pop(FILENAME)
                    validate(read_json(self.artifacts.verify(method_ref)),digest(source) if source else None)
                    frozen['proposal_ref']=method_ref
                if not snapshot:
                    raise ContractError("empty tool implementation")
                if source:
                    unchanged={name:ref for name,ref in source.items() if name not in snapshot}
                    for ref in unchanged.values():self.artifacts.verify(ref)
                    if self.artifacts.snapshot(task_dir/'model',unchanged)!=unchanged:
                        raise ContractError('tool changed inherited files without declaring them')
                    snapshot={**source,**snapshot}
            frozen.update(status="implemented", implementation=snapshot, implementation_hash=digest(snapshot))
            status = "success"
            self.store.event("candidate_implemented", frozen)
            write_json(directory / "candidate.json", frozen)
            return frozen
        finally:
            if not recovered:self.store.settle(task_id, plan["agent_cost"], status)
            # Failed transports also retain available requests, logs and receipts.
            # Avoid a second event for successful or already archived attempts.
            if not any(e['kind']=='candidate_generation_archived' and e['payload']['task_id']==task_id
                       for e in self.store.events()):record_generation()
            self.store.event("candidate_development_finished", {"candidate_id": candidate_id, "status": status, "task_id":task_id, "recovered":recovered})

    @serialized
    def run(self,candidate_id,kind,context,plan,*,agent=None,source=None):
        """Execute the frozen ladder; scientific rejection is terminal, never retried into a pass."""
        identifier(candidate_id)
        directory=self.root/'candidates'/candidate_id
        if (directory/'candidate.json').exists():
            frozen=read_json(directory/'candidate.json')
            if frozen['kind']!=kind or frozen['plan_hash']!=digest(plan) or frozen['context_hash']!=digest(context) or frozen.get('source_hash')!=digest(source):
                raise ContractError('frozen evolution episode changed')
        stopped=[e['payload'] for e in self.store.events() if e['kind']=='evolution_stopped' and e['payload']['candidate_id']==candidate_id]
        from .evolution_graph import sync_lifecycle
        graph_requested=any(e['kind']=='evolution_graph_requested' and e['payload']['candidate_id']==candidate_id for e in self.store.events())
        if stopped:
            if graph_requested:sync_lifecycle(self.store)
            return stopped[-1]
        if not graph_requested:self.store.event('evolution_graph_requested',{'candidate_id':candidate_id})
        try:
            candidate=self.propose(candidate_id,kind,context,plan,agent=agent,source=source)
            if candidate['status']=='measurement_required':
                result={'candidate_id':candidate_id,'kind':kind,'status':'measurement_required','measurement_gap':candidate['measurement_gap']}
            else:
                for stage in STAGES[kind]:
                    rejected=[e for e in self.store.events() if e['kind']=='validation_rejected' and e['payload']['candidate_id']==candidate_id and e['payload']['stage']==stage]
                    if rejected:raise ContractError('frozen validation previously rejected: '+stage)
                    self.validate(candidate_id,stage)
                version=self.promote(candidate_id)
                result={'candidate_id':candidate_id,'kind':kind,'status':'promoted','version':version}
        except ValidationRecoveryRequired:
            raise
        except (ContractError,OSError,ValueError,KeyError) as error:
            result={'candidate_id':candidate_id,'kind':kind,'status':'rejected','reason':str(error)}
            rejections=[e['payload'] for e in self.store.events() if e['kind']=='validation_rejected'
                        and e['payload']['candidate_id']==candidate_id]
            if rejections and rejections[-1].get('rejection_kind')=='validation_gate':
                result['rejection_kind']='validation_gate'
                result['rejected_stage']=rejections[-1]['stage']
        result['budget']=self.store.budget()
        self.store.event('evolution_stopped',result)
        from .evolution_graph import sync_lifecycle
        sync_lifecycle(self.store)
        write_json(directory/'run-report.json',result)
        return result

    @serialized
    def validate(self, candidate_id, stage):
        identifier(candidate_id)
        directory = self.root / "candidates" / candidate_id
        candidate, plan = read_json(directory / "candidate.json"), read_json(directory / "plan.json")
        if digest(plan) != candidate["plan_hash"]:
            raise ContractError("validation plan changed")
        if digest(candidate.get("implementation")) != candidate.get("implementation_hash"):
            raise ContractError("candidate implementation identity changed")
        for path, sha in candidate["validator_hashes"].items():
            if file_hash(path) != sha:
                raise ContractError("frozen validator changed")
        stages = STAGES[candidate["kind"]]
        if stage not in stages:
            raise ContractError("unknown validation stage")
        if candidate["status"] == "proposed":
            raise ContractError("candidate not implemented")
        validations = directory / "validation"
        for previous in stages[:stages.index(stage)]:
            if self._accepted_stage(candidate,plan,previous) is None:
                raise ContractError("validation ladder cannot be skipped")
        cached=self._accepted_stage(candidate,plan,stage)
        if cached is not None:return cached
        if any(e['kind']=='validation_rejected' and e['payload']['candidate_id']==candidate_id and e['payload']['stage']==stage for e in self.store.events()):
            raise ContractError('frozen validation previously rejected; create a new candidate and plan')
        target = validations / stage
        report = None
        try:
            from .validation_stage import report as stage_report
            report=stage_report(self,candidate,plan,stage,run_process)
            self._check_report(candidate, plan, stage, report)
            frozen_report = {"report": report, "report_hash": digest(report), "implementation_hash": candidate["implementation_hash"], "plan_hash": candidate["plan_hash"]}
            write_json(target / "accepted.json", frozen_report)
            self.store.event("validation_passed", {"candidate_id": candidate_id, "stage": stage, **frozen_report})
            candidate["status"] = {"robustness": "offline_validated", "shadow": "shadow_validated", "utility": "utility_validated",
                                   "prefix_audit": "offline_validated", "development": "development_validated", "confirmation": "utility_validated"}.get(stage, candidate["status"])
            write_json(directory / "candidate.json", candidate)
            return frozen_report
        except ValidationRecoveryRequired:
            raise
        except (ContractError, OSError, ValueError) as error:
            rejection={"candidate_id":candidate_id,"stage":stage,"reason":str(error)}
            if report is not None:
                try:public_only(report)
                except (ContractError,ValueError):report=None
            if report is not None:
                # A returned report remains evidence of attempted conditions even
                # when correctness or report-contract checks reject the tool.
                rejection.update(report=report,report_hash=digest(report),
                    implementation_hash=candidate['implementation_hash'],plan_hash=candidate['plan_hash'])
            rejection['rejection_kind']='validation_gate' if isinstance(error,ValidationGateRejected) else 'validation_contract'
            if isinstance(error,ValidationGateRejected):rejection['failed_gates']=error.failed_gates
            self.store.event("validation_rejected",rejection)
            raise

    def _accepted_stage(self,candidate,plan,stage):
        records=[e['payload'] for e in self.store.events() if e['kind']=='validation_passed'
                 and e['payload']['candidate_id']==candidate['candidate_id'] and e['payload']['stage']==stage]
        if not records:return None
        if len({r['report_hash'] for r in records})!=1:raise ContractError('conflicting committed validation reports')
        record=records[-1]
        accepted={key:record[key] for key in ['report','report_hash','implementation_hash','plan_hash']}
        if accepted['report_hash']!=digest(accepted['report']) or accepted['plan_hash']!=candidate['plan_hash'] or accepted['implementation_hash']!=candidate['implementation_hash']:
            raise ContractError('committed validation identity changed')
        self._check_report(candidate,plan,stage,accepted['report'])
        path=self.root/'candidates'/candidate['candidate_id']/'validation'/stage/'accepted.json'
        if path.exists() and read_json(path)!=accepted:raise ContractError('validation projection differs from committed report')
        if not path.exists():write_json(path,accepted)
        return accepted

    def _check_report(self, candidate, plan, stage, report):
        if not isinstance(report,dict):raise ContractError("validation report must be an object")
        public_only(report)
        groups = report.get("comparison_groups", report.get("arms"))
        if "comparison_groups" in report and "arms" in report and report["comparison_groups"] != report["arms"]:
            raise ContractError("conflicting comparison group fields")
        required({**report, "comparison_groups": groups}, ["passed", "gain", "regression", "total_cost", "evaluation_ids", "identity", "comparison_groups", "online"])
        rule = plan["stages"][stage]
        if report["passed"] is not True or report["identity"] != plan["identity"]:
            raise ContractError("validation failed or paired identity differs")
        failures=numerical_gate_failures(report,rule)
        ids = set(report["evaluation_ids"])
        if not ids:
            raise ContractError("missing validation identities")
        if stage in {"shadow", "utility", "development", "confirmation"} and ids & set(plan["development_ids"]):
            raise ContractError("validation overlaps candidate development")
        if stage == "shadow" and len(ids) < 2:
            raise ContractError("shadow requires two independent conditions")
        if stage in {"utility", "development", "confirmation"}:
            if report["online"] is not True:
                raise ContractError("replay is not online utility evidence")
            expected_groups = {"no_tool", "existing_tool", "candidate_tool"} if candidate["kind"] == "tool" else {"baseline", "candidate"}
            if set(groups) != expected_groups:
                raise ContractError("missing comparison groups")
        if stage == "confirmation":
            previous = read_json(self.root / "candidates" / candidate["candidate_id"] / "validation" / "development" / "accepted.json")
            if ids & set(previous["report"]["evaluation_ids"]):
                raise ContractError("confirmation reused development tasks")
        if failures:raise ValidationGateRejected('utility/cost/regression gate failed',failures)

    def verified_candidate(self, candidate_id):
        identifier(candidate_id)
        candidate = read_json(self.root / "candidates" / candidate_id / "candidate.json")
        if candidate.get('status')=='measurement_required':raise ContractError('measurement handoff is not a promotable implementation')
        plan = read_json(self.root / "candidates" / candidate_id / "plan.json")
        if digest(plan) != candidate["plan_hash"]:
            raise ContractError("plan changed")
        if digest(candidate["implementation"]) != candidate["implementation_hash"]:
            raise ContractError("candidate implementation identity changed")
        self._verify_proposal(candidate)
        for stage in STAGES[candidate["kind"]]:
            if self._accepted_stage(candidate,plan,stage) is None:
                raise ContractError(f"missing independent validation: {stage}")
        for ref in candidate["implementation"].values():
            self.artifacts.verify(ref)
        return candidate

    def _verify_proposal(self,candidate):
        committed=[e['payload'] for e in self.store.events() if e['kind'] in {'candidate_implemented','measurement_handoff_recorded'}
                   and e['payload']['candidate_id']==candidate['candidate_id']]
        if not committed:return
        reference=committed[-1].get('proposal_ref')
        if committed[-1].get('status')=='measurement_required' and candidate!=committed[-1]:raise ContractError('measurement handoff projection changed')
        if candidate.get('proposal_ref')!=reference:raise ContractError('candidate proposal projection changed')
        if reference:self.artifacts.verify(reference)

    @serialized
    def promote(self, candidate_id):
        candidate = self.verified_candidate(candidate_id)
        # Version activation is consumed only by a new episode, never a running controller.
        previous = self._active(candidate['kind'])
        if previous and previous['candidate_id']==candidate_id:
            write_json(self.root / f"active-{candidate['kind']}.json",previous)
            self.artifacts.restore(candidate['implementation'],self.root/'versions'/candidate_id)
            return previous
        version = {"candidate_id": candidate_id, "kind": candidate["kind"], "implementation": candidate["implementation"], "previous": previous, "plan_hash": candidate["plan_hash"]}
        self.store.event("candidate_promoted", version)
        candidate["status"] = "promoted"
        write_json(self.root / "candidates" / candidate_id / "candidate.json", candidate)
        write_json(self.root / f"active-{candidate['kind']}.json", version)
        self.artifacts.restore(candidate["implementation"], self.root / "versions" / candidate_id)
        return version

    def _active(self,kind):
        active=None
        for event in self.store.events():
            value=event['payload']
            if event['kind']=='candidate_promoted' and value['kind']==kind:
                if value['previous']!=active:raise ContractError('promotion predecessor differs from committed history')
                active=value
            elif event['kind']=='candidate_rollback' and active and value['from']==active['candidate_id']:
                expected=active['previous']['candidate_id'] if active['previous'] else None
                if value['to']!=expected:raise ContractError('invalid committed rollback')
                active=active['previous']
        return active

    def verify(self):
        integrity=self.store.verify()
        count=0
        for event in self.store.events():
            if event['kind'] in {'validation_stage_started','validation_stage_report_recorded'}:
                key='cost_context_ref' if event['kind']=='validation_stage_started' else 'report_ref'
                self.artifacts.verify(event['payload'][key]);count+=1
            if event['kind']=='validation_stage_launched' and 'cost_context_ref' in event['payload']:
                self.artifacts.verify(event['payload']['cost_context_ref']);count+=1
            if event['kind'] in {'candidate_implemented','measurement_handoff_recorded'} and event['payload'].get('proposal_ref'):
                self.artifacts.verify(event['payload']['proposal_ref']);count+=1
            if event['kind'] in {'candidate_generation_recovered','candidate_generation_archived'}:
                for ref in event['payload']['receipts'].values():self.artifacts.verify(ref);count+=1
        for path in (self.root/'candidates').glob('*/candidate.json'):
            candidate=read_json(path);plan=read_json(path.parent/'plan.json')
            self._verify_proposal(candidate)
            if digest(plan)!=candidate['plan_hash'] or digest(read_json(path.parent/'context.json'))!=candidate['context_hash']:
                raise ContractError('frozen candidate inputs changed')
            if 'implementation' in candidate:
                if digest(candidate['implementation'])!=candidate['implementation_hash']:raise ContractError('candidate implementation changed')
                for ref in candidate['implementation'].values():self.artifacts.verify(ref);count+=1
            for stage in STAGES[candidate['kind']]:self._accepted_stage(candidate,plan,stage)
        for kind in STAGES:
            active=self._active(kind)
            if active:self.verified_candidate(active['candidate_id'])
        if any(e['kind']=='evolution_graph_requested' for e in self.store.events()) and any(e['kind']=='evolution_stopped' for e in self.store.events()):
            from .evolution_graph import sync_lifecycle
            sync_lifecycle(self.store,check_only=True)
        return {**integrity,'verified_artifacts':count}

    @serialized
    def rollback(self, kind):
        if kind not in STAGES:
            raise ContractError("unknown kind")
        path = self.root / f"active-{kind}.json"
        active = self._active(kind)
        if not active:
            raise ContractError("no active promoted version")
        self.store.event("candidate_rollback", {"kind":kind,"from": active["candidate_id"], "to": active["previous"]["candidate_id"] if active['previous'] else None})
        write_json(path, active["previous"])
        if any(e["kind"]=="evolution_graph_requested" for e in self.store.events()):
            from .evolution_graph import sync_lifecycle
            sync_lifecycle(self.store)
        return active["previous"]
