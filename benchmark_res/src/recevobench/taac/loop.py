"""Karpathy-style incumbent editing, with durable trials and optional research evidence.

This is a trusted local development runner, not a sandbox against hostile agents.
Private labels are never included in agent requests; OS isolation is not claimed.
"""

from __future__ import annotations

import csv
import difflib
import fcntl
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean

from .common import bounded_run, digest, file_hash, read, verify_data, write
from .evaluate import diagnose, score
from .knowledge import consolidate, memory_cards, model_facts, validate_design
from .retrieval import freeze_config, history_cards, load_cards, materialize, validate_plan


def controller_identity():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def data_signature(root, files):
    root = Path(root)
    return {name: [p.stat().st_size, p.stat().st_mtime_ns] for name in files for p in [root / name]}


def initialize(config_path, root):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Study output must not already exist")
    for key in ["data", "candidate"]:
        config[key] = str((config_path.parent / config[key]).resolve())
    manifest = verify_data(config["data"])
    for key in ["trial_seconds", "total_seconds", "agent_seconds"]:
        v = config["limits"][key]
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError("Positive finite limit required: " + key)
    if not isinstance(config["limits"]["max_trials"], int) or config["limits"]["max_trials"] < 1:
        raise ValueError("Positive max_trials required, including baseline")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Distinct integer training seeds required")
    for key in ["epochs", "batch_size"]:
        if type(config["train"][key]) is not int or config["train"][key] < 1:
            raise ValueError("Positive integer training parameter required: " + key)
    if config["train"].get("max_steps") is not None and (
        type(config["train"]["max_steps"]) is not int or config["train"]["max_steps"] < 1
    ):
        raise ValueError("max_steps must be null or positive")
    if not math.isfinite(config["min_delta"]) or config["min_delta"] < 0:
        raise ValueError("Nonnegative finite calibrated min_delta required")
    agent = config.get("agent")
    if agent and (
        not isinstance(agent.get("argv"), list)
        or not agent["argv"]
        or any(not isinstance(a, str) for a in agent["argv"])
    ):
        raise ValueError("Agent argv must be a nonempty string array; no shell evaluation")
    if config.get("enhanced"):
        if "research_query" in config:
            raise ValueError("Remove research_query; the Agent now generates a retrieval plan")
        config["retrieval"] = freeze_config(config.get("retrieval", {}), config_path.parent)
        seconds = config["limits"].setdefault(
            "retrieval_seconds", config["limits"]["agent_seconds"]
        )
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite retrieval_seconds required")
    root.mkdir(parents=True)
    (root / "trials").mkdir()
    (root / "agents").mkdir()
    (root / "retrieval").mkdir()
    shutil.copyfile(config["candidate"], root / "baseline.py")
    config["initial_source_sha256"] = file_hash(root / "baseline.py")
    config["data_identity"] = manifest["identity"]
    config["data_signature"] = data_signature(config["data"], manifest["files"])
    config["controller_identity"] = controller_identity()
    config["knowledge"] = None
    if config.get("enhanced", False):
        source = read(config_path).get("knowledge")
        if not source:
            raise ValueError("Enhanced mode requires explicit knowledge JSONL")
        shutil.copyfile((config_path.parent / source).resolve(), root / "knowledge.jsonl")
        config["knowledge"] = "knowledge.jsonl"
        config["knowledge_sha256"] = file_hash(root / "knowledge.jsonl")
    config["personal"] = None
    personal = read(config_path).get("personal")
    if personal:
        shutil.copyfile((config_path.parent / personal).resolve(), root / "personal.json")
        config["personal"] = "personal.json"
        config["personal_sha256"] = file_hash(root / "personal.json")
    config["agent_memory"] = None
    prior = read(config_path).get("agent_memory")
    if prior:
        export_path = (config_path.parent / prior).resolve()
        memory_cards(read(export_path))
        shutil.copyfile(export_path, root / "prior-agent-memory.json")
        config["agent_memory"] = "prior-agent-memory.json"
        config["agent_memory_sha256"] = file_hash(root / "prior-agent-memory.json")
    if config.get("enhanced"):
        load_cards(
            root / config["knowledge"],
            memory_cards(read(root / config["agent_memory"])) if config["agent_memory"] else [],
        )
    # Do not read mutable instructions from the repository during a live study.
    program = read(config_path).get("program")
    if program:
        shutil.copyfile((config_path.parent / program).resolve(), root / "program.md")
        config["program_sha256"] = file_hash(root / "program.md")
    write(root / "config.json", config)
    write(
        root / "state.json",
        {
            "config_hash": digest(config),
            "trials": [],
            "incumbent": None,
            "spent_seconds": 0.0,
            "pending": None,
            "closed": False,
        },
    )
    write(root / "protocol.json", config["train"])
    _results(root, [])
    return {
        "root": str(root),
        "data_identity": manifest["identity"],
        "status": "initialized",
        "fixture": read(Path(config["data"]) / "public/metadata.json")["fixture"],
    }


def load(root):
    root = Path(root).resolve()
    config, state = read(root / "config.json"), read(root / "state.json")
    if (
        digest(config) != state["config_hash"]
        or controller_identity() != config["controller_identity"]
    ):
        raise ValueError("Frozen configuration/controller changed; create a new study")
    if file_hash(root / "baseline.py") != config["initial_source_sha256"]:
        raise ValueError("Baseline source changed")
    if read(root / "protocol.json") != config["train"]:
        raise ValueError("Frozen training protocol changed")
    for name, key in [
        ("knowledge.jsonl", "knowledge_sha256"),
        ("personal.json", "personal_sha256"),
        ("program.md", "program_sha256"),
        ("prior-agent-memory.json", "agent_memory_sha256"),
    ]:
        if key in config and file_hash(root / name) != config[key]:
            raise ValueError("Frozen read-only input changed: " + name)
    manifest = read(Path(config["data"]) / "manifest.json")
    if manifest["identity"] != config["data_identity"] or manifest["identity"] != digest(
        manifest["files"]
    ):
        raise ValueError("Prepared dataset identity changed")
    if data_signature(config["data"], manifest["files"]) != config["data_signature"]:
        raise ValueError("Frozen prepared files changed; re-verify and initialize a new study")
    for t in state["trials"]:
        directory = root / "trials" / t["id"]
        if read(directory / "receipt.json") != t:
            raise ValueError("Committed trial receipt changed")
        source = directory / "candidate.py"
        if source.exists() and file_hash(source) != t["source_sha256"]:
            raise ValueError("Committed candidate source changed")
    return root, config, state


@contextmanager
def locked(root):
    with (Path(root) / "controller.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _results(root, trials):
    path = root / "results.tsv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ["trial", "parent", "source_sha256", "score", "status", "wall_seconds", "description"]
        )
        for t in trials:
            writer.writerow(
                [
                    t["id"],
                    t["parent"],
                    t["source_sha256"],
                    t.get("score"),
                    t["status"],
                    t["wall_seconds"],
                    t["hypothesis"],
                ]
            )


def _save(root, state):
    write(root / "state.json", state)
    _results(root, state["trials"])


def _ready(config, state):
    if state["closed"]:
        raise ValueError("Study closed; hidden evaluation cannot feed back into research")
    if state["pending"]:
        raise ValueError("Unsettled work: run recover to retain its reserved cost")
    if state["spent_seconds"] >= config["limits"]["total_seconds"]:
        raise ValueError("Study wall-time budget exhausted")


def _env():
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def context(root):
    root, config, state = load(root)
    meta = read(Path(config["data"]) / "public/metadata.json")
    current = next((t for t in state["trials"] if t["id"] == state["incumbent"]), None)
    source = root / "trials" / current["id"] / "candidate.py" if current else root / "baseline.py"
    diagnostic = None
    if current:
        path = root / "trials" / current["id"] / "diagnostics.json"
        if path.exists():
            diagnostic = read(path)
    facts = model_facts(source, meta, diagnostic)
    history = []
    for trial_record in state["trials"]:
        directory = root / "trials" / trial_record["id"]
        entry = dict(trial_record)
        for name in ["design", "analysis", "diagnostics"]:
            path = directory / (name + ".json")
            if path.exists():
                value = read(path)
                entry[name] = (
                    {k: v for k, v in value.items() if k != "observations"}
                    if name == "diagnostics"
                    else value
                )
        history.append(entry)
    result = {
        "task": meta,
        "model_state": facts,
        "history": history,
        "remaining_seconds": config["limits"]["total_seconds"] - state["spent_seconds"],
        "protocol": config["train"],
        "selection": "feedback, provisional until final audit",
        "candidate_source": str(source),
        "knowledge": [],
        "personal": read(root / "personal.json") if config["personal"] else None,
        "instructions": "Only candidate.py is editable model code. Keep controls and "
        "uncertainty explicit. No private labels, gate or final scores are supplied.",
    }
    if config.get("enhanced"):
        if diagnostic:
            result["diagnostics"] = {k: v for k, v in diagnostic.items() if k != "observations"}
        saved = root / "research-context.json"
        if saved.exists():
            prepared = read(saved)
            expected = state.get("research_context_hash")
            if digest(prepared) != expected:
                raise ValueError("Prepared retrieval context changed")
            if prepared["evidence_digest"] == evidence_digest(result):
                result.update(knowledge=prepared["knowledge"], retrieval=prepared["retrieval"])
    return result


def evidence_digest(ctx):
    return digest(
        {k: v for k, v in ctx.items() if k not in {"remaining_seconds", "knowledge", "retrieval"}}
    )


def _research_context(root, config, state):
    _ready(config, state)
    ctx = context(root)
    if not config.get("enhanced"):
        return ctx
    # Always regenerate from current evidence, never reuse a previous round's query.
    ctx["knowledge"] = []
    ctx.pop("retrieval", None)
    work = _agent(root, config, state, "query", ctx, Path(ctx["candidate_source"]))
    plan = read(work / "retrieval-plan.json")
    cards = load_cards(
        root / config["knowledge"],
        (memory_cards(read(root / config["agent_memory"])) if config["agent_memory"] else [])
        + history_cards(ctx),
    )
    _ready(config, state)
    directory = root / "retrieval" / work.name
    reserve = min(
        config["limits"]["retrieval_seconds"],
        config["limits"]["total_seconds"] - state["spent_seconds"],
    )
    state["pending"] = {"kind": "retrieval", "id": directory.name, "reserved_seconds": reserve}
    _save(root, state)
    started = time.monotonic()
    directory.mkdir()
    request = {
        "schema": "taac-retrieval-request-v1",
        "config": config["retrieval"],
        "plan": plan,
        "cards": cards,
        "model_state": ctx["model_state"],
        "evidence_digest": evidence_digest(ctx),
    }
    write(directory / "request.json", request)
    output = directory / "ranking.json"
    try:
        if config["retrieval"]["backend"] == "hybrid":
            argv = [
                sys.executable,
                "-m",
                "recevobench.taac",
                "_retrieve",
                "--request",
                str(directory / "request.json"),
                "--output",
                str(output),
            ]
        else:
            for path, expected in config["retrieval"]["artifact_hashes"].items():
                if file_hash(path) != expected:
                    raise ValueError("Frozen retrieval backend artifact changed")
            argv = [
                a.format(
                    request=str(directory / "request.json"),
                    output=str(output),
                    workspace=str(directory),
                )
                for a in config["retrieval"]["argv"]
            ]
        execution = bounded_run(argv, directory, directory / "retrieval.log", reserve, _env())
        write(directory / "execution.json", execution)
        if execution["status"] != "success":
            raise ValueError("Retrieval failed; no lexical fallback; inspect " + str(directory))
        result = read(output)
        if result.get("schema") != "taac-retrieval-ranking-v1" or result.get(
            "request_digest"
        ) != digest(request):
            raise ValueError("Retrieval response is not bound to the current request")
        knowledge = materialize(
            cards, ctx["model_state"], result["ranking"], config["retrieval"]["top_k"]
        )
        retrieval = {
            "plan": plan,
            "request_digest": digest(request),
            "ranking": result["ranking"],
            "backend": config["retrieval"]["backend"],
            "artifacts": str(directory),
            "query_workspace": str(work),
        }
        prepared = {
            "evidence_digest": evidence_digest(ctx),
            "knowledge": knowledge,
            "retrieval": retrieval,
        }
        if time.monotonic() - started > reserve:
            raise ValueError("Retrieval budget exceeded including hydration")
        write(directory / "context.json", prepared)
        write(root / "research-context.json", prepared)
        state["research_context_hash"] = digest(prepared)
    except Exception as error:
        write(directory / "failure.json", {"error": str(error)})
        raise
    finally:
        state["spent_seconds"] += time.monotonic() - started
        state["pending"] = None
        _save(root, state)
    return context(root)


def research_context(root):
    """Explicit, budgeted preparation for enhanced manual trials; context stays read-only."""
    with locked(root):
        root, config, state = load(root)
        return _research_context(root, config, state)


def _trial(root, config, state, source, design, ctx):
    _ready(config, state)
    if len(state["trials"]) >= config["limits"]["max_trials"]:
        raise ValueError("Trial budget exhausted (baseline is included)")
    baseline = state["incumbent"] is None
    if config.get("enhanced") and not baseline and not ctx.get("retrieval"):
        raise ValueError(
            "Run research-context to prepare a fresh Agent retrieval plan before trial"
        )
    validate_design(design, ctx, config.get("enhanced", False) and not baseline)
    if (
        not baseline
        and any(t["source_sha256"] == file_hash(source) for t in state["trials"])
        and design.get("kind") != "repeat"
    ):
        raise ValueError("Previously evaluated source; declare kind=repeat for intentional retest")
    trial_id = f"t{len(state['trials']):04d}"
    directory = root / "trials" / trial_id
    remaining = config["limits"]["total_seconds"] - state["spent_seconds"]
    reservation = min(config["limits"]["trial_seconds"], remaining)
    pending = {
        "kind": "trial",
        "id": trial_id,
        "reserved_seconds": reservation,
        "parent": state["incumbent"],
        "hypothesis": design["hypothesis"],
        "source_sha256": file_hash(source),
    }
    # Reserve before filesystem setup or launching any computation.
    state["pending"] = pending
    _save(root, state)
    started = time.monotonic()
    directory.mkdir()
    shutil.copyfile(source, directory / "candidate.py")
    parent_source = (
        root / "trials" / state["incumbent"] / "candidate.py"
        if state["incumbent"]
        else root / "baseline.py"
    )
    (directory / "code.diff").write_text(
        "".join(
            difflib.unified_diff(
                parent_source.read_text().splitlines(keepends=True),
                (directory / "candidate.py").read_text().splitlines(keepends=True),
                fromfile="incumbent.py",
                tofile="candidate.py",
            )
        )
    )
    write(directory / "design.json", design)
    write(directory / "context.json", ctx)
    metrics, executions, failure = [], [], None
    data = Path(config["data"])
    try:
        for seed in config["seeds"]:
            seed_root = directory / f"seed-{seed}"
            argv = [
                sys.executable,
                "-m",
                "recevobench.taac",
                "_fit",
                "--public",
                str(data / "public"),
                "--candidate",
                str(directory / "candidate.py"),
                "--output",
                str(seed_root),
                "--protocol",
                str(root / "protocol.json"),
                "--device",
                config["device"],
                "--seed",
                str(seed),
            ]
            receipt = bounded_run(
                argv,
                directory,
                directory / f"seed-{seed}.log",
                reservation - (time.monotonic() - started),
                _env(),
            )
            executions.append(receipt)
            if receipt["status"] != "success":
                raise ValueError("Worker " + receipt["status"] + "; inspect log")
            metrics.append(
                score(
                    seed_root / "feedback/predictions.jsonl",
                    data / "private/feedback.jsonl",
                    read(data / "public/candidate_ids.json"),
                    seed_root / "metrics.json",
                )
            )
        if config.get("enhanced"):
            # One declared seed for cheap screening; all seed metrics remain in the receipt.
            diagnostic = diagnose(
                data / "public",
                data / "private/feedback.jsonl",
                directory / f"seed-{config['seeds'][0]}/feedback",
                directory / "diagnostics.json",
            )
            diagnostic["seed"] = config["seeds"][0]
            write(directory / "diagnostics.json", diagnostic)
        if time.monotonic() - started > reservation:
            raise ValueError("Trial budget exceeded including scoring/diagnostics")
        if file_hash(directory / "candidate.py") != pending["source_sha256"]:
            raise ValueError("Candidate source changed during execution")
    except Exception as error:
        failure = str(error)
    wall = time.monotonic() - started
    mean = fmean(m["score"] for m in metrics) if not failure else None
    incumbent = next((t for t in state["trials"] if t["id"] == state["incumbent"]), None)
    delta = mean - incumbent["score"] if mean is not None and incumbent else None
    keep = mean is not None and (baseline or delta > config["min_delta"])
    trial = {
        **{k: v for k, v in pending.items() if k not in {"kind", "reserved_seconds"}},
        "score": mean,
        "delta": delta,
        "seed_metrics": metrics,
        "wall_seconds": wall,
        "status": "crash" if failure else ("keep" if keep else "discard"),
        "failure": failure,
        "executions": executions,
        "candidate_artifacts": {
            str(p.relative_to(directory)): file_hash(p)
            for p in directory.rglob("*")
            if p.is_file() and p.name not in {"receipt.json", "analysis.json"}
        },
    }
    write(directory / "receipt.json", trial)
    state["trials"].append(trial)
    state["spent_seconds"] += wall
    state["pending"] = None
    if keep:
        state["incumbent"] = trial_id
    _save(root, state)
    return trial


def trial(root, source, design_path):
    with locked(root):
        root, config, state = load(root)
        return _trial(root, config, state, Path(source).resolve(), read(design_path), context(root))


def _baseline(root, config, state):
    design = {
        "hypothesis": "Establish unchanged adapted Tencent baseline",
        "change": "baseline",
        "falsification": "Invalid training/inference or nonfinite score",
        "controls": ["unchanged source, fixed data and protocol"],
        "budget_rationale": "Frozen baseline calibration budget",
    }
    return _trial(root, config, state, root / "baseline.py", design, context(root))


def baseline(root):
    with locked(root):
        root, config, state = load(root)
        if state["trials"]:
            raise ValueError("Baseline already attempted; preserve its evidence")
        return _baseline(root, config, state)


def _agent(root, config, state, phase, payload, source):
    _ready(config, state)
    if not config.get("agent"):
        raise ValueError("Configure agent.argv or use manual trial submissions")
    call_index = len(list((root / "agents").iterdir()))
    directory = root / "agents" / f"{call_index:04d}-{phase}"
    directory.mkdir(exist_ok=False)
    shutil.copyfile(source, directory / "candidate.py")
    write(
        directory / "request.json",
        {
            "phase": phase,
            "context": payload,
            "outputs": {
                "plan": ["candidate.py", "design.json"],
                "query": ["retrieval-plan.json"],
                "analyze": ["analysis.json"],
            }[phase],
            "instructions": "Query: read model code, diagnostics and history; "
            "write retrieval-plan.json "
            "with 1-3 queries, each text, intent (mechanism/method/counterevidence), "
            "keywords, "
            "conditions (field/op/value), rationale and evidence_refs (fact keys or trial:<id>). "
            "Include counterevidence intent; cite model, available diagnostics and history. "
            "Plan: assess every knowledge card and its inline/linked counterexamples via "
            "knowledge_assessments; retain condition_checks; "
            "then edit candidate.py and write design. "
            "Analyze: do not edit code; explain results, alternatives and next test. "
            "Do not run training yourself. Host performs execution and scoring.",
        },
    )
    if (root / "program.md").exists():
        shutil.copyfile(root / "program.md", directory / "program.md")
    argv = [
        a.format(workspace=str(directory), request=str(directory / "request.json"))
        for a in config["agent"]["argv"]
    ]
    reserve = min(
        config["limits"]["agent_seconds"],
        config["limits"]["total_seconds"] - state["spent_seconds"],
    )
    state["pending"] = {"kind": "agent", "id": directory.name, "reserved_seconds": reserve}
    _save(root, state)
    receipt = bounded_run(argv, directory, directory / "agent.log", reserve, _env())
    write(directory / "execution.json", receipt)
    state["spent_seconds"] += receipt["wall_seconds"]
    state["pending"] = None
    _save(root, state)
    load(root)  # Recheck frozen task, controller and source after the agent process.
    if receipt["status"] != "success":
        raise ValueError("Agent execution failed; cost retained, inspect " + str(directory))
    if phase == "plan":
        validate_design(read(directory / "design.json"), payload, config.get("enhanced", False))
    elif phase == "query":
        if file_hash(directory / "candidate.py") != file_hash(source):
            raise ValueError("Query phase must not edit candidate code")
        validate_plan(read(directory / "retrieval-plan.json"), payload)
    else:
        if file_hash(directory / "candidate.py") != file_hash(source):
            raise ValueError("Analyze phase must not edit candidate code")
        value = read(directory / "analysis.json")
        for field in ["effect", "alternatives", "next_action", "evidence_trials"]:
            if field not in value:
                raise ValueError("Analysis missing " + field)
        known = {t["id"] for t in state["trials"]}
        if (
            not value["evidence_trials"]
            or not set(value["evidence_trials"]) <= known
            or state["trials"][-1]["id"] not in value["evidence_trials"]
        ):
            raise ValueError("Analysis references nonexistent trials")
    return directory


def run(root, rounds=1):
    if rounds < 1:
        raise ValueError("Positive rounds required")
    with locked(root):
        root, config, state = load(root)
        _ready(config, state)
        if not state["trials"]:
            result = _baseline(root, config, state)
            if result["status"] == "crash":
                return state
        if state["incumbent"] is None:
            raise ValueError("No valid baseline; diagnose the failed baseline first")
        for _ in range(rounds):
            if len(state["trials"]) >= config["limits"]["max_trials"]:
                break
            ctx = _research_context(root, config, state)
            work = _agent(root, config, state, "plan", ctx, Path(ctx["candidate_source"]))
            result = _trial(
                root, config, state, work / "candidate.py", read(work / "design.json"), ctx
            )
            if config.get("enhanced"):
                after = {
                    **context(root),
                    "knowledge": ctx["knowledge"],
                    "retrieval": ctx["retrieval"],
                    "latest_trial": result,
                    "latest_design": read(root / "trials" / result["id"] / "design.json"),
                }
                analysis = _agent(
                    root,
                    config,
                    state,
                    "analyze",
                    after,
                    root / "trials" / result["id"] / "candidate.py",
                )
                shutil.copyfile(
                    analysis / "analysis.json", root / "trials" / result["id"] / "analysis.json"
                )
        return state


def recover(root):
    with locked(root):
        root, _, state = load(root)
        pending = state["pending"]
        if pending is None:
            return state
        work = (
            root / "trials" / pending["id"]
            if pending["kind"] == "trial"
            else root / "agents" / pending["id"]
            if pending["kind"] == "agent"
            else root / "retrieval" / pending["id"]
            if pending["kind"] == "retrieval"
            else root / "final"
        )
        for launch in work.rglob("*.launch.json") if work.exists() else []:
            pgid = read(launch)["pgid"]
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            raise ValueError(
                "Worker process group still exists; verify/stop it before recovery: " + str(pgid)
            )
        state["spent_seconds"] += pending["reserved_seconds"]
        if pending["kind"] == "trial":
            directory = root / "trials" / pending["id"]
            directory.mkdir(exist_ok=True)
            receipt_path = directory / "receipt.json"
            # A durable terminal receipt can be adopted without a second execution.
            if receipt_path.exists():
                record = read(receipt_path)
                if record["source_sha256"] != pending["source_sha256"]:
                    raise ValueError("Recovered receipt identity mismatch")
                state["spent_seconds"] -= pending["reserved_seconds"]
                state["spent_seconds"] += record["wall_seconds"]
            else:
                record = {
                    **{k: v for k, v in pending.items() if k not in {"kind", "reserved_seconds"}},
                    "status": "interrupted",
                    "score": None,
                    "delta": None,
                    "wall_seconds": pending["reserved_seconds"],
                    "failure": "Unsettled execution; full reservation retained",
                    "seed_metrics": [],
                }
                write(receipt_path, record)
            state["trials"].append(record)
            if record["status"] == "keep":
                state["incumbent"] = record["id"]
        state["pending"] = None
        _save(root, state)
        return state


def finish(root):
    """Close research before evaluating gate/final; never feed those scores back."""
    with locked(root):
        root, config, state = load(root)
        _ready(config, state)
        if not state["incumbent"]:
            raise ValueError("No valid incumbent to finalize")
        verify_data(config["data"])
        state["closed"] = True
        state["selected"] = state["incumbent"]
        state["pending"] = {
            "kind": "final",
            "id": "final",
            "reserved_seconds": min(
                config["limits"]["trial_seconds"],
                config["limits"]["total_seconds"] - state["spent_seconds"],
            ),
        }
        _save(root, state)
        start = time.monotonic()
        limit = state["pending"]["reserved_seconds"]
        data = Path(config["data"])
        result = {}
        try:
            for tid in dict.fromkeys([state["trials"][0]["id"], state["selected"]]):
                result[tid] = {}
                for part in ["gate", "final"]:
                    values = []
                    for seed in config["seeds"]:
                        directory = root / "trials" / tid
                        out = root / "final" / tid / part / str(seed)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        committed = next(t for t in state["trials"] if t["id"] == tid)
                        checkpoint_rel = f"seed-{seed}/train/checkpoint.pt"
                        if (
                            file_hash(directory / checkpoint_rel)
                            != committed["candidate_artifacts"][checkpoint_rel]
                        ):
                            raise ValueError("Committed checkpoint changed before final evaluation")
                        cmd = [
                            sys.executable,
                            "-m",
                            "recevobench.taac",
                            "predict",
                            "--public",
                            str(data / "public"),
                            "--candidate",
                            str(directory / "candidate.py"),
                            "--checkpoint",
                            str(directory / f"seed-{seed}/train/checkpoint.pt"),
                            "--queries",
                            str(data / f"public/{part}.jsonl"),
                            "--output",
                            str(out),
                            "--device",
                            config["device"],
                        ]
                        receipt = bounded_run(
                            cmd,
                            root,
                            out.parent / f"{seed}.log",
                            limit - (time.monotonic() - start),
                            _env(),
                        )
                        if receipt["status"] != "success":
                            raise ValueError("Final inference failed; no automatic retry")
                        values.append(
                            score(
                                out / "predictions.jsonl",
                                data / f"private/{part}.jsonl",
                                read(data / "public/candidate_ids.json"),
                                out / "metrics.json",
                            )
                        )
                    result[tid][part] = {
                        "score": fmean(v["score"] for v in values),
                        "seeds": values,
                    }
            write(
                root / "final.json",
                {
                    "selected": state["selected"],
                    "results": result,
                    "scope": "Local frozen holdout, not original leaderboard; "
                    "feedback selected model",
                },
            )
        finally:
            state["spent_seconds"] += time.monotonic() - start
            state["pending"] = None
            _save(root, state)
        return result


def stage_memory(root):
    with locked(root):
        root, _, state = load(root)
        return consolidate(root, state, context(root), root / "stage-memory.json")
