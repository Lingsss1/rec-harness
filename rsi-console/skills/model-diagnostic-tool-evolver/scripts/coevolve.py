#!/usr/bin/env python3
"""Persist trigger-derived probes and gate their handoff to model experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import manage_candidate as candidates


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def require(value, keys):
    for key in keys:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"nonempty string required: {key}")


def artifact(workspace, name):
    path = (workspace.resolve() / name).resolve()
    if not path.is_relative_to(workspace.resolve()) or not path.is_file():
        raise ValueError(f"evidence must be a file inside workspace: {name}")
    return {"path": str(path.relative_to(workspace.resolve())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(workspace, ref):
    if artifact(workspace, ref["path"]) != ref:
        raise ValueError(f"evidence changed: {ref['path']}")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def propose(args):
    spec = read(args.input)
    require(spec, ("trigger_id", "kind", "observation", "component", "code_identity",
                   "protocol_id", "gap", "decision", "baseline", "budget",
                   "measurement", "null_behavior", "action", "falsification"))
    if spec["kind"] not in {"structural_change", "user_issue"}:
        raise ValueError("kind must be structural_change or user_issue")
    source = artifact(args.workspace, spec["source"])
    # Identity includes context: the same complaint on a changed model is a new probe.
    candidate_id = "probe-" + digest(spec)[:24]
    directory = candidates.candidate_dir(args.workspace, candidate_id)
    if directory.exists():
        state = candidates.read_state(args.workspace, candidate_id)
        if state.get("trigger_evidence") != source:
            raise ValueError("trigger evidence changed; use a new trigger/context")
        return {"candidate_id": candidate_id, "status": state["status"], "duplicate": True}
    candidates.command_init(argparse.Namespace(
        workspace=args.workspace, id=candidate_id, gap=spec["gap"],
        decision=spec["decision"], baseline=spec["baseline"], budget=spec["budget"]))
    state = candidates.read_state(args.workspace, candidate_id)
    state["proposal"] = spec
    state["trigger_evidence"] = source
    candidates.write_state(args.workspace, candidate_id, state)
    candidates.append_event(args.workspace, candidate_id, "probe_proposed",
                            proposal=spec, trigger_evidence=source)
    return {"candidate_id": candidate_id, "status": "idea", "duplicate": False}


def handoff(args):
    state = candidates.read_state(args.workspace, args.id)
    if state["status"] != "accepted":
        raise ValueError("handoff requires an accepted probe")
    candidates.require_transition_gates(state, "accepted")
    proposal = state["proposal"]
    verify(args.workspace, state["trigger_evidence"])
    # Freeze actual validation artifacts as well as the recorded gate decisions.
    gate_evidence = {g: artifact(args.workspace, state["validation"][g]["evidence"])
                     for g in candidates.GATES}
    for gate, ref in gate_evidence.items():
        if ref["sha256"] != state["validation"][gate].get("sha256"):
            raise ValueError(f"gate evidence changed or was never fingerprinted: {gate}")
    report = read(args.input)
    require(report, ("code_identity", "protocol_id", "split", "finding", "conclusion"))
    if report["split"] != "validation":
        raise ValueError("only validation evidence may drive a model edit")
    if report["conclusion"] != "supported":
        raise ValueError("null, contradicted or inconclusive findings do not authorize an edit")
    for key in ("code_identity", "protocol_id"):
        if report[key] != proposal[key]:
            raise ValueError(f"stale or mismatched {key}; propose for the new context")
    evidence = artifact(args.workspace, report["evidence"])
    task = {"candidate_id": args.id, "candidate_hash": digest(state),
            "trigger_id": proposal["trigger_id"], "code_identity": report["code_identity"],
            "protocol_id": report["protocol_id"], "finding": report["finding"],
            "action": proposal["action"], "falsification": proposal["falsification"],
            "decision_contract": state["decision_contract"],
            "acceptance_thresholds": state["acceptance"]["thresholds"],
            "budget": proposal["budget"], "gate_evidence": gate_evidence,
            "evidence": evidence, "report": artifact(args.workspace, str(args.input.resolve())),
            "trigger_evidence": state["trigger_evidence"]}
    task_id = "edit-" + digest(task)[:24]
    path = candidates.evolution_root(args.workspace) / "handoffs" / f"{task_id}.json"
    if not path.exists():
        save(path, {**task, "task_id": task_id, "status": "ready"})
        candidates.append_event(args.workspace, args.id, "edit_queued", task_id=task_id,
                                handoff=str(path))
    return {"task_id": task_id, "handoff": str(path), "status": read(path)["status"]}


def checked_task(args):
    candidates.validate_id(args.task_id)
    path = candidates.evolution_root(args.workspace) / "handoffs" / f"{args.task_id}.json"
    task = read(path)
    if task["status"] != "ready":
        raise ValueError("handoff is already completed")
    state = candidates.read_state(args.workspace, task["candidate_id"])
    if digest(state) != task["candidate_hash"]:
        raise ValueError("candidate changed since handoff; regenerate the handoff")
    if args.code_identity != task["code_identity"] or args.protocol_id != task["protocol_id"]:
        raise ValueError("handoff does not match current incumbent/protocol")
    for ref in [task["evidence"], task["report"], task["trigger_evidence"],
                *task["gate_evidence"].values()]:
        verify(args.workspace, ref)
    return path, task


def check(args):
    return checked_task(args)[1]


def close(args):
    path, task = checked_task(args)
    episode = read(args.input)
    require(episode, ("episode_id", "task_id", "outcome", "parent_code_identity",
                      "result_code_identity", "protocol_id", "evidence"))
    if episode["task_id"] != args.task_id or episode["protocol_id"] != task["protocol_id"]:
        raise ValueError("episode task/protocol mismatch")
    if episode["parent_code_identity"] != task["code_identity"]:
        raise ValueError("episode parent mismatch")
    if episode["outcome"] not in {"keep", "discard", "provisional", "crash", "invalid"}:
        raise ValueError("unknown experiment outcome")
    ref = artifact(args.workspace, episode["evidence"])
    task.update(status="completed", episode=episode, outcome_evidence=ref)
    save(path, task)
    candidates.append_event(args.workspace, task["candidate_id"], "edit_evaluated",
                            task_id=args.task_id, episode=episode, evidence=ref)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("propose", propose), ("handoff", handoff), ("check", check), ("close", close)):
        p = sub.add_parser(name)
        p.add_argument("workspace", type=Path)
        if name in {"propose", "handoff", "close"}:
            p.add_argument("--input", type=Path, required=True)
        if name == "handoff":
            p.add_argument("--id", required=True)
        if name in {"check", "close"}:
            p.add_argument("--task-id", required=True)
            p.add_argument("--code-identity", required=True)
            p.add_argument("--protocol-id", required=True)
        p.set_defaults(handler=handler)
    args = parser.parse_args()
    print(json.dumps(args.handler(args), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
