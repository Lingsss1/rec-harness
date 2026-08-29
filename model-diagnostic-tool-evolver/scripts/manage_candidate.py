#!/usr/bin/env python3
"""Manage an auditable diagnostic-tool candidate lifecycle."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parent.parent
ASSET_DIR = SKILL_DIR / "assets"
ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
GATES = (
    "paper_fidelity",
    "synthetic_correctness",
    "robustness",
    "shadow_real_data",
    "incremental_utility",
    "cost",
)
TRANSITIONS = {
    "idea": {"researched", "rejected"},
    "researched": {"implemented", "rejected"},
    "implemented": {"synthetic_pass", "rejected"},
    "synthetic_pass": {"shadow", "rejected"},
    "shadow": {"accepted", "rejected"},
    "accepted": {"superseded"},
    "rejected": set(),
    "superseded": set(),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_id(candidate_id: str) -> None:
    if not ID_PATTERN.fullmatch(candidate_id):
        raise ValueError("candidate ID must use 1-64 lowercase letters, digits, or internal hyphens")


def evolution_root(workspace: Path) -> Path:
    return workspace.resolve() / "memory" / "tool-evolution"


def candidate_dir(workspace: Path, candidate_id: str) -> Path:
    validate_id(candidate_id)
    return evolution_root(workspace) / "candidates" / candidate_id


def state_path(workspace: Path, candidate_id: str) -> Path:
    return candidate_dir(workspace, candidate_id) / "candidate.json"


def registry_path(workspace: Path) -> Path:
    return evolution_root(workspace) / "registry.jsonl"


def replace_tokens(value: Any, tokens: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in tokens.items():
            value = value.replace("{{" + key + "}}", replacement)
        return value
    if isinstance(value, list):
        return [replace_tokens(item, tokens) for item in value]
    if isinstance(value, dict):
        return {key: replace_tokens(item, tokens) for key, item in value.items()}
    return value


def read_state(workspace: Path, candidate_id: str) -> dict[str, Any]:
    path = state_path(workspace, candidate_id)
    if not path.is_file():
        raise FileNotFoundError(f"candidate not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"candidate state must be an object: {path}")
    return value


def write_state(workspace: Path, candidate_id: str, state: dict[str, Any]) -> None:
    path = state_path(workspace, candidate_id)
    state["updated_at"] = now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_event(workspace: Path, candidate_id: str, event: str, **fields: Any) -> None:
    path = registry_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": now(),
        "candidate_id": candidate_id,
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def command_init(args: argparse.Namespace) -> dict[str, Any]:
    validate_id(args.id)
    directory = candidate_dir(args.workspace, args.id)
    if directory.exists():
        raise FileExistsError(f"candidate already exists: {directory}")
    directory.mkdir(parents=True)
    for child in ("src", "tests", "artifacts"):
        (directory / child).mkdir()

    created_at = now()
    tokens = {
        "candidate_id": args.id,
        "created_at": created_at,
        "gap": args.gap,
        "decision": args.decision,
        "baseline": args.baseline,
        "budget": args.budget,
    }
    template = json.loads((ASSET_DIR / "candidate.template.json").read_text(encoding="utf-8"))
    state = replace_tokens(template, tokens)
    state_path(args.workspace, args.id).write_text(
        json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    method_card = replace_tokens(
        (ASSET_DIR / "method-card.template.md").read_text(encoding="utf-8"), tokens
    )
    (directory / "method-card.md").write_text(method_card, encoding="utf-8")
    append_event(
        args.workspace,
        args.id,
        "initialized",
        from_status=None,
        to_status="idea",
        gap=args.gap,
        target_decision=args.decision,
        budget=args.budget,
    )
    return {"candidate_dir": str(directory), "status": "idea"}


def command_paper(args: argparse.Namespace) -> dict[str, Any]:
    state = read_state(args.workspace, args.id)
    if state["status"] not in {"idea", "researched"}:
        raise ValueError("papers may be recorded only before implementation")
    record = {
        "title": args.title,
        "url": args.url,
        "version": args.version,
        "official_code_url": args.code_url,
        "license": args.license,
        "role": args.role,
    }
    if any(item.get("url") == args.url for item in state["papers"]):
        raise ValueError(f"paper URL already recorded: {args.url}")
    state["papers"].append(record)
    write_state(args.workspace, args.id, state)
    append_event(args.workspace, args.id, "paper_recorded", paper=record)
    return {"candidate_id": args.id, "paper_count": len(state["papers"])}


def command_precommit(args: argparse.Namespace) -> dict[str, Any]:
    state = read_state(args.workspace, args.id)
    if state["status"] not in {"idea", "researched"}:
        raise ValueError("acceptance gates must be precommitted before implementation")
    if any(item["result"] != "pending" for item in state["validation"].values()):
        raise ValueError("cannot change precommitment after a validation result exists")
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("thresholds file must contain a non-empty JSON object")
    state["opportunity_score"] = args.opportunity_score
    state["decision_contract"] = {
        "input_schema": args.input_schema,
        "output_schema": args.output_schema,
        "null_behavior": args.null_behavior,
        "action_mapping": args.action_mapping,
    }
    state["acceptance"]["thresholds"] = thresholds
    state["acceptance"]["preset_before_results"] = True
    write_state(args.workspace, args.id, state)
    append_event(
        args.workspace,
        args.id,
        "acceptance_precommitted",
        opportunity_score=args.opportunity_score,
        thresholds=thresholds,
        decision_contract=state["decision_contract"],
    )
    return {"candidate_id": args.id, "preset_before_results": True}


def command_gate(args: argparse.Namespace) -> dict[str, Any]:
    state = read_state(args.workspace, args.id)
    if not state["acceptance"]["preset_before_results"]:
        raise ValueError("precommit acceptance thresholds before recording gate results")
    entry = state["validation"][args.gate]
    previous = entry["result"]
    entry.update({
        "result": args.result,
        "evidence": args.evidence,
        "note": args.note,
        "updated_at": now(),
    })
    if args.evidence not in state["artifacts"]:
        state["artifacts"].append(args.evidence)
    write_state(args.workspace, args.id, state)
    append_event(
        args.workspace,
        args.id,
        "gate_recorded",
        gate=args.gate,
        previous_result=previous,
        result=args.result,
        evidence=args.evidence,
        note=args.note,
    )
    return {"candidate_id": args.id, "gate": args.gate, "result": args.result}


def require_transition_gates(state: dict[str, Any], target: str) -> None:
    required_by_target = {
        "synthetic_pass": ("paper_fidelity", "synthetic_correctness", "robustness"),
        "shadow": ("shadow_real_data",),
        "accepted": GATES,
    }
    required = required_by_target.get(target, ())
    missing = [gate for gate in required if state["validation"][gate]["result"] != "pass"]
    if missing:
        raise ValueError(f"cannot transition to {target}; gates not passed: {missing}")
    if target == "researched":
        if not state["papers"]:
            raise ValueError("record at least one primary paper before researched")
        if not state["acceptance"]["preset_before_results"]:
            raise ValueError("precommit acceptance thresholds before researched")
    if target == "accepted" and not state["acceptance"]["preset_before_results"]:
        raise ValueError("accepted requires precommitted thresholds")


def command_transition(args: argparse.Namespace) -> dict[str, Any]:
    state = read_state(args.workspace, args.id)
    current = state["status"]
    if args.to not in TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid transition: {current} -> {args.to}")
    require_transition_gates(state, args.to)
    state["status"] = args.to
    if args.to in {"accepted", "rejected", "superseded"}:
        state["decision"] = {"status": args.to, "note": args.note, "evidence": args.evidence}
    write_state(args.workspace, args.id, state)
    append_event(
        args.workspace,
        args.id,
        "transitioned",
        from_status=current,
        to_status=args.to,
        evidence=args.evidence,
        note=args.note,
    )
    return {"candidate_id": args.id, "from": current, "to": args.to}


def command_promote(args: argparse.Namespace) -> dict[str, Any]:
    state = read_state(args.workspace, args.id)
    if state["status"] != "accepted":
        raise ValueError("only an accepted candidate may be recorded as promoted")
    append_event(
        args.workspace,
        args.id,
        "promoted",
        evidence=args.evidence,
        note=args.note,
    )
    return {"candidate_id": args.id, "promoted": True}


def readiness(state: dict[str, Any]) -> dict[str, Any]:
    pending = [name for name in GATES if state["validation"][name]["result"] == "pending"]
    failed = [name for name in GATES if state["validation"][name]["result"] == "fail"]
    return {
        "candidate_id": state["candidate_id"],
        "status": state["status"],
        "paper_count": len(state["papers"]),
        "preset_before_results": state["acceptance"]["preset_before_results"],
        "pending_gates": pending,
        "failed_gates": failed,
        "ready_for_acceptance": not pending and not failed and state["acceptance"]["preset_before_results"],
    }


def command_status(args: argparse.Namespace) -> dict[str, Any]:
    return readiness(read_state(args.workspace, args.id))


def add_common_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--id", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="create an isolated candidate")
    add_common_id(initialize)
    initialize.add_argument("--gap", required=True)
    initialize.add_argument("--decision", required=True)
    initialize.add_argument("--baseline", required=True)
    initialize.add_argument("--budget", required=True)
    initialize.set_defaults(handler=command_init)

    paper = subparsers.add_parser("paper", help="record a primary paper")
    add_common_id(paper)
    paper.add_argument("--title", required=True)
    paper.add_argument("--url", required=True)
    paper.add_argument("--version", required=True)
    paper.add_argument("--code-url")
    paper.add_argument("--license")
    paper.add_argument("--role", choices=("chosen", "alternative", "counterevidence"), required=True)
    paper.set_defaults(handler=command_paper)

    precommit = subparsers.add_parser("precommit", help="freeze decision contract and thresholds")
    add_common_id(precommit)
    precommit.add_argument("--opportunity-score", type=int, choices=range(7, 11), required=True)
    precommit.add_argument("--thresholds", type=Path, required=True)
    precommit.add_argument("--input-schema", required=True)
    precommit.add_argument("--output-schema", required=True)
    precommit.add_argument("--null-behavior", required=True)
    precommit.add_argument("--action-mapping", required=True)
    precommit.set_defaults(handler=command_precommit)

    gate = subparsers.add_parser("gate", help="record a validation gate")
    add_common_id(gate)
    gate.add_argument("--gate", choices=GATES, required=True)
    gate.add_argument("--result", choices=("pending", "pass", "fail"), required=True)
    gate.add_argument("--evidence", required=True)
    gate.add_argument("--note", required=True)
    gate.set_defaults(handler=command_gate)

    transition = subparsers.add_parser("transition", help="advance or reject a candidate")
    add_common_id(transition)
    transition.add_argument("--to", choices=tuple(TRANSITIONS), required=True)
    transition.add_argument("--evidence", required=True)
    transition.add_argument("--note", required=True)
    transition.set_defaults(handler=command_transition)

    promote = subparsers.add_parser("promote", help="append the stable-tool promotion event")
    add_common_id(promote)
    promote.add_argument("--evidence", required=True)
    promote.add_argument("--note", required=True)
    promote.set_defaults(handler=command_promote)

    status = subparsers.add_parser("status", help="show candidate readiness")
    add_common_id(status)
    status.set_defaults(handler=command_status)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
