"""Read-only source cards, explicit applicability and evidence-bound diff records."""

from __future__ import annotations

import ast
import operator
from pathlib import Path

from .common import digest, file_hash, read, write

OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def literal_settings(source):
    """Keep inspectable settings aligned with the actual imported candidate."""
    tree = ast.parse(Path(source).read_text())
    declarations = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SETTINGS" for t in node.targets
        ):
            declarations.append(ast.literal_eval(node.value))
    if len(declarations) != 1 or not isinstance(declarations[0], dict):
        raise ValueError("Candidate must declare exactly one literal SETTINGS dictionary")
    return declarations[0]


def model_facts(source, metadata, diagnostics=None):
    """Read literal SETTINGS without importing or executing agent-owned source."""
    settings = literal_settings(source)
    facts = {
        "task.candidate_count": metadata["n_candidates"],
        "task.history_limit": metadata["history_limit"],
        "task.candidate_mm_coverage": metadata["candidate_mm_coverage"],
        "task.release": metadata["release"],
    }
    facts.update({"model." + k: v for k, v in settings.items()})
    if diagnostics:
        facts.update(
            {"diagnostic." + k: diagnostics[k] for k in ["random_win_rate", "mean_target_rank"]}
        )
    return {
        "facts": facts,
        "source_sha256": file_hash(source),
        "metadata_identity": digest(metadata),
        "limitations": "SETTINGS declarations and measured quantities; not full architecture proof",
    }


def applicability(card, facts):
    checks = []
    for condition in card.get("conditions", []):
        field, op = condition["field"], condition["op"]
        if op not in OPS:
            raise ValueError("Unsupported condition operator")
        status = "unknown"
        if field in facts:
            status = "satisfied" if OPS[op](facts[field], condition["value"]) else "not_satisfied"
        checks.append({**condition, "observed": facts.get(field), "status": status})
    return checks


def memory_cards(export):
    if export.get("schema") != "taac-stage-memory-v1":
        raise ValueError("Unsupported prior Agent memory export")
    cards = []
    for r in export["records"]:
        cards.append(
            {
                "id": "prior-" + digest(r)[:24],
                "scene": r["scene"],
                "method": r["method"],
                "gain": r["gain"],
                "reason": r["reason"],
                "sources": [r["evidence"]],
                "source_status": "prior_run_unverified",
                "conditions": [
                    {"field": "task.release", "op": "eq", "value": r["scene"]["release"]}
                ],
                "counterexamples": ["A prior outcome is not independent current evidence"],
                "previous_status": r["status"],
                "previous_parent": r["parent"],
            }
        )
    return cards


def validate_assessments(design, context):
    """Validate coverage/provenance, not the truth of the Agent's scientific judgment."""
    from .retrieval import evidence_refs

    cards = {c["id"]: c for c in context["knowledge"]}
    assessments = design.get("knowledge_assessments", [])
    ids = [a.get("knowledge_id") for a in assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(cards):
        raise ValueError("Agent must assess every supplied card, including linked counterexamples")
    valid_refs = evidence_refs(context) | {"knowledge:" + cid for cid in cards}

    def judgment(value):
        if value.get("status") not in {"applicable", "not_applicable", "unknown"}:
            raise ValueError("Assessment needs applicable/not_applicable/unknown status")
        if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
            raise ValueError("Assessment requires rationale")
        refs = value.get("evidence_refs")
        if not isinstance(refs, list) or not set(refs) <= valid_refs:
            raise ValueError("Assessment evidence_refs must resolve")
        if value["status"] != "unknown" and not refs:
            raise ValueError("Definite assessment must cite evidence")

    for a in assessments:
        card = cards[a["knowledge_id"]]
        judgment(a)
        if a.get("condition_checks") != card["condition_checks"]:
            raise ValueError("Assessment must retain host applicability including unknowns")
        checks = a.get("counterexample_checks", [])
        if [c.get("counterexample") for c in checks] != card.get("counterexamples", []):
            raise ValueError("Assessment must retain every inline counterexample in order")
        for check in checks:
            judgment(check)
        linked = a.get("linked_counterexample_checks", [])
        if [c.get("knowledge_id") for c in linked] != card.get("counterexample_ids", []):
            raise ValueError("Assessment must review every linked counterexample in order")
        for check in linked:
            judgment(check)
    return {a["knowledge_id"]: a for a in assessments}


def validate_design(design, context, enhanced):
    for key in ["hypothesis", "change", "falsification", "controls", "budget_rationale"]:
        if not design.get(key):
            raise ValueError("Design missing " + key)
    if not enhanced:
        return
    assessments = validate_assessments(design, context)
    cards = {c["id"]: c for c in context["knowledge"]}
    diffs = design.get("diffs", [])
    if not diffs or sum(d.get("selected") is True for d in diffs) != 1:
        raise ValueError("Enhanced design requires diffs and exactly one selected direction")
    for diff in diffs:
        if diff.get("knowledge_id") not in cards:
            raise ValueError("Diff references knowledge outside supplied context")
        assessment = assessments[diff["knowledge_id"]]
        if (
            diff.get("selected")
            and assessment["status"] != "applicable"
            and not diff.get("exploration_rationale")
        ):
            raise ValueError("Selecting unknown/inapplicable knowledge needs exploration_rationale")
        refs = diff.get("current_refs")
        if not refs or not set(refs) <= set(context["model_state"]["facts"]):
            raise ValueError("Diff current_refs must resolve to supplied facts")
        for key in ["alternatives", "missing_evidence", "experiment", "priority_reason"]:
            if key not in diff:
                raise ValueError("Diff missing " + key)
        expected = cards[diff["knowledge_id"]]["condition_checks"]
        if diff.get("condition_checks") != expected:
            raise ValueError("Diff must retain host applicability results including unknowns")


def consolidate(root, state, contexts, output):
    """Stage summary preserves observations and explanations as separate records."""
    if state.get("pending"):
        raise ValueError("Unsettled trial cannot be consolidated")
    summary = []
    for trial in state["trials"]:
        directory = Path(root) / "trials" / trial["id"]
        design = (
            read(directory / "design.json")
            if (directory / "design.json").exists()
            else {"change": "interrupted before design archive"}
        )
        analysis = (
            read(directory / "analysis.json") if (directory / "analysis.json").exists() else None
        )
        summary.append(
            {
                "scene": contexts["task"],
                "method": design["change"],
                "gain": trial.get("delta"),
                "reason": analysis,
                "reason_status": "provisional" if analysis else "unknown",
                "status": trial["status"],
                "trial_id": trial["id"],
                "parent": trial["parent"],
                "source_sha256": trial["source_sha256"],
                "resource_seconds": trial["wall_seconds"],
                "evidence": {
                    "receipt": file_hash(directory / "receipt.json"),
                    "design": file_hash(directory / "design.json"),
                },
                "scope": "Single-run episodic evidence, not independent replication",
            }
        )
    write(
        output,
        {
            "schema": "taac-stage-memory-v1",
            "records": summary,
            "limitations": "No automatic causal confirmation or cross-scene generalization",
        },
    )
    return summary
