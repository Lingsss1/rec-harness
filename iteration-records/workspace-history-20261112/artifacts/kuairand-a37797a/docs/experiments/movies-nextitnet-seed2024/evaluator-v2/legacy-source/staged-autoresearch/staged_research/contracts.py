"""Small explicit contracts for prospective plans, evidence, and decisions."""
from __future__ import annotations

import math

from .storage import ProtocolError, digest


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1}
NUMBER = {"type": "integer", "minimum": 0}
REFS = {"type": "array", "items": NUMBER, "minItems": 1}
DIRECTION = obj({"bottleneck": TEXT, "mechanism": TEXT})
STEP = obj({"slot": {"type": "integer", "minimum": 1, "maximum": 10},
            "instruction": TEXT, "condition": TEXT, "expected_observation": TEXT})
PLAN = obj({"direction": DIRECTION, "hypothesis": TEXT, "falsification": TEXT,
            "observables": {"type": "array", "items": TEXT, "minItems": 1},
            "steps": {"type": "array", "items": STEP, "minItems": 10, "maxItems": 10},
            "revision_reason": TEXT})
DIAGNOSTIC = obj({"kind": {"enum": ["ablation", "rollback", "paired", "repeat"]},
                  "parent_trial": NUMBER, "paired_with": NUMBER, "instruction": TEXT,
                  "uncertainty": TEXT, "decision_if_supported": TEXT,
                  "decision_if_refuted": TEXT})
REVIEW = obj({
    "facts": {"type": "array", "minItems": 1,
              "items": obj({"claim": TEXT, "trial_refs": REFS})},
    "interpretations": {"type": "array", "items": obj({"explanation": TEXT, "trial_refs": REFS})},
    "uncertainties": {"type": "array", "items": TEXT},
    "decision": {"enum": ["continue", "switch", "revisit"]},
    "next_direction": DIRECTION, "reason_refs": REFS,
    "diagnostics": {"type": "array", "items": DIAGNOSTIC, "maxItems": 2},
})
EXPERIMENT = obj({"hypothesis": TEXT, "change_description": TEXT,
                  "expected_observation": TEXT})


def validate_shape(value, schema, path="response"):
    if "enum" in schema and value not in schema["enum"]:
        raise ProtocolError(path + ": invalid choice")
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ProtocolError(path + ": expected object")
        if set(value) - set(schema["properties"]) or set(schema["required"]) - set(value):
            raise ProtocolError(path + ": unexpected or missing fields")
        for key, child in value.items():
            validate_shape(child, schema["properties"][key], path + "." + key)
    elif kind == "array":
        if not isinstance(value, list) or len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ProtocolError(path + ": invalid array length")
        for child in value:
            validate_shape(child, schema["items"], path + "[]")
    elif kind == "string":
        if not isinstance(value, str) or len(value.strip()) < schema.get("minLength", 0):
            raise ProtocolError(path + ": expected non-empty text")
    elif kind == "integer":
        if not isinstance(value, int) or isinstance(value, bool) or not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ProtocolError(path + ": invalid integer")


def direction_id(direction):
    clean = {k: " ".join(v.lower().split()) for k, v in direction.items()}
    return digest(clean)[:16]


def validate_plan(plan, expected_direction=None):
    validate_shape(plan, PLAN)
    if sorted(s["slot"] for s in plan["steps"]) != list(range(1, 11)):
        raise ProtocolError("A plan must cover each of the 10 slots exactly once")
    if expected_direction is not None and direction_id(plan["direction"]) != direction_id(expected_direction):
        raise ProtocolError("Plan silently changed the direction selected at the previous review")


def validate_review(review, records, current_direction, known_directions, diagnostics_allowed, final=False):
    validate_shape(review, REVIEW)
    valid_refs = {0} | {r["trial"] for r in records}
    for refs in ([review["reason_refs"]] + [x["trial_refs"] for x in review["facts"] + review["interpretations"]]):
        if not set(refs) <= valid_refs:
            raise ProtocolError("Review cites an experiment that has not occurred")
    current, chosen = direction_id(current_direction), direction_id(review["next_direction"])
    if review["decision"] == "continue" and chosen != current:
        raise ProtocolError("Parameter refinement must retain the same direction identity")
    if review["decision"] == "switch" and (chosen == current or chosen in known_directions):
        raise ProtocolError("Switch must open a previously untried direction; use revisit otherwise")
    if review["decision"] == "revisit" and (chosen == current or chosen not in known_directions):
        raise ProtocolError("Revisit must reference a distinct, previously executed direction")
    if review["diagnostics"] and (not diagnostics_allowed or final):
        raise ProtocolError("Diagnostics are only admitted in D's pre-tail review")
    scored = {0} | {r["trial"] for r in records if r["valid"]}
    for probe in review["diagnostics"]:
        if probe["parent_trial"] not in scored or probe["paired_with"] not in scored:
            raise ProtocolError("Diagnostic anchor/comparator must be a scored version")
        # A changed parent is a confound. Pair against the very artifact being intervened on.
        if probe["parent_trial"] != probe["paired_with"]:
            raise ProtocolError("Paired diagnostic must start from its comparator's exact version")
        if probe["decision_if_supported"] == probe["decision_if_refuted"]:
            raise ProtocolError("Diagnostic must distinguish consequential decisions")


def score_result(payload, task):
    if not isinstance(payload, dict) or payload.get("constraints_ok") is not True:
        raise ProtocolError("Missing or failed constraint check")
    score = payload.get("metric")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ProtocolError("Metric must be finite and on the frozen [0,1] scale")
    if task["fml_name"] == "Privacy_opacus":
        checks = payload.get("constraint_checks", {})
        if checks.get("target_epsilon") != 8 or checks.get("target_delta") != 1e-5:
            raise ProtocolError("Opacus target privacy budget changed")
        for key, maximum in (("actual_epsilon", 8.0), ("actual_delta", 1e-5)):
            v = checks.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= maximum:
                raise ProtocolError("Opacus privacy accounting failed: " + key)
    seconds = payload.get("training_seconds")
    if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
        raise ProtocolError("Invalid measured training time")
    return float(score)
