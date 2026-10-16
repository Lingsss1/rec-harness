"""Report every preregistered cell, including failed and missing trajectories."""
from __future__ import annotations

import csv
import datetime as dt
import math
import random
import statistics
from pathlib import Path

from .contracts import direction_id
from .storage import digest, read, write
from .study import matrix


def normalized(score, baseline, task):
    if score is None:
        return None
    sign = 1 if task["direction"] == "higher" else -1
    return max(0.0, sign * (score - baseline) / abs(task["metric_best"] - task["metric_worst"]))


def mean(values):
    return statistics.mean(values) if values else None


def bootstrap(values):
    if len(values) < 2:
        return None
    rng = random.Random(20261015)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[249], draws[9749]]


def trajectory_summary(state, task, directory):
    records = state["records"]
    curve = [normalized(r["best_metric"], task["baseline_validation"], task) for r in records]
    final = curve[-1] if curve else 0
    late = (final - curve[49]) / final if len(curve) == 100 and final > 0 else None
    seen, duplicates, optimization_count = set(), 0, 0
    for record in records:
        # Deliberate confirmation is not accidental search redundancy.
        if record["diagnostic"] is not None:
            continue
        optimization_count += 1
        fingerprint = digest({"parent": 0 if record["parent_trial"] == 0 else records[record["parent_trial"] - 1]["artifact_hash"], "diff": record["diff"]})
        duplicates += fingerprint in seen
        seen.add(fingerprint)
    post_decision = []
    for stage, decision in state["final_reviews"].items():
        end = int(stage) * 10
        if len(records) >= end + 10:
            gain = curve[end + 9] - curve[end - 1]
            raw_gain = records[end + 9]["best_metric"] - records[end - 1]["best_metric"]
            post_decision.append({"stage": int(stage), "decision": decision["decision"],
                                  "next_stage_normalized_gain": gain, "next_stage_raw_change": raw_gain})
    changed = 0
    for stage, pre in state["pre_reviews"].items():
        final_review = state["final_reviews"].get(stage)
        if pre["diagnostics"] and final_review:
            changed += (pre["decision"], direction_id(pre["next_direction"])) != (final_review["decision"], direction_id(final_review["next_direction"]))
    usage = {k: 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "total_cost_usd", "elapsed_seconds")}
    missing_usage = 0
    for call in sorted((directory / "private" / "calls").glob("*")):
        path = call / "usage.json"
        if not path.exists():
            missing_usage += 1
            continue
        item = read(path)
        if any(item.get(k) is None for k in ("input_tokens", "output_tokens")):
            missing_usage += 1
        for key in usage:
            value = item.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                usage[key] += value
    result = state.get("test") or {}
    metric = result.get("metric") if result.get("status") == "completed" else None
    complete = len(records) == 100 and state["status"] == "completed" and metric is not None
    elapsed = None
    if state.get("completed_at"):
        elapsed = (dt.datetime.fromisoformat(state["completed_at"]) - dt.datetime.fromisoformat(state["created_at"])).total_seconds()
    return {"task": state["task"], "arm": state["arm"], "replicate": state["replicate"],
            "complete": complete, "status": state["status"], "trials": len(records),
            "test_metric": metric, "normalized_test_improvement": normalized(metric, task["baseline_test"], task),
            "validation_auc": mean(curve) if len(curve) == 100 else None,
            "late_gain_fraction": late,
            "invalid_trial_fraction": mean([not r["valid"] for r in records]),
            "exact_repeat_fraction": duplicates / optimization_count if optimization_count else None,
            "exact_repeat_definition": "non-diagnostic re-proposal with identical parent artifact and code diff",
            "diagnostic_fraction": mean([r["diagnostic"] is not None for r in records]),
            "diagnostic_decisions_changed": changed, "post_decision_gains": post_decision,
            "training_seconds_observed": sum(r["training_seconds"] or 0 for r in records),
            "training_time_missing_trials": sum(r["training_seconds"] is None for r in records),
            "validation_elapsed_seconds_observed": sum(r["elapsed_seconds"] or 0 for r in records),
            "trajectory_wall_seconds": elapsed, "agent_usage_observed": usage,
            "agent_calls_missing_usage": missing_usage,
            "direction_labels": "agent-declared; semantic equivalence requires separate blinded audit"}


def report(root, registration, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Report output must be disjoint from the registered study")
    rows, curves = [], []
    for cell in matrix(registration["config"]):
        directory = root / "trajectories" / cell["task"] / cell["arm"] / str(cell["replicate"])
        path = directory / "state.json"
        if not path.exists():
            rows.append(dict(cell, complete=False, status="missing", trials=0))
            continue
        state = read(path)
        if state["registration_id"] != registration["registration_id"]:
            raise ValueError("Trajectory registration mismatch")
        # Validate receipt chain and artifact identities before trusting a published report.
        from .engine import Trajectory
        checker = Trajectory(root, registration, cell["task"], cell["arm"], cell["replicate"], None)
        checker.verify_events(state)
        task = registration["config"]["tasks"][cell["task"]]
        rows.append(trajectory_summary(state, task, directory))
        for record in state["records"]:
            curves.append({**cell, "trial": record["trial"], "valid": record["valid"],
                           "metric": record["metric"], "best_metric": record["best_metric"],
                           "normalized_validation_gain": normalized(record["best_metric"], task["baseline_validation"], task)})
    comparisons = []
    for task_name, task in registration["config"]["tasks"].items():
        for first, second in (("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")):
            differences = []
            for rep in range(1, 6):
                a = next(r for r in rows if (r["task"], r["arm"], r["replicate"]) == (task_name, first, rep))
                b = next(r for r in rows if (r["task"], r["arm"], r["replicate"]) == (task_name, second, rep))
                if a["complete"] and b["complete"]:
                    differences.append((b["test_metric"] - a["test_metric"]) * (1 if task["direction"] == "higher" else -1))
            comparisons.append({"task": task_name, "contrast": second + "-" + first,
                "complete_pairs": len(differences), "raw_improvement_differences": differences,
                "mean_raw_improvement": mean(differences), "bootstrap_95_interval": bootstrap(differences),
                "confirmatory_complete": len(differences) == 5})
    aggregates = []
    for arm in "ABCD":
        selected = [r for r in rows if r["arm"] == arm]
        complete = all(r["complete"] for r in selected)
        aggregates.append({"arm": arm, "completed": sum(r["complete"] for r in selected),
                           "expected": 15, "macro_normalized_test_improvement":
                           mean([r["normalized_test_improvement"] for r in selected]) if complete else None})
    output.mkdir(parents=True, exist_ok=True)
    summary = {"registration_id": registration["registration_id"], "expected_trajectories": 60,
               "completed_trajectories": sum(r["complete"] for r in rows), "trajectories": rows,
               "comparisons": comparisons, "aggregates": aggregates,
               "limitations": ["Five repetitions give imprecise exploratory bootstrap intervals.",
                  "Same validation-call budget is not the same total cost.",
                  "Exact code repetition is not semantic repetition.",
                  "Decision changes do not prove causal attribution is correct.",
                  "Missing/failed cells are retained; no complete-suite aggregate is published for incomplete arms."]}
    execution = registration["config"].get("execution", {"backend": "docker"})
    summary["execution"] = execution
    if execution.get("isolation") == "process":
        summary["limitations"].append("Trusted local process mode does not enforce filesystem/network isolation or prevent access to old sessions and hidden assets; full isolation claims require an audited OS boundary.")
    write(output / "summary.json", summary)
    with (output / "curves.csv").open("w", newline="") as f:
        fields = ["task", "arm", "replicate", "trial_budget", "trial", "valid", "metric", "best_metric", "normalized_validation_gain"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curves)
    text = ["# Staged autoresearch results", "", "Completed: %d / 60. Missing and failed cells remain in summary.json." % summary["completed_trajectories"], "",
            "| Task | Contrast | Complete pairs | Mean raw improvement |", "|---|---|---:|---:|"]
    for item in comparisons:
        value = item["mean_raw_improvement"]
        text.append("| %s | %s | %d / 5 | %s |" % (item["task"], item["contrast"], item["complete_pairs"], "unavailable" if value is None else "%.6f" % value))
    text += ["", "Positive differences favor the later arm. Partial comparisons are exploratory.", ""] + ["- " + x for x in summary["limitations"]]
    (output / "report.md").write_text("\n".join(text) + "\n")
    return summary
