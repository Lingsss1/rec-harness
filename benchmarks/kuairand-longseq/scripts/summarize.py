#!/usr/bin/env python3
"""Summarize measured final checkpoints, refusing to mix smoke and real runs."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longseq_bench.registry import MODEL_NAMES, MODEL_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--require-complete", action="store_true",
                        help="Fail unless all four fixed-final-epoch results exist")
    args = parser.parse_args()
    results = []
    identities = set()
    for name in MODEL_NAMES:
        directory = args.runs / name
        if not (directory / "final.json").exists():
            results.append({"model": name, "status": "not_completed"})
            continue
        final = json.loads((directory / "final.json").read_text())
        run = json.loads((directory / "run.json").read_text())
        if run["model"]["name"] != name:
            raise ValueError("Wrong model in result directory: " + name)
        if final["fingerprint"] != run["fingerprint"]:
            raise ValueError("Final/run identity mismatch: " + name)
        if run["run_kind"] == "synthetic_smoke" and not args.include_smoke:
            raise ValueError("Synthetic smoke output is not a KuaiRand result: " + name)
        if not final.get("training_complete") or final["epoch"] != run["training"]["epochs"] - 1:
            raise ValueError("Run is not at its fixed final epoch: " + name)
        identities.add(json.dumps({"data": run["source_csv_sha256"],
            "cache_files": run["cache_files"], "history": run["model"]["max_history"],
            "catalog": run["model"]["num_items"], "dim": run["model"]["dim"],
            "statistics": run.get("data_statistics"), "paper_statistics_match": run["paper_statistics_match"],
            "preprocessing": run["preprocessing_manifest"], "evaluation": run["evaluation"],
            "precision": run["precision"], "kind": run["run_kind"],
            "training": run["training"], "world_size": run["world_size"],
            "accumulation": run["gradient_accumulation"], "code": run["code_files"]}, sort_keys=True))
        # Interrupted runs may have logged an epoch before saving its checkpoint.
        # Use the last record for each epoch rather than double-counting it.
        records = {}
        metrics_path = directory / "metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                record = json.loads(line)
                if record["fingerprint"] != run["fingerprint"]:
                    raise ValueError("Metric/run identity mismatch: " + name)
                if record["epoch"] <= final["epoch"]:
                    records[record["epoch"]] = record
        peaks = [cost["peak_allocated_bytes"] for record in records.values()
                 for cost in record.get("rank_training_costs", []) if cost["peak_allocated_bytes"] is not None]
        costs = {"training_seconds": sum(max((cost["seconds"] for cost in record.get("rank_training_costs", [])),
                                              default=0) for record in records.values()),
                 "evaluation_seconds": sum(record.get("evaluation_seconds", 0) for record in records.values()),
                 "peak_training_allocated_bytes": max(peaks) if peaks else None,
                 "logged_epochs": len(records)}
        results.append({"model": name, "status": "completed", "run_kind": run["run_kind"],
                        "model_label": MODEL_LABELS[name],
                        "adaptation": run.get("adaptations", {}).get(name),
                        "epoch": final["epoch"], "metrics": final["metrics"],
                        "parameters": run["parameters"], "fingerprint": run["fingerprint"],
                        "source_csv_sha256": run["source_csv_sha256"],
                        "data_statistics": run.get("data_statistics"),
                        "paper_statistics_match": run["paper_statistics_match"],
                        "preprocessing_profile": (run.get("preprocessing_manifest") or {}).get("profile"),
                        "precision": run["precision"], "max_history": run["model"]["max_history"],
                        "world_size": run["world_size"], "costs": costs})
    if len(identities) > 1:
        raise ValueError("Runs use different data, training protocol, runtime topology or code")
    if args.require_complete and any(row["status"] != "completed" for row in results):
        raise ValueError("All four models must complete before reporting success")
    (args.runs / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["| Model | Status | NDCG@10 | NDCG@50 | HR@10 | HR@50 | MRR (reference) | MRR (exact) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        metrics = result.get("metrics", {})
        numbers = ["%.6f" % metrics[key] if key in metrics else "—"
                   for key in ("ndcg@10", "ndcg@50", "hr@10", "hr@50", "mrr", "mrr_exact")]
        lines.append("| " + " | ".join([result.get("model_label", MODEL_LABELS[result["model"]]), result["status"], *numbers]) + " |")
    for result in results:
        if result["status"] == "completed":
            lines.append("\n%s: kind=%s; history=%s; precision=%s; profile=%s; paper statistics match=%s; statistics=%s" %
                         (result["model"], result["run_kind"], result["max_history"], result["precision"],
                          result["preprocessing_profile"], result["paper_statistics_match"], result["data_statistics"]))
            if result.get("adaptation"):
                lines.append("\nAdaptation: " + result["adaptation"])
    markdown = "\n".join(lines) + "\n"
    (args.runs / "summary.md").write_text(markdown)
    csv_rows = []
    for row in results:
        csv_rows.append({**{key: value for key, value in row.items() if key not in ("metrics", "costs", "data_statistics")},
                         **row.get("metrics", {}), **row.get("costs", {}),
                         **{"data_" + key: value for key, value in (row.get("data_statistics") or {}).items()}})
    with (args.runs / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in csv_rows for key in row)))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(markdown, end="")


if __name__ == "__main__":
    main()
