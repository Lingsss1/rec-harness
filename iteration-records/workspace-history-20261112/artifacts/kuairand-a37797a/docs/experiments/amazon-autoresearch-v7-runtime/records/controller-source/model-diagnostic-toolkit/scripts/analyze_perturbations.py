#!/usr/bin/env python3
"""Analyze framework-independent paired Shuffle, Zero, or ablation results."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"candidate", "seed", "step", "metric", "original", "perturbed"}


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def block_bootstrap(values: list[float], block: int, iterations: int, rng: random.Random):
    if len(values) < 2 or iterations == 0:
        return None
    block = min(max(1, block), len(values))
    means = []
    for _ in range(iterations):
        sampled = []
        while len(sampled) < len(values):
            start = rng.randrange(len(values))
            sampled.extend(values[(start + offset) % len(values)] for offset in range(block))
        means.append(statistics.mean(sampled[:len(values)]))
    return [quantile(means, 0.025), quantile(means, 0.975)]


def stats(rows: list[dict[str, Any]], block: int, iterations: int, rng: random.Random):
    ordered = sorted(rows, key=lambda row: row["step"])
    deltas = [row["perturbed"] - row["original"] for row in ordered]
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    original_mean = statistics.mean(row["original"] for row in ordered)
    return {
        "n": len(deltas),
        "original_mean": original_mean,
        "perturbed_mean": statistics.mean(row["perturbed"] for row in ordered),
        "delta_mean": mean,
        "delta_median": statistics.median(deltas),
        "delta_std": std,
        "delta_standard_error": std / math.sqrt(len(deltas)),
        "delta_p05": quantile(deltas, 0.05),
        "delta_p95": quantile(deltas, 0.95),
        "relative_delta": mean / original_mean if original_mean != 0 else None,
        "positive_delta_fraction": sum(value > 0 for value in deltas) / len(deltas),
        "block_bootstrap_ci95": block_bootstrap(deltas, block, iterations, rng),
        "step_min": ordered[0]["step"],
        "step_max": ordered[-1]["step"],
    }


def load_rows(path: Path):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    result = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            raise ValueError(f"missing required columns: {missing}")
        for index, row in enumerate(reader, 2):
            try:
                result.append({
                    "candidate": row["candidate"].strip(),
                    "seed": row["seed"].strip(),
                    "step": int(float(row["step"])),
                    "metric": row["metric"].strip(),
                    "original": float(row["original"]),
                    "perturbed": float(row["perturbed"]),
                    "intervention": (row.get("intervention") or "unknown").strip(),
                    "width": int(float(row["width"])) if row.get("width") else None,
                })
            except ValueError as exc:
                raise ValueError(f"invalid numeric value on row {index}: {exc}") from exc
    if not result:
        raise ValueError("input contains no data rows")
    return result


def build_report(path: Path, block: int, iterations: int, seed: int):
    rows = load_rows(path)
    grouped: dict[tuple[str, str, int | None, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = (row["candidate"], row["intervention"], row["width"], row["seed"], row["metric"])
        grouped[key].append(row)
    rng = random.Random(seed)
    by_seed = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        candidate, intervention, width, run_seed, metric = key
        by_seed.append({
            "candidate": candidate,
            "intervention": intervention,
            "width": width,
            "seed": run_seed,
            "metric": metric,
            **stats(values, block, iterations, rng),
        })

    rollup_groups: dict[tuple[str, str, int | None, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for item in by_seed:
        rollup_groups[(item["candidate"], item["intervention"], item["width"], item["metric"])].append(item)
    rollups = []
    for key, values in sorted(rollup_groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        candidate, intervention, width, metric = key
        seed_means = [item["delta_mean"] for item in values]
        rollups.append({
            "candidate": candidate,
            "intervention": intervention,
            "width": width,
            "metric": metric,
            "seed_count": len(seed_means),
            "seeds": [item["seed"] for item in values],
            "mean_of_seed_means": statistics.mean(seed_means),
            "min_seed_mean": min(seed_means),
            "max_seed_mean": max(seed_means),
        })

    width_warnings = []
    comparison_groups: dict[tuple[str, str], set[int | None]] = collections.defaultdict(set)
    for item in rollups:
        comparison_groups[(item["intervention"], item["metric"])].add(item["width"])
    for (intervention, metric), widths in comparison_groups.items():
        concrete = {width for width in widths if width is not None}
        if len(concrete) > 1:
            width_warnings.append(
                f"{intervention}/{metric} contains multiple widths {sorted(concrete)}; do not compare candidates as equal-cost controls."
            )

    return {
        "input": str(path.resolve()),
        "row_count": len(rows),
        "configuration": {"block_size": block, "bootstrap_iterations": iterations, "bootstrap_seed": seed},
        "by_seed": by_seed,
        "cross_seed_rollups": rollups,
        "warnings": width_warnings,
        "caveats": [
            "All deltas are computed as perturbed minus original.",
            "Paired effects measure current-model reliance, not guaranteed retraining benefit.",
            "Near-zero Shuffle should be checked with Zero and raw representation variance.",
            "Use independent seeds and unchanged validation protocol before durable conclusions.",
        ],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired Perturbation Diagnostic",
        "",
        f"- Input: `{report['input']}`",
        f"- Rows: `{report['row_count']}`",
        "",
        "## Per-seed results",
        "",
        "| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in report["by_seed"]:
        interval = item["block_bootstrap_ci95"]
        interval_text = "n/a" if interval is None else f"[{fmt(interval[0])}, {fmt(interval[1])}]"
        lines.append(
            f"| `{item['candidate']}` | `{item['intervention']}` | {fmt(item['width'])} | "
            f"`{item['seed']}` | `{item['metric']}` | {item['n']} | {fmt(item['delta_mean'])} | "
            f"{interval_text} | {fmt(item['positive_delta_fraction'])} |"
        )
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in report["caveats"])
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Long-form CSV or TSV")
    parser.add_argument("--output-dir", type=Path, default=Path("perturbation-report"))
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.block_size < 1 or args.bootstrap < 0:
        raise ValueError("--block-size must be >= 1 and --bootstrap must be >= 0")
    report = build_report(args.input, args.block_size, args.bootstrap, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (args.output_dir / "perturbation_report.json").resolve()
    md_path = (args.output_dir / "perturbation_report.md").resolve()
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
