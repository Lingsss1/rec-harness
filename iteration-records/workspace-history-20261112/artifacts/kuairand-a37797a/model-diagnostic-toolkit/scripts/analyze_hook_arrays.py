#!/usr/bin/env python3
"""Analyze activation and gradient arrays exported from any model framework."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


ACTIVATION_PREFIX = "activation::"
GRADIENT_PREFIX = "gradient::"


def finite_values(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def activation_stats(array: np.ndarray) -> dict[str, Any]:
    values = finite_values(array)
    if values.size == 0:
        return {"finite_elements": 0}
    return {
        "finite_elements": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "rms": float(np.sqrt(np.mean(values * values))),
        "zero_fraction": float(np.mean(values == 0)),
        "max_abs": float(np.max(np.abs(values))),
    }


def sensitivity_stats(activation: np.ndarray, gradient: np.ndarray) -> dict[str, Any]:
    if activation.shape != gradient.shape:
        raise ValueError(
            f"activation/gradient shape mismatch: {activation.shape} vs {gradient.shape}"
        )
    a = np.asarray(activation, dtype=np.float64).reshape(-1)
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(g)
    a, g = a[finite], g[finite]
    if a.size == 0:
        return {"finite_elements": 0}
    product = np.abs(a * g)
    return {
        "finite_elements": int(a.size),
        "gradient_abs_mean": float(np.mean(np.abs(g))),
        "gradient_l2": float(np.linalg.norm(g)),
        "gradient_zero_fraction": float(np.mean(g == 0)),
        "activation_x_gradient_abs_mean": float(product.mean()),
        "activation_x_gradient_abs_sum": float(product.sum()),
    }


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def hook_metadata(name: str, activation: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("hooks", {}).get(name, {})
    if not isinstance(configured, dict):
        raise ValueError(f"config hooks.{name} must be an object")
    inferred_width = int(activation.shape[-1]) if activation.ndim >= 2 else 1
    return {
        "group": str(configured.get("group", "default")),
        "width": int(configured.get("width", inferred_width)),
        "description": str(configured.get("description", "")),
    }


def select_candidates(rows: list[dict[str, Any]], count: int, seed: int):
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["objective"], row["group"], row["width"])
        buckets.setdefault(key, []).append(row)
    rng = random.Random(seed)
    result = []
    for (objective, group, width), items in sorted(buckets.items()):
        ordered = sorted(
            items,
            key=lambda item: item["activation_x_gradient_abs_mean"],
            reverse=True,
        )
        if len(ordered) < 3:
            continue
        take = min(count, max(1, len(ordered) // 3))
        top = [item["hook"] for item in ordered[:take]]
        bottom = [item["hook"] for item in ordered[-take:]]
        middle = [
            item["hook"] for item in ordered[take:-take]
            if item["hook"] not in top and item["hook"] not in bottom
        ]
        random_control = rng.sample(middle, min(take, len(middle))) if middle else []
        result.append({
            "objective": objective,
            "group": group,
            "width": width,
            "comparable_hook_count": len(ordered),
            "top": top,
            "random": random_control,
            "bottom": bottom,
        })
    return result


def analyze(path: Path, config: dict[str, Any], candidate_count: int, seed: int):
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        activation_keys = [key for key in keys if key.startswith(ACTIVATION_PREFIX)]
        if not activation_keys:
            raise ValueError("no activation::<hook_name> arrays found")

        hooks: dict[str, Any] = {}
        rank_rows = []
        for activation_key in sorted(activation_keys):
            name = activation_key[len(ACTIVATION_PREFIX):]
            activation = np.asarray(archive[activation_key])
            if not np.issubdtype(activation.dtype, np.number):
                raise TypeError(f"{activation_key} must be numeric")
            metadata = hook_metadata(name, activation, config)
            hook_report = {
                "shape": list(activation.shape),
                **metadata,
                "activation": activation_stats(activation),
                "objectives": {},
            }
            suffix = f"::{name}"
            gradient_keys = [
                key for key in keys
                if key.startswith(GRADIENT_PREFIX) and key.endswith(suffix)
            ]
            for gradient_key in sorted(gradient_keys):
                objective = gradient_key[len(GRADIENT_PREFIX):-len(suffix)]
                gradient = np.asarray(archive[gradient_key])
                stats = sensitivity_stats(activation, gradient)
                hook_report["objectives"][objective] = stats
                rank_rows.append({
                    "hook": name,
                    "objective": objective,
                    "group": metadata["group"],
                    "width": metadata["width"],
                    **stats,
                })
            hooks[name] = hook_report

    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    objective_groups = sorted({(row["objective"], row["group"]) for row in rank_rows})
    for objective, group in objective_groups:
        selected = [
            row for row in rank_rows
            if row["objective"] == objective and row["group"] == group
        ]
        selected.sort(
            key=lambda item: item.get("activation_x_gradient_abs_mean", -math.inf),
            reverse=True,
        )
        for rank, row in enumerate(selected, 1):
            row["rank_within_group"] = rank
        rankings.setdefault(objective, {})[group] = selected

    return {
        "input": str(path.resolve()),
        "hook_count": len(hooks),
        "hooks": hooks,
        "rankings": rankings,
        "equal_width_candidates": select_candidates(rank_rows, candidate_count, seed),
        "caveats": [
            "Activation×gradient is a local first-order screening proxy, not causal importance.",
            "Ranks are comparable only within the same objective and configured Hook group.",
            "A present gradient array proves export, not that every loss component reaches the Hook.",
            "Confirm candidates with paired interventions and retrained validation outcomes.",
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
        "# Hook Activation and Gradient Diagnostic",
        "",
        f"- Input: `{report['input']}`",
        f"- Hooks: `{report['hook_count']}`",
    ]
    for objective, groups in report["rankings"].items():
        for group, rows in groups.items():
            lines.extend([
                "",
                f"## `{objective}` / `{group}`",
                "",
                "| Rank | Hook | Width | mean abs(a×g) | sum abs(a×g) | mean abs(g) |",
                "|---:|---|---:|---:|---:|---:|",
            ])
            for row in rows:
                lines.append(
                    f"| {row['rank_within_group']} | `{row['hook']}` | {row['width']} | "
                    f"{fmt(row.get('activation_x_gradient_abs_mean'))} | "
                    f"{fmt(row.get('activation_x_gradient_abs_sum'))} | "
                    f"{fmt(row.get('gradient_abs_mean'))} |"
                )
    if report["equal_width_candidates"]:
        lines.extend(["", "## Equal-width candidate suggestions", ""])
        for item in report["equal_width_candidates"]:
            lines.append(
                f"- `{item['objective']}` / `{item['group']}` / {item['width']}d: "
                f"Top={item['top']}, Random={item['random']}, Bottom={item['bottom']}"
            )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in report["caveats"])
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ containing activation:: and gradient:: arrays")
    parser.add_argument("--config", type=Path, help="Optional Hook group/width metadata JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("hook-report"))
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.candidate_count < 1:
        raise ValueError("--candidate-count must be >= 1")
    report = analyze(args.input, load_config(args.config), args.candidate_count, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (args.output_dir / "hook_report.json").resolve()
    md_path = (args.output_dir / "hook_report.md").resolve()
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
