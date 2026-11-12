#!/usr/bin/env python3
"""Diagnose representation collapse in a matrix of embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LoadedArray:
    array: np.ndarray
    source: str
    key: str | None


def _select_key(value: Any, key: str | None, source: Path) -> Any:
    if isinstance(value, dict) or hasattr(value, "keys"):
        keys = list(value.keys())
        if key is None:
            if len(keys) != 1:
                shown = ", ".join(map(str, keys[:20]))
                raise ValueError(
                    f"{source} contains multiple values ({shown}); select one with --key"
                )
            key = str(keys[0])
        if key not in value:
            raise ValueError(f"key {key!r} not found in {source}; available: {keys}")
        return value[key]
    if key is not None:
        raise ValueError(f"--key was supplied, but {source} is not a keyed container")
    return value


def _load_delimited(path: Path) -> np.ndarray:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        return np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    except ValueError:
        return np.genfromtxt(path, delimiter=delimiter, dtype=np.float64, skip_header=1)


def load_array(path_text: str, key: str | None = None) -> LoadedArray:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            value = _select_key(archive, key, path)
            value = np.asarray(value)
            key = key or archive.files[0]
    elif suffix in {".csv", ".tsv"}:
        value = _load_delimited(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = _select_key(json.load(handle), key, path)
    elif suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load .pt/.pth files") from exc
        value = torch.load(path, map_location="cpu", weights_only=False)
        value = _select_key(value, key, path)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
    else:
        raise ValueError(
            f"unsupported extension {suffix!r}; use .npy, .npz, .csv, .tsv, .json, .pt, or .pth"
        )

    return LoadedArray(np.asarray(value), str(path), key)


def _python_number(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_python_number(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _python_number(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_number(item) for item in value]
    return value


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def _sample_pairs(
    n: int, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, min(count, n * max(1, n - 1)))
    left = rng.integers(0, n, size=count)
    right = rng.integers(0, n, size=count)
    equal = left == right
    while np.any(equal):
        right[equal] = rng.integers(0, n, size=int(equal.sum()))
        equal = left == right
    return left, right


def _cosine_pairs(z: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1)
    denom = norms[left] * norms[right]
    dots = np.einsum("ij,ij->i", z[left], z[right])
    result = np.zeros_like(dots, dtype=np.float64)
    valid = denom > np.finfo(np.float64).eps
    result[valid] = dots[valid] / denom[valid]
    both_zero = (~valid) & (norms[left] == 0) & (norms[right] == 0)
    result[both_zero] = 1.0
    return np.clip(result, -1.0, 1.0)


def _spectrum(
    z: np.ndarray, max_rows: int, rng: np.random.Generator
) -> tuple[dict[str, Any], int]:
    n = len(z)
    if n > max_rows:
        sample = z[rng.choice(n, size=max_rows, replace=False)]
    else:
        sample = z
    centered = sample - sample.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular * singular / max(1, len(sample) - 1)
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        ratios = np.zeros_like(eigenvalues)
        effective_rank = 0.0
        stable_rank = 0.0
        participation_ratio = 0.0
    else:
        ratios = eigenvalues / total
        positive = ratios[ratios > 0]
        effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
        stable_rank = float(total / max(float(eigenvalues[0]), np.finfo(np.float64).eps))
        participation_ratio = float(1.0 / np.square(ratios).sum())

    available_rank = max(1, min(len(sample) - 1, z.shape[1]))
    return {
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / available_rank,
        "stable_rank": stable_rank,
        "participation_ratio": participation_ratio,
        "available_rank": available_rank,
        "top1_variance_ratio": float(ratios[:1].sum()),
        "top5_variance_ratio": float(ratios[:5].sum()),
        "top10_variance_ratio": float(ratios[:10].sum()),
        "leading_variance_ratios": ratios[: min(20, len(ratios))],
    }, len(sample)


def analyze(
    raw: np.ndarray,
    labels: np.ndarray | None,
    pair_count: int,
    spectrum_rows: int,
    seed: int,
) -> dict[str, Any]:
    if raw.ndim != 2:
        raise ValueError(f"embeddings must be 2D [samples, dimensions], got shape {raw.shape}")
    if raw.shape[0] < 2 or raw.shape[1] < 1:
        raise ValueError(f"need at least 2 samples and 1 dimension, got shape {raw.shape}")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"embeddings must be numeric, got dtype {raw.dtype}")

    z = raw.astype(np.float64, copy=False)
    finite_rows = np.isfinite(z).all(axis=1)
    removed = int((~finite_rows).sum())
    z = z[finite_rows]
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(raw):
            raise ValueError(f"labels length {len(labels)} does not match sample count {len(raw)}")
        labels = labels[finite_rows]
    if len(z) < 2:
        raise ValueError("fewer than 2 finite embedding rows remain")

    rng = np.random.default_rng(seed)
    n, d = z.shape
    centered = z - z.mean(axis=0, keepdims=True)
    dim_std = z.std(axis=0, ddof=1)
    max_std = float(dim_std.max(initial=0.0))
    dead_threshold = max(1e-12, max_std * 1e-4)
    dead_ratio = float(np.mean(dim_std <= dead_threshold))
    raw_energy = float(np.mean(np.square(z)))
    centered_energy = float(np.mean(np.square(centered)))
    centered_energy_ratio = centered_energy / max(raw_energy, np.finfo(np.float64).eps)
    norms = np.linalg.norm(z, axis=1)

    left, right = _sample_pairs(n, pair_count, rng)
    pair_cosine = _cosine_pairs(z, left, right)
    spectrum, spectrum_sample_count = _spectrum(z, spectrum_rows, rng)

    findings: list[dict[str, str]] = []
    if centered_energy <= max(1e-14, raw_energy * 1e-10):
        findings.append({
            "type": "complete_collapse",
            "severity": "high",
            "evidence": "Centered embedding energy is negligible relative to raw energy.",
        })
    if spectrum["effective_rank_ratio"] < 0.05 and spectrum["available_rank"] >= 8:
        findings.append({
            "type": "dimensional_concentration",
            "severity": "high" if spectrum["top5_variance_ratio"] > 0.95 else "medium",
            "evidence": "The singular-value spectrum uses less than 5% of the available rank.",
        })
    elif spectrum["effective_rank_ratio"] < 0.15 and spectrum["available_rank"] >= 8:
        findings.append({
            "type": "dimensional_concentration",
            "severity": "medium",
            "evidence": "The singular-value spectrum uses less than 15% of the available rank.",
        })
    cosine_q = _quantiles(pair_cosine)
    if cosine_q["p05"] > 0.95:
        findings.append({
            "type": "directional_concentration",
            "severity": "high" if cosine_q["median"] > 0.99 else "medium",
            "evidence": "Even the 5th percentile random-pair cosine similarity exceeds 0.95.",
        })
    if dead_ratio > 0.5:
        findings.append({
            "type": "dead_dimensions",
            "severity": "high" if dead_ratio > 0.9 else "medium",
            "evidence": "More than half of embedding dimensions have negligible variance relative to the most active dimension.",
        })

    severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    overall = "none"
    for finding in findings:
        if severity_order[finding["severity"]] > severity_order[overall]:
            overall = finding["severity"]

    label_metrics = None
    if labels is not None:
        same = labels[left] == labels[right]
        label_metrics = {
            "unique_labels": int(len(np.unique(labels))),
            "sampled_same_label_pairs": int(same.sum()),
            "sampled_different_label_pairs": int((~same).sum()),
        }
        if same.any():
            label_metrics["within_label_cosine_mean"] = float(pair_cosine[same].mean())
        if (~same).any():
            label_metrics["between_label_cosine_mean"] = float(pair_cosine[~same].mean())
        if same.any() and (~same).any():
            label_metrics["within_between_cosine_gap"] = float(
                pair_cosine[same].mean() - pair_cosine[~same].mean()
            )

    return _python_number({
        "shape": {
            "input_samples": int(raw.shape[0]),
            "finite_samples": n,
            "removed_nonfinite_samples": removed,
            "dimensions": d,
            "spectrum_sample_count": spectrum_sample_count,
            "sampled_pair_count": len(pair_cosine),
        },
        "metrics": {
            "raw_energy_mean_square": raw_energy,
            "centered_energy_mean_square": centered_energy,
            "centered_to_raw_energy_ratio": centered_energy_ratio,
            "dimension_std": {
                "mean": float(dim_std.mean()),
                "median": float(np.median(dim_std)),
                "min": float(dim_std.min()),
                "max": max_std,
                "dead_threshold": dead_threshold,
                "dead_dimension_ratio": dead_ratio,
            },
            "vector_norm": {
                "mean": float(norms.mean()),
                "std": float(norms.std(ddof=1)),
                "min": float(norms.min()),
                "max": float(norms.max()),
                "zero_vector_ratio": float(np.mean(norms <= np.finfo(np.float64).eps)),
            },
            "random_pair_cosine": {
                "mean": float(pair_cosine.mean()),
                "std": float(pair_cosine.std(ddof=1)) if len(pair_cosine) > 1 else 0.0,
                **cosine_q,
            },
            "spectrum": spectrum,
            "labels": label_metrics,
        },
        "diagnosis": {
            "overall_risk": overall,
            "findings": findings,
            "caveats": [
                "Heuristic flags are not proof; compare against healthy baselines and downstream performance.",
                "High cosine similarity can reflect a common mean or anisotropy rather than complete collapse.",
                "Low effective rank can be natural when data lies on a low-dimensional manifold.",
            ],
        },
    })


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    shape = report["shape"]
    metrics = report["metrics"]
    diag = report["diagnosis"]
    std = metrics["dimension_std"]
    cos = metrics["random_pair_cosine"]
    spec = metrics["spectrum"]
    lines = [
        "# Representation Collapse Diagnostic",
        "",
        f"Overall heuristic risk: **{diag['overall_risk'].upper()}**",
        "",
        "## Data",
        "",
        f"- Shape: `{shape['finite_samples']} × {shape['dimensions']}`",
        f"- Removed non-finite rows: `{shape['removed_nonfinite_samples']}`",
        f"- Random pairs sampled: `{shape['sampled_pair_count']}`",
        f"- Rows used for spectrum: `{shape['spectrum_sample_count']}`",
        "",
        "## Core metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Centered/raw energy ratio | {_fmt(metrics['centered_to_raw_energy_ratio'])} |",
        f"| Dead-dimension ratio | {_fmt(std['dead_dimension_ratio'])} |",
        f"| Random-pair cosine mean | {_fmt(cos['mean'])} |",
        f"| Random-pair cosine p05 / median / p95 | {_fmt(cos['p05'])} / {_fmt(cos['median'])} / {_fmt(cos['p95'])} |",
        f"| Effective rank / available rank | {_fmt(spec['effective_rank'])} / {spec['available_rank']} |",
        f"| Effective-rank ratio | {_fmt(spec['effective_rank_ratio'])} |",
        f"| Top-1 variance ratio | {_fmt(spec['top1_variance_ratio'])} |",
        f"| Top-5 variance ratio | {_fmt(spec['top5_variance_ratio'])} |",
        "",
        "## Findings",
        "",
    ]
    if diag["findings"]:
        for item in diag["findings"]:
            lines.append(f"- **{item['type']} ({item['severity']})**: {item['evidence']}")
    else:
        lines.append("- No collapse pattern crossed the built-in heuristic thresholds.")

    label_metrics = metrics.get("labels")
    if label_metrics:
        lines.extend(["", "## Label-aware metrics", ""])
        for key, value in label_metrics.items():
            lines.append(f"- {key.replace('_', ' ').title()}: `{_fmt(value)}`")

    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in diag["caveats"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose complete, dimensional, and directional embedding collapse."
    )
    parser.add_argument("embeddings", help="Embedding matrix file")
    parser.add_argument("--key", help="Array/tensor key for keyed embedding containers")
    parser.add_argument("--labels", help="Optional one-dimensional label file")
    parser.add_argument("--labels-key", help="Array/tensor key for keyed label containers")
    parser.add_argument("--output-dir", default="collapse-report", help="Report directory")
    parser.add_argument("--pairs", type=int, default=20000, help="Random pairs to sample")
    parser.add_argument(
        "--spectrum-rows", type=int, default=10000, help="Maximum rows used for SVD"
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pairs < 1 or args.spectrum_rows < 2:
        raise ValueError("--pairs must be >= 1 and --spectrum-rows must be >= 2")

    loaded = load_array(args.embeddings, args.key)
    label_array = None
    label_source = None
    if args.labels:
        loaded_labels = load_array(args.labels, args.labels_key)
        label_array = loaded_labels.array
        label_source = loaded_labels.source

    report = analyze(
        loaded.array,
        label_array,
        pair_count=args.pairs,
        spectrum_rows=args.spectrum_rows,
        seed=args.seed,
    )
    report["input"] = {
        "embeddings": loaded.source,
        "embeddings_key": loaded.key,
        "labels": label_source,
        "labels_key": args.labels_key,
    }
    report["configuration"] = {
        "seed": args.seed,
        "requested_pairs": args.pairs,
        "maximum_spectrum_rows": args.spectrum_rows,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "collapse_report.json"
    markdown_path = output_dir / "collapse_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    summary = {
        "overall_risk": report["diagnosis"]["overall_risk"],
        "finding_types": [item["type"] for item in report["diagnosis"]["findings"]],
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
