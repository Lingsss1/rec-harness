#!/usr/bin/env python3
"""Audit FuXi-Linear's processed CSV without importing training dependencies."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
PAPER_STATS = {"users": 27284, "observed_items": 131090, "interactions": 97010279}
REQUIRED = {"user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_values(value, kind):
    return [kind(x.strip()) for x in value.split(",")] if value.strip() else []


def audit(path, max_history=1024):
    path = Path(path).resolve()
    users, items, lengths, errors = set(), set(), [], []
    repeated_targets, supervised_pairs, rows = 0, 0, 0
    def error(message):
        # Count all invalid rows without emitting a potentially enormous report.
        if len(errors) < 50:
            errors.append(message)
    invalid_rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, 2):
            rows += 1
            try:
                uid = int(row["user_id"])
                ids = parse_values(row["sequence_item_ids"], int)
                ts = parse_values(row["sequence_timestamps"], int)
                ratings = parse_values(row["sequence_ratings"], float)
                problems = []
                if uid in users:
                    problems.append("duplicate user_id")
                users.add(uid)
                if len(ids) < 3:
                    problems.append("fewer than 3 interactions")
                if len(ids) != len(ts) or len(ids) != len(ratings):
                    problems.append("sequence column lengths differ")
                if any(x < 0 for x in ids):
                    problems.append("negative raw item_id")
                if any(not math.isfinite(x) for x in ratings):
                    problems.append("nonfinite rating")
                if any(a > b for a, b in zip(ts, ts[1:])):
                    problems.append("timestamps are not chronological")
                if problems:
                    invalid_rows += 1
                    error("line %d: %s" % (line, "; ".join(problems)))
                    continue
                items.update(ids)
                lengths.append(len(ids))
                # Same holdout/window convention as DatasetV2.
                eval_history = ids[:-1][-max_history:]
                repeated_targets += int(ids[-1] in eval_history)
                train_history = ids[:-2][-max_history:]
                supervised_pairs += len(train_history)
            except (ValueError, TypeError, KeyError) as exc:
                invalid_rows += 1
                error("line %d: %s" % (line, exc))
    if not rows:
        error("empty dataset")
    contiguous = bool(items) and min(items) == 0 and max(items) + 1 == len(items)
    if not contiguous:
        error("item IDs must be contiguous and zero-based before reference +1 shift")
    stats = {"users": len(users), "observed_items": len(items), "interactions": sum(lengths)}
    differences = {k: {"observed": stats[k], "paper": v}
                   for k, v in PAPER_STATS.items() if stats[k] != v}
    return {
        "path": str(path), "sha256": file_hash(path),
        "statistics": stats,
        "rows": rows, "valid_rows": len(lengths), "invalid_rows": invalid_rows,
        "average_history_length": sum(lengths) / len(lengths) if lengths else None,
        "max_history_length_in_data": max(lengths) if lengths else None,
        "model_history_cap": max_history,
        "raw_item_ids_contiguous_zero_based": contiguous,
        "min_raw_item_id": min(items) if items else None,
        "max_raw_item_id": max(items) if items else None,
        "train_supervised_positions_per_epoch": supervised_pairs,
        "holdout_target_repeated_in_model_history": repeated_targets,
        "holdout_target_repeated_fraction": repeated_targets / len(lengths) if lengths else None,
        "paper_statistics_match": not differences,
        "paper_statistics_differences": differences,
        "preprocessing_identity_verified": False,
        "note": "Matching counts do not establish preprocessing identity; record source and raw hashes separately.",
        "errors": errors,
        "valid": not errors and invalid_rows == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.csv)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, "Data audit failed: %s\n" % exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("statistics", "valid", "paper_statistics_match", "errors")}))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
