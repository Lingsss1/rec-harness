"""Prepare a cache, or reproduce the released author-code preprocessing profile."""
import argparse
import csv
import json
from pathlib import Path

from .data import make_cache, sha256
from .download import LOGS

PROFILE = "author-code-350-unfiltered-df1"


def preprocess_author_code(raw_dir, output):
    import pandas as pd
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    files = [Path(raw_dir) / name for name in LOGS]
    # Loading only these three columns preserves the row membership and sorting
    # of the released df1 path, reducing RAM. Do NOT introduce the unused df mask.
    frame = pd.concat([pd.read_csv(path, usecols=["user_id", "video_id", "time_ms"]) for path in files])
    before = (frame.user_id.nunique(), frame.video_id.nunique())
    while True:
        counts = frame.video_id.value_counts()
        frame = frame[frame.video_id.isin(counts[counts >= 350].index)]
        counts = frame.user_id.value_counts()
        frame = frame[frame.user_id.isin(counts[counts >= 20].index)]
        after = (frame.user_id.nunique(), frame.video_id.nunique())
        if before == after:
            break
        before = after
    if frame.empty:
        raise ValueError("Filtering removed all rows")
    mapping = {item: i for i, item in enumerate(sorted(frame.video_id.unique()))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"])
        for i, (uid, group) in enumerate(frame.groupby("user_id")):
            group = group.sort_values("time_ms")  # Same pandas default quicksort.
            writer.writerow([i, uid, ",".join(group.video_id.map(mapping).astype(str)),
                             ",".join(["4.0"] * len(group)), ",".join(group.time_ms.astype(str))])
    manifest = {
        "profile": PROFILE, "reference_commit": "5a704061a7ebf0b81afb465f12f93f2747ba4667",
        "warning": "Preserves released df1 behavior; differs from paper-described positive filtering/threshold.",
        "pandas_version": pd.__version__, "raw_files": {p.name: sha256(p) for p in files},
        "output_sha256": sha256(output), "item_frequency_min": 350, "user_frequency_min": 20,
        "positive_feedback_filter_applied": False,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="Optional: create processed CSV using the released-code profile")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=1024)
    args = parser.parse_args()
    provenance = preprocess_author_code(args.raw_dir, args.processed_csv) if args.raw_dir else None
    sidecar = args.processed_csv.with_suffix(".provenance.json")
    if provenance is None and sidecar.exists():
        provenance = json.loads(sidecar.read_text())
        if provenance.get("output_sha256") != sha256(args.processed_csv):
            raise ValueError("Processed CSV no longer matches its provenance sidecar")
    print(json.dumps(make_cache(args.processed_csv, args.cache, args.max_history, provenance), indent=2))


if __name__ == "__main__":
    main()
