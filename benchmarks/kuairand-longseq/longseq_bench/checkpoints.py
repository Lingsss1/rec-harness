"""Preserve uncheckpointed attempts and reconcile epoch-boundary logs."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .download import atomic_json
from .registry import MODEL_NAMES


def prepare_run(output, model):
    directory = Path(output) / model
    if not directory.exists() or (directory / "last.pt").exists():
        return None
    if (directory / "final.json").exists():
        raise ValueError("Completed result is missing its last.pt: " + str(directory))
    manifest = directory / "run.json"
    if manifest.exists():
        run = json.loads(manifest.read_text())
        if run.get("model", {}).get("name") != model or run.get("run_kind") != "kuairand27k_benchmark":
            raise ValueError("Refusing to restart an unrelated output directory: " + str(directory))
    elif any(directory.iterdir()):
        raise ValueError("Unrecognized nonempty output directory: " + str(directory))
    # A failed first epoch has no usable checkpoint. Preserve the attempt and
    # start from seed 42 instead of failing because the output directory exists.
    parent = Path(output) / "interrupted"
    parent.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=model + "-", dir=parent)) / "run"
    directory.rename(backup)
    return backup


def recover_metrics(output, fingerprint, next_epoch, last_metrics):
    output = Path(output)
    metrics = output / "metrics.jsonl"
    records = {}
    lines = metrics.read_text().splitlines() if metrics.exists() else []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # An interrupted trailing write is not a complete epoch.
            raise
        if record["fingerprint"] != fingerprint:
            raise ValueError("Metrics belong to a different run")
        if record["epoch"] < next_epoch:
            records[record["epoch"]] = record
    if next_epoch:
        if (last_metrics is None or last_metrics["epoch"] != next_epoch - 1
                or last_metrics["fingerprint"] != fingerprint):
            raise ValueError("Checkpoint is missing its matching last-epoch metrics")
        records[last_metrics["epoch"]] = last_metrics
        atomic_json(output / "latest.json", last_metrics)
    temporary = metrics.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(records[epoch]) + "\n" for epoch in sorted(records)))
    os.replace(temporary, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("model", choices=MODEL_NAMES)
    args = parser.parse_args()
    backup = prepare_run(args.output, args.model)
    if backup:
        print("Preserved attempt without an epoch checkpoint at " + str(backup), flush=True)


if __name__ == "__main__":
    main()
