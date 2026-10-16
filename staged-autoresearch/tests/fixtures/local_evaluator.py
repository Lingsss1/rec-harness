"""SYNTHETIC process fixture. No ML/GPU/data libraries and no real metrics."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
for key in ("candidate", "data", "result"):
    parser.add_argument("--" + key, required=True, type=Path)
parser.add_argument("--checkpoint", type=Path)
args = parser.parse_args()
assert "ANTHROPIC_API_KEY" not in os.environ
assert "PYTHONPATH" not in os.environ
source = args.candidate / "domainbed/algorithms.py"
trial = int(source.read_text())
if args.checkpoint:
    assert (args.checkpoint / "weights.fixture").read_text() == "SYNTHETIC"
else:
    (args.result / "checkpoint").mkdir()
    (args.result / "checkpoint/weights.fixture").write_text("SYNTHETIC")
(args.result / "metrics.json").write_text(json.dumps({"metric": 0.2 + trial / 1000,
    "constraints_ok": True, "training_seconds": 0,
    "constraint_checks": {"target_epsilon": 8, "target_delta": 1e-5,
                          "actual_epsilon": 7.99, "actual_delta": 1e-5}}))
