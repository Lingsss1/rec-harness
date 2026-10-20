"""Trusted Movies evaluator using the frozen common NextItNet trainer.

The final scalar is independently recomputed outside the candidate process.
Prediction children receive shuffled candidate features, never evaluation labels.
All phase arrays and scoring seeds are identical to the previous comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

# This module is run by absolute path from a clean evaluator environment.
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from mle_star_codex.run import source_parts
from mle_star_codex.worker import read as read_split, rows_for, metrics, sha


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def predict_child(args):
    import numpy as np
    import torch
    from mle_star_codex.worker import load, predict, check
    torch.set_num_threads(4)
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    request = json.loads(Path(args.features).read_text())
    module, model = load(Path(args.candidate) / "model.py", request["itemnum"], device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "checkpoint.pt", map_location=device, weights_only=True))
    check(model, module.MAXLEN, device)
    # The histories and randomly reordered candidates contain no positive index.
    scores = predict(model, request["rows"], module.MAXLEN, device)
    np.save(args.scores, scores)


def evaluate(args):
    import numpy as np
    candidate, data, output = Path(args.candidate), Path(args.data), Path(args.result)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise RuntimeError("refusing to overwrite evaluator result")
    source_parts((candidate / "model.py").read_text())
    scratch = output / "adapter_scratch"
    scratch.mkdir(exist_ok=True)
    started = time.monotonic()
    common = WORKSPACE / "mle_star_codex/worker.py"
    if args.phase == "validation":
        training = scratch / "training"
        argv = [sys.executable, str(common), "--source", str(candidate / "model.py"),
            "--data", str(data), "--output", str(training), "--seed", "2024",
            "--epochs", str(args.epochs), "--batches", str(args.batches), "--eval-users", str(args.eval_users)]
        with (scratch / "training.log").open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
        trained = json.loads((training / "result.json").read_text())
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        shutil.copyfile(training / "checkpoint.pt", checkpoint / "checkpoint.pt")
        save(checkpoint / "identity.json", {"model_sha256": sha(candidate / "model.py"),
            "checkpoint_sha256": sha(checkpoint / "checkpoint.pt"), "seed": 2024,
            "common_worker_sha256": sha(common)})
    else:
        checkpoint = Path(args.checkpoint)
        identity = json.loads((checkpoint / "identity.json").read_text())
        if identity["model_sha256"] != sha(candidate / "model.py") or identity["checkpoint_sha256"] != sha(checkpoint / "checkpoint.pt"):
            raise RuntimeError("test source/checkpoint identity changed")
        trained = None
    train = read_split(data / "trainval/Movies_and_TV_train.txt")
    valid = read_split(data / "trainval/Movies_and_TV_valid.txt")
    itemnum = json.loads((data / "trainval/dataset_stats.json").read_text())["Movies_and_TV"]["itemnum"]
    if args.phase == "validation":
        rows = rows_for(train, valid, itemnum, args.eval_users)
    else:
        test = read_split(data / "test/Movies_and_TV_test.txt")
        rows = rows_for(train, test, itemnum, 10000, valid)
    row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    rng = random.Random(2024)
    visible, private = [], []
    for row in rows:
        order = list(range(100))
        rng.shuffle(order)
        visible.append({"history": row["history"], "candidates": [row["candidates"][i] for i in order]})
        private.append(order.index(0))
    features, scores_path = scratch / "features.json", scratch / "predictions.npy"
    save(features, {"itemnum": itemnum, "rows": visible})
    with (scratch / "prediction.log").open("w") as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--predict-child",
            "--candidate", str(candidate), "--checkpoint", str(checkpoint),
            "--features", str(features), "--scores", str(scores_path)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    scores = np.load(scores_path, allow_pickle=False)
    if scores.shape != (len(rows), 100) or not np.isfinite(scores).all():
        raise RuntimeError("invalid independent score matrix")
    ordered = np.empty_like(scores)
    for i, target in enumerate(private):
        order = [target] + [j for j in range(100) if j != target]
        ordered[i] = scores[i, order]
    measured = metrics(ordered)
    if trained is not None and abs(trained["metrics"]["HR@10"] - measured["HR@10"]) > 1e-12:
        raise RuntimeError("independent validation score disagrees with frozen training worker")
    payload = {"metric": measured["HR@10"], "constraints_ok": True,
        "metrics": measured, "training_seconds": time.monotonic() - started if trained is not None else 0,
        "rows_sha256": row_hash, "model_sha256": sha(candidate / "model.py"),
        "common_worker_sha256": sha(common), "n_users": len(rows), "phase": args.phase,
        "scoring": "trusted external parent; prediction child receives shuffled unlabeled candidates"}
    save(output / "metrics.json", payload)
    print(json.dumps(payload), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--data")
    p.add_argument("--result")
    p.add_argument("--checkpoint")
    p.add_argument("--phase", choices=["validation", "test"], default="validation")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--eval-users", type=int, default=200)
    p.add_argument("--predict-child", action="store_true")
    p.add_argument("--features")
    p.add_argument("--scores")
    args = p.parse_args()
    if args.predict_child:
        predict_child(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
