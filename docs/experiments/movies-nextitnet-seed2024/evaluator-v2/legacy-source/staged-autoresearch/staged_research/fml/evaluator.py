"""Trusted FML scorer. Candidate Python runs only in disposable child processes."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..storage import ProtocolError, digest, inventory, read, write
from ..study import TASKS
from .data import verify_data
from .specs import DATA_PROTOCOL


def run_worker(argv, directory, label, isolation="process"):
    env = dict(os.environ)
    for name in list(env):
        if name.endswith("API_KEY") or name in ("PYTHONPATH", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(name)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    binary = [sys.executable, "-m", "staged_research.fml.worker"]
    if isolation == "bubblewrap":
        from ..local import bubblewrap_command
        bindings = []
        argv = list(argv)
        for name, target in (("source", "/source"), ("data", "/input"), ("checkpoint", "/checkpoint"), ("inputs", "/features.npy")):
            flag = "--" + name
            if flag in argv:
                index = argv.index(flag) + 1
                bindings.append((Path(argv[index]), target, True))
                argv[index] = target
        index = argv.index("--output") + 1
        output = Path(argv[index])
        bindings.append((output.parent, "/output", False))
        argv[index] = "/output/" + output.name
        env.update(HOME="/tmp", TMPDIR="/tmp")
        binary = bubblewrap_command(binary + argv, bindings, cwd="/tmp", network=False,
            roots=[sys.prefix], files=[], gpu=True)
        argv = []
    with (directory / (label + ".stdout.log")).open("w") as stdout, (directory / (label + ".stderr.log")).open("w") as stderr:
        result = subprocess.run(binary + argv,
            cwd=str(directory), env=env, stdout=stdout, stderr=stderr)
    if result.returncode:
        # Error logs are retained as files; do not swallow worker failure.
        tail = (directory / (label + ".stderr.log")).read_text(errors="replace")[-2000:]
        raise ProtocolError("Candidate worker failed (%s, exit=%d); inspect %s\n%s" % (label, result.returncode, directory, tail))


def accuracy(predictions, labels):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if len(predictions) != len(labels) or not len(labels) or not np.isfinite(predictions).all():
        raise ProtocolError("Invalid classifier predictions")
    if predictions.ndim != 2 or predictions.shape[1] not in (1, 2, 10):
        raise ProtocolError("Expected classifier logits")
    classes = (predictions[:, 0] > 0).astype(int) if predictions.shape[1] == 1 else predictions.argmax(axis=1)
    return float(np.mean(classes == labels))


def ood_metrics(id_scores, other_scores):
    from sklearn.metrics import roc_auc_score
    id_scores, other_scores = np.asarray(id_scores), np.asarray(other_scores)
    if id_scores.ndim != 1 or other_scores.ndim != 1 or not len(id_scores) or not len(other_scores):
        raise ProtocolError("OOD scores must be nonempty 1D arrays")
    scores = np.concatenate([id_scores, other_scores])
    if not np.isfinite(scores).all():
        raise ProtocolError("Nonfinite OOD scores")
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(other_scores))])
    ordered = labels[np.argsort(-scores)]
    target = int(0.95 * len(id_scores))
    reached = np.flatnonzero(np.cumsum(ordered) >= target)
    stop = reached[0] + 1 if len(reached) else len(ordered)
    fpr = float(np.sum(1 - ordered[:stop]) / len(other_scores))
    return {"auroc": float(roc_auc_score(labels, scores)), "fpr95": fpr}


def privacy_checks(spec):
    """Recompute epsilon in the scorer; ignore worker-supplied scalar epsilon."""
    from opacus.accountants import create_accountant
    config, state = spec["training_config"], spec["privacy"]
    if config.get("target_epsilon") != 8 or config.get("target_delta") != 1e-5 or state.get("mechanism") != "prv":
        raise ProtocolError("Privacy target/accountant changed")
    expected_steps = spec["logical_steps"]
    sample_rate, sigma = state["sample_rate"], state["noise_multiplier"]
    if not isinstance(expected_steps, int) or expected_steps <= 0 or not math.isfinite(sigma) or sigma <= 0 or not 0 < sample_rate <= 1:
        raise ProtocolError("Invalid privacy trace parameters")
    batches = spec["population"] // config["batch_size"]
    if not batches or sample_rate != 1 / batches or expected_steps != config["epochs"] * int(1 / sample_rate):
        raise ProtocolError("Accountant sampling schedule differs from the frozen training loop")
    history = []
    for item in state["history"]:
        noise, rate, steps = item
        if noise != sigma or rate != sample_rate or isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ProtocolError("Inconsistent actual accountant trace")
        history.append((noise, rate, steps))
    if sum(item[2] for item in history) != expected_steps:
        raise ProtocolError("Accounted optimizer steps differ from completed training")
    accountant = create_accountant(mechanism="prv")
    accountant.history = history
    epsilon = float(accountant.get_epsilon(delta=1e-5))
    if not math.isfinite(epsilon) or not 0 < epsilon <= 8:
        raise ProtocolError("Actual privacy budget exceeds epsilon=8")
    return {"target_epsilon": 8, "target_delta": 1e-5, "actual_epsilon": epsilon, "actual_delta": 1e-5}


def source_identity(candidate):
    return digest(inventory(candidate))


def evaluate(task, phase, candidate, data, result, checkpoint=None, device="cuda", smoke=False, worker_isolation="process"):
    candidate, data, result = (Path(p).resolve() for p in (candidate, data, result))
    if phase not in ("validation", "test") or task not in TASKS:
        raise ProtocolError("Invalid FML task/phase")
    metadata = verify_data(data, task, phase, allow_smoke=smoke)
    if smoke != (metadata["profile"] == "smoke"):
        raise ProtocolError("Smoke flag and dataset profile must agree")
    result.mkdir(parents=True, exist_ok=True)
    if (result / "metrics.json").exists() or (result / "checkpoint").exists():
        raise ProtocolError("Refuse to overwrite previous evaluator results")
    scratch = result / "adapter_scratch"
    scratch.mkdir(exist_ok=False)
    code = scratch / "source"
    shutil.copytree(candidate, code)
    # Do not import any candidate into this scorer, including when loading weights.
    fingerprint = source_identity(candidate)
    common = ["--task", task, "--source", str(code), "--device", device] + (["--smoke"] if smoke else [])
    if phase == "validation":
        training = scratch / "train_input"
        training.mkdir()
        for item in data.iterdir():
            if item.name.startswith("train_") or item.name == "rng_state.npy":
                if item.suffix == ".npy":
                    shutil.copyfile(item, training / item.name)
        trained = scratch / "trained"
        run_worker(common + ["--action", "train", "--data", str(training), "--output", str(trained)], scratch, "train", worker_isolation)
        if source_identity(candidate) != fingerprint or source_identity(code) != fingerprint:
            raise ProtocolError("Candidate/source was changed during training")
        spec = read(trained / "model_spec.json")
        spec.update(source_sha256=fingerprint, protocol=DATA_PROTOCOL, profile=metadata["profile"])
        write(trained / "model_spec.json", spec)
        if not (trained / "model.pt").is_file() or (trained / "model.pt").is_symlink():
            raise ProtocolError("Training did not return a real model checkpoint")
        checkpoint = result / "checkpoint"
        shutil.copytree(trained, checkpoint)
    else:
        if checkpoint is None:
            raise ProtocolError("Hidden test requires the validation-selected checkpoint")
        checkpoint = Path(checkpoint).resolve()
        spec = read(checkpoint / "model_spec.json")
        if spec.get("source_sha256") != fingerprint or spec.get("protocol") != DATA_PROTOCOL or spec.get("profile") != metadata["profile"]:
            raise ProtocolError("Checkpoint does not belong to the selected source/protocol/profile")
    checkpoint_hash = digest(inventory(checkpoint))
    constraints = privacy_checks(spec) if task == "opacus" else None
    which = "val" if phase == "validation" else "test"
    subsets = [which, "ood"] if task == "openood" else [which]
    predictions = []
    # Labels never enter the predictor directory. Each candidate forward pass
    # executes in a fresh interpreter, so monkeypatches cannot alter the scorer.
    inference = scratch / "inference"
    inference.mkdir()
    for index, subset in enumerate(subsets):
        inputs = inference / ("features-%d.npy" % index)
        shutil.copyfile(data / (subset + "_x.npy"), inputs)
        scores = inference / "scores"
        scores.mkdir(exist_ok=True)
        prediction_file = scores / ("predictions-%d.npy" % index)
        run_worker(common + ["--action", "infer", "--checkpoint", str(checkpoint),
            "--inputs", str(inputs), "--output", str(prediction_file)], inference, "infer-%d" % index, worker_isolation)
        predictions.append(np.load(prediction_file, allow_pickle=False))
    if source_identity(candidate) != fingerprint or source_identity(code) != fingerprint or digest(inventory(checkpoint)) != checkpoint_hash:
        raise ProtocolError("Candidate or selected checkpoint changed during scoring")
    if task == "openood":
        details = ood_metrics(*predictions)
        metric = details["auroc"]
    else:
        metric = accuracy(predictions[0], np.load(data / (which + "_y.npy"), allow_pickle=False))
        details = {"accuracy": metric}
    value = {"metric": metric, "constraints_ok": True, "task": task, "phase": phase,
        "profile": metadata["profile"], "data_id": metadata["id"], "source_sha256": fingerprint,
        "checkpoint_sha256": checkpoint_hash, "training_seconds": spec["training_seconds"] if phase == "validation" else 0,
        "details": details}
    if constraints is not None:
        value["constraint_checks"] = constraints
    write(result / "metrics.json", value, exclusive=True)
    # Keep only selected weights/spec and logs; scratch arrays/models would
    # otherwise multiply CIFAR/ColoredMNIST storage by 6,000 evaluations.
    for path in scratch.rglob("*.log"):
        shutil.copyfile(path, result / path.name)
    shutil.rmtree(scratch)
    print(json.dumps({"metric": metric, "phase": phase, "profile": metadata["profile"]}), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    for name in ("candidate", "data", "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-isolation", choices=("process", "bubblewrap"), default="process")
    args = parser.parse_args()
    try:
        evaluate(args.task, args.phase, args.candidate, args.data, args.result, args.checkpoint, args.device, args.smoke, args.worker_isolation)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
