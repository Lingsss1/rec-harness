"""Frozen public toy evaluator; fresh parameters per paired seed, no external data."""
import argparse
import importlib.util
import json
import math
import random
from pathlib import Path


def load_model(path):
    spec = importlib.util.spec_from_file_location("candidate", Path(path) / "model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def samples(seed, n):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.uniform(-1, 1), rng.uniform(-1, 1)
        rows.append(((a, b), 0.25 * a + 0.15 * b + 1.5 * a * b))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    model = load_model(args.model)
    data = samples(args.seed, 180)
    weights = model.train(data, args.seed)
    # Repeated fresh initialization must be deterministic for a fixed seed.
    if args.check:
        duplicate = model.train(data, args.seed)
        assert weights == duplicate, "nondeterministic initialization/training"
        for row in [(0.0, 0.0), (1.0, -1.0), (-1.0, 1.0)]:
            assert math.isfinite(model.predict(weights, row))
        return
    validation = samples(args.seed + 10000, 120)
    mse = sum((model.predict(weights, row) - target) ** 2 for row, target in validation) / len(validation)
    metrics = {"quality": 1 / (1 + mse), "mse": mse, "parameters": len(weights)}
    Path(args.output).write_text(json.dumps(metrics))


if __name__ == "__main__":
    main()
