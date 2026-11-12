"""Public synthetic preference regressor: intentionally lacks feature interactions."""
import random


def features(row):
    return [1.0, row[0], row[1]]


def train(samples, seed):
    rng = random.Random(seed)
    weights = [rng.uniform(-0.01, 0.01) for _ in features(samples[0][0])]
    for _ in range(120):
        gradients = [0.0] * len(weights)
        for row, target in samples:
            values = features(row)
            error = sum(w * x for w, x in zip(weights, values)) - target
            for i, value in enumerate(values):
                gradients[i] += 2 * error * value / len(samples)
        weights = [w - 0.15 * g for w, g in zip(weights, gradients)]
    return weights


def predict(weights, row):
    return sum(w * x for w, x in zip(weights, features(row)))
