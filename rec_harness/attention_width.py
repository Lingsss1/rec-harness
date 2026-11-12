"""Frozen, query-conditioned attention width measurements.

Spectra are diagnostics; held-out nonlinear merging measures compressibility.
No function in this module trains or changes a model.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def responses(queries, keys, bias, activation="silu", scale=1.0):
    logits = queries @ keys.T + bias
    if activation == "silu":
        return F.silu(logits) / scale
    if activation == "softmax":
        return logits.softmax(-1)
    raise ValueError("unknown activation: " + activation)


def spectrum(matrix, centered=False):
    """Singular-value entropy rank and squared-energy rank, explicitly distinct."""
    x = matrix.double()
    if centered:
        x = x - x.mean(0, keepdim=True)
    gram = x @ x.T if x.shape[0] <= x.shape[1] else x.T @ x
    energy = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = float(energy.sum())
    if total <= 1e-24:
        return dict(effective_rank=0.0, energy_rank_99=0, stable_rank=0.0,
                    energy=total, sample_ceiling=min(x.shape), ceiling_reached=False)
    # Roundoff in a Gram matrix creates tiny fictitious singular values.
    energy = energy.masked_fill(energy < energy[0] * 1e-12, 0)
    singular = energy.sqrt()
    p = singular / singular.sum()
    positive = p > 0
    rank99 = int(torch.searchsorted(energy.cumsum(0), .99 * energy.sum())) + 1
    return dict(effective_rank=float(torch.exp(-(p[positive] * p[positive].log()).sum())),
                energy_rank_99=rank99, stable_rank=float(energy.sum() / energy[0]),
                energy=total, sample_ceiling=min(x.shape),
                ceiling_reached=rank99 >= .8 * min(x.shape))


def nmse(actual, reference):
    return float((actual.double() - reference.double()).square().sum() /
                 reference.double().square().sum().clamp_min(1e-24))


def balanced_groups(features, budgets):
    """Nested median splits along the largest-variance feature; no test queries.

    All group sizes are equal. This permits exactly size-matched random controls
    and makes group cardinality an explicit part of the intervention.
    """
    n = len(features)
    if n < 1 or n & (n - 1):
        raise ValueError("balanced diagnostic requires a power-of-two memory size")
    requested = set(budgets)
    if any(m <= 0 or m > n or m & (m - 1) for m in requested):
        raise ValueError("budgets must be powers of two no larger than memory")
    groups = torch.arange(n, device=features.device)[None]
    result = {}
    while True:
        count, width = groups.shape
        if count in requested:
            result[count] = groups.clone()
        if count >= max(requested):
            break
        values = features[groups]
        axis = values.var(1, unbiased=False).argmax(-1)
        coordinates = values.gather(2, axis[:, None, None].expand(-1, width, 1)).squeeze(-1)
        order = coordinates.argsort(dim=1, stable=True)
        groups = groups.gather(1, order).reshape(count * 2, width // 2)
    return result


def response_features(calibration_responses, seed, dimensions=32):
    """Approximate response cosines using a fixed random projection."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(calibration_responses.shape[0], dimensions,
                             generator=generator, dtype=calibration_responses.dtype)
    features = calibration_responses.T @ projection.to(calibration_responses.device)
    return F.normalize(features, dim=-1, eps=1e-12)


def merged_parameters(keys, values, bias, groups, activation="silu"):
    """Exact for equal within-group logits, approximate otherwise.

    SiLU: average input weights, sum output weights, original scale unchanged.
    Softmax: average values and add log group size to preserve attention mass.
    """
    key = keys[groups].mean(1)
    offset = bias[groups].mean(1)
    if activation == "silu":
        value = values[groups].sum(1)
    elif activation == "softmax":
        value = values[groups].mean(1)
        offset = offset + math.log(groups.shape[1])
    else:
        raise ValueError("unknown activation: " + activation)
    return key, value, offset


def merged_output(queries, keys, values, bias, groups, activation="silu", scale=1.0):
    k, v, b = merged_parameters(keys, values, bias, groups, activation)
    return responses(queries, k, b, activation, scale) @ v


def paired_interval(values, seed=42, repeats=2000):
    """User-level paired bootstrap; input contains one difference per user."""
    x = torch.as_tensor(values, dtype=torch.float64)
    if not len(x):
        raise ValueError("empty panel")
    generator = torch.Generator().manual_seed(seed)
    # Keep memory bounded even for a full-catalog, full-user confirmation.
    means = []
    for start in range(0, repeats, 100):
        indices = torch.randint(len(x), (min(100, repeats - start), len(x)), generator=generator)
        means.append(x[indices].mean(1))
    samples = torch.cat(means)
    return dict(mean=float(x.mean()), low=float(samples.quantile(.025)),
                high=float(samples.quantile(.975)), users=len(x), bootstrap_repeats=repeats)
