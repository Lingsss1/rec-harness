"""Minimal, auditable MISO-style primitives for a PyTorch reproduction.

This is a reference reconstruction, not Meta's unreleased implementation.
The paper specifies Eq. (1), Eq. (2), and a high-level closed loop, but it does
not specify hooks, density estimation, action thresholds, or comparison scores.

Assumptions:
  * A probed module returns one tensor whose final axis is the neuron/channel axis.
  * The ranking task uses a scalar batch loss (for example BCEWithLogitsLoss).
  * Evaluation loaders are deterministic and representative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


Batch = Any
BatchToArgs = Callable[[Batch, torch.device], tuple[tuple[Any, ...], dict[str, Any], Tensor]]
LossFn = Callable[[Tensor, Tensor], Tensor]


def _as_tensor(output: Any) -> Tensor:
    """Extract the single tensor that represents a module's activation."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) == 1 and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("The reference probe expects a Tensor or a one-Tensor tuple/list output.")


def _flatten_neurons(tensor: Tensor) -> Tensor:
    """Flatten every sample/token axis while preserving the final neuron axis."""
    if tensor.ndim < 2:
        raise ValueError("Expected at least [batch, neurons].")
    return tensor.reshape(-1, tensor.shape[-1])


@dataclass
class TensorProbe:
    """Forward probe whose activation also captures its backward gradient."""

    activation: Tensor | None = None
    gradient: Tensor | None = None

    def hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        activation = _as_tensor(output)
        self.activation = activation.detach()
        self.gradient = None
        if activation.requires_grad:
            activation.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: Tensor) -> None:
        self.gradient = gradient.detach()


def gradient_taylor_importance(
    model: nn.Module,
    module: nn.Module,
    loader: Iterable[Batch],
    batch_to_args: BatchToArgs,
    loss_fn: LossFn,
    device: torch.device,
) -> Tensor:
    """Estimate E[|a_n * dL/da_n|], the first term in paper Eq. (1).

    Returns one score per final-axis neuron. Dropout should be disabled and the
    same fixed validation loader should be used for every candidate model.
    """

    model.eval()
    probe = TensorProbe()
    handle = module.register_forward_hook(probe.hook)
    score_sum: Tensor | None = None
    observation_count = 0

    try:
        for batch in loader:
            args, kwargs, target = batch_to_args(batch, device)
            model.zero_grad(set_to_none=True)
            prediction = model(*args, **kwargs)
            loss = loss_fn(_as_tensor(prediction), target)
            loss.backward()

            if probe.activation is None or probe.gradient is None:
                raise RuntimeError("The module did not produce a differentiable probed activation.")
            activation = _flatten_neurons(probe.activation)
            gradient = _flatten_neurons(probe.gradient)
            batch_score = (activation * gradient).abs().sum(dim=0)
            score_sum = batch_score if score_sum is None else score_sum + batch_score
            observation_count += activation.shape[0]
    finally:
        handle.remove()

    if score_sum is None or observation_count == 0:
        raise ValueError("The loader produced no observations.")
    return score_sum / observation_count


@torch.no_grad()
def mean_loader_loss(
    model: nn.Module,
    loader: Iterable[Batch],
    batch_to_args: BatchToArgs,
    loss_fn: LossFn,
    device: torch.device,
) -> float:
    """Compute an example-weighted mean evaluation loss."""

    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        args, kwargs, target = batch_to_args(batch, device)
        prediction = _as_tensor(model(*args, **kwargs))
        loss = loss_fn(prediction, target)
        batch_size = int(target.shape[0])
        total += float(loss.detach()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("The loader produced no examples.")
    return total / count


def _group_shuffle_hook(group: Tensor, generator: torch.Generator | None):
    """Return a hook that shuffles selected channels across flattened observations."""

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Tensor:
        activation = _as_tensor(output)
        flat = _flatten_neurons(activation).clone()
        permutation = torch.randperm(flat.shape[0], device=flat.device, generator=generator)
        flat[:, group] = flat[permutation][:, group]
        return flat.reshape_as(activation)

    return hook


@torch.no_grad()
def permutation_group_importance(
    model: nn.Module,
    module: nn.Module,
    loader_factory: Callable[[], Iterable[Batch]],
    groups: Sequence[Sequence[int]],
    batch_to_args: BatchToArgs,
    loss_fn: LossFn,
    device: torch.device,
    repeats: int = 3,
    seed: int = 0,
) -> Tensor:
    """Estimate loss increase after shuffling each neuron group.

    The paper writes neuron-wise shuffling. Group-wise shuffling is a practical
    approximation because exact neuron-wise evaluation needs one extra forward
    pass per neuron and repeat. ``loader_factory`` must recreate the same data
    order for every pass.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    baseline = mean_loader_loss(model, loader_factory(), batch_to_args, loss_fn, device)
    deltas = torch.zeros(len(groups), dtype=torch.float64)

    for group_index, indices in enumerate(groups):
        group = torch.as_tensor(indices, dtype=torch.long, device=device)
        for repeat in range(repeats):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + 1009 * group_index + repeat)
            handle = module.register_forward_hook(_group_shuffle_hook(group, generator))
            try:
                perturbed = mean_loader_loss(
                    model, loader_factory(), batch_to_args, loss_fn, device
                )
            finally:
                handle.remove()
            deltas[group_index] += perturbed - baseline

    return (deltas / repeats).to(dtype=torch.float32)


def percentile_rank(values: Tensor) -> Tensor:
    """Map scores to [0, 1] ranks so heterogeneous signals can be combined."""

    flat = values.detach().flatten()
    if flat.numel() <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    order = torch.argsort(flat)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(flat.numel(), device=flat.device, dtype=torch.float32)
    return (ranks / (flat.numel() - 1)).reshape_as(values)


def combine_importance(
    gradient_score: Tensor,
    permutation_score: Tensor,
    alpha: float,
    normalize: bool = False,
) -> Tensor:
    """Combine Eq. (1) terms.

    ``normalize=False`` follows the paper literally. ``normalize=True`` first
    percentile-ranks both terms; that is an engineering completion, not stated
    in the paper, and should be reported as a separate ablation.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    left, right = gradient_score, permutation_score
    if normalize:
        left, right = percentile_rank(left), percentile_rank(right)
    return alpha * left + (1.0 - alpha) * right


def gaussian_kl_to_standard_normal(samples: Tensor, eps: float = 1e-8) -> Tensor:
    """KL(N(mu, sigma^2) || N(0,1)); a cheap approximation to empirical KL."""

    flat = samples.detach().float().flatten()
    variance = flat.var(unbiased=False).clamp_min(eps)
    mean = flat.mean()
    return 0.5 * (mean.square() + variance - 1.0 - variance.log())


def empirical_kl_to_standard_normal(
    samples: Tensor,
    bins: int = 201,
    low: float = -8.0,
    high: float = 8.0,
    eps: float = 1e-12,
) -> Tensor:
    """Histogram estimate of D_KL(p_empirical || N(0,1)).

    Fixed bin edges make scores comparable between layers/checkpoints. Values
    outside [low, high] are clipped into the edge bins.
    """

    flat = samples.detach().float().flatten().clamp(low, high)
    counts = torch.histc(flat, bins=bins, min=low, max=high)
    p = (counts + eps) / (counts.sum() + eps * bins)

    edges = torch.linspace(low, high, bins + 1, device=flat.device)
    normal_cdf = 0.5 * (1.0 + torch.erf(edges / (2.0**0.5)))
    q = (normal_cdf[1:] - normal_cdf[:-1]).clamp_min(eps)
    q = q / q.sum()
    return torch.sum(p * (p.log() - q.log()))


def alignment_score(
    pre_normalization: Tensor,
    post_normalization: Tensor,
    estimator: Callable[[Tensor], Tensor] = empirical_kl_to_standard_normal,
) -> Tensor:
    """Paper Eq. (2). It is a ranking score and may be negative."""

    return 1.0 - 0.5 * (
        estimator(pre_normalization) + estimator(post_normalization)
    )


@dataclass(frozen=True)
class LayerSignature:
    mean: float
    std: float
    q01: float
    q50: float
    q99: float
    rms: float
    zero_fraction: float


def layer_signature(activation: Tensor) -> LayerSignature:
    """A compact state summary for an explicit comparison primitive."""

    flat = activation.detach().float().flatten()
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.50, 0.99], device=flat.device))
    return LayerSignature(
        mean=float(flat.mean()),
        std=float(flat.std(unbiased=False)),
        q01=float(quantiles[0]),
        q50=float(quantiles[1]),
        q99=float(quantiles[2]),
        rms=float(flat.square().mean().sqrt()),
        zero_fraction=float((flat == 0).float().mean()),
    )


def signature_distance(
    left: LayerSignature,
    right: LayerSignature,
    eps: float = 1e-8,
) -> float:
    """Concrete comparison score; the paper does not prescribe one.

    Each feature is scaled by the pair's characteristic magnitude before the
    mean absolute difference is taken.
    """

    distances = []
    for field in LayerSignature.__dataclass_fields__:
        a = float(getattr(left, field))
        b = float(getattr(right, field))
        scale = max(abs(a), abs(b), eps)
        distances.append(abs(a - b) / scale)
    return sum(distances) / len(distances)


def normalized_entropy(probability: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Binary normalized entropy: model log loss divided by base-rate entropy."""

    probability = probability.float().flatten().clamp(eps, 1.0 - eps)
    target = target.float().flatten()
    if probability.shape != target.shape:
        raise ValueError("probability and target must have the same flattened shape")

    log_loss = -(
        target * probability.log() + (1.0 - target) * (1.0 - probability).log()
    ).mean()
    base_rate = target.mean().clamp(eps, 1.0 - eps)
    base_entropy = -(
        base_rate * base_rate.log() + (1.0 - base_rate) * (1.0 - base_rate).log()
    )
    return log_loss / base_entropy


def rank_modules(scores: Mapping[str, Tensor]) -> list[tuple[str, float]]:
    """Reduce neuron scores to module means and rank descending."""

    return sorted(
        ((name, float(value.detach().float().mean())) for name, value in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
