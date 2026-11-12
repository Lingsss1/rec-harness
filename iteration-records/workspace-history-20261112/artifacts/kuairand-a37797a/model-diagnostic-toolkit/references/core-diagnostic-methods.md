# Core diagnostic methods

These diagnostics are deliberately independent of the training framework. Export small, fixed-validation artifacts from the GPU process and analyze them offline.

## Freeze the comparison first

Before interpreting any diagnostic, record:

- code/checkpoint identity, dataset version, split, and sampled example IDs;
- primary validation metric and guardrails;
- random seeds, batch order, precision, and evaluation mode;
- model width, candidate width, parameter count, runtime, and peak GPU memory;
- the sign convention for every metric and perturbation delta.

Use validation data for iteration. Keep the held-out test split untouched until the incumbent and protocol are frozen.

## Sample sufficiency

For sparse ranking objectives, report at least:

- total examples and batches;
- positives and negatives per task;
- users, queries, or sessions represented;
- candidate frequency and missing-value rate;
- per-batch metric variance and per-seed variance.

An absent gradient can mean the sampled window contained no useful supervision. Treat a task-level sensitivity result as provisional when positives or represented groups are too few. Prefer stratified fixed subsets over simply increasing an untracked time window.

## Representation geometry

Given embeddings `X` with shape `[n, d]`, inspect both raw and row-normalized values. The bundled collapse tool measures:

- centered energy: `mean(||x_i - mean(X)||²)`;
- mean pairwise cosine similarity;
- covariance effective rank: `exp(-sum(p_i log p_i))`, where `p_i` is normalized spectral energy;
- top singular-value energy share;
- per-dimension variance and near-zero dimensions;
- within-label and between-label geometry when labels exist.

Interpret the metrics jointly:

- near-zero centered energy suggests complete collapse;
- high cosine similarity with nonzero norm suggests directional collapse;
- low effective rank or high top-spectrum share suggests dimensional collapse;
- weak between-label separation relative to within-label spread suggests class-conditional collapse.

Do not use one universal threshold across models. Compare against an untrained or incumbent baseline, previous checkpoints, and a shuffled-label/control representation on the same sample.

## Activation and gradient export

The analyzer expects a compressed NumPy archive:

```text
activation::<hook_name>
gradient::<objective>::<hook_name>
```

For each Hook, activation and gradient must have identical shapes. Aggregate sequence or spatial axes consistently across candidates; preserve the feature-width axis. A minimal PyTorch export pattern is:

```python
captured = {}

def retain(name):
    def hook(_module, _inputs, output):
        output.retain_grad()
        captured[name] = output
    return hook

handles = [module.register_forward_hook(retain(name)) for name, module in hooks.items()]
model.eval()
model.zero_grad(set_to_none=True)
losses = model(batch)
losses[objective].backward()

arrays = {}
for name, tensor in captured.items():
    arrays[f"activation::{name}"] = tensor.detach().float().cpu().numpy()
    arrays[f"gradient::{objective}::{name}"] = tensor.grad.detach().float().cpu().numpy()
np.savez_compressed("hooks.npz", **arrays)
for handle in handles:
    handle.remove()
```

Adapt the forward and objective lookup to the repository. If several objectives are relevant, rerun backward from identical weights and batches, clearing gradients between objectives. Do not optimize on these diagnostic backward passes.

Optional Hook metadata groups only comparable candidates:

```json
{
  "hooks": {
    "tower.block2.ffn": {
      "group": "tower_ffn",
      "width": 256,
      "description": "post-activation output"
    }
  }
}
```

## Activation-times-gradient screening

For activation `a` and objective gradient `g`, the tool reports `mean(abs(a * g))`. This approximates the local first-order effect of suppressing a Hook. Use it to rank candidates only when they share:

- objective, validation sample, checkpoint, and numerical precision;
- Hook semantics and aggregation method;
- comparable widths or an explicit width-normalized comparison.

Select Top, middle/random, and Bottom candidates of equal width. The random control reveals whether any observed perturbation is specific to the rank or merely common to the layer family.

Sensitivity measures current-model reliance. It does not say that expanding a high-ranked module or deleting a low-ranked one will improve a retrained model.

## Paired perturbations

Run original and perturbed forwards with identical weights, examples, and stochastic state. Write one row per paired observation:

```text
candidate,seed,step,metric,original,perturbed,intervention,width
ffn_2,0,17,loss,0.431,0.447,shuffle,256
```

Use `delta = perturbed - original`; interpret the sign according to the metric. Recommended ladder:

1. Shuffle across examples while preserving marginal values.
2. Zero the same representation.
3. Remove or resize the component and retrain under the fixed protocol.

Shuffle tests whether example-specific information matters. Zero also removes fixed bias and scale. A near-zero Shuffle result with a harmful Zero result can indicate a non-informative but structurally relied-on offset, not a safely removable module.

Use block bootstrap intervals for correlated sequential rows and multiple independent seeds for the final comparison. A paired intervention validates a mechanism in the current checkpoint; only retraining validates a structural model change.

## Agent decision boundary

The diagnostic Skill may recommend the next controlled experiment, but it must not declare a new incumbent or rewrite durable memory. Pass the reports, artifact paths, validity flags, and code identity to `$model-optimization-harness-updater` after outcomes are available.
