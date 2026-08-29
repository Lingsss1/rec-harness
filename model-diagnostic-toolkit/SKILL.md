---
name: model-diagnostic-toolkit
description: Run framework-independent diagnostics for GPU-based ranking and recommendation model optimization on open datasets, including representation collapse, activation and gradient health, activation×gradient screening, sample sufficiency, and paired Shuffle/Zero/ablation analysis. Use when Codex needs diagnostic evidence from NumPy arrays or experiment tables before choosing or evaluating a model change. Do not use it to update long-term research memory by itself.
---

# Model Diagnostic Toolkit

Produce auditable measurements from files available on the GPU machine through framework-independent arrays and tables. Keep measurement separate from memory evolution: this Skill creates evidence; `$model-optimization-harness-updater` decides what survives across rounds.

## Route by input

### Representations

Export a fixed validation subset as a matrix `[samples, dimensions]`, then run:

```bash
python3 <skill-dir>/scripts/diagnose_collapse.py EMBEDDINGS \
  --labels LABELS \
  --output-dir OUTPUT_DIR
```

Supported formats are `.npy`, `.npz`, `.csv`, `.tsv`, `.json`, `.pt`, and `.pth`. Omit labels when unavailable. Read both JSON and Markdown reports.

Use the result to distinguish:

- complete collapse: centered energy is nearly zero;
- directional concentration: unrelated samples share almost the same direction;
- dimensional concentration: variance occupies very few singular directions;
- class-conditional or local collapse: labels or task groups lose separation.

### Hook activations and gradients

Export a `.npz` file with keys:

```text
activation::<hook_name>
gradient::<objective>::<hook_name>
```

Each array must have the same shape for a given Hook, usually `[samples, width]`. Then run:

```bash
python3 <skill-dir>/scripts/analyze_hook_arrays.py HOOKS.npz \
  --config HOOKS.json \
  --output-dir OUTPUT_DIR
```

The optional config assigns Hooks to comparable groups. Read [references/core-diagnostic-methods.md](references/core-diagnostic-methods.md) before instrumenting the model.

### Shuffle, Zero, or ablation results

Export a long-form CSV/TSV with one row per paired observation:

```text
candidate,seed,step,metric,original,perturbed,intervention,width
```

Run:

```bash
python3 <skill-dir>/scripts/analyze_perturbations.py RESULTS.tsv \
  --output-dir OUTPUT_DIR
```

The analyzer computes paired deltas, block-bootstrap intervals, positive-delta frequency, and cross-seed summaries without depending on a training framework.

## Core workflow

1. Freeze the incumbent, validation subset, metric definitions, and experiment budget.
2. Check sample sufficiency, class/task composition, missing values, and metric variance.
3. Measure representation geometry and activation health.
4. Use `mean(abs(activation × gradient))` only to screen comparable Hooks within an objective.
5. Select equal-width Top, middle/random, and Bottom candidates; fix seeds.
6. Run same-weight, same-batch paired Shuffle or Zero interventions.
7. Confirm structural hypotheses with retraining under the unchanged validation protocol.
8. Pass compact reports and run metadata to `$model-optimization-harness-updater`.

If the result remains decision-relevant but ambiguous, the same ad-hoc analysis has been needed more than once, or a missing measurement is wasting experiment budget, invoke `$model-diagnostic-tool-evolver`. Let it validate a candidate in isolation; do not add exploratory code directly to this stable toolkit.

## Evidence boundaries

- Activation×gradient is a local first-order proxy, not causal importance.
- High sensitivity means the current model relies on a signal; it does not imply that expanding the module improves validation performance.
- Near-zero Shuffle is ambiguous until Zero and raw-variance checks separate redundancy from fixed bias or scale.
- A removable module requires deletion plus retraining evidence.
- Compare Hook ranks only within the same objective and comparable layer/group.
- Keep perturbation losses out of the optimizer and do not update weights between original and perturbed evaluations.
- Use validation metrics for iterative decisions; reserve test data for the final frozen incumbent.
- Record runtime, peak GPU memory, validity, and code identity alongside quality metrics.

## Bundled tools

- `diagnose_collapse.py`: representation geometry and collapse report.
- `analyze_hook_arrays.py`: activation/gradient summaries, within-group rankings, and equal-width candidate suggestions.
- `analyze_perturbations.py`: generic paired intervention statistics from CSV/TSV.
