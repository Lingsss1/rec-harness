# Matched optimizer-state replay: method card

This candidate diagnoses the frozen local HeavyBall 3.2.1 implementation. No claim is made that synthetic CPU eager effects explain the eight Taobao AUC declines. The original 12-step design is `recurrence-readout-retrieval/reports/optimizer-layout-replay-design.md`; source/threshold identities will be frozen before execution.

## Decision and alternatives

1. Gradient-norm logging: cheapest; detects a changed global clip coefficient, but does not identify Q initialization or probes. Retained as an observable, insufficient as the full method.
2. Q-initialization inspection: tests the common fresh-bucket scale; cannot distinguish stochastic preconditioner updates. Retained as one factor.
3. Matched one-step state replay: selected. Restore identical inputs/RNG; intervene separately on old clipped gradients C, initial Q slices S and random vectors R. Compare all eight C/S/R combinations for shared layout plus task-layout endpoints. This can reveal interactions and unresolved residuals without assuming additivity.
4. Repeated full training: rejected for this measurement because it does not isolate these mechanisms and would spend GPU time; additional training seeds remain prohibited.

## Primary sources and boundaries

- [HeavyBall official implementation](https://github.com/HomebrewML/HeavyBall): compiled composable optimizer architecture. Local source, not mutable upstream HEAD, defines the measured buckets and transforms; BSD-2-Clause license retained in the installed package. We call the existing library and add repository-local observation wrappers, without copying or modifying its optimizer implementation.
- [PyTorch RNG API](https://docs.pytorch.org/docs/stable/random.html): CPU RNG state can be captured/restored and fork_rng restores prior state. The installed API is checked locally; `devices=[]` excludes accelerators.
- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html): reproducibility is not guaranteed across platforms/releases. CPU eager replay cannot stand in for CUDA compiled replay, even with a shared seed.
- [Li, Preconditioned Stochastic Gradient Descent, 2015](https://arxiv.org/abs/1512.04202): motivates estimating preconditioners from stochastic perturbations. It does not establish this library's bucket/RNG coupling or any Taobao model outcome.

`paper_claim`: only the API/algorithm statements above. `agent_inference`: adding parameters may alter old updates through the three local implementation paths. `local_result`: none at registration. Complexity is tiny fixed tensors with 12 optimizer steps; no dataset scaling or statistical sampling claim.

## Inputs and outputs

Formula-generated FP32 parameters and gradients with fixed member names/order; baseline has two small matrix parameters and four [1] biases. Shared insertion has [1] gradient4; task insertion has [4] gradient[1,1,1,1]. Save exact bytes and captured RNG; do not call manual_seed. Keep lr1e-4, decay0, original defaults and clip1. The synthetic experiment has no sparse optimizer and does not model its cost.

Output JSON plus small tensor artifacts: raw/clipped gradients, bucket members, initial/final Q and other state, V, dampened gradients, update vectors, exact/difference metrics, restored-wrapper/RNG guards, per-condition completion and measured resources. Signed update differences are descriptive; vector discrepancies use nonnegative absolute/L2 differences. No confidence intervals across parameter coordinates, no AUC estimate, no training-seed inference.

## Controls, validity and action

Two baseline steps from the same snapshots compare unwrapped and observation-only paths; outputs, state and ending RNG must be exact. C/S/R fixing must restore the respective old-member quantity exactly; extra members remain native. Synthetic correctness requires detection of the analytically changed clip input, correct member/shape accounting, exact negative control, and honest residual classification. All3-fixed need not prove every update path explained; residuals prohibit an optimizer-preserving claim. R fixing does not restore the layout's final global RNG by itself.

Pure metadata edge tests reject missing/duplicate names, wrong shapes, nonfinite formula inputs and malformed freeze/output paths without optimizer steps. Actual captures must be finite and map all members. Unexpected wrapper calls or missing local injection points stop the attempt; no automatic retries or estimator changes after results.

Before any model-edit handoff, require separate native compiled shadow on at least two preselected conditions and a decision-utility comparison against the two baselines; these gates are pending and are not run in the first-stage budget. A successful synthetic result is at most synthetic_pass, never accepted or promoted. Quantitative training gains require the original unchanged training and full-validation comparison.

## Cost

12 synthetic optimizer steps at most; CPU4,120s wall,480s accumulated CPU,2GiB RAM,64MiB artifacts; no real data, model forward/backward, GPU or training. Record optimizer_replay_steps separately from training_updates=0. Preserve any failed receipt and measured partial costs. A tiny fixed shape study does not validate arbitrary tensor sizes or compiled execution.
