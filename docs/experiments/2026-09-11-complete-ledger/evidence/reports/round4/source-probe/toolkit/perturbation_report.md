# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round4/source-probe/perturbations.tsv`
- Rows: `171`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `context` | `source_selection` | 30 | `20262027` | `logloss` | 57 | 6.953833e-06 | [-0.0004194477, 0.0004038842] | 0.5438596 |
| `no_extra_norm` | `norm_ablation` | 37 | `20262027` | `logloss` | 57 | -3.1377e-06 | [-1.972613e-05, 1.319841e-05] | 0.4035088 |
| `target` | `source_selection` | 7 | `20262027` | `logloss` | 57 | 0.005372996 | [0.004610049, 0.006152304] | 1 |

## Warnings

- source_selection/logloss contains multiple widths [7, 30]; do not compare candidates as equal-cost controls.

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
