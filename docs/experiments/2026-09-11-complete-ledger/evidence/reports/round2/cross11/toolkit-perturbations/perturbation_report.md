# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round2/cross11/perturbations.tsv`
- Rows: `1026`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0.0002326056 | [-0.0005265548, 0.001009245] | 0.4912281 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.01080727 | [0.009543154, 0.0120183] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004826906 | [0.004198929, 0.005462975] | 1 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004464657 | [0.003718748, 0.005211155] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.009664704 | [0.008774726, 0.01056592] | 1 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.0007103342 | [0.0004466658, 0.001002776] | 0.7017544 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.02341171 | [0.02180078, 0.02495331] | 1 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.000704637 | [-0.001106886, -0.0002803361] | 0.3333333 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002584082 | [0.001957047, 0.00325338] | 0.877193 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.01901177 | [0.01767078, 0.0203011] | 1 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002658376 | [0.001992088, 0.003364633] | 0.8947368 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.008147634 | [0.007374103, 0.008892514] | 1 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008878827 | [0.007579796, 0.0101528] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005476522 | [0.004626781, 0.00637664] | 0.9824561 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002253092 | [0.001538514, 0.002954692] | 0.8245614 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002198474 | [0.001790911, 0.00257298] | 0.9122807 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
