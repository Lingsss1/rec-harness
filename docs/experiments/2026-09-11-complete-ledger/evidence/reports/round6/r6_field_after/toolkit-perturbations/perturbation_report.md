# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round6/r6_field_after/perturbations.tsv`
- Rows: `1140`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.05184271 | [0.04911365, 0.05438849] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.007650358 | [0.006628974, 0.008787717] | 0.9824561 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.007605437 | [0.006711179, 0.008485054] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002581488 | [0.002122249, 0.003046949] | 0.9824561 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002243533 | [0.001860485, 0.00269339] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0005222549 | [-0.0002509911, 0.00126717] | 0.5614035 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002471828 | [0.002070256, 0.002872155] | 0.9298246 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.0007412483 | [-0.001235983, -0.0003352932] | 0.3859649 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.001126614 | [-0.001678117, -0.0005025027] | 0.2982456 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.0001812538 | [-0.0001620325, 0.000528722] | 0.5614035 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002810932 | [0.001684228, 0.003999336] | 0.7017544 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00652012 | [0.005670398, 0.007385279] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002062833 | [0.001578355, 0.002571471] | 0.8421053 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004639689 | [0.003962816, 0.005270991] | 0.9649123 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00172941 | [0.0008487311, 0.002636254] | 0.7017544 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005924464 | [0.005151357, 0.006692978] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001043336 | [0.0002757058, 0.00175543] | 0.6666667 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006559523 | [0.005650534, 0.007447447] | 0.9824561 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.000169285 | [-0.0003558862, 0.0006788822] | 0.5964912 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
