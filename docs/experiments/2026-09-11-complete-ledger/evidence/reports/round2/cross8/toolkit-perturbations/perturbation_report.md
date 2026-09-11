# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round2/cross8/perturbations.tsv`
- Rows: `1026`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003390172 | [0.002774447, 0.003960801] | 0.9473684 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001080788 | [0.0005050449, 0.001732993] | 0.6666667 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.007572251 | [0.006612955, 0.00842679] | 0.9649123 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002438951 | [0.001901577, 0.002952004] | 0.8245614 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003159286 | [0.002648986, 0.00369933] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001110334 | [0.0005840857, 0.001581941] | 0.7192982 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.0001606646 | [-0.0003600275, 3.324568e-05] | 0.4385965 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003301524 | [0.002851077, 0.003752871] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008093934 | [0.0004500583, 0.001175055] | 0.6666667 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003694748 | [0.00319082, 0.004242231] | 0.9473684 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0002274539 | [-0.0001737863, 0.0005992193] | 0.5614035 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005252205 | [0.004564185, 0.005899885] | 0.9649123 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003785452 | [0.002934156, 0.004602834] | 0.9298246 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008095708 | [0.006980055, 0.009298657] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003616444 | [0.002863904, 0.004406391] | 0.8596491 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004941523 | [0.004331178, 0.005492286] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002524366 | [0.001978619, 0.003054435] | 0.877193 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
