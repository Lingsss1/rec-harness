# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round3/place_feedback2/perturbations.tsv`
- Rows: `1140`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.008435817 | [0.007473182, 0.009435986] | 0.9824561 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.001510868 | [0.001208011, 0.001820267] | 0.877193 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006776386 | [0.005981773, 0.007598304] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001962931 | [0.001376985, 0.002503114] | 0.7894737 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.007794569 | [0.006995572, 0.008686037] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002297899 | [0.001851188, 0.002705334] | 0.9298246 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001292439 | [0.00104144, 0.00153282] | 0.9122807 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001354851 | [0.0006518571, 0.001992119] | 0.7368421 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -9.808624e-05 | [-0.0006263553, 0.0004163908] | 0.5087719 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00461761 | [0.00400818, 0.00520048] | 0.9473684 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005351717 | [0.004277157, 0.006386418] | 0.8947368 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006405881 | [0.005650669, 0.00723473] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002461336 | [0.001750463, 0.003155718] | 0.8421053 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00383208 | [0.003304457, 0.004357143] | 0.9649123 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001772689 | [0.001082097, 0.002501301] | 0.7719298 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004861139 | [0.004186237, 0.005494252] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001291113 | [0.0008270023, 0.001744253] | 0.8070175 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001460467 | [0.001179025, 0.001753097] | 0.9473684 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001178576 | [0.0004256581, 0.001841969] | 0.6842105 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
