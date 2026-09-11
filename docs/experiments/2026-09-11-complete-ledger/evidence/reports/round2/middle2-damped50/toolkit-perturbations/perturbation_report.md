# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round2/middle2-damped50/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `inference_loops1` | `loops1` | 256 | `20262027` | `logloss` | 57 | 0.0007398965 | [0.0003832341, 0.001080141] | 0.7719298 |
| `inference_loops3` | `loops3` | 256 | `20262027` | `logloss` | 57 | 0.000525148 | [0.0001604245, 0.0008789712] | 0.6315789 |
| `inference_loops4` | `loops4` | 256 | `20262027` | `logloss` | 57 | 0.002041688 | [0.001446719, 0.002727547] | 0.8245614 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001147086 | [0.0008290371, 0.001432749] | 0.7894737 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004480837 | [0.003412709, 0.005566788] | 0.8245614 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008572406 | [0.007649968, 0.009528032] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001284314 | [0.000782725, 0.00177128] | 0.8070175 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006269652 | [0.005454009, 0.007126346] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001152231 | [0.000567025, 0.00173362] | 0.754386 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.001171374 | [0.0009231246, 0.001459933] | 0.8947368 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002905215 | [0.00240557, 0.003449988] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003071761 | [0.002577433, 0.00357701] | 0.9649123 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006922371 | [0.006088968, 0.007743404] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002850677 | [0.00208735, 0.003511272] | 0.877193 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005834935 | [0.004978675, 0.006692442] | 0.9649123 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001832038 | [0.001203506, 0.002358054] | 0.8070175 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004127516 | [0.003470469, 0.00481024] | 0.9473684 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008760468 | [0.0004665845, 0.001247566] | 0.7192982 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004251601 | [0.003617017, 0.004930882] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002118294 | [0.00132662, 0.002858303] | 0.754386 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
