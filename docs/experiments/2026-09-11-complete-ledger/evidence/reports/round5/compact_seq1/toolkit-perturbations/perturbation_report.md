# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round5/compact_seq1/perturbations.tsv`
- Rows: `1140`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.01147091 | [0.01025862, 0.01267046] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01425645 | [0.01270023, 0.01563807] | 0.9649123 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004025353 | [0.003383211, 0.00470776] | 0.9473684 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006811609 | [0.005703351, 0.00806362] | 0.9122807 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.009059978 | [0.008148264, 0.01004583] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003735322 | [0.003094652, 0.004403028] | 0.9298246 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003421617 | [0.002834535, 0.004004763] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001111235 | [0.0005227334, 0.001646215] | 0.7368421 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.0002127172 | [-0.0004934813, 3.925965e-05] | 0.4210526 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003229427 | [0.002731102, 0.003744058] | 0.9473684 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0004693 | [-0.0001764131, 0.0009924702] | 0.5964912 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004241578 | [0.003598116, 0.004876225] | 0.9649123 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001208275 | [0.0007646079, 0.001671974] | 0.754386 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008548394 | [0.007523735, 0.009599098] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001581595 | [0.001136391, 0.002051194] | 0.8070175 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004835259 | [0.003992588, 0.005599674] | 0.9473684 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001201131 | [0.0006007721, 0.001816312] | 0.754386 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003882254 | [0.003286912, 0.004457407] | 0.9298246 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00302584 | [0.002073883, 0.003910226] | 0.8070175 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
