# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round3/place_field2/perturbations.tsv`
- Rows: `1140`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.003650103 | [0.003111641, 0.004294538] | 0.9649123 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.0001989658 | [-8.816379e-05, 0.0005249392] | 0.5614035 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004693609 | [0.004036865, 0.005357068] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00236757 | [0.00156013, 0.003114941] | 0.8070175 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002557595 | [0.002077979, 0.003090926] | 0.9298246 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001448141 | [0.0005841212, 0.002376878] | 0.7368421 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004205007 | [0.003642075, 0.004814806] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0009817667 | [0.0003606345, 0.00151193] | 0.7368421 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 3.342147e-05 | [-0.0004643198, 0.0005341936] | 0.5438596 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005375251 | [0.004660704, 0.006117272] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004198053 | [0.003174974, 0.005202092] | 0.8947368 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.007656483 | [0.006778143, 0.008560017] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005224658 | [0.004517298, 0.0059122] | 0.9649123 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005627137 | [0.004886619, 0.006325825] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00289296 | [0.002060806, 0.003684547] | 0.8421053 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002164312 | [0.001771896, 0.002543625] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0002428445 | [-0.0002104139, 0.0006473936] | 0.5614035 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004190785 | [0.003557313, 0.004753134] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001184146 | [0.000431159, 0.00186082] | 0.7192982 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
