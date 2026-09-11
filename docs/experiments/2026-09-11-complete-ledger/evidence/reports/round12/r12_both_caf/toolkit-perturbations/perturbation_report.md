# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round12/r12_both_caf/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.005162593 | [0.004376732, 0.006083325] | 0.9473684 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.03996795 | [0.0377924, 0.04204777] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.0104227 | [0.009174183, 0.01157282] | 0.9473684 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002721786 | [0.002189284, 0.003354782] | 0.9122807 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001984961 | [0.001322201, 0.002614588] | 0.8070175 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002454021 | [0.001874569, 0.003134215] | 0.8947368 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004296834 | [0.003307891, 0.005145222] | 0.8596491 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002431688 | [0.00187746, 0.003010749] | 0.8947368 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001617853 | [0.00091069, 0.002284508] | 0.7894737 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0006629025 | [0.0003099416, 0.0009882776] | 0.6842105 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006331989 | [0.005485375, 0.007163381] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002516264 | [0.001908412, 0.003134858] | 0.8947368 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003093851 | [0.002575654, 0.003618201] | 0.9473684 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001944602 | [0.001197069, 0.002739767] | 0.8245614 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005410114 | [0.004583502, 0.006259883] | 0.9649123 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002190212 | [0.001508984, 0.002779932] | 0.8245614 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001779411 | [0.001409104, 0.002127742] | 0.8947368 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0006407204 | [4.121804e-05, 0.001245984] | 0.6491228 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005561455 | [0.004709585, 0.006405321] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002061696 | [0.001709273, 0.002443134] | 0.9298246 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
