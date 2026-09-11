# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round11/r11_history_gate/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.003307355 | [0.002665774, 0.003970495] | 0.9122807 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.03699119 | [0.03514388, 0.0387786] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.02115744 | [0.01969692, 0.02264978] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004831639 | [0.004123952, 0.005413048] | 0.9649123 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001870561 | [0.001419448, 0.00227724] | 0.8421053 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003780447 | [0.003266009, 0.004332095] | 0.9649123 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00119108 | [0.0008280746, 0.001530386] | 0.7894737 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006931609 | [0.006063764, 0.007801788] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002832005 | [0.002122149, 0.003523233] | 0.8421053 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 5.054239e-05 | [-0.0004322581, 0.0005347102] | 0.5614035 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003166686 | [0.00267063, 0.003707507] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002585674 | [0.001730122, 0.003439111] | 0.8245614 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003443018 | [0.003006848, 0.003921352] | 0.9649123 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0002475742 | [-9.388068e-05, 0.0005407521] | 0.6315789 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004597761 | [0.003955712, 0.005231336] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001714381 | [0.001308696, 0.002115471] | 0.8596491 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006670249 | [0.005861325, 0.007428137] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.01002644 | [0.008588241, 0.01139225] | 0.9473684 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001812979 | [0.001423744, 0.002166513] | 0.9122807 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0009420125 | [0.0004602687, 0.001371206] | 0.7368421 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
