# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round17/r17_base_phase132/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.004434367 | [0.003570892, 0.00524985] | 0.9122807 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.02088193 | [0.01920697, 0.02265703] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.006946468 | [0.005719554, 0.008302679] | 0.8947368 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006297907 | [0.005426312, 0.007142072] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0004959716 | [4.815271e-05, 0.0009061987] | 0.6842105 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005541652 | [0.004848529, 0.006283838] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002401883 | [0.001693131, 0.003051185] | 0.8245614 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004367768 | [0.003654256, 0.005117141] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008902675 | [0.0004333744, 0.001340184] | 0.7894737 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 6.411395e-05 | [-0.0001026571, 0.0002367779] | 0.4912281 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00460606 | [0.003906235, 0.005279741] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0003831209 | [7.191487e-05, 0.0007101678] | 0.5614035 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001890739 | [0.001376783, 0.002496129] | 0.8596491 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006188866 | [0.005000437, 0.007332343] | 0.9122807 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.0009342738 | [0.0006807071, 0.001158831] | 0.8421053 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.0002068814 | [-0.000496922, 6.962197e-05] | 0.5263158 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001958295 | [0.001503986, 0.002449427] | 0.8947368 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004528493 | [0.003586744, 0.005538029] | 0.8947368 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00510059 | [0.004381532, 0.005818275] | 0.9473684 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0006405424 | [-3.670391e-05, 0.001214495] | 0.6842105 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
