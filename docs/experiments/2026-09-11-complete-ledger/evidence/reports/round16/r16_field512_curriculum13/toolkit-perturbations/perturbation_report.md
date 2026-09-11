# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round16/r16_field512_curriculum13/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.00698144 | [0.006272267, 0.007695255] | 1 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.002566443 | [0.001805103, 0.003275428] | 0.8245614 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | -0.00205553 | [-0.002636026, -0.001508989] | 0.1929825 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005493199 | [0.004630886, 0.006311492] | 0.9649123 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005637995 | [0.004924169, 0.006335884] | 0.9649123 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005459549 | [0.00473483, 0.006279413] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002266072 | [0.00184888, 0.002662021] | 0.877193 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002761181 | [0.00237214, 0.003123054] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001964795 | [0.001625608, 0.002304362] | 0.9122807 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.001057479 | [0.0009064278, 0.001197263] | 0.9473684 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003605209 | [0.003046002, 0.004192485] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002804147 | [0.002543674, 0.00307294] | 0.9824561 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001495547 | [0.001069377, 0.002012031] | 0.8421053 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002683797 | [0.002004249, 0.003307336] | 0.8947368 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002256717 | [0.001801049, 0.002748508] | 0.877193 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002480929 | [0.001782986, 0.003181906] | 0.8245614 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001895464 | [0.001458348, 0.002385356] | 0.8245614 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.009463634 | [0.008418417, 0.01054204] | 0.9649123 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005322048 | [0.004579785, 0.006028958] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004623717 | [0.003973503, 0.005230839] | 0.9824561 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
