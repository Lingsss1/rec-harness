# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round12/r12_history_afc/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.001054093 | [0.0006084282, 0.001493155] | 0.754386 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.02734986 | [0.02535309, 0.02909756] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.008074033 | [0.007088321, 0.009052109] | 0.9473684 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002233576 | [0.001839715, 0.002611004] | 0.9298246 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.000633784 | [0.0002829159, 0.0009776672] | 0.6842105 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004027845 | [0.003412136, 0.004738832] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002244486 | [0.001641108, 0.002794061] | 0.8421053 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003163191 | [0.002513143, 0.00391599] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001975992 | [0.001428858, 0.002509072] | 0.877193 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0001209065 | [-0.0001645686, 0.0003890269] | 0.4912281 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002179024 | [0.001699655, 0.002678425] | 0.9298246 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001835069 | [0.001338035, 0.002360823] | 0.8421053 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004000414 | [0.00351546, 0.00446917] | 0.9649123 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001906126 | [0.00113738, 0.002562052] | 0.8245614 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002400871 | [0.001943791, 0.002860573] | 0.8596491 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0005794889 | [0.000210254, 0.0009292079] | 0.6842105 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00703548 | [0.006060011, 0.00803035] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003195753 | [0.002576151, 0.003849947] | 0.8596491 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004330755 | [0.003543621, 0.005137201] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004645632 | [0.003813654, 0.005461164] | 0.9298246 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
