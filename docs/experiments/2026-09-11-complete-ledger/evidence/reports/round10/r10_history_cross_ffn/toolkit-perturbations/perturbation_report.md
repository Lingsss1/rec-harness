# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round10/r10_history_cross_ffn/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.002273907 | [0.001776653, 0.002759467] | 0.8245614 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.05558543 | [0.05300407, 0.05814116] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01303185 | [0.01179376, 0.01436268] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008504748 | [0.007442667, 0.009542008] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001743143 | [0.001308897, 0.002272878] | 0.8421053 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003623451 | [0.003005743, 0.004284441] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002322795 | [0.001558008, 0.003037606] | 0.8421053 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004548463 | [0.003821085, 0.005247531] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002777056 | [0.001973167, 0.00356409] | 0.8421053 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.0001137476 | [-0.0003875079, 0.0001427619] | 0.4385965 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003584795 | [0.003121049, 0.004052472] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0004155835 | [-0.0001602606, 0.0009174998] | 0.5614035 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003404934 | [0.002904213, 0.003940274] | 0.9473684 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001669118 | [0.001198508, 0.002091841] | 0.8421053 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003553776 | [0.003038024, 0.004085841] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0007505791 | [2.753748e-05, 0.001442322] | 0.6140351 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002055408 | [0.001713677, 0.002441621] | 0.8947368 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002015411 | [0.001230793, 0.002810331] | 0.7719298 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002942677 | [0.002398299, 0.003470024] | 0.9298246 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001976518 | [0.001191775, 0.002708474] | 0.7719298 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
