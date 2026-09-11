# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round16/r16_base_curriculum13/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.0006891567 | [-0.0002855671, 0.001736987] | 0.5438596 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.01568363 | [0.01424948, 0.01717139] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | -0.002684157 | [-0.003678698, -0.001463215] | 0.1403509 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005151217 | [0.00384188, 0.00651864] | 0.8596491 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.001576323 | [-0.002557392, -0.0006127063] | 0.3333333 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001724289 | [0.001033219, 0.002476852] | 0.7719298 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.001208387 | [-0.001917311, -0.0005262515] | 0.3508772 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003286747 | [0.002336426, 0.004419424] | 0.8596491 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.001598593 | [-0.002086953, -0.001109213] | 0.2280702 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.0007980978 | [-0.001083562, -0.0005132662] | 0.2105263 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001318356 | [0.0008058615, 0.00194779] | 0.7719298 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.0007158128 | [-0.001277307, -0.0001060994] | 0.4035088 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002022631 | [0.001249895, 0.002962552] | 0.7368421 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.0006522202 | [-0.001666709, 0.0002398555] | 0.4035088 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.0007374998 | [0.000414469, 0.001100053] | 0.754386 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.001239385 | [-0.001572914, -0.0009164393] | 0.122807 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | -0.0001820033 | [-0.0005595633, 0.0002279621] | 0.3859649 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0005617466 | [-0.0004264466, 0.001610052] | 0.5614035 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004519033 | [0.003424186, 0.005701419] | 0.9122807 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.001288136 | [-0.00198588, -0.000648449] | 0.2631579 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
