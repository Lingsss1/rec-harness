# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round18/r18_base_norm_phase132/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.00477384 | [0.004010993, 0.005506429] | 0.8947368 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.0165692 | [0.01517371, 0.01813009] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.00850578 | [0.007331371, 0.009754708] | 0.9649123 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006288348 | [0.005338685, 0.007107483] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001316244 | [0.0008085433, 0.001827413] | 0.7894737 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003963128 | [0.003388906, 0.004565334] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0004765381 | [-2.489738e-05, 0.000933733] | 0.5964912 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002775483 | [0.002243372, 0.003373165] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002170596 | [0.001364872, 0.002984859] | 0.7719298 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0002032348 | [-8.141138e-05, 0.0004548543] | 0.6140351 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005280716 | [0.004482563, 0.006079105] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.000844647 | [0.0003621156, 0.001318959] | 0.7192982 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.0028894 | [0.00245765, 0.003380019] | 0.9473684 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006905072 | [0.005690985, 0.007995764] | 0.8947368 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.0008048146 | [0.0005695078, 0.001032907] | 0.8070175 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.0002890676 | [-0.0005972519, 2.535709e-05] | 0.3859649 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002944393 | [0.002495193, 0.003364453] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0007657207 | [0.0001870229, 0.001375693] | 0.7192982 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005728694 | [0.004998077, 0.006494675] | 0.9824561 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001815766 | [0.00103927, 0.002567694] | 0.7719298 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
