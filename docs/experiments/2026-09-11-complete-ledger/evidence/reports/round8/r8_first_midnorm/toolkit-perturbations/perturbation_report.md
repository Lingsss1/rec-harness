# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round8/r8_first_midnorm/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.002301555 | [0.001755721, 0.002890975] | 0.8947368 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.03589982 | [0.03411496, 0.0377288] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.03040914 | [0.0286478, 0.03220735] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003685263 | [0.003166728, 0.004139616] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001975917 | [0.001449217, 0.002487616] | 0.877193 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001659591 | [0.001304434, 0.002074161] | 0.877193 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001124879 | [0.0005220431, 0.001659459] | 0.7192982 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.007612284 | [0.006750642, 0.0084414] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003047314 | [0.002363132, 0.003737313] | 0.877193 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -4.112537e-05 | [-0.0004710591, 0.0003457042] | 0.5263158 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005889896 | [0.005159313, 0.006625642] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002522156 | [0.001814101, 0.003253248] | 0.8070175 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00382748 | [0.003398674, 0.004292285] | 0.9473684 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001096013 | [0.0005195477, 0.001675796] | 0.7192982 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003649799 | [0.003125163, 0.004159186] | 0.9649123 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0005694296 | [0.000285799, 0.0008437304] | 0.754386 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003195125 | [0.002708016, 0.003719602] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006527138 | [0.005413356, 0.007622708] | 0.9298246 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003522879 | [0.003030138, 0.003998442] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00139919 | [0.0009589406, 0.001811894] | 0.8245614 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
