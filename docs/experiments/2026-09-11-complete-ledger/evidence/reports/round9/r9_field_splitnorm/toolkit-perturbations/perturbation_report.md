# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round9/r9_field_splitnorm/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.002435488 | [0.001693102, 0.003329838] | 0.7894737 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.03927458 | [0.03740428, 0.0411725] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.03250578 | [0.03078626, 0.03428749] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004925145 | [0.004228026, 0.005534275] | 0.9473684 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003080817 | [0.002529101, 0.003598895] | 0.9122807 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001492094 | [0.001198069, 0.00181267] | 0.877193 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0005896708 | [0.0002468311, 0.0009301071] | 0.7017544 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005944118 | [0.005180041, 0.006682427] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00374699 | [0.003018624, 0.004456924] | 0.9298246 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0005200905 | [9.150132e-05, 0.000892673] | 0.6842105 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003669883 | [0.003138613, 0.004244693] | 0.9473684 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003196686 | [0.002470411, 0.003922695] | 0.8947368 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005060506 | [0.004349447, 0.005717029] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002165778 | [0.001502802, 0.002792328] | 0.8245614 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00351786 | [0.002888448, 0.004177373] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002026424 | [0.001494014, 0.002543904] | 0.9122807 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004140559 | [0.003500043, 0.004786904] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00731084 | [0.006184143, 0.0085359] | 0.9298246 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00165221 | [0.001253508, 0.002048142] | 0.8421053 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001554338 | [0.001049984, 0.002039252] | 0.8421053 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
