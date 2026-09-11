# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round17/r17_field512_phase132/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.007777718 | [0.006731965, 0.008993627] | 0.9824561 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.009194869 | [0.008139307, 0.01032572] | 0.9824561 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.007133696 | [0.00625764, 0.008042925] | 0.9473684 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003291583 | [0.002640665, 0.003898161] | 0.9298246 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002039698 | [0.001330927, 0.002673761] | 0.8421053 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005196413 | [0.00445015, 0.005977222] | 0.9649123 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0002219983 | [-7.370484e-05, 0.0004852971] | 0.5789474 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004004724 | [0.003241845, 0.004708154] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001696217 | [0.001130293, 0.00223878] | 0.8070175 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.0001454348 | [-0.0005315905, 0.0002077707] | 0.4912281 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004361938 | [0.003648352, 0.005141055] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001649185 | [0.000914817, 0.002346594] | 0.7719298 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005397279 | [0.004614249, 0.006213412] | 0.9473684 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008788804 | [0.0003774808, 0.001297459] | 0.7017544 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003195877 | [0.002546649, 0.003909454] | 0.9122807 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001555375 | [0.0009754786, 0.002082329] | 0.8070175 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004014042 | [0.003363874, 0.004706793] | 0.9298246 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002377302 | [0.001661373, 0.003074144] | 0.8070175 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00284052 | [0.002275909, 0.003387403] | 0.9298246 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001046721 | [0.0005811545, 0.001493529] | 0.7368421 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
