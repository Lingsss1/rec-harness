# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round15/r15_w512_field/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.003478407 | [0.002906105, 0.004033234] | 0.9122807 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.0268975 | [0.02479519, 0.02873759] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01538873 | [0.01420775, 0.01662027] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004659792 | [0.003812409, 0.005417999] | 0.9649123 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003969526 | [0.003138405, 0.004714884] | 0.9122807 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008299826 | [0.007244976, 0.009338092] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001990036 | [0.001547167, 0.002449299] | 0.8421053 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003927947 | [0.003155429, 0.004700126] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003683558 | [0.002812921, 0.004549373] | 0.8947368 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0005743634 | [8.033758e-05, 0.001060811] | 0.6140351 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002799691 | [0.002346257, 0.003238863] | 0.9473684 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004014828 | [0.003164732, 0.004923268] | 0.9122807 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003126126 | [0.002545451, 0.003766004] | 0.9298246 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0009561424 | [0.0004949676, 0.00137718] | 0.6842105 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005302634 | [0.004509155, 0.006142138] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001587065 | [0.001269222, 0.001908565] | 0.9122807 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001690702 | [0.001248565, 0.002180263] | 0.8947368 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001223243 | [0.0007092781, 0.001753101] | 0.7368421 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003494116 | [0.002870171, 0.004099003] | 0.9473684 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002118058 | [0.001432171, 0.00274027] | 0.8245614 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
