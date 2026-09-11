# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round9/r9_untied_clone/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.002947463 | [0.002398719, 0.003535658] | 0.877193 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.03380595 | [0.03207761, 0.03550754] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01708368 | [0.0156545, 0.01865286] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003888084 | [0.003342816, 0.004371615] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001270032 | [0.0008168069, 0.001686109] | 0.7719298 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006633877 | [0.005816349, 0.007497139] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004064748 | [0.00328767, 0.004783819] | 0.9473684 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005098312 | [0.004315978, 0.005893318] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001665182 | [0.00109869, 0.002281295] | 0.8070175 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -6.237156e-05 | [-0.0004470382, 0.000326956] | 0.5614035 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002513493 | [0.002034868, 0.002991743] | 0.9473684 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002214584 | [0.001485402, 0.003010231] | 0.754386 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002969961 | [0.002512329, 0.003437662] | 0.9122807 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004943267 | [0.00395309, 0.005996672] | 0.8947368 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005134407 | [0.004424511, 0.00579409] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002783121 | [0.002277853, 0.003293947] | 0.8947368 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002788259 | [0.002417777, 0.003159405] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003981921 | [0.00307216, 0.004904994] | 0.8596491 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003768562 | [0.003272373, 0.004278227] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0002377744 | [-3.561613e-06, 0.0004764912] | 0.6315789 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
