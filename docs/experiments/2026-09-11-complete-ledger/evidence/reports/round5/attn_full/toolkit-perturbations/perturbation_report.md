# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round5/attn_full/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.004410553 | [0.003650367, 0.005214274] | 0.9298246 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.008911037 | [0.007907723, 0.009926843] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01360312 | [0.01232023, 0.01499985] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005147417 | [0.004414897, 0.005822495] | 0.9473684 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001525052 | [0.001138655, 0.001912844] | 0.8421053 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004809089 | [0.00420298, 0.005424043] | 0.9649123 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001806314 | [0.00143919, 0.002148175] | 0.877193 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00174172 | [0.001347481, 0.002139434] | 0.877193 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0006091244 | [0.0001452752, 0.001074118] | 0.6842105 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0004149139 | [-0.0001320866, 0.0009338723] | 0.5964912 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002742456 | [0.002263948, 0.003218935] | 0.9473684 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006118877 | [0.005064935, 0.007242668] | 0.9298246 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006758959 | [0.00588962, 0.007573015] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002898164 | [0.002326957, 0.003465544] | 0.9122807 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004254621 | [0.00363486, 0.004897316] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.01603 | [0.01446549, 0.01759264] | 0.9824561 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005451611 | [0.004701026, 0.006162243] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 4.508778e-06 | [-0.0002086313, 0.0002245886] | 0.5087719 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003333364 | [0.002792996, 0.003859965] | 0.9298246 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001787332 | [0.001105979, 0.002480169] | 0.7894737 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
