# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round13/r13_query_between_zero/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.003252467 | [0.002596929, 0.003911058] | 0.8947368 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.04786307 | [0.04576563, 0.04999998] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.02794627 | [0.02629619, 0.02966723] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006212482 | [0.00541792, 0.007018154] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003048765 | [0.002489556, 0.003639451] | 0.8947368 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002612028 | [0.002111951, 0.003178761] | 0.9649123 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001538879 | [0.001035713, 0.002026308] | 0.8245614 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006203664 | [0.005490816, 0.006881713] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00476425 | [0.003866674, 0.005647764] | 0.9122807 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.000868092 | [0.0003013319, 0.001403423] | 0.6842105 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004631022 | [0.004040892, 0.005271154] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006668248 | [0.005612055, 0.007737658] | 0.9122807 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003109451 | [0.002672561, 0.003538096] | 0.9473684 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002752058 | [0.001949228, 0.003560204] | 0.8596491 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004102711 | [0.003335161, 0.004938213] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002101332 | [0.001545156, 0.002723747] | 0.9122807 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.0069092 | [0.006141104, 0.007608576] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002438058 | [0.001740721, 0.003107264] | 0.8245614 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003902502 | [0.003139277, 0.004767289] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001593967 | [0.001047831, 0.002107723] | 0.8421053 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
