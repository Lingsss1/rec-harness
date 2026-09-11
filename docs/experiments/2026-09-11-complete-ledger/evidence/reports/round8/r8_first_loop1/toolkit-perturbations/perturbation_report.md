# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round8/r8_first_loop1/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.003699163 | [0.002741012, 0.004619121] | 0.8596491 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.1056372 | [0.1022599, 0.1093226] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005961437 | [0.005057326, 0.006812975] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005201392 | [0.004100883, 0.006172073] | 0.9298246 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005481535 | [0.004813102, 0.006150512] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002796059 | [0.002345061, 0.003331806] | 0.9298246 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003130986 | [0.002579062, 0.003736237] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008706581 | [0.0004737852, 0.001279943] | 0.7719298 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0006141699 | [0.0004173419, 0.0008204172] | 0.8070175 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002400298 | [0.001927517, 0.002932629] | 0.9122807 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00182735 | [0.001411608, 0.002277291] | 0.877193 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00642474 | [0.00560164, 0.007336742] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004132852 | [0.003119326, 0.005048771] | 0.9122807 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003286432 | [0.002777255, 0.00382591] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002018152 | [0.001188207, 0.002813797] | 0.754386 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003178012 | [0.002574552, 0.003880319] | 0.9473684 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006222221 | [0.005139417, 0.007424113] | 0.9473684 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003188546 | [0.002605141, 0.003783856] | 0.9473684 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0007103509 | [0.0001597077, 0.001244051] | 0.7192982 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
