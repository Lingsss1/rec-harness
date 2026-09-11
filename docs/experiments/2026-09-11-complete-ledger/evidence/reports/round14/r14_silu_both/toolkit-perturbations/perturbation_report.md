# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round14/r14_silu_both/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.00524139 | [0.004522942, 0.006051316] | 0.9473684 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.02812753 | [0.02544773, 0.03064293] | 0.9824561 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.00946106 | [0.008422056, 0.01058053] | 0.9824561 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008064734 | [0.007125276, 0.008897853] | 1 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002764788 | [0.002166223, 0.003326838] | 0.8596491 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001244922 | [0.0008974453, 0.00164837] | 0.8596491 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0003968054 | [0.0001728675, 0.000635766] | 0.6666667 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006801298 | [0.00593845, 0.007635694] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.000443079 | [-9.711315e-05, 0.001000257] | 0.6140351 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0008891191 | [0.0006188462, 0.001124135] | 0.8421053 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006919952 | [0.006158299, 0.007639786] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002709353 | [0.00218133, 0.003238668] | 0.9298246 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003511883 | [0.002891308, 0.004189822] | 0.8947368 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0004548676 | [-0.0001586319, 0.001010833] | 0.5964912 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005110201 | [0.004507067, 0.00573627] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001451913 | [0.0006627942, 0.002247303] | 0.7017544 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006130663 | [0.005461034, 0.00676351] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00161111 | [0.0007775222, 0.002491143] | 0.7368421 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003193623 | [0.002697528, 0.003716474] | 0.9298246 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00307782 | [0.002090483, 0.004104218] | 0.8070175 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
