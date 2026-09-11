# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round14/r14_swiglu_both/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.007590191 | [0.006350201, 0.00894461] | 1 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.02623585 | [0.02389398, 0.02836419] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01305607 | [0.01132265, 0.01509014] | 0.9649123 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002938974 | [0.002220401, 0.003673922] | 0.9122807 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004105718 | [0.003170887, 0.004981189] | 0.9298246 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003926714 | [0.003368108, 0.004586615] | 0.9298246 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001327868 | [0.000779229, 0.001871981] | 0.7192982 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008918889 | [0.007763524, 0.01024907] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003364999 | [0.002510202, 0.004213662] | 0.877193 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0003574166 | [0.0001449731, 0.0005712288] | 0.7192982 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005008898 | [0.004259441, 0.005750315] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001486219 | [0.00102726, 0.001942068] | 0.754386 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004125731 | [0.003477667, 0.004875627] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0004824729 | [4.400622e-05, 0.0008563705] | 0.5614035 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002761467 | [0.002198236, 0.003408375] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0009237826 | [0.0003903304, 0.001416972] | 0.7017544 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004051867 | [0.003454617, 0.004671752] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001139674 | [0.0005799837, 0.001648132] | 0.7017544 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004040299 | [0.003357818, 0.004753735] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001635953 | [0.000957572, 0.002349959] | 0.7192982 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
