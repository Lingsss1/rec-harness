# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round11/r11_history_embedding/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.003847338 | [0.003064, 0.004668467] | 0.8947368 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.05678136 | [0.05445552, 0.05911114] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01786777 | [0.01652187, 0.01929762] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004959521 | [0.004365754, 0.005569966] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002195732 | [0.001657497, 0.002719309] | 0.8421053 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001224601 | [0.0009786779, 0.001477971] | 0.8596491 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00101279 | [0.0003693115, 0.001552989] | 0.7017544 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006641883 | [0.00569133, 0.00758588] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.006348459 | [0.005141037, 0.007474522] | 0.9298246 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0001928116 | [-0.0003498196, 0.000683226] | 0.5964912 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002286427 | [0.001879577, 0.002681723] | 0.9298246 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003760091 | [0.002773099, 0.004756755] | 0.8596491 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004295316 | [0.003658372, 0.004930261] | 0.9649123 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0006750406 | [9.772727e-05, 0.001202428] | 0.6842105 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005897007 | [0.005155863, 0.006588406] | 0.9649123 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002036174 | [0.001556982, 0.002482985] | 0.8421053 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004862159 | [0.00424529, 0.005431028] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001338996 | [0.0007230712, 0.001957043] | 0.7719298 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003891785 | [0.003242554, 0.004530162] | 0.9473684 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002275763 | [0.001673661, 0.002786208] | 0.9122807 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
