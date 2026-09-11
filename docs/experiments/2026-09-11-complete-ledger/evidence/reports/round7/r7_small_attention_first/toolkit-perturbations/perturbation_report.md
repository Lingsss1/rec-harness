# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round7/r7_small_attention_first/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.003123702 | [0.002350445, 0.003941984] | 0.8596491 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.04677416 | [0.04463655, 0.04905062] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01879967 | [0.0174745, 0.02015241] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006560205 | [0.005725654, 0.007308349] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001683155 | [0.001261256, 0.002162283] | 0.8245614 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001454347 | [0.001118915, 0.001825408] | 0.8245614 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0006566798 | [0.0003994617, 0.0008867694] | 0.754386 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008334238 | [0.00737455, 0.009267359] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003510011 | [0.002738475, 0.004307364] | 0.9122807 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.000108492 | [-0.000483494, 0.0002850431] | 0.5087719 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004182918 | [0.003607466, 0.004705389] | 0.9473684 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0009770205 | [0.000366268, 0.001612001] | 0.6666667 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002410797 | [0.002005436, 0.00280565] | 0.9298246 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 4.14119e-05 | [-0.0003835283, 0.0004510382] | 0.5614035 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002688935 | [0.002251231, 0.003167021] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002387833 | [0.001631689, 0.003112124] | 0.8421053 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003466232 | [0.002943878, 0.003977594] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004096318 | [0.00326813, 0.00497727] | 0.877193 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003994943 | [0.003303162, 0.004623035] | 0.9649123 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002004347 | [0.001400629, 0.002589475] | 0.7894737 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
