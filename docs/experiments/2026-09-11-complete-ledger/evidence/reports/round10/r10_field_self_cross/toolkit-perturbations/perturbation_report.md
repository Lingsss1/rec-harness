# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round10/r10_field_self_cross/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.001085859 | [0.0006110766, 0.001624228] | 0.754386 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.03163903 | [0.0300986, 0.03319346] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.00995869 | [0.008992079, 0.01113915] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00417408 | [0.003294311, 0.00491799] | 0.9122807 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00101883 | [0.0004616458, 0.001716519] | 0.6842105 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003996222 | [0.00333143, 0.004722196] | 0.9298246 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001632776 | [0.001051015, 0.002140944] | 0.8421053 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003521571 | [0.00286793, 0.004239561] | 0.9298246 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001970201 | [0.001363176, 0.002572022] | 0.8421053 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0001366303 | [-0.0001159549, 0.0003603811] | 0.5789474 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002509976 | [0.002028525, 0.003011971] | 0.8947368 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001034046 | [0.0005888894, 0.001496602] | 0.7719298 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001670136 | [0.001295325, 0.002041265] | 0.8947368 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002819731 | [0.001954688, 0.003637332] | 0.8245614 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005104651 | [0.004273795, 0.005948227] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002137255 | [0.0014971, 0.002729539] | 0.8596491 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003037607 | [0.002479127, 0.003607197] | 0.9473684 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004944788 | [0.004002577, 0.005861517] | 0.9122807 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002241558 | [0.001791793, 0.002745522] | 0.9473684 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001144107 | [0.0006719076, 0.001597404] | 0.7894737 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
