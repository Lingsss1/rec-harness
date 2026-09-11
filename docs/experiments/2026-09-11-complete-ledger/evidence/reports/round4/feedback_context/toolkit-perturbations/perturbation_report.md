# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round4/feedback_context/perturbations.tsv`
- Rows: `1140`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.004942468 | [0.003991109, 0.005886348] | 0.9298246 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.002548787 | [0.001885184, 0.003211925] | 0.8245614 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003582804 | [0.00303056, 0.00411927] | 0.9649123 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0003612983 | [-0.0001436395, 0.0008237932] | 0.6666667 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004357355 | [0.003724379, 0.005061927] | 0.9824561 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008834451 | [0.0004611243, 0.001356795] | 0.7368421 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004328463 | [0.003767595, 0.004944396] | 0.9649123 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003684392 | [0.002640139, 0.004574439] | 0.8245614 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 7.4001e-05 | [-0.0001553143, 0.000314403] | 0.5614035 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00483651 | [0.004117613, 0.005482584] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0007577842 | [0.0002530289, 0.001176312] | 0.6842105 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004999504 | [0.004352315, 0.005680434] | 0.9824561 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002505504 | [0.001616628, 0.00341208] | 0.7719298 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002842507 | [0.002296494, 0.00337425] | 0.8947368 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00372411 | [0.002797461, 0.004650698] | 0.9298246 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006349214 | [0.005508929, 0.007071248] | 0.9824561 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001984713 | [0.001271589, 0.002621674] | 0.7894737 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.007266347 | [0.006364259, 0.008001891] | 0.9824561 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0007243752 | [0.0003718543, 0.001048303] | 0.7894737 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
