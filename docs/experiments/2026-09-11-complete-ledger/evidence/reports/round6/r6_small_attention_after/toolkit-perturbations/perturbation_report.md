# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round6/r6_small_attention_after/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.01649377 | [0.0146701, 0.0180714] | 1 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.09887617 | [0.09622548, 0.101311] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.02626663 | [0.02391218, 0.02871302] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004205455 | [0.003556384, 0.004819235] | 0.9649123 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.004544435 | [0.003424686, 0.005548808] | 0.8947368 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003088262 | [0.002562206, 0.003616217] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008292967 | [0.0001814427, 0.001432209] | 0.6491228 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004194147 | [0.003476933, 0.00505312] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003983244 | [0.002992638, 0.004976107] | 0.877193 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0001911748 | [-0.000154899, 0.0005196915] | 0.5964912 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003895222 | [0.00328223, 0.004506446] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001444612 | [0.0007759874, 0.002095269] | 0.7894737 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003236875 | [0.002644258, 0.003871305] | 0.9298246 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002886609 | [0.001929708, 0.003758809] | 0.8421053 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002436307 | [0.002004551, 0.002896125] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00020507 | [-0.0002644412, 0.0006425926] | 0.4912281 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004679827 | [0.004019063, 0.005298686] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0002814212 | [-6.113397e-06, 0.0005724832] | 0.5438596 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004883821 | [0.004154347, 0.005640385] | 0.9824561 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001213975 | [0.0006943922, 0.001670686] | 0.8245614 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
