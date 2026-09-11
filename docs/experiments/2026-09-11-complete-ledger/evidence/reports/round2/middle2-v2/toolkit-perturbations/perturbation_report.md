# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round2/middle2-v2/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | -0.003106859 | [-0.003836665, -0.002289677] | 0.1403509 |
| `inference_loops1` | `loops1` | 256 | `20262027` | `logloss` | 57 | 0.002253227 | [0.001811554, 0.002676946] | 0.9473684 |
| `inference_loops3` | `loops3` | 256 | `20262027` | `logloss` | 57 | 0.005643936 | [0.005081929, 0.006237669] | 1 |
| `inference_loops4` | `loops4` | 256 | `20262027` | `logloss` | 57 | 0.01964372 | [0.0184858, 0.02078277] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001845692 | [0.001565307, 0.002139665] | 0.9473684 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.01322959 | [0.01218326, 0.01429096] | 1 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.008762268 | [0.00797115, 0.009639158] | 1 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.009373445 | [0.008776266, 0.009961733] | 1 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006640069 | [0.005816382, 0.007472251] | 0.9824561 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005022253 | [0.004635574, 0.005392229] | 1 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | -0.002757559 | [-0.003166536, -0.00229732] | 0.05263158 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.006979265 | [0.00629608, 0.007740732] | 0.9649123 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.008750557 | [0.008134943, 0.009298733] | 1 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.007322952 | [0.006530943, 0.008130747] | 1 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.01408652 | [0.01305583, 0.01503596] | 1 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001134797 | [0.0008753518, 0.001392328] | 0.9122807 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.01223762 | [0.01115367, 0.01340969] | 1 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005186393 | [0.004563715, 0.005826057] | 0.9824561 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005775882 | [0.005230908, 0.00633895] | 1 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
