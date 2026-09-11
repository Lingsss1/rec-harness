# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round7/r7_small_midnorm/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.01042675 | [0.008781387, 0.01185916] | 0.9298246 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.07340005 | [0.07111423, 0.07556997] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01693326 | [0.01511073, 0.01884561] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004250288 | [0.003572408, 0.00497064] | 0.9824561 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0003590416 | [-0.000116768, 0.0008373199] | 0.6315789 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002676722 | [0.002265543, 0.003085421] | 0.9649123 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0004240329 | [-0.0002420477, 0.001027743] | 0.6842105 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003671405 | [0.002943431, 0.004508746] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002443628 | [0.001547505, 0.003317133] | 0.7368421 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0002743215 | [-0.0001119194, 0.0006357209] | 0.6666667 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004046954 | [0.003334472, 0.004763245] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002000259 | [0.001296579, 0.002715488] | 0.7719298 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004643149 | [0.003903337, 0.005387819] | 0.9649123 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002461675 | [0.001574118, 0.003312818] | 0.8245614 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005905991 | [0.005132148, 0.006721326] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001165439 | [0.0006895217, 0.001596673] | 0.7719298 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002345271 | [0.001805166, 0.002826576] | 0.8947368 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | -0.0002221975 | [-0.0005393264, 9.465529e-05] | 0.4561404 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003511221 | [0.002765417, 0.00434285] | 0.9122807 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001425696 | [0.0006272688, 0.002181021] | 0.6842105 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
