# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round18/r18_field512_norm_phase132/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.006088541 | [0.00521656, 0.006944103] | 0.9824561 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.004050342 | [0.002746358, 0.005225397] | 0.877193 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.006973942 | [0.005891522, 0.008277268] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002617341 | [0.002084443, 0.003151531] | 0.8947368 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0006144496 | [5.135917e-05, 0.001128943] | 0.6666667 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003984245 | [0.003372251, 0.004585539] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001213387 | [0.000849853, 0.001581172] | 0.8070175 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001619664 | [0.001188197, 0.002060148] | 0.8947368 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008874483 | [0.0004151009, 0.001342895] | 0.7192982 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.00044177 | [3.610282e-05, 0.0008127721] | 0.6315789 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004422921 | [0.003710139, 0.005188715] | 0.9824561 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00271341 | [0.00203833, 0.00341972] | 0.8947368 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00243968 | [0.001980358, 0.002941226] | 0.9122807 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002640504 | [0.00184167, 0.003354159] | 0.8596491 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00329377 | [0.002745603, 0.003886888] | 0.9824561 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0006899826 | [0.0001909692, 0.001144571] | 0.7017544 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004026667 | [0.003429777, 0.00469691] | 0.9649123 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00203401 | [0.001424798, 0.002701291] | 0.8245614 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002508687 | [0.00207518, 0.002955604] | 0.9473684 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001126529 | [0.0005620113, 0.001696994] | 0.754386 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
