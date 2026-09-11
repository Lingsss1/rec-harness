# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round15/r15_w1536_history/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.004170231 | [0.003521703, 0.004901968] | 0.9298246 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.01706874 | [0.01553992, 0.01859783] | 1 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.01470274 | [0.0134716, 0.01595937] | 1 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003630053 | [0.002974125, 0.004287085] | 0.9649123 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.0008177386 | [0.0003712438, 0.001255613] | 0.7017544 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005331402 | [0.004576071, 0.006077286] | 0.9473684 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002187377 | [0.001556123, 0.002722355] | 0.7894737 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003572666 | [0.002874236, 0.00432171] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001040469 | [0.0005958604, 0.001511796] | 0.7368421 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0002446588 | [-0.0001360352, 0.0005882634] | 0.5789474 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.005845041 | [0.005035406, 0.006624124] | 0.9649123 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.00219795 | [0.001498389, 0.002838479] | 0.8245614 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002479361 | [0.001927156, 0.003060547] | 0.8947368 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002915434 | [0.002046566, 0.003742224] | 0.8245614 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002832474 | [0.002301047, 0.003344926] | 0.8947368 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001259344 | [0.0006916093, 0.001800372] | 0.7719298 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003020193 | [0.002541263, 0.003529066] | 0.9473684 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.005798361 | [0.00482817, 0.006858806] | 0.9122807 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002666419 | [0.002040424, 0.003333564] | 0.877193 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001545647 | [0.0008348798, 0.002224709] | 0.754386 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
