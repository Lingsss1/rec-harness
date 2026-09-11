# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/reports/round16/r16_base_mixed1232/perturbations.tsv`
- Rows: `1197`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `all_queries` | `queries` | 256 | `20262027` | `logloss` | 57 | 0 | [0, 0] | 0 |
| `extra_attention_off` | `off` | 256 | `20262027` | `logloss` | 57 | 0.0007223442 | [0.0004477987, 0.001006718] | 0.7894737 |
| `inference_placement_steps1` | `steps1` | 256 | `20262027` | `logloss` | 57 | 0.0008703146 | [0.0004354337, 0.001300742] | 0.7192982 |
| `inference_placement_steps3` | `steps3` | 256 | `20262027` | `logloss` | 57 | 0.0004164425 | [0.0001837823, 0.0006983618] | 0.7017544 |
| `q0_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.003201672 | [0.002618226, 0.003797659] | 0.9298246 |
| `q0_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001542235 | [0.001032372, 0.002038839] | 0.8421053 |
| `q1_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002295957 | [0.001806972, 0.002899493] | 0.9298246 |
| `q1_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002249831 | [0.001604234, 0.002820516] | 0.8421053 |
| `q2_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.00317688 | [0.002592031, 0.003769087] | 0.9473684 |
| `q2_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002619309 | [0.002054849, 0.003192938] | 0.9122807 |
| `q3_half` | `half` | 256 | `20262027` | `logloss` | 57 | 0.0003514355 | [4.77801e-05, 0.0006466556] | 0.6842105 |
| `q3_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.001417124 | [0.001082544, 0.001749945] | 0.8421053 |
| `q3_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001534283 | [0.001018942, 0.002073092] | 0.8596491 |
| `q4_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002097024 | [0.00161854, 0.002685525] | 0.877193 |
| `q4_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.002011464 | [0.001443825, 0.002521346] | 0.877193 |
| `q5_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.004167654 | [0.003413502, 0.005064719] | 0.9473684 |
| `q5_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003226645 | [0.002588734, 0.003844878] | 0.9122807 |
| `q6_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002105894 | [0.001661166, 0.002593853] | 0.877193 |
| `q6_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.003199227 | [0.002522812, 0.00385289] | 0.9122807 |
| `q7_shuffle` | `shuffle` | 256 | `20262027` | `logloss` | 57 | 0.002293521 | [0.00177959, 0.00285274] | 0.8947368 |
| `q7_zero` | `zero` | 256 | `20262027` | `logloss` | 57 | 0.001215384 | [0.0007293807, 0.001695399] | 0.8245614 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
