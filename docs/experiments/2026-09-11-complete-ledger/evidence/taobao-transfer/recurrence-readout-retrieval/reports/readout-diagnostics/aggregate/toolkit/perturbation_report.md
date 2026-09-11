# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/readout-diagnostics/aggregate/paired.tsv`
- Rows: `74352`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `role_q4_read_task` | `mean` | 4 | `20262027` | `buy_weighted_bce` | 4647 | 3.338265e-07 | n/a | 0.6115774 |
| `role_q4_read_task` | `mean` | 4 | `20262027` | `cart_weighted_bce` | 4647 | -2.642426e-07 | n/a | 0.4340435 |
| `role_q4_read_task` | `mean` | 4 | `20262027` | `fav_weighted_bce` | 4647 | -1.047204e-07 | n/a | 0.7443512 |
| `role_q4_read_task` | `mean` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | 9.272548e-06 | n/a | 0.4945126 |
| `role_q4_read_task` | `zero` | 4 | `20262027` | `buy_weighted_bce` | 4647 | 4.797014e-06 | n/a | 0.6113622 |
| `role_q4_read_task` | `zero` | 4 | `20262027` | `cart_weighted_bce` | 4647 | 3.976146e-05 | n/a | 0.5655261 |
| `role_q4_read_task` | `zero` | 4 | `20262027` | `fav_weighted_bce` | 4647 | 4.722955e-06 | n/a | 0.255864 |
| `role_q4_read_task` | `zero` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | -0.0001032322 | n/a | 0.5037659 |
| `role_q8_read_task` | `mean` | 4 | `20262027` | `buy_weighted_bce` | 4647 | -4.995026e-08 | n/a | 0.4191952 |
| `role_q8_read_task` | `mean` | 4 | `20262027` | `cart_weighted_bce` | 4647 | -6.447224e-06 | n/a | 0.5196901 |
| `role_q8_read_task` | `mean` | 4 | `20262027` | `fav_weighted_bce` | 4647 | -7.516245e-07 | n/a | 0.5192597 |
| `role_q8_read_task` | `mean` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | -3.289511e-06 | n/a | 0.4958037 |
| `role_q8_read_task` | `zero` | 4 | `20262027` | `buy_weighted_bce` | 4647 | 2.823004e-06 | n/a | 0.5797289 |
| `role_q8_read_task` | `zero` | 4 | `20262027` | `cart_weighted_bce` | 4647 | 4.680072e-05 | n/a | 0.4796643 |
| `role_q8_read_task` | `zero` | 4 | `20262027` | `fav_weighted_bce` | 4647 | -9.228902e-06 | n/a | 0.5190445 |
| `role_q8_read_task` | `zero` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | -3.823542e-05 | n/a | 0.4951582 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
