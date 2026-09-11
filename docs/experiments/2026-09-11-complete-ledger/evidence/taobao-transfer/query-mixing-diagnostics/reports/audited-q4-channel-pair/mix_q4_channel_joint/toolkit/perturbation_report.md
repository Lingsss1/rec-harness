# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_joint/export/paired.tsv`
- Rows: `55764`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_identity` | `identity` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_identity` | `identity` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_identity` | `identity` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_identity` | `identity` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_shuffle` | `shuffle` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 5.783149e-08 | [-2.355855e-07, 3.74504e-07] | 0.5121584 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_shuffle` | `shuffle` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 1.116612e-06 | [-1.028702e-06, 3.616513e-06] | 0.4738541 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_shuffle` | `shuffle` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | -3.402158e-06 | [-5.685329e-06, -1.30593e-06] | 0.4914999 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_shuffle` | `shuffle` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 7.168291e-06 | [-3.744813e-06, 1.906688e-05] | 0.4880568 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_zero` | `zero` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | -4.477918e-07 | [-8.410828e-07, 4.029908e-08] | 0.5175382 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_zero` | `zero` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | -3.158469e-07 | [-1.713943e-06, 1.495008e-06] | 0.423499 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_zero` | `zero` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | -3.005876e-06 | [-5.66988e-06, -6.058645e-07] | 0.6257801 |
| `mix_q4_channel_joint-s20262027-20260909T035059317050Z_zero` | `zero` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 1.020477e-05 | [-5.030837e-06, 2.43578e-05] | 0.5007532 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
