# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/audited-q4-channel-pair/mix_q4_channel_constant/export/paired.tsv`
- Rows: `55764`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_identity` | `identity` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_identity` | `identity` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_identity` | `identity` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_identity` | `identity` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_shuffle` | `shuffle` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_shuffle` | `shuffle` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_shuffle` | `shuffle` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_shuffle` | `shuffle` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_zero` | `zero` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | -8.171541e-09 | [-1.750565e-08, 5.546847e-10] | 0.3053583 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_zero` | `zero` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 2.050518e-07 | [-9.721513e-09, 6.071387e-07] | 0.4992468 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_zero` | `zero` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 1.272602e-08 | [-1.70148e-08, 3.98473e-08] | 0.3712072 |
| `mix_q4_channel_constant-s20262027-20260909T035109726499Z_zero` | `zero` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 2.028535e-07 | [5.164492e-09, 4.337339e-07] | 0.4861201 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
