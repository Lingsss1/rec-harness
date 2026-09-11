# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/q4-work/paired.tsv`
- Rows: `130116`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `query_q4_work-s20262027-20260908T171234834290Z_drop_interaction` | `drop_interaction` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | -2.108014e-07 | [-5.17995e-07, 1.232141e-07] | 0.4157521 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_interaction` | `drop_interaction` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 8.082515e-07 | [-6.044064e-07, 1.964928e-06] | 0.5222724 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_interaction` | `drop_interaction` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | -1.585604e-07 | [-9.531073e-07, 6.520889e-07] | 0.6040456 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_interaction` | `drop_interaction` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 6.519956e-06 | [-7.140643e-06, 1.973007e-05] | 0.4701958 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_residual` | `drop_residual` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | -1.163205e-06 | [-1.58736e-06, -7.678324e-07] | 0.3212825 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_residual` | `drop_residual` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 7.316432e-07 | [-7.473157e-06, 1.536938e-05] | 0.2902948 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_residual` | `drop_residual` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | -4.302038e-06 | [-6.574564e-06, -1.485834e-06] | 0.377663 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_residual` | `drop_residual` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | -9.86639e-08 | [-6.907793e-06, 7.579001e-06] | 0.5022595 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_retrieval` | `drop_retrieval` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 5.177959e-07 | [2.243745e-07, 8.754998e-07] | 0.5599311 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_retrieval` | `drop_retrieval` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 8.198682e-08 | [-9.367817e-06, 5.190464e-06] | 0.5663869 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_retrieval` | `drop_retrieval` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 2.853727e-06 | [1.453687e-07, 4.883274e-06] | 0.5319561 |
| `query_q4_work-s20262027-20260908T171234834290Z_drop_retrieval` | `drop_retrieval` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 2.469851e-06 | [-7.422502e-06, 1.275731e-05] | 0.512804 |
| `query_q4_work-s20262027-20260908T171234834290Z_identity` | `identity` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q4_work-s20262027-20260908T171234834290Z_identity` | `identity` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q4_work-s20262027-20260908T171234834290Z_identity` | `identity` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q4_work-s20262027-20260908T171234834290Z_identity` | `identity` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q4_work-s20262027-20260908T171234834290Z_retrieval_only` | `retrieval_only` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | -1.375591e-06 | [-1.901845e-06, -7.74447e-07] | 0.3344093 |
| `query_q4_work-s20262027-20260908T171234834290Z_retrieval_only` | `retrieval_only` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 1.515958e-06 | [-6.338194e-06, 1.502863e-05] | 0.3146116 |
| `query_q4_work-s20262027-20260908T171234834290Z_retrieval_only` | `retrieval_only` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | -4.504872e-06 | [-6.840717e-06, -1.9386e-06] | 0.471487 |
| `query_q4_work-s20262027-20260908T171234834290Z_retrieval_only` | `retrieval_only` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 6.38707e-06 | [-7.408993e-06, 2.038987e-05] | 0.464816 |
| `query_q4_work-s20262027-20260908T171234834290Z_shuffle` | `shuffle` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | -4.067557e-07 | [-1.062382e-06, 3.173737e-07] | 0.5201205 |
| `query_q4_work-s20262027-20260908T171234834290Z_shuffle` | `shuffle` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | -3.229837e-06 | [-1.646212e-05, 2.331747e-06] | 0.4884872 |
| `query_q4_work-s20262027-20260908T171234834290Z_shuffle` | `shuffle` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | -2.686779e-06 | [-5.221529e-06, 1.534636e-07] | 0.4641704 |
| `query_q4_work-s20262027-20260908T171234834290Z_shuffle` | `shuffle` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 6.969598e-06 | [-2.918715e-06, 1.803141e-05] | 0.489348 |
| `query_q4_work-s20262027-20260908T171234834290Z_zero` | `zero` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | -8.595095e-07 | [-1.356576e-06, -2.911142e-07] | 0.3477512 |
| `query_q4_work-s20262027-20260908T171234834290Z_zero` | `zero` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 1.535614e-06 | [-1.728211e-06, 6.166513e-06] | 0.3544222 |
| `query_q4_work-s20262027-20260908T171234834290Z_zero` | `zero` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | -1.673206e-06 | [-3.625348e-06, 5.055878e-07] | 0.5026899 |
| `query_q4_work-s20262027-20260908T171234834290Z_zero` | `zero` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 8.716912e-06 | [-2.103382e-06, 1.982578e-05] | 0.4678287 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
