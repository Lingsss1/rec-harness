# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/q8-entry/paired.tsv`
- Rows: `130116`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_interaction` | `drop_interaction` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | -4.827913e-08 | [-4.201449e-07, 4.188048e-07] | 0.5347536 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_interaction` | `drop_interaction` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | -2.675204e-06 | [-1.251413e-05, 1.946087e-06] | 0.5532602 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_interaction` | `drop_interaction` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 6.743383e-07 | [-7.004406e-07, 1.778618e-06] | 0.5315257 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_interaction` | `drop_interaction` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | 2.707013e-06 | [-1.755214e-06, 7.779822e-06] | 0.4729933 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_residual` | `drop_residual` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | -6.448507e-07 | [-1.178524e-06, -9.412671e-08] | 0.3393587 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_residual` | `drop_residual` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 3.742781e-06 | [-2.152305e-06, 1.40839e-05] | 0.3294599 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_residual` | `drop_residual` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | -6.276184e-07 | [-2.062589e-06, 1.289833e-06] | 0.3068646 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_residual` | `drop_residual` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | -2.914486e-06 | [-7.598639e-06, 1.482494e-06] | 0.5201205 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_retrieval` | `drop_retrieval` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | -6.574839e-07 | [-1.13442e-06, -1.681747e-07] | 0.4986012 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_retrieval` | `drop_retrieval` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | -9.586936e-07 | [-6.036382e-06, 6.97452e-06] | 0.506133 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_retrieval` | `drop_retrieval` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | -9.200657e-07 | [-2.457857e-06, 1.266148e-06] | 0.5082849 |
| `query_q8_entry-s20262027-20260908T171234804182Z_drop_retrieval` | `drop_retrieval` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | -1.686887e-06 | [-6.644895e-06, 3.608276e-06] | 0.512804 |
| `query_q8_entry-s20262027-20260908T171234804182Z_identity` | `identity` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q8_entry-s20262027-20260908T171234804182Z_identity` | `identity` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q8_entry-s20262027-20260908T171234804182Z_identity` | `identity` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q8_entry-s20262027-20260908T171234804182Z_identity` | `identity` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `query_q8_entry-s20262027-20260908T171234804182Z_retrieval_only` | `retrieval_only` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | -6.94347e-07 | [-1.263359e-06, -8.785159e-08] | 0.4064988 |
| `query_q8_entry-s20262027-20260908T171234804182Z_retrieval_only` | `retrieval_only` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 1.049432e-06 | [-9.337865e-07, 3.707936e-06] | 0.3748655 |
| `query_q8_entry-s20262027-20260908T171234804182Z_retrieval_only` | `retrieval_only` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 2.689431e-08 | [-7.214701e-07, 7.143623e-07] | 0.3393587 |
| `query_q8_entry-s20262027-20260908T171234804182Z_retrieval_only` | `retrieval_only` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | -2.081381e-07 | [-4.779918e-06, 4.01352e-06] | 0.4934366 |
| `query_q8_entry-s20262027-20260908T171234804182Z_shuffle` | `shuffle` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | -4.083121e-07 | [-1.231991e-06, 2.893964e-07] | 0.5345384 |
| `query_q8_entry-s20262027-20260908T171234804182Z_shuffle` | `shuffle` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 8.645621e-06 | [-2.478978e-06, 3.622255e-05] | 0.5257155 |
| `query_q8_entry-s20262027-20260908T171234804182Z_shuffle` | `shuffle` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 1.204313e-06 | [-3.746964e-07, 3.54262e-06] | 0.5179686 |
| `query_q8_entry-s20262027-20260908T171234804182Z_shuffle` | `shuffle` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | -1.082785e-06 | [-1.050775e-05, 7.549612e-06] | 0.4923607 |
| `query_q8_entry-s20262027-20260908T171234804182Z_zero` | `zero` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | -1.350796e-06 | [-2.067252e-06, -6.818282e-07] | 0.406714 |
| `query_q8_entry-s20262027-20260908T171234804182Z_zero` | `zero` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 7.425863e-08 | [-5.927242e-06, 1.042645e-05] | 0.373144 |
| `query_q8_entry-s20262027-20260908T171234804182Z_zero` | `zero` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | -8.921277e-07 | [-2.553925e-06, 1.585655e-06] | 0.3660426 |
| `query_q8_entry-s20262027-20260908T171234804182Z_zero` | `zero` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | -1.902512e-06 | [-7.951506e-06, 3.866797e-06] | 0.5085001 |

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
