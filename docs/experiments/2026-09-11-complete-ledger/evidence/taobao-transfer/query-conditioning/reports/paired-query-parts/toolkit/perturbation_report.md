# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/taobao-transfer/query-conditioning/reports/paired-query-parts/paired.tsv`
- Rows: `185880`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `ctx_q4_entry_shuffle` | `entry_shuffle` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 0.0001032237 | [7.508807e-05, 0.0001347873] | 0.5917796 |
| `ctx_q4_entry_shuffle` | `entry_shuffle` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | -7.900219e-05 | [-0.0009040931, 0.0003361899] | 0.5672477 |
| `ctx_q4_entry_shuffle` | `entry_shuffle` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 0.0001022577 | [4.203167e-05, 0.0001626195] | 0.5470196 |
| `ctx_q4_entry_shuffle` | `entry_shuffle` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 6.995296e-05 | [-0.0007760467, 0.0007804215] | 0.4904239 |
| `ctx_q4_entry_zero` | `entry_zero` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 1.523384e-05 | [7.50784e-06, 2.210563e-05] | 0.6145901 |
| `ctx_q4_entry_zero` | `entry_zero` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | -7.419958e-05 | [-0.0003058864, 5.960913e-05] | 0.5020443 |
| `ctx_q4_entry_zero` | `entry_zero` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 6.65257e-05 | [3.319989e-05, 9.929339e-05] | 0.3638907 |
| `ctx_q4_entry_zero` | `entry_zero` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | -0.0001972108 | [-0.000677649, 0.0002371846] | 0.510652 |
| `ctx_q4_identity` | `identity` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `ctx_q4_identity` | `identity` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `ctx_q4_identity` | `identity` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `ctx_q4_identity` | `identity` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `ctx_q4_work_shuffle` | `work_shuffle` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 0.0002368065 | [0.0001797236, 0.0002927565] | 0.5739187 |
| `ctx_q4_work_shuffle` | `work_shuffle` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 0.0001845592 | [-0.000835307, 0.0007853725] | 0.5483107 |
| `ctx_q4_work_shuffle` | `work_shuffle` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 0.0001900836 | [4.213911e-05, 0.0003311444] | 0.5523994 |
| `ctx_q4_work_shuffle` | `work_shuffle` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 0.001469444 | [6.219344e-05, 0.002939787] | 0.4906391 |
| `ctx_q4_work_zero` | `work_zero` | 1536 | `20262027` | `buy_weighted_bce` | 4647 | 1.620306e-05 | [2.357613e-06, 3.01496e-05] | 0.436841 |
| `ctx_q4_work_zero` | `work_zero` | 1536 | `20262027` | `cart_weighted_bce` | 4647 | 0.0001546555 | [-3.066117e-05, 0.0004841365] | 0.5069938 |
| `ctx_q4_work_zero` | `work_zero` | 1536 | `20262027` | `fav_weighted_bce` | 4647 | 8.059903e-05 | [-1.305347e-05, 0.0001874853] | 0.7120723 |
| `ctx_q4_work_zero` | `work_zero` | 1536 | `20262027` | `is_click_weighted_bce` | 4647 | 0.002028667 | [0.001063873, 0.00309704] | 0.4906391 |
| `explore_dual_no_feedback_entry_shuffle` | `entry_shuffle` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | 0.000183947 | [0.0001356681, 0.0002379762] | 0.5907037 |
| `explore_dual_no_feedback_entry_shuffle` | `entry_shuffle` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 0.0002469817 | [-9.189675e-05, 0.0005046005] | 0.5491715 |
| `explore_dual_no_feedback_entry_shuffle` | `entry_shuffle` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 0.0003412391 | [0.0001407856, 0.0004808661] | 0.5700452 |
| `explore_dual_no_feedback_entry_shuffle` | `entry_shuffle` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | -8.75144e-06 | [-0.0006529854, 0.0006773978] | 0.4693351 |
| `explore_dual_no_feedback_entry_zero` | `entry_zero` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | 7.096336e-05 | [4.662615e-05, 0.0001016899] | 0.528513 |
| `explore_dual_no_feedback_entry_zero` | `entry_zero` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | -9.861853e-05 | [-0.0007958505, 0.0003097249] | 0.4325371 |
| `explore_dual_no_feedback_entry_zero` | `entry_zero` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 0.0001108414 | [2.795748e-05, 0.0001706312] | 0.401119 |
| `explore_dual_no_feedback_entry_zero` | `entry_zero` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | -0.0001771142 | [-0.0004736143, 0.0001056542] | 0.5205509 |
| `explore_dual_no_feedback_identity` | `identity` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `explore_dual_no_feedback_identity` | `identity` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `explore_dual_no_feedback_identity` | `identity` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `explore_dual_no_feedback_identity` | `identity` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | 0 | [0, 0] | 0 |
| `explore_dual_no_feedback_work_shuffle` | `work_shuffle` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | 0.0007220228 | [0.0006023943, 0.000847059] | 0.5898429 |
| `explore_dual_no_feedback_work_shuffle` | `work_shuffle` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 0.00086393 | [-0.0001652484, 0.001708198] | 0.5569184 |
| `explore_dual_no_feedback_work_shuffle` | `work_shuffle` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 0.001023818 | [0.0005848696, 0.001359724] | 0.586615 |
| `explore_dual_no_feedback_work_shuffle` | `work_shuffle` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | 0.001442134 | [0.0001673729, 0.002839308] | 0.4792339 |
| `explore_dual_no_feedback_work_zero` | `work_zero` | 3072 | `20262027` | `buy_weighted_bce` | 4647 | 0.0001141027 | [8.973672e-05, 0.0001424242] | 0.5724123 |
| `explore_dual_no_feedback_work_zero` | `work_zero` | 3072 | `20262027` | `cart_weighted_bce` | 4647 | 0.0001568288 | [-0.0001802803, 0.0004380832] | 0.6251345 |
| `explore_dual_no_feedback_work_zero` | `work_zero` | 3072 | `20262027` | `fav_weighted_bce` | 4647 | 0.0001065201 | [-7.961267e-05, 0.0002223364] | 0.6825909 |
| `explore_dual_no_feedback_work_zero` | `work_zero` | 3072 | `20262027` | `is_click_weighted_bce` | 4647 | 0.0009902153 | [0.0004289537, 0.001604195] | 0.4549172 |

## Warnings

- entry_shuffle/buy_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- entry_shuffle/cart_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- entry_shuffle/fav_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- entry_shuffle/is_click_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- entry_zero/buy_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- entry_zero/cart_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- entry_zero/fav_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- entry_zero/is_click_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- identity/buy_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- identity/cart_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- identity/fav_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- identity/is_click_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_shuffle/buy_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_shuffle/cart_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_shuffle/fav_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_shuffle/is_click_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_zero/buy_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_zero/cart_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_zero/fav_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.
- work_zero/is_click_weighted_bce contains multiple widths [1536, 3072]; do not compare candidates as equal-cost controls.

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
