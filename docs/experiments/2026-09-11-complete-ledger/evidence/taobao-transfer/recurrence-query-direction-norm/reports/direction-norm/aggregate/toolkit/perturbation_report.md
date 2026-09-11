# Paired Perturbation Diagnostic

- Input: `workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/reports/direction-norm/aggregate/paired.tsv`
- Rows: `185880`

## Per-seed results

| Candidate | Intervention | Width | Seed | Metric | n | Mean delta | 95% block CI | Positive fraction |
|---|---|---:|---|---|---:|---:|---:|---:|
| `ctx_q4` | `both_minus_native` | 4 | `20262027` | `buy_weighted_bce` | 4647 | 1.431166e-05 | n/a | 0.575425 |
| `ctx_q4` | `both_minus_native` | 4 | `20262027` | `cart_weighted_bce` | 4647 | 2.63857e-05 | n/a | 0.5896277 |
| `ctx_q4` | `both_minus_native` | 4 | `20262027` | `fav_weighted_bce` | 4647 | 0.0001270243 | n/a | 0.5205509 |
| `ctx_q4` | `both_minus_native` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | 0.0001640562 | n/a | 0.4669679 |
| `ctx_q4` | `direction_minus_native` | 4 | `20262027` | `buy_weighted_bce` | 4647 | 1.420982e-05 | n/a | 0.575425 |
| `ctx_q4` | `direction_minus_native` | 4 | `20262027` | `cart_weighted_bce` | 4647 | 2.714134e-05 | n/a | 0.5885518 |
| `ctx_q4` | `direction_minus_native` | 4 | `20262027` | `fav_weighted_bce` | 4647 | 0.000126655 | n/a | 0.5205509 |
| `ctx_q4` | `direction_minus_native` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | 0.0001641189 | n/a | 0.4667527 |
| `ctx_q4` | `direction_minus_norm` | 4 | `20262027` | `buy_weighted_bce` | 4647 | 1.409108e-05 | n/a | 0.5747794 |
| `ctx_q4` | `direction_minus_norm` | 4 | `20262027` | `cart_weighted_bce` | 4647 | 2.781844e-05 | n/a | 0.5887669 |
| `ctx_q4` | `direction_minus_norm` | 4 | `20262027` | `fav_weighted_bce` | 4647 | 0.0001260702 | n/a | 0.5220572 |
| `ctx_q4` | `direction_minus_norm` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | 0.0001640548 | n/a | 0.4673983 |
| `ctx_q4` | `interaction` | 4 | `20262027` | `buy_weighted_bce` | 4647 | -1.689486e-08 | n/a | 0.5119432 |
| `ctx_q4` | `interaction` | 4 | `20262027` | `cart_weighted_bce` | 4647 | -7.853737e-08 | n/a | 0.4783731 |
| `ctx_q4` | `interaction` | 4 | `20262027` | `fav_weighted_bce` | 4647 | -2.155252e-07 | n/a | 0.5100065 |
| `ctx_q4` | `interaction` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | -1.26905e-07 | n/a | 0.4803099 |
| `ctx_q4` | `norm_minus_native` | 4 | `20262027` | `buy_weighted_bce` | 4647 | 1.18736e-07 | n/a | 0.5493867 |
| `ctx_q4` | `norm_minus_native` | 4 | `20262027` | `cart_weighted_bce` | 4647 | -6.770981e-07 | n/a | 0.5900581 |
| `ctx_q4` | `norm_minus_native` | 4 | `20262027` | `fav_weighted_bce` | 4647 | 5.848235e-07 | n/a | 0.4811706 |
| `ctx_q4` | `norm_minus_native` | 4 | `20262027` | `is_click_weighted_bce` | 4647 | 6.415345e-08 | n/a | 0.5029051 |
| `explore_dual_no_feedback` | `both_minus_native` | 8 | `20262027` | `buy_weighted_bce` | 4647 | -1.506872e-05 | n/a | 0.4738541 |
| `explore_dual_no_feedback` | `both_minus_native` | 8 | `20262027` | `cart_weighted_bce` | 4647 | 0.0003003949 | n/a | 0.4484614 |
| `explore_dual_no_feedback` | `both_minus_native` | 8 | `20262027` | `fav_weighted_bce` | 4647 | -5.290292e-05 | n/a | 0.4473854 |
| `explore_dual_no_feedback` | `both_minus_native` | 8 | `20262027` | `is_click_weighted_bce` | 4647 | 0.0003333303 | n/a | 0.4878416 |
| `explore_dual_no_feedback` | `direction_minus_native` | 8 | `20262027` | `buy_weighted_bce` | 4647 | -1.486655e-05 | n/a | 0.4751453 |
| `explore_dual_no_feedback` | `direction_minus_native` | 8 | `20262027` | `cart_weighted_bce` | 4647 | 0.0003013317 | n/a | 0.4488918 |
| `explore_dual_no_feedback` | `direction_minus_native` | 8 | `20262027` | `fav_weighted_bce` | 4647 | -5.308927e-05 | n/a | 0.448031 |
| `explore_dual_no_feedback` | `direction_minus_native` | 8 | `20262027` | `is_click_weighted_bce` | 4647 | 0.0003335225 | n/a | 0.4891328 |
| `explore_dual_no_feedback` | `direction_minus_norm` | 8 | `20262027` | `buy_weighted_bce` | 4647 | -1.458214e-05 | n/a | 0.4755756 |
| `explore_dual_no_feedback` | `direction_minus_norm` | 8 | `20262027` | `cart_weighted_bce` | 4647 | 0.0003021566 | n/a | 0.4484614 |
| `explore_dual_no_feedback` | `direction_minus_norm` | 8 | `20262027` | `fav_weighted_bce` | 4647 | -5.338433e-05 | n/a | 0.4467398 |
| `explore_dual_no_feedback` | `direction_minus_norm` | 8 | `20262027` | `is_click_weighted_bce` | 4647 | 0.0003332819 | n/a | 0.4876264 |
| `explore_dual_no_feedback` | `interaction` | 8 | `20262027` | `buy_weighted_bce` | 4647 | 8.223151e-08 | n/a | 0.4878416 |
| `explore_dual_no_feedback` | `interaction` | 8 | `20262027` | `cart_weighted_bce` | 4647 | -1.119351e-07 | n/a | 0.4869808 |
| `explore_dual_no_feedback` | `interaction` | 8 | `20262027` | `fav_weighted_bce` | 4647 | -1.087116e-07 | n/a | 0.4630945 |
| `explore_dual_no_feedback` | `interaction` | 8 | `20262027` | `is_click_weighted_bce` | 4647 | -4.327538e-07 | n/a | 0.4940822 |
| `explore_dual_no_feedback` | `norm_minus_native` | 8 | `20262027` | `buy_weighted_bce` | 4647 | -2.844081e-07 | n/a | 0.5013988 |
| `explore_dual_no_feedback` | `norm_minus_native` | 8 | `20262027` | `cart_weighted_bce` | 4647 | -8.249192e-07 | n/a | 0.4990316 |
| `explore_dual_no_feedback` | `norm_minus_native` | 8 | `20262027` | `fav_weighted_bce` | 4647 | 2.950623e-07 | n/a | 0.5175382 |
| `explore_dual_no_feedback` | `norm_minus_native` | 8 | `20262027` | `is_click_weighted_bce` | 4647 | 2.405879e-07 | n/a | 0.5227028 |

## Warnings

- both_minus_native/buy_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- both_minus_native/cart_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- both_minus_native/fav_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- both_minus_native/is_click_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_native/buy_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_native/cart_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_native/fav_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_native/is_click_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_norm/buy_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_norm/cart_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_norm/fav_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- direction_minus_norm/is_click_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- interaction/buy_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- interaction/cart_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- interaction/fav_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- interaction/is_click_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- norm_minus_native/buy_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- norm_minus_native/cart_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- norm_minus_native/fav_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.
- norm_minus_native/is_click_weighted_bce contains multiple widths [4, 8]; do not compare candidates as equal-cost controls.

## Caveats

- All deltas are computed as perturbed minus original.
- Paired effects measure current-model reliance, not guaranteed retraining benefit.
- Near-zero Shuffle should be checked with Zero and raw representation variance.
- Use independent seeds and unchanged validation protocol before durable conclusions.
