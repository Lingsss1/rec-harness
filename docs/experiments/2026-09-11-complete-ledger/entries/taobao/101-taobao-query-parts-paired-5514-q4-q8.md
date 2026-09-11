## 101. taobao-query-parts-paired-5514-q4-q8

- 假设：Entry and first-work sample-specific information both contribute to second query
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T16:13:18.364275+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-parts-paired-5514-q4-q8",
  "family": "query_conditioning_diagnostic",
  "hypothesis": "Entry and first-work sample-specific information both contribute to second query",
  "change": "Same-weight second-query-only Shuffle/Zero and exact identity control; rec-harness paired analyzer",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "state": "completed_audited",
    "measurements": [
      {
        "candidate": "ctx_q4_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "buy_weighted_bce",
        "delta_mean": 0.00010322366882255501,
        "block_bootstrap_ci95": [
          7.508807025863665e-05,
          0.00013478731834673464
        ]
      },
      {
        "candidate": "ctx_q4_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "cart_weighted_bce",
        "delta_mean": -7.90021933219134e-05,
        "block_bootstrap_ci95": [
          -0.0009040931441227132,
          0.00033618987711953627
        ]
      },
      {
        "candidate": "ctx_q4_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "fav_weighted_bce",
        "delta_mean": 0.00010225767597654927,
        "block_bootstrap_ci95": [
          4.203166645680115e-05,
          0.00016261947524946632
        ]
      },
      {
        "candidate": "ctx_q4_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "is_click_weighted_bce",
        "delta_mean": 6.995295965039079e-05,
        "block_bootstrap_ci95": [
          -0.0007760466884264117,
          0.000780421539916113
        ]
      },
      {
        "candidate": "ctx_q4_entry_zero",
        "intervention": "entry_zero",
        "metric": "buy_weighted_bce",
        "delta_mean": 1.5233844713850135e-05,
        "block_bootstrap_ci95": [
          7.507840350677737e-06,
          2.2105633764975693e-05
        ]
      },
      {
        "candidate": "ctx_q4_entry_zero",
        "intervention": "entry_zero",
        "metric": "cart_weighted_bce",
        "delta_mean": -7.41995839757143e-05,
        "block_bootstrap_ci95": [
          -0.0003058863644775795,
          5.960912540771074e-05
        ]
      },
      {
        "candidate": "ctx_q4_entry_zero",
        "intervention": "entry_zero",
        "metric": "fav_weighted_bce",
        "delta_mean": 6.652570306871963e-05,
        "block_bootstrap_ci95": [
          3.319989464640376e-05,
          9.929339233691296e-05
        ]
      },
      {
        "candidate": "ctx_q4_entry_zero",
        "intervention": "entry_zero",
        "metric": "is_click_weighted_bce",
        "delta_mean": -0.00019721081513485828,
        "block_bootstrap_ci95": [
          -0.0006776490425721724,
          0.00023718459926114742
        ]
      },
      {
        "candidate": "ctx_q4_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "buy_weighted_bce",
        "delta_mean": 0.0002368065095239089,
        "block_bootstrap_ci95": [
          0.00017972361730280394,
          0.0002927564655245339
        ]
      },
      {
        "candidate": "ctx_q4_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "cart_weighted_bce",
        "delta_mean": 0.0001845591582754636,
        "block_bootstrap_ci95": [
          -0.0008353069689764805,
          0.0007853724668506576
        ]
      },
      {
        "candidate": "ctx_q4_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "fav_weighted_bce",
        "delta_mean": 0.00019008358445869847,
        "block_bootstrap_ci95": [
          4.213911347613627e-05,
          0.00033114444827684003
        ]
      },
      {
        "candidate": "ctx_q4_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "is_click_weighted_bce",
        "delta_mean": 0.0014694437878683168,
        "block_bootstrap_ci95": [
          6.21934410904698e-05,
          0.00293978685656828
        ]
      },
      {
        "candidate": "ctx_q4_work_zero",
        "intervention": "work_zero",
        "metric": "buy_weighted_bce",
        "delta_mean": 1.6203058887456045e-05,
        "block_bootstrap_ci95": [
          2.3576129412929878e-06,
          3.0149602649786403e-05
        ]
      },
      {
        "candidate": "ctx_q4_work_zero",
        "intervention": "work_zero",
        "metric": "cart_weighted_bce",
        "delta_mean": 0.00015465545200326475,
        "block_bootstrap_ci95": [
          -3.066116576941142e-05,
          0.00048413645943697986
        ]
      },
      {
        "candidate": "ctx_q4_work_zero",
        "intervention": "work_zero",
        "metric": "fav_weighted_bce",
        "delta_mean": 8.059902580084013e-05,
        "block_bootstrap_ci95": [
          -1.3053473344222491e-05,
          0.00018748527861611546
        ]
      },
      {
        "candidate": "ctx_q4_work_zero",
        "intervention": "work_zero",
        "metric": "is_click_weighted_bce",
        "delta_mean": 0.002028667005275599,
        "block_bootstrap_ci95": [
          0.0010638734406405836,
          0.003097040318449612
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "buy_weighted_bce",
        "delta_mean": 0.00018394704451901007,
        "block_bootstrap_ci95": [
          0.0001356681158315647,
          0.00023797622792157724
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "cart_weighted_bce",
        "delta_mean": 0.00024698168089309957,
        "block_bootstrap_ci95": [
          -9.189675200456036e-05,
          0.000504600459304319
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "fav_weighted_bce",
        "delta_mean": 0.00034123913003204954,
        "block_bootstrap_ci95": [
          0.0001407856293356885,
          0.00048086608158719017
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_shuffle",
        "intervention": "entry_shuffle",
        "metric": "is_click_weighted_bce",
        "delta_mean": -8.751440233958306e-06,
        "block_bootstrap_ci95": [
          -0.0006529853809771286,
          0.0006773977692427321
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_zero",
        "intervention": "entry_zero",
        "metric": "buy_weighted_bce",
        "delta_mean": 7.096335985582e-05,
        "block_bootstrap_ci95": [
          4.662615359888666e-05,
          0.00010168994404467057
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_zero",
        "intervention": "entry_zero",
        "metric": "cart_weighted_bce",
        "delta_mean": -9.861852690334089e-05,
        "block_bootstrap_ci95": [
          -0.0007958505244350993,
          0.00030972490816961204
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_zero",
        "intervention": "entry_zero",
        "metric": "fav_weighted_bce",
        "delta_mean": 0.00011084138559828756,
        "block_bootstrap_ci95": [
          2.7957482566930375e-05,
          0.00017063121837582472
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_entry_zero",
        "intervention": "entry_zero",
        "metric": "is_click_weighted_bce",
        "delta_mean": -0.000177114150159786,
        "block_bootstrap_ci95": [
          -0.00047361428110452293,
          0.0001056542308729102
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "buy_weighted_bce",
        "delta_mean": 0.0007220228317005084,
        "block_bootstrap_ci95": [
          0.0006023942567998275,
          0.0008470590392322846
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "cart_weighted_bce",
        "delta_mean": 0.0008639300245093999,
        "block_bootstrap_ci95": [
          -0.00016524839852308508,
          0.0017081975512065505
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "fav_weighted_bce",
        "delta_mean": 0.0010238184330126816,
        "block_bootstrap_ci95": [
          0.0005848696097573456,
          0.001359723750542884
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_shuffle",
        "intervention": "work_shuffle",
        "metric": "is_click_weighted_bce",
        "delta_mean": 0.0014421338985878896,
        "block_bootstrap_ci95": [
          0.0001673729240958238,
          0.0028393083445856296
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_zero",
        "intervention": "work_zero",
        "metric": "buy_weighted_bce",
        "delta_mean": 0.00011410271891606406,
        "block_bootstrap_ci95": [
          8.973672189841942e-05,
          0.00014242424719643475
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_zero",
        "intervention": "work_zero",
        "metric": "cart_weighted_bce",
        "delta_mean": 0.00015682880408400323,
        "block_bootstrap_ci95": [
          -0.00018028028421220962,
          0.0004380831744174761
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_zero",
        "intervention": "work_zero",
        "metric": "fav_weighted_bce",
        "delta_mean": 0.00010652014145709216,
        "block_bootstrap_ci95": [
          -7.961267352722941e-05,
          0.0002223363988902816
        ]
      },
      {
        "candidate": "explore_dual_no_feedback_work_zero",
        "intervention": "work_zero",
        "metric": "is_click_weighted_bce",
        "delta_mean": 0.000990215260449554,
        "block_bootstrap_ci95": [
          0.00042895371833578456,
          0.0016041948034050233
        ]
      }
    ],
    "conclusion": "Preserve entry and first-work information; their example-specific contributions matter for fav/buy in both checkpoints; work shuffle additionally harms click. Cart remains inconclusive.",
    "scope": "Fixed stratified subset and checkpoints, CPU float32, user bootstrap conditional on selected subset, not full-population selection variance or training-seed variance",
    "training_updates": 0
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/query-conditioning/reports/paired-interpretation.json",
    "workspace/qformer-loop/taobao-transfer/query-conditioning/reports/paired-query-parts/toolkit/perturbation_report.json"
  ]
}
```

</details>
