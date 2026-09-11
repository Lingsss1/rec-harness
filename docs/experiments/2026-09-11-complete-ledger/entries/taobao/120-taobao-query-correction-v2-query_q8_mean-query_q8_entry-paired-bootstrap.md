## 120. taobao-query-correction-v2-query_q8_mean-query_q8_entry-paired-bootstrap

- 假设：Conditional source and matched-reference AUC differences
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:23:07.342867+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-correction-v2-query_q8_mean-query_q8_entry-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "code_id": "05eab0e32d1b31729f2b2d2bea8321d42bef2af844449dd0cabbb2715bb79091",
  "hypothesis": "Conditional source and matched-reference AUC differences",
  "change": "500 paired user resamples; all five complete source-vs-mean pairs plus q4_work-vs-ctx_q4",
  "budget": {
    "training_updates": 0,
    "bootstrap_seed": 20260906,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q8_mean",
    "candidate": "query_q8_entry",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.004100375630004427,
      "cart": 0.0033965689099487717,
      "fav": 0.003613077178069135,
      "buy": 0.0005869634721303862
    },
    "ci95_auc": {
      "is_click": [
        0.0035110922229819375,
        0.0046909147777043775
      ],
      "cart": [
        0.0019360865990669385,
        0.004938323840371894
      ],
      "fav": [
        0.002074383409158176,
        0.005270053456475373
      ],
      "buy": [
        -0.0018011947396040337,
        0.0029563215280712647
      ]
    },
    "delta_macro_auc": 0.00292424629753818,
    "ci95_macro_auc": [
      0.002119145044531312,
      0.003820982325002739
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "provisional",
  "confidence": "One training seed; fixed checkpoints and released validation protocol; selection uncorrected; no training-seed variance or equivalence claim.",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-paired-bootstrap.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/partial-summary.json"
  ]
}
```

</details>
