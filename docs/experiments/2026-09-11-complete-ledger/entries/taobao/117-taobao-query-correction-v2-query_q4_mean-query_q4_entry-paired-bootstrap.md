## 117. taobao-query-correction-v2-query_q4_mean-query_q4_entry-paired-bootstrap

- 假设：Conditional source and matched-reference AUC differences
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:23:07.342867+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-correction-v2-query_q4_mean-query_q4_entry-paired-bootstrap",
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
    "reference": "query_q4_mean",
    "candidate": "query_q4_entry",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0012857165312695207,
      "cart": -0.0007330336286652805,
      "fav": 0.001452312334141781,
      "buy": -0.001145888673227513
    },
    "ci95_auc": {
      "is_click": [
        0.000737043966803383,
        0.0018691352503661571
      ],
      "cart": [
        -0.002420758378441637,
        0.0009209566069001165
      ],
      "fav": [
        -0.00016775274109475933,
        0.002960641838954413
      ],
      "buy": [
        -0.003365740875313647,
        0.0010217808202261595
      ]
    },
    "delta_macro_auc": 0.00021477664087965476,
    "ci95_macro_auc": [
      -0.0005623838707544133,
      0.0009783773545280073
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
