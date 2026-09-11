## 121. taobao-query-correction-v2-query_q8_mean-query_q8_work-paired-bootstrap

- 假设：Conditional source and matched-reference AUC differences
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:23:07.342867+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-correction-v2-query_q8_mean-query_q8_work-paired-bootstrap",
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
    "candidate": "query_q8_work",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.002158101721792982,
      "cart": -0.001005683161173776,
      "fav": -0.0002843113910905304,
      "buy": -0.0018902436653888133
    },
    "ci95_auc": {
      "is_click": [
        0.0016808721782525932,
        0.002664389425215474
      ],
      "cart": [
        -0.002443173179953803,
        0.0006064875405822386
      ],
      "fav": [
        -0.0018867565722344936,
        0.0013077762940695227
      ],
      "buy": [
        -0.003918165079762443,
        0.00040884172780958957
      ]
    },
    "delta_macro_auc": -0.00025553412396506214,
    "ci95_macro_auc": [
      -0.0010806146494780281,
      0.0005904822100743165
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
