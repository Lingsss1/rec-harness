## 119. taobao-query-correction-v2-query_q8_mean-query_q8_difference-paired-bootstrap

- 假设：Conditional source and matched-reference AUC differences
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:23:07.342867+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-correction-v2-query_q8_mean-query_q8_difference-paired-bootstrap",
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
    "candidate": "query_q8_difference",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.0012566377620000235,
      "cart": -0.0014103274759088569,
      "fav": 0.0004423509547810678,
      "buy": -0.0026984945560012408
    },
    "ci95_auc": {
      "is_click": [
        -0.0018536652356977046,
        -0.0007269728896501824
      ],
      "cart": [
        -0.002952265736481302,
        3.4199695468764144e-05
      ],
      "fav": [
        -0.001159432599056123,
        0.0019341398672904906
      ],
      "buy": [
        -0.0048981254757008656,
        -0.0003924323233125234
      ]
    },
    "delta_macro_auc": -0.0012307772097823744,
    "ci95_macro_auc": [
      -0.0020491276973670864,
      -0.00042260563908793513
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
