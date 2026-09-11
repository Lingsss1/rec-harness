## 118. taobao-query-correction-v2-query_q4_mean-query_q4_work-paired-bootstrap

- 假设：Conditional source and matched-reference AUC differences
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:23:07.342867+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-correction-v2-query_q4_mean-query_q4_work-paired-bootstrap",
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
    "candidate": "query_q4_work",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0014013237001535561,
      "cart": 0.0013202795631915842,
      "fav": 0.001933943103199831,
      "buy": 0.002230042684326139
    },
    "ci95_auc": {
      "is_click": [
        0.0008700551576478671,
        0.0019931821512234687
      ],
      "cart": [
        1.734806294756709e-05,
        0.00287747286338657
      ],
      "fav": [
        0.0004072628389658606,
        0.0037043213913475173
      ],
      "buy": [
        -0.0001436095969437611,
        0.004205596400103997
      ]
    },
    "delta_macro_auc": 0.0017213972627179164,
    "ci95_macro_auc": [
      0.0008525996105997538,
      0.0024866644337607356
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
