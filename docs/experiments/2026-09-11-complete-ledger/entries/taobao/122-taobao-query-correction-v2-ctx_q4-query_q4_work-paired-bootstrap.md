## 122. taobao-query-correction-v2-ctx_q4-query_q4_work-paired-bootstrap

- 假设：Conditional source and matched-reference AUC differences
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:23:07.342867+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-correction-v2-ctx_q4-query_q4_work-paired-bootstrap",
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
    "reference": "ctx_q4",
    "candidate": "query_q4_work",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0011027535201866279,
      "cart": 0.00020634737945157688,
      "fav": -0.00011416036563804965,
      "buy": -0.001222095873015916
    },
    "ci95_auc": {
      "is_click": [
        0.0006414599064769434,
        0.0016085138821537086
      ],
      "cart": [
        -0.001475593284591059,
        0.0018732404727273067
      ],
      "fav": [
        -0.0016760500319913801,
        0.0014109457778608947
      ],
      "buy": [
        -0.0031892156682593624,
        0.00048077678255460297
      ]
    },
    "delta_macro_auc": -6.7888347538014315e-06,
    "ci95_macro_auc": [
      -0.0008123229695614021,
      0.0007222719546049674
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
