## 127. taobao-query-difference-retry-query_q4_mean-query_q4_difference-paired-bootstrap

- 假设：Conditional AUC difference for the repaired q4 difference candidate
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:26:50.558466+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-difference-retry-query_q4_mean-query_q4_difference-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "code_id": "f9e9d73f33b3b5250cc5da7f5bd60927ac21a9cfd2ac181da6736ae23272524b",
  "hypothesis": "Conditional AUC difference for the repaired q4 difference candidate",
  "change": "500 paired user resamples; previous six bootstrap comparisons reused without re-appending",
  "budget": {
    "training_updates": 0,
    "bootstrap_seed": 20260906,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q4_mean",
    "candidate": "query_q4_difference",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.00035078811153266276,
      "cart": -0.00249455174625568,
      "fav": -0.0011302141469782256,
      "buy": -0.0007996417578494119
    },
    "ci95_auc": {
      "is_click": [
        -0.0009008415960933297,
        0.0002279069749941187
      ],
      "cart": [
        -0.004205905914404656,
        -0.0007017622641132436
      ],
      "fav": [
        -0.0025798012705587846,
        0.0003878690045789084
      ],
      "buy": [
        -0.003152447451549492,
        0.0012122667161488714
      ]
    },
    "delta_macro_auc": -0.0011937989406538563,
    "ci95_macro_auc": [
      -0.0020513313711588385,
      -0.00037679422165299794
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights; retry has the audited narrow initialization guard"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap-audit.json"
  ]
}
```

</details>
