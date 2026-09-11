## 128. taobao-query-difference-retry-ctx_q4-query_q4_difference-paired-bootstrap

- 假设：Conditional AUC difference for the repaired q4 difference candidate
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:26:50.558466+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-difference-retry-ctx_q4-query_q4_difference-paired-bootstrap",
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
    "reference": "ctx_q4",
    "candidate": "query_q4_difference",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.000649358291499591,
      "cart": -0.0036084839299956872,
      "fav": -0.0031783176158161064,
      "buy": -0.004251780315191467
    },
    "ci95_auc": {
      "is_click": [
        -0.0011875563355810803,
        -8.60070613026067e-05
      ],
      "cart": [
        -0.005362704665546688,
        -0.001973386116724124
      ],
      "fav": [
        -0.004529010719271623,
        -0.001789183917923288
      ],
      "buy": [
        -0.0065061739691032435,
        -0.0020104083797157957
      ]
    },
    "delta_macro_auc": -0.002921985038125574,
    "ci95_macro_auc": [
      -0.003753409740340901,
      -0.0021092758058287773
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
