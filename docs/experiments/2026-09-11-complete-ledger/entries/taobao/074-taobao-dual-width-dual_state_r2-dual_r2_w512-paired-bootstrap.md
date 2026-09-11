## 74. taobao-dual-width-dual_state_r2-dual_r2_w512-paired-bootstrap

- 假设：Conditional core update gain
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T11:49:50.265831+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-dual-width-dual_state_r2-dual_r2_w512-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Conditional core update gain",
  "change": "500 paired user resamples",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "dual_state_r2",
    "candidate": "dual_r2_w512",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.0016116110351258595,
      "cart": -0.000544252883807661,
      "fav": -0.0001511695127153212,
      "buy": 0.004092859570019591
    },
    "ci95_auc": {
      "is_click": [
        -0.00223691657191237,
        -0.0009354073340982206
      ],
      "cart": [
        -0.0021766858516847286,
        0.0013475358010714273
      ],
      "fav": [
        -0.0021901036289165093,
        0.0016272742411746076
      ],
      "buy": [
        0.0015420543464195607,
        0.006600824689496385
      ]
    },
    "delta_macro_auc": 0.0004464565345926319,
    "ci95_macro_auc": [
      -0.00048120875815612914,
      0.0014366718223338686
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-dual-width/reports/paired-bootstrap.json"
  ]
}
```

</details>
