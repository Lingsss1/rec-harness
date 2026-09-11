## 56. taobao-dual-state-dual_state_r1-dual_state_r2-paired-bootstrap

- 假设：Conditional core update gain
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T09:45:10.169569+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-dual-state-dual_state_r1-dual_state_r2-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Conditional core update gain",
  "change": "500 paired user resamples",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "dual_state_r1",
    "candidate": "dual_state_r2",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.00031218979397706903,
      "cart": 0.004063721006305143,
      "fav": -0.00021610097552882124,
      "buy": -0.0011089479004676717
    },
    "ci95_auc": {
      "is_click": [
        -0.00030860508748696276,
        0.0009073200560874734
      ],
      "cart": [
        0.002225742660435695,
        0.005830028510958245
      ],
      "fav": [
        -0.0020350644504237097,
        0.0018138984514218018
      ],
      "buy": [
        -0.00369910768687961,
        0.0016454626588159793
      ]
    },
    "delta_macro_auc": 0.0007627154810713188,
    "ci95_macro_auc": [
      -0.00022430915777606886,
      0.0017334739123308165
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-dual-state/reports/paired-bootstrap.json"
  ]
}
```

</details>
