## 41. taobao-middle-middle_r2-paired-bootstrap

- 假设：Conditional middle recurrence gain
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T16:59:16.030007+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-middle-middle_r2-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Conditional middle recurrence gain",
  "change": "500 paired user resamples",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "psgd_w384_single",
    "candidate": "middle_r2",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.0008816746221884664,
      "cart": 0.00038184561785337134,
      "fav": 0.0009445740954674431,
      "buy": -0.00029776072824172495
    },
    "ci95_auc": {
      "is_click": [
        -0.0015185330275847618,
        -0.000262128733641268
      ],
      "cart": [
        -0.001412407359209264,
        0.002135253999489786
      ],
      "fav": [
        -0.0007669938266757158,
        0.0025276017763292435
      ],
      "buy": [
        -0.002441684202186553,
        0.0022847260022598485
      ]
    },
    "delta_macro_auc": 3.674609072257251e-05,
    "ci95_macro_auc": [
      -0.0007907269024882597,
      0.0010020121193010493
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-middle/reports/paired-bootstrap.json"
  ]
}
```

</details>
