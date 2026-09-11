## 33. taobao-psgd_post_half-paired-bootstrap

- 假设：Assess optimizer gain conditional on checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T13:17:57.780472+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-psgd_post_half-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Assess optimizer gain conditional on checkpoints",
  "change": "500 paired user resamples, shared weights across heads",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "t3_post_half",
    "candidate": "psgd_post_half",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.003206153359553765,
      "cart": 0.002233626089167662,
      "fav": 0.001650815548383866,
      "buy": 0.0013806296787379768
    },
    "ci95_auc": {
      "is_click": [
        0.0026327195377299994,
        0.0037860617777662805
      ],
      "cart": [
        0.00039134204694001864,
        0.004001230717938991
      ],
      "fav": [
        -0.00013621896204576154,
        0.0034057433243006694
      ],
      "buy": [
        -0.0010987393301721132,
        0.0040161086884229645
      ]
    },
    "delta_macro_auc": 0.002117806168960845,
    "ci95_macro_auc": [
      0.0011739746275845919,
      0.0030948610835725473
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/paired-bootstrap.json"
  ]
}
```

</details>
