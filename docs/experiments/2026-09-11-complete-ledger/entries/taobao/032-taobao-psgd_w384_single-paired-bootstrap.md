## 32. taobao-psgd_w384_single-paired-bootstrap

- 假设：Assess optimizer gain conditional on checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T13:17:57.779902+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-psgd_w384_single-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Assess optimizer gain conditional on checkpoints",
  "change": "500 paired user resamples, shared weights across heads",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "t2_cross8_w384_loop1",
    "candidate": "psgd_w384_single",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0022181628301505274,
      "cart": 0.00335112534664872,
      "fav": 0.0026544250002585867,
      "buy": 0.005421282619586187
    },
    "ci95_auc": {
      "is_click": [
        0.0016102084642900915,
        0.0028090792568192546
      ],
      "cart": [
        0.0015846838899466421,
        0.005313753160170014
      ],
      "fav": [
        0.0010137002986155496,
        0.0043928167711030424
      ],
      "buy": [
        0.0027282616115734347,
        0.007951301119111864
      ]
    },
    "delta_macro_auc": 0.0034112489491611164,
    "ci95_macro_auc": [
      0.0023696681044264984,
      0.004415754909658568
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
