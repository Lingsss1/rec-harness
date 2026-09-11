## 37. taobao-tune-psgd_beta095-paired-bootstrap

- 假设：Conditional optimizer gain
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T16:14:08.326743+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-tune-psgd_beta095-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Conditional optimizer gain",
  "change": "500 paired user resamples",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "psgd_w384_single",
    "candidate": "psgd_beta095",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.000332018275061019,
      "cart": 0.0013978611255235363,
      "fav": 0.0021023016109559256,
      "buy": -0.0010579026944415038
    },
    "ci95_auc": {
      "is_click": [
        -0.0008803886042271003,
        0.0001862609269602855
      ],
      "cart": [
        -0.0003016939272494079,
        0.0032683979221558718
      ],
      "fav": [
        0.0006534263253855727,
        0.0037003130942986546
      ],
      "buy": [
        -0.0032669844623983424,
        0.0011344916099492119
      ]
    },
    "delta_macro_auc": 0.0005275604417441793,
    "ci95_macro_auc": [
      -0.0002266175298421019,
      0.001436193043529698
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/optimizer-psgd-tune/reports/paired-bootstrap.json"
  ]
}
```

</details>
