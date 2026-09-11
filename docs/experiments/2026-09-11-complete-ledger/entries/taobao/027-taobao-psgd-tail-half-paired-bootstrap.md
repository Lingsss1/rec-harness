## 27. taobao-psgd-tail-half-paired-bootstrap

- 假设：Assess optimizer gain conditional on fixed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T13:03:47.141600+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-psgd-tail-half-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Assess optimizer gain conditional on fixed checkpoints",
  "change": "500 paired user resamples, shared weights across heads",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "decision": "provisional",
  "metrics": {
    "reference": "t3_tail_half",
    "candidate": "psgd_t3_tail_half",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0007743420525723188,
      "cart": -0.0001985145936593824,
      "fav": 0.001168309898755382,
      "buy": 0.0013616314571823773
    },
    "ci95_auc": {
      "is_click": [
        0.00019222947346077268,
        0.0013849759930816133
      ],
      "cart": [
        -0.002101097739404539,
        0.0016405904251271392
      ],
      "fav": [
        -0.000677167720969482,
        0.0030036028449697705
      ],
      "buy": [
        -0.001120192164669387,
        0.0040782762635037636
      ]
    },
    "delta_macro_auc": 0.0007764422037126462,
    "ci95_macro_auc": [
      -0.0001912824473878249,
      0.0016948710897357222
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/optimizer-psgd/reports/paired-bootstrap.json"
  ]
}
```

</details>
