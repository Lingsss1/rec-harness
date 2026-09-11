## 31. taobao-psgd_cross8_clip1-paired-bootstrap

- 假设：Assess optimizer gain conditional on checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T13:17:57.779355+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-psgd_cross8_clip1-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Assess optimizer gain conditional on checkpoints",
  "change": "500 paired user resamples, shared weights across heads",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "taobao_cross8_clip1",
    "candidate": "psgd_cross8_clip1",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.004408348391725769,
      "cart": 0.00018438756131011758,
      "fav": 0.0021971984741917527,
      "buy": 0.003965041464517927
    },
    "ci95_auc": {
      "is_click": [
        0.00376594391586938,
        0.004947953190160986
      ],
      "cart": [
        -0.0014783048672256966,
        0.0017515150199266026
      ],
      "fav": [
        0.0006366941388361886,
        0.0038074219747923853
      ],
      "buy": [
        0.0015276542570843655,
        0.00654668024596603
      ]
    },
    "delta_macro_auc": 0.0026887439729363916,
    "ci95_macro_auc": [
      0.0017247593871374401,
      0.0035902692496661676
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
