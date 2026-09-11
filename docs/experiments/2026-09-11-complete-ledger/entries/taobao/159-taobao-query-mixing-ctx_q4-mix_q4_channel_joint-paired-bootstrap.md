## 159. taobao-query-mixing-ctx_q4-mix_q4_channel_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.093667+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-ctx_q4-mix_q4_channel_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "ctx_q4",
    "candidate": "mix_q4_channel_joint",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "reference_role": "original_recurrence",
    "mandatory": true,
    "mandatory_joint_constant_contrast": false,
    "mandatory_original_recurrence_contrast": true,
    "delta_auc": {
      "is_click": 0.0010477879755581565,
      "cart": -0.0005357268052302855,
      "fav": -0.0001330516824029626,
      "buy": -0.0013545075722954936
    },
    "ci95_auc": {
      "is_click": [
        0.0004861313039931425,
        0.0015792725353308583
      ],
      "cart": [
        -0.002235592051036747,
        0.0010961163527792054
      ],
      "fav": [
        -0.0016314016624072946,
        0.001227642832564915
      ],
      "buy": [
        -0.0036294354134415283,
        0.0006782916088449606
      ]
    },
    "delta_macro_auc": -0.0002438745210926463,
    "ci95_macro_auc": [
      -0.001100299388456618,
      0.000557029552632559
    ],
    "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/paired-bootstrap-identities.json"
  ]
}
```

</details>
