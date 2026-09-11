## 155. taobao-query-mixing-mix_q4_channel_constant-mix_q4_channel_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.079904+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-mix_q4_channel_constant-mix_q4_channel_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "mix_q4_channel_constant",
    "candidate": "mix_q4_channel_joint",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "reference_role": "conditioning_control",
    "mandatory": true,
    "mandatory_joint_constant_contrast": true,
    "mandatory_original_recurrence_contrast": false,
    "delta_auc": {
      "is_click": 0.0009396760089108724,
      "cart": -0.00030779292300375793,
      "fav": 0.0003297028138634639,
      "buy": 0.0012254920583231188
    },
    "ci95_auc": {
      "is_click": [
        0.0003943137650293805,
        0.0014970511368180475
      ],
      "cart": [
        -0.0018891050093504669,
        0.0012896822093320906
      ],
      "fav": [
        -0.0011219161088202034,
        0.0016688241167196354
      ],
      "buy": [
        -0.0011389509503691025,
        0.0036465499866002904
      ]
    },
    "delta_macro_auc": 0.0005467694895233688,
    "ci95_macro_auc": [
      -0.00034580346755074193,
      0.0014149764367677583
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
