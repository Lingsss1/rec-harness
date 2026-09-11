## 157. taobao-query-mixing-mix_q8_channel_constant-mix_q8_channel_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.086669+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-mix_q8_channel_constant-mix_q8_channel_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "mix_q8_channel_constant",
    "candidate": "mix_q8_channel_joint",
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
      "is_click": -0.003176234324373972,
      "cart": -0.003630568236864251,
      "fav": -0.001617365217203015,
      "buy": -0.001735807866823147
    },
    "ci95_auc": {
      "is_click": [
        -0.003709854482947536,
        -0.002597738623642038
      ],
      "cart": [
        -0.005280279111062139,
        -0.002204917418873953
      ],
      "fav": [
        -0.0032190323571760714,
        9.241166552236895e-06
      ],
      "buy": [
        -0.004123537443484607,
        0.0008153443603363585
      ]
    },
    "delta_macro_auc": -0.0025399939113160963,
    "ci95_macro_auc": [
      -0.00339650489045982,
      -0.0017128938910583942
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
