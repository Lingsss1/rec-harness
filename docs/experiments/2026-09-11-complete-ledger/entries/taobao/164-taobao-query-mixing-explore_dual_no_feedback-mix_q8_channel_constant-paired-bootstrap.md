## 164. taobao-query-mixing-explore_dual_no_feedback-mix_q8_channel_constant-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.111449+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-explore_dual_no_feedback-mix_q8_channel_constant-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "explore_dual_no_feedback",
    "candidate": "mix_q8_channel_constant",
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
      "is_click": -0.0014902314774186687,
      "cart": -0.0030522020401672068,
      "fav": -0.002188739734391376,
      "buy": -0.003092635002234001
    },
    "ci95_auc": {
      "is_click": [
        -0.0021199715335199405,
        -0.00097348431138746
      ],
      "cart": [
        -0.0046886865729858646,
        -0.0015056325840211028
      ],
      "fav": [
        -0.0037316955363647596,
        -0.0005708480873126682
      ],
      "buy": [
        -0.0052659927506580025,
        -0.0010059651883430447
      ]
    },
    "delta_macro_auc": -0.0024559520635528687,
    "ci95_macro_auc": [
      -0.0032940635403033297,
      -0.0016133641263571176
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
