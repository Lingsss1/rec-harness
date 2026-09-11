## 163. taobao-query-mixing-explore_dual_no_feedback-mix_q8_channel_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.107834+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-explore_dual_no_feedback-mix_q8_channel_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "explore_dual_no_feedback",
    "candidate": "mix_q8_channel_joint",
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
      "is_click": -0.004666465801792641,
      "cart": -0.006682770277031458,
      "fav": -0.0038061049515943912,
      "buy": -0.004828442869057148
    },
    "ci95_auc": {
      "is_click": [
        -0.005303926320941235,
        -0.004020554237367991
      ],
      "cart": [
        -0.008424294821025326,
        -0.004947287797701411
      ],
      "fav": [
        -0.005325637926526164,
        -0.0021637805524471645
      ],
      "buy": [
        -0.007317971514083537,
        -0.002309674782114167
      ]
    },
    "delta_macro_auc": -0.004995945974868965,
    "ci95_macro_auc": [
      -0.005948306359540133,
      -0.004113655213096361
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
