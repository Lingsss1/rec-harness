## 165. taobao-query-mixing-explore_dual_no_feedback-mix_q8_query_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.114941+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-explore_dual_no_feedback-mix_q8_query_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "explore_dual_no_feedback",
    "candidate": "mix_q8_query_joint",
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
      "is_click": -0.00318780413545261,
      "cart": -0.002927991196695623,
      "fav": -0.003313307731145132,
      "buy": -0.0025492624169622946
    },
    "ci95_auc": {
      "is_click": [
        -0.00377688877041345,
        -0.0026360426485845497
      ],
      "cart": [
        -0.004382965118201504,
        -0.0011609163136749258
      ],
      "fav": [
        -0.005025863481237708,
        -0.0016432432616736634
      ],
      "buy": [
        -0.0045612975915463125,
        -0.0002295001348551552
      ]
    },
    "delta_macro_auc": -0.0029945913700639704,
    "ci95_macro_auc": [
      -0.003824687661344195,
      -0.0021077088514912207
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
