## 166. taobao-query-mixing-explore_dual_no_feedback-mix_q8_query_constant-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.118499+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-explore_dual_no_feedback-mix_q8_query_constant-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "explore_dual_no_feedback",
    "candidate": "mix_q8_query_constant",
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
      "is_click": -0.002135809562843094,
      "cart": -0.0015591420609105677,
      "fav": -0.0013614398129414562,
      "buy": -0.0006261882978763067
    },
    "ci95_auc": {
      "is_click": [
        -0.002732984634094346,
        -0.0015873871743291219
      ],
      "cart": [
        -0.002929317925476585,
        -0.0001227159747359377
      ],
      "fav": [
        -0.0029469159526564954,
        0.00013214074761747896
      ],
      "buy": [
        -0.002721334952174195,
        0.0014771292637797944
      ]
    },
    "delta_macro_auc": -0.0014206449336429117,
    "ci95_macro_auc": [
      -0.0021973074694157215,
      -0.0006391588014134531
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
