## 65. taobao-dual-feedback-removal-paired-bootstrap

- 假设：Does no-feedback working-state update improve versus feedback R2?
- 判定：retain_candidate
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T10:26:53.545415+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-dual-feedback-removal-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Does no-feedback working-state update improve versus feedback R2?",
  "change": "500 paired user resamples on frozen full-validation predictions",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "dual_state_r2",
    "candidate": "explore_dual_no_feedback",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0006175072169222195,
      "cart": 3.727132830300128e-05,
      "fav": 0.0012146088011447675,
      "buy": 0.0026399166234053473
    },
    "ci95_auc": {
      "is_click": [
        -9.45526074271341e-06,
        0.001305786526179649
      ],
      "cart": [
        -0.00150406237966815,
        0.0017769361724243604
      ],
      "fav": [
        -0.0008559802918301106,
        0.0029619419850794608
      ],
      "buy": [
        0.00016896425774538004,
        0.005223309885911338
      ]
    },
    "delta_macro_auc": 0.0011273259924439172,
    "ci95_macro_auc": [
      0.000220408837119241,
      0.0022419871031725317
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "retain_candidate",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-explore/reports/feedback-paired-bootstrap.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-explore/feedback_bootstrap.py"
  ]
}
```

</details>
