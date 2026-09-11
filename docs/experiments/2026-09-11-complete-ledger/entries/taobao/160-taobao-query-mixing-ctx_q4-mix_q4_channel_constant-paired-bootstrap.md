## 160. taobao-query-mixing-ctx_q4-mix_q4_channel_constant-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.097254+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-ctx_q4-mix_q4_channel_constant-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "ctx_q4",
    "candidate": "mix_q4_channel_constant",
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
      "is_click": 0.00010811196664728406,
      "cart": -0.00022793388222652755,
      "fav": -0.0004627544962664265,
      "buy": -0.0025799996306186124
    },
    "ci95_auc": {
      "is_click": [
        -0.00040503714291564575,
        0.0006541029600254017
      ],
      "cart": [
        -0.001821141295435813,
        0.0014881386366081685
      ],
      "fav": [
        -0.002042880667650912,
        0.0009823466517281938
      ],
      "buy": [
        -0.004687336377400162,
        -0.0004541224120995038
      ]
    },
    "delta_macro_auc": -0.0007906440106160151,
    "ci95_macro_auc": [
      -0.0015777067009283365,
      6.777985181484609e-05
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
