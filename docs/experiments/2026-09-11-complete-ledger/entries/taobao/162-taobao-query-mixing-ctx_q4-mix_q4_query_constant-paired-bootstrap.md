## 162. taobao-query-mixing-ctx_q4-mix_q4_query_constant-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.104359+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-ctx_q4-mix_q4_query_constant-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "ctx_q4",
    "candidate": "mix_q4_query_constant",
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
      "is_click": -0.0028246750906115725,
      "cart": -0.001569524896642971,
      "fav": -0.0027389294867412595,
      "buy": -0.004420306463930879
    },
    "ci95_auc": {
      "is_click": [
        -0.0034464809343528803,
        -0.0022335174474880544
      ],
      "cart": [
        -0.003158353599223049,
        4.170809789978412e-05
      ],
      "fav": [
        -0.004562041673711182,
        -0.0012425652951645967
      ],
      "buy": [
        -0.006733434564550341,
        -0.002332306765200038
      ]
    },
    "delta_macro_auc": -0.002888358984481587,
    "ci95_macro_auc": [
      -0.0037251126323888406,
      -0.0020567260238820934
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
