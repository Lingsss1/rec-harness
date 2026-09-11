## 161. taobao-query-mixing-ctx_q4-mix_q4_query_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.100807+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-ctx_q4-mix_q4_query_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "ctx_q4",
    "candidate": "mix_q4_query_joint",
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
      "is_click": 0.0012016268340766523,
      "cart": -0.0012313756430578637,
      "fav": 0.001037502030625248,
      "buy": -0.0017425635177172882
    },
    "ci95_auc": {
      "is_click": [
        0.0006715907249102065,
        0.0017842724149109783
      ],
      "cart": [
        -0.002744995310557466,
        0.00033933726732431605
      ],
      "fav": [
        -0.0005161908320194086,
        0.0024694815205208915
      ],
      "buy": [
        -0.003806432781218372,
        0.00020590958116525608
      ]
    },
    "delta_macro_auc": -0.00018370257401822965,
    "ci95_macro_auc": [
      -0.0009206318506348954,
      0.0005333416620977589
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
