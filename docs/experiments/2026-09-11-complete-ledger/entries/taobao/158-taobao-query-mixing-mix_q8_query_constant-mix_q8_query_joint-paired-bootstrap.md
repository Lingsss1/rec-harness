## 158. taobao-query-mixing-mix_q8_query_constant-mix_q8_query_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.090244+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-mix_q8_query_constant-mix_q8_query_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "mix_q8_query_constant",
    "candidate": "mix_q8_query_joint",
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
      "is_click": -0.001051994572609516,
      "cart": -0.0013688491357850552,
      "fav": -0.0019518679182036758,
      "buy": -0.001923074119085988
    },
    "ci95_auc": {
      "is_click": [
        -0.0016327013381994908,
        -0.0004522785970629773
      ],
      "cart": [
        -0.002993602416902938,
        0.00022663232321096734
      ],
      "fav": [
        -0.0034549692956826,
        -0.0005346780531825391
      ],
      "buy": [
        -0.0040314029894937455,
        0.00020800352740136724
      ]
    },
    "delta_macro_auc": -0.0015739464364210587,
    "ci95_macro_auc": [
      -0.002349156794962986,
      -0.0007747138269309635
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
