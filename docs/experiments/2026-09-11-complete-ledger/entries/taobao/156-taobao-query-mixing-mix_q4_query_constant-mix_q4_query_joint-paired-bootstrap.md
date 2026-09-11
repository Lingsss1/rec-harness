## 156. taobao-query-mixing-mix_q4_query_constant-mix_q4_query_joint-paired-bootstrap

- 假设：Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T04:50:34.083239+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-mixing-mix_q4_query_constant-mix_q4_query_joint-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify joint-vs-constant and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "mix_q4_query_constant",
    "candidate": "mix_q4_query_joint",
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
      "is_click": 0.004026301924688225,
      "cart": 0.00033814925358510717,
      "fav": 0.0037764315173665075,
      "buy": 0.0026777429462135904
    },
    "ci95_auc": {
      "is_click": [
        0.0033928000699526588,
        0.00464941256216283
      ],
      "cart": [
        -0.0012317892293798455,
        0.002091886799480741
      ],
      "fav": [
        0.002033056015334997,
        0.005640834004541323
      ],
      "buy": [
        0.0005798320885039538,
        0.004631683856751228
      ]
    },
    "delta_macro_auc": 0.0027046564104633575,
    "ci95_macro_auc": [
      0.0018877894099423621,
      0.0034858632233484375
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
