## 139. taobao-query-routes-query_q4_work-route_q4_interaction-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.559634+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q4_work-route_q4_interaction-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q4_work",
    "candidate": "route_q4_interaction",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.0006969844377778811,
      "cart": -0.0018230978597447711,
      "fav": -0.0005237597308971331,
      "buy": -0.0011259482668550813
    },
    "ci95_auc": {
      "is_click": [
        -0.0012221655063016535,
        -0.00018319823987017502
      ],
      "cart": [
        -0.003755365781442868,
        -0.00022164820755717453
      ],
      "fav": [
        -0.001970508552103814,
        0.0009667000274656992
      ],
      "buy": [
        -0.002913621835426747,
        0.0009297679440420327
      ]
    },
    "delta_macro_auc": -0.0010424475738187722,
    "ci95_macro_auc": [
      -0.0018051809894652972,
      -0.0003267512035853428
    ],
    "scope": "Paired user-cluster bootstrap conditional on fixed checkpoints and released validation; shared user weights across four heads; no training-seed uncertainty and no candidate/source-selection correction"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/paired-bootstrap-identities.json"
  ]
}
```

</details>
