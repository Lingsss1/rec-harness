## 143. taobao-query-routes-query_q8_entry-route_q8_interaction-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.570225+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q8_entry-route_q8_interaction-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q8_entry",
    "candidate": "route_q8_interaction",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.0023880215844530595,
      "cart": -0.004532707761128685,
      "fav": -0.0020752318926116065,
      "buy": -0.00036893529599957287
    },
    "ci95_auc": {
      "is_click": [
        -0.0029112057256314187,
        -0.001876213285541753
      ],
      "cart": [
        -0.006090808719728419,
        -0.0030547044557458465
      ],
      "fav": [
        -0.003871508110467586,
        -0.0006476439856097721
      ],
      "buy": [
        -0.002621559492969841,
        0.0020638901505910735
      ]
    },
    "delta_macro_auc": -0.002341224133548203,
    "ci95_macro_auc": [
      -0.003260880697020183,
      -0.00151798204853904
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
