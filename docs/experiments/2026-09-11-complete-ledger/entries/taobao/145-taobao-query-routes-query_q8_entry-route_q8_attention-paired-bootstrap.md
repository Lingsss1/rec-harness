## 145. taobao-query-routes-query_q8_entry-route_q8_attention-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.575633+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q8_entry-route_q8_attention-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q8_entry",
    "candidate": "route_q8_attention",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.0019389878450730702,
      "cart": -0.004795425811231047,
      "fav": -0.002276787449856821,
      "buy": -0.0014190206212415513
    },
    "ci95_auc": {
      "is_click": [
        -0.0025169183563057166,
        -0.0014643022434374434
      ],
      "cart": [
        -0.006603904142379008,
        -0.003297326211658351
      ],
      "fav": [
        -0.003795467474348299,
        -0.0006384997869894798
      ],
      "buy": [
        -0.003908334348609812,
        0.0009401608190591796
      ]
    },
    "delta_macro_auc": -0.00260755543185065,
    "ci95_macro_auc": [
      -0.003517084885309231,
      -0.0017464702277697282
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
