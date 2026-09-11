## 142. taobao-query-routes-query_q8_entry-route_q8_retrieval-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.567513+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q8_entry-route_q8_retrieval-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q8_entry",
    "candidate": "route_q8_retrieval",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.0035101699785703433,
      "cart": -0.005462728705010189,
      "fav": -0.0038275792846406143,
      "buy": -0.0019681202300498857
    },
    "ci95_auc": {
      "is_click": [
        -0.004106837279099121,
        -0.0029366203297663345
      ],
      "cart": [
        -0.0072662193722900035,
        -0.0039862639460661155
      ],
      "fav": [
        -0.0054317599367244285,
        -0.0023941700022121677
      ],
      "buy": [
        -0.0041958565219229835,
        0.0005787094301961365
      ]
    },
    "delta_macro_auc": -0.003692149549567758,
    "ci95_macro_auc": [
      -0.004526139306515168,
      -0.002840970183414894
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
