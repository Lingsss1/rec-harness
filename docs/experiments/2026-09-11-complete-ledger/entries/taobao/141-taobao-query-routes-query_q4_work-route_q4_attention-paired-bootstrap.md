## 141. taobao-query-routes-query_q4_work-route_q4_attention-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.564854+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q4_work-route_q4_attention-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q4_work",
    "candidate": "route_q4_attention",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.00615657540659631,
      "cart": -0.001842093358151109,
      "fav": -0.002520994028748391,
      "buy": -0.004540954115300755
    },
    "ci95_auc": {
      "is_click": [
        -0.006702206593744872,
        -0.005637278903941437
      ],
      "cart": [
        -0.0035445002070031402,
        -0.0002729133207827435
      ],
      "fav": [
        -0.003916951512801156,
        -0.0010640527304267959
      ],
      "buy": [
        -0.00658802200234408,
        -0.0022281329838223563
      ]
    },
    "delta_macro_auc": -0.003765154227199252,
    "ci95_macro_auc": [
      -0.004567699385545,
      -0.0029791525489986643
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
