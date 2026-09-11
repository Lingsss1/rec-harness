## 138. taobao-query-routes-query_q4_work-route_q4_retrieval-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.557016+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q4_work-route_q4_retrieval-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q4_work",
    "candidate": "route_q4_retrieval",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.0004622748162819157,
      "cart": -0.0015662881059558265,
      "fav": -0.0006671693585835747,
      "buy": 0.0008664028744558294
    },
    "ci95_auc": {
      "is_click": [
        -0.0009434891334504896,
        5.6222812030701e-05
      ],
      "cart": [
        -0.0032106638337418363,
        5.604299879227124e-05
      ],
      "fav": [
        -0.00217890406848098,
        0.0007680356772535146
      ],
      "buy": [
        -0.0009210465548232321,
        0.003152878766117981
      ]
    },
    "delta_macro_auc": -0.00045733235159151064,
    "ci95_macro_auc": [
      -0.0012380908214365501,
      0.00038367189431029393
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
