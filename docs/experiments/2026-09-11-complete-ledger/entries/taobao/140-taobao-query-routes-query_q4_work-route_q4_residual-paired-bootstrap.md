## 140. taobao-query-routes-query_q4_work-route_q4_residual-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.562244+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q4_work-route_q4_residual-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q4_work",
    "candidate": "route_q4_residual",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.00018398864549418192,
      "cart": -0.0032051886648571948,
      "fav": -0.003109561726142962,
      "buy": -0.003429361833394151
    },
    "ci95_auc": {
      "is_click": [
        -0.0006990199088502902,
        0.00031010286377100295
      ],
      "cart": [
        -0.004748143614072051,
        -0.001525712242504765
      ],
      "fav": [
        -0.004683282889441473,
        -0.0016483394934014412
      ],
      "buy": [
        -0.005645931523643777,
        -0.0013518331743412506
      ]
    },
    "delta_macro_auc": -0.0024820252174722057,
    "ci95_macro_auc": [
      -0.003322204222685762,
      -0.0016184830848244827
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
