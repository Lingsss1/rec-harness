## 144. taobao-query-routes-query_q8_entry-route_q8_residual-paired-bootstrap

- 假设：Quantify route-vs-all and matched baseline differences conditional on completed checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:27:40.572924+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-routes-query_q8_entry-route_q8_residual-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify route-vs-all and matched baseline differences conditional on completed checkpoints",
  "change": "500 paired user resamples with shared task weights",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "query_q8_entry",
    "candidate": "route_q8_residual",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "training_updates": 0,
    "new_training_seeds": [],
    "mandatory_all_route_contrast": true,
    "delta_auc": {
      "is_click": -0.005057084673297241,
      "cart": -0.004419185956446081,
      "fav": -0.004329441596192973,
      "buy": -0.004388690180176735
    },
    "ci95_auc": {
      "is_click": [
        -0.005668965304377477,
        -0.004459193940990561
      ],
      "cart": [
        -0.006110195812641622,
        -0.002981770901350861
      ],
      "fav": [
        -0.006056666536700475,
        -0.0026699342814827222
      ],
      "buy": [
        -0.006765286528549677,
        -0.0018239656984592215
      ]
    },
    "delta_macro_auc": -0.004548600601528285,
    "ci95_macro_auc": [
      -0.005454212685787415,
      -0.003618478271830941
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
