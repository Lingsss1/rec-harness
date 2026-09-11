## 91. taobao-q4-r1-r2-paired-bootstrap

- 假设：Second shared work/prediction cycle benefits the four-query no-feedback candidate
- 判定：supports_retaining_two_cycles_conditional_on_checkpoints
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T15:38:01.638698+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-q4-r1-r2-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Second shared work/prediction cycle benefits the four-query no-feedback candidate",
  "change": "500 paired user resamples of matched R1 and R2 predictions",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "q4_r1",
    "candidate": "ctx_q4",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 5.341478532294719e-05,
      "cart": 0.0027787112829202876,
      "fav": 0.0010524773672074828,
      "buy": 0.003956897505808188
    },
    "ci95_auc": {
      "is_click": [
        -0.0005595859446540397,
        0.0006640344557538225
      ],
      "cart": [
        0.0010717410411633988,
        0.004443218618955511
      ],
      "fav": [
        -0.0007056328773794834,
        0.0027809485381162236
      ],
      "buy": [
        0.0015420398270702019,
        0.006444152831939763
      ]
    },
    "delta_macro_auc": 0.001960375235314671,
    "ci95_macro_auc": [
      0.0009976644527789867,
      0.002777110216781185
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "supports_retaining_two_cycles_conditional_on_checkpoints",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/paired-bootstrap-q4-depth.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-q4-capacity/reports/q4-depth-identities.json"
  ]
}
```

</details>
