## 129. taobao-query-difference-retry-psgd_precond_every_step-query_q4_difference-paired-bootstrap

- 假设：Conditional AUC difference for the repaired q4 difference candidate
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T03:26:50.558466+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-query-difference-retry-psgd_precond_every_step-query_q4_difference-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "code_id": "f9e9d73f33b3b5250cc5da7f5bd60927ac21a9cfd2ac181da6736ae23272524b",
  "hypothesis": "Conditional AUC difference for the repaired q4 difference candidate",
  "change": "500 paired user resamples; previous six bootstrap comparisons reused without re-appending",
  "budget": {
    "training_updates": 0,
    "bootstrap_seed": 20260906,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "psgd_precond_every_step",
    "candidate": "query_q4_difference",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": -0.002465089659410058,
      "cart": -0.003912532723411077,
      "fav": -0.002865635281870005,
      "buy": -0.0030485508615361034
    },
    "ci95_auc": {
      "is_click": [
        -0.0030891235326970775,
        -0.001737233078033338
      ],
      "cart": [
        -0.005738028271759915,
        -0.0021337252292220887
      ],
      "fav": [
        -0.004489896336556082,
        -0.0011344396593385825
      ],
      "buy": [
        -0.005630487389686914,
        -0.00035356983158002996
      ]
    },
    "delta_macro_auc": -0.003072952131556672,
    "ci95_macro_auc": [
      -0.004032821387491707,
      -0.002135923314224895
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights; retry has the audited narrow initialization guard"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/paired-bootstrap-audit.json"
  ]
}
```

</details>
