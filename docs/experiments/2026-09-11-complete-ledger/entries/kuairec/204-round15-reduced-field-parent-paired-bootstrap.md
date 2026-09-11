## 204. round15-reduced-field-parent-paired-bootstrap

- 假设：Quantify conditional AUC uncertainty for smaller field FFN versus r7
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T05:40:04.934339+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "round15-reduced-field-parent-paired-bootstrap",
  "parent_experiment_id": "r15_w512_field-s20262027-20260907T052153858349Z",
  "family": "paired_full_validation_user_bootstrap",
  "hypothesis": "Quantify conditional AUC uncertainty for smaller field FFN versus r7",
  "change": "Resample existing validation users; no new training seed or model updates",
  "budget": {
    "training_updates": 0,
    "training_seed": 20262027,
    "new_training_seed_values": []
  },
  "decision": "provisional",
  "confidence": "fixed_checkpoint_full_validation",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "checkpoint-conditional; no equivalence or cross-seed stability claim"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round15/paired-bootstrap.json"
  ]
}
```

</details>
