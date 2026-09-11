## 193. round14-gated-parent-paired-bootstrap

- 假设：Quantify conditional AUC uncertainty between best gated model and r7
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T05:10:48.032836+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "round14-gated-parent-paired-bootstrap",
  "parent_experiment_id": "r14_swiglu_both-s20262027-20260907T045344417858Z",
  "family": "paired_full_validation_user_bootstrap",
  "hypothesis": "Quantify conditional AUC uncertainty between best gated model and r7",
  "change": "Resample existing validation users with the existing bootstrap seed; no training",
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
      "conditional on checkpoints; no test evaluation"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round14/paired-bootstrap.json"
  ]
}
```

</details>
