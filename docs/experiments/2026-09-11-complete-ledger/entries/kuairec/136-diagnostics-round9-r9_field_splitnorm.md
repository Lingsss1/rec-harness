## 136. diagnostics-round9-r9_field_splitnorm

- 假设：Measure actual Query output reliance in split-norm and independently trained history controls
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T17:56:27.641691+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round9-r9_field_splitnorm",
  "parent_experiment_id": "r9_field_splitnorm-s20262027-20260906T173725574982Z",
  "family": "fixed_seed_feedback_sharing_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure actual Query output reliance in split-norm and independently trained history controls",
  "change": "Query Zero/Shuffle, extra attention off and inference-depth interventions",
  "budget": {
    "training_updates": 0,
    "training_seed": 20262027,
    "new_training_seed_values": []
  },
  "decision": "provisional",
  "confidence": "fixed_checkpoint_subset",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "checkpoint unchanged; no test evaluation"
    ]
  },
  "runtime_sec": 41.205081693828106,
  "peak_vram_mb": 2083.08935546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round9/r9_field_splitnorm/summary.json",
    "workspace/qformer-loop/reports/round9/r9_field_splitnorm/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round9/r9_field_splitnorm/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
