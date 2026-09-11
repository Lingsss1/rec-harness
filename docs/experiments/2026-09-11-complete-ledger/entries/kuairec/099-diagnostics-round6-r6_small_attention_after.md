## 99. diagnostics-round6-r6_small_attention_after

- 假设：Check trained terminal feedback and compact post-history attention reliance
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "diagnostics-round6-r6_small_attention_after",
  "parent_experiment_id": "r6_small_attention_after-s20262027-20260906T160133918683Z",
  "timestamp": "2026-09-06T16:20:45.509419+00:00",
  "code_id": "b2d3fb3949e1b0a6a0f3c7ee09326744ce8ab497dc8f88434b5edc31e1c2389c",
  "dataset_id": "kuairec-loop-v1",
  "family": "fixed_seed_schedule_diagnostics",
  "hypothesis": "Check trained terminal feedback and compact post-history attention reliance",
  "change": "frozen checkpoint Query/depth interventions; attention off where present",
  "budget": {
    "training_updates": 0,
    "training_seed": 20262027,
    "new_training_seed_values": []
  },
  "decision": "provisional",
  "confidence": "single_existing_seed",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "same fixed diagnostic subset; no training or test evaluation"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round6/r6_small_attention_after/summary.json",
    "workspace/qformer-loop/reports/round6/r6_small_attention_after/toolkit-perturbations/perturbation_report.json",
    "workspace/qformer-loop/reports/round6/r6_small_attention_after/toolkit-hooks/hook_report.json"
  ],
  "runtime_sec": 41.74988625943661,
  "peak_vram_mb": 2091.09423828125
}
```

</details>
