## 85. diagnostics-round5-attn_full

- 假设：Check recurrence and extra attention reliance after compact/attention retraining
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "diagnostics-round5-attn_full",
  "parent_experiment_id": "attn_full-s20262027-20260906T150445276311Z",
  "timestamp": "2026-09-06T15:37:46.076701+00:00",
  "code_id": "b2d3fb3949e1b0a6a0f3c7ee09326744ce8ab497dc8f88434b5edc31e1c2389c",
  "dataset_id": "kuairec-loop-v1",
  "family": "compact_recurrence_diagnostics",
  "hypothesis": "Check recurrence and extra attention reliance after compact/attention retraining",
  "change": "same checkpoint depth and Query interventions; extra attention off where present; shared mixer state hooks",
  "decision": "provisional",
  "confidence": "provisional",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "same fixed validation subset; no optimizer step or test evaluation"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round5/attn_full/summary.json",
    "workspace/qformer-loop/reports/round5/attn_full/toolkit-perturbations/perturbation_report.json",
    "workspace/qformer-loop/reports/round5/attn_full/toolkit-hooks/hook_report.json"
  ],
  "runtime_sec": 49.744571566581726,
  "peak_vram_mb": 2885.98095703125
}
```

</details>
