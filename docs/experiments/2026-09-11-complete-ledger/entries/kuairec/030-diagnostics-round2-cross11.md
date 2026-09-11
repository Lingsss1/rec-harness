## 30. diagnostics-round2-cross11

- 假设：Measure query reliance, finite gradients and recurrence geometry before structural edits
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.004800+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "test not evaluated"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "diagnostics-round2-cross11",
  "parent_experiment_id": "baseline-s20262027-20260906T112348253214Z",
  "code_id": "710ebecdf3b722b35cd51748fe9a31485d92b3c20e447b9bc61e810ef8755f66",
  "family": "diagnostics",
  "hypothesis": "Measure query reliance, finite gradients and recurrence geometry before structural edits",
  "change": "same-weight query Zero/Shuffle, depth perturbation and hook export",
  "decision": "provisional",
  "runtime_sec": 38.25068625062704,
  "peak_vram_mb": 2510.4814453125,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2/cross11/summary.json",
    "workspace/qformer-loop/reports/round2/cross11/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round2/cross11/toolkit-perturbations/perturbation_report.json"
  ],
  "note": "subset diagnostics only; no incumbent selection from these scores"
}
```

</details>
