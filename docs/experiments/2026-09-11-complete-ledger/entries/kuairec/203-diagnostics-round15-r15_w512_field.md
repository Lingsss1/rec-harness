## 203. diagnostics-round15-r15_w512_field

- 假设：Measure Query and recurrence reliance after dense FFN width changes
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T05:40:04.930358+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round15-r15_w512_field",
  "parent_experiment_id": "r15_w512_field-s20262027-20260907T052153858349Z",
  "family": "fixed_seed_width_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure Query and recurrence reliance after dense FFN width changes",
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
  "runtime_sec": 41.18094394356012,
  "peak_vram_mb": 2064.58935546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round15/r15_w512_field/summary.json",
    "workspace/qformer-loop/reports/round15/r15_w512_field/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round15/r15_w512_field/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
