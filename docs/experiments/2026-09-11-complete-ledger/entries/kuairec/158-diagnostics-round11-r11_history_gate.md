## 158. diagnostics-round11-r11_history_gate

- 假设：Measure Query and recurrence reliance in initially neutral conditioning models
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T03:46:27.707799+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round11-r11_history_gate",
  "parent_experiment_id": "r11_history_gate-s20262027-20260907T032608594282Z",
  "family": "fixed_seed_conditioning_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure Query and recurrence reliance in initially neutral conditioning models",
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
  "runtime_sec": 40.02630262076855,
  "peak_vram_mb": 2082.96630859375,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round11/r11_history_gate/summary.json",
    "workspace/qformer-loop/reports/round11/r11_history_gate/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round11/r11_history_gate/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
