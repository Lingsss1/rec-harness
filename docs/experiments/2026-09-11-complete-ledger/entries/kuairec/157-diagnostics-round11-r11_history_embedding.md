## 157. diagnostics-round11-r11_history_embedding

- 假设：Measure Query and recurrence reliance in initially neutral conditioning models
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T03:46:27.704861+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round11-r11_history_embedding",
  "parent_experiment_id": "r11_history_embedding-s20262027-20260907T032608681208Z",
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
  "runtime_sec": 40.259452909231186,
  "peak_vram_mb": 2090.08837890625,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round11/r11_history_embedding/summary.json",
    "workspace/qformer-loop/reports/round11/r11_history_embedding/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round11/r11_history_embedding/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
