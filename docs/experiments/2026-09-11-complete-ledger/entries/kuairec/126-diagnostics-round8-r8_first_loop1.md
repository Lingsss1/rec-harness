## 126. diagnostics-round8-r8_first_loop1

- 假设：Measure reliance at the actual output and compare depth perturbations with complete depth retraining
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T17:26:06.789590+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round8-r8_first_loop1",
  "parent_experiment_id": "r8_first_loop1-s20262027-20260906T170409366470Z",
  "family": "fixed_seed_refinement_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure reliance at the actual output and compare depth perturbations with complete depth retraining",
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
  "runtime_sec": 34.17644868046045,
  "peak_vram_mb": 1325.01904296875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round8/r8_first_loop1/summary.json",
    "workspace/qformer-loop/reports/round8/r8_first_loop1/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round8/r8_first_loop1/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
