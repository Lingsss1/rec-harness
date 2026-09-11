## 179. diagnostics-round13-r13_query_between_zero

- 假设：Measure Query output and depth reliance after history source updates
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T04:49:44.102199+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round13-r13_query_between_zero",
  "parent_experiment_id": "r13_query_between_zero-s20262027-20260907T042433943622Z",
  "family": "fixed_seed_source_update_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure Query output and depth reliance after history source updates",
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
  "runtime_sec": 44.67018497735262,
  "peak_vram_mb": 2565.57568359375,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round13/r13_query_between_zero/summary.json",
    "workspace/qformer-loop/reports/round13/r13_query_between_zero/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round13/r13_query_between_zero/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
