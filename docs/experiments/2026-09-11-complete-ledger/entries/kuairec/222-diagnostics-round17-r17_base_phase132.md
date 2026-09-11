## 222. diagnostics-round17-r17_base_phase132

- 假设：Measure Query and inference-depth reliance after variable-depth training
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T07:51:18.820338+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round17-r17_base_phase132",
  "parent_experiment_id": "r17_base_phase132-s20262027-20260907T073714696072Z",
  "family": "fixed_seed_training_depth_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure Query and inference-depth reliance after variable-depth training",
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
  "runtime_sec": 41.4253718778491,
  "peak_vram_mb": 2097.08740234375,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round17/r17_base_phase132/summary.json",
    "workspace/qformer-loop/reports/round17/r17_base_phase132/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round17/r17_base_phase132/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
