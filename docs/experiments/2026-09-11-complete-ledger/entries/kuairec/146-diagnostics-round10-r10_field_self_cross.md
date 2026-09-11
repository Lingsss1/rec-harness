## 146. diagnostics-round10-r10_field_self_cross

- 假设：Measure actual Query output reliance when repeated blocks execute selected sublayers
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T03:21:41.913630+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round10-r10_field_self_cross",
  "parent_experiment_id": "r10_field_self_cross-s20262027-20260906T181435546256Z",
  "family": "fixed_seed_selective_sublayer_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure actual Query output reliance when repeated blocks execute selected sublayers",
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
  "runtime_sec": 40.73712758719921,
  "peak_vram_mb": 2038.02490234375,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round10/r10_field_self_cross/summary.json",
    "workspace/qformer-loop/reports/round10/r10_field_self_cross/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round10/r10_field_self_cross/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
