## 108. diagnostics-round7-r7_small_attention_first

- 假设：Check first-pass attention and moved norm at the actual final Query output
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "diagnostics-round7-r7_small_attention_first",
  "parent_experiment_id": "r7_small_attention_first-s20262027-20260906T163655782605Z",
  "timestamp": "2026-09-06T16:50:57.272950+00:00",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "dataset_id": "kuairec-loop-v1",
  "family": "fixed_seed_memory_interface_diagnostics",
  "hypothesis": "Check first-pass attention and moved norm at the actual final Query output",
  "change": "Frozen checkpoint Query Zero/Shuffle, depth and extra-attention-off probes",
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
      "fixed diagnostic subset; checkpoint unchanged; no test evaluation"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round7/r7_small_attention_first/summary.json",
    "workspace/qformer-loop/reports/round7/r7_small_attention_first/toolkit-perturbations/perturbation_report.json",
    "workspace/qformer-loop/reports/round7/r7_small_attention_first/toolkit-hooks/hook_report.json"
  ],
  "runtime_sec": 41.35688207298517,
  "peak_vram_mb": 2097.08740234375
}
```

</details>
