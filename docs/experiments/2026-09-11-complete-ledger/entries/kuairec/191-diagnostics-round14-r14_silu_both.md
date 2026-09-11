## 191. diagnostics-round14-r14_silu_both

- 假设：Measure Query and recurrence reliance with gated or SiLU FFNs
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T05:10:48.025110+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round14-r14_silu_both",
  "parent_experiment_id": "r14_silu_both-s20262027-20260907T050135045692Z",
  "family": "fixed_seed_gated_ffn_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure Query and recurrence reliance with gated or SiLU FFNs",
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
  "runtime_sec": 41.18251349776983,
  "peak_vram_mb": 2101.09716796875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round14/r14_silu_both/summary.json",
    "workspace/qformer-loop/reports/round14/r14_silu_both/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round14/r14_silu_both/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
