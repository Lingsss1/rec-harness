## 147. diagnostics-round10-r10_history_cross_ffn

- 假设：Measure actual Query output reliance when repeated blocks execute selected sublayers
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T03:21:41.915993+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round10-r10_history_cross_ffn",
  "parent_experiment_id": "r10_history_cross_ffn-s20262027-20260906T181435553962Z",
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
  "runtime_sec": 40.62165704369545,
  "peak_vram_mb": 2017.39990234375,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round10/r10_history_cross_ffn/summary.json",
    "workspace/qformer-loop/reports/round10/r10_history_cross_ffn/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round10/r10_history_cross_ffn/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
