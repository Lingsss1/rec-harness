## 170. diagnostics-round12-r12_history_afc

- 假设：Measure Query output and depth reliance with reordered complete shared blocks
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T04:06:03.548963+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round12-r12_history_afc",
  "parent_experiment_id": "r12_history_afc-s20262027-20260907T035559125014Z",
  "family": "fixed_seed_order_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure Query output and depth reliance with reordered complete shared blocks",
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
  "runtime_sec": 41.17439292371273,
  "peak_vram_mb": 2123.52490234375,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round12/r12_history_afc/summary.json",
    "workspace/qformer-loop/reports/round12/r12_history_afc/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round12/r12_history_afc/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
