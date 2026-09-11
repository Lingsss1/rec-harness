## 160. probe-round11-r11_history_gate-conditioning

- 假设：Measure learned repeat conditioning reliance without retraining
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T03:46:27.713841+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "probe-round11-r11_history_gate-conditioning",
  "parent_experiment_id": "r11_history_gate-s20262027-20260907T032608594282Z",
  "family": "fixed_checkpoint_condition_neutralization",
  "code_id": "53154c9dad7571e90cc05fdbce8a922941e40f6e905c7a9807706fe15988177e",
  "hypothesis": "Measure learned repeat conditioning reliance without retraining",
  "change": "Neutralize learned field/history conditioning separately and jointly",
  "budget": {
    "training_updates": 0,
    "training_seed": 20262027,
    "new_training_seed_values": []
  },
  "decision": "provisional",
  "confidence": "fixed_checkpoint_subset",
  "runtime_sec": 3.747044526040554,
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "parameters restored; checkpoint unchanged; no test evaluation"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round11/r11_history_gate-conditioning/result.json"
  ]
}
```

</details>
