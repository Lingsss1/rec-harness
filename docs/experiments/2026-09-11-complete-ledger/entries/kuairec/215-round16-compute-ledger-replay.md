## 215. round16-compute-ledger-replay

- 假设：Disambiguate raw historical sequence lengths from truncated valid/padded attention positions
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T07:32:19.772027+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "round16-compute-ledger-replay",
  "family": "read_only_training_batch_replay",
  "code_id": "dfa26dc7e4f2c57e1c570abdae5cb9cb198b32367ca801ff9b4bcb3230948ca9",
  "hypothesis": "Disambiguate raw historical sequence lengths from truncated valid/padded attention positions",
  "change": "Replay original epoch0 loader once; confirm original sample/raw-length ledgers then count actual masks",
  "budget": {
    "training_updates": 0,
    "new_training_seed_values": []
  },
  "decision": "provisional",
  "confidence": "exact_recorded_ledger_match",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "read-only; no new model training or test evaluation"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round16/replayed-compute.json"
  ]
}
```

</details>
