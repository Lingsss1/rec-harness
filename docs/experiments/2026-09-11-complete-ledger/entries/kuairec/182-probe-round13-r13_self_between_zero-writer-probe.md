## 182. probe-round13-r13_self_between_zero-writer-probe

- 假设：Measure reliance on learned history updates and matching Query feedback
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T04:49:44.113364+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "probe-round13-r13_self_between_zero-writer-probe",
  "parent_experiment_id": "r13_self_between_zero-s20262027-20260907T042433943647Z",
  "family": "fixed_checkpoint_history_writer_probe",
  "code_id": "4fa3a88cc61ce427d9958f7f5bfd22e981a2f11d656ce4429de312d38ed7cd57",
  "hypothesis": "Measure reliance on learned history updates and matching Query feedback",
  "change": "Bypass writer; deterministically roll feedback Query within a batch when applicable",
  "budget": {
    "training_updates": 0,
    "training_seed": 20262027,
    "new_training_seed_values": []
  },
  "decision": "provisional",
  "confidence": "fixed_checkpoint_subset",
  "runtime_sec": 3.447512835264206,
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "checkpoint unchanged; no parameter edits; no test evaluation"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round13/r13_self_between_zero-writer-probe/result.json"
  ]
}
```

</details>
