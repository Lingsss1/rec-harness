## 148. round11-diagnostic-launch-failure-1

- 假设：Run fixed-checkpoint conditioning probe
- 判定：crashed
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T03:46:27.677484+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "round11-diagnostic-launch-failure-1",
  "family": "diagnostic_runtime_failure",
  "hypothesis": "Run fixed-checkpoint conditioning probe",
  "change": "initialize project dependency paths by importing diagnostic_round2 first",
  "decision": "crashed",
  "confidence": "observed_runtime_error",
  "budget": {
    "training_updates": 0,
    "new_training_seed_values": []
  },
  "validity": {
    "valid": false,
    "violations": [
      "sklearn imported before diagnostic_round2 configured project runtime/deps"
    ],
    "notes": [
      "failed before checkpoint loading; corrected probe subsequently completed"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round11/failures.json",
    "workspace/qformer-loop/reports/round11/conditioning-probe-import-failure.log"
  ]
}
```

</details>
