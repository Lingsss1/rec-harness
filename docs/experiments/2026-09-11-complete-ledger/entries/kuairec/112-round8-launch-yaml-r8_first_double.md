## 112. round8-launch-yaml-r8_first_double

- 假设：Launch the frozen structural configuration
- 判定：crash
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T17:05:39.572317+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "round8-launch-yaml-r8_first_double",
  "family": "r8_first_double",
  "decision": "crash",
  "hypothesis": "Launch the frozen structural configuration",
  "change": "YAML append reused an existing anchor name",
  "budget": {
    "training_updates": 0,
    "seeds": [
      20262027
    ]
  },
  "confidence": "execution_failure",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round8/training-launch-failed/r8_first_double-s20262027.log",
    "workspace/qformer-loop/reports/round8/launch-failed-model_config.yaml"
  ],
  "superseded_by_family": "r8_first_double"
}
```

</details>
