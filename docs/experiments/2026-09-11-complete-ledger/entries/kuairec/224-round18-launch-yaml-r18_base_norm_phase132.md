## 224. round18-launch-yaml-r18_base_norm_phase132

- 假设：Launch depth-specific normalization trial
- 判定：crash
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T08:11:01.682378+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "round18-launch-yaml-r18_base_norm_phase132",
  "family": "r18_base_norm_phase132",
  "code_id": "bd201c01a076204b5159e00334d1c9dd97cf5ea84e296c094cc673145bab15eb",
  "hypothesis": "Launch depth-specific normalization trial",
  "change": "Configuration serialization failed before training",
  "budget": {
    "training_updates": 0,
    "training_seed": 20262027
  },
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "YAML duplicate anchor"
    ],
    "notes": [
      "No model constructed; original seed retried after parse/semantic checks"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round18/training-config-failure/r18_base_norm_phase132-s20262027.log",
    "workspace/qformer-loop/reports/round18/launch-failure.json"
  ]
}
```

</details>
