## 32. round2-diagnostic-bce-autocast

- 假设：Export native-route activation gradients
- 判定：crash
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.005086+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": false,
    "violations": [
      "BCE executed inside autocast"
    ],
    "notes": [
      "fixed by FP32 BCE outside autocast; successful rerun is diagnostics-round2-middle2-v2"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "round2-diagnostic-bce-autocast",
  "family": "diagnostic_export",
  "hypothesis": "Export native-route activation gradients",
  "change": "initial export attempt",
  "decision": "crash",
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2-middle2.log"
  ]
}
```

</details>
