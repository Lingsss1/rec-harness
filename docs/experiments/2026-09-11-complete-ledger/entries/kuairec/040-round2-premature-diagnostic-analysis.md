## 40. round2-premature-diagnostic-analysis

- 假设：Analyze newly exported checkpoint arrays
- 判定：invalid
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "round2-premature-diagnostic-analysis",
  "timestamp": "2026-09-06T13:55:02.209556+00:00",
  "decision": "invalid",
  "family": "diagnostic_analysis",
  "hypothesis": "Analyze newly exported checkpoint arrays",
  "change": "toolkit invoked before exporter completed",
  "validity": {
    "valid": false,
    "violations": [
      "exporter still writing arrays; required hooks.json/states.npz not present"
    ],
    "notes": [
      "does not invalidate the model or completed rerun"
    ]
  },
  "artifact_paths": [
    "reports/round2/toolkit-post-training-premature-status.json"
  ],
  "next_question": "Require export summary.json before toolkit invocation"
}
```

</details>
