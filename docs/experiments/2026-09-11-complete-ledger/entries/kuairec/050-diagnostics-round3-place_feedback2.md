## 50. diagnostics-round3-place_feedback2

- 假设：Check the new field/history call graph with shared existing probes
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "diagnostics-round3-place_feedback2",
  "parent_experiment_id": "place_feedback2-s20262027-20260906T135252383734Z",
  "timestamp": "2026-09-06T14:24:32.295054+00:00",
  "code_id": "a910db8bc6cdfd4a4b5219e0cda03dfac2a462609cec04aa407cd6c4cc7c6198",
  "dataset_id": "kuairec-loop-v1",
  "family": "placement_diagnostics",
  "hypothesis": "Check the new field/history call graph with shared existing probes",
  "change": "export final norm and individual field/sequence block hooks; paired equal-width Query Zero/Shuffle",
  "decision": "provisional",
  "confidence": "provisional",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "same frozen user-stratified validation subset; no test evaluation"
    ]
  },
  "artifact_paths": [
    "qformer-loop/reports/round3/place_feedback2/summary.json",
    "qformer-loop/reports/round3/place_feedback2/toolkit-hooks/hook_report.json",
    "qformer-loop/reports/round3/place_feedback2/toolkit-perturbations/perturbation_report.json"
  ],
  "runtime_sec": 45.42711931467056,
  "peak_vram_mb": 2777.08056640625
}
```

</details>
