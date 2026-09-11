## 64. diagnostics-round4-source-probe

- 假设：Check feedback source dependence and recurrent query use
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "diagnostics-round4-source-probe",
  "parent_experiment_id": "place_feedback2-s20262027-20260906T135252383734Z",
  "timestamp": "2026-09-06T15:05:55.006462+00:00",
  "code_id": "ebddd9d7d0b2c51d8bbcfcc63eb2e07546afc80f39207001f5d48425e190ba7f",
  "dataset_id": "kuairec-loop-v1",
  "family": "feedback_diagnostics",
  "hypothesis": "Check feedback source dependence and recurrent query use",
  "change": "frozen checkpoint intervention using existing rec-harness probes",
  "decision": "provisional",
  "confidence": "provisional",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "fixed user-stratified validation subset; no test; diagnostic effects do not establish retraining benefits"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round4/source-probe/summary.json"
  ],
  "runtime_sec": 5.502126969397068,
  "peak_vram_mb": 696.20458984375
}
```

</details>
