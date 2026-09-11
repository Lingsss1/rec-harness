## 137. diagnostics-round9-r9_untied_clone

- 假设：Measure actual Query output reliance in split-norm and independently trained history controls
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T17:56:27.643988+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "diagnostics-round9-r9_untied_clone",
  "parent_experiment_id": "r9_untied_clone-s20262027-20260906T174430682740Z",
  "family": "fixed_seed_feedback_sharing_diagnostics",
  "code_id": "b301525f14832b6f280249b80ca263681c8688a2152028397db939fa450f22ea",
  "hypothesis": "Measure actual Query output reliance in split-norm and independently trained history controls",
  "change": "Query Zero/Shuffle, extra attention off and inference-depth interventions",
  "budget": {
    "training_updates": 0,
    "training_seed": 20262027,
    "new_training_seed_values": []
  },
  "decision": "provisional",
  "confidence": "fixed_checkpoint_subset",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "checkpoint unchanged; no test evaluation"
    ]
  },
  "runtime_sec": 41.67877929657698,
  "peak_vram_mb": 2088.490234375,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round9/r9_untied_clone/summary.json",
    "workspace/qformer-loop/reports/round9/r9_untied_clone/toolkit-hooks/hook_report.json",
    "workspace/qformer-loop/reports/round9/r9_untied_clone/toolkit-perturbations/perturbation_report.json"
  ]
}
```

</details>
