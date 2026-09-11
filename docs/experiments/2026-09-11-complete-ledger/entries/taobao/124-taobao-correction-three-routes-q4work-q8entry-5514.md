## 124. taobao-correction-three-routes-q4work-q8entry-5514

- 假设：Separate retrieval, interaction and residual dependence of learned second-work input corrections
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:27:07.851662+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-correction-three-routes-q4work-q8entry-5514",
  "family": "fixed_checkpoint_query_route_probe",
  "hypothesis": "Separate retrieval, interaction and residual dependence of learned second-work input corrections",
  "change": "Identity/zero/shuffle and correction-route ablations on two fixed checkpoints",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "scope": "Fixed 5514-row stratified subset, 4647 selected users, CPU FP32, IPW BCE; paired bootstrap conditional on checkpoint and subset, not sampling-design uncertainty or training-seed variance. Multiple interventions and candidate selection uncorrected. No test evaluation.",
    "findings": [
      "q4_work: removing retrieval correction harms fav and buy while removing residual correction improves fav and buy, with conditional intervals excluding zero. Interaction removal remains inconclusive.",
      "q8_entry: removing retrieval correction improves buy, so q4 retrieval dependence does not replicate here; removing residual correction also improves buy.",
      "Zeroing the entire correction improves buy in both selected checkpoints. Shuffle intervals cross zero for all four tasks in both checkpoints. This does not support a general beneficial sample-specific correction claim."
    ]
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/interpretation.json",
    "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/q4-work/toolkit/perturbation_report.json",
    "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/q8-entry/toolkit/perturbation_report.json"
  ]
}
```

</details>
