## 47. taobao-depth-training-mixed123-depth-curve

- 假设：Mixed depths reduce deterioration at greater recurrence depth
- 判定：Improved depth robustness but lower fixed-R2 AUC
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T05:05:08.817960+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-depth-training-mixed123-depth-curve",
  "family": "fixed_checkpoint_depth_diagnostic",
  "hypothesis": "Mixed depths reduce deterioration at greater recurrence depth",
  "change": "Frozen mixed123 checkpoint evaluated at depths1/2/3/4; only4 unseen in training",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "macro_auc_by_depth": {
      "1": 0.825977728690046,
      "2": 0.8264726829772377,
      "3": 0.8264130165451168,
      "4": 0.8259148748088244
    },
    "native_max_prediction_difference": 0
  },
  "decision": "Improved depth robustness but lower fixed-R2 AUC",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-training/reports/depth-mixed123/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-training/DEPTH.md"
  ]
}
```

</details>
