## 48. taobao-depth-training-supervised12-depth-curve

- 假设：Intermediate supervision reduces deterioration at greater recurrence depth
- 判定：Less depth deterioration but lower peak AUC
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T05:07:07.056399+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-depth-training-supervised12-depth-curve",
  "family": "fixed_checkpoint_depth_diagnostic",
  "hypothesis": "Intermediate supervision reduces deterioration at greater recurrence depth",
  "change": "Frozen supervised12 checkpoint at1/2/3/4;3/4 unseen",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "macro_auc_by_depth": {
      "1": 0.8249836113180737,
      "2": 0.8254845285900987,
      "3": 0.8251059201766087,
      "4": 0.8235831741058937
    },
    "native_max_prediction_difference": 0
  },
  "decision": "Less depth deterioration but lower peak AUC",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-training/reports/depth-supervised12/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-training/DEPTH.md"
  ]
}
```

</details>
