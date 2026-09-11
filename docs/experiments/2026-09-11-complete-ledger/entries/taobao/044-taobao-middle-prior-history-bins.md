## 44. taobao-middle-prior-history-bins

- 假设：Recurrence benefits may depend on raw history length
- 判定：No evidence that longest histories benefit most; descriptive only
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T03:56:10.798589+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-middle-prior-history-bins",
  "family": "fixed_checkpoint_history_diagnostic",
  "hypothesis": "Recurrence benefits may depend on raw history length",
  "change": "Fixed bins 0-9,10-29,30-59,>=60 on saved validation predictions",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "delta_macro_auc_by_bin": [
      0.0032245900267859895,
      0.0004401339098651924,
      0.0004957415201107818,
      -0.0008169462917205772
    ]
  },
  "decision": "No evidence that longest histories benefit most; descriptive only",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-training/PRIOR_HISTORY.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-training/reports/prior-history-diagnostic.json"
  ]
}
```

</details>
