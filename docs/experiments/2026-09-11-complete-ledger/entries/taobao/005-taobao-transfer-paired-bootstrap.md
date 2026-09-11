## 5. taobao-transfer-paired-bootstrap

- 假设：Quantify r7 transfer deltas conditional on frozen checkpoints
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T09:30:33.497314+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-transfer-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify r7 transfer deltas conditional on frozen checkpoints",
  "change": "500 paired user resamples with shared user weights across four tasks",
  "budget": {
    "training_updates": 0,
    "bootstrap_seed": 20260906,
    "new_training_seeds": []
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/reports/paired-bootstrap.json"
  ]
}
```

</details>
