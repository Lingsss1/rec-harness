## 12. taobao-round2-paired-bootstrap

- 假设：Quantify best widened checkpoint gain
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T11:02:29.662169+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-round2-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Quantify best widened checkpoint gain",
  "change": "500 paired user resamples",
  "budget": {
    "training_updates": 0,
    "bootstrap_seed": 20260906,
    "new_training_seeds": []
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/round2/reports/paired-bootstrap.json"
  ]
}
```

</details>
