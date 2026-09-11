## 125. taobao-q4-difference-near-zero-psgd-reproduction

- 假设：Distinguish actual zero gradient from an approximate-zero rejection
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T18:27:07.853912+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-q4-difference-near-zero-psgd-reproduction",
  "family": "optimizer_failure_diagnostic",
  "hypothesis": "Distinguish actual zero gradient from an approximate-zero rejection",
  "change": "Same seed and original two-step smoke with read-only gradient/scale instrumentation",
  "budget": {
    "training_updates": 0,
    "smoke_steps_completed": 1,
    "new_training_seeds": []
  },
  "metrics": {
    "grad_elements_nonzero": 6144,
    "clipped_a_gradient_max_abs": 8.77190586834331e-09,
    "finite_default_automatic_scale": 140.19190910151133,
    "fp32_fourth_moment_underflow": false
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/psgd-failure-q4-difference/attempt-20260908T181924351492Z/result.json",
    "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/psgd-failure-q4-difference/attempt-20260908T181924351492Z/observations.jsonl"
  ]
}
```

</details>
