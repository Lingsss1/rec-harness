## 205. round16-synthetic-fixture-failures

- 假设：Verify real train_step before launching variable-depth experiments
- 判定：crashed
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T07:32:19.728586+00:00",
  "dataset_id": "kuairec-loop-v1",
  "experiment_id": "round16-synthetic-fixture-failures",
  "family": "correctness_fixture_failure",
  "hypothesis": "Verify real train_step before launching variable-depth experiments",
  "change": "Correct forward-only fixture labels and feature-map mismatch; two CPU and one GPU failed invocations preserved",
  "budget": {
    "training_runs": 0,
    "new_training_seed_values": []
  },
  "decision": "crashed",
  "confidence": "observed_test_failure_corrected",
  "validity": {
    "valid": false,
    "violations": [
      "synthetic fixture missing labels or containing undeclared seq_len"
    ],
    "notes": [
      "fixed before full training; final CPU/GPU tests passed"
    ]
  },
  "artifact_paths": [
    "workspace/qformer-loop/reports/round16/failures.json",
    "workspace/qformer-loop/reports/round16/cpu-tests-fixture-failure.txt",
    "workspace/qformer-loop/reports/round16/cpu-tests-seqlen-fixture-failure.txt",
    "workspace/qformer-loop/reports/round16/gpu-tests-seqlen-fixture-failure.txt"
  ]
}
```

</details>
