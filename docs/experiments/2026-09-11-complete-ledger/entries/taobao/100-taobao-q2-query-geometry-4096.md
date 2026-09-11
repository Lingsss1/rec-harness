## 100. taobao-q2-query-geometry-4096

- 假设：Second query may differ in geometry but causal utility is unknown
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T16:00:32.565758+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-q2-query-geometry-4096",
  "family": "query_conditioning_diagnostic",
  "hypothesis": "Second query may differ in geometry but causal utility is unknown",
  "change": "Export q1/q2 and run four stable rec-harness geometry analyses",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "measurements": {
      "q1_mean": {
        "effective_rank": 52.5980855610032,
        "top1_variance_ratio": 0.0999913748786532,
        "centered_to_raw_energy_ratio": 0.20457801566366285,
        "random_pair_cosine": 0.7987646649124985
      },
      "q2_mean": {
        "effective_rank": 49.52891277685158,
        "top1_variance_ratio": 0.14986175833661217,
        "centered_to_raw_energy_ratio": 0.19691388471143081,
        "random_pair_cosine": 0.8090898524895148
      },
      "q1_slot0": {
        "effective_rank": 56.138983126626194,
        "top1_variance_ratio": 0.0993177037774296,
        "centered_to_raw_energy_ratio": 0.23590496987164464,
        "random_pair_cosine": 0.7647655216373164
      },
      "q2_slot0": {
        "effective_rank": 53.373652390051696,
        "top1_variance_ratio": 0.14683708584013083,
        "centered_to_raw_energy_ratio": 0.22785155500151172,
        "random_pair_cosine": 0.7777417796509472
      }
    },
    "scope": {
      "samples": 4096,
      "users": 375,
      "task_positives": {
        "is_click": 193,
        "cart": 11,
        "fav": 11,
        "buy": 4
      },
      "task_negatives": {
        "is_click": 3903,
        "cart": 4085,
        "fav": 4085,
        "buy": 4092
      },
      "query_shape": [
        4096,
        4,
        384
      ],
      "relative_change": 0.5330536365509033,
      "mean_paired_cosine": 0.9093810319900513,
      "finite": true,
      "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-context-read/runs/ctx_q4-s20262027-20260908T133705147096Z",
      "manifest_sha256": "487a2481d19bd689e852a5fff647669a466fb7b2fb45e991402bfc021d16a9a6",
      "checkpoint_sha256": "12d438880b0eb8882328981fa79734143fe3857c7ef490a1e31c291135080104",
      "exporter_sha256": "43f3d4e31ca35298d0a0ddad6ff747252e65daae53644f538c7c71ab6fc35a90",
      "subset": "first 4096 validation rows in released order; diagnostic only",
      "precision": "CPU float32; not comparable to BF16 full validation metrics",
      "training_updates": 0,
      "test_evaluated": false,
      "runtime_seconds": 9.34495746344328
    },
    "interpretation": "q2 changes but becomes slightly more concentrated; no demonstrated retrieval quality defect or causal bottleneck",
    "next_action": "stratified same-weight entry/work shuffle and zero at second query only",
    "task_sufficiency": "buy4/cart11/fav11 positives insufficient for sparse-task ranking"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/query-conditioning/reports/geometry-interpretation.json",
    "workspace/qformer-loop/taobao-transfer/query-conditioning/RESULTS.md"
  ]
}
```

</details>
