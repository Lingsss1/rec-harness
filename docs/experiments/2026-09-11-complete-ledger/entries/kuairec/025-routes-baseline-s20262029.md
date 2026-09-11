## 25. routes-baseline-s20262029

- 假设：Removing train/eval query mismatch improves validation quality
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.003909+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "test not evaluated"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "routes-baseline-s20262029",
  "parent_experiment_id": "baseline-s20262029-20260906T112713728411Z",
  "code_id": "66f468d0f439e8c302cf04c65d746c36d5381cb7fb9577b71e160efc30a59b5a",
  "family": "query_routing",
  "hypothesis": "Removing train/eval query mismatch improves validation quality",
  "change": "Q3 eval scale 0 -> 0.5 / 1, no weight update",
  "budget": {
    "training_updates": 0,
    "seeds": [
      20262029
    ],
    "evaluation_routes": 3
  },
  "metrics": {
    "primary": {
      "name": "validation_AUC_all_queries",
      "value": 0.86828197053844,
      "delta": 0.0003781723090752198,
      "reference": "same checkpoint native route"
    },
    "guardrails": {
      "logloss": 0.19027100510930803,
      "native_logloss": 0.19450827060502757
    }
  },
  "all_route_metrics": {
    "native": {
      "logloss": 0.19450827060502757,
      "AUC": 0.8679037982293648,
      "UAUC": 0.7881302453925081,
      "gAUC": 0.8009165947008892,
      "NDCG(10)": 0.40587375487173927
    },
    "q3_half": {
      "logloss": 0.19041137102680025,
      "AUC": 0.8681563209548276,
      "UAUC": 0.7882930343079069,
      "gAUC": 0.8010689807454254,
      "NDCG(10)": 0.40643926564621746
    },
    "all_queries": {
      "logloss": 0.19027100510930803,
      "AUC": 0.86828197053844,
      "UAUC": 0.7884917927304682,
      "gAUC": 0.8012409287547017,
      "NDCG(10)": 0.4070366518073264
    }
  },
  "decision": "provisional",
  "runtime_sec": 63.00751442462206,
  "peak_vram_mb": 5152.185546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2/routes/baseline-s20262029/result.json"
  ],
  "diagnostics": [
    {
      "kind": "full_validation_paired_route",
      "path": "workspace/qformer-loop/reports/round2/routes/baseline-s20262029/result.json"
    }
  ]
}
```

</details>
