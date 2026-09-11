## 26. routes-middle2-s20262027

- 假设：Removing train/eval query mismatch improves validation quality
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.004074+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "test not evaluated"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "routes-middle2-s20262027",
  "parent_experiment_id": "middle2-s20262027-20260906T121257263362Z",
  "code_id": "66f468d0f439e8c302cf04c65d746c36d5381cb7fb9577b71e160efc30a59b5a",
  "family": "query_routing",
  "hypothesis": "Removing train/eval query mismatch improves validation quality",
  "change": "Q3 eval scale 0 -> 0.5 / 1, no weight update",
  "budget": {
    "training_updates": 0,
    "seeds": [
      20262027
    ],
    "evaluation_routes": 3
  },
  "metrics": {
    "primary": {
      "name": "validation_AUC_all_queries",
      "value": 0.8699935258802891,
      "delta": 0.000280445868362289,
      "reference": "same checkpoint native route"
    },
    "guardrails": {
      "logloss": 0.1895778943355105,
      "native_logloss": 0.1930701100529127
    }
  },
  "all_route_metrics": {
    "native": {
      "logloss": 0.1930701100529127,
      "AUC": 0.8697130800119268,
      "UAUC": 0.7909455311050113,
      "gAUC": 0.8022054652359927,
      "NDCG(10)": 0.4054413702340968
    },
    "q3_half": {
      "logloss": 0.1900501701661698,
      "AUC": 0.8700475933799412,
      "UAUC": 0.7897962284730009,
      "gAUC": 0.8013711489191282,
      "NDCG(10)": 0.40559456854562875
    },
    "all_queries": {
      "logloss": 0.1895778943355105,
      "AUC": 0.8699935258802891,
      "UAUC": 0.7888721927140412,
      "gAUC": 0.800601770767669,
      "NDCG(10)": 0.40636998639654404
    }
  },
  "decision": "provisional",
  "runtime_sec": 70.52465556561947,
  "peak_vram_mb": 5216.185546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2/routes/middle2-s20262027/result.json"
  ],
  "diagnostics": [
    {
      "kind": "full_validation_paired_route",
      "path": "workspace/qformer-loop/reports/round2/routes/middle2-s20262027/result.json"
    }
  ]
}
```

</details>
