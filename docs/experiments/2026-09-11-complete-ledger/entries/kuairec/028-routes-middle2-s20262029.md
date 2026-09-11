## 28. routes-middle2-s20262029

- 假设：Removing train/eval query mismatch improves validation quality
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.004401+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "test not evaluated"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "routes-middle2-s20262029",
  "parent_experiment_id": "middle2-s20262029-20260906T121647530746Z",
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
      "value": 0.8686460110468807,
      "delta": -8.041393531799113e-05,
      "reference": "same checkpoint native route"
    },
    "guardrails": {
      "logloss": 0.18956905312026695,
      "native_logloss": 0.19791504445468114
    }
  },
  "all_route_metrics": {
    "native": {
      "logloss": 0.19791504445468114,
      "AUC": 0.8687264249821987,
      "UAUC": 0.7885984850343909,
      "gAUC": 0.8008145512702152,
      "NDCG(10)": 0.40218507238254
    },
    "q3_half": {
      "logloss": 0.19058143560470597,
      "AUC": 0.8687048108614993,
      "UAUC": 0.7890605086219511,
      "gAUC": 0.8012406777959209,
      "NDCG(10)": 0.4031401545216601
    },
    "all_queries": {
      "logloss": 0.18956905312026695,
      "AUC": 0.8686460110468807,
      "UAUC": 0.7892864197280802,
      "gAUC": 0.8015273887069919,
      "NDCG(10)": 0.40294509944552725
    }
  },
  "decision": "provisional",
  "runtime_sec": 71.46386430412531,
  "peak_vram_mb": 5216.185546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2/routes/middle2-s20262029/result.json"
  ],
  "diagnostics": [
    {
      "kind": "full_validation_paired_route",
      "path": "workspace/qformer-loop/reports/round2/routes/middle2-s20262029/result.json"
    }
  ]
}
```

</details>
