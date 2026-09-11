## 23. routes-baseline-s20262027

- 假设：Removing train/eval query mismatch improves validation quality
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.003542+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "test not evaluated"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "routes-baseline-s20262027",
  "parent_experiment_id": "baseline-s20262027-20260906T112348253214Z",
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
      "value": 0.8699734166101216,
      "delta": 0.0004903207626300032,
      "reference": "same checkpoint native route"
    },
    "guardrails": {
      "logloss": 0.1894329521538222,
      "native_logloss": 0.1900833987939914
    }
  },
  "all_route_metrics": {
    "native": {
      "logloss": 0.1900833987939914,
      "AUC": 0.8694830958474916,
      "UAUC": 0.787044954674599,
      "gAUC": 0.799239771256098,
      "NDCG(10)": 0.40624328421898004
    },
    "q3_half": {
      "logloss": 0.1889174655518989,
      "AUC": 0.8698074503150168,
      "UAUC": 0.7866713477666737,
      "gAUC": 0.798863808333554,
      "NDCG(10)": 0.4045911605678066
    },
    "all_queries": {
      "logloss": 0.1894329521538222,
      "AUC": 0.8699734166101216,
      "UAUC": 0.7863618680476449,
      "gAUC": 0.7986001871297632,
      "NDCG(10)": 0.40212930460754337
    }
  },
  "decision": "provisional",
  "runtime_sec": 61.6642125621438,
  "peak_vram_mb": 5152.185546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2/routes/baseline-s20262027/result.json"
  ],
  "diagnostics": [
    {
      "kind": "full_validation_paired_route",
      "path": "workspace/qformer-loop/reports/round2/routes/baseline-s20262027/result.json"
    }
  ]
}
```

</details>
