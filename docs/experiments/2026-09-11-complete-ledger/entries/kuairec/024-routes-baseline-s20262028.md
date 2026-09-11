## 24. routes-baseline-s20262028

- 假设：Removing train/eval query mismatch improves validation quality
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.003739+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "test not evaluated"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "routes-baseline-s20262028",
  "parent_experiment_id": "baseline-s20262028-20260906T112713704038Z",
  "code_id": "66f468d0f439e8c302cf04c65d746c36d5381cb7fb9577b71e160efc30a59b5a",
  "family": "query_routing",
  "hypothesis": "Removing train/eval query mismatch improves validation quality",
  "change": "Q3 eval scale 0 -> 0.5 / 1, no weight update",
  "budget": {
    "training_updates": 0,
    "seeds": [
      20262028
    ],
    "evaluation_routes": 3
  },
  "metrics": {
    "primary": {
      "name": "validation_AUC_all_queries",
      "value": 0.8708279342804846,
      "delta": 0.0002877513818125843,
      "reference": "same checkpoint native route"
    },
    "guardrails": {
      "logloss": 0.18874293464019073,
      "native_logloss": 0.18915727542425867
    }
  },
  "all_route_metrics": {
    "native": {
      "logloss": 0.18915727542425867,
      "AUC": 0.870540182898672,
      "UAUC": 0.7904060541982754,
      "gAUC": 0.8016584711480392,
      "NDCG(10)": 0.4100094471094947
    },
    "q3_half": {
      "logloss": 0.18877889474480325,
      "AUC": 0.8707151495808146,
      "UAUC": 0.7897159557390945,
      "gAUC": 0.8010376938429986,
      "NDCG(10)": 0.40966624623494546
    },
    "all_queries": {
      "logloss": 0.18874293464019073,
      "AUC": 0.8708279342804846,
      "UAUC": 0.7891687001511011,
      "gAUC": 0.8005020266777836,
      "NDCG(10)": 0.4089483220842704
    }
  },
  "decision": "provisional",
  "runtime_sec": 64.03322860598564,
  "peak_vram_mb": 5152.185546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2/routes/baseline-s20262028/result.json"
  ],
  "diagnostics": [
    {
      "kind": "full_validation_paired_route",
      "path": "workspace/qformer-loop/reports/round2/routes/baseline-s20262028/result.json"
    }
  ]
}
```

</details>
