## 27. routes-middle2-s20262028

- 假设：Removing train/eval query mismatch improves validation quality
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-06T13:28:12.004239+00:00",
  "dataset_id": "kuairec-loop-v1",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "test not evaluated"
    ]
  },
  "confidence": "provisional",
  "experiment_id": "routes-middle2-s20262028",
  "parent_experiment_id": "middle2-s20262028-20260906T121647640962Z",
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
      "value": 0.8692897691125018,
      "delta": 0.0007590030082039734,
      "reference": "same checkpoint native route"
    },
    "guardrails": {
      "logloss": 0.18956021900561082,
      "native_logloss": 0.18987440936561303
    }
  },
  "all_route_metrics": {
    "native": {
      "logloss": 0.18987440936561303,
      "AUC": 0.8685307661042978,
      "UAUC": 0.7887072104817362,
      "gAUC": 0.800687640805724,
      "NDCG(10)": 0.4064630943658545
    },
    "q3_half": {
      "logloss": 0.18906610736985813,
      "AUC": 0.8689521481185943,
      "UAUC": 0.7893463302051071,
      "gAUC": 0.8012230630164584,
      "NDCG(10)": 0.4074332204407207
    },
    "all_queries": {
      "logloss": 0.18956021900561082,
      "AUC": 0.8692897691125018,
      "UAUC": 0.7898319485374516,
      "gAUC": 0.8016849363198547,
      "NDCG(10)": 0.4074616989137068
    }
  },
  "decision": "provisional",
  "runtime_sec": 72.29246528446674,
  "peak_vram_mb": 5216.185546875,
  "artifact_paths": [
    "workspace/qformer-loop/reports/round2/routes/middle2-s20262028/result.json"
  ],
  "diagnostics": [
    {
      "kind": "full_validation_paired_route",
      "path": "workspace/qformer-loop/reports/round2/routes/middle2-s20262028/result.json"
    }
  ]
}
```

</details>
