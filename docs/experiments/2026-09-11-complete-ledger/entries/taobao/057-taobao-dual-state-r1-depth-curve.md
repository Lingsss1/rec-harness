## 57. taobao-dual-state-r1-depth-curve

- 假设：Does the trained recurrence help at fixed weights?
- 判定：diagnostic
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-08T09:47:37.835617+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-dual-state-r1-depth-curve",
  "family": "fixed_checkpoint_depth_diagnostic",
  "hypothesis": "Does the trained recurrence help at fixed weights?",
  "change": "Evaluate one and two cycles on full validation; reproduce native predictions",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-dual-state/runs/dual_state_r1-s20262027-20260908T090746489340Z",
    "scores": {
      "1": {
        "validation": {
          "is_click": {
            "AUC": 0.6659401890142416,
            "logloss": 0.1825684773692603
          },
          "cart": {
            "AUC": 0.8633958433724217,
            "logloss": 0.017832093972120534
          },
          "fav": {
            "AUC": 0.8810120376512781,
            "logloss": 0.013391776227325186
          },
          "buy": {
            "AUC": 0.8961766175648992,
            "logloss": 0.006739058499943639
          }
        },
        "macro_AUC": 0.8266311719007102
      },
      "2": {
        "validation": {
          "is_click": {
            "AUC": 0.6556024746913354,
            "logloss": 0.1912323196693901
          },
          "cart": {
            "AUC": 0.8560118098478146,
            "logloss": 0.019462208975003708
          },
          "fav": {
            "AUC": 0.8588879408973733,
            "logloss": 0.01690342981149384
          },
          "buy": {
            "AUC": 0.8853757279551876,
            "logloss": 0.008548448091195803
          }
        },
        "macro_AUC": 0.8139694883479276
      }
    },
    "validation_rows": 2871767,
    "native_max_prediction_difference": 0.0,
    "checkpoint_sha256": "ad49348ace7f0115d33dc2e50d2ac3783b6118b63aa72d0cb22a615a67b2346e",
    "script_sha256": "3b1fad5e3bf1bdc443a3e069fec2eea5e2a8012057a60645ce9b56b85f52c00d",
    "manifest_sha256": "38bca3a52a490142f0dc3d05107fffbe072cbfa8bf1705ab2567bbce53896f95",
    "training_updates": 0,
    "test_evaluated": false,
    "native_depth": 1,
    "scope": "Same frozen checkpoint at one/two cycles; off-native depth is an inference ablation, not independently trained performance."
  },
  "decision": "diagnostic",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-dual-state/reports/depth-r1/result.json"
  ]
}
```

</details>
