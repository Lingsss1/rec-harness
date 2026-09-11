## 42. taobao-middle-depth-middle-depth-curve

- 假设：Does additional recurrence improve one frozen checkpoint?
- 判定：diagnostic
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T17:02:55.489220+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-middle-depth-middle-depth-curve",
  "family": "fixed_checkpoint_depth_diagnostic",
  "parent_experiment_id": "middle_r2-s20262027-20260907T162109565214Z",
  "hypothesis": "Does additional recurrence improve one frozen checkpoint?",
  "change": "Evaluate middle depth1/2/3/4",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "decision": "diagnostic",
  "metrics": {
    "1": {
      "validation": {
        "is_click": {
          "AUC": 0.6597119815419278,
          "logloss": 0.18359743765503217
        },
        "cart": {
          "AUC": 0.86018674800029,
          "logloss": 0.01789054053240356
        },
        "fav": {
          "AUC": 0.8808096928396866,
          "logloss": 0.013460140410715676
        },
        "buy": {
          "AUC": 0.8953517080604937,
          "logloss": 0.006826552397128003
        }
      },
      "macro_AUC": 0.8240150326105995
    },
    "2": {
      "validation": {
        "is_click": {
          "AUC": 0.665850019952777,
          "logloss": 0.1822364208983367
        },
        "cart": {
          "AUC": 0.865639189234401,
          "logloss": 0.017739375238749012
        },
        "fav": {
          "AUC": 0.8819196652414157,
          "logloss": 0.01337362568854116
        },
        "buy": {
          "AUC": 0.8966026553087438,
          "logloss": 0.0067563097070809435
        }
      },
      "macro_AUC": 0.8275028824343343
    },
    "3": {
      "validation": {
        "is_click": {
          "AUC": 0.6648576661849764,
          "logloss": 0.18216563687006437
        },
        "cart": {
          "AUC": 0.8643506787619998,
          "logloss": 0.018146876498510718
        },
        "fav": {
          "AUC": 0.880493150765133,
          "logloss": 0.013501525219107894
        },
        "buy": {
          "AUC": 0.8932798807139176,
          "logloss": 0.007048671754391135
        }
      },
      "macro_AUC": 0.8257453441065067
    },
    "4": {
      "validation": {
        "is_click": {
          "AUC": 0.6605068677596102,
          "logloss": 0.18256670282353848
        },
        "cart": {
          "AUC": 0.8576264205430084,
          "logloss": 0.01913435711989578
        },
        "fav": {
          "AUC": 0.8778918187065845,
          "logloss": 0.014051977107172564
        },
        "buy": {
          "AUC": 0.8831093492934906,
          "logloss": 0.007888425006923667
        }
      },
      "macro_AUC": 0.8197836140756735
    }
  },
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-middle/reports/depth-middle/result.json"
  ]
}
```

</details>
