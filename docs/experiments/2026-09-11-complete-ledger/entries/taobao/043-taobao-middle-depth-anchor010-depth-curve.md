## 43. taobao-middle-depth-anchor010-depth-curve

- 假设：Does additional recurrence improve one frozen checkpoint?
- 判定：diagnostic
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T17:02:55.490202+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-middle-depth-anchor010-depth-curve",
  "family": "fixed_checkpoint_depth_diagnostic",
  "parent_experiment_id": "middle_r2_anchor010-s20262027-20260907T161618129374Z",
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
          "AUC": 0.6610125507459184,
          "logloss": 0.18305584457927632
        },
        "cart": {
          "AUC": 0.8625661133052164,
          "logloss": 0.017776426513920102
        },
        "fav": {
          "AUC": 0.8794428200711135,
          "logloss": 0.013399472581234423
        },
        "buy": {
          "AUC": 0.8957851799492597,
          "logloss": 0.006889105491798823
        }
      },
      "macro_AUC": 0.8247016660178769
    },
    "2": {
      "validation": {
        "is_click": {
          "AUC": 0.6643270175816082,
          "logloss": 0.182251681713331
        },
        "cart": {
          "AUC": 0.865675525598574,
          "logloss": 0.017663315656501342
        },
        "fav": {
          "AUC": 0.8789390840722765,
          "logloss": 0.01338531019841893
        },
        "buy": {
          "AUC": 0.8968444553996352,
          "logloss": 0.006792790688774889
        }
      },
      "macro_AUC": 0.8264465206630235
    },
    "3": {
      "validation": {
        "is_click": {
          "AUC": 0.6632867883324925,
          "logloss": 0.18226554665443448
        },
        "cart": {
          "AUC": 0.8651837925129034,
          "logloss": 0.017731096511784054
        },
        "fav": {
          "AUC": 0.8770308103310435,
          "logloss": 0.013501148523637372
        },
        "buy": {
          "AUC": 0.896148197581576,
          "logloss": 0.006868475749067778
        }
      },
      "macro_AUC": 0.8254123971895039
    },
    "4": {
      "validation": {
        "is_click": {
          "AUC": 0.6608846781108332,
          "logloss": 0.1825805805103598
        },
        "cart": {
          "AUC": 0.862310916006368,
          "logloss": 0.018097249262265518
        },
        "fav": {
          "AUC": 0.8742057545043856,
          "logloss": 0.013845232552363506
        },
        "buy": {
          "AUC": 0.8930639256035815,
          "logloss": 0.007150180062991446
        }
      },
      "macro_AUC": 0.822616318556292
    }
  },
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-middle/reports/depth-anchor010/result.json"
  ]
}
```

</details>
