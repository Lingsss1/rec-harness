## 169. taobao-second-read-attention-originals-5514

- 假设：Determine whether second-round attention redistribution affects predictions independently of query interaction and residual
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-09T05:32:57.744707+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-second-read-attention-originals-5514",
  "parent_experiment_ids": [
    "ctx_q4-s20262027-20260908T133705147096Z",
    "explore_dual_no_feedback-s20262027-20260908T091954887719Z"
  ],
  "family": "fixed_checkpoint_second_read_attention_probe",
  "code_id": "8b1a31d43177c96b7855e3d3dec5e8f81ca830f82cb0753f8db4585b0b6e0554",
  "diagnostic_candidate_id": "taobao-second-read-attention-v1",
  "model_source_identity": "bd65990dd5cd216464424e357a1d87267fece6b6b292f97099519c2bb0d0da90",
  "hypothesis": "Determine whether second-round attention redistribution affects predictions independently of query interaction and residual",
  "change": "Original SDPA in native/identity/reuse; replace only second w_q input with first entry; original q2 interaction/residual preserved",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": [],
    "gpu_hours": 0
  },
  "metrics": {
    "samples_each": 5514,
    "users": 4647,
    "tasks": {
      "ctx_q4": {
        "is_click": {
          "native_ipw_bce": 0.18094589604724506,
          "identity_ipw_bce": 0.18094589604724506,
          "reuse_ipw_bce": 0.1811099522098907,
          "reuse_minus_identity": {
            "estimate": 0.00016405616264536792,
            "descriptive_ci95": [
              -8.458306429502778e-05,
              0.0004569532453446214
            ],
            "directional_ci99375": [
              -0.00016665589821951866,
              0.0005859963680992394
            ]
          },
          "reuse_minus_native": {
            "estimate": 0.00016405616264536792,
            "descriptive_ci95": [
              -8.458306429502778e-05,
              0.0004569532453446214
            ],
            "directional_ci99375": [
              -0.00016665589821951866,
              0.0005859963680992394
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "inconclusive",
          "action": "abstain_from_attention_specific_model_edits",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        },
        "cart": {
          "native_ipw_bce": 0.018065485271994586,
          "identity_ipw_bce": 0.018065485271994586,
          "reuse_ipw_bce": 0.018091870975892373,
          "reuse_minus_identity": {
            "estimate": 2.6385703897761454e-05,
            "descriptive_ci95": [
              -2.185444086053926e-05,
              6.294701653637133e-05
            ],
            "directional_ci99375": [
              -4.564488380140991e-05,
              7.452098516804465e-05
            ]
          },
          "reuse_minus_native": {
            "estimate": 2.6385703897761454e-05,
            "descriptive_ci95": [
              -2.185444086053926e-05,
              6.294701653637133e-05
            ],
            "directional_ci99375": [
              -4.564488380140991e-05,
              7.452098516804465e-05
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "inconclusive",
          "action": "abstain_from_attention_specific_model_edits",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        },
        "fav": {
          "native_ipw_bce": 0.013235783087343612,
          "identity_ipw_bce": 0.013235783087343612,
          "reuse_ipw_bce": 0.013362807415134967,
          "reuse_minus_identity": {
            "estimate": 0.00012702432779136035,
            "descriptive_ci95": [
              2.5421693398959408e-05,
              0.00025115045855211556
            ],
            "directional_ci99375": [
              -5.954712104536138e-06,
              0.00031352640927223953
            ]
          },
          "reuse_minus_native": {
            "estimate": 0.00012702432779136035,
            "descriptive_ci95": [
              2.5421693398959408e-05,
              0.00025115045855211556
            ],
            "directional_ci99375": [
              -5.954712104536138e-06,
              0.00031352640927223953
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "inconclusive",
          "action": "abstain_from_attention_specific_model_edits",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        },
        "buy": {
          "native_ipw_bce": 0.006609427434078904,
          "identity_ipw_bce": 0.006609427434078904,
          "reuse_ipw_bce": 0.0066237390903886315,
          "reuse_minus_identity": {
            "estimate": 1.4311656309733042e-05,
            "descriptive_ci95": [
              9.73809531246234e-06,
              1.9220549705229177e-05
            ],
            "directional_ci99375": [
              8.107827581993967e-06,
              2.1101592077506038e-05
            ]
          },
          "reuse_minus_native": {
            "estimate": 1.4311656309733042e-05,
            "descriptive_ci95": [
              9.73809531246234e-06,
              1.9220549705229177e-05
            ],
            "directional_ci99375": [
              8.107827581993967e-06,
              2.1101592077506038e-05
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "positive",
          "action": "preserve_second_redistribution_for_this_frozen_checkpoint_task",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        }
      },
      "explore_dual_no_feedback": {
        "is_click": {
          "native_ipw_bce": 0.18145874781177582,
          "identity_ipw_bce": 0.18145874781177582,
          "reuse_ipw_bce": 0.1817920781338053,
          "reuse_minus_identity": {
            "estimate": 0.0003333303220294364,
            "descriptive_ci95": [
              4.672020819842186e-05,
              0.0006292641523202294
            ],
            "directional_ci99375": [
              -5.125708461276574e-05,
              0.000760694329372942
            ]
          },
          "reuse_minus_native": {
            "estimate": 0.0003333303220294364,
            "descriptive_ci95": [
              4.672020819842186e-05,
              0.0006292641523202294
            ],
            "directional_ci99375": [
              -5.125708461276574e-05,
              0.000760694329372942
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "inconclusive",
          "action": "abstain_from_attention_specific_model_edits",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        },
        "cart": {
          "native_ipw_bce": 0.01801679791264649,
          "identity_ipw_bce": 0.01801679791264649,
          "reuse_ipw_bce": 0.018317192767206936,
          "reuse_minus_identity": {
            "estimate": 0.00030039485456040453,
            "descriptive_ci95": [
              -0.0001975344196785441,
              0.001081134147214237
            ],
            "directional_ci99375": [
              -0.00026331076419997495,
              0.0014940796627174852
            ]
          },
          "reuse_minus_native": {
            "estimate": 0.00030039485456040453,
            "descriptive_ci95": [
              -0.0001975344196785441,
              0.001081134147214237
            ],
            "directional_ci99375": [
              -0.00026331076419997495,
              0.0014940796627174852
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "inconclusive",
          "action": "abstain_from_attention_specific_model_edits",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        },
        "fav": {
          "native_ipw_bce": 0.01332012841318057,
          "identity_ipw_bce": 0.01332012841318057,
          "reuse_ipw_bce": 0.013267225495969303,
          "reuse_minus_identity": {
            "estimate": -5.290291721130152e-05,
            "descriptive_ci95": [
              -9.829220823064127e-05,
              -2.530767665451306e-06
            ],
            "directional_ci99375": [
              -0.000113471932076058,
              2.206656817021924e-05
            ]
          },
          "reuse_minus_native": {
            "estimate": -5.290291721130152e-05,
            "descriptive_ci95": [
              -9.829220823064127e-05,
              -2.530767665451306e-06
            ],
            "directional_ci99375": [
              -0.000113471932076058,
              2.206656817021924e-05
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "inconclusive",
          "action": "abstain_from_attention_specific_model_edits",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        },
        "buy": {
          "native_ipw_bce": 0.006566427378615031,
          "identity_ipw_bce": 0.006566427378615031,
          "reuse_ipw_bce": 0.0065513586568363745,
          "reuse_minus_identity": {
            "estimate": -1.5068721778670674e-05,
            "descriptive_ci95": [
              -3.23440543160087e-05,
              2.4604153342927415e-06
            ],
            "directional_ci99375": [
              -3.8730893810857566e-05,
              8.949244607792113e-06
            ]
          },
          "reuse_minus_native": {
            "estimate": -1.5068721778670674e-05,
            "descriptive_ci95": [
              -3.23440543160087e-05,
              2.4604153342927415e-06
            ],
            "directional_ci99375": [
              -3.8730893810857566e-05,
              8.949244607792113e-06
            ]
          },
          "epsilon": 1e-06,
          "ipw_mean_absolute_identity_drift": 0.0,
          "direction": "inconclusive",
          "action": "abstain_from_attention_specific_model_edits",
          "exact_row_bce_null": false,
          "directional_interval_within_epsilon": false,
          "native_consistency_checked": true
        }
      }
    },
    "bootstrap_replicates": 10000,
    "directional_ci": 0.99375,
    "epsilon": 1e-06,
    "identity_exact": true,
    "actual_allpadding_identity_exact": true,
    "tool_six_gates_passed": true
  },
  "decision": "provisional",
  "tool_decision": "accepted",
  "global_best_changed": false,
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/outcome.json",
    "workspace/qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/shadow/aggregate.json",
    "workspace/qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/shadow-validation.json",
    "workspace/qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/memory-consolidation-status.json"
  ],
  "confidence": "Checkpoint/subset conditional; only q4 buy passes the predeclared nominal multiplicity-adjusted direction gate"
}
```

</details>
