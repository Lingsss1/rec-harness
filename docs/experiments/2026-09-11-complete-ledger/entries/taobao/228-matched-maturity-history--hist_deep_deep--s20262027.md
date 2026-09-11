## 228. matched-maturity-history--hist_deep_deep--s20262027

- 假设：The depth and order of history representations may change second-read utility; compare against matched Cross8.
- 判定：discard
- 原因：Macro regression versus A has a negative paired interval and latency rises 46.63%.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--hist_deep_deep--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "hist_deep_deep",
  "code_id": "db2c4b4cc7829e0693010dcddad436bc26a7ead0d1f8cae94af6cb0e0630e01b",
  "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8.",
  "change": "{\"model\": \"QFormerCross8HistoryVersions\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"sequence_loop_steps\": null, \"history_version_mode\": \"deep_deep\", \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8HistoryVersions",
    "maturity_mode": "mature",
    "work_steps": 2,
    "sequence_loop_steps": null,
    "history_version_mode": "deep_deep",
    "num_queries": 4
  },
  "input_protocol_id": "strict-past-native-last100-action19-mature",
  "budget": {
    "formal_training_runs": 1,
    "epochs": 1,
    "seeds": [
      20262027
    ],
    "new_training_seeds": [],
    "optimizer_steps": 2193,
    "dense_optimizer": "HeavyBall.PSGDKron",
    "preconditioner_update_probability": 1.0,
    "dense_lr": 0.0001,
    "sparse_optimizer": "Adagrad",
    "sparse_lr": 0.05,
    "clip": 1,
    "batch_size": 8192,
    "precision": "BF16"
  },
  "metrics": {
    "primary": {
      "name": "macro_AUC",
      "value": 0.7616711000084829,
      "delta": -0.005423851306377486
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18424566433500367,
        "AUC": 0.6497457487064112
      },
      "cart": {
        "logloss": 0.02192193038187438,
        "AUC": 0.7906974804029819
      },
      "fav": {
        "logloss": 0.01732778739075879,
        "AUC": 0.7837235581950441
      },
      "buy": {
        "logloss": 0.009719312491575791,
        "AUC": 0.8225176127294948
      }
    },
    "macro_source_logloss": 0.058303673649803156,
    "matched_parent_effect": {
      "coefficients": {
        "released_r1": 0.0,
        "released_r2": 0.0,
        "mature_r1": 0.0,
        "mature_r2": -1.0,
        "hist_deep_deep": 1.0,
        "hist_shallow_deep": 0.0,
        "hist_deep_shallow": 0.0,
        "cross8_mature": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.005423851306377486,
          "ci95": [
            -0.008412503504719962,
            -0.0026454209667909805
          ]
        },
        "auc.is_click": {
          "estimate": -0.00019054449457822642,
          "ci95": [
            -0.0009138213918716498,
            0.0005321794802823903
          ]
        },
        "auc.cart": {
          "estimate": -0.0017367899543617726,
          "ci95": [
            -0.006185804529578232,
            0.0033248502757166052
          ]
        },
        "auc.fav": {
          "estimate": -0.018739932180003915,
          "ci95": [
            -0.025372328830792136,
            -0.012174312343978888
          ]
        },
        "auc.buy": {
          "estimate": -0.0010281385965660306,
          "ci95": [
            -0.00922978984998619,
            0.006836202683316097
          ]
        },
        "bce.is_click": {
          "estimate": -0.00014598193278481109,
          "ci95": [
            -0.00021506546027438844,
            -7.353338523247641e-05
          ]
        },
        "bce.cart": {
          "estimate": 9.397601251867616e-05,
          "ci95": [
            -3.6698986122155206e-06,
            0.00018837555750332352
          ]
        },
        "bce.fav": {
          "estimate": 0.0005296296965203867,
          "ci95": [
            0.0004010178874869381,
            0.0006693927687444701
          ]
        },
        "bce.buy": {
          "estimate": 7.551056066901728e-05,
          "ci95": [
            -9.464848442267785e-06,
            0.00016591146442635377
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0015462454597835462,
          "ci95": [
            -0.003142284530924455,
            0.005922861859455457
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.01854938768542569,
          "ci95": [
            0.012144775190802166,
            0.025282930981281062
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.0008375941019878042,
          "ci95": [
            -0.007038853144111845,
            0.009097348462186135
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.017003142225642143,
          "ci95": [
            0.008691724204889197,
            0.024689806525275866
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": -0.0007086513577957421,
          "ci95": [
            -0.0091450068446889,
            0.008255733649983903
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.017711793583437885,
          "ci95": [
            -0.027522734947077583,
            -0.007191656047186967
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00023995794530348724,
          "ci95": [
            -0.0003563642708393695,
            -0.00012161916899428135
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0006756116293051978,
          "ci95": [
            -0.0008255571872592501,
            -0.0005319855159088829
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00022149249345382836,
          "ci95": [
            -0.00034159621864055846,
            -0.00011610285278845947
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00043565368400171056,
          "ci95": [
            -0.000593519835986029,
            -0.0002786051152977905
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 1.846545184965888e-05,
          "ci95": [
            -0.00011118883292660711,
            0.0001444575856194869
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 0.00045411913585136944,
          "ci95": [
            0.00030223156572321315,
            0.0006294317738324492
          ]
        }
      }
    },
    "parameter_count": 51955556,
    "initializer_sha256": "914ccddd39668aa2cd2e294f413889bc9659d29ba06f8e62a8ea750f46e1e357",
    "formal_worker_cost": {
      "wall_seconds": 2885.2715956643224,
      "cpu_seconds": 3398.598635535,
      "peak_rss_bytes": 17073332224,
      "gpu_peak_bytes": 37873516544,
      "samples": 2651
    },
    "forward_latency": {
      "mean_ms": 20.788499212265016,
      "p50_ms": 20.763855934143066,
      "p95_ms": 20.848753452301025,
      "batch_size": 1024,
      "warmups": 10,
      "iterations": 40
    }
  },
  "validity": {
    "valid": true,
    "violations": [],
    "state": "passed",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-quality-summary.json",
    "no_prediction_to_work_feedback": true,
    "test_evaluated": false,
    "source_and_input_identity_scope": "Original N audit supplies the 122 full input-content SHA256 values. Production B/C runners verified input stat snapshots before and after their actual traversals, and both original metadata hashes before those traversals; this audit now verifies the 122 current stat records and two original metadata hashes. No Parquet, checkpoint or prediction array is opened or rehashed here. Model-output SHA256 values are inherited from successful production completions and matched worker/launcher receipts. Stat continuity cannot independently rule out content changes that preserve stat fields."
  },
  "decision": "discard",
  "decision_reason": "Macro regression versus A has a negative paired interval and latency rises 46.63%.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/Taobao_Action/hist_deep_deep.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/result.json",
    "result_sha256": "beef2eece0bd568ecaf13be7b0448c380647b1e1e664ea38c6e57f030f1c6430",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/manifest.json",
    "manifest_sha256": "d802e843a901c9fc96e995d5e64cf8b121363e2d5b2eb038a46c1c0213fd4de5",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_deep/Taobao_Action/hist_deep_deep.model",
    "checkpoint_sha256": "757a6cbdc9c0f914cd9c994436e8759a15bc6958fb46daef668e4983f2f31378",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
