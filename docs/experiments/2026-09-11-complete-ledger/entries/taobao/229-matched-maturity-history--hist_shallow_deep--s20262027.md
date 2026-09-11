## 229. matched-maturity-history--hist_shallow_deep--s20262027

- 假设：The depth and order of history representations may change second-read utility; compare against matched Cross8.
- 判定：retain_research_tradeoff
- 原因：Positive macro point meets candidate policy but interval crosses zero; latency rises 46.65%, cart/buy point estimates decline; no promotion.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--hist_shallow_deep--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "hist_shallow_deep",
  "code_id": "db2c4b4cc7829e0693010dcddad436bc26a7ead0d1f8cae94af6cb0e0630e01b",
  "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8.",
  "change": "{\"model\": \"QFormerCross8HistoryVersions\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"sequence_loop_steps\": null, \"history_version_mode\": \"shallow_deep\", \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8HistoryVersions",
    "maturity_mode": "mature",
    "work_steps": 2,
    "sequence_loop_steps": null,
    "history_version_mode": "shallow_deep",
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
      "value": 0.7673842438958248,
      "delta": 0.00028929258096432875
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18363844976930296,
        "AUC": 0.6532613183153021
      },
      "cart": {
        "logloss": 0.021811821982936203,
        "AUC": 0.7906579507328771
      },
      "fav": {
        "logloss": 0.01682381561455162,
        "AUC": 0.8066565087647535
      },
      "buy": {
        "logloss": 0.009714505924374887,
        "AUC": 0.8189611977703666
      }
    },
    "macro_source_logloss": 0.057997148322791424,
    "matched_parent_effect": {
      "coefficients": {
        "released_r1": 0.0,
        "released_r2": 0.0,
        "mature_r1": 0.0,
        "mature_r2": -1.0,
        "hist_deep_deep": 0.0,
        "hist_shallow_deep": 1.0,
        "hist_deep_shallow": 0.0,
        "cross8_mature": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": 0.00028929258096432875,
          "ci95": [
            -0.002641094433663372,
            0.0030838629067439114
          ]
        },
        "auc.is_click": {
          "estimate": 0.003325025114312652,
          "ci95": [
            0.002552821477465256,
            0.004142619520820806
          ]
        },
        "auc.cart": {
          "estimate": -0.0017763196244665602,
          "ci95": [
            -0.006242021447639426,
            0.0028211409772939487
          ]
        },
        "auc.fav": {
          "estimate": 0.004193018389705427,
          "ci95": [
            -0.0018074643720757677,
            0.009920015085292317
          ]
        },
        "auc.buy": {
          "estimate": -0.004584553555694204,
          "ci95": [
            -0.013001132956183709,
            0.002617144503444021
          ]
        },
        "bce.is_click": {
          "estimate": -0.0007531964984849038,
          "ci95": [
            -0.0008314742597532538,
            -0.0006734761755542732
          ]
        },
        "bce.cart": {
          "estimate": -1.6132385910066355e-05,
          "ci95": [
            -0.00010883785954870972,
            7.698846923413823e-05
          ]
        },
        "bce.fav": {
          "estimate": 2.5657920210489515e-05,
          "ci95": [
            -0.00011019522346056949,
            0.00016514093052077798
          ]
        },
        "bce.buy": {
          "estimate": 7.070401254058792e-05,
          "ci95": [
            -7.251077712635894e-06,
            0.00015301185737824882
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.005101344738779212,
          "ci95": [
            0.0004856527364410781,
            0.009652835106274323
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.0008679932753927755,
          "ci95": [
            -0.006916677383449374,
            0.005265897182774135
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.007909578670006856,
          "ci95": [
            0.0005575962829397292,
            0.01651875415504198
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.005969338014171988,
          "ci95": [
            -0.01318053556392374,
            0.001555735965022289
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.002808233931227644,
          "ci95": [
            -0.004865801360991351,
            0.011711435802989239
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.008777571945399631,
          "ci95": [
            -0.0007690734683098853,
            0.018868567681647005
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0007370641125748374,
          "ci95": [
            -0.0008511397561138915,
            -0.0006193881289777369
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.0007788544186953933,
          "ci95": [
            -0.0009354256721203936,
            -0.0006243096307012996
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0008239005110254917,
          "ci95": [
            -0.0009424501405048776,
            -0.0007162224248533547
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -4.179030612055587e-05,
          "ci95": [
            -0.0002154476278340656,
            0.00012635959718274674
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -8.683639845065427e-05,
          "ci95": [
            -0.00021266664265950528,
            2.3428007227234754e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -4.50460923300984e-05,
          "ci95": [
            -0.00020295309852636986,
            0.00011365310115141802
          ]
        }
      }
    },
    "parameter_count": 51955556,
    "initializer_sha256": "914ccddd39668aa2cd2e294f413889bc9659d29ba06f8e62a8ea750f46e1e357",
    "formal_worker_cost": {
      "wall_seconds": 2898.3198798447847,
      "cpu_seconds": 3421.1364017429996,
      "peak_rss_bytes": 17097760768,
      "gpu_peak_bytes": 37873516544,
      "samples": 2662
    },
    "forward_latency": {
      "mean_ms": 20.791291189193725,
      "p50_ms": 20.775184631347656,
      "p95_ms": 20.83083095550537,
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
  "decision": "retain_research_tradeoff",
  "decision_reason": "Positive macro point meets candidate policy but interval crosses zero; latency rises 46.65%, cart/buy point estimates decline; no promotion.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/Taobao_Action/hist_shallow_deep.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/result.json",
    "result_sha256": "8b5385dff3136403baa61cbc8550f453f2cd12a446d122be6d1995d7f5248aa7",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/manifest.json",
    "manifest_sha256": "18df9929a44abebfb9a83f5888db43c9b25f659c2dc46d4bd4102cf911332cce",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_shallow_deep/Taobao_Action/hist_shallow_deep.model",
    "checkpoint_sha256": "21c90fb61c4bfc6f6c74ce97fab319bbdf9f7c90279166937187099574e056f8",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
