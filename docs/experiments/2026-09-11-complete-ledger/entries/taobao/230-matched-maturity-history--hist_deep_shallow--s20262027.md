## 230. matched-maturity-history--hist_deep_shallow--s20262027

- 假设：The depth and order of history representations may change second-read utility; compare against matched Cross8.
- 判定：discard
- 原因：Macro regression versus A has a negative paired interval and latency rises 46.35%.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--hist_deep_shallow--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "hist_deep_shallow",
  "code_id": "db2c4b4cc7829e0693010dcddad436bc26a7ead0d1f8cae94af6cb0e0630e01b",
  "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8.",
  "change": "{\"model\": \"QFormerCross8HistoryVersions\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"sequence_loop_steps\": null, \"history_version_mode\": \"deep_shallow\", \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8HistoryVersions",
    "maturity_mode": "mature",
    "work_steps": 2,
    "sequence_loop_steps": null,
    "history_version_mode": "deep_shallow",
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
      "value": 0.7626726252984801,
      "delta": -0.00442232601638029
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18429814205668352,
        "AUC": 0.648959602931274
      },
      "cart": {
        "logloss": 0.021861702486078695,
        "AUC": 0.7916853382329715
      },
      "fav": {
        "logloss": 0.016969594386391583,
        "AUC": 0.7929455504742192
      },
      "buy": {
        "logloss": 0.009851115939254376,
        "AUC": 0.8171000095554561
      }
    },
    "macro_source_logloss": 0.058245138717102046,
    "matched_parent_effect": {
      "coefficients": {
        "released_r1": 0.0,
        "released_r2": 0.0,
        "mature_r1": 0.0,
        "mature_r2": -1.0,
        "hist_deep_deep": 0.0,
        "hist_shallow_deep": 0.0,
        "hist_deep_shallow": 1.0,
        "cross8_mature": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.00442232601638029,
          "ci95": [
            -0.007850020183400443,
            -0.0007105993787787589
          ]
        },
        "auc.is_click": {
          "estimate": -0.0009766902697154745,
          "ci95": [
            -0.0017492456952240304,
            -0.0001265834132608117
          ]
        },
        "auc.cart": {
          "estimate": -0.0007489321243722147,
          "ci95": [
            -0.005822061535492665,
            0.004799326070624706
          ]
        },
        "auc.fav": {
          "estimate": -0.009517939900828831,
          "ci95": [
            -0.016917269412454975,
            -0.00384680038610531
          ]
        },
        "auc.buy": {
          "estimate": -0.0064457417706046405,
          "ci95": [
            -0.018258906576421895,
            0.0040935215947381694
          ]
        },
        "bce.is_click": {
          "estimate": -9.350421110174145e-05,
          "ci95": [
            -0.00017151177295081227,
            -2.1719813288732834e-05
          ]
        },
        "bce.cart": {
          "estimate": 3.374811664491953e-05,
          "ci95": [
            -8.819640919804292e-05,
            0.00013879253364238356
          ]
        },
        "bce.fav": {
          "estimate": 0.00017143669203935688,
          "ci95": [
            4.5087526967790405e-05,
            0.00034342742564929857
          ]
        },
        "bce.buy": {
          "estimate": 0.00020731400255669281,
          "ci95": [
            9.869355278767601e-05,
            0.0003323942590152036
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": -0.00022775814534325978,
          "ci95": [
            -0.0057800264844969805,
            0.00497327775931991
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.008541249631113357,
          "ci95": [
            0.0028672450606610778,
            0.015892876858460206
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.005469051500889166,
          "ci95": [
            -0.005020354299115959,
            0.017477220626995405
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.008769007776456617,
          "ci95": [
            0.0008970524551275978,
            0.0176478846259393
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": 0.005696809646232426,
          "ci95": [
            -0.006657844621396081,
            0.01836093035322969
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.0030721981302241907,
          "ci95": [
            -0.015152627998566296,
            0.009458778307061315
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.00012725232774666098,
          "ci95": [
            -0.0002622425857728945,
            1.7219766282416834e-06
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00026494090314109833,
          "ci95": [
            -0.00045492478912983537,
            -0.00010847402918817166
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.00030081821365843427,
          "ci95": [
            -0.0004536166084948883,
            -0.00017213568670678485
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": -0.00013768857539443735,
          "ci95": [
            -0.0003377216738290318,
            3.078882578139395e-05
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": -0.00017356588591177328,
          "ci95": [
            -0.000339538681299138,
            -1.8160151660604833e-05
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -3.5877310517335934e-05,
          "ci95": [
            -0.0002043392987233152,
            0.00017142278640028773
          ]
        }
      }
    },
    "parameter_count": 51955556,
    "initializer_sha256": "914ccddd39668aa2cd2e294f413889bc9659d29ba06f8e62a8ea750f46e1e357",
    "formal_worker_cost": {
      "wall_seconds": 2879.7387092188,
      "cpu_seconds": 3399.227580377,
      "peak_rss_bytes": 17094365184,
      "gpu_peak_bytes": 37875613696,
      "samples": 2641
    },
    "forward_latency": {
      "mean_ms": 20.748981523513795,
      "p50_ms": 20.741984367370605,
      "p95_ms": 20.79793539047241,
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
  "decision_reason": "Macro regression versus A has a negative paired interval and latency rises 46.35%.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/Taobao_Action/hist_deep_shallow.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/result.json",
    "result_sha256": "70715ebad43b2a86605bfe2a1a76c9ea5fa3e61cf3b7e7b7f742a9a160cb3dd2",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/manifest.json",
    "manifest_sha256": "29d4d701fb97e12a478024218ac2cf9be2d7cb11efdc28170eed37635982696b",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/hist_deep_shallow/Taobao_Action/hist_deep_shallow.model",
    "checkpoint_sha256": "de0f20486a97d2f4fb9fb46660f5fea96ffcdd15bba0a068e77ba285a26a42e1",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
