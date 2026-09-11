## 227. matched-maturity-history--mature_r2--s20262027

- 假设：Recurrence gain may depend on action maturity under matched strict history support.
- 判定：retain_protocol_anchor
- 原因：Keep the existing mature-input recurrent anchor; no confirmed SD macro improvement and lower cost than SD.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--mature_r2--s20262027",
  "parent_experiment_id": "matched-maturity-history--mature_r1--s20262027",
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "mature_r2",
  "code_id": "547d90bf0c6acc6513ecfb9ea55844c34f70047492832f6a951b9c40a1bc9c49",
  "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support.",
  "change": "{\"model\": \"QFormerCross8MaturityDepth\", \"maturity_mode\": \"mature\", \"work_steps\": 2, \"sequence_loop_steps\": null, \"history_version_mode\": null, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MaturityDepth",
    "maturity_mode": "mature",
    "work_steps": 2,
    "sequence_loop_steps": null,
    "history_version_mode": null,
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
      "value": 0.7670949513148605,
      "delta": 0.0009077124632053679
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18439164626778223,
        "AUC": 0.6499362932009894
      },
      "cart": {
        "logloss": 0.021827954343642188,
        "AUC": 0.7924342703573437
      },
      "fav": {
        "logloss": 0.016798157691351123,
        "AUC": 0.802463490375048
      },
      "buy": {
        "logloss": 0.00964380193474217,
        "AUC": 0.8235457513260608
      }
    },
    "macro_source_logloss": 0.05816539005937943,
    "matched_parent_effect": {
      "coefficients": {
        "released_r1": 0.0,
        "released_r2": 0.0,
        "mature_r1": -1.0,
        "mature_r2": 1.0,
        "hist_deep_deep": 0.0,
        "hist_shallow_deep": 0.0,
        "hist_deep_shallow": 0.0,
        "cross8_mature": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": 0.0009077124632053679,
          "ci95": [
            -0.0018596814515439701,
            0.003713361672625221
          ]
        },
        "auc.is_click": {
          "estimate": 0.0006681279337776846,
          "ci95": [
            -0.0002222790281181136,
            0.0015248920487231806
          ]
        },
        "auc.cart": {
          "estimate": -0.0003809077547397566,
          "ci95": [
            -0.007219485733060266,
            0.005351430430537084
          ]
        },
        "auc.fav": {
          "estimate": 0.003051269293169012,
          "ci95": [
            -0.003700557919273023,
            0.00973557540334511
          ]
        },
        "auc.buy": {
          "estimate": 0.0002923603806145314,
          "ci95": [
            -0.0063462088024723974,
            0.008093264275765342
          ]
        },
        "bce.is_click": {
          "estimate": 0.00014229040664570647,
          "ci95": [
            6.520096542023124e-05,
            0.0002194413469561021
          ]
        },
        "bce.cart": {
          "estimate": -7.237996600449256e-05,
          "ci95": [
            -0.00019406865767628698,
            7.113242861453778e-05
          ]
        },
        "bce.fav": {
          "estimate": -0.00010240622628154145,
          "ci95": [
            -0.00028504904458396694,
            7.176755364155076e-05
          ]
        },
        "bce.buy": {
          "estimate": -0.00018592581472353167,
          "ci95": [
            -0.00026568298933008665,
            -0.00010524364462295747
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0010490356885174412,
          "ci95": [
            -0.004540713092571852,
            0.007999056243490945
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": -0.0023831413593913275,
          "ci95": [
            -0.009107419602031383,
            0.004824513109296432
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.00037576755316315325,
          "ci95": [
            -0.007390793712508286,
            0.007150213925997585
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": -0.0034321770479087688,
          "ci95": [
            -0.012398049711609743,
            0.005304547760193692
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": -0.000673268135354288,
          "ci95": [
            -0.010617252602211585,
            0.009146860871393703
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": 0.002758908912554481,
          "ci95": [
            -0.007409032777870089,
            0.011785135126581659
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": 0.00021467037265019903,
          "ci95": [
            5.007325854027734e-05,
            0.0003694990098782714
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": 0.0002446966329272479,
          "ci95": [
            6.0274382380685634e-05,
            0.0004290412135605134
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": 0.00032821622136923814,
          "ci95": [
            0.0002030526031044885,
            0.0004469074248974131
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 3.0026260277048883e-05,
          "ci95": [
            -0.00017158135970314113,
            0.0002448841485041507
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 0.00011354584871903911,
          "ci95": [
            -4.471255640658058e-05,
            0.00027330562688591313
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": 8.351958844199023e-05,
          "ci95": [
            -0.00011157301012666723,
            0.00027179791943239486
          ]
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "formal_worker_cost": {
      "wall_seconds": 2534.6803352087736,
      "cpu_seconds": 3041.4287067490004,
      "peak_rss_bytes": 17010442240,
      "gpu_peak_bytes": 28369223680,
      "samples": 2324
    },
    "forward_latency": {
      "mean_ms": 14.177265572547913,
      "p50_ms": 14.120863914489746,
      "p95_ms": 14.295497226715087,
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
  "decision": "retain_protocol_anchor",
  "decision_reason": "Keep the existing mature-input recurrent anchor; no confirmed SD macro improvement and lower cost than SD.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/Taobao_Action/mature_r2.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/result.json",
    "result_sha256": "15c18a7aee32a1dd3015c9a8c9613fe532887416cd49e5a063943f3eee4d246b",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/manifest.json",
    "manifest_sha256": "a8439ee9ab331449eb1625f5822deb9a2ad15bf4ad5be8fefa451e78a17949de",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2/Taobao_Action/mature_r2.model",
    "checkpoint_sha256": "c514fb78a24c2bbf4460c65d5b6d97e642e5d9a2359be3f9b4dd43558f872ecd",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
