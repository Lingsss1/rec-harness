## 225. matched-maturity-history--released_r2--s20262027

- 假设：Recurrence gain may depend on action maturity under matched strict history support.
- 判定：discard
- 原因：R2-R1 macro estimate and paired interval are negative; keep the completed result as evidence.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--released_r2--s20262027",
  "parent_experiment_id": "matched-maturity-history--released_r1--s20262027",
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "released_r2",
  "code_id": "547d90bf0c6acc6513ecfb9ea55844c34f70047492832f6a951b9c40a1bc9c49",
  "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support.",
  "change": "{\"model\": \"QFormerCross8MaturityDepth\", \"maturity_mode\": \"released\", \"work_steps\": 2, \"sequence_loop_steps\": null, \"history_version_mode\": null, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MaturityDepth",
    "maturity_mode": "released",
    "work_steps": 2,
    "sequence_loop_steps": null,
    "history_version_mode": null,
    "num_queries": 4
  },
  "input_protocol_id": "strict-past-native-last100-action19-released",
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
      "value": 0.8055046313408105,
      "delta": -0.002127112748927823
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18369704714688748,
        "AUC": 0.6517419871375195
      },
      "cart": {
        "logloss": 0.019455012263802775,
        "AUC": 0.8396346776078314
      },
      "fav": {
        "logloss": 0.01449862306718589,
        "AUC": 0.8572007687710154
      },
      "buy": {
        "logloss": 0.007807169117126676,
        "AUC": 0.8734410918468754
      }
    },
    "macro_source_logloss": 0.0563644628987507,
    "matched_parent_effect": {
      "coefficients": {
        "released_r1": -1.0,
        "released_r2": 1.0,
        "mature_r1": 0.0,
        "mature_r2": 0.0,
        "hist_deep_deep": 0.0,
        "hist_shallow_deep": 0.0,
        "hist_deep_shallow": 0.0,
        "cross8_mature": 0.0
      },
      "metrics": {
        "macro_auc": {
          "estimate": -0.002127112748927823,
          "ci95": [
            -0.003414442618004629,
            -0.0006945099171742354
          ]
        },
        "auc.is_click": {
          "estimate": -0.0003097774546739762,
          "ci95": [
            -0.0012298788938377408,
            0.0006113453491059431
          ]
        },
        "auc.cart": {
          "estimate": -0.003128840963602997,
          "ci95": [
            -0.005484449429297375,
            -0.0007393192363301972
          ]
        },
        "auc.fav": {
          "estimate": -0.004386308855889087,
          "ci95": [
            -0.007532196862459095,
            -0.001722278728872384
          ]
        },
        "auc.buy": {
          "estimate": -0.0006835237215452317,
          "ci95": [
            -0.004301213433377563,
            0.002887416586999068
          ]
        },
        "bce.is_click": {
          "estimate": -0.00047153858775503266,
          "ci95": [
            -0.0005579041673170442,
            -0.0003867263382009455
          ]
        },
        "bce.cart": {
          "estimate": 9.593568243794481e-05,
          "ci95": [
            2.423921742186301e-05,
            0.0001627424466462827
          ]
        },
        "bce.fav": {
          "estimate": 8.2470366193476e-06,
          "ci95": [
            -5.7608898177930774e-05,
            7.508718343882624e-05
          ]
        },
        "bce.buy": {
          "estimate": 4.198788671385279e-05,
          "ci95": [
            -1.3193391757272166e-05,
            0.00010293963717053298
          ]
        },
        "auc.is_click_minus_cart": {
          "estimate": 0.0028190635089290206,
          "ci95": [
            0.00027940894707772227,
            0.005394355942149159
          ]
        },
        "auc.is_click_minus_fav": {
          "estimate": 0.004076531401215111,
          "ci95": [
            0.001239809723544777,
            0.007360098021071884
          ]
        },
        "auc.is_click_minus_buy": {
          "estimate": 0.00037374626687125545,
          "ci95": [
            -0.0030903368093852133,
            0.003967027656922535
          ]
        },
        "auc.cart_minus_fav": {
          "estimate": 0.0012574678922860905,
          "ci95": [
            -0.002636582803486742,
            0.0052622921516682014
          ]
        },
        "auc.cart_minus_buy": {
          "estimate": -0.002445317242057765,
          "ci95": [
            -0.00657955137594537,
            0.001732054384370327
          ]
        },
        "auc.fav_minus_buy": {
          "estimate": -0.0037027851343438556,
          "ci95": [
            -0.008258634288762631,
            0.00071958058134674
          ]
        },
        "bce.is_click_minus_cart": {
          "estimate": -0.0005674742701929775,
          "ci95": [
            -0.0006648050273271466,
            -0.00045896553138239553
          ]
        },
        "bce.is_click_minus_fav": {
          "estimate": -0.00047978562437438026,
          "ci95": [
            -0.0005872967595772318,
            -0.0003715318823726459
          ]
        },
        "bce.is_click_minus_buy": {
          "estimate": -0.0005135264744688854,
          "ci95": [
            -0.0006113505422282135,
            -0.0004084848742313566
          ]
        },
        "bce.cart_minus_fav": {
          "estimate": 8.768864581859721e-05,
          "ci95": [
            -7.851157691751699e-06,
            0.00018509813422627495
          ]
        },
        "bce.cart_minus_buy": {
          "estimate": 5.394779572409202e-05,
          "ci95": [
            -3.4829038660859475e-05,
            0.00013866591657591418
          ]
        },
        "bce.fav_minus_buy": {
          "estimate": -3.374085009450519e-05,
          "ci95": [
            -0.00011850187689834248,
            5.3441358461832976e-05
          ]
        }
      }
    },
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "formal_worker_cost": {
      "wall_seconds": 2526.1776882708073,
      "cpu_seconds": 3035.017940275,
      "peak_rss_bytes": 17027457024,
      "gpu_peak_bytes": 28367126528,
      "samples": 2318
    },
    "forward_latency": {
      "mean_ms": 14.120417666435241,
      "p50_ms": 14.114336013793945,
      "p95_ms": 14.174235391616822,
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
  "decision_reason": "R2-R1 macro estimate and paired interval are negative; keep the completed result as evidence.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/Taobao_Action/released_r2.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/result.json",
    "result_sha256": "55575cad30b2b41e3c4be6ad42b750c9165a6528dfce19c63ec827320ec21a3d",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/manifest.json",
    "manifest_sha256": "8277d9582ccb4302e340493826c44181f6b64ae6bd843122b43190f8ee17d823",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r2/Taobao_Action/released_r2.model",
    "checkpoint_sha256": "aa0a7d26991d07b963c1d0a186569a62b1c91b841dd7d17a87154819699d6726",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
