## 170. taobao-recurrence-task-effects-saved-validation

- 假设：Predictive benefit of a second cycle may differ across task heads; distinguish differences in magnitude from exclusive task utility.
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "taobao-recurrence-task-effects-saved-validation",
  "timestamp": "2026-09-09T06:06:00.273415+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "parent_experiment_ids": [
    "q4_r1-s20262027-20260908T150233962301Z",
    "ctx_q4-s20262027-20260908T133705147096Z",
    "explore_dual_no_feedback-s20262027-20260908T091954887719Z"
  ],
  "family": "saved_prediction_taskwise_recurrence_effects",
  "code_id": "f1bf123b463d24000789290c488c67ede29e48749871b4fb4cc13b787c9433c9",
  "hypothesis": "Predictive benefit of a second cycle may differ across task heads; distinguish differences in magnitude from exclusive task utility.",
  "change": "Replay historical q4 trained-depth 500 user draws and add q8 frozen-depth paired intervals; retain all four AUC/BCE effects and all six task contrasts.",
  "budget": {
    "training_updates": 0,
    "new_training_runs": 0,
    "new_training_seeds": [],
    "model_forwards": 0,
    "gpu_hours": 0,
    "bootstrap_repetitions_each": 500,
    "bootstrap_seed": 20260906
  },
  "metrics": {
    "q4_delta_auc": {
      "is_click": 5.341478532294719e-05,
      "cart": 0.0027787112829202876,
      "fav": 0.0010524773672074828,
      "buy": 0.003956897505808188
    },
    "q8_delta_auc": {
      "is_click": 0.010860654502252487,
      "cart": 0.013065211709695657,
      "fav": 0.012096708103326748,
      "buy": 0.013753872259069433
    },
    "q4_click_delta_bce": 0.000163370153419059,
    "q4_click_bce_ci95": [
      0.00010765296501773477,
      0.00022282304819187275
    ],
    "direct_auc_contrasts": {
      "q4_trained": [
        {
          "first_task": "is_click",
          "second_task": "cart",
          "sign": "first effect minus second effect",
          "delta_auc": -0.0027252964975973404,
          "ci95_auc": [
            -0.0045223999597569595,
            -0.0010514665313744466
          ],
          "delta_bce": 0.00015971645345358635,
          "ci95_bce": [
            6.68007504797972e-05,
            0.0002502592073984823
          ]
        },
        {
          "first_task": "is_click",
          "second_task": "buy",
          "sign": "first effect minus second effect",
          "delta_auc": -0.0039034827204852407,
          "ci95_auc": [
            -0.006529910768944316,
            -0.0015277135121195885
          ],
          "delta_bce": 0.00017012846290499225,
          "ci95_bce": [
            9.470652524180969e-05,
            0.0002429032208488879
          ]
        }
      ],
      "q8_frozen": [
        {
          "first_task": "is_click",
          "second_task": "cart",
          "sign": "first effect minus second effect",
          "delta_auc": -0.0022045572074431696,
          "ci95_auc": [
            -0.003986746174442055,
            -0.0006855924601310963
          ],
          "delta_bce": -0.002490392766784844,
          "ci95_bce": [
            -0.0026361725597461643,
            -0.002333991413218153
          ]
        },
        {
          "first_task": "is_click",
          "second_task": "buy",
          "sign": "first effect minus second effect",
          "delta_auc": -0.002893217756816946,
          "ci95_auc": [
            -0.005305512896996029,
            -0.0009480822150964473
          ],
          "delta_bce": -0.0026088349479567556,
          "ci95_bce": [
            -0.0027389334566999624,
            -0.0024464906463447854
          ]
        }
      ]
    },
    "all_other_auc_contrast_intervals": "cross_zero_in_both_families",
    "q8_all_four_auc_positive_bce_negative_ci95": true,
    "q4_legacy_auc_intervals_exact": true,
    "measured_cost": {
      "wall_seconds": 199.51010938733816,
      "cpu_seconds": 565.0630770559999,
      "peak_rss_bytes": 1059528704
    }
  },
  "validity": {
    "valid": true,
    "violations": [],
    "scope": "Exploratory uncorrected 95% user intervals conditional on saved checkpoints; q4 separately trained R1/R2 differs from q8 same-checkpoint whole-cycle deletion; neither tests new readout architecture.",
    "test_evaluated": false
  },
  "decision": "provisional",
  "confidence": "fixed_checkpoint_task_effects_not_training_seed_or_mechanism_proof",
  "global_best_changed": false,
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-task-effects/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-task-effects/reports/interpretation.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-task-effects/result-review.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-task-effects/memory-consolidation-status.json"
  ],
  "next_question": "Does a task-specific early/final readout improve beyond a shared readout mixture while preserving both original cycles?"
}
```

</details>
