## 180. taobao-optimizer-layout-synthetic-replay

- 假设：New parameter layouts can alter old updates through clip,Q initialization and stochastic probe assignment despite matching initial old weights/raw gradients.
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "taobao-optimizer-layout-synthetic-replay",
  "timestamp": "2026-09-09T08:27:06.498176+00:00",
  "dataset_id": "synthetic_optimizer_inputs_no_dataset",
  "family": "optimizer_layout_single_step_replay",
  "code_id": "0ac44b0dc5b1e84b9cc4cff4c9d7f08514a49800d91f0703e6b60b1d1f0d0621",
  "candidate_id": "probe-302bf4e60eb7a4e126b2d1b1",
  "hypothesis": "New parameter layouts can alter old updates through clip,Q initialization and stochastic probe assignment despite matching initial old weights/raw gradients.",
  "change": "12 formula-input CPU eager one-step comparisons;shared8-cell C/S/R factorial plus task endpoints and baseline observation control.",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": [],
    "optimizer_replay_steps": 12,
    "max_wall_seconds": 120
  },
  "metrics": {
    "baseline_exact": true,
    "shared111_old_update_and_state_exact": true,
    "task111_old_update_and_state_exact": true,
    "isolated_evidence": {
      "clip": {
        "baseline_raw_norm": 1.260119080543518,
        "shared_raw_norm": 4.193792819976807,
        "native_pre_update_relative_error": 0.3934230099865539,
        "fixed_C_pre_update_exact": true,
        "natural_Q_response_contrast": "shared_000 versus shared_100, pre.weight only; first bucket has unchanged probe assignment and no added member",
        "contrast": "shared_011 versus shared_111: S and R fixed, change only old clipped gradients",
        "old_update_relative_error_with_Q_and_V_fixed": {
          "pre.weight": 0.6547437975341337,
          "tower.0.bias": 0.6536405246355629,
          "tower.1.bias": 0.6627762926256765,
          "tower.2.bias": 0.6636027967632964,
          "tower.3.bias": 0.6232987589184004,
          "work.weight": 0.5996865849450637
        }
      },
      "bucket_initialization": {
        "contrast": "shared_101 versus shared_111: old clipped gradients and probes matched; fix old Q slices only",
        "baseline_bias_scale": 1.3041059627549998,
        "shared_bias_scale_with_C_fixed": 1.1143194200276063,
        "old_bias_update_relative_error_without_S": {
          "tower.0.bias": 0.2233162813954383,
          "tower.1.bias": 0.22996674708221948,
          "tower.2.bias": 0.25076596747584257,
          "tower.3.bias": 0.20384915162779543
        }
      },
      "random_probes": {
        "contrast": "shared_110 versus shared_111: old clipped gradients/Q initial slices matched; old V fixing restores updates",
        "old_update_relative_error_without_R": {
          "pre.weight": 0.0,
          "tower.0.bias": 0.1854686718719951,
          "tower.1.bias": 0.20056073547629916,
          "tower.2.bias": 0.27032759839736037,
          "tower.3.bias": 0.0118712153006775,
          "work.weight": 0.14404810105279392
        }
      }
    },
    "cost": {
      "parent_observed_wall_seconds": 3.0219235494732857,
      "worker_tracked_cpu_seconds": 3.342409,
      "worker_final_recorded_resources": {
        "artifact_bytes": 7434864,
        "cpu_seconds": 3.342409,
        "optimizer_replay_steps": 12,
        "rss_bytes": 682704896,
        "wall_seconds": 2.5927098616957664
      },
      "sampled_worker_tree_peak": {
        "cpu_seconds": 3.66,
        "live_tree_members": 1,
        "rss_bytes": 694075392
      },
      "artifact_bytes": 7440813,
      "scope": "Parent wall includes worker lifecycle; CPU and memory are recorded worker and sampled tree measurements, not an exact aggregate of every process shutdown cost."
    }
  },
  "validity": {
    "valid": true,
    "violations": [],
    "test_evaluated": false,
    "scope": "Fixed synthetic CPU eager single-step mechanism;not native compiled,multistep or AUC causation"
  },
  "decision": "provisional",
  "confidence": "verified_controlled_synthetic_mechanism_only",
  "global_best_changed": false,
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/optimizer-layout-replay/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/optimizer-layout-replay/reports/interpretation.json",
    "workspace/qformer-loop/taobao-transfer/optimizer-layout-replay/reports/replay-result-review.json",
    "workspace/qformer-loop/taobao-transfer/optimizer-layout-replay/reports/candidate-after-synthetic.json"
  ],
  "next_question": "Can direction and norm changes in the second query be separated using original saved validation artifacts before choosing a parameter-free retrieval variant?"
}
```

</details>
