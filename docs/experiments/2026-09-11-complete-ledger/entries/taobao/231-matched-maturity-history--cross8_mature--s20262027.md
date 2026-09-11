## 231. matched-maturity-history--cross8_mature--s20262027

- 假设：The depth and order of history representations may change second-read utility; compare against matched Cross8.
- 判定：retain_reference
- 原因：Fresh matched Cross8 baseline; architecture/width differences prevent isolating recurrence from A-X.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--cross8_mature--s20262027",
  "parent_experiment_id": null,
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "cross8_mature",
  "code_id": "db2c4b4cc7829e0693010dcddad436bc26a7ead0d1f8cae94af6cb0e0630e01b",
  "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8.",
  "change": "{\"model\": \"QFormerCross8MaturityReference\", \"maturity_mode\": \"mature\", \"work_steps\": null, \"sequence_loop_steps\": 1, \"history_version_mode\": null, \"num_queries\": 8}",
  "config_change": {
    "model": "QFormerCross8MaturityReference",
    "maturity_mode": "mature",
    "work_steps": null,
    "sequence_loop_steps": 1,
    "history_version_mode": null,
    "num_queries": 8
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
      "value": 0.7594064200954598,
      "delta": null
    },
    "per_task": {
      "is_click": {
        "logloss": 0.1843179378485924,
        "AUC": 0.6499367701429657
      },
      "cart": {
        "logloss": 0.021834411636786143,
        "AUC": 0.789206735371116
      },
      "fav": {
        "logloss": 0.016973379157766585,
        "AUC": 0.7965020110274496
      },
      "buy": {
        "logloss": 0.010007390316809587,
        "AUC": 0.8019801638403081
      }
    },
    "macro_source_logloss": 0.05828327973998868,
    "matched_parent_effect": null,
    "parameter_count": 81616356,
    "initializer_sha256": "802ad70615a37bf4da7fc220405a4b4e667aeb7275bdcbb392e7124cefe6cbd9",
    "formal_worker_cost": {
      "wall_seconds": 2522.7814552783966,
      "cpu_seconds": 3028.683010009,
      "peak_rss_bytes": 16983564288,
      "gpu_peak_bytes": 30013390848,
      "samples": 2316
    },
    "forward_latency": {
      "mean_ms": 14.525848817825317,
      "p50_ms": 14.508959770202637,
      "p95_ms": 14.638473320007325,
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
  "decision": "retain_reference",
  "decision_reason": "Fresh matched Cross8 baseline; architecture/width differences prevent isolating recurrence from A-X.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/Taobao_Action/cross8_mature.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/result.json",
    "result_sha256": "e301e2959cd9d8aa5443bdc8ee9bdee02576bc62946241c34996435bf9ef6103",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/manifest.json",
    "manifest_sha256": "f1a54a6ebd39c772cb1b858b69cb5a132d3b2773da264276c402eb689567e50f",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/runs/cross8_mature/Taobao_Action/cross8_mature.model",
    "checkpoint_sha256": "e685bda693b75c6e838a36bfa48f7f74d0fdd6fcd50cb44c81c6858b578056ae",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
