## 226. matched-maturity-history--mature_r1--s20262027

- 假设：Recurrence gain may depend on action maturity under matched strict history support.
- 判定：retain_reference
- 原因：Matched maturity/depth R1 reference; R2 gain interval crosses zero.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--mature_r1--s20262027",
  "parent_experiment_id": null,
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "mature_r1",
  "code_id": "547d90bf0c6acc6513ecfb9ea55844c34f70047492832f6a951b9c40a1bc9c49",
  "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support.",
  "change": "{\"model\": \"QFormerCross8MaturityDepth\", \"maturity_mode\": \"mature\", \"work_steps\": 1, \"sequence_loop_steps\": null, \"history_version_mode\": null, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MaturityDepth",
    "maturity_mode": "mature",
    "work_steps": 1,
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
      "value": 0.7661872388516551,
      "delta": null
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18424935586113908,
        "AUC": 0.6492681652672118
      },
      "cart": {
        "logloss": 0.02190033426191309,
        "AUC": 0.7928151781120835
      },
      "fav": {
        "logloss": 0.016900563788149952,
        "AUC": 0.799412221081879
      },
      "buy": {
        "logloss": 0.009829727751449309,
        "AUC": 0.8232533909454463
      }
    },
    "macro_source_logloss": 0.058219995415662855,
    "matched_parent_effect": null,
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "formal_worker_cost": {
      "wall_seconds": 2390.003037005663,
      "cpu_seconds": 2898.495570814,
      "peak_rss_bytes": 17044246528,
      "gpu_peak_bytes": 23166189568,
      "samples": 2192
    },
    "forward_latency": {
      "mean_ms": 11.372623252868653,
      "p50_ms": 11.367936134338379,
      "p95_ms": 11.411023950576782,
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
  "decision_reason": "Matched maturity/depth R1 reference; R2 gain interval crosses zero.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/Taobao_Action/mature_r1.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/result.json",
    "result_sha256": "f20cc0055017503b428368f2db8ab676ada98fca7f5f6f91dbe99cd480b26f85",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/manifest.json",
    "manifest_sha256": "b0cd61cf86b51afa3dcf9d3bde93a987e464abf1ebde4a1577714dac8c67b2f6",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r1/Taobao_Action/mature_r1.model",
    "checkpoint_sha256": "5d0f2ad86760cb3a208d4eae71cf9728e20871f706cd19cd9ea7aae975dc9792",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
