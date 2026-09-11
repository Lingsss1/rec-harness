## 224. matched-maturity-history--released_r1--s20262027

- 假设：Recurrence gain may depend on action maturity under matched strict history support.
- 判定：retain_reference
- 原因：Strict-past released-label matched R1 reference; do not compare directly to old release protocol.

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "matched-maturity-history--released_r1--s20262027",
  "parent_experiment_id": null,
  "timestamp": "2026-09-09T20:58:43.632223+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_type": "formal_training",
  "family": "released_r1",
  "code_id": "547d90bf0c6acc6513ecfb9ea55844c34f70047492832f6a951b9c40a1bc9c49",
  "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support.",
  "change": "{\"model\": \"QFormerCross8MaturityDepth\", \"maturity_mode\": \"released\", \"work_steps\": 1, \"sequence_loop_steps\": null, \"history_version_mode\": null, \"num_queries\": 4}",
  "config_change": {
    "model": "QFormerCross8MaturityDepth",
    "maturity_mode": "released",
    "work_steps": 1,
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
      "value": 0.8076317440897383,
      "delta": null
    },
    "per_task": {
      "is_click": {
        "logloss": 0.18416858573464248,
        "AUC": 0.6520517645921935
      },
      "cart": {
        "logloss": 0.01935907669234433,
        "AUC": 0.8427635185714344
      },
      "fav": {
        "logloss": 0.014490376215334775,
        "AUC": 0.8615870776269045
      },
      "buy": {
        "logloss": 0.007765181240830164,
        "AUC": 0.8741246155684206
      }
    },
    "macro_source_logloss": 0.05644580497078794,
    "matched_parent_effect": null,
    "parameter_count": 51724260,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "formal_worker_cost": {
      "wall_seconds": 2387.3118413984776,
      "cpu_seconds": 2889.456151949,
      "peak_rss_bytes": 17057886208,
      "gpu_peak_bytes": 23166189568,
      "samples": 2192
    },
    "forward_latency": {
      "mean_ms": 11.438752007484435,
      "p50_ms": 11.368256092071533,
      "p95_ms": 11.52149748802185,
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
  "decision_reason": "Strict-past released-label matched R1 reference; do not compare directly to old release protocol.",
  "confidence": "single_seed_exploratory_paired_users_no_multiplicity_or_selection_correction",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/manifest.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/Taobao_Action/released_r1.model",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  ],
  "source_identity": {
    "run_dir": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1",
    "result": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/result.json",
    "result_sha256": "388a958bb69de83038489a4fd94ad4fb42b3f0a1b0db6cbd147bbb532b2b5c69",
    "manifest": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/manifest.json",
    "manifest_sha256": "9c308e243f9ab08b5090ba293947c94e3162d822c5af6dbe689c5a1470d80df6",
    "checkpoint": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/released_r1/Taobao_Action/released_r1.model",
    "checkpoint_sha256": "69906114f951801228cbc527c113977ea64598626b9050cf04d13fc5685430a0",
    "joint_paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "final_report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
  },
  "global_best_changed": false
}
```

</details>
