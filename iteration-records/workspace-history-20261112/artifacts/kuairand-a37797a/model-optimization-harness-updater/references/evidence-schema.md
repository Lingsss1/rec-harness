# Experiment evidence schema

Keep raw logs and large artifacts outside memory. Memory stores compact facts with stable paths and identities so Codex can recover details without loading full logs into context.

## Episode digest

Append one object per attempted experiment to `memory/episodes.jsonl`:

```json
{
  "experiment_id": "exp-0007",
  "parent_experiment_id": "exp-0003",
  "timestamp": "2026-08-28T12:00:00+08:00",
  "code_id": "git-commit-or-content-hash",
  "dataset_id": "dataset-version-and-split",
  "hypothesis": "Increasing cross depth improves long-tail NDCG without hurting head AUC.",
  "change": "cross_depth: 2 -> 3",
  "family": "feature_interaction_depth",
  "budget": {"epochs": 20, "seeds": [0, 1, 2]},
  "metrics": {
    "primary": {"name": "val_ndcg@10", "value": 0.4123, "delta": 0.0018},
    "guardrails": {"val_auc": 0.781, "peak_vram_mb": 18140, "runtime_sec": 2530}
  },
  "diagnostics": [
    {"kind": "paired_shuffle", "summary": "mean loss delta +0.0041", "path": "artifacts/exp-0007/perturbation_report.json"}
  ],
  "validity": {"valid": true, "violations": [], "notes": []},
  "decision": "keep",
  "confidence": "provisional",
  "artifact_paths": ["logs/exp-0007.log"],
  "falsification_result": "passed",
  "next_question": "Does the gain survive a second dataset?"
}
```

Allowed decisions are `keep`, `discard`, `provisional`, `crash`, and `invalid`. Append an episode even when the run fails.

## Failure digest

Append exact, reusable avoids to `memory/failures.jsonl`:

```json
{
  "failure_id": "fail-0004",
  "experiment_id": "exp-0008",
  "kind": "oom",
  "signature": "CUDA out of memory at attention logits",
  "scope": {"model": "ranker-v3", "batch_size": 2048, "sequence_length": 256},
  "reproduction": "python train.py --config configs/exp-0008.yaml",
  "likely_cause": "quadratic attention allocation exceeded 24GB",
  "avoid": "Do not combine batch_size>=2048 and sequence_length>=256 on 24GB GPU.",
  "artifact_path": "logs/exp-0008.log",
  "superseded_by": null
}
```

Execution failure is not evidence against the scientific hypothesis.

## Lesson object

Store active, consolidated knowledge in `memory/lessons.json`:

```json
{
  "schema_version": 1,
  "last_consolidated_episode": "exp-0009",
  "lessons": [
    {
      "lesson_id": "lesson-0002",
      "statement": "On sparse long-tail data, deeper cross layers help only with stronger regularization.",
      "scope": {"dataset_regime": "sparse-long-tail", "component": "cross_network"},
      "status": "supported",
      "confidence": 0.72,
      "supporting_episode_ids": ["exp-0004", "exp-0007"],
      "counterevidence_episode_ids": ["exp-0005"],
      "falsification_condition": "No gain across three seeds after matched regularization search.",
      "use_count": 1,
      "successful_use_count": 1,
      "last_used_at": "2026-08-28T12:30:00+08:00",
      "supersedes": null
    }
  ]
}
```

Statuses are `provisional`, `supported`, `confirmed`, `demoted`, and `superseded`. A new lesson requires two independent valid outcomes for `supported`; `confirmed` requires mechanism evidence plus retrained validation evidence.

## Strategy state

`memory/strategy.json` is mutable working state:

```json
{
  "schema_version": 1,
  "incumbent": {"experiment_id": "exp-0007", "code_id": "abc123", "primary_metric": 0.4123},
  "phase": "exploit",
  "plateau": {"consecutive_misses": 1, "recent_slope": 0.0004},
  "active_hypothesis": "Regularized depth improves long-tail interactions.",
  "exhausted_families": ["embedding_width_local_search"],
  "next_queue": [
    {"hypothesis": "Retune regularization at depth 3", "cost": "medium", "gate": ">=0.001 val_ndcg@10 over incumbent"}
  ],
  "rejected_memory_edits": [],
  "last_consolidated_episode": "exp-0009"
}
```

Do not treat this file as history. Reconstruct history from the append-only ledgers.

## Contradictions and supersession

Do not average incompatible model basins, datasets, budgets, widths, or objectives. Narrow lesson scope or leave the question unresolved. When evidence reverses a lesson, keep both evidence sets and use `supersedes`/`superseded_by`; never rewrite past episodes.
