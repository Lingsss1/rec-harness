---
name: model-optimization-harness-updater
description: Update persistent research memory for a Codex-driven GPU model-optimization loop after multiple experiments. Use when repeated ranking/recommendation trials, diagnostics, or ablations should be consolidated into an autoresearch-style keep/discard ledger, episodic evidence, recurring lessons, failure avoids, plateau state, and next-hypothesis guidance. Evolve memory only by default; do not modify diagnostic tools, Skills, prompts, evaluation code, or test protocol unless the user explicitly requests it.
---

# Model Optimization Harness Updater

Evolve the research Harness through memory, not by continuously rewriting the tool or Skill. Use `$model-diagnostic-toolkit` to produce measurements, then update bounded memory from verified experiment outcomes.

## Memory contract

Locate or create this structure inside the experiment workspace:

```text
memory/
├── results.tsv
├── episodes.jsonl
├── failures.jsonl
├── lessons.json
└── strategy.json
```

- `results.tsv`: fast autoresearch-style ledger for every run.
- `episodes.jsonl`: immutable, append-only experiment digests.
- `failures.jsonl`: exact reproducible crashes, OOMs, invalid runs, and protocol violations.
- `lessons.json`: consolidated conditional knowledge supported across rounds.
- `strategy.json`: current incumbent, search phase, plateau state, active hypothesis, and next queue.

When memory does not exist, copy the bundled `assets/results.template.tsv`, `assets/lessons.template.json`, and `assets/strategy.template.json`; create empty `episodes.jsonl` and `failures.jsonl`. Read [references/memory-design.md](references/memory-design.md) for the architecture and [references/evidence-schema.md](references/evidence-schema.md) for record schemas.

## Per-experiment loop

1. Read the incumbent, recent episodes, relevant lessons, failure avoids, and current strategy.
2. Propose one falsifiable change. Keep the evaluation split, budget, and primary metric fixed.
3. Snapshot the incumbent code before editing.
4. Run training with logs redirected to a file; extract only compact metrics into context.
5. Classify the run as `keep`, `discard`, `provisional`, `crash`, or `invalid`.
6. Append one row to `results.tsv` and one full digest to `episodes.jsonl` regardless of outcome.
7. Keep the code only when it passes the configured validation gate and guardrail metrics. Otherwise restore the incumbent.
8. Do not update durable lessons after every run. Continue experimentation until a consolidation trigger occurs.

Use `provisional` for architecture jumps that require a fixed retuning window before fair comparison.

## Consolidation triggers

Update `lessons.json` and `strategy.json` after any of:

- 5 valid experiments;
- two independent runs support the same mechanism;
- the incumbent improves;
- three consecutive valid experiments fail to improve;
- the same failure pattern repeats;
- the active hypothesis family is exhausted;
- the user requests a checkpoint review.

## Consolidation procedure

### 1. Reconstruct evidence

Read all new episodes since `last_consolidated_episode`. Do not infer from the current code alone; discarded and failed experiments are part of the evidence.

### 2. Produce textual gradients

For each repeated pattern, write a short correction direction:

```text
observation → likely mechanism → bounded next change → falsifying metric
```

Accumulate only recurring patterns as momentum. One-off noise remains episodic.

### 3. Apply bounded memory edits

Propose small `add`, `replace`, `demote`, or `delete` operations on individual lessons. Every lesson must include scope, supporting episode IDs, confidence, counterevidence, and a falsification condition.

### 4. Gate the update

Accept a new lesson as `supported` only after at least two independent valid outcomes agree. Mark it `confirmed` only when mechanism evidence and retrained validation outcome agree. Otherwise keep it `provisional`.

Store rejected memory edits in `strategy.json.rejected_memory_edits` so later consolidation does not repeat them blindly.

### 5. Update search strategy

Refresh:

- incumbent identity and score;
- recent improvement slope;
- phase: `explore`, `exploit`, `retune`, or `plateau_escape`;
- exhausted hypothesis families;
- next ranked hypotheses with estimated cost and falsification gate.

When plateauing, change the hypothesis family or diagnostic level instead of retrying nearby scalar values indefinitely.

If plateau evidence points to a recurring measurement gap rather than an exhausted model hypothesis, invoke `$model-diagnostic-tool-evolver`. Keep its candidate registry separate from model lessons until a new diagnostic passes real-data utility gates.

### 6. Audit memory size

Demote or delete lessons that are duplicated, contradicted, never retrieved, repeatedly unhelpful, or too specific to an obsolete model basin. Preserve their original evidence in `episodes.jsonl`; deletion affects the active summary, not history.

## Retrieval policy for Codex

Before proposing the next experiment, load only:

- `strategy.json`;
- the last 5–10 episodes;
- lessons matching the current model component, metric, or failure mode;
- exact avoids relevant to the proposed edit.

Do not load full training logs or the entire historical memory into context. Use paths and stable IDs to recover details when needed.

## Conditional native delegation

The Harness is not an agent runtime. When an investigation divides into independent, context-bounded workstreams, Codex may use its available native subagent capability. The root Codex remains the only research-state writer: it creates minimal context packs, validates returned evidence and sources, resolves conflicts, and performs one reconciled Harness update. Subagents may return candidate evidence or isolated implementation artifacts; they must not update posteriors, select an incumbent, modify the protocol, or launch training.

Delegation is optional and capability-dependent. Do not create permanent roles or require subagents for coupled, small, destructive, or state-mutating work. Agent agreement is review signal, never independent scientific evidence. For the context-pack, artifact, reconciliation, and isolated-worktree contracts, read [references/native-delegation.md](references/native-delegation.md) only when delegation is actually being considered.

When the target Harness exposes a deterministic `next` action, use it before open-ended routing. Treat it as a read-only workflow gate, preserve its decision receipt, and use its state hash to reject stale mutations. Planning delegation does not mean dispatching it: only explicitly dispatched tasks require root reconciliation.

## Acceptance rules

- Choose one primary validation metric and explicit guardrails before the run.
- Never route experiments or update memory from held-out test metrics.
- Prefer simpler code when validation is statistically indistinguishable.
- Treat OOM, timeout, and crash as execution evidence, not proof that the scientific hypothesis is false.
- Do not promote activation×gradient ranks, collapse flags, or Shuffle effects directly into model winners; they guide which experiment to run.
- Do not overwrite old episodes or failures. Use supersession links for changed conclusions.
- Keep memory and code changes independently reversible.

## Deliver the checkpoint review

Report:

- experiments since the previous consolidation;
- incumbent changes and validation deltas;
- kept, discarded, provisional, crashed, and invalid counts;
- lessons added, strengthened, demoted, or deleted;
- plateau diagnosis and next hypothesis queue;
- files updated and rollback point.
