# Native delegation contract

Use Codex-native subagents to widen bounded research work without turning the Harness into an agent runtime.

## Qualification

Delegate only when at least two workstreams are independent, have explicit stopping conditions, and can return useful results without writing global research state. Good targets include Arm-local Idea Burst, local versus external evidence search, independent code-module inspection, competing explanations, and read-only candidate review.

Keep work in the root when it updates the Research Graph, posterior, selection receipt, incumbent, protocol, lessons, experiment validity, or training controller. Small or tightly coupled tasks do not benefit from delegation.

## Context pack

Give each task only its objective, typed lineage, verified local facts, explicitly unverified diagnoses, information gap, allowed tools/sources, forbidden actions, write scope, output schema, and stop condition. Do not pass the full research history. A pack is temporary unless it was actually dispatched and influenced a decision; persist its hash when retained.

Planning and dispatch are distinct. Previewing packs is read-only. Dispatch freezes the exact context hash and changes the advisory lifecycle from `planned` to `dispatched`; only dispatched work requires reconciliation. Unused or unsuitable plans should be cancelled. A non-parallel plan must never block local Hypothesis formation.

## Artifact contract

A delegated artifact reports:

```json
{
  "task_id": "...",
  "context_hash": "...",
  "artifact_hash": "...",
  "artifact_path": "...",
  "status": "complete|abstained|blocked",
  "claims": [
    {"claim_id": "...", "claim": "...", "evidence_refs": ["..."]}
  ],
  "evidence_refs": [],
  "counterevidence": [],
  "competing_explanations": [],
  "confidence": 0.0,
  "methods": [],
  "uncertainties": [],
  "recommended_next_action": "",
  "protocol_violations": []
}
```

Missing provenance keeps a claim at brainstorming status. Multiple agents repeating a claim does not strengthen its scientific weight.

Compute `artifact_hash` over the canonical artifact content excluding the hash field itself. Reject artifacts whose task ID is unknown, context hash is stale, artifact hash does not verify, or schema is incomplete.

## Root reconciliation

The root Codex verifies source identity, local applicability, code claims, contradictions, and protocol compliance. Every accepted claim names its `source_task_id` and `source_claim_id`; accepted evidence must be a non-empty subset of that source claim's evidence. It records accepted and rejected claims, conflicts, resolutions, evidence references, and the root decision in one reconciliation receipt. Persist the full receipt in an append-only event log and keep only its hashes, paths, accepted summaries, and lineage in the scheduling projection. Only accepted, attributable claims may enter Investigation evidence. Register a formal Hypothesis only after this gate.

Use the Harness `next` projection before deciding the next workflow transition. It is deterministic and read-only; use its state hash as an optimistic-concurrency guard on subsequent root mutations.

## Code worktrees

Use isolated worktrees only for multiple evidence-gated candidates that require conflicting code changes. The root creates and assigns them, fixes each task's working directory and write scope, reviews diffs and tests, and performs any integration. Subagents do not merge, launch training, or alter shared evaluation files. Prefer ordinary configuration candidates or read-only shared-workspace analysis when isolation adds no value.
