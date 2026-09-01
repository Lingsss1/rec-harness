# Native delegation contract

Use Codex-native subagents to widen bounded research work without turning the Harness into an agent runtime.

## Qualification

Delegate only when at least two workstreams are independent, have explicit stopping conditions, and can return useful results without writing global research state. Good targets include Arm-local Idea Burst, local versus external evidence search, independent code-module inspection, competing explanations, and read-only candidate review.

Keep work in the root when it updates the Research Graph, posterior, selection receipt, incumbent, protocol, lessons, experiment validity, or training controller. Small or tightly coupled tasks do not benefit from delegation.

## Context pack

Give each task only its objective, typed lineage, verified local facts, explicitly unverified diagnoses, information gap, allowed tools/sources, forbidden actions, write scope, output schema, and stop condition. Do not pass the full research history. A pack is temporary unless it was actually dispatched and influenced a decision; persist its hash when retained.

## Artifact contract

A delegated artifact reports:

```json
{
  "task_id": "...",
  "status": "complete|abstained|blocked",
  "claims": [],
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

## Root reconciliation

The root Codex verifies source identity, local applicability, code claims, contradictions, and protocol compliance. It records accepted and rejected claims, conflicts, resolutions, evidence references, and the root decision in one reconciliation receipt. Only accepted, attributable claims may enter Investigation evidence. Register a formal Hypothesis only after this gate.

## Code worktrees

Use isolated worktrees only for multiple evidence-gated candidates that require conflicting code changes. The root creates and assigns them, fixes each task's working directory and write scope, reviews diffs and tests, and performs any integration. Subagents do not merge, launch training, or alter shared evaluation files. Prefer ordinary configuration candidates or read-only shared-workspace analysis when isolation adds no value.
