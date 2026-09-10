# Trigger → probe → evidence → model edit

This workflow adapts [Self-EvolveRec §3.3](https://arxiv.org/abs/2602.12612), which evolves diagnosis from structural changes and qualitative user feedback, then feeds quantitative findings into code evolution. The paper motivates the two triggers; the six validation gates, file fingerprints and experiment receipts below are local engineering choices, not claims that the paper specifies this protocol.

## Observe and automatically propose

At each round boundary, Codex compares the parent and current code (execution flow, added/removed modules, representations, losses and input contracts), and reads new user complaints or grounded simulator feedback. Produce a compact structural report with both code identities; preserve the original complaint and its user/history/list evidence. A complaint is a hypothesis, never quantitative proof.

For each decision-relevant gap, first check stable toolkit coverage and the candidate registry. Reuse a compatible accepted probe; record a coverage decision when an existing measurement suffices. Avoid re-proposing rejected estimators without new evidence. A new component alone does not establish a defect. Removed components retire incompatible probes; changed tensor or metric semantics require revalidation under a new candidate/context.

When coverage is missing, **automatically formulate** the smallest falsifiable probe and persist it with `coevolve.py propose`. Codex does the semantic analysis and open-ended probe design; the script validates and records the proposal, and does not call an LLM or infer metrics from keywords. Do not ask for permission just to propose a probe within an authorized research task. If data or budget is missing, record what is needed and leave the candidate pending.

Example proposal JSON (paths are relative to the experiment workspace):

```json
{
  "trigger_id": "round-12-diversity",
  "kind": "user_issue",
  "observation": "The validation recommendation lists repeat the same category",
  "source": "reports/user-feedback.json",
  "component": "ranking-head",
  "code_identity": "parent-commit-and-dirty-diff-hash",
  "protocol_id": "frozen-validation-v1",
  "gap": "No measurement distinguishes category concentration from user preference",
  "decision": "Whether to test category-aware reranking",
  "baseline": "No-tool accuracy decisions and existing collapse/perturbation reports",
  "budget": "CPU analysis plus two preallocated validation conditions",
  "measurement": "Top-k category entropy relative to matched user-history controls",
  "null_behavior": "No residual deficit under matched controls means no edit",
  "action": "Test one category-aware reranking change on an isolated incumbent copy",
  "falsification": "Discard if fixed validation quality or diversity guardrail fails"
}
```

For a structural trigger use `kind: structural_change`; source points to the structural report. For example a new temporal encoder suggests a paired sequence-order perturbation with identical weights/batches and a null control. These are examples, not a fixed probe catalog.

```bash
python3 <evolver>/scripts/coevolve.py propose WORKSPACE --input PROPOSAL.json
```

The returned candidate ID is deterministic for an identical proposal. Replays preserve existing state. The proposal is a design sketch; complete the method card, paper search, implementation and precommitted decision contract using `manage_candidate.py`. Its six gates remain mandatory. Record gate evidence as existing workspace files; each receives a SHA-256 fingerprint. Terminal candidates cannot rewrite gates. New evidence/estimators use a new candidate ID.

## Quantitative feedback and next-round handoff

Tool acceptance establishes measurement reliability, not a model defect or a model winner. Run the accepted probe read-only on the proposal's current code and frozen validation protocol. Interpret the output against the precommitted action mapping, including controls and null behavior; record a report:

```json
{
  "code_identity": "parent-commit-and-dirty-diff-hash",
  "protocol_id": "frozen-validation-v1",
  "split": "validation",
  "conclusion": "supported",
  "finding": "Measured deficit exceeds the precommitted threshold under matched controls",
  "evidence": "reports/diversity-measurements.json"
}
```

`conclusion` may instead be `null`, `contradicted`, or `inconclusive`: retain those reports as evidence, but do not queue an edit. A pass flag is a recorded research assessment; scripts verify lifecycle and artifact integrity, not scientific truth. The agent must inspect raw measurements, controls, leakage and thresholds before declaring support.

```bash
python3 <evolver>/scripts/coevolve.py handoff WORKSPACE --id PROBE_ID --input WORKSPACE/reports/finding.json
python3 <evolver>/scripts/coevolve.py check WORKSPACE --task-id EDIT_ID \
  --code-identity CURRENT_CODE_ID --protocol-id CURRENT_PROTOCOL_ID
```

Reports and all referenced evidence must reside in WORKSPACE. A handoff freezes gate evidence, candidate state, trigger source and finding report. `check` is mandatory immediately before editing: reject stale code/protocol, changed artifacts, superseded candidates and completed tasks. Repeating an unchanged handoff returns the existing task without duplicating it. Old candidates without fingerprinted gates remain readable but cannot drive this new handoff; revalidate a new candidate.

The consuming Codex loads ready tasks from `memory/tool-evolution/handoffs/`, checks them, and adds the action, task ID, cost and falsification condition to `strategy.json`'s next hypothesis queue. Within the user's existing model-edit/training authorization, execute one bounded change on an isolated/snapshotted incumbent and run the frozen validation comparison. A handoff never authorizes a test-set-driven decision or automatically replaces the incumbent. Missing training budget leaves the task queued.

## Close and trigger the next round

Append the normal results/episode records through the Harness workflow, including `diagnostic_task_id`, candidate ID and supporting evidence. Export a compact episode receipt:

```json
{
  "episode_id": "episode-13",
  "task_id": "EDIT_ID",
  "outcome": "discard",
  "parent_code_identity": "parent-commit-and-dirty-diff-hash",
  "result_code_identity": "trial-commit-and-dirty-diff-hash",
  "protocol_id": "frozen-validation-v1",
  "evidence": "reports/paired-retraining-evaluation.json"
}
```

```bash
python3 <evolver>/scripts/coevolve.py close WORKSPACE --task-id EDIT_ID \
  --code-identity PARENT_CODE_ID --protocol-id CURRENT_PROTOCOL_ID --input EPISODE.json
```

Outcomes are `keep`, `discard`, `provisional`, `crash`, `invalid`. The receipt links the outcome to the probe but does not itself validate training metrics or mutate the incumbent. The agent applies existing keep/discard rules and rolls back unsuccessful trials. Rescan the resulting incumbent and new complaints at the next round; preserve discarded trials and null findings. Promote durable lessons only under the existing multi-run evidence rules. This repository supplies Skills and audit helpers, not a background training runtime.
