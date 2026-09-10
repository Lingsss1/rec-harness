---
name: model-diagnostic-tool-evolver
description: Discover, implement, validate, and promote new diagnostic tools for Codex-driven GPU ranking and recommendation model research. Use when model structure changes, user or simulator complaints expose missing measurements, an existing diagnostic remains ambiguous, Codex repeatedly performs the same ad-hoc analysis, a plateau appears caused by missing measurements, a paper-derived method could reduce experiment waste, or additional information is likely to improve optimization decisions. Search current primary literature, build candidates in isolation, run correctness and real-data utility experiments, and promote only evidence-gated tools into $model-diagnostic-toolkit.
---

# Model Diagnostic Tool Evolver

Turn a decision-relevant information gap into a reusable diagnostic tool. Keep candidate evolution separate from stable tools and from long-term model-optimization memory.

## Close the model–diagnosis loop

At each model round boundary, compare structural changes and new user/simulator issues against existing diagnostic coverage. Automatically propose a falsifiable candidate for each qualified uncovered gap; preserve the trigger and parent code/protocol identities. Read [references/co-evolution-loop.md](references/co-evolution-loop.md) for proposal design, runnable commands, schemas and the next-round contract.

Use `scripts/coevolve.py propose` to persist the proposal, then the validation ladder below. After acceptance, run the probe on the current validation context and use `handoff` only for a quantitatively supported finding. The consuming model-research agent must `check` the handoff before making the bounded code change, evaluate under the frozen protocol, and `close` it with the experiment outcome. Null findings and unvalidated probes cannot drive edits. Automatically continue within the user's existing model-edit and experiment budget authorization; queue work whose artifacts or budget are unavailable.

## Start only for a qualified gap

Write the target decision before searching papers:

```text
observed ambiguity -> missing information -> decision it could change -> cheapest falsifying test
```

Start a candidate only when all are true:

- the gap affects experiment selection, interpretation, or GPU budget;
- the existing `$model-diagnostic-toolkit` cannot answer it;
- the required artifacts can be obtained without test-set leakage;
- expected reuse or saved experiment cost justifies implementation;
- a validation result can prove the method adds information.

Do not create a tool for a one-off formatting task, a result already available in logs, or an unfalsifiable intuition.

## Create an isolated candidate

Initialize the audit trail inside the experiment workspace:

```bash
python3 <skill-dir>/scripts/manage_candidate.py init WORKSPACE \
  --id TOOL_ID \
  --gap "MISSING_INFORMATION" \
  --decision "DECISION_THE_TOOL_MAY_CHANGE" \
  --baseline "CURRENT_NO_TOOL_OR_EXISTING_TOOL_BASELINE" \
  --budget "PRECOMMITTED_VALIDATION_BUDGET"
```

This creates `memory/tool-evolution/candidates/TOOL_ID/` and an append-only `registry.jsonl`. Never prototype directly inside the installed diagnostic Skill.

Read [references/research-validation-protocol.md](references/research-validation-protocol.md) before literature search or experimentation. Read [references/candidate-schema.md](references/candidate-schema.md) when updating candidate state.

## Search and distill the literature

1. Search the current literature because relevant methods and implementations change.
2. Prefer original papers, proceedings pages, official repositories, and official documentation. Use surveys only to discover primary sources.
3. Compare 3–8 plausible methods when available; record negative search results when fewer exist.
4. Extract the exact measured quantity, assumptions, estimator, required tensors, complexity, known failure modes, and available license.
5. Separate `paper_claim`, `agent_inference`, and `local_result` in the method card.
6. Choose the smallest method that can falsify the information-gap hypothesis. Do not select by novelty alone.

Transition to `researched` only after the method card cites primary sources and defines a no-tool/existing-tool baseline.

## Implement the minimum candidate

- Use framework-independent NumPy arrays or long-form tables at the analysis boundary when practical.
- Keep GPU instrumentation repository-local and thin; keep aggregation deterministic and testable offline.
- Define stable input/output schemas and explicit metric sign conventions.
- Return machine-readable JSON plus a compact human-readable report.
- Handle empty inputs, shape mismatches, non-finite values, seeds, and large-array limits.
- Avoid new dependencies unless they materially reduce risk. Request approval before installing packages or cloning code.
- Preserve source citations and comply with code licenses; reimplement equations when reuse terms are unclear.

Do not change the training objective, validation split, or incumbent while testing a diagnostic candidate.

## Run the validation ladder

Precommit acceptance gates in `candidate.json` before observing results, then test in order:

1. `paper_fidelity`: match equations, reference outputs, or analytically known toy cases.
2. `synthetic_correctness`: recover injected signal and return null on negative controls.
3. `robustness`: pass edge cases, determinism, shape, non-finite, runtime, and memory checks.
4. `shadow_real_data`: run read-only on at least two relevant checkpoints, seeds, or experiment conditions.
5. `incremental_utility`: beat the no-tool and existing-tool baselines on a predeclared decision-utility measure.
6. `cost`: stay inside the declared artifact, runtime, GPU-memory, and maintenance budget.

Record each gate:

```bash
python3 <skill-dir>/scripts/manage_candidate.py gate WORKSPACE \
  --id TOOL_ID --gate GATE --result pass \
  --evidence PATH --note "COMPACT_RESULT"
```

Use the configured Harness budget. When no real-data experiment budget exists, stop after offline validation, mark the candidate `synthetic_pass`, and queue the shadow gate instead of launching an expensive run.

## Accept, reject, or revise

- `accepted`: every required gate passes and the result is nonredundant, reproducible, decision-relevant, and affordable.
- `rejected`: correctness fails, information is redundant, utility gate fails, assumptions do not hold, or cost exceeds value.
- `superseded`: a later accepted tool covers the same decision contract more effectively.

Use a new candidate ID for a materially different estimator. Do not silently retune acceptance thresholds after seeing results. Preserve rejected candidates and their evidence so Codex does not repeat them.

## Promote an accepted tool

After `manage_candidate.py transition ... --to accepted` succeeds:

1. Copy only the minimal stable analyzer, tests, and method reference into the source tree of `$model-diagnostic-toolkit`.
2. Add its input route, output contract, decision boundary, and bundled-tool entry to the toolkit `SKILL.md`.
3. Run candidate unit tests, a realistic fixture, `py_compile`, and Skill validation.
4. Confirm the new tool does not regress existing toolkit tests or duplicate an existing output.
5. Sync the validated source Skill into the Codex installation; request approval if the destination is outside the writable workspace.
6. Append a `promoted` event with source paths, code hash, paper citations, and rollback point.
7. Send the accepted experiment digest to `$model-optimization-harness-updater` only when it changes future experiment policy; do not promote a one-off tool result directly into a durable model lesson.

Never promote candidate artifacts, downloaded repositories, raw logs, or paper PDFs into the stable toolkit.

## Report the evolution outcome

Return:

- information gap and target decision;
- papers considered and why the chosen method fits;
- implementation and artifact schemas;
- gate results against both baselines;
- accepted/rejected/synthetic-pass status;
- promoted files or the remaining real-data gate;
- runtime/GPU cost and rollback point.
