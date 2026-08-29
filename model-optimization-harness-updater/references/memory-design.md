# Memory design for Codex model research

The Harness separates durable history from a small working policy. This borrows the disciplined experiment ledger and keep/discard loop from autoresearch, while adding evidence-gated consolidation and utility-aware forgetting for long-running Codex research.

## Design principles

1. **Fixed outer loop.** Freeze the validation protocol, experiment budget, primary metric, and guardrails before search.
2. **Append-only experience.** Record every keep, discard, crash, and invalid run. The current code is not a substitute for experiment history.
3. **Reversible incumbents.** Snapshot code before each experiment; retain an edit only after its validation gate passes.
4. **Two memory speeds.** Episodes update after every run; lessons and search strategy update only at consolidation triggers.
5. **Bounded edits.** Consolidation proposes local `add`, `replace`, `demote`, or `delete` operations rather than rewriting all memory.
6. **Evidence gates.** Repeated valid outcomes are required for durable rules. Mechanism diagnostics and retrained validation serve different roles.
7. **Selective retrieval.** Codex loads current strategy, recent episodes, and only scope-matching lessons/failures.
8. **Utility-aware forgetting.** Active summaries are pruned when duplicated, contradicted, obsolete, or repeatedly unhelpful; raw episodes remain immutable.

## File responsibilities

### `results.tsv`

One-line scanning ledger inspired by autoresearch. The first line is:

```text
experiment_id\tparent_id\tcode_id\tprimary_metric\tprimary_delta\tguardrails\tpeak_vram_mb\truntime_sec\tstatus\thypothesis\tchange
```

Append immediately after each attempt. Never remove poor results.

### `episodes.jsonl`

The episodic store is the source of truth for research reasoning. Each line contains one complete but compact experiment digest. It is immutable and suited to streaming or filtered retrieval.

### `failures.jsonl`

This is procedural memory: reproducible crash signatures and exact avoids. It prevents Codex from wasting GPU budget on known-invalid configurations without conflating execution failures with scientific outcomes.

### `lessons.json`

This is semantic memory: scoped, falsifiable conclusions synthesized from multiple episodes. Each lesson carries evidence IDs, counterevidence, confidence, and retrieval utility. A lesson is an advisory prior, never permission to bypass validation.

### `strategy.json`

This is working memory: incumbent, phase, plateau state, active hypothesis, exhausted families, and next queue. It may change at every consolidation checkpoint and can be reconstructed from history.

## Consolidation as self-evolution

At a trigger, compare new episodes with retrieved lessons and emit textual gradients:

```text
observation -> likely mechanism -> bounded correction -> falsifying measurement
```

Recurring gradients accumulate momentum. Translate them into a small edit proposal, then validate the proposal against held-out episodes or the next controlled experiments. Keep rejected edits in `strategy.json` to avoid oscillation.

Forward evolution expands or corrects memory from new evidence. Backward consolidation audits whether older lessons were actually retrieved and useful. This prevents unchecked memory growth and overfitting to an obsolete model basin.

## Search phase transitions

- `explore`: compare distinct hypothesis families with cheap diagnostics or short budgets.
- `exploit`: refine a supported family with one controlled variable at a time.
- `retune`: grant a bounded tuning window after an architecture or objective change; mark results provisional until the window closes.
- `plateau_escape`: after repeated valid misses, change representation, objective, or architecture family instead of continuing a local scalar sweep.

Phase changes must cite episode IDs and update the next queue with an explicit cost and falsification gate.

## Source methods

- [autoresearch program](https://github.com/karpathy/autoresearch/blob/master/program.md): fixed evaluation, baseline-first iteration, results ledger, keep/discard decisions, and crash recovery.
- [A-Evolve design](https://github.com/A-EVO-Lab/a-evolve/blob/main/DESIGN.md): filesystem workspace contract, append-only JSONL memory, versioned rollback, and selectively loaded skills.
- [SkillOpt](https://arxiv.org/abs/2605.23904): bounded skill edits, validation gates, rejected-edit memory, and slower meta-updates.
- [SkillGrad](https://arxiv.org/abs/2605.27760): textual gradients and recurring-diagnosis momentum.
- [SkillProx](https://arxiv.org/abs/2608.07449): forward evolution combined with backward, utility-aware consolidation.

These ideas inform the memory mechanism; they do not make external repositories or services runtime dependencies of either Skill.
