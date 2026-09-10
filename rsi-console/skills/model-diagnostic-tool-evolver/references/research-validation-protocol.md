# Research and validation protocol

## Contents

1. Opportunity triage
2. Literature search
3. Method card
4. Validation design
5. Incremental utility
6. Promotion and rollback
7. Source methods

## 1. Opportunity triage

Score each dimension from 0 to 2 before starting:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Decision leverage | descriptive only | may refine a choice | can eliminate or prioritize experiments |
| Recurrence | one-off | likely repeated | already repeated |
| Artifact readiness | unavailable | requires small instrumentation | already available |
| Method plausibility | no known estimator | indirect estimator | established estimator with controls |
| Cost ratio | likely costs more than it saves | uncertain | clearly cheaper than wasted experiments |

Research a tool when the total is at least 7/10 and decision leverage is 2. Record the score; do not inflate it after literature search.

Examples of qualified gaps include distinguishing dead gradients from insufficient positive samples, separating anisotropy from true collapse, testing whether a feature group contributes unique information, or measuring instability that aggregate validation metrics hide.

## 2. Literature search

Construct queries from four facets:

```text
model/domain + missing quantity + estimator family + failure mode
```

Search current primary sources. Use citation graphs and surveys only to locate original work. Prefer:

1. original peer-reviewed paper or latest author revision;
2. official paper repository;
3. official implementation and documentation;
4. independent reproduction only as counterevidence.

For each candidate method, record:

- paper URL, title, version/date, and official-code URL;
- exact quantity and equation implemented;
- assumptions and data requirements;
- computational complexity and numerical risks;
- evaluation evidence in the paper;
- known limitations and mismatches with the local model;
- code license and whether code was reused or reimplemented.

Do not cite a search result page. Keep quotations short and distinguish source facts from inferences.

## 3. Method card

Complete `method-card.md` before implementation. It must define:

- the target decision and current baseline;
- the method's causal or statistical scope;
- inputs, outputs, sign convention, and null behavior;
- competing methods rejected and why;
- paper claims versus agent inferences;
- synthetic positive and negative controls;
- real-data shadow comparison;
- precommitted acceptance thresholds;
- expected runtime, memory, and maintenance cost.

If the output cannot map to a concrete next action, keep it as exploratory analysis rather than a reusable tool.

## 4. Validation design

### Paper fidelity

Implement the smallest auditable core. Test against a paper example, official reference output, exact small-matrix calculation, or an analytically derived case. Tolerance must be declared before running.

### Synthetic correctness

Use at least one positive control with injected signal and one negative/null control. Where relevant, sweep signal strength and verify monotonic response. Prevent seed leakage by generating evaluation fixtures independently from development fixtures.

### Robustness

Test:

- empty and minimal shapes;
- mismatched shapes and schema errors;
- `NaN`, `Inf`, zeros, constants, and extreme scales;
- deterministic output under a fixed seed;
- stability across reasonable seeds or sampling budgets;
- maximum intended array size and streaming/subsampling behavior;
- runtime, peak host memory, and GPU memory if instrumentation is used.

### Shadow real data

Run without changing optimization on at least two checkpoints, seeds, datasets, or conditions relevant to the claimed scope. Use validation or train-derived artifacts, never held-out test results for iterative routing.

Choose conditions before viewing candidate output. Include a condition expected to differ and a null or near-null comparison when feasible.

## 5. Incremental utility

Compare three arms:

1. no new diagnostic;
2. existing toolkit evidence;
3. existing toolkit plus candidate.

Predeclare one utility measure appropriate to the gap, such as:

- fewer GPU trials needed to reject a hypothesis family;
- improved retrospective selection of known-valid next experiments on held-out episodes;
- reduced ambiguity or abstention under a fixed decision rubric;
- detection of injected faults missed by existing tools;
- reproducible separation of conditions the baseline treats as equivalent.

Do not require every diagnostic to predict the final ranking metric. Mechanism tools may pass by reliably falsifying a decision-relevant assumption. Reject a candidate that merely correlates with an existing metric or produces a more attractive visualization without changing a decision.

Use held-out episodes for retrospective tests; do not design the tool and score it on the same episodes. For forward GPU tests, cap budget in advance and compare against the unchanged Harness protocol.

## 6. Promotion and rollback

A promoted tool needs:

- a stable CLI and machine-readable report;
- unit tests and one realistic fixture;
- a method reference with primary citations and limitations;
- a clear route and decision boundary in the toolkit Skill;
- no hidden platform dependency;
- code identity and a rollback point.

Promotion is a small patch, not a wholesale toolkit rewrite. Observe the tool in subsequent experiments. Supersede or remove it from active routing if it is never retrieved, repeatedly unhelpful, too costly, or invalidated; retain its registry history.

## 7. Source methods

- [LLM Agents Making Agent Tools](https://arxiv.org/abs/2502.11705): paper-to-tool transformation, execution feedback, self-correction, and objective unit tests.
- [Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126): separation of expensive tool making from repeated low-cost tool use and functional caching.
- [ResearchCodeAgent](https://arxiv.org/abs/2504.20117): iterative research-method implementation with planning and memory.
- [The AI Scientist](https://arxiv.org/abs/2408.06292): iterative idea, implementation, experiment, and review loop.
- [Tool-Making and Self-Evolving LLM Agents](https://arxiv.org/abs/2607.08010): validation against labeled cases, versioned tools, execution traces, and deployment auditability.

These sources motivate the workflow. Candidate diagnostic methods must cite their own domain-specific primary papers separately.
