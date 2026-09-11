# Target decision written before literature search

Observed ambiguity: repeated reads of the same history and the failure of source/route corrections do not establish whether the second read redistributes useful attention. Existing whole-query and gate interventions also alter the explicit query-feature interaction and residual state.

Missing information: the fixed-checkpoint effect of replacing only second-cycle L2 attention probabilities P2 with same-example, same-head, same-query-slot first-cycle P1, while retaining second-cycle w_qi(q2), the original history values, downstream projections and residual q2.

Decision it could change: preserve adaptive second-read redistribution, abstain from attention-specific edits if unsupported, or later consider a separately authorized reuse hypothesis only after reliable negligible-effect evidence. High attention overlap alone does not select a model change, and a harmful replacement does not demonstrate that anti-repetition penalties would help.

Cheapest falsifying test: an offline analytical fixture distinguishes identical P1/P2, distinct probabilities with equal attended values, and distinct probabilities selecting causally relevant values. Any failed identity or null control rejects the measurement before running either checkpoint. A future fixed-checkpoint shadow test must preserve w_qi, residual and memory and compare original versus P2->P1 under matched numerical execution.

Opportunity score, fixed before paper search: decision leverage 2, recurrence 2 (several rounds remain unable to separate the probability path), artifact readiness 1 (checkpoints/subset exist, thin instrumentation is missing), method plausibility 1 (controlled intervention is plausible but this model-specific estimator is unvalidated), cost ratio 2 (two subset probes should be cheaper than another eight-model training round). Total 8/10.

Scope now: research and proposal only. All paper-fidelity, synthetic, robustness, shadow, utility and cost measurements remain pending. No diagnostic implementation, real-data execution, training, test read, model edit, stable toolkit change, global lesson or handoff is permitted in this task.
