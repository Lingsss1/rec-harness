# Round 2 frozen comparison contract

- Primary baseline: Cross8 original; optimized references: Cross11 and matched-settings Cross8. Preserve all round-1 outcomes.
- Protocol: kuairec-loop-v1, unchanged training/validation files and labels. No model test evaluation.
- Primary metric: full-validation global AUC; report logloss, UAUC, gAUC, NDCG(10), parameter count, runtime, memory and inference latency.
- Diagnostics: frozen seed 20262027 Cross8/Cross11/middle2 checkpoints; 8 uniformly sampled validation events per user; no weight updates. Export final-query activation/gradient arrays, Q0 layer-state geometry, all equal-width query Zero/Shuffle controls, and same-weight loop-depth interventions. Subset metrics do not select a winner.
- Same-checkpoint query routing: native Q3=0, Q3=0.5, all queries, full validation for Cross11 and middle2 under all three seeds. Equal evaluation budget for both models.
- Retraining budget: choose at most two loop modifications after diagnostics, paired seeds 20262027/28/29, same one epoch / batch 8192 / 1240 updates as round 1. One matched nonloop control may be added if a training hyperparameter changes. Snapshot source/config before edits.
- Retention: AUC close to either primary or optimized baseline remains an optimization candidate per user instruction. For this round, a mean absolute shortfall <=0.0015 is a provisional screening window, not a statistical equivalence margin; retain cost and guardrail differences explicitly.
- Promotion is stricter than retention: inspect all three paired seed deltas and paired user bootstrap, require positive mean AUC delta and no mean logloss deterioration >0.001 to call a quality improvement. A stable-improvement claim additionally requires consistent seed signs and positive user-bootstrap interval; three seeds still do not establish generalization to other datasets.
- Existing toolkit probes cover current routing, gradient-health and collapse questions. No new diagnostic method or upstream toolkit modification is needed. No tool-evolution handoffs exist yet.
- Search findings remain validation evidence. Keep candidate status separate from incumbent status and preserve all failed/null outcomes.
