# Round 8: first-pass attention refinement and trained recurrence controls

All eight experiments use existing training seed 20262027 only, no replications.
Parent is r7_small_attention_first: N(all) S A N(context) S, shared N/S weights,
one independent field block and one independent history block, final sequence norm.
Same kuairec-loop-v1 fingerprints, 1 epoch, 1240 updates, batch8192, BF16,
AdamW1e-4/Adagrad.05, clip1, all eight Queries. No test evaluation or selection.
Cross8 original is primary reference; Cross11 all Queries is optimized reference.

| Family | Change from parent |
|---|---|
| r8_first_midnorm | sequence norm after every S, before A if present; omit terminal duplicate |
| r8_first_half | A attention residual scale 0.5 before its post norm |
| r8_first_double | A attention residual scale 2 before its post norm |
| r8_first_prenorm | move A norm to its attention input; raw residual bypass |
| r8_first_prenorm_half | combine A pre norm and residual scale 0.5 |
| r8_first_midnorm_half | combine per-history sequence norm and A residual scale 0.5 |
| r8_first_loop1 | train one N S A cycle from scratch; no second context/history read |
| r8_first_loop3 | train N S A N S N S from scratch; third context/history read shares weights |

For A: post-norm computes LN(q + alpha*Attn(q)); pre-norm computes
q + alpha*Attn(LN(q)). Dropout remains in the residual branch as in the parent.
All configurations keep identical parameter keys, shapes, counts and seed initialization;
the wrapper reuses existing parameter objects already registered in optimizers.
One/three cycles change compute and information-read count, not parameter budget.
These controls test this trained recurrence schedule, not an equal-compute proof that
weight sharing itself is responsible for every gain. Previous depth-at-inference probes
are not substitutes for these complete retraining experiments.

Select by full-validation AUC; inspect Logloss, UAUC, gAUC, NDCG@10, parameters,
same-card batch1024 inference and training cost. Retain near-Cross8/Cross11 and
ranking/cost tradeoffs even when they do not beat the parent. No diagnostic-subset
model selection. Existing hooks follow diagnostic_query_output, so norm moves do not
change the meaning of final Query interventions. Keep model/exporter frozen after tests.
