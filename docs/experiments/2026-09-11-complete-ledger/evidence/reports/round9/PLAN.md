# Round 9: feedback placement, shared history and pass-local interfaces

Eight full runs use existing seed 20262027 only. No additional seeds or replicates.
Parent r7_small_attention_first: N(all) S A N(context) S -> final sequence norm.
N/S share parameters across calls; extra A occurs only after the first S.
Cross8 original remains primary baseline; Cross11 all Queries is optimized reference.
Same kuairec-loop-v1 fingerprints, full validation, 1 epoch/1240 updates, batch8192,
AdamW1e-4 + Adagrad.05, clip1, BF16, all Queries, no test evaluation.

| Family | Change | Main comparison |
|---|---|---|
| r9_feedback_omit | N S A S; remove second field read | r7 first-attention parent |
| r9_history_replace | N S A S S; replace second field block with shared history | parent; same block-call count, different source |
| r9_feedback_before | N N(context) S A S | parent; same block-call count |
| r9_feedback_after | N S A S N(context) | parent; same block-call count |
| r9_field_splitnorm | independent norm for second field read | parent; +512 parameters |
| r9_history_splitnorm | sequence norm after each S; second one independent | r8_first_midnorm; +512 parameters |
| r9_both_splitnorm | both field/history interface norms independent by pass | r8_first_midnorm; +1024 parameters |
| r9_untied_clone | second S initialized as exact independent clone of first S | parent; added history-block parameters |

History/both splitnorm also compare with r7 parent, but that contrast changes norm
placement and sharing together. Compare to r8_first_midnorm to isolate norm sharing.
Clones draw no new random numbers and preserve the corresponding initial function.
All shared transforms keep parent initialization. Register cloned parameters in
the same optimizer type and copied group settings; tests verify actual updates.
Extra diagnostic passes reuse the final available cloned module, without creating
parameters. Existing actual-output Identity remains the intervention target.

Complete validation AUC is primary; keep Logloss, UAUC/gAUC, NDCG and compute tradeoffs.
The untied model is not parameter matched; omit is not compute matched. Same block
call counts do not imply equal FLOPs because field and history sources differ.
Use CPU/GPU compatibility, real graph/source/mask and optimizer coverage tests, then
freeze model/config/test/plan/exporter before training. Record all failed and null
outcomes. Keep all work/data/artifacts within external/home/lhr.
