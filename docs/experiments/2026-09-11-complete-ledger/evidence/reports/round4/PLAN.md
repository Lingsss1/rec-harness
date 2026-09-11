# Round 4: feedback source and normalization interface

Continue structural optimization, with Cross8 as primary baseline and Cross11 as optimized reference. Retain near-baseline models as optimization candidates; retention is not incumbent promotion. Preserve all preceding evidence and keep the test set untouched.

Fixed protocol: kuairec-loop-v1; identical data hashes and labels; train/validation all Query; 1 epoch, batch=8192, 1240 updates; AdamW/Adagrad and learning rates unchanged. Four variants × seeds 20262027/28/29 = 12 new full runs. Every variant receives three seeds, with no single-seed selection.

| Variant | Additional field read | Field final norms | Reference |
|---|---|---:|---|
| feedback_target | only 7 target-item fields after history | 2 | previous all-field feedback |
| feedback_context | only 30 user/context fields after history | 2 | previous all-field feedback |
| feedback_no_norm | all 37 fields after history | 1 | previous all-field feedback with 2 norms |
| field_norm_matched | all 37 fields before history, norm after each N1 | 2 | previous field loop with 1 norm and feedback with 2 norms |

All variants retain the same 6 QFormer block calls and 11.408M parameters. Source subsets alter actual attention cost; record same-GPU latency. The initial field encoding still reads all 37 fields. Source restriction affects only the additional read, so no initial user/target features are removed.

Primary metric: complete validation global AUC. Guardrails: logloss, UAUC, gAUC, NDCG@10, GPU memory, runtime and same-GPU forward latency. Compare paired seeds against Cross8 original/matched, Cross11 all Query, previous feedback, field loop and middle2_damped50. Report all results, including losses and high variance. Bootstrap intervals condition on fixed checkpoints and do not prove statistical equivalence or account for model search.

Correctness gates: old field/feedback defaults exactly reproduce outputs and gradients; one pass recovers Cross11; excluded feedback fields have zero direct gradient into the additional read; selected fields remain differentiable; actual source lengths, normalization counts and optimizer registration match the configuration; no source mutation or nonfinite outputs.

Existing rec-harness geometry, hook and paired-intervention tools cover these tensor contracts. There are no tool-evolution handoffs to consume; no new diagnostic method is required. Use fixed-subset source/norm interventions before interpreting retrained outcomes, and never use subset AUC to choose the winner.
