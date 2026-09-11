# Round15: plain GELU FFN width controls

Frozen before training. Eight variants, existing seed20262027 only; kuairec-loop-v1 inputs unchanged, one epoch/1240 updates, batch8192, BF16, AdamW1e-4/Adagrad.05, clip1 and all8 Queries. No test evaluation. Cross8 primary and Cross11 allQ optimized reference.

Parent r7_small_attention_first N S A N S, one shared field block and one shared history block. Preserve GELU, recurrence placement, attention and all norms. Width512 and680 at field/history/both (six runs); width1536 at history/both (two expansion controls). Original width1024. No gates. At D256, each replaced FFN parameter delta=(2D+1)*(hidden-1024)=513*(hidden-1024), multiplied by two for both stages. Original model8115473; both5127590161, both6807762529, both15368640785.

Initialize resized up/down matrices with Xavier then copy overlapping parent rows/columns, scaled by sqrt((D+old_hidden)/(D+new_hidden)); biases copied on overlap, new biases zero. Both stage candidates constructed inside restored CPU RNG context without reseeding, so same-stage initialization independent of scope and remaining model/RNG unchanged. At original width conversion exactly preserves output and gradients. At other widths initial function changes; this is a from-scratch capacity screen, not a function-preserving compression claim. Expanded new coordinates use Xavier. Optimizer removes old FFN and inserts replacement exactly once before training.

CPU/GPU five tests each: original-width compatibility including BF16, parameter arithmetic/GELU/overlap and RNG, scope-independent initialization/optimizer updates, shared recurrence/masks/empty history/final hook, invalid contracts. Training GPUs1-7, reserveGPU0.

Select by full validation AUC and guardrails, retain near-baseline/cost/ranking tradeoffs. Same-GPU benchmark; fixed-subset rec-harness diagnosis of best reduced and best expanded model by full-validation AUC. Include prior budget-matched gated reference for width680 interpretation. Do not select on diagnostic subset; no new seeds or replication.
