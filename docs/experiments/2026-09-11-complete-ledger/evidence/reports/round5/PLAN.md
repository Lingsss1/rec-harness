# Round 5: fewer blocks and additional self-attention inside recurrence

Frozen before training, 2026-09-06. User requested fewer layers, self-attention
followed by recurrence, and reducing blocks after adding attention.

Cross11 already has self-attention in every original QFormer block. The new
`A` is a separate shared Query-only self-attention + residual LayerNorm, with
no new FFN. It runs before each recurrent field read. All K/V sources remain
immutable, and only the second field read restricts K/V to user/context.
Initial queries and the first field read always see every field.

`N`: original field block; `S`: original history block. Identical names within
a schedule share weights. Every N read ends with the existing field final norm;
the complete history output ends with the existing history final norm.

| Family | Field / history independent blocks | Actual schedule | Full QFormer / extra attention calls |
|---|---|---|---|
| prior feedback_context | 2 / 3 | N0 N1 S0 N1(context) S1 S2 | 6 / 0 |
| compact_seq2 | 2 / 2 | N0 N1 S0 N1(context) S0 S1 | 6 / 0 |
| compact_seq1 | 2 / 1 | N0 N1 S0 N1(context) S0 | 5 / 0 |
| attn_full | 2 / 3 | N0 A N1 S0 A N1(context) S1 S2 | 6 / 2 |
| attn_seq2 | 2 / 2 | N0 A N1 S0 A N1(context) S0 S1 | 6 / 2 |
| attn_seq1 | 2 / 1 | N0 A N1 S0 A N1(context) S0 | 5 / 2 |
| attn_small | 1 / 1 | A N0 S0 A N0(context) S0 | 4 / 2 |

Six new families × all three fixed seeds 20262027/28/29 = 18 complete runs.
No selection by the first seed. The comparisons isolate extra attention at each
history depth and compare independent depth with recurrent reuse. `attn_small`
also removes the field prelude; it is a joint compression candidate, not a
single-factor control. `attn_full` isolates attention against the existing parent.
Every module is installed by a stage factory before the optimizers are built.

Keep kuairec-loop-v1 data, identical hashes, 1 epoch / 1240 updates,
batch 8192, AdamW 1e-4 + Adagrad .05, clip 1, BF16, fixed loader order,
all Queries in train/eval. No test evaluation. Primary baseline Cross8 original,
optimized references Cross8 matched and Cross11 all Queries, plus parent context
feedback. Retain candidates close to either baseline, separate from promotion.

Measure parameter counts, update/fit time, VRAM and GPU 7 BF16 B1024 forward
latency; equal numbers of full and attention-only calls are not equal compute.
Rank on complete-validation mean AUC, retain logloss / UAUC / NDCG tradeoffs.
Use paired user bootstrap for promising compact candidates; it conditions on
these three seeds and does not correct architecture selection or seed variance.

Verify full-depth/no-extra-attention output and gradient compatibility, real
call order / sharing / source widths, optimizer inclusion, masked/empty history,
immutable sources, and reduced parameter counts on CPU and GPU before training.
