# Round 7: memory and normalization interfaces, same existing seed

Eight complete runs, existing seed 20262027 only. No new seed or replication.
Same kuairec-loop-v1 inputs and training budget: 1 epoch / 1240 updates /
batch 8192 / AdamW 1e-4 + Adagrad .05 / clip 1 / BF16 / all Queries.
Primary reference Cross8 original; optimized Cross11 all Queries; direct
parents r6_field_after and r6_small_attention_after, all at this same seed.

Parent terminal schedule: N0 N1 S0 S0 N1(context), with field final norm
after each N1 read and sequence final norm at output.
Parent compact schedule: N0 S0 A N0(context) S0 A, with field norm per read
and sequence norm at output. A already includes its own residual LayerNorm.

| Family | Change from corresponding parent |
|---|---|
| r7_terminal_prenorm | move sequence final norm before terminal field read |
| r7_terminal_query_memory | append initial Q to terminal context K/V |
| r7_terminal_field_memory | append first full-field encoded Q to terminal context K/V |
| r7_small_query_memory | append initial Q to second compact field read |
| r7_small_field_memory | append first full-field encoded Q to second compact field read |
| r7_small_midnorm | sequence norm after each history read, before A; no final duplicate |
| r7_small_attention_last | extra A only after the last history read |
| r7_small_attention_first | extra A only after the first history read |

Memory tensors are fixed within a forward, keep gradients, and contain no future
data. Initial Q and first field state see the same original full field tokens;
observed historical K/V never change. Default feedback is 30 context fields;
adding 8 Query memory tokens makes it 38. This is not an equal-cost comparison.
All configurations retain parent parameter keys/shapes and counts. Norm relocation
and mixer schedule use existing modules; no unused newly allocated modules.

Default no-memory/end-norm/both-mixers matches each parent exactly. CPU/GPU tests
cover that compatibility, real order, cached tensor identity and direct memory
gradients, optimizer coverage, mask/empty-history invariance and invalid options.
Evaluate full validation, parameter counts and same-card forward latency. Retain
quality, Logloss, user-ranking and cost tradeoffs. Test split is not evaluated.
