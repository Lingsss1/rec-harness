# Round 11: zero-initialized repeated-pass conditioning

Eight complete runs, existing seed 20262027 only; no replicates or new seeds.
Parent r7_small_attention_first retains full N S A N(context) S shared blocks.
First calls are unchanged. Only repeated N/S calls use the new condition.

| Family | Type | Scope | Added parameters |
|---|---|---|---:|
| r11_field_embedding | broadcast vector | field | 256 |
| r11_history_embedding | broadcast vector | history | 256 |
| r11_both_embedding | separate broadcast vectors | both | 512 |
| r11_query_embedding | separate 8x256 Query vectors | both | 4096 |
| r11_field_gate | input-Query linear gate | field | 257 |
| r11_history_gate | input-Query linear gate | history | 257 |
| r11_both_gate | separate input-Query gates | both | 514 |
| r11_embedding_gate | vectors plus gates | both | 1026 |

Embeddings are added before the repeated full block. Gate uses original incoming
Query q, with delta=tanh(Wq+b), output=y+delta*(y-q); the full-update scale is
1+delta in (0,2), initially exactly one. Field gating follows field final norm;
history gating precedes the final sequence norm. All added parameters start zero,
draw no random numbers and use parent dense optimizer settings. Parent initialization
and initial output are preserved, including BF16. Diagnostic third calls reuse the
same repeated-pass parameters; this encodes first versus repeated roles, not a
separate learned vector for every possible depth. No source tokens are modified.

Same kuairec-loop-v1 fingerprints, one epoch/1240 updates, batch8192,
AdamW1e-4/Adagrad.05, clip1, BF16, all Queries; full validation AUC is primary.
Cross8 original remains primary baseline and Cross11 all Queries optimized reference.
Retain Logloss/UAUC/gAUC/NDCG and cost tradeoffs; test split is not evaluated.
CPU/GPU tests check parent initial functions/gradients, real new-parameter updates,
repeat-only effects, gate math, masks/empty history and true output hooks.
Freeze model/config/plan/tests/exporter before training, preserve all outcomes.
Project artifacts remain within external/home/lhr.
