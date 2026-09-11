# Round 6: compact recurrence schedules, fixed existing seed

User asked to continue optimization without additional seeds. This round uses
only the existing seed 20262027 for all eight structures. No seed extension or
replication is scheduled. Compare full validation against the SAME existing seed
of Cross8 original, Cross8 matched, Cross11 all Queries, compact_seq1 and attn_full.
Do not compare new single-seed scores against old three-seed means.

All use unchanged kuairec-loop-v1 inputs, label/split/history, 1 epoch,
1240 updates, batch 8192, AdamW 1e-4 + Adagrad .05, clip 1, BF16, all Queries.
No test evaluation. Source snapshots and CPU/GPU checks precede training.

N: field block; S: history block; A: extra Query self-attention. Repeated names
share weights. Each field read is followed by the same field final norm, and
the final output always has the same history final norm. Only additional N
reads use context fields; initial Query and first field read see all fields.

| Family | Schedule | Purpose |
|---|---|---|
| existing compact_seq1 | N0 N1 S0 N1(context) S0 | parent |
| r6_shallow | N0 N1 S0 | trained no-loop shallow control |
| r6_small | N0 S0 N0(context) S0 | remove independent field prelude, no extra A |
| r6_field_before | N0 N1 N1(context) S0 S0 | place second field read before history |
| r6_field_after | N0 N1 S0 S0 N1(context) | place second field read after history |
| r6_history_only | N0 N1 S0 S0 S0 | allocate five calls to history recurrence |
| r6_loop3 | N0 (N1 S0) × 3 | train three cycles from scratch |
| r6_attention_after | N0 N1 S0 A N1(context) S0 A | move shared A after history, compare attn_seq1 |
| r6_untied | N0 N1 S0 N1(context) S1 | independent history weights at same five-call depth |

history_only uses one field final norm versus two in parent; this interface
change accompanies reallocating the field call. untied increases parameter
count and changes initialization consumption; equal seed is not identical
initialization of every shared-shaped tensor. No claim of isolating every
interaction or matching GPU time. Three cycles cost seven full calls.

Report AUC, Logloss, UAUC, NDCG, parameters and same-card B1024 forward latency.
Retain candidates near Cross8 OR Cross11; promotion is separate. Single seed
is a structural screen, not a stable improvement claim. No new seed uncertainty
study is requested; use current checkpoint diagnostics only if they address the
remaining structural question. Eight full runs are the initial bounded matrix.
