# Transfer the current KuaiRec winner to the released Taobao task

User request: 尝试把目前最好的迭代模型应用到淘宝任务.

Use the current full-validation AUC winner r7_small_attention_first (KuaiRec AUC0.8712340584580865), not the latest experimental candidate. Reuse its frozen QFormerCross11Memory implementation: one field block and one history block, both shared across two cycles; additional self-attention after the first history call only; first field read all fields, repeated field read context only; all8 output Query slots feed each task tower. Train from scratch on Taobao embeddings with four independent task heads. No KuaiRec checkpoint weights are loaded.

Data: UniRank-linked salmon1802/Taobao, revision fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5. Download train/valid and metadata only, validate LFS SHA256. Keep released encoded rows, labels and histories unchanged. Train17822296 rows (May6–11), valid2871767 rows (May12). Labels is_click/cart/fav/buy. See reports/protocol.json for behavior-window and timestamp availability limitations. This is a within-release architecture comparison, not evidence of causal online performance.

Four configurations: original Cross8 clip10; Cross8 clip1 control; Cross11 allQuery clip1; r7 loop2 allQuery clip1. Cross8 is primary baseline. Same seed20262027, epoch1, batch8192, AdamW1e-4/Adagrad.05, BF16, max_len100, shuffle, drop_lastFalse. Exact optimizer budget follows unchanged blocked loader; assert every batch completes. No extra seeds or quality replications. Report per-task AUC/logloss and equal-weight macro AUC (the upstream monitor); do not compare Taobao AUC numerically to KuaiRec AUC.

Correctness: CPU/GPU tests for four heads and real updates, all parameter ownership, exact N S A N S recurrence, padding and target-action exclusion, checkpoint roundtrip. Audit data ranges/indices/labels, then real-data smoke before full matrix. Four jobs on GPU1–4, reserveGPU0. Benchmark all four on same GPU7 with existing benchmark script. Save configs, source/input hashes, predictions and checkpoints under taobao-transfer/runs; leave frozen KuaiRec runner/config/runs unchanged. Preserve null results and failures. No test evaluation.
