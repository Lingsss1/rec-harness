# Task-conditioned second history query

The research question is whether four prediction tasks benefit from different second-read conditions. Existing task-depth comparisons support heterogeneous gains, but do not establish that recurrence helps only particular tasks. The q4 independently trained two-round model gains more AUC on cart and buy than click; the q8 same-checkpoint whole-round deletion hurts all four tasks. These comparisons have different scopes. See [task-wise evidence](../recurrence-research/TASKWISE_RECURRENCE_EVIDENCE.md).

The previous [work-relay round](../recurrence-work-relay/RESULTS.md) is closed: eight candidates fail the original macro/cost policy, all eight parent macro intervals are negative, and forward latency increases 7.93–9.40%. Its q4 all-versus-own comparison improves fav but hurts click; both models remain worse than the original. This motivates testing task conditions, without forecasting improvement from a failed model's task tradeoff.

## Four planned configurations

| Parent | Shared condition | Task-specific condition | GPU assignments |
|---|---|---|---|
| Original ctx_q4 | taskq_q4_shared | taskq_q4_task | 0, 1 |
| Original explore_dual_no_feedback | taskq_q8_shared | taskq_q8_task | 4, 5 |

All four retain two rounds, original task towers, no prediction-to-work feedback, seed20262027, BF16, clip1, dense HeavyBall.PSGDKron with preconditioning every step and LR1e-4, sparse Adagrad LR.05, batch8192, and2193 updates. No additional training seed or test evaluation. The [protocol](protocol.json) fixes six mandatory paired comparisons and seven-model same-device latency measurements.

## Intervention and control

The original shared prefix executes once: `e=L0(q,H); z1=L1(e,H); y1=L2(e,z1); x=(e+z1)/2`.

For task t, the second work layer uses the original outer query x. Only the output of its query projection changes, from `u=Wq(x)` to `u+b[t].to(u.dtype)`, before the original head reshape and query normalization. Shared mode uses `b.mean(0)` instead. The original interaction projection, outer residual, complete history K/V and mask remain. Each task retains all S query slots; task IDs are not query-slot IDs.

Both modes register one raw zero-initialized `[4,384]` bias (1536 parameters) after the original parameter traversal and before optimizer construction. Both execute four complete second-work, second-prediction and final-post branches, including at zero initialization. For each task, `y2_t=L2(y1,z2_t)` passes through the original final post modules and its matching task tower. Prediction state never supplies a work query.

The shared control has the same registered parameter shape and branch count but fewer effective condition dimensions. Its raw gradient rows are mathematically equal; PSGDKron may update them unequally. Four-branch gradient accumulation, the appended parameter, clipping and optimizer grouping can change the training trajectory. We record those identities and do not repair or claim matching native optimizer dynamics. Initial forward equivalence is checked, not inferred from zero initialization.

## Decision and interpretation

Per parent, compare shared versus original, task versus original, and task versus shared. Report macro AUC, all four task AUC/BCE effects, all six direct task-effect differences, measured training/VRAM cost and forward latency. An internal task-versus-shared gain alone is insufficient for a candidate: the frozen policy still compares against the original recurrence. Zero-crossing intervals do not establish equivalence. Saved-checkpoint user-paired intervals do not estimate training-seed uncertainty.

Task-specific bias tests task-conditioned retrieval in a controlled later-branch schedule. It does not by itself prove improved use of the first-work update, causal information quantity or task-exclusive recurrence benefit. Labels and outcome masks do not condition the query; only the fixed prediction-task index selects the bias.

## Literature motivation and bounded scope

[DMN+ §3.3](https://proceedings.mlr.press/v48/xiong16.pdf) conditions attention on the question and prior memory; [MAC §2.2](https://arxiv.org/pdf/1803.03067) separates control from memory. Both concern language/question reasoning and provide motivation for conditioning reads, not evidence for four fixed Taobao task IDs. The [FiLM paper](https://arxiv.org/pdf/1709.07871) compares affine and restricted feature modulation in a different setting. We defer an additional gain/affine sweep: its relevance before query normalization is unproven, and the first unresolved comparison is task-specific versus shared bias. Previous shared second-query norm/projection untying and task-specific depth readout already failed the original macro/cost policy; this intervention changes each task's actual second retrieval.

Execution order is prior-round closeout, new offline/source review, one bounded native CUDA stage check, four real-batch two-step smokes with independent review, then four full trainings. At creation of this note, only the seven CPU model tests have executed and passed; GPU checks and training remain pending. Their fixture optimizer is AdamW, so they do not substitute for the real PSGDKron smoke gate. CPU autograd issued a CUDA device-count warning while CUDA remained uninitialized.
