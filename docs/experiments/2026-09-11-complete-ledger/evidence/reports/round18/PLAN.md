# Round18: depth-specific final normalization

Four new from-scratch runs, existing seed20262027 only. Two capacities (r7 and field FFN512) × two depth schedules (phase132 and mixed1232). Compare each against its exact shared-norm schedule control as well as fixed2 parents, Cross8 primary and Cross11 allQuery reference. Retain ranking/cost tradeoffs; no test evaluation.

Only final history-stage LayerNorm affine parameters are separate for executed depths1/2/3. Original norm retained for depth2; clone identical depth1/3 parameters without consuming RNG. Recurrent N/S blocks, first-only extra self-attention and prediction tower remain shared. Adds1024 training parameters; only one norm executes at inference. Full backprop; no auxiliary losses or gradient detach.

Frozen kuairec-loop-v1, same input hashes, seed20262027, 1epoch/1240 updates, batch8192, AdamW1e-4/Adagrad.05, BF16, clip1, all8Queries. Depth counts310/620/310; total2480 cycle calls; eval2. Record raw history lengths separately from capped valid mask tokens and padded positions. No extra training seeds or replications. Reserve GPU0; train1–4, benchmark7, diagnostics5/6.

Verify CPU/GPU tests: exact initial output/RNG/core gradients at all depths and both capacities; optimizer ownership and only active norm updates under real BF16 training; schedule/checkpoint roundtrip; final diagnostic hook; exception restoration and invalid input rejection. Freeze code/config before training. Select one candidate per capacity on full validation AUC for existing rec-harness diagnostics, never diagnostic subset or test. Compare matching schedule controls to isolate norm splitting. Preserve null results and append memory.
