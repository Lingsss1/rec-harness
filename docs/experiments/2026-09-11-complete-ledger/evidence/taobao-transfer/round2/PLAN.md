# Taobao round2: increase capacity plus shared recurrence

User steering: 尝试增加参数量+循环. Primary metric is four-task equal-weight macroAUC; report each task AUC/logloss. Cross8 remains primary baseline; taobao_cross8_clip1 is the current strongest reference (macroAUC0.823049126701138). The KuaiRec winner transferred to Taobao at0.8179245073004873, so it is not assumed to be the Taobao winner.

Five Cross8 experiments form capacity/recurrence controls: width256 loop2; width384 loop1/2; width512 loop1/2. Existing width256 loop1 clip1 reference is reused, never retrained. Width means token_dim; embedding_dim16, heads4, queries8, field layers2/history layers3, FFN ratio4 and four task towers[256,128] remain fixed. At loop2, the complete shared history stage (three CrossValue layers, post-self-attention, FFN and norms) runs twice against identical original history K/V, carrying Query states forward. The field stage executes once. Loop1 is exactly the original Cross8 function, parameter keys and initialization. No extra parameters for recurrence, full backprop through both passes.

Sixth experiment: widen r7's original two-cycle N S A N S architecture to384. It compares with existing r7 width256 to test increased capacity on that loop design. This is a separate architecture comparison, not the same per-pass computation as Cross8 stage recurrence.

Keep taobao-unirank-release-v1 data, four labels, seed20262027, epoch1,2193updates,batch8192,clip1,AdamW1e-4/Adagrad.05,BF16,max_len100 unchanged. Six new from-scratch full runs on GPU1–6. ReserveGPU0; benchmarkGPU7. No extra seeds, repeats or test evaluation. Model width increases active parameters; report total parameters and the original implementation's unused task residual-gate parameters separately to avoid crediting inactive parameter growth.

Verify CPU/GPU5 tests: one-pass exact parent output/gradient/RNG; twice-shared call graph and manual unroll; capacity vs recurrence parameter counts; four-task real BF16 update, masks and checkpoint; invalid contracts. Run real-data smoke at the most demanding width512 loop2 before full matrix. Preserve the initial test-fixture failure (omitted scalar task specification) separately. Freeze model/config/runner/tests before training. Expected published data/label availability limitations remain unchanged; see ../reports/protocol.json.

Broader non-QFormer alternatives remain deferred after user steering. Their CPU fixture/initialization checks were completed; none were trained.
