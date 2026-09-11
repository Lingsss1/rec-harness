# Interpreting the matched recurrence study

Use A=mature_r2, DD=hist_deep_deep (H2/H2), SD=hist_shallow_deep (H1/H2), DS=hist_deep_shallow (H2/H1), X=cross8_mature. No results are asserted by this guide.

| Prespecified effect | Question supported by this comparison | Limit |
|---|---|---|
| D_released, D_mature | Separately trained R2 minus R1 under each input protocol | Not inference-only extra depth or a comparison to old17-class results |
| I=D_mature-D_released | Whether conservative action maturity changes the recurrence benefit | Not actual online leakage rate or a causal attribution of old AUC |
| DD-A, SD-A, DS-A | Overall value of history encoding and its read schedule | Added capacity/compute and training dynamics also change |
| SD-DD | Change only the first work read from H2 to H1 in the training recipe | Not a fixed-checkpoint inference intervention |
| DS-DD | Change only the second work read from H2 to H1 in the training recipe | Not a fixed-checkpoint inference intervention |
| SD-DS | Order of the same two history versions | Both positions change; no H1/H1 cell, so not a full version interaction |
| A-X, DD-X, SD-X, DS-X | Quality and cost against matched mature-input Cross8 | q4/q8 architectures and capacity differ; not isolated recurrence |

For all13 effects, retain macro AUC, four task AUC and BCE effects, and all six direct task differences for AUC/BCE:21 metrics each,273 intervals. Use the same500 user draws for all models/effects. Compute I and task differences within each draw; never subtract confidence-interval endpoints. AUC positive and BCE negative are favorable. A significant task and an inconclusive task do not establish task heterogeneity. A confidence interval crossing zero does not establish equivalence. These exploratory percentile intervals are conditional on one training seed's checkpoints; no seed or multiplicity correction is implied.

B is primarily an explanatory maturity-by-depth study and does not define new numerical keep/near thresholds. C's queue policy defines a quality candidate by positive macro point difference versus A, and a near candidate floor of -0.0001 together with forward-latency ratio at most1.05. Report quality point/interval, near condition and measured costs separately; the renderer must not invent an extra boolean policy. Root decides actual retention and states cost tradeoffs. No automatic old-protocol incumbent promotion.

If only DD-A improves, the result supports additional history encoding as a candidate, without establishing a benefit of read order. SD beating DD/DS and A supports this particular shallow-to-deep schedule as a candidate; inference-time history writeback conditioned on z1 remains untested because both H1 and H2 are precomputed. Positive D_mature can coexist with negative I. Single-task improvement cannot replace the macro objective.

Report eight unique models on one physical GPU, BF16,1024 common raw rows, two separately bound transformed inputs,10warmup+40timed per model. Keep model CUDA Event latency separate from data/metadata construction, action conversion, transfer, model construction, checkpoint load and training cost. Same parameter count and module call counts do not establish equal actual latency or optimizer trajectories.

This guide was prepared from the frozen B/C protocols and the owning agent's read-only interpretation review. It does not alter protocols, execute new comparisons or preselect a winner.
