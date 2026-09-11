# Second-work value content: six fixed cells

Prior E all eight failed the quality/cost policy. Its work gate improved buy BCE versus entry but did not establish buy AUC or macro improvement. E changed both K and V; no inference that K was damaged is justified.

Test event/pooled × entry/work/constant. Keep the original second query `(e+z1)/2`, original Q/K and original base SDPA, w_qi interaction, external residual, two work rounds, prediction chain and shared post. Event adds an auxiliary SDPA over per-event nonlinear value residuals. Pooled adds the same conditional nonlinear form after aggregation, with no query averaging. Both add before w_qi multiplication. Fully empty histories suppress the pooled residual to avoid an extra no-history decoder.

Rank16 raw AH/AC/B matrices add 18,432 dense parameters. Fixed B Xavier gain0.01 avoids deliberately zero initialization; no guarantee PSGDKron will accept actual gradients. No repair, deferred step, gain grid or automatic retry. Constant has fixed nonzero RMS1 channels and fewer effective condition degrees of freedom; equal raw matrix shapes are not equal information/capacity.

Six complete matched-parameter initializations are required. Original mature_r2 parameter/buffer values and constructor RNG must match after filtering only the new residual members. Initial functions/gradients and subsequent training trajectories need not match. Extra gradients can alter global clip and shared projections, so local Q/K preservation is not a post-training attention-probability guarantee.

Fixed seed20262027, 2193 updates, PSGDKron every step, LR1e-4, clip1, BF16, original sparse optimizer and strict mature input protocol. No new training seeds or test split. Six meaningful cells use GPUs0..5; defer the two expensive slot-specific cells pending evidence instead of repeating exhausted hypotheses to occupy GPUs.

Primary comparisons: six versus mature_r2 and Cross8; three condition contrasts within each position; three event-minus-pooled contrasts. 21 effects × 21 metrics = 441 intervals, one500 shared-user draw stream. Report all task AUC/BCE and direct task effect differences, runtime and same-session latency. No candidate selected by interval crossing zero alone. All artifacts and failed/null outcomes retained.

This supplements read content; it does not directly establish improved second-round retrieval addressing or reproduce ITT routing. Previous query/difference reruns and generic history writers are completed and are not repeated.
