# Single history encoding and first-work conditioning

The completed mature-input study supports a read-order question, but its SD-versus-A macro interval crosses zero and SD costs46.65% more latency. This round keeps two work cycles and asks whether one encoding, cheaper token-wise encoding, or a work-conditioned increment offers a useful alternative. No claim of exact ITT reproduction or expected improvement.

All inputs, training seed20262027, PSGDKron every-step/LR1e-4, sparse Adagrad.05, clip1, BF16, batch8192,2193updates and original four heads remain fixed. Prior mature_r2, cross8_mature and hist_shallow_deep checkpoints/predictions are reused as references, with no retraining or extra training seeds. Original17-label released incumbent remains a separate protocol.

| GPU | Name | Encoder | First/second history | Increment condition |
|---:|---|---|---|---|
|0|sa_01|C attention G once|H0/H1|none|
|1|sa_11|C attention G once|H1/H1|none|
|2|sa_10|C attention G once|H1/H0|none|
|3|mlp_01|token MLP G once|H0/H1|none|
|4|mlp_11|token MLP G once|H1/H1|none|
|5|mlp_10|token MLP G once|H1/H0|none|
|6|mlp_gate_entry|token MLP G once|H0/H0+gate*delta|entry e|
|7|mlp_gate_work|token MLP G once|H0/H0+gate*delta|first work z1|

Entry always readsH0. G is computed once before entry; only the gated pair applies its modulation after z1 exists. Both gate variants execute identical modules/calls; their sole structural difference is the source of the condition. No prediction y feeds work or gate. The original second query is(e+z1)/2.

MLP delta is explicit: up(FFN(LN(down(mask(H0))))), with bottleneck128,FFN256,GELU. Do not subtract close BF16 H1 and H0 to recover it. Gate rank32 uses separate LN on history/condition and two bias-free projections:2*sigmoid(dot(Wh(LN(H0)),Wc(mean(LN(condition),slots)))/sqrt32). Mask history before new sublayers and after updates. Original model input builder already masks padding; new-module NaN handling does not claim that arbitrary invalid NaNs are safe in unchanged native attention.

Initial parameter/buffer/RNG/static optimizer layout must match within attention3,MLP3,gated2 groups, allowing only physical CUDA index normalization. Architecture/parameter additions may change optimizer dynamics versus parents and between groups. Actual counts and gradients are checked during native smokes. Predicted totals are51,955,556/51,889,252/51,915,364; these are predictions until measured.

The fixed29 contrasts are in protocol.json: eight versusA; eight versusCross8; three reading-order pairs per encoder; three matched-position MLP-minus-attention pairs; work-gate minus entry-gate; each gate versus ungatedMLP01; SA01 versus completed SD. One common500 user-bootstrap stream computes all11 models and609 intervals. Full taskAUC/BCE, direct task differences, macro and costs stay visible. No posthoc extra pairs/seeds/draws.

Positive macro point difference versusA defines a quality candidate. Near-baseline efficiency requires point delta>=-0.0001 and forward-latency ratio<=1.05. Report these conditions separately from intervals and actual retention; do not treat crossingzero as equivalence. work-gate beating entry-gate while losing toA would support only a within-family condition difference, not an improved model. Gate-vs-ungated also changes modulation/capacity; MLP-vs-attention changes both interaction and capacity. SA01-vs-oldSD changes first and second read depths plus computation, not only second depth.

Existing rec-harness measurements cover the selection question: native route/gradient/update smoke checks plus full paired retraining and latency. Handoffs are currently empty; no diagnostic task ID is fabricated. Old fixed-checkpoint sensitivity and query-mixing negatives remain relevant cautions but are not equivalent mature-input history-update trials.

Root owns protocol, runtime launch and memory. Delegates may prepare isolated sources/tests and return read-only reviews. Production sources freeze only after necessary new-code checks; successful old loader,attention-kernel andprocess-cleanup fixtures are reused without rerunning them. No source dataset, old model, old episode or old failure is rewritten.
