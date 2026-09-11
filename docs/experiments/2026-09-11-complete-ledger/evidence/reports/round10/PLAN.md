# Round 10: selective shared sublayers in repeated block calls

Eight complete from-scratch runs, existing seed 20262027 only. Parent is
r7_small_attention_first: N(all) S A N(context) S, final sequence norm.
Primary reference Cross8 original; optimized Cross11 all Queries; cost reference
r8_first_loop1. Same kuairec-loop-v1 inputs, one epoch/1240 updates/batch8192,
AdamW1e-4/Adagrad.05, clip1, BF16, all Queries; no test evaluation.

Every first N/S call still executes self-attention + CrossValue + FFN, each with
its own residual and LayerNorm. Only second and later calls use the selection.
The extra A after the first S remains unchanged. Field final norm is still applied
after every N read; final sequence norm is applied at model output.

| Family | Repeated N | Repeated S |
|---|---|---|
| r10_field_cross | CrossValue only | full |
| r10_field_cross_ffn | CrossValue + FFN | full |
| r10_field_self_cross | self-attention + CrossValue | full |
| r10_history_cross | full | CrossValue only |
| r10_history_cross_ffn | full | CrossValue + FFN |
| r10_history_self_cross | full | self-attention + CrossValue |
| r10_both_cross_ffn | CrossValue + FFN | CrossValue + FFN |
| r10_both_cross | CrossValue only | CrossValue only |

Skipping a sublayer also skips that sublayer's residual LayerNorm, so this tests
selective block structure, not attention/FFN computation at fixed norm count.
All selected modules share the original weights and use immutable original source
tokens. Wrappers reuse exact original parameter objects, optimizer groups and
state-dict keys; no extra parameters or random initialization. Every parameter
remains trained via the full first pass. All models have 8,115,473 parameters.

Tests cover full-parent initialization/output/gradient equality, real sublayer and
norm call order, original source identity/masks, optimizer coverage, full first-pass
invariance, finite empty-history behavior, and actual final Query interventions.
Freeze model/config/plan/tests/exporter before training. Compare complete validation
AUC plus Logloss/UAUC/gAUC/NDCG, GPU latency and training cost. Keep near-baseline
and cost/ranking candidates; preserve negative outcomes. Diagnostic subset effects
do not select a winner. All project artifacts stay within external/home/lhr.
