import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.
    
    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}
    
    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}
    
    Your task: produce a concise diagnosis summary that captures ONLY the core findings and implications.
    Do NOT propose web searches or action plans. Do NOT list "what to look up".

    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": [
        "<1-2 sentences: the most important interpretation of the metrics>",
        "<1-2 sentences: the second most important interpretation (only if truly necessary)>"
    ],
    "key_implications": [
        "<1 sentence: what this implies about model behavior>",
        "<1 sentence: what this implies about training dynamics or representation>",
        "<optional 1 sentence: risk/trade-off if relevant>"
    ],
    "evidence": {{
        "headline_metrics": {{
        "<metric_name>": <value>,
        "<metric_name>": <value>
        }},
        "brief_metric_read": {{
        "<metric_name>": "<very short interpretation tied to its definition>",
        "<metric_name>": "<very short interpretation tied to its definition>"
        }}
    }}
    }}

    Rules:
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Separate the diagnostic view from the trained module
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = device
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in the shared task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute contextual alignment without a quadratic Gram matrix
    @torch.no_grad()
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if not isinstance(sample_size, int) or sample_size < 2:
            raise ValueError("sample_size must be an integer >= 2")
        emb = self.view.item_emb.weight.detach()[1:]
        n = emb.size(0)
        if n < 2:
            return None

        g = torch.Generator(device=emb.device)
        g.manual_seed(seed)
        k = min(sample_size, n)
        idx = torch.randperm(n, generator=g, device=emb.device)[:k]
        # Convert only sampled rows; retain the existing cosine definition for
        # zero vectors. Nonfinite embeddings indicate an explicit model error.
        x = emb.index_select(0, idx).float()
        if not torch.isfinite(x).all().item():
            raise FloatingPointError("Nonfinite sampled contextual item embeddings")
        x = F.normalize(x, p=2, dim=-1).double()
        # Sum of off-diagonal dot products equals ||sum_i x_i||² - sum_i ||x_i||².
        offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
        mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
        return round(float((mean_cos + 1.0) * 0.5), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Validate combined-score features and restore every module's diagnostic mode
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        # Keep the public method signature; category metadata is unavailable.
        previous_modes = [(module, module.training) for module in self.model.modules()]
        margins_all = []
        valid_count = nonfinite_count = batches = 0
        base_sq = residual_sq = 0.0
        decomposed_scores = 0
        try:
            self.model.eval()
            iterator = iter(dataloader)
            for _ in range(num_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                _, sequence, positive, negative = batch
                seq = torch.as_tensor(sequence, device=self.device, dtype=torch.long)
                pos = torch.as_tensor(positive, device=self.device, dtype=torch.long)
                neg = torch.as_tensor(negative, device=self.device, dtype=torch.long)
                if seq.ndim != 2 or pos.shape != seq.shape or neg.shape != seq.shape:
                    raise ValueError(
                        "Diagnostic sequence, positive and negative must be B x L"
                    )
                batches += 1
                features = self.view.log2feats(seq)
                dimension = self.view.item_emb.weight.shape[-1]
                if (
                    features.ndim != 3
                    or features.shape[:2] != seq.shape
                    or features.shape[-1] < dimension
                ):
                    raise ValueError(
                        "log2feats must return B x L x F with F >= contextual embedding dimension"
                    )
                items = torch.stack((pos, neg), dim=-1)
                logits = self.view.predict_score(features, items).float()
                if logits.shape != items.shape:
                    raise ValueError(
                        "predict_score must return B x L x 2 paired scores"
                    )
                margin = logits[..., 0] - logits[..., 1]
                valid = seq.ne(0) & pos.ne(0) & neg.ne(0)
                finite = torch.isfinite(logits).all(-1) & torch.isfinite(margin)
                valid_count += int(valid.sum().item())
                nonfinite_count += int((valid & ~finite).sum().item())
                measured = valid & finite
                if measured.any():
                    margins_all.append(margin[measured].cpu())

                # Under the concatenated-feature contract, the first D coordinates
                # score against the contextual table; remaining coordinates encode
                # the directed transition branch. Scoring still uses the shared API.
                dimension = self.view.item_emb.weight.shape[-1]
                if features.shape[-1] > dimension and measured.any():
                    contextual = features[..., :dimension].float()
                    embeddings = self.view.item_emb(items).float()
                    base = (contextual.unsqueeze(-2) * embeddings).sum(-1)
                    residual = logits - base
                    decomposable = measured & torch.isfinite(base).all(-1)
                    decomposable &= torch.isfinite(residual).all(-1)
                    if decomposable.any():
                        b = base[decomposable].double()
                        r = residual[decomposable].double()
                        base_sq += float(b.square().sum().item())
                        residual_sq += float(r.square().sum().item())
                        decomposed_scores += b.numel()
        finally:
            # Preserve mixed modes, including intentionally frozen eval modules.
            # Calling train(previous_mode) recursively would overwrite those modes.
            for module, was_training in previous_modes:
                module.training = was_training

        metrics = dict.fromkeys(
            ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate"), None
        )
        measured_count = 0
        low_total = wrong_total = 0
        if margins_all:
            margins = torch.cat(margins_all)
            measured_count = margins.numel()
            low_total = int((margins < eps).sum().item())
            wrong_total = int((margins < 0).sum().item())
            k = max(1, int(np.ceil(0.1 * measured_count)))
            metrics.update(
                {
                    "mean": float(margins.mean().item()),
                    "p10": float(margins.kthvalue(k).values.item()),
                    "neg_beats_pos_rate": wrong_total / measured_count,
                    "small_margin_rate": low_total / measured_count,
                }
            )
        metrics.update(
            {
                "valid_positions": valid_count,
                "measured_positions": measured_count,
                "nonfinite_positions": nonfinite_count,
                "batches": batches,
                "decomposed_score_count": decomposed_scores,
                "contextual_score_rms": (
                    (base_sq / decomposed_scores) ** 0.5 if decomposed_scores else None
                ),
                "transition_score_rms": (
                    (residual_sq / decomposed_scores) ** 0.5
                    if decomposed_scores
                    else None
                ),
            }
        )
        breakdown = {
            "eps": float(eps),
            "low_margin_total_positions": low_total,
            "wrong_rank_total_positions": wrong_total,
            "low_margin_top_categories": None,
            "wrong_rank_top_categories": None,
            "unavailable_reason": "Item categories are not supplied by the shared task.",
        }
        return metrics, breakdown

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Report measured quantities and explicitly unobserved attributes
    def run_full_diagnosis(self, train_loader):
        print(">>> Running training-score diagnosis...")
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        margins, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=0.5, topk_categories=20
        )
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margins["mean"],
                "pos_neg_margin_p10": margins["p10"],
                "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margins["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "valid_position_count": margins["valid_positions"],
                "measured_position_count": margins["measured_positions"],
                "nonfinite_position_count": margins["nonfinite_positions"],
                "diagnostic_batch_count": margins["batches"],
                "decomposed_score_count": margins["decomposed_score_count"],
                "contextual_score_rms": margins["contextual_score_rms"],
                "transition_score_rms": margins["transition_score_rms"],
                "simulator_diversity_trait": None,
                "simulator_conformity_trait": None,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "(1 + mean off-diagonal cosine)/2 among up to 2048 contextual "
                    "item embeddings, sampled without replacement with local seed 42; "
                    "padding and transition tables excluded. Null for fewer than two items. "
                    "Higher indicates greater directional alignment; this alone does not "
                    "establish representation collapse or recommendation diversity."
                ),
                "pos_neg_margin_mean": (
                    "Mean combined positive-minus-negative logit on measured training "
                    "positions. Score-scale dependent; null without finite observations."
                ),
                "pos_neg_margin_p10": (
                    "Nearest-rank 10th percentile of measured training margins; "
                    "score-scale dependent and null without finite observations."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of measured training margins strictly below zero; ties excluded. "
                    "This is a sampled-pair diagnostic, not validation HR@10. Null if empty."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of measured training margins below eps=0.5; score-scale "
                    "dependent and null without finite observations."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Counts of measured margins below 0.5 and zero. Category breakdowns "
                    "are null because item categories are unavailable."
                ),
                "valid_position_count": (
                    "Number of inspected positions with nonzero sequence, positive and negative IDs."
                ),
                "measured_position_count": (
                    "Valid positions with finite positive score, negative score and margin; "
                    "the denominator for all margin rates."
                ),
                "nonfinite_position_count": (
                    "Valid positions excluded because a paired score or its margin is nonfinite."
                ),
                "diagnostic_batch_count": "Number of batches inspected, at most ten.",
                "decomposed_score_count": (
                    "Number of individual positive/negative scores included in the RMS "
                    "decomposition, requiring finite combined, contextual and residual scores."
                ),
                "contextual_score_rms": (
                    "sqrt(mean(contextual_logit squared)) over decomposed scores, using "
                    "the first D feature coordinates and the contextual item table. "
                    "Null without concatenated transition features or usable scores. "
                    "This contextual component comes from joint training, not a separate baseline fit."
                ),
                "transition_score_rms": (
                    "sqrt(mean((combined_logit - contextual_logit) squared)) over the "
                    "same decomposed scores. Includes the learned transition gate. "
                    "Null without concatenated transition features or usable scores. "
                    "Magnitude alone does not measure ranking benefit."
                ),
                "simulator_diversity_trait": "Explicitly unobserved; always null.",
                "simulator_conformity_trait": "Explicitly unobserved; always null.",
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
