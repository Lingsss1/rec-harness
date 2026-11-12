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
    ### >>> Self_EvolveRec-BLOCK-START: Keep the view contract and track the actual trained module
    def __init__(self, view, device):
        self.model = view
        self.trained_model = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.residual_metrics = {}

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute sampled directional alignment without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if sample_size < 2:
            raise ValueError("Cosine alignment requires sample_size >= 2")
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not bool(torch.isfinite(x).all().item()):
                raise FloatingPointError("Nonfinite sampled item embeddings")
            # Exclude undefined cosines rather than treating zero vectors as directions.
            norms = x.norm(dim=-1)
            self.residual_metrics["embedding_near_zero_sample_count"] = int(
                (norms <= 1e-8).sum().item()
            )
            x = x[norms > 1e-8]
            k = x.shape[0]
            if k < 2:
                return None
            x = F.normalize(x, p=2, dim=-1, eps=1e-8)

            # sum_{i != j} x_i dot x_j = ||sum_i x_i||^2 - sum_i ||x_i||^2.
            # Float64 accumulation limits cancellation without allocating K x K.
            x = x.double()
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
            return round(float((mean_cos + 1.0) * 0.5), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Accumulate implemented branch geometry and signed margin contributions
        margins_all = []
        residual_sums = {
            name: 0.0
            for name in (
                "encoder_norm",
                "branch_norm",
                "scaled_branch_norm",
                "raw_item_norm",
                "unit_item_norm",
                "raw_mean_norm",
                "unit_mean_norm",
                "near_zero_item_count",
                "near_zero_mean_rate",
                "residual_backbone_ratio",
                "backbone_margin_contribution",
                "branch_margin_contribution",
                "absolute_branch_margin_contribution",
                "gate_mean",
                "gate_second_moment",
            )
        }
        coherence_sums = Counter()
        coherence_counts = Counter()
        length_counts = Counter()
        residual_count = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            ### >>> Self_EvolveRec-BLOCK-START: Reject malformed diagnostic batches instead of silently losing observations
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnosis requires (None, sequence, positive, negative) batches"
                )
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Pass device tensors to the trained augmented encoder
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic tensor shapes before candidate scoring
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence and targets must share shape B x L"
                )
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            ### <<< Self_EvolveRec-BLOCK-END
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Decompose the actual selected branch without undefined coherence values
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())
                actual = self.trained_model
                if hasattr(actual, "recent_components"):
                    parts = actual.recent_components(seq_t)
                    branch = parts["branch"]
                    scaled = actual.alpha * branch
                    encoder = log_feats - scaled
                    contribution = actual.predict_score(scaled, all_items)
                    contribution = contribution[..., 0] - contribution[..., 1]
                    encoder_norm = encoder.norm(dim=-1)
                    scaled_norm = scaled.norm(dim=-1)
                    unit_mean_norm = parts["unit_mean"].norm(dim=-1)
                    values = {
                        "encoder_norm": encoder_norm,
                        "branch_norm": branch.norm(dim=-1),
                        "scaled_branch_norm": scaled_norm,
                        "raw_item_norm": parts["raw_item_norm"],
                        "unit_item_norm": parts["unit_item_norm"],
                        "raw_mean_norm": parts["raw_mean"].norm(dim=-1),
                        "unit_mean_norm": unit_mean_norm,
                        "near_zero_item_count": parts["near_zero_items"].float(),
                        "near_zero_mean_rate": (
                            unit_mean_norm <= actual.residual_eps
                        ).float(),
                        "residual_backbone_ratio": scaled_norm
                        / encoder_norm.clamp_min(actual.residual_eps),
                        "backbone_margin_contribution": margin - contribution,
                        "branch_margin_contribution": contribution,
                        "absolute_branch_margin_contribution": contribution.abs(),
                        "gate_mean": parts["gate"],
                        "gate_second_moment": parts["gate"].square(),
                    }
                    residual_count += int(valid.sum().item())
                    for name, value in values.items():
                        observed = value[valid]
                        if not bool(torch.isfinite(observed).all().item()):
                            raise FloatingPointError(
                                f"Nonfinite residual diagnostic: {name}"
                            )
                        residual_sums[name] += float(observed.sum().item())
                    for length in range(1, actual.residual_window + 1):
                        group = valid & (parts["length"] == length)
                        defined = group & parts["coherence_defined"]
                        length_counts[length] += int(group.sum().item())
                        coherence_counts[length] += int(defined.sum().item())
                        coherence_sums[length] += float(
                            parts["coherence"][defined].sum().item()
                        )
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical validity masks for all margin summaries
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Count observed margins directly when categories are unavailable
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            # Category counters remain empty under the matched data contract.
            ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing margin observations from measured zero values
        if len(margins_all) == 0:
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
        else:
            m = torch.cat(margins_all, dim=0)
            if not bool(torch.isfinite(m).all().item()):
                raise FloatingPointError(
                    "Nonfinite diagnostic positive-negative margins"
                )
            N = m.numel()
            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())
            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }
        ### <<< Self_EvolveRec-BLOCK-END

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        ### >>> Self_EvolveRec-BLOCK-START: Report all controls and distinguish undefined measurements
        actual = self.trained_model
        if hasattr(actual, "recent_components"):
            self.residual_metrics.update(
                {
                    "recent_variant": actual.residual_variant,
                    "recent_alpha": float(actual.alpha.detach().item()),
                    "recent_valid_positions": residual_count,
                    **{
                        "recent_"
                        + name: value / residual_count if residual_count else None
                        for name, value in residual_sums.items()
                    },
                    "recent_coherence_by_length": {
                        str(length): {
                            "positions": length_counts[length],
                            "defined_positions": coherence_counts[length],
                            "mean": (
                                coherence_sums[length] / coherence_counts[length]
                                if coherence_counts[length]
                                else None
                            ),
                        }
                        for length in range(1, actual.residual_window + 1)
                    },
                }
            )
            mean = self.residual_metrics.pop("recent_gate_mean")
            second = self.residual_metrics.pop("recent_gate_second_moment")
            self.residual_metrics["recent_gate_statistics"] = (
                {"mean": mean, "std": max(0.0, second - mean * mean) ** 0.5}
                if residual_count and actual.residual_variant == "gated"
                else None
            )
        grads = [p.grad.detach() for p in actual.parameters() if p.grad is not None]
        self.residual_metrics["existing_gradients_finite"] = (
            bool(all(torch.isfinite(g).all().item() for g in grads)) if grads else None
        )
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Restore model mode and describe measured residual controls without unavailable metadata
    def run_full_diagnosis(self, train_loader):
        self.residual_metrics = {}
        prior_mode = self.trained_model.training
        try:
            self.trained_model.eval()
            with torch.no_grad():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
            return {
                "metrics": {
                    "embedding_collapse_score": collapse_score,
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    **self.residual_metrics,
                },
                "metric_definitions": {
                    "embedding_collapse_score": (
                        "Sampled mean off-diagonal item cosine mapped to [0,1]. "
                        "Higher indicates directional alignment, not definitive representation "
                        "collapse or recommendation diversity. Null if fewer than two items exist."
                    ),
                    "pos_neg_margin_mean": "Mean augmented positive-minus-negative logit over valid training positions; null if none were observed.",
                    "pos_neg_margin_p10": "10th percentile of augmented positive-minus-negative logits; null if no valid positions were observed.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid positions with margin below zero; null if none were observed.",
                    "pos_neg_small_margin_rate": "Fraction of valid positions with margin below 0.5; null if none were observed.",
                    "pos_neg_margin_category_breakdown": (
                        "Position totals for margins below 0.5 or zero. Category lists are empty "
                        "because item category metadata is unavailable."
                    ),
                    "recent_alpha": "Direct unconstrained scalar, initialized to zero; may become negative.",
                    "recent_valid_positions": "Number of valid training positions used for branch diagnostics.",
                    "recent_encoder_norm": "Mean L2 norm of the masked backbone state before residual addition; null if no valid positions.",
                    "recent_branch_norm": "Mean L2 norm of the selected implemented branch before alpha scaling; null if no valid positions.",
                    "recent_variant": "Selected control: baseline, raw mean, unit mean (recent), or coherence-gated unit mean.",
                    "embedding_near_zero_sample_count": "Sampled nonpadding embeddings with norm at most 1e-8; excluded from item cosine alignment.",
                    "recent_raw_item_norm": "Mean within-window raw embedding norm, then averaged over supervised positions.",
                    "recent_unit_item_norm": "Mean within-window epsilon-normalized embedding norm, then averaged over supervised positions.",
                    "recent_raw_mean_norm": "Mean norm of the unnormalized causal-window embedding mean.",
                    "recent_unit_mean_norm": "Mean norm of the mean of individually epsilon-normalized embeddings; the pooled mean is not renormalized.",
                    "recent_near_zero_item_count": "Mean number of window embeddings with norm at most the serialized epsilon per supervised position.",
                    "recent_near_zero_mean_rate": "Fraction of supervised positions whose unit-mean vector has norm at most the serialized epsilon.",
                    "recent_residual_backbone_ratio": "Mean per-position ratio of the norm of alpha times the selected branch to the epsilon-clamped backbone norm.",
                    "recent_backbone_margin_contribution": "Mean signed backbone contribution to the sampled positive-minus-negative logit.",
                    "recent_branch_margin_contribution": "Mean signed alpha-scaled branch contribution to the sampled positive-minus-negative logit; negative alpha is retained without a positive-interest interpretation.",
                    "recent_coherence_by_length": "Counts and mean distinct-position-pair cosine for each causal nonpadding window length. Padding and diagonal pairs are excluded. Mean is null for singleton windows or when no windows have all embedding norms above epsilon.",
                    "recent_gate_statistics": "Mean and population standard deviation of the implemented coherence gate over supervised positions for the gated control; null for other controls or no observations.",
                    "recent_scaled_branch_norm": "Mean L2 norm of alpha times the recent branch; null if no valid positions.",
                    "recent_absolute_branch_margin_contribution": (
                        "Mean absolute alpha-scaled branch contribution to the positive-minus-negative "
                        "logit using the trained tied scoring function; null if no valid positions."
                    ),
                    "existing_gradients_finite": (
                        "Whether all currently retained parameter gradients are finite; null when "
                        "no gradients are retained. Diagnosis performs no backward pass."
                    ),
                },
            }
        finally:
            self.trained_model.train(prior_mode)

    ### <<< Self_EvolveRec-BLOCK-END
