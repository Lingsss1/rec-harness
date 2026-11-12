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
    ### >>> Self_EvolveRec-BLOCK-START: Retain the compatibility view and evaluate the actual deployed module
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("view.model must be the actual deployed nn.Module")
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._residual_pairs = []

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute directional concentration without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
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
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite sampled item embeddings")
            x = F.normalize(x, p=2, dim=-1, eps=1e-8)

            # Sum of all pairwise dot products equals the squared norm
            # of the vector sum. Subtract self-pairs, including zero vectors.
            total = x.sum(dim=0, dtype=torch.float64)
            diagonal = x.double().square().sum()
            mean_cos = (
                ((total.square().sum() - diagonal) / (k * (k - 1))).clamp(-1, 1).item()
            )
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
        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic consumption and require explicit residual-off semantics
        from itertools import islice

        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        if not np.isfinite(eps):
            raise ValueError("eps must be finite")
        self._residual_pairs = []
        margins_all = []
        base_scorer = getattr(self.model, "predict_score_base", None)
        if base_scorer is not None and not callable(base_scorer):
            raise TypeError("view.predict_score_base must be callable when provided")
        feature_layout = getattr(self.model, "feature_layout", None)
        if feature_layout is None:
            feature_layout = getattr(self.module, "feature_layout", None)
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Reject malformed batches without consuming an extra diagnostic batch
        for step, batch in enumerate(islice(dataloader, num_batches)):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    f"Diagnostic batch {step} must contain (None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned diagnostic tensors before model scoring
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative tensors must share shape B x L"
                )
            log_feats = self.model.log2feats(seq_t)
            if log_feats.ndim != 3 or log_feats.shape[:2] != seq_t.shape:
                raise ValueError(
                    "view.log2feats must return aligned B x L x D features"
                )
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError(
                    "view.predict_score must return aligned B x L x 2 scores"
                )
            margin = all_logits[..., 0].float() - all_logits[..., 1].float()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Compare residual-off scores only through an explicit scoring contract
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            if valid.any():
                if (
                    not torch.isfinite(all_logits[valid]).all()
                    or not torch.isfinite(margin[valid]).all()
                ):
                    raise ValueError("Nonfinite deployed diagnostic scores or margins")
                margins_all.append(margin[valid].detach().float().cpu())
                base_logits = None
                if base_scorer is not None:
                    base_logits = base_scorer(log_feats, all_items)
                elif feature_layout == "sasrec_recent_concat":
                    dim = self.model.item_emb.weight.shape[-1]
                    if log_feats.shape[-1] != 2 * dim:
                        raise ValueError(
                            "sasrec_recent_concat requires exactly 2D features"
                        )
                    base_logits = (
                        log_feats[..., :dim].unsqueeze(-2)
                        * self.model.item_emb(all_items)
                    ).sum(-1)
                if base_logits is not None:
                    if base_logits.shape != all_logits.shape:
                        raise ValueError(
                            "Residual-off scores must share the deployed B x L x 2 shape"
                        )
                    if not torch.isfinite(base_logits[valid]).all():
                        raise ValueError("Nonfinite residual-off diagnostic scores")
                    self._residual_pairs.append(
                        torch.stack([base_logits[valid], all_logits[valid]], dim=1)
                        .detach()
                        .float()
                        .cpu()
                    )
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical valid positions for margins and breakdown counts
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Missing diagnostic samples are unavailable rather than healthy zeros
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
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

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Restore deployed mode and report evidence-qualified paired residual diagnostics
    def run_full_diagnosis(self, train_loader):
        # Preserve mixed submodule modes as well as the root module's mode.
        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            with torch.no_grad():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=eps, topk_categories=20
                    )
                )
        finally:
            for module, training in prior_modes:
                module.training = training

        paired = {
            "status": "unavailable",
            "reason": "No valid paired observations with view.predict_score_base or explicit feature_layout='sasrec_recent_concat'",
        }
        if self._residual_pairs:
            scores = torch.cat(self._residual_pairs, dim=0)
            # Axis 1: residual off / deployed; axis 2: positive / negative.
            outranks = scores[..., 1] > scores[..., 0]
            ties = scores[..., 1] == scores[..., 0]
            paired = {
                "status": "available",
                "candidate_protocol": "Identical supplied positive/negative pairs; positive-first stable ties",
                "positions": int(scores.shape[0]),
                "residual_off_mean_positive_rank": float(
                    (1 + outranks[:, 0].long()).float().mean()
                ),
                "deployed_mean_positive_rank": float(
                    (1 + outranks[:, 1].long()).float().mean()
                ),
                "residual_off_negative_outranking_fraction": float(
                    outranks[:, 0].float().mean()
                ),
                "deployed_negative_outranking_fraction": float(
                    outranks[:, 1].float().mean()
                ),
                "residual_off_tie_fraction": float(ties[:, 0].float().mean()),
                "deployed_tie_fraction": float(ties[:, 1].float().mean()),
                "corrected_pairs": int((outranks[:, 0] & ~outranks[:, 1]).sum()),
                "regressed_pairs": int((~outranks[:, 0] & outranks[:, 1]).sum()),
                "mean_absolute_score_change": float(
                    (scores[:, 1] - scores[:, 0]).abs().mean()
                ),
            }
        self._residual_pairs = []

        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "residual_fixed_candidate_comparison": paired,
                "unavailable_observations": {
                    "item_categories": None,
                    "reviews_and_ratings": None,
                    "item_prices_and_stores": None,
                    "product_specifications": None,
                    "simulator_diversity_and_conformity_traits": None,
                },
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Mean sampled pairwise embedding cosine mapped to [0,1]; approximately 0.5 "
                    "means mean cosine zero. Measures directional concentration only, not recommendation "
                    "diversity, satisfaction, or simulator traits. Zero vectors use F.normalize semantics; "
                    "null means fewer than two catalog embeddings."
                ),
                "pos_neg_margin_mean": (
                    "Mean deployed positive-minus-negative score on valid training positions. "
                    "Depends on score scale; null means no valid observations."
                ),
                "pos_neg_margin_p10": "Empirical lower tenth percentile of deployed training-pair margins; scale dependent.",
                "pos_neg_neg_beats_pos_rate": "Fraction of valid supplied training pairs where the negative strictly outscores the positive.",
                "pos_neg_small_margin_rate": "Fraction of deployed training-pair margins below eps=0.5; scale dependent and not calibrated across models.",
                "pos_neg_margin_category_breakdown": (
                    "Counts valid low-margin and wrong-rank positions. Category lists remain empty "
                    "because category observations are unavailable."
                ),
                "residual_fixed_candidate_comparison": (
                    "Paired comparison on identical supplied positive/negative training candidates. "
                    "Residual-off scores use view.predict_score_base when provided, or the first D features "
                    "dotted with raw candidate embeddings when feature_layout='sasrec_recent_concat' is explicitly declared. "
                    "Feature width alone does not establish scoring semantics. Deployed scores use view.predict_score. "
                    "Positive rank is 1 or 2 with positive-first ties. These training-pair "
                    "diagnostics are not pinned validation HR@10 or full-catalog ranks and do not select coefficients."
                ),
                "unavailable_observations": (
                    "Null denotes unavailable evidence. Histories and embedding similarity do not establish "
                    "ownership, dislike, satisfaction, compatibility, complementarity, redundancy, current need, "
                    "or simulator traits."
                ),
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
