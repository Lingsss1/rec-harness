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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the diagnostic view contract and exclude unavailable metadata
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories, review contents and ratings are unavailable here.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Diagnose compatibility embeddings and represent unavailable estimates explicitly
            emb = self.view.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2 or sample_size < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Compute the same cosine diagnostic without a quadratic Gram matrix
            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise ValueError("Non-finite sampled item embeddings")
            x = F.normalize(x, p=2, dim=-1)
            # Sum of all pairwise dot products equals squared norm of the sum.
            # Subtract the actual diagonal, including any zero embeddings.
            summed = x.sum(dim=0, dtype=torch.float64)
            offdiag_sum = summed.square().sum() - x.square().sum(dtype=torch.float64)
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)
            ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Preserve actual module modes during margin diagnostics
        previous_modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            return self._measure_margins(dataloader, num_batches, eps, topk_categories)
        finally:
            for module, training in previous_modes:
                module.training = training

    @torch.no_grad()
    def _measure_margins(self, dataloader, num_batches, eps, topk_categories):
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic consumption exactly and reject malformed batches explicitly
        from itertools import islice

        if not isinstance(num_batches, int) or num_batches < 0:
            raise ValueError("num_batches must be a nonnegative integer")
        if not np.isfinite(eps):
            raise ValueError("eps must be finite")
        for batch in islice(dataloader, num_batches):
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError(
                    "Diagnosis expects (None, sequence, positive, negative) batches"
                )
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Use architecture-compatible view hooks with device-resident sequences
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis expects sequence, positive and negative tensors of shape B x L"
                )
            log_feats = self.view.log2feats(seq_t)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.view.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError("predict_score must return one score per candidate")
            margin = all_logits[..., 0] - all_logits[..., 1]
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            if not torch.isfinite(margin[valid]).all():
                raise ValueError("Non-finite diagnostic margins")
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unused item-ID transfers for unavailable category breakdowns
            # Aggregate counts require margins and validity only.
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Use the same valid positions for aggregate and breakdown counts
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize observed error counts while keeping unavailable category lists empty
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing diagnostic observations from measured zero
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

        ### >>> Self_EvolveRec-BLOCK-START: Mark category breakdowns unavailable while retaining observed margin counts
        breakdown = {
            "category_metadata_available": False,
            "valid_positions": sum(int(values.numel()) for values in margins_all),
            "eps": float(eps),
            ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Restore caller modes even when full diagnosis raises
    def run_full_diagnosis(self, train_loader):
        previous_modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            return self._run_full_diagnosis(train_loader)
        finally:
            for module, training in previous_modes:
                module.training = training

    def _run_full_diagnosis(self, train_loader):
        print(
            ">>> Running training-history diagnostics (category metadata unavailable)..."
        )
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Restrict diagnostic interpretations to measured training quantities
                "embedding_collapse_score": (
                    "Sampled mean pairwise item-embedding cosine mapped from [-1,1] to [0,1]. "
                    "A value near 0.5 means mean cosine near zero; larger values indicate "
                    "directional alignment. This is not recommendation diversity, effective "
                    "rank, or evidence of unobserved simulator traits. Null means unavailable."
                ),
                "pos_neg_margin_mean": (
                    "Mean positive-minus-negative score on valid training-loader positions. "
                    "Describes fitted-model separation, not independent generalization. Null means no observations."
                ),
                "pos_neg_margin_p10": (
                    "Lower empirical tenth percentile of observed training score margins; null means no observations."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of valid training positions with negative score above positive score; null means no observations."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of valid training margins below eps=0.5. Depends on score scale; null means no observations."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Observed valid, low-margin and wrong-rank position counts. Category lists "
                    "remain empty because category metadata is unavailable, not because no errors occurred. "
                    "These diagnostics do not replace pinned host validation HR@10."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
