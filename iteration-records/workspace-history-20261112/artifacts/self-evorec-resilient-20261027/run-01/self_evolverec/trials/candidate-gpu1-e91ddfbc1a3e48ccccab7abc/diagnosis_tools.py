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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret missing measurements and diagnostic limits explicitly
    - Every statement must be grounded in the provided metric values/definitions.
    - Null measurements are unavailable, not zero or evidence of healthy behavior. Report their support counts.
    - Nonfinite scores and excluded embeddings must be considered before interpreting aggregate measurements.
    - Training margins and embedding geometry cannot establish validation HR@10 improvements, recommendation diversity, simulator traits, or correct implementation of the pooling residual.
    - If available measurements look healthy, status should remain STABLE; if support is absent, state that model health cannot be assessed from these measurements.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Respect the diagnostic view and unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.embedding_support = {}

    def _get_categories(self, item_id: int):
        # Category annotations are unavailable in this benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            ### >>> Self_EvolveRec-BLOCK-START: Report embedding geometry with explicit measurement support
            catalog_count = emb.size(0)
            usable = torch.isfinite(emb).all(dim=-1) & (emb.norm(dim=-1) > 0)
            emb = emb[usable]
            n = emb.size(0)
            self.embedding_support = {
                "catalog_items": int(catalog_count),
                "usable_items": int(n),
                "excluded_items": int(catalog_count - n),
                "sampled_items": int(min(max(sample_size, 0), n)),
                "sampled_pairs": 0,
            }
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute identical pairwise geometry without a quadratic Gram matrix
            # Sum of all pairwise dot products equals the squared norm of the
            # vector sum. Subtract actual diagonal norms to retain the original
            # normalization semantics, including near-zero embedding vectors.
            x = x.to(torch.float64)
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            self.embedding_support["sampled_pairs"] = int(offdiag_cnt // 2)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    ### >>> Self_EvolveRec-BLOCK-START: Restore actual module modes after diagnostic scoring
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            return self._measure_pos_neg_margins(
                dataloader, num_batches, eps, topk_categories
            )
        finally:
            for module, training in modes:
                module.training = training

    def _measure_pos_neg_margins(self, dataloader, num_batches, eps, topk_categories):
        requested_positions = 0
        invalid_negative_positions = 0
        nonfinite_positions = 0
        measured_batches = 0
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume only the declared number of diagnostic batches
        from itertools import islice

        for step, batch in enumerate(islice(dataloader, max(0, num_batches))):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score the trained architecture using device tensors
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic tensor contracts and subtract scores in float32
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis requires sequence, positive and negative tensors of shape B x L"
                )
            if log_feats.shape[:2] != seq_t.shape or log_feats.ndim != 3:
                raise ValueError("Diagnostic log2feats must return B x L x H features")
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError(
                    "Diagnostic predict_score must return B x L x 2 scores"
                )
            # Avoid half-precision subtraction overflow for otherwise finite scores.
            all_logits = all_logits.float()
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing negatives and nonfinite diagnostic scores
            requested = pos_t != 0
            paired = requested & (neg_t != 0)
            finite = torch.isfinite(all_logits).all(dim=-1) & torch.isfinite(margin)
            valid = paired & finite
            requested_positions += int(requested.sum().item())
            invalid_negative_positions += int((requested & ~paired).sum().item())
            nonfinite_positions += int((paired & ~finite).sum().item())
            measured_batches += 1
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count measured failures without unavailable category lookups
            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            # Category counters remain empty because annotations are unavailable.
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Represent empty margin measurements as unavailable
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

        ### >>> Self_EvolveRec-BLOCK-START: Expose diagnostic support and unavailable category coverage
        measured_positions = sum(int(m.numel()) for m in margins_all)
        breakdown = {
            "support": {
                "measured_batches": measured_batches,
                "requested_positions": requested_positions,
                "measured_positions": measured_positions,
                "invalid_negative_positions": invalid_negative_positions,
                "nonfinite_positions": nonfinite_positions,
            },
            "category_annotations_available": False,
            "category_annotated_positions": 0,
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

    def run_full_diagnosis(self, train_loader):
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Include embedding measurement support
                "embedding_collapse_score": collapse_score,
                "embedding_support": self.embedding_support,
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Ground interpretations in geometry and sampled training separation
                "embedding_collapse_score": (
                    "Mean pairwise cosine of sampled finite, nonzero nonpadding item embeddings, "
                    "mapped to [0,1]. Near 0.5 indicates mean cosine near zero; larger values "
                    "indicate directional alignment. This measures geometry, not recommendation "
                    "diversity or user preferences. Null means fewer than two usable sampled items."
                ),
                "embedding_support": (
                    "Catalog, usable, excluded, and sampled item counts and unordered sampled-pair "
                    "count supporting the embedding geometry measurement."
                ),
                "pos_neg_margin_mean": (
                    "Mean positive-minus-sampled-negative logit on training positions with nonzero "
                    "positive and negative IDs and finite scores. Scale-dependent; not validation HR. "
                    "Null means no measured positions."
                ),
                "pos_neg_margin_p10": (
                    "Nearest-rank 10th percentile of measured training margins; scale-dependent. "
                    "Null means no measured positions."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of measured training pairs with negative score strictly above positive "
                    "score. Ties are excluded from this event. Measures sampled separation, not "
                    "full-catalog ranking or validation HR. Null means no measured positions."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of measured training margins below the reported eps threshold. "
                    "Depends on logit scale. Null means no measured positions."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Includes measurement support, excluded missing-negative and nonfinite counts, "
                    "and counts of measured margins below eps and zero. Category metadata is "
                    "unavailable: category lists are empty and annotated support is zero. "
                    "Simulator diversity/conformity traits and review attributes are unobserved."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
