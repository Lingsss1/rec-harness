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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the compatibility view and actual trained module
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.embedding_sample_count = 0

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable directional alignment as null
            n = emb.size(0)
            self.embedding_sample_count = min(sample_size, n)
            if self.embedding_sample_count < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Compute directional alignment in float32 and reject invalid embeddings
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all().item():
                raise FloatingPointError(
                    "Non-finite sampled item embeddings in diagnosis"
                )
            x = F.normalize(x, p=2, dim=-1)
            gram = x @ x.t()
            ### <<< Self_EvolveRec-BLOCK-END
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Let the full-diagnosis entry point manage module evaluation mode
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Diagnose the same blended tensor features used in training
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Validate candidate scoring and vectorize training-margin diagnosis
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis expects matching B x L sequence, positive and negative tensors"
                )
            if log_feats.shape[:2] != seq_t.shape:
                raise ValueError("Diagnosis features must have shape B x L x D")

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError("predict_score must return B x L x 2 candidate logits")

            # Match the training loss mask exactly. Check logits before subtraction
            # so infinities cannot masquerade as ordinary ranking failures.
            valid = pos_t != 0
            selected_logits = all_logits[valid].detach().float()
            if not torch.isfinite(selected_logits).all().item():
                raise FloatingPointError(
                    "Non-finite valid candidate logits in diagnosis"
                )
            measured = (selected_logits[:, 0] - selected_logits[:, 1]).cpu()
            if not torch.isfinite(measured).all().item():
                raise FloatingPointError(
                    "Non-finite positive-negative margins in diagnosis"
                )
            if measured.numel():
                margins_all.append(measured)
                low_total += int((measured < float(eps)).sum().item())
                wrong_total += int((measured < 0.0).sum().item())
            # Category counters remain empty because categories are unavailable.
            ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Preserve missing margin measurements rather than reporting zero failures
        if len(margins_all) == 0:
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

        ### >>> Self_EvolveRec-BLOCK-START: Count measured positions and explicitly mark unavailable categories
        breakdown = {
            "valid_sample_count": sum(int(m.numel()) for m in margins_all),
            "categories_available": False,
            "category_unavailability_reason": "Item categories are not provided in this task",
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

    ### >>> Self_EvolveRec-BLOCK-START: Measure training margins without changing model mode or inventing unavailable evidence
    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        print(">>> Running training-history alignment and margin diagnosis...")
        # Preserve mixed submodule modes as well as the root module's mode.
        modes = [(module, module.training) for module in self.module.modules()]
        self.module.eval()
        try:
            alignment = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=0.5, topk_categories=20
                )
            )
            return {
                "metrics": {
                    # Retained key for compatibility; interpretation is directional alignment.
                    "embedding_collapse_score": alignment,
                    "embedding_sample_count": self.embedding_sample_count,
                    "embedding_pair_count": self.embedding_sample_count
                    * max(0, self.embedding_sample_count - 1)
                    // 2,
                    "pos_neg_valid_sample_count": breakdown["valid_sample_count"],
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_category_breakdown": breakdown,
                },
                "metric_definitions": {
                    "embedding_collapse_score": (
                        "Legacy key: (mean pairwise item-embedding cosine + 1)/2. "
                        "Measures directional similarity, not proof of collapse or recommendation diversity. "
                        "Null when fewer than two items can be sampled."
                    ),
                    "embedding_sample_count": "Number of non-padding item embeddings sampled.",
                    "embedding_pair_count": "Number of distinct unordered sampled embedding pairs.",
                    "pos_neg_valid_sample_count": "Number of sampled training positions with positive != 0; repeated positions may occur.",
                    "pos_neg_margin_mean": "Mean positive minus negative logit on sampled training positions; null if empty.",
                    "pos_neg_margin_p10": "Nearest-rank 10th percentile of sampled training margins; null if empty.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of sampled training margins below zero; null if empty.",
                    "pos_neg_small_margin_rate": "Fraction of sampled training margins below 0.5; scale dependent; null if empty.",
                    "pos_neg_margin_category_breakdown": (
                        "Counts low-margin and wrong-rank training positions. Category lists remain empty "
                        "because item categories are unavailable. These are training diagnostics, not held-out accuracy."
                    ),
                },
            }
        finally:
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END
