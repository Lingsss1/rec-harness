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
    ### >>> Self_EvolveRec-BLOCK-START: Keep compatibility view distinct from the trained module and unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute finite supported mean cosine without a quadratic Gram matrix
    @torch.no_grad()
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        emb = self.model.item_emb.weight.detach()[1:]
        n = emb.shape[0]
        self.embedding_sample_count = 0
        self.embedding_excluded_count = 0
        if n == 0 or sample_size < 1:
            return None

        generator = torch.Generator(device=emb.device)
        generator.manual_seed(seed)
        indices = torch.randperm(n, generator=generator, device=emb.device)[
            : min(sample_size, n)
        ]
        sampled = emb.index_select(0, indices).float()
        finite = torch.isfinite(sampled).all(dim=-1)
        sampled = sampled[finite]
        norms = sampled.norm(dim=-1)
        sampled = sampled[norms > 0]
        k = sampled.shape[0]
        self.embedding_sample_count = int(k)
        self.embedding_excluded_count = int(indices.numel() - k)
        if k < 2:
            return None

        x = F.normalize(sampled, p=2, dim=-1).double()
        # sum(i != j) dot(x_i, x_j) = ||sum(x_i)||^2 - sum ||x_i||^2.
        offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
        mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
        return round(float((mean_cos + 1.0) * 0.5), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Restore actual module mode even when a probe fails
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(self, *args, **kwargs):
        previous_mode = self.module.training
        try:
            self.module.eval()
            return self._analyze_pos_neg_margin_with_category_breakdown(*args, **kwargs)
        finally:
            self.module.train(previous_mode)

    ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def _analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained module through the compatibility view
        self.module.eval()
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

            ### >>> Self_EvolveRec-BLOCK-START: Use device tensors for the SASRec compatibility contract
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and nonfinite margins
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            valid = valid & torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the identical valid-position mask for breakdowns
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
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Represent absent margin evidence explicitly
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                ### <<< Self_EvolveRec-BLOCK-END
            }
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
            ### >>> Self_EvolveRec-BLOCK-START: Report the valid margin sample count
            "valid_positions": sum(int(m.numel()) for m in margins_all),
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

    ### >>> Self_EvolveRec-BLOCK-START: Run every diagnostic without gradients and restore the trained module mode
    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        previous_mode = self.module.training
        try:
            self.module.eval()
            return self._run_full_diagnosis(train_loader)
        finally:
            self.module.train(previous_mode)

    ### <<< Self_EvolveRec-BLOCK-END

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Report actual valid embedding support and excluded samples
                "embedding_sample_count": self.embedding_sample_count,
                "embedding_excluded_count": self.embedding_excluded_count,
                "embedding_collapse_score": collapse_score,
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Avoid equating average cosine with effective rank or recommendation diversity
                    "Mean pairwise cosine of sampled nonpadding item embeddings mapped to [0,1]. "
                    "Higher indicates greater average directional alignment; 0.5 corresponds to zero mean cosine. "
                    "This statistic alone does not establish embedding collapse or recommendation diversity. "
                    "Null when fewer than two embeddings are available."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                ### >>> Self_EvolveRec-BLOCK-START: Define finite embedding support and training-margin interpretation
                "embedding_sample_count": "Number of sampled nonpadding embeddings with finite coordinates and strictly positive FP32 norm used for cosine; cosine is null with fewer than two valid embeddings.",
                "embedding_excluded_count": "Number of sampled embeddings excluded because coordinates are nonfinite or FP32 norm is zero. Exclusions require investigation and are not evidence of healthy embeddings.",
                "pos_neg_margin_mean": "Mean positive-minus-negative training logit over finite, nonpadding positions; null without samples. Pairwise separation does not establish sampled HR@10 or full-catalog ranking.",
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                "pos_neg_margin_category_breakdown": (
                    ### >>> Self_EvolveRec-BLOCK-START: Declare unavailable category evidence
                    "Training-position counts for margin<eps and margin<0, with valid_positions reporting "
                    "finite nonpadding sample support. Category lists remain empty because item categories "
                    "are unavailable in this task."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
            },
        }
