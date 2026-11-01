import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic batch consumption exactly
from collections import Counter
from itertools import islice

### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Bind the actual BERT4Rec module and respect unavailable metadata
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this shared task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Measure catalog embeddings only and report empty geometry as unavailable
            emb = self.view.item_emb.weight.detach()[1 : self.model.itemnum + 1]
            n = emb.size(0)
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute the same mean cosine without a quadratic Gram matrix
            x = x.float()
            if not torch.isfinite(x).all():
                raise RuntimeError(
                    "Nonfinite catalog embeddings in diagnostic geometry"
                )
            # Sum of all pairwise dot products equals the squared norm of
            # the vector sum. Subtract the actual diagonal, including zeros.
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Preserve the trained module's mode during diagnostic inference
        was_training = self.model.training
        self.model.eval()
        try:
            return self._measure_masked_margins(
                dataloader, num_batches, eps, topk_categories
            )
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def _measure_masked_margins(self, dataloader, num_batches, eps, topk_categories):
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic consumption and reject malformed batches explicitly
        if not isinstance(num_batches, int) or num_batches < 0:
            raise ValueError("num_batches must be a nonnegative integer")
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnosis requires (None, sequence, positive, negative) batches"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score only valid supervised BERT4Rec mask positions without exposing labels
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis requires matching B x L sequence/positive/negative tensors"
                )
            itemnum = self.model.itemnum
            valid = (
                (seq_t == self.model.mask_token)
                & (pos_t >= 1)
                & (pos_t <= itemnum)
                & (neg_t >= 1)
                & (neg_t <= itemnum)
                & (neg_t != pos_t)
            )
            if not valid.any():
                continue

            # Labels are used exclusively for scoring, never as encoder inputs.
            features = self.model.log2feats(seq_t)[valid]
            all_items = torch.stack([pos_t[valid], neg_t[valid]], dim=-1)
            all_logits = self.model.predict_score(features, all_items)
            measured = all_logits[..., 0] - all_logits[..., 1]
            if not torch.isfinite(measured).all():
                raise RuntimeError("Nonfinite BERT4Rec diagnostic margins")
            margins_all.append(measured.detach().float().cpu())

            # The measured margins are already on CPU in margins_all[-1].
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count valid margins directly without dense tensors or extra device transfers
            measured_cpu = margins_all[-1]
            low_total += int((measured_cpu < float(eps)).sum().item())
            wrong_total += int((measured_cpu < 0.0).sum().item())
            # Category counters remain empty because annotations are unavailable.
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Represent absent supervised measurements as unavailable
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

        ### >>> Self_EvolveRec-BLOCK-START: Report measured sample counts and unavailable category annotations
        breakdown = {
            "category_status": "unavailable: item categories are not supplied in this task",
            "supervised_masked_positions": sum(int(m.numel()) for m in margins_all),
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
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Interpret catalog geometry without inferring recommendation diversity
                    "Mean off-diagonal cosine of sampled catalog embeddings, mapped by (cosine+1)/2. "
                    "PAD and MASK are excluded. Higher values indicate directional similarity; "
                    "0.5 corresponds to mean cosine zero. This does not measure recommendation diversity. "
                    "Null means fewer than two sampled catalog embeddings."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                ### >>> Self_EvolveRec-BLOCK-START: Define masked-position diagnostics separately from host ranking metrics
                "pos_neg_margin_mean": (
                    "Mean positive-minus-negative logit at supervised MASK positions with valid, distinct "
                    "catalog positive/negative IDs. Uses one training diagnostic negative per position; "
                    "does not estimate host 100-candidate HR@10. Null means no measured positions."
                ),
                "pos_neg_margin_p10": (
                    "Nearest-rank 10th percentile of measured masked-position margins. "
                    "Null means no measured positions."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of measured masked positions with margin < 0. "
                    "Null means no measured positions."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of measured masked positions with margin < eps=0.5. "
                    "This threshold depends on score scale. Null means no measured positions."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Reports measured supervised-mask count, margin threshold and low/wrong-margin counts. "
                    "Category annotations are unavailable; empty category lists are not measured zero counts. "
                    "Reviews, ownership and simulator traits are unobserved."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
