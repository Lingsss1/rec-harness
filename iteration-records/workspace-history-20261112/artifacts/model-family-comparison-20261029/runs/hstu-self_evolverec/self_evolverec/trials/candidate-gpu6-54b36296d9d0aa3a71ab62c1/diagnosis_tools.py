import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Reproducible bounded candidate-panel diagnostics
import os
import time
import unicodedata
from itertools import islice, combinations
from collections import Counter

### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Interpret measured panels with explicit scope and censoring
# DEBUG: Remove the unexpected top-level indentation and restore the required callable.
def diagnosis_interpreter_prompt(raw_diagnosis):
    return (
        "Interpret the following measured HSTU training diagnostics using their "
        "metric_definitions. Report missing measurements explicitly. Raw margin "
        "values depend on score scale. The embedding cosine probe measures raw "
        "ID embeddings, whereas margins use the model's joint candidate scoring. "
        "A mapped cosine near 0.5 corresponds to mean cosine near zero and does "
        "not establish substantial directional collapse. Item categories, reviews "
        "and simulator traits are unavailable. These training measurements do not "
        "establish validation improvement, compatibility or user preferences. "
        "Controlled candidate panels are not implemented in this probe; do not "
        "invent panel results or interpret sampled negatives as exact catalog ranks.\n\n"
        + json.dumps(raw_diagnosis, indent=2, ensure_ascii=False, allow_nan=False)
    )


### <<< Self_EvolveRec-BLOCK-END


class DiagnosisProbe:
    def __init__(self, view, device):
        # DEBUG: Preserve the host ModelView contract and its actual trained module.
        self.view = view
        self.model = view
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # DEBUG: Item categories are unavailable in this benchmark.
        return []

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            # DEBUG: Pairwise cosine requires at least two real item embeddings.
            if n < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
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
        # DEBUG: ModelView is an adapter; evaluation mode belongs to view.model.
        self.view.model.eval()

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            # DEBUG: HSTU expects device-resident B x L integer tensors.
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.view.log2feats(seq_t)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            # DEBUG: Reject invalid scores instead of silently reporting fabricated metrics.
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if valid.any() and not bool(torch.isfinite(all_logits[valid]).all()):
                raise FloatingPointError("Nonfinite HSTU diagnostic candidate scores")
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            # DEBUG: Use the same valid causal positions for metrics and counts.
            valid_cpu = valid.detach().cpu().numpy()

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
            # DEBUG: No observed positions is missing evidence, not perfect separation.
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
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
            # DEBUG: Distinguish unavailable category metadata from measured zero counts.
            "available": False,
            "reason": "Item categories are not supplied by this benchmark",
            "valid_positions": sum(int(values.numel()) for values in margins_all),
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

    def run_full_diagnosis(self, train_loader):
        # DEBUG: Execute probes without gradients and restore the actual module's mode,
        # including HSTU's cache invalidation on train/eval transitions.
        was_training = self.view.model.training
        try:
            self.view.model.eval()
            with torch.no_grad():
                return self._run_full_diagnosis(train_loader)
        finally:
            self.view.model.train(was_training)

    def _run_full_diagnosis(self, train_loader):
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
                    # DEBUG: Describe the measured raw-ID similarity without unsupported diversity claims.
                    "Mean pairwise cosine of sampled raw ID embeddings, mapped to [0,1]. "
                    "A value near 0.5 means mean cosine near zero. This does not measure "
                    "joint title-aware vectors or recommendation diversity. Null if fewer "
                    "than two real catalog embeddings exist."
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                "pos_neg_margin_category_breakdown": (
                    # DEBUG: Keep unsupported metadata explicitly unavailable.
                    "Category breakdown is unavailable because item categories are not supplied. "
                    "Valid-position and low/wrong-margin totals describe measured training pairs; "
                    "empty category lists are not evidence of zero category-specific errors. "
                    "Null scalar margins mean no valid positions were observed."
                ),
            },
        }
