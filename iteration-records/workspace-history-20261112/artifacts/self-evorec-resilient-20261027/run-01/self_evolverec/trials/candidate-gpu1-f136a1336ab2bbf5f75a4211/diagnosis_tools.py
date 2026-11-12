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
    ### >>> Self_EvolveRec-BLOCK-START: Ground interpretation in available measurements and their limitations
    - Null measurements are unavailable evidence, never zero or healthy observations.
    - Distinguish measured observations from hypotheses. If evidence is insufficient, explicitly say so; the required status is only a diagnostic summary.
    - Embedding alignment does not measure recommendation diversity or establish collapse by itself.
    - Neural training margins do not measure fused ranking quality, validation HR@10, or generalization.
    - Do not infer causal intent, simulator diversity/conformity traits, compatibility, rejection, or unavailable metadata.
    - Do not invent gains, uncertainty intervals, preference labels, or validation/test results.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the diagnostic view and explicit metadata boundaries
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        if not isinstance(self.network, torch.nn.Module):
            raise TypeError("view.model must be the trained torch.nn.Module")
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Category metadata is unavailable under the shared task contract.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute embedding alignment without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if int(sample_size) < 2:
            raise ValueError("Embedding alignment requires sample_size >= 2")
        self._embedding_undefined_reason = None
        self._embedding_sample_count = 0
        with torch.inference_mode():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            k = min(int(sample_size), n)
            self._embedding_sample_count = k
            if k < 2:
                self._embedding_undefined_reason = (
                    "Fewer than two nonpadding item embeddings"
                )
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            # Double precision stabilizes the subtraction in the identity below.
            x = emb.index_select(0, idx).to(dtype=torch.float64)
            if not torch.isfinite(x).all().item():
                self._embedding_undefined_reason = (
                    "Sampled item embeddings contain nonfinite values"
                )
                return None
            x = F.normalize(x, p=2, dim=-1)
            # sum_{i != j} <x_i,x_j> = ||sum_i x_i||^2 - sum_i ||x_i||^2.
            # Subtract actual norms so zero embeddings retain F.normalize semantics.
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore network mode for direct and aggregate diagnostic calls
        prior_mode = self.network.training
        try:
            self.network.eval()
            with torch.inference_mode():
                return self._measure_prefix_margins(
                    dataloader, num_batches, eps, topk_categories
                )
        finally:
            self.network.train(prior_mode)

    def _measure_prefix_margins(self, dataloader, num_batches, eps, topk_categories):
        margins_all = []
        queried_positions = 0
        nonfinite_positions = 0
        suffix_audit = {
            "queries": 0,
            "nonfinite_queries": 0,
            "failed_queries": 0,
            "max_feature_absolute_error": None,
            "max_score_absolute_error": None,
        }
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume at most ten diagnostic batches without fetching an extra batch
        from itertools import islice

        for batch in islice(dataloader, min(10, max(0, int(num_batches)))):
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Reject malformed diagnostic batches explicitly
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score actual prefixes and audit suffix independence once per batch
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative must have shape B x L"
                )

            catalog_size = self.model.item_emb.weight.shape[0] - 1
            for name, tensor in (
                ("sequence", seq_t),
                ("positive", pos_t),
                ("negative", neg_t),
            ):
                if ((tensor < 0) | (tensor > catalog_size)).any().item():
                    raise ValueError(
                        f"Diagnostic {name} contains IDs outside the model catalog"
                    )
            valid = (pos_t > 0) & (neg_t > 0) & (seq_t > 0)
            margin = torch.full(seq_t.shape, float("nan"), device=self.device)
            audited_batch = False
            # Explicitly truncate before feature extraction. This also protects
            # against suffix leakage from pooling added to a causal backbone.
            for column in range(seq_t.shape[1]):
                rows = torch.nonzero(valid[:, column], as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                prefix = seq_t.index_select(0, rows)[:, : column + 1]
                features = self.model.log2feats(prefix)[:, -1:, :]
                items = torch.stack(
                    (pos_t[rows, column], neg_t[rows, column]), dim=-1
                ).unsqueeze(1)
                logits = self.model.predict_score(features, items)
                if logits.shape != items.shape:
                    raise ValueError(
                        "view.predict_score must return one logit per requested item"
                    )
                margin[rows, column] = (logits[:, 0, 0] - logits[:, 0, 1]).float()

                # One additional forward per batch at most. Keep the prefix and
                # absolute positions fixed, and append a deterministically changed
                # suffix. Both representations and raw logits must agree.
                if not audited_batch and column + 1 < seq_t.shape[1]:
                    audited_batch = True
                    changed = seq_t[rows[:1]].clone()
                    suffix = changed[:, column + 1 :]
                    changed[:, column + 1 :] = torch.where(
                        suffix == 0, torch.ones_like(suffix), torch.zeros_like(suffix)
                    )
                    changed_features = self.model.log2feats(changed)[
                        :, column : column + 1, :
                    ]
                    changed_logits = self.model.predict_score(
                        changed_features, items[:1]
                    )
                    reference_features = features[:1]
                    reference_logits = logits[:1]
                    if (
                        changed_features.shape != reference_features.shape
                        or changed_logits.shape != reference_logits.shape
                    ):
                        raise ValueError(
                            "Suffix audit returned inconsistent feature or score shapes"
                        )
                    suffix_audit["queries"] += 1
                    tensors = (
                        reference_features,
                        changed_features,
                        reference_logits,
                        changed_logits,
                    )
                    if not all(torch.isfinite(t).all().item() for t in tensors):
                        suffix_audit["nonfinite_queries"] += 1
                    else:
                        feature_error = float(
                            (reference_features.double() - changed_features.double())
                            .abs()
                            .max()
                            .item()
                        )
                        score_error = float(
                            (reference_logits.double() - changed_logits.double())
                            .abs()
                            .max()
                            .item()
                        )
                        for key, error in (
                            ("max_feature_absolute_error", feature_error),
                            ("max_score_absolute_error", score_error),
                        ):
                            previous = suffix_audit[key]
                            suffix_audit[key] = (
                                error if previous is None else max(previous, error)
                            )
                        passed = torch.allclose(
                            reference_features, changed_features, rtol=1e-4, atol=1e-5
                        ) and torch.allclose(
                            reference_logits, changed_logits, rtol=1e-4, atol=1e-5
                        )
                        suffix_audit["failed_queries"] += int(not passed)

            queried_positions += int(valid.sum().item())
            finite = torch.isfinite(margin)
            nonfinite_positions += int((valid & ~finite).sum().item())
            valid = valid & finite
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
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
            ### >>> Self_EvolveRec-BLOCK-START: Do not turn missing diagnostic evidence into zero measurements
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

        ### >>> Self_EvolveRec-BLOCK-START: Report diagnostic denominators, suffix audits and unavailable categories
        breakdown = {
            "queried_positions": queried_positions,
            "prefix_suffix_audit": {
                **suffix_audit,
                "undefined_reason": (
                    "No valid prefix with room for an appended suffix"
                    if suffix_audit["queries"] == 0
                    else (
                        "All audited queries contained nonfinite features or scores"
                        if suffix_audit["queries"] == suffix_audit["nonfinite_queries"]
                        else None
                    )
                ),
            },
            "finite_positions": sum(int(m.numel()) for m in margins_all),
            "nonfinite_positions": nonfinite_positions,
            "undefined_reason": (
                "No finite margins at positions with a real source, target and negative"
                if not margins_all
                else None
            ),
            "categories_available": False,
            "category_unavailable_reason": "Category metadata is unavailable in this task",
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

    ### >>> Self_EvolveRec-BLOCK-START: Report neural measurements, explicit failures and unavailable fused evidence
    def run_full_diagnosis(self, train_loader):
        prior_mode = self.network.training
        try:
            self.network.eval()
            with torch.inference_mode():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
        finally:
            self.network.train(prior_mode)

        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "embedding_sample_count": self._embedding_sample_count,
                "embedding_undefined_reason": self._embedding_undefined_reason,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "fused_training_ranking": None,
                "fused_training_ranking_unavailable_reason": (
                    "The compatibility view does not specify sparse transition artifacts, "
                    "frozen score scales, or a selected fusion configuration. No fused "
                    "ranking or leave-one-occurrence-out measurement was performed."
                ),
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Mean off-diagonal cosine similarity of up to 2048 nonpadding item "
                    "embeddings sampled with seed 42, mapped by (cosine+1)/2. Zero vectors "
                    "use torch normalization semantics. This measures embedding alignment, "
                    "not recommendation diversity, user preference, or ranking quality."
                ),
                "embedding_sample_count": "Number of sampled item embedding rows, excluding padding.",
                "embedding_undefined_reason": "Reason alignment is null, or null when defined.",
                "pos_neg_margin_mean": (
                    "Mean raw neural positive-minus-loader-negative logit over finite valid "
                    "positions in the first ten batches. Features use only sequence through "
                    "the source position, preserving its positional indices. In-sample training "
                    "diagnostic; excludes transition and popularity residuals."
                ),
                "pos_neg_margin_p10": "Nearest-rank empirical 10th percentile of those same neural margins.",
                "pos_neg_neg_beats_pos_rate": "Fraction of those neural margins strictly below zero; ties are excluded.",
                "pos_neg_small_margin_rate": (
                    "Fraction of those neural margins below 0.5. This threshold depends on "
                    "logit scale and cannot certify ranking quality."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Includes queried, finite and nonfinite position counts, and counts of "
                    "margins below 0.5 and zero. Valid positions require nonzero source, "
                    "positive and negative IDs. Undefined margin statistics are null with "
                    "a reason. Category lists remain empty because categories are unavailable. "
                    "prefix_suffix_audit checks at most one valid query per batch by comparing "
                    "explicit-prefix features and neural scores against the same position with "
                    "a changed appended suffix. Finite comparisons use rtol=1e-4, atol=1e-5; "
                    "nonfinite queries are reported separately. Zero audited queries are "
                    "undefined, not a passed check. This bounded audit cannot prove causality "
                    "for every possible input and does not measure fused score invariance."
                ),
                "fused_training_ranking": (
                    "Unavailable here. A valid fused diagnostic requires actual prefix "
                    "histories, frozen training-derived scales, and subtraction of one "
                    "queried adjacent occurrence from its pair count, source total, target "
                    "frequency and frequency total, recomputing all affected probabilities. "
                    "Such subtraction would not make neural scores out-of-sample."
                ),
                "fused_training_ranking_unavailable_reason": "Explains missing fused evidence; never a measured result.",
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
