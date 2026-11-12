import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic consumption and restore module modes on every exit
from collections import Counter
from functools import wraps
from itertools import islice


def _evaluation_probe(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            return method(self, *args, **kwargs)
        finally:
            # Preserve mixed modes, such as a frozen backbone in evaluation mode.
            for module, training in modes:
                module.training = training

    return wrapped


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
    - Null or unavailable values indicate missing evidence, not healthy measurements.
    - Nonfinite counts indicate numerical problems; do not ignore them when reading finite-only summaries.
    - Training pairwise ranks do not establish validation HR@10 or full-catalog recommendation quality.
    - Simulator diversity and conformity traits are unobserved; do not infer them from embedding geometry.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the view contract and exclude unavailable attributes
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = device
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in the matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Compute cosine evidence without a quadratic Gram matrix
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            k = min(max(0, int(sample_size)), n)
            self.embedding_sample = {
                "sample_count": k,
                "finite_item_count": 0,
                "nonfinite_item_count": 0,
                "zero_vector_count": 0,
                "pair_count": 0,
                "status": "unavailable_insufficient_items",
            }
            if k == 0:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            finite = torch.isfinite(x).all(dim=-1)
            self.embedding_sample["nonfinite_item_count"] = int((~finite).sum().item())
            x = x[finite].double()
            count = x.shape[0]
            self.embedding_sample["finite_item_count"] = count
            self.embedding_sample["pair_count"] = count * (count - 1) // 2
            self.embedding_sample["zero_vector_count"] = int(
                (x.norm(dim=-1) == 0).sum().item()
            )
            if count < 2:
                return None

            x = F.normalize(x, p=2, dim=-1)
            # Sum_{i != j} <x_i, x_j> = ||sum_i x_i||^2 - sum_i ||x_i||^2.
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (count * (count - 1))).clamp(-1, 1)
            self.embedding_sample["status"] = (
                "measured_finite_subset"
                if self.embedding_sample["nonfinite_item_count"]
                else "measured"
            )
            return round(float((mean_cos.item() + 1.0) * 0.5), 4)
            ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Restore diagnostic mode even when loading or scoring fails
    @torch.no_grad()
    @_evaluation_probe
    def analyze_pos_neg_margin_with_category_breakdown(
        ### <<< Self_EvolveRec-BLOCK-END
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic limits and collect supported observations
        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        if not np.isfinite(eps):
            raise ValueError("eps must be finite")
        margins_all = []
        final_margins = []
        ratios = []
        valid_count = nonfinite_count = 0
        zero_backbone_count = nonfinite_feature_count = 0
        captured_backbone = []
        backbone_layer = getattr(self.module, "last_layernorm", None)
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume exactly the requested batches and reject invalid adapter output
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic batches must be (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Tensor inputs, masked pairwise ranks, and observed residual ratios
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
            if seq_t.shape[1] == 0:
                raise ValueError("Diagnostic sequences must have positive length")
            captured_backbone.clear()
            hook = None
            if isinstance(backbone_layer, torch.nn.Module):
                hook = backbone_layer.register_forward_hook(
                    lambda module, inputs, output: captured_backbone.append(
                        output.detach().clone()
                    )
                )
            try:
                log_feats = self.model.log2feats(seq_t)
            finally:
                if hook is not None:
                    hook.remove()

            if captured_backbone and captured_backbone[-1].shape == log_feats.shape:
                base = captured_backbone[-1].float()
                residual = log_feats.float() - base
                base_norm = base.norm(dim=-1)
                residual_norm = residual.norm(dim=-1)
                finite_features = torch.isfinite(base_norm) & torch.isfinite(
                    residual_norm
                )
                observed = seq_t != 0
                nonfinite_feature_count += int(
                    (observed & ~finite_features).sum().item()
                )
                zero_backbone_count += int(
                    (observed & finite_features & (base_norm == 0)).sum().item()
                )
                supported = observed & finite_features & (base_norm > 0)
                if supported.any():
                    ratios.append(
                        (residual_norm[supported] / base_norm[supported]).cpu()
                    )

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError(
                    "predict_score must return B x L x 2 scores for paired items"
                )
            margin = all_logits[..., 0].float() - all_logits[..., 1].float()
            eligible = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            finite = torch.isfinite(all_logits).all(dim=-1) & torch.isfinite(margin)
            valid_count += int(eligible.sum().item())
            nonfinite_count += int((eligible & ~finite).sum().item())
            valid = eligible & finite
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            # Select the actual final nonpadding sequence position, rather than
            # substituting an earlier position when its target is unavailable.
            positions = torch.arange(seq_t.shape[1], device=seq_t.device)
            last = (
                positions.unsqueeze(0)
                .expand_as(seq_t)
                .masked_fill(seq_t == 0, -1)
                .max(dim=1)
                .values
            )
            rows = torch.arange(seq_t.shape[0], device=seq_t.device)
            safe_last = last.clamp_min(0)
            final_valid = (last >= 0) & valid[rows, safe_last]
            if final_valid.any():
                final_margins.append(
                    margin[rows[final_valid], safe_last[final_valid]].float().cpu()
                )

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = valid.cpu().numpy()
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
            ### >>> Self_EvolveRec-BLOCK-START: Represent absent training margins explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Report supported measurements without unavailable category claims
        finals = torch.cat(final_margins) if final_margins else torch.empty(0)
        ratio_values = torch.cat(ratios) if ratios else torch.empty(0)
        margin_metrics.update(
            {
                "valid_pair_count": valid_count,
                "finite_pair_count": valid_count - nonfinite_count,
                "nonfinite_pair_count": nonfinite_count,
                "status": "measured" if margins_all else "unavailable_no_finite_pairs",
                "final_position_pairwise_rank": {
                    "sample_count": int(finals.numel()),
                    "positive_wins_rank_0": int((finals > 0).sum().item()),
                    "exact_ties_rank_interval_0_1": int((finals == 0).sum().item()),
                    "negative_wins_rank_1": int((finals < 0).sum().item()),
                    "status": (
                        "measured"
                        if finals.numel()
                        else "unavailable_no_finite_final_pairs"
                    ),
                },
                "residual_backbone_norm_ratio": {
                    "status": (
                        "measured"
                        if ratio_values.numel()
                        else "unavailable_no_supported_backbone_outputs"
                    ),
                    "sample_count": int(ratio_values.numel()),
                    "zero_backbone_positions": zero_backbone_count,
                    "nonfinite_feature_positions": nonfinite_feature_count,
                    "quantiles": (
                        {
                            name: float(torch.quantile(ratio_values, q).item())
                            for name, q in (
                                ("p50", 0.5),
                                ("p90", 0.9),
                                ("p99", 0.99),
                                ("max", 1.0),
                            )
                        }
                        if ratio_values.numel()
                        else None
                    ),
                },
            }
        )
        breakdown = {
            "status": "unavailable",
            "reason": "Item categories are not provided.",
        }
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve all module modes across the complete diagnostic workflow
    @_evaluation_probe
    def run_full_diagnosis(self, train_loader):
        ### <<< Self_EvolveRec-BLOCK-END
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        ### >>> Self_EvolveRec-BLOCK-START: Run protected probes and summarize public interaction histories
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        histories = self.user_train.get("History", self.user_train)
        repeated_events = total_events = history_count = 0
        for history in histories.values():
            if not isinstance(history, (list, tuple, np.ndarray)):
                continue
            items = [int(item) for item in history if int(item) != 0]
            history_count += 1
            total_events += len(items)
            repeated_events += len(items) - len(set(items))
        # Embedding support is recorded by the probe after finite-value filtering.
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                ### >>> Self_EvolveRec-BLOCK-START: Expose sample counts, missing states, and residual evidence
                "pos_neg_margin_category_breakdown": breakdown,
                "training_pair_diagnostics": margin_metrics,
                "embedding_sample": dict(self.embedding_sample),
                "history_duplicate_prevalence": {
                    "history_count": history_count,
                    "event_count": total_events,
                    "repeat_event_count": repeated_events,
                    "repeat_event_fraction": (
                        repeated_events / total_events if total_events else None
                    ),
                },
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Limit embedding similarity interpretation to measured geometry
                    "Mean off-diagonal cosine similarity of sampled nonpadding item embeddings, mapped to [0,1]. "
                    "Approximately 0.5 represents mean cosine zero. Higher indicates directional alignment, "
                    "not a direct measure of recommendation diversity, coverage, or user preferences. "
                    "Nonfinite sampled embeddings are excluded and counted. Zero vectors follow torch normalization "
                    "conventions; null means fewer than two finite sampled embeddings."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define diagnostic limits, tie handling, and residual measurement support
                "pos_neg_margin_category_breakdown": "Unavailable: item categories are not provided in this task.",
                "training_pair_diagnostics": (
                    "At most ten training batches in evaluation mode, using the trained scoring path. "
                    "Eligible pairs require nonpadding sequence, positive and negative IDs; nonfinite scores are excluded "
                    "and counted separately. These are sampled training margins, not validation HR@10 or full-catalog ranks. "
                    "Final-position pairwise ranks compare one positive with one sampled negative at the final nonpadding "
                    "sequence position: strict positive wins have rank 0, strict losses rank 1, exact ties span ranks 0..1. "
                    "Residual ratios are ||returned features - captured last_layernorm output|| / ||captured output|| "
                    "at finite nonpadding positions with nonzero backbone norm. This hook assumes last_layernorm is the "
                    "final backbone normalization; zero-backbone and nonfinite positions are reported separately. "
                    "Quantiles summarize observed ratios only; they do not prove an architectural bound or bounded logits. "
                    "Missing measurements use null values or explicit unavailable states."
                ),
                "embedding_sample": (
                    "Sampled nonpadding item count, finite and nonfinite item counts, zero-vector count among finite "
                    "items, and unordered pair count among finite sampled items. Partial measurements are labeled."
                ),
                "history_duplicate_prevalence": (
                    "Public training histories only: repeat_event_count is the sum of history length minus unique-item "
                    "count, excluding padding; its fraction uses all observed events as denominator. This describes "
                    "whole histories, not specifically the residual pooling window."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
