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
    - Treat null values and empty unavailable breakdowns as missing evidence, never as measured zeros.
    - Training-row sensitivity is descriptive and cannot establish validation improvements or observed user traits.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Separate the compatibility view from the trained module and enforce evidence boundaries
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        if not isinstance(self.network, torch.nn.Module):
            raise TypeError("view.model must be the actual trained torch.nn.Module")
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    def _eval_call(self, function, *args, **kwargs):
        # Preserve every submodule's mode, including any deliberately mixed modes.
        modes = [(module, module.training) for module in self.network.modules()]
        try:
            self.network.eval()
            with torch.no_grad():
                return function(*args, **kwargs)
        finally:
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute pairwise cosine alignment without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2 or sample_size < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite sampled item embeddings")
            # Cosine direction is undefined for zero vectors. Exclude them
            # rather than interpreting them as orthogonal observed directions.
            x = x[x.norm(dim=-1) > 0]
            k = x.size(0)
            if k < 2:
                return None
            x = F.normalize(x, p=2, dim=-1)
            # Sum of off-diagonal dot products equals ||sum(x)||^2 - sum(||x||^2).
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
            return round(float((mean_cos + 1.0) * 0.5), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Restore the actual module mode after bounded margin diagnostics
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        return self._eval_call(
            self._measure_pos_neg_margins,
            dataloader,
            num_batches=num_batches,
            eps=eps,
            topk_categories=topk_categories,
        )

    @torch.no_grad()
    def _measure_pos_neg_margins(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and reject invalid diagnostic batches explicitly
        from itertools import islice

        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        for batch in islice(iter(dataloader), num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                raise ValueError(
                    "Diagnosis expects (None, sequence, positive, negative) batches"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Pass device tensors through the SASRec compatibility contract
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or seq_t.shape != pos_t.shape
                or seq_t.shape != neg_t.shape
            ):
                raise ValueError(
                    "Diagnosis expects sequence, positive and negative tensors of shape B x L"
                )
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate compatibility scoring before computing diagnostic margins
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError("predict_score must return logits of shape B x L x 2")
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count only observed positions with distinct nonpadding candidates
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0) & pos_t.ne(neg_t)
            if not torch.isfinite(margin[valid]).all():
                raise FloatingPointError(
                    "Nonfinite positive-negative diagnostic margins"
                )
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize threshold counts while leaving unavailable categories empty
            low_total += int(np.count_nonzero(valid_cpu & (margin_cpu < float(eps))))
            wrong_total += int(np.count_nonzero(valid_cpu & (margin_cpu < 0.0)))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing margin evidence from measured zero
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

        ### >>> Self_EvolveRec-BLOCK-START: Report diagnostic support and unavailable category evidence
        margin_metrics["num_positions"] = int(sum(m.numel() for m in margins_all))
        breakdown = {
            "availability": "Item categories are unavailable; category lists are empty.",
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

    ### >>> Self_EvolveRec-BLOCK-START: Validate bounded paired interventions and report training rows without implying distinct users
    @torch.no_grad()
    def _measure_rank_sensitivity(self, batches, max_users=16):
        # The loader carries no user identities; this bounds rows, not unique users.
        if max_users < 1:
            raise ValueError("The diagnostic row limit must be positive")
        rows = []
        for batch in batches:
            if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                raise ValueError(
                    "Diagnosis expects (None, sequence, positive, negative) batches"
                )
            seq, pos, neg = [
                torch.as_tensor(x, dtype=torch.long).detach().cpu() for x in batch[1:4]
            ]
            if seq.ndim != 2 or seq.shape != pos.shape or seq.shape != neg.shape:
                raise ValueError(
                    "Diagnosis expects sequence, positive and negative tensors of shape B x L"
                )
            for s, p, n in zip(seq, pos, neg):
                # Require right alignment, a valid target, and enough history for
                # both a recent deletion and an equally sized older deletion.
                positions = s.ne(0).nonzero(as_tuple=True)[0]
                if positions.numel() < 7 or s[-1] == 0 or p[-1] == 0:
                    continue
                expected = torch.arange(s.numel() - positions.numel(), s.numel())
                if not torch.equal(positions, expected):
                    raise ValueError(
                        "Rank interventions require contiguous, left-padded histories"
                    )
                history = set(s.tolist())
                target = int(p[-1])
                negatives = sorted(set(n.tolist()) - history - {0, target})[:31]
                if not negatives:
                    continue
                rows.append((s, positions, [target] + negatives))
                if len(rows) == max_users:
                    break
            if len(rows) == max_users:
                break

        result = {
            "num_rows": len(rows),
            "distinct_users": None,
            "num_candidates": [],
            "comparisons": {},
            "sampling_unit": "Training rows; user identities are unavailable in loader batches.",
        }
        names = ("delete_recent", "delete_older", "shuffle_recent")
        if not rows:
            result["availability"] = (
                "No eligible diagnostic rows; comparisons are unmeasured."
            )
            return result

        sequences = torch.stack([row[0] for row in rows])
        variants = {name: sequences.clone() for name in names}
        rng = np.random.default_rng(42)
        for i, (s, positions, candidates) in enumerate(rows):
            # Preserve the final input item in every intervention. Recent means
            # the preceding four nonpadding items in the five-item window.
            history = s[positions]
            width = min(4, int(history.numel()) - 1)
            delete_count = min(width, int(history.numel()) - width - 1)
            recent_indices = torch.arange(
                history.numel() - width - 1, history.numel() - 1
            )
            for name, indices in (
                ("delete_recent", recent_indices[-delete_count:]),
                ("delete_older", torch.arange(delete_count)),
            ):
                keep = torch.ones(history.numel(), dtype=torch.bool)
                keep[indices] = False
                shortened = history[keep]
                variants[name][i].zero_()
                variants[name][i, -shortened.numel() :] = shortened
            shuffled = history.clone()
            permutation = torch.as_tensor(rng.permutation(width), dtype=torch.long)
            shuffled[recent_indices] = history[recent_indices[permutation]]
            variants["shuffle_recent"][i, positions] = shuffled
            result["num_candidates"].append(len(candidates))

        candidate_count = max(result["num_candidates"])
        candidate_tensor = torch.zeros(
            (len(rows), candidate_count), dtype=torch.long, device=self.device
        )
        candidate_mask = torch.zeros_like(candidate_tensor, dtype=torch.bool)
        for i, (_, _, candidates) in enumerate(rows):
            candidate_tensor[i, : len(candidates)] = torch.tensor(
                candidates, device=self.device
            )
            candidate_mask[i, : len(candidates)] = True

        def ranks(sequence):
            features = self.model.log2feats(sequence.to(self.device))[:, -1:, :]
            scores = self.model.predict_score(
                features, candidate_tensor.unsqueeze(1)
            ).squeeze(1)
            if scores.shape != candidate_tensor.shape:
                raise ValueError("predict_score must preserve the candidate dimension")
            if not torch.isfinite(scores[candidate_mask]).all():
                raise FloatingPointError("Nonfinite rank-sensitivity diagnostic scores")
            # Average ranks for ties, independent of candidate order.
            above = ((scores > scores[:, :1]) & candidate_mask).sum(-1).float()
            tied = ((scores == scores[:, :1]) & candidate_mask).sum(-1).float() - 1
            return above + 0.5 * tied

        baseline = ranks(sequences)
        denominators = torch.tensor(result["num_candidates"], device=self.device) - 1
        result["baseline_mean_normalized_rank"] = float(
            (baseline / denominators).mean().item()
        )
        for name, variant in variants.items():
            changed = ranks(variant)
            result["comparisons"][name] = {
                "mean_normalized_rank_change": float(
                    ((changed - baseline) / denominators).mean().item()
                ),
                "target_rank_worsens_rate": float(
                    (changed > baseline).float().mean().item()
                ),
            }
        return result

    def run_full_diagnosis(self, train_loader):
        return self._eval_call(self._run_full_diagnosis, train_loader)

    def _run_full_diagnosis(self, train_loader):
        # Cache at most ten batches once, supporting single-pass iterators and
        # paired comparisons without resampling targets or negatives.
        from itertools import islice

        batches = list(islice(iter(train_loader), 10))
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            batches, num_batches=10, eps=eps, topk_categories=20
        )
        sensitivity = self._measure_rank_sensitivity(batches, max_users=16)
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_num_positions": margin_metrics["num_positions"],
                "pos_neg_margin_category_breakdown": breakdown,
                "paired_rank_sensitivity": sensitivity,
                "evidence_availability": {
                    "missing_values": "null means unmeasured, not zero",
                    "item_categories": "unavailable",
                    "ratings_reviews_prices_stores": "unavailable",
                    "simulator_diversity_conformity_traits": "unobserved",
                },
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Mean off-diagonal cosine among nonzero vectors sampled from up to 2048 nonpadding item embeddings, mapped by (cos+1)/2. "
                    "About 0.5 indicates mean cosine near zero; larger values indicate directional alignment. "
                    "This alone does not measure effective dimension, recommendation diversity, or user preferences."
                ),
                "pos_neg_margin_mean": "Mean positive-minus-negative logit over observed positions with distinct nonpadding candidates; scale dependent.",
                "pos_neg_margin_p10": "Nearest-rank lower 10th percentile of those margins; scale dependent.",
                "pos_neg_neg_beats_pos_rate": "Fraction of measured positions with negative logit strictly above positive logit; ties are excluded from this rate.",
                "pos_neg_small_margin_rate": "Fraction of measured positions with positive-minus-negative logit below 0.5; scale dependent.",
                "pos_neg_margin_num_positions": "Number of measured training positions; zero means no eligible margin evidence.",
                "pos_neg_margin_category_breakdown": "Position counts at margin thresholds; item-category lists remain empty because categories are unavailable.",
                "paired_rank_sensitivity": (
                    "Descriptive evaluation-mode interventions on at most 16 contiguous, left-padded training rows with at least seven input items. "
                    "Rows may repeat users; distinct user counts are unmeasured. "
                    "For each row, the last positive target and up to 31 unique supplied training negatives excluding input history "
                    "are held fixed across all interventions. Zero-based target rank uses average ties and is divided by candidate_count-1; "
                    "smaller ranks are better. Comparisons report mean normalized rank change and fraction with worsening rank. "
                    "Recent deletion removes up to four items immediately before the final input item; older deletion removes an equal "
                    "number of oldest input items. Both compact and left-pad the surviving history, so positional shifts also contribute. "
                    "Shuffling permutes the four items preceding the final input item while preserving membership and the final item. "
                    "These interventions can create distribution shift; they do not establish causal user preferences. "
                    "This is not pinned validation HR@10 and must not select configurations."
                ),
                "evidence_availability": "Explicit evidence boundaries; unavailable attributes and simulator traits must not be inferred as observations.",
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
