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
    ### >>> Self_EvolveRec-BLOCK-START: Respect the diagnostic view and unavailable metadata contract
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._probe_batch = None

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in the matched benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Report unmeasurable embedding similarity explicitly
            if n < 2 or sample_size < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute exact mean cosine without a quadratic Gram matrix
            # Sum of all pairwise dot products equals the squared norm of
            # the vector sum. Subtract the actual diagonal, including any
            # zero vectors produced by normalization.
            x = x.float()
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
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
        ### >>> Self_EvolveRec-BLOCK-START: Cache one bounded diagnostic batch without resampling histories
        self._probe_batch = None
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic sampling without consuming an extra batch
        from itertools import islice

        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        batches = islice(dataloader, num_batches)
        ### <<< Self_EvolveRec-BLOCK-END
        for batch in batches:
            ### >>> Self_EvolveRec-BLOCK-START: Fail clearly on violations of the diagnostic loader contract
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Use device tensors and preserve the B x L diagnostic contract
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative must be B x L"
                )
            if seq_t.shape[1] == 0:
                raise ValueError("Diagnostic sequences must have positive length")
            if self._probe_batch is None:
                final_valid = (
                    seq_t[:, -1].ne(0)
                    & pos_t[:, -1].ne(0)
                    & neg_t[:, -1].ne(0)
                    & pos_t[:, -1].ne(neg_t[:, -1])
                )
                rows = final_valid.nonzero(as_tuple=True)[0][:16]
                if rows.numel():
                    self._probe_batch = tuple(
                        value.index_select(0, rows).detach().clone()
                        for value in (seq_t, pos_t, neg_t)
                    )
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and invalid sampled-negative comparisons
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0) & (pos_t != neg_t)
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Report nonfinite diagnostic scores as errors
            if valid.any():
                measured = margin[valid].detach().float()
                if not bool(torch.isfinite(measured).all()):
                    raise FloatingPointError(
                        "Nonfinite positive/negative margins on valid diagnostic positions"
                    )
                margins_all.append(measured.cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize measured totals without unavailable category lookups
            counts = (
                torch.stack(
                    (
                        (valid & margin.lt(float(eps))).sum(),
                        (valid & margin.lt(0.0)).sum(),
                    )
                )
                .cpu()
                .tolist()
            )
            low_total += int(counts[0])
            wrong_total += int(counts[1])
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Empty measurements are unavailable rather than perfect scores
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

        ### >>> Self_EvolveRec-BLOCK-START: Do not interpret missing item categories as measured category counts
        return margin_metrics, {
            "available": False,
            "reason": "Item categories are not provided by this benchmark",
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
        }

    @torch.no_grad()
    def analyze_recent_dependence(self):
        result = {
            "available": False,
            "reason": "No valid final-position diagnostic comparisons",
            "recent_alpha": None,
            "backbone_norm": None,
            "recent_pool_norm": None,
            "scaled_recent_pool_norm": None,
            "branch_removal_logit_rms": None,
            "unchanged_history_max_logit_difference": None,
            "recent_removal_logit_rms": None,
            "distant_removal_logit_rms": None,
            "recent_changed_users": 0,
            "distant_changed_users": 0,
            "measured_users": 0,
        }
        if self._probe_batch is None:
            return result
        seq, pos, neg = self._probe_batch
        if seq.shape[1] == 0:
            return result
        valid_users = (
            seq[:, -1].ne(0)
            & pos[:, -1].ne(0)
            & neg[:, -1].ne(0)
            & pos[:, -1].ne(neg[:, -1])
        )
        if not bool(valid_users.any()):
            return result
        seq, pos, neg = (value[valid_users] for value in (seq, pos, neg))
        items = torch.stack((pos[:, -1], neg[:, -1]), dim=-1).unsqueeze(1)

        def score(sequence):
            features = self.model.log2feats(sequence)[:, -1:, :]
            logits = self.model.predict_score(features, items).float()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    "Nonfinite encoder logits during recent-history dependence diagnosis"
                )
            return logits

        scores = score(seq)
        unchanged = score(seq.clone())
        result.update(
            {
                "available": True,
                "reason": None,
                "measured_users": int(seq.shape[0]),
                "unchanged_history_max_logit_difference": float(
                    (scores - unchanged).abs().max().item()
                ),
            }
        )
        for name, start, stop in (
            ("recent", max(0, seq.shape[1] - 5), seq.shape[1]),
            ("distant", max(0, seq.shape[1] - 10), max(0, seq.shape[1] - 5)),
        ):
            changed = seq[:, start:stop].ne(0).any(dim=1)
            result[f"{name}_changed_users"] = int(changed.sum().item())
            if bool(changed.any()):
                perturbed = seq.clone()
                perturbed[:, start:stop] = 0
                difference = (score(perturbed) - scores)[changed]
                result[f"{name}_removal_logit_rms"] = float(
                    difference.square().mean().sqrt().item()
                )

        backbone_fn = getattr(self.module, "backbone_log2feats", None)
        pool_fn = getattr(self.module, "pool_recent_items", None)
        alpha = getattr(self.module, "recent_alpha", None)
        if callable(backbone_fn) and callable(pool_fn) and alpha is not None:
            backbone = backbone_fn(seq)
            recent = pool_fn(seq)
            active = seq.ne(0)
            result["recent_alpha"] = float(alpha.detach().item())
            for name, values in (
                ("backbone_norm", backbone),
                ("recent_pool_norm", recent),
                ("scaled_recent_pool_norm", alpha * recent),
            ):
                result[name] = float(values.norm(dim=-1)[active].mean().item())
            removed = self.model.predict_score(backbone[:, -1:, :], items).float()
            result["branch_removal_logit_rms"] = float(
                (removed - scores).square().mean().sqrt().item()
            )
        return result

    def run_full_diagnosis(self, train_loader):
        # Preserve mixed submodule modes and global RNG streams. A loader's
        # private generator or iterator state remains owned by the caller.
        modes = [(module, module.training) for module in self.module.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        self.module.eval()
        try:
            with torch.no_grad():
                result = self._run_full_diagnosis(train_loader)
                result["metrics"][
                    "recent_dependence"
                ] = self.analyze_recent_dependence()
                result["metric_definitions"]["recent_dependence"] = (
                    "Fixed training-history, fixed-positive/negative candidate encoder-logit "
                    "dependence measurements; these are not validation HR or causal effects. "
                    "Norms average over nonpadding positions. Alpha is an unconstrained scalar. "
                    "Branch removal scores backbone features with identical candidates. Recent "
                    "removal masks the last five positions; distant removal masks the preceding "
                    "five. Perturbation RMS uses only users whose nonpadding history changed. "
                    "The unchanged-history maximum difference is a numerical control. Null "
                    "values denote unavailable measurements. Graph score contributions, if "
                    "present outside predict_score, are not measured by these encoder probes."
                )
                return result
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self._probe_batch = None

        # Review checks: diagnostic view contract, bounded loader consumption,
        # valid-position masks, finite scores, unavailable category metadata,
        # mode/RNG restoration, and linear-memory mean-cosine computation.
        # Static review only; no training or validation improvement is claimed.

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Keep the diagnostic implementation behind mode restoration
    def _run_full_diagnosis(self, train_loader):
        ### <<< Self_EvolveRec-BLOCK-END
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
                    "Range ~[0,1]. Estimated by mean pairwise cosine similarity of sampled item embeddings "
                    ### >>> Self_EvolveRec-BLOCK-START: Separate embedding alignment from unobserved recommendation diversity
                    "(mapped to [0,1]). Higher indicates stronger directional alignment; 0.5 "
                    "corresponds to zero mean cosine. This does not measure recommendation "
                    "diversity or simulator traits. Null means fewer than two sampled items."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                "pos_neg_margin_category_breakdown": (
                    ### >>> Self_EvolveRec-BLOCK-START: Document unavailable category metadata honestly
                    "Category breakdown is unavailable because item categories are not provided. "
                    "Only total low-margin and wrong-rank counts over valid training positions "
                    "are measured. Null margin statistics mean no valid comparisons."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
            },
        }
