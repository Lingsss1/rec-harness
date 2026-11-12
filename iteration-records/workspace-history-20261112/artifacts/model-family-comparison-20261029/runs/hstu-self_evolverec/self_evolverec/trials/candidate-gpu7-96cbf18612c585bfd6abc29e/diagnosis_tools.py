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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret exact-catalog training audits and numerical checks conservatively
    - Ground every statement in measured metrics and their definitions; distinguish observations from hypotheses.
    - Catalog and sampled-hit expectations use in-sample training targets and an auxiliary uniform-tie convention. They do not establish validation performance, intent, compatibility, shortcuts, or generalization.
    - Candidate-pool and tie equivalence to the pinned evaluator are unknown through this adapter. Numerical path disagreements make analytic results conditional on the reference path.
    - Block-dependent target scores imply blockwise operational ranks, not a proven shape-independent ordering. Near ties never replace exact equality.
    - Null means unavailable, never zero. Report exclusions, retained support, sampling fractions, arithmetic failures and numerical flags.
    - Whole-user intervals at at most sixteen users are exploratory and conditional on fitted weights and diagnostic arithmetic. Degenerate intervals do not establish certainty; resamples are not additional users.
    - Wilson intervals apply only to binary catalog hits. Analytic sampled probabilities and fractional tie expectations are not binomial outcomes.
    - Sampling discrepancy concerns one frozen model, not a model-comparison reversal or an expected validation gain. Five negative draws are not independent users.
    - Do not infer representation rank or recommendation diversity from mean cosine, or infer unavailable metadata, simulator traits or elapsed-time effects.
    - Only pinned four-category validation macro HR@10 selects models. Never recommend independently fitted score fusion or ensembles.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the HSTU ModelView contract and unavailable metadata boundaries
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = getattr(view, "user_train", {})
        self.itemnum = int(self.network.itemnum)
        self.maxlen = int(self.network.maxlen)
        self.datasets = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        names = [
            getattr(owner, key, None)
            for owner in (view, self.network)
            for key in ("dataset_name", "dataset", "category_name")
        ]
        routes = {
            name for name in names if isinstance(name, str) and name in self.datasets
        }
        self.category = next(iter(routes)) if len(routes) == 1 else None
        self.category_index = (
            self.datasets.index(self.category) if self.category is not None else 4
        )

    def _get_categories(self, item_id: int):
        # Item-category attributes are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Sample actual effective HSTU embeddings without materializing the catalog
        with torch.inference_mode():
            n = self.itemnum
            if n < 2:
                return None
            rng = np.random.default_rng(seed)
            k = min(int(sample_size), n)
            if k < 2:
                return None
            idx = torch.as_tensor(
                rng.choice(n, size=k, replace=False) + 1,
                dtype=torch.long,
                device=self.device,
            )
            x = self.model.item_emb(idx).detach().float()
            if not torch.isfinite(x).all() or (x.norm(dim=-1) == 0).any():
                return None
            ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Manage the actual module mode at the diagnostic boundary
        # run_full_diagnosis owns evaluation mode and restores it in finally.
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

            ### >>> Self_EvolveRec-BLOCK-START: Validate loader tensors before HSTU embedding lookups
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic loader must supply matching B x L tensors")
            if not 1 <= seq_t.size(1) <= self.maxlen:
                raise ValueError(
                    "Diagnostic sequence length exceeds the HSTU position table"
                )
            for name, tensor in (
                ("sequence", seq_t),
                ("positive", pos_t),
                ("negative", neg_t),
            ):
                if ((tensor < 0) | (tensor > self.itemnum)).any():
                    raise ValueError(
                        f"Diagnostic {name} contains IDs outside 0..itemnum"
                    )
            log_feats = self.model.log2feats(seq_t.detach().cpu().numpy())
            # Two aligned calls use the same scoring path as HSTU training.
            pos_logits = self.model.predict_score(log_feats, pos_t)
            neg_logits = self.model.predict_score(log_feats, neg_t)
            if pos_logits.shape != seq_t.shape or neg_logits.shape != seq_t.shape:
                raise ValueError("Aligned HSTU scores must have shape B x L")
            margin = pos_logits - neg_logits
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Restrict pair diagnostics to finite ordinary-item training positions
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            valid = (
                (seq_t != 0)
                & (pos_t > 0)
                & (pos_t <= self.itemnum)
                & (neg_t > 0)
                & (neg_t <= self.itemnum)
                & torch.isfinite(margin)
            )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same validity mask for pair summaries
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize position counts when item categories are unavailable
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Represent undefined sampled-pair statistics explicitly
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

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Bounded exact-catalog HSTU audit with analytic sampling and numerical checks
    def _audit_prefixes(self, batches):
        histories = (
            self.user_train.get("History", {})
            if isinstance(self.user_train, dict)
            else {}
        )
        if not isinstance(histories, dict):
            raise ValueError(
                "ModelView.user_train['History'] must map user IDs to histories"
            )
        exclusions = Counter()
        records = []
        coverage = {
            "source": "identified_training_users" if histories else "training_rows",
            "eligible_users": 0,
            "selected_users": 0,
            "unique_users": 0 if histories else None,
            "selected_rows": 0,
            "prefixes_requested": 0,
            "user_cap": 32,
            "targets_per_user_or_row": 2,
        }
        if histories:
            ordered = sorted(histories, key=lambda key: (str(key), type(key).__name__))
            eligible = []
            for index, uid in enumerate(ordered):
                try:
                    h = [int(item) for item in histories[uid]]
                except (TypeError, ValueError):
                    exclusions["malformed_history_user"] += 1
                    continue
                if any(item < 1 or item > self.itemnum for item in h):
                    exclusions["noncatalog_history_user"] += 1
                    continue
                if len(h) < 5 or self.maxlen < 4:
                    exclusions["history_too_short_user"] += 1
                    continue
                eligible.append((index, uid, h))
            coverage["eligible_users"] = len(eligible)
            rng = np.random.default_rng(
                np.random.SeedSequence([42, self.category_index])
            )
            chosen = sorted(
                rng.choice(
                    len(eligible), min(32, len(eligible)), replace=False
                ).tolist()
            )
            coverage["selected_users"] = len(chosen)
            for picked in chosen:
                index, uid, h = eligible[picked]
                rng = np.random.default_rng(
                    np.random.SeedSequence([42, self.category_index, index])
                )
                positions = sorted(
                    rng.choice(
                        np.arange(4, len(h)), min(2, len(h) - 4), replace=False
                    ).tolist()
                )
                for j in positions:
                    records.append((index, j, h[max(0, j - self.maxlen) : j], h[j]))
        else:
            # Rows have no reliable user identity. Never infer one from sequences.
            row_index = 0
            for batch in batches:
                if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                    continue
                seq = torch.as_tensor(batch[1]).detach().cpu().numpy()
                pos = torch.as_tensor(batch[2]).detach().cpu().numpy()
                for sequence, positive in zip(seq, pos):
                    if coverage["selected_rows"] >= 32:
                        break
                    candidates = []
                    for t in range(len(sequence)):
                        h = [int(x) for x in sequence[: t + 1] if int(x) != 0][
                            -self.maxlen :
                        ]
                        target = int(positive[t])
                        if (
                            len(h) >= 4
                            and 1 <= target <= self.itemnum
                            and all(1 <= x <= self.itemnum for x in h)
                        ):
                            candidates.append((t, h, target))
                    if not candidates:
                        exclusions["no_eligible_target_row"] += 1
                    else:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([42, self.category_index, row_index])
                        )
                        selected = sorted(
                            rng.choice(
                                len(candidates), min(2, len(candidates)), replace=False
                            ).tolist()
                        )
                        for selected_index in selected:
                            t, h, target = candidates[selected_index]
                            records.append((row_index, t, h, target))
                        coverage["selected_rows"] += 1
                    row_index += 1
                if coverage["selected_rows"] >= 32:
                    break
            coverage["rows_examined"] = row_index
        coverage["prefixes_requested"] = len(records)
        return records, coverage, exclusions

    def _score_prefix(self, index, position, history, target, counters):
        rng = np.random.default_rng(
            np.random.SeedSequence([42, self.category_index, index, position])
        )
        # This implementation has only padding=0 and ordinary IDs 1..itemnum.
        # Exclude this prefix, never future events or held-out interactions.
        pool = np.setdiff1d(
            np.arange(1, self.itemnum + 1, dtype=np.int64),
            np.asarray(list(set(history) | {target}), dtype=np.int64),
            assume_unique=False,
        )
        if not len(pool):
            return None, "no_eligible_negative"
        negatives = rng.choice(pool, min(999, len(pool)), replace=False)
        candidates = np.concatenate(([target], negatives))
        n = len(history)
        k = min(3, (n - 2) // 2)
        names = [
            "original",
            "recent_removed",
            "oldest_removed",
            "final_removed",
            "recent_preserving_final",
            "shuffle_0",
            "shuffle_1",
            "shuffle_2",
        ]
        variants = [
            history,
            history[:-k],
            history[k:],
            history[:-1],
            history[: -(k + 1)] + history[-1:],
        ]
        variants.extend(
            rng.permutation(history[:-1]).tolist() + history[-1:] for _ in range(3)
        )
        sequences = np.zeros((len(variants), self.maxlen), dtype=np.int64)
        for row, items in enumerate(variants):
            sequences[row, -len(items) :] = items
        seq_t = torch.as_tensor(sequences, dtype=torch.long, device=self.device)
        features = self.model.log2feats(sequences)
        counters["sequence_evaluations"] += len(variants)
        if features.ndim != 3 or features.shape[:2] != seq_t.shape:
            raise ValueError("HSTU log2feats must return [variants, maxlen, hidden]")
        indices = torch.arange(self.maxlen, device=features.device).expand_as(seq_t)
        last = indices.masked_fill(seq_t == 0, -1).amax(dim=1)
        readout = features[torch.arange(len(variants), device=features.device), last]
        scores = []
        for start in range(0, len(candidates), 256):
            items = torch.as_tensor(
                candidates[start : start + 256], dtype=torch.long, device=self.device
            )
            scores.append(
                self.model.predict_score(readout, items).detach().float().cpu().numpy()
            )
            counters["candidate_score_calls"] += 1
        scores = np.concatenate(scores, axis=-1)
        counters["candidate_scores"] += len(variants) * len(candidates)
        if scores.shape != (len(variants), len(candidates)):
            raise ValueError(
                "HSTU diagnostic score contract must return [variants, candidates]"
            )
        if not np.isfinite(scores).all():
            return None, "nonfinite_scores"
        target_scores = scores[:, :1]
        ranks = (
            1
            + (scores > target_scores).sum(axis=1)
            + ((scores == target_scores) & (candidates[None, :] < target)).sum(axis=1)
        )
        values = np.stack([ranks, 1.0 / ranks, (ranks <= 10).astype(float)], axis=1)
        outcomes = {name: value for name, value in zip(names, values)}
        outcomes["shuffle"] = np.mean(values[5:8], axis=0)
        effects = {
            name: outcomes[name] - outcomes["original"]
            for name in (
                "recent_removed",
                "oldest_removed",
                "final_removed",
                "recent_preserving_final",
                "shuffle",
            )
        }
        # Direct equal-count comparison, in addition to effects versus original.
        effects["recent_minus_oldest_removal"] = (
            outcomes["recent_removed"] - outcomes["oldest_removed"]
        )
        effects["final_minus_prefinal_removal"] = (
            outcomes["final_removed"] - outcomes["recent_preserving_final"]
            if k == 1
            else None
        )
        effects = {key: value for key, value in effects.items() if value is not None}
        tied = ((scores == target_scores) & (candidates[None, :] != target)).any(axis=1)
        return {
            "original": outcomes["original"],
            "effects": effects,
            "candidate_count": len(candidates),
            "tie_frequency": float(tied.mean()),
        }, None

    def _history_audit(self, batches):
        records, coverage, exclusions = self._audit_prefixes(batches)
        counters = {
            "sequence_evaluations": 0,
            "candidate_score_calls": 0,
            "candidate_scores": 0,
        }
        grouped = {}
        counts, ties = [], []
        for index, position, history, target in records:
            result, reason = self._score_prefix(
                index, position, history, target, counters
            )
            if reason:
                exclusions[reason] += 1
                continue
            grouped.setdefault(index, []).append(result)
            counts.append(result["candidate_count"])
            ties.append(result["tie_frequency"])
        reliable = coverage["source"] == "identified_training_users"
        coverage["prefixes_scored"] = len(counts)
        coverage["unique_users"] = len(grouped) if reliable else None
        coverage["rows_scored"] = len(grouped) if not reliable else None
        coverage["exclusions"] = dict(exclusions)
        labels = ("rank", "reciprocal_rank", "hit10")

        def as_metrics(values):
            return {label: float(value) for label, value in zip(labels, values)}

        baseline = None
        effects = {}
        if grouped:
            baseline = as_metrics(
                np.mean(
                    [
                        np.mean([record["original"] for record in rows], axis=0)
                        for rows in grouped.values()
                    ],
                    axis=0,
                )
            )
            names = sorted(
                set.intersection(
                    *[
                        set(record["effects"])
                        for rows in grouped.values()
                        for record in rows
                    ]
                )
            )
            rng = np.random.default_rng(42)
            bootstrap_indices = (
                rng.integers(0, len(grouped), size=(2000, len(grouped)))
                if reliable
                else None
            )
            for name in names:
                clusters = np.asarray(
                    [
                        np.mean([record["effects"][name] for record in rows], axis=0)
                        for rows in grouped.values()
                    ]
                )
                interval = None
                if reliable and len(grouped) >= 2:
                    boot = clusters[bootstrap_indices].mean(axis=1)
                    lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
                    interval = {
                        label: [float(lo), float(hi)]
                        for label, lo, hi in zip(labels, lower, upper)
                    }
                effects[name] = {
                    "mean": as_metrics(clusters.mean(axis=0)),
                    "user_cluster_ci95": interval,
                    "ci_null_reason": (
                        "Loader rows have no verified user IDs"
                        if not reliable
                        else (
                            "At least two identified users are required for a user-cluster interval"
                            if len(grouped) < 2
                            else None
                        )
                    ),
                    "cluster_effect_p10": as_metrics(
                        np.quantile(clusters, 0.1, axis=0)
                    ),
                    "cluster_effect_p90": as_metrics(
                        np.quantile(clusters, 0.9, axis=0)
                    ),
                }
        result = {
            "category": self.category,
            "category_null_reason": (
                None
                if self.category
                else "ModelView does not identify a matched category"
            ),
            "coverage": coverage,
            "cost_counts": counters,
            "original": baseline,
            "effects": effects,
            "null_reason": None if grouped else "No complete eligible matched prefixes",
            "candidate_counts": {
                "min": int(min(counts)) if counts else None,
                "max": int(max(counts)) if counts else None,
                "mean": float(np.mean(counts)) if counts else None,
                "fraction_at_most_10": (
                    float(np.mean(np.asarray(counts) <= 10)) if counts else None
                ),
            },
            "target_tie_frequency": (
                float(
                    np.mean(
                        [
                            np.mean([record["tie_frequency"] for record in rows])
                            for rows in grouped.values()
                        ]
                    )
                )
                if grouped
                else None
            ),
            "undefined_effects": {
                "final_minus_prefinal_removal": (
                    "Equal-count comparison requires k=1 for every scored prefix"
                    if grouped and "final_minus_prefinal_removal" not in effects
                    else (
                        "No complete eligible matched prefixes" if not grouped else None
                    )
                ),
            },
            "protocol": {
                "seed": 42,
                "maxlen": self.maxlen,
                "negative_cap": 999,
                "shuffle_repeats": 3,
                "bootstrap_replicates": 2000,
                "input_channels": "item IDs plus learned absolute positions; no elapsed-time channel",
                "position_policy": "compact deletions, restore left padding; shuffle nonfinal items with final item fixed",
                "identity_scope": "category-local user IDs; cross-category identity is unverified",
                "selection_metric": "Host macro validation HR@10 remains unchanged",
            },
        }
        # A probe receives one actual trained category model. Do not manufacture
        # a four-category estimate from one category or pool local user IDs.
        return {
            "per_category": {
                name: result if name == self.category else None
                for name in self.datasets
            },
            "unidentified_category_result": result if self.category is None else None,
            "macro_four_category_mean": None,
            "macro_four_category_ci95": None,
            "macro_null_reason": "This invocation contains one category checkpoint; all four category user-cluster records are required to recompute a macro estimate and bootstrap",
        }

    @staticmethod
    def _catalog_outcome(g, e):
        ranks = np.arange(1 + g, 2 + g + e, dtype=np.float64)
        return {
            "g": int(g),
            "e": int(e),
            "rank_optimistic": int(1 + g),
            "rank_pessimistic": int(1 + g + e),
            "hit10": float(np.clip((10 - g) / (e + 1), 0, 1)),
            "reciprocal_rank": float(np.mean(1.0 / ranks)),
        }

    @staticmethod
    def _analytic_sample(N, g, e):
        from scipy.special import logsumexp
        from scipy.stats import hypergeom

        if not (0 <= g <= N and 0 <= e <= N - g):
            raise ValueError("Invalid catalog counts: require 0 <= g, e and g+e <= N")
        if N < 99:
            return {"hit10": None, "null_reason": "Fewer than 99 eligible competitors"}
        m, lower = 99, N - g - e

        # Factor the joint distribution into two hypergeometrics. SciPy's
        # logpmf avoids subtracting large independently evaluated log factorials.
        a, b = np.meshgrid(
            np.arange(min(g, m) + 1), np.arange(min(e, m) + 1), indexing="ij"
        )
        c = m - a - b
        valid = (c >= 0) & (c <= lower)
        a, b, c = a[valid], b[valid], c[valid]
        logp = hypergeom.logpmf(a, N, g, m)
        remaining_population = N - g
        if remaining_population:
            logp = logp + hypergeom.logpmf(b, remaining_population, e, m - a)
        else:
            # Every competitor is strictly above the target: A=m, B=0.
            logp = np.where((a == m) & (b == 0), logp, -np.inf)
        mass = float(np.exp(logsumexp(logp)))
        value = float(np.sum(np.exp(logp) * np.clip((10 - a) / (b + 1), 0, 1)))
        checks = {"probability_mass": mass, "mass_error": abs(mass - 1.0)}
        if e == 0:
            checks["untied_cdf_error"] = abs(value - float(hypergeom.cdf(9, N, g, m)))
        full = float(np.clip((10 - g) / (e + 1), 0, 1))
        if N == m:
            checks["complete_sample_error"] = abs(value - full)
        failed = (
            not np.isfinite(value)
            or not np.isfinite(mass)
            or any(v > 1e-8 for k, v in checks.items() if k.endswith("error"))
            or value < full - 1e-8
            or value > 1.0 + 1e-8
        )
        return {
            "hit10": None if failed else value,
            "checks": checks,
            "null_reason": "Analytic probability checks failed" if failed else None,
        }

    @staticmethod
    def _user_summary(rows, key):
        values = np.asarray(
            [r[key] for r in rows if r.get(key) is not None], dtype=float
        )
        n = len(values)
        result = {
            "n": n,
            "mean": float(values.mean()) if n else None,
            "user_bootstrap_ci95": None,
            "degenerate_interval": None,
            "interval_null_reason": "At least two supported users required",
        }
        if n >= 2:
            rng = np.random.default_rng(42)
            boot = values[rng.integers(n, size=(2000, n))].mean(1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            result.update(
                user_bootstrap_ci95=[float(lo), float(hi)],
                degenerate_interval=bool(lo == hi),
                interval_null_reason=None,
            )
        if key == "H_full":
            binary = bool(n and np.all((values == 0) | (values == 1)))
            result["binary_hits"] = int(values.sum()) if binary else None
            result["descriptive_wilson_ci95"] = None
            result["wilson_null_reason"] = (
                None if binary else "No outcomes or fractional tie expectations"
            )
            if binary:
                p, z = float(values.mean()), 1.959963984540054
                center = (p + z * z / (2 * n)) / (1 + z * z / n)
                half = (
                    z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                )
                result["descriptive_wilson_ci95"] = [
                    float(center - half),
                    float(center + half),
                ]
        return result

    def _catalog_audit(self):
        histories = (
            self.user_train.get("History", {})
            if isinstance(self.user_train, dict)
            else {}
        )
        if not isinstance(histories, dict):
            raise TypeError(
                "ModelView.user_train['History'] must map user IDs to histories"
            )
        eligible, exclusions, counts = [], Counter(), Counter()
        for uid in sorted(histories, key=lambda x: (str(x), type(x).__name__)):
            try:
                h = [int(x) for x in histories[uid]]
            except (TypeError, ValueError):
                exclusions["malformed_history"] += 1
                continue
            if any(x < 1 or x > self.itemnum for x in h):
                exclusions["nonordinary_history_id"] += 1
                continue
            counts.update(h)
            if len(h) >= 2:
                eligible.append((uid, h))
            else:
                exclusions["fewer_than_two_events"] += 1
        rng = np.random.default_rng(42)
        selected = sorted(
            rng.choice(len(eligible), min(16, len(eligible)), replace=False).tolist()
        )
        draws_rng = np.random.default_rng(42)
        quantiles = (
            np.quantile(list(counts.values()), [0.25, 0.75], method="linear")
            if counts
            else None
        )
        costs = {
            "sequence_evaluations": 0,
            "candidate_score_calls": 0,
            "candidate_scores": 0,
        }
        rows, scored_dtypes = [], set()
        fp32 = all(
            p.dtype == torch.float32
            for p in self.network.parameters()
            if p.is_floating_point()
        )
        protocol = {
            "seed": 42,
            "user_cap": 16,
            "reference_chunk_size": 1024,
            "comparison_chunk_size": 257,
            "numerical_users": 4,
            "draws": 5,
            "sample_negatives": 99,
            "bootstrap_replicates": 2000,
            "near_tie_threshold": 1e-5,
            "maxlen": self.maxlen,
            "ordinary_mapping": "Supplied HSTU ordinary IDs 1..itemnum; padding=0; no other reserved IDs",
            "pool": "Ordinary catalog excluding truncated prefix and target; target remains eligible when repeated in context",
            "reference": "One user; ascending competitor IDs; target first in every block; same-vector target comparisons",
            "top10_comparison": "Include target at margin zero; competitors use their own block target; descending operational margin then ascending item ID",
            "ties": "Exact equality; auxiliary uniform tie breaking; top10 comparisons sort descending score then ascending ID",
            "evaluator_pool_and_ties": "Unknown through supplied ModelView; no evaluator equivalence asserted",
            "draw_order": "Selected users in canonical order, five local seed-42 draws per finite reference user",
            "population": "Final public training event, preceding context truncated once and left padded; no replacement of excluded selected users",
            "interval_scope": "Exploratory whole-user intervals, conditional on checkpoint and arithmetic; finite cohort and exclusions limit population interpretation; no training or selection uncertainty",
            "selection_metric": "Unchanged host four-category macro validation HR@10",
        }
        configuration = {
            "device": str(self.device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "parameter_dtypes": sorted(
                {str(p.dtype) for p in self.network.parameters()}
            ),
            "fp32_parameters": fp32,
            "autocast": False,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }

        def score(features, ids):
            costs["candidate_score_calls"] += 1
            costs["candidate_scores"] += len(ids)
            tensor = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            output = self.model.predict_score(features.float(), tensor)
            if output.numel() != len(ids):
                raise ValueError("Expected one HSTU score per candidate")
            scored_dtypes.add(str(output.dtype))
            if output.dtype != torch.float32:
                raise TypeError(
                    f"FP32 reference scorer returned {output.dtype}; "
                    "converting its output would not establish FP32 scoring"
                )
            return output.detach().reshape(-1).double().cpu().numpy()

        def catalog(features, target, pool, width):
            scores, targets = [], []
            for start in range(0, max(1, len(pool)), width - 1):
                block = pool[start : start + width - 1]
                values = score(features, np.r_[target, block])
                targets.append(float(values[0]))
                scores.extend(values[1:].tolist())
            scores = np.asarray(scores, dtype=float)
            block_targets = np.repeat(targets, width - 1)[: len(pool)]
            return scores, np.asarray(targets), block_targets

        def compare(left, right, left_target, right_target, ids, target):
            finite = (
                np.isfinite(left).all()
                and np.isfinite(right).all()
                and np.isfinite(left_target).all()
                and np.isfinite(right_target).all()
            )
            result = {
                "nonfinite_values": int(
                    (~np.isfinite(right)).sum() + (~np.isfinite(right_target)).sum()
                ),
                "null_reason": None if finite else "Nonfinite numerical comparison",
            }
            if not finite:
                return result
            delta = np.abs(left - right)
            lm, rm = left - left_target, right - right_target
            # Include the target in recommendation membership comparisons.
            # Its margin is zero under both blockwise operational paths.
            ranked_ids = np.r_[target, ids]
            ltop = ranked_ids[np.lexsort((ranked_ids, -np.r_[0.0, lm]))[:10]]
            rtop = ranked_ids[np.lexsort((ranked_ids, -np.r_[0.0, rm]))[:10]]
            result.update(
                max_absolute_score_difference=float(delta.max(initial=0)),
                max_relative_score_difference=float(
                    (delta / np.maximum(np.abs(left), 1e-12)).max(initial=0)
                ),
                max_absolute_target_difference=float(
                    np.abs(left_target - right_target).max(initial=0)
                ),
                ordering_or_tie_disagreements=int(
                    np.count_nonzero(np.sign(lm) != np.sign(rm))
                ),
                reference_exact_ties=int(np.count_nonzero(lm == 0)),
                compared_exact_ties=int(np.count_nonzero(rm == 0)),
                minimum_absolute_reference_margin=(
                    float(np.abs(lm).min()) if len(lm) else None
                ),
                near_tie_count=int(np.count_nonzero(np.abs(lm) <= 1e-5)),
                rank_bound_change=[
                    int((rm > 0).sum() - (lm > 0).sum()),
                    int((rm >= 0).sum() - (lm >= 0).sum()),
                ],
                top10_membership_changes=int(len(set(ltop) ^ set(rtop))),
            )
            return result

        valid_numerical_users = 0
        with torch.autocast(device_type=self.device.type, enabled=False):
            for picked in selected:
                uid, history = eligible[picked]
                row = {"user_id": str(uid), "target": history[-1], "excluded": None}
                rows.append(row)
                if not fp32:
                    row["excluded"] = (
                        "FP32 reference unsupported without casting trained parameters"
                    )
                    exclusions[row["excluded"]] += 1
                    continue
                context, target = history[:-1][-self.maxlen :], history[-1]
                if not context:
                    row["excluded"] = "Empty truncated context"
                    exclusions[row["excluded"]] += 1
                    continue
                sequence = np.zeros((1, self.maxlen), dtype=np.int64)
                sequence[0, -len(context) :] = context
                features = self.model.log2feats(sequence)
                costs["sequence_evaluations"] += 1
                if features.ndim != 3 or tuple(features.shape[:2]) != sequence.shape:
                    raise ValueError("HSTU log2feats must return B x L x hidden")
                last = int(np.flatnonzero(sequence[0])[-1])
                if features.dtype != torch.float32:
                    row["excluded"] = (
                        "Feature interface did not provide FP32 reference operands"
                    )
                    exclusions[row["excluded"]] += 1
                    continue
                feature = features[0, last]
                pool = np.setdiff1d(
                    np.arange(1, self.itemnum + 1), list(set(context) | {target})
                )
                scores, targets, block_targets = catalog(feature, target, pool, 1024)
                row.update(
                    context=context,
                    repeat_target=target in context,
                    eligible_competitors=len(pool),
                    # Preserve nonfinite evidence without emitting invalid JSON.
                    block_target_scores=[
                        float(value) if np.isfinite(value) else None
                        for value in targets
                    ],
                    target_training_count=counts[target],
                )
                if not (np.isfinite(scores).all() and np.isfinite(targets).all()):
                    row["excluded"] = "Nonfinite reference catalog scores"
                    row["nonfinite_values"] = int(
                        (~np.isfinite(scores)).sum() + (~np.isfinite(targets)).sum()
                    )
                    exclusions[row["excluded"]] += 1
                    continue
                margin = scores - block_targets
                g, e = int((margin > 0).sum()), int((margin == 0).sum())
                outcome = self._catalog_outcome(g, e)
                analytic = self._analytic_sample(len(pool), g, e)
                varying = bool(np.any(targets != targets[0]))
                row.update(
                    catalog_outcome=outcome,
                    H_full=outcome["hit10"],
                    H_sample=analytic["hit10"],
                    analytic=analytic,
                    D=(
                        analytic["hit10"] - outcome["hit10"]
                        if analytic["hit10"] is not None
                        else None
                    ),
                    operational_blockwise_ranks=varying,
                    reference_top10_semantics="Target-relative blockwise scores, descending then ID",
                    near_tie_count=int((np.abs(margin) <= 1e-5).sum()),
                    minimum_absolute_margin=(
                        float(np.abs(margin).min()) if len(pool) else None
                    ),
                )
                row["popularity_stratum"] = (
                    "low"
                    if counts[target] <= quantiles[0]
                    else "middle" if counts[target] <= quantiles[1] else "high"
                )
                draws = []
                if len(pool) >= 99:
                    for _ in range(5):
                        positions = draws_rng.choice(len(pool), 99, replace=False)
                        ids = pool[positions]
                        candidate = score(feature, np.r_[target, ids])
                        ref = self._catalog_outcome(
                            int((margin[positions] > 0).sum()),
                            int((margin[positions] == 0).sum()),
                        )
                        finite = bool(np.isfinite(candidate).all())
                        actual = (
                            self._catalog_outcome(
                                int((candidate[1:] > candidate[0]).sum()),
                                int((candidate[1:] == candidate[0]).sum()),
                            )
                            if finite
                            else None
                        )
                        check = compare(
                            scores[positions],
                            candidate[1:],
                            block_targets[positions],
                            np.full(99, candidate[0]),
                            ids,
                            target,
                        )
                        draws.append(
                            {
                                "candidate_ids": [int(target)] + ids.tolist(),
                                "reference": ref,
                                "candidate_path": actual,
                                "numerical_check": check,
                            }
                        )
                row["draws"] = draws
                hits = [d["reference"]["hit10"] for d in draws]
                row["draw_summary"] = {
                    "mean": float(np.mean(hits)) if hits else None,
                    "sample_std": (
                        float(np.std(hits, ddof=1)) if len(hits) > 1 else None
                    ),
                    "definite_hits_and_misses": bool(1.0 in hits and 0.0 in hits),
                    "fractional_hit_range": (
                        [float(min(hits)), float(max(hits))] if hits else None
                    ),
                    "null_reason": (
                        None if hits else "Fewer than 99 eligible competitors"
                    ),
                }
                if valid_numerical_users < 4:
                    other, other_targets, other_block_targets = catalog(
                        feature, target, pool, 257
                    )
                    row["chunk_comparison"] = compare(
                        scores, other, block_targets, other_block_targets, pool, target
                    )
                    row["comparison_block_target_scores"] = [
                        float(value) if np.isfinite(value) else None
                        for value in other_targets
                    ]
                    valid_numerical_users += 1
                row["analytic_conditional_on_reference_path"] = bool(
                    varying
                    or any(
                        d["numerical_check"].get("ordering_or_tie_disagreements", 0)
                        or d["numerical_check"]["null_reason"]
                        for d in draws
                    )
                    or row.get("chunk_comparison", {}).get(
                        "ordering_or_tie_disagreements", 0
                    )
                    or row.get("chunk_comparison", {}).get("null_reason")
                )
        retained = [r for r in rows if r["excluded"] is None]
        configuration["scoring_output_dtypes"] = sorted(scored_dtypes)
        result = {
            "category": self.category,
            "category_null_reason": (
                None if self.category else "Authorized route unavailable or conflicting"
            ),
            "eligible_users": len(eligible),
            "selected_users": len(selected),
            "selected_ids": [str(eligible[i][0]) for i in selected],
            "sampling_fraction": len(selected) / len(eligible) if eligible else None,
            "retained_users": len(retained),
            "exclusions": dict(exclusions),
            "users": rows,
            "cost_counts": costs,
            "protocol": protocol,
            "configuration": configuration,
            "training_item_count_q25_q75": (
                quantiles.tolist() if quantiles is not None else None
            ),
            "summary": {
                key: self._user_summary(retained, key)
                for key in ("H_full", "H_sample", "D")
            },
            "popularity_strata": {
                name: {
                    key: self._user_summary(
                        [r for r in retained if r["popularity_stratum"] == name], key
                    )
                    for key in ("H_full", "H_sample", "D")
                }
                for name in ("low", "middle", "high")
            },
            "null_reason": None if retained else "No finite supported selected users",
        }
        return {
            "per_category": {
                name: result if name == self.category else None
                for name in self.datasets
            },
            "unidentified_category_result": result if self.category is None else None,
            "macro_four_category_mean": None,
            "macro_four_category_ci95": None,
            "macro_null_reason": "One category invocation; host must independently resample users within all four identified supported categories and average means equally",
        }

    def run_full_diagnosis(self, train_loader):
        import time
        from itertools import islice

        print(">>> Running frozen HSTU training-prefix diagnosis...")
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        generators = []
        for owner in (train_loader, getattr(train_loader, "sampler", None)):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator) and all(
                generator is not old for old, _ in generators
            ):
                generators.append((generator, generator.get_state()))
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        rng_devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        numerical_settings = (
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.random.fork_rng(
                devices=rng_devices
            ), torch.inference_mode(), torch.autocast(
                device_type=self.device.type, enabled=False
            ):
                self.network.eval()
                if devices:
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(torch.cuda.current_stream(self.device))
                start = time.perf_counter()
                batches = list(islice(iter(train_loader), 10))
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=10, eps=eps, topk_categories=20
                    )
                )
                audit = self._catalog_audit()
                gpu_seconds, peak = None, None
                if devices:
                    end_event.record(torch.cuda.current_stream(self.device))
                    torch.cuda.synchronize(self.device)
                    gpu_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
                    peak = int(torch.cuda.max_memory_allocated(self.device))
                wall_seconds = time.perf_counter() - start
                breakdown["null_reason"] = (
                    "Item-category attributes are unavailable; raw counts would not adjust for exposure"
                )
                result = {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "embedding_null_reason": (
                            None
                            if collapse_score is not None
                            else "Fewer than two items, zero-norm embeddings, or nonfinite embeddings"
                        ),
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_null_reason": (
                            None
                            if margin_metrics["mean"] is not None
                            else "No finite valid sampled training pairs"
                        ),
                        "pos_neg_margin_category_breakdown": breakdown,
                        "exact_catalog_training_audit": audit,
                        "diagnostic_cost": {
                            "wall_seconds": float(wall_seconds),
                            "cuda_event_seconds": gpu_seconds,
                            "peak_allocated_bytes": peak,
                            "gpu_null_reason": None if devices else "CPU execution",
                        },
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "(Mean off-diagonal cosine + 1)/2 for up to 2048 effective ordinary-item embeddings sampled with local seed 42. Approximately 0.5 means mean cosine zero. This measures angular alignment, not recommendation diversity or representation rank. Null for undefined inputs.",
                        "pos_neg_margin_mean": "Mean positive-minus-loader-negative score over finite, nonpadding, ordinary-item training positions in at most ten batches. Supplied HSTU features are causal: position t sees sequence[:t+1], not later events.",
                        "pos_neg_margin_p10": "Nearest-order-statistic 10th percentile of sampled pair margins.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of valid sampled pairs with negative score strictly greater than positive score.",
                        "pos_neg_small_margin_rate": "Fraction of valid sampled pairs with positive-minus-negative score below eps=0.5; depends on score scale.",
                        "pos_neg_margin_category_breakdown": "Empty category lists because item categories are unavailable. Position counts remain descriptive; no exposure-adjusted conclusion is supported.",
                        "exact_catalog_training_audit": "One frozen supplied HSTU, at most sixteen uniformly selected canonical eligible training users, final training event as target, preceding context truncated once then left padded. Readout uses greatest nonpadding index. Ordinary IDs are supplied HSTU IDs 1..itemnum, excluding prefix-seen IDs and target from competitors. No replacement of selected scoring exclusions. These are in-sample diagnostic pools, not established equivalents of pinned validation pools. Production inference, model fitting and simulator cohort are unchanged.",
                        "catalog_outcome": "For N eligible competitors, g strictly exceed their same-block target score and e exactly equal it. Optimistic rank=1+g; pessimistic rank=1+g+e; H_full=clip((10-g)/(e+1),0,1). Expected reciprocal rank averages 1/r across every tied rank. Uniform tie breaking is an auxiliary convention. Varying block target scores imply blockwise operational ranks rather than proven invariant scores.",
                        "analytic_sample": "H_sample sums multivariate hypergeometric probabilities C(g,a)C(e,b)C(N-g-e,99-a-b)/C(N,99), weighted by clip((10-a)/(b+1),0,1). Feasible mass must agree with one within 1e-8; e=0 is checked against hypergeometric CDF and N=99 against H_full. D=H_sample-H_full should be nonnegative within numerical tolerance. Failed checks yield null sampled expectations. Conditional on reference ordering and uniform tie breaking, D isolates candidate-sampling discrepancy for this one model; it is not a model-comparison reversal or validation estimate.",
                        "numerical_checks": "Reference uses disabled autocast, unchanged FP32 trained parameters, highest matmul precision and disabled TF32; unsupported parameter dtypes yield explicit exclusions. Every catalog block includes target first; comparisons use that same block's target. First four finite reference users compare all pairs under widths 257 and 1024. All five actual sampled candidate vectors are rescored. Absolute/relative differences, nonfinite values, exact ties, target-relative ordering and rank-bound changes are measured. Relative denominator=max(abs(reference score),1e-12). Near-tie threshold 1e-5 never changes equality or ranks. Descriptive top10 compares target-relative blockwise scores with ascending ID ties; it includes target-relative score zero for the target only when explicitly supplied. Numerical disagreements make analytic results conditional on the reference path, not predictions of host arithmetic.",
                        "draws": "Five local seed-42 samples of 99 unique uniform competitors without replacement per supported user. IDs and reference/candidate-path tie-aware outcomes are retained. Means and sample standard deviations describe five candidate realizations, not independent users; fractional hit range is reported separately from definite hits/misses. Fewer than 99 competitors yields null sampled comparisons.",
                        "uncertainty_and_popularity": "Means weight supported users equally; D is paired within user. Whole-user percentile bootstrap uses 2000 seed-42 draws when n>=2; intervals are exploratory at n<=16 and degeneracy is reported. Wilson intervals apply only when every H_full is binary, reporting hits/n; finite cohort and approximate binomial assumptions limit interpretation. No interval includes training or adaptive selection uncertainty. Popularity counts use only valid public training histories; q25/q75 use NumPy linear quantiles over training-observed items, with low<=q25, middle<=q75, high>q75. Empty support is null. Four-category macro diagnostics require host aggregation of four identified supported category user records; never pool local user IDs or average interval endpoints.",
                        "diagnostic_cost": "Wall time includes loader access and all probes. CUDA-event elapsed time measures the diagnostic stream interval, including host-induced idle gaps, not kernel-active GPU time. Peak allocated memory is the absolute PyTorch allocation high-water mark after resetting it at audit entry. History cost_counts record actual sequence evaluations, candidate-score calls and scalar candidate scores. CPU GPU statistics are null. No claim of inexpensive auditing is made.",
                    },
                }
                import os
                import tempfile

                artifact_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
                )
                os.makedirs(artifact_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=artifact_dir,
                    prefix=(self.category or "unidentified") + "_",
                    suffix=".json",
                    delete=False,
                ) as stream:
                    result["metrics"]["artifact_path"] = stream.name
                    json.dump(result, stream, indent=2, allow_nan=False)
                return result
        finally:
            torch.set_float32_matmul_precision(numerical_settings[0])
            torch.backends.cuda.matmul.allow_tf32 = numerical_settings[1]
            torch.backends.cudnn.allow_tf32 = numerical_settings[2]
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in generators:
                generator.set_state(state)
            for module, training in modes:
                module.training = training

    # Static review: traced analytic boundary cases, scorer dtype checks,
    # target-inclusive top10 comparisons and both comparison call sites.
    # Nonfinite block targets retain explicit nulls/counts for JSON artifacts.
    # Training, checkpoints, predictor, routing and selection remain unchanged.
    # No execution, benchmark gains or runtime measurements are claimed.
    ### <<< Self_EvolveRec-BLOCK-END
