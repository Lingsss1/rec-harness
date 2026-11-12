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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measured training sensitivity conservatively
    - Every statement must be grounded in the provided metric values/definitions.
    - History interventions measure sensitivity of a frozen fitted model on training prefixes, not validation performance, user intent, shortcut learning, or generalization.
    - Null statistics are unavailable, not zero. Do not infer elapsed-time effects: this supplied HSTU consumes item IDs and learned absolute positions only.
    - Confidence intervals condition on the fitted checkpoint and fixed diagnostic protocol. They do not include training uncertainty.
    - Do not infer diversity, conformity, exposure-adjusted category effects, or unavailable review/metadata attributes.
    # Review checks: traced HSTU feature/scoring shapes, validated IDs before
    # lookups, checked final-nonpadding readout and matched candidates, removed
    # unnecessary category loops, and checked undefined interval reporting.
    # Four-category macro statistics remain explicitly unavailable in this
    # single-category invocation. No training, checkpoint, or predictor changes.
    # Static review only; execution and benchmark performance were not measured.
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
        self.category = next(
            (name for name in names if isinstance(name, str) and name in self.datasets),
            None,
        )
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

    ### >>> Self_EvolveRec-BLOCK-START: Matched frozen-HSTU history interventions with clustered rank diagnostics
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
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
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
                audit = self._history_audit(batches)
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
                return {
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
                        "history_interventions": audit,
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
                        "history_interventions": "Frozen-model training-prefix sampled diagnostics, distinct from the host validation benchmark. Up to 32 eligible users/category and two distinct training targets/user, with >=4 context events truncated to model.maxlen. Fallback uses sequence[:t+1] and positive[t], never later events, and reports rows without user confidence intervals. Earlier occurrences of the target remain valid context. One target plus up to 999 unique negatives excludes only prefix-seen items and target, not later events. All variants share candidates. Rank is one plus scores strictly above target plus tied candidates with smaller item ID; target tie frequency counts any equal-scoring other candidate. Greatest absolute nonpadding index supplies readout. Recent/oldest removal delete k=min(3,floor((n-2)/2)) items; final removal deletes one; recent-preserving-final deletes k immediately preceding the final item. Three nonfinal shuffles are averaged within prefix. Rank, reciprocal rank and hit10 effects are perturbed-minus-original: positive rank is worse; positive reciprocal rank/hit10 is better. Recent-minus-oldest compares equal-count deletions; final-minus-prefinal is available only when k=1 for every included prefix. Repeats average within prefixes, prefixes within user (or unidentified row), then users within category. Bootstrap resamples complete category-local users 2000 times with seed 42, retaining all paired outcomes. Intervals condition on fitted weights and fixed candidates. Fallback empirical quantiles describe row effects, not user uncertainty. Candidate counts <=10 make hit10 trivially one. A four-category macro summary is null until all four category user-cluster records are available; do not average category interval endpoints or assume matching local IDs denote the same person. Sensitivity does not identify intent, shortcuts, elapsed-time effects, or generalization; invariance does not establish that history is ignored.",
                        "diagnostic_cost": "Wall time includes loader access and all probes. CUDA-event elapsed time measures the diagnostic stream interval, including host-induced idle gaps, not kernel-active GPU time. Peak allocated memory is the absolute PyTorch allocation high-water mark after resetting it at audit entry. History cost_counts record actual sequence evaluations, candidate-score calls and scalar candidate scores. CPU GPU statistics are null. No claim of inexpensive auditing is made.",
                    },
                }
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in generators:
                generator.set_state(state)
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END
