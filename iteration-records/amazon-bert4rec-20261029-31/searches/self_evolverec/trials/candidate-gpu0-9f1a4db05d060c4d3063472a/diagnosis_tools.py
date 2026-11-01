import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption without fetching an extra batch
from collections import Counter
from itertools import islice

### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Ground interpretation in diagnostic provenance and availability
def diagnosis_interpreter_prompt(raw_diagnosis):
    return _diagnosis_interpreter_prompt(raw_diagnosis) + """
    Additional interpretation requirements:
    Distinguish supervised training separation, descriptive in-sample prefix
    dependence, and held-out predictive usefulness. Prefix diagnostics are never
    host validation performance. Events in these prefixes may have contributed
    to fitting. Interventions are not causal user counterfactuals.
    Cite measured values, counts, availability and definitions for every finding.
    Missing measurements and provenance are unknown, not evidence of health.
    Embedding cosine similarity does not establish recommendation diversity.
    Input-target item equality does not establish leakage: repeated events and
    legitimate corruption policies can share item IDs.
    Interpret bootstrap intervals and strata as exploratory. Account for no-ops,
    effective edits, small samples and unrealistic order-corruption interventions.
    Timestamp inversions describe chronological disorder, not proven violations
    of item release availability, which is unobserved.
    Do not invent ratings, reviews, categories, ownership or simulator traits.
    Validation usefulness requires explicitly authorized pinned diagnostic inputs;
    do not infer it from this loader or request hidden files.
    Selection remains host macro validation HR@10; acknowledge validation adaptation
    and the shared compute budget. Do not recommend independent predictor fusion.
    If required evidence is unavailable, use NEEDS_IMPROVEMENT and explain the
    measurement limitation without claiming the fitted model itself has failed.
    These requirements take precedence over any default suggestion of STABLE.
    """


def _diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and use the actual BERT4Rec module
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are not available in this matched task.
        return []

    def _features(self, sequence):
        # The supplied BERT4Rec path accepts tensors; retain the view fallback.
        method = getattr(self.network, "log2feats", None)
        if callable(method):
            return method(sequence)
        return self.model.log2feats(sequence.detach().cpu().numpy())

    def _scores(self, features, items):
        method = getattr(self.network, "predict_score", None)
        if callable(method):
            return method(features, items)
        return self.model.predict_score(features, items)

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable pairwise similarity instead of fabricated zero collapse
            n = emb.size(0)
            if min(int(sample_size), n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Audit selected masked supervision without inferring event provenance
        margins_all = []
        audit = {
            "selected_positions": 0,
            "selected_with_reserved_mask": 0,
            "selected_without_reserved_mask": 0,
            "invalid_target_or_negative": 0,
            "original_labels_and_corruption_provenance": "unavailable",
            "leakage_conclusion": "unknown; item-ID equality is not a leakage test",
        }
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume exactly the requested number of diagnostic batches
        for batch in islice(dataloader, max(0, int(num_batches))):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score aligned BERT4Rec masked targets through compatible interfaces
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or seq_t.shape != pos_t.shape
                or seq_t.shape != neg_t.shape
            ):
                raise ValueError(
                    "diagnostic batches must contain aligned B x L tensors"
                )
            itemnum = int(self.network.itemnum)
            selected = pos_t != 0
            real_labels = (pos_t >= 1) & (pos_t <= itemnum)
            real_negatives = (neg_t >= 1) & (neg_t <= itemnum)
            masked = seq_t == int(self.network.mask_token)
            audit["selected_positions"] += int(selected.sum().item())
            audit["selected_with_reserved_mask"] += int(
                (selected & masked).sum().item()
            )
            audit["selected_without_reserved_mask"] += int(
                (selected & ~masked).sum().item()
            )
            audit["invalid_target_or_negative"] += int(
                (selected & ~(real_labels & real_negatives)).sum().item()
            )
            valid = selected & masked & real_labels & real_negatives
            log_feats = self._features(seq_t)
            # Separate aligned calls avoid B x L x 2 broadcasting ambiguity.
            safe_pos = pos_t.masked_fill(~valid, 0)
            safe_neg = neg_t.masked_fill(~valid, 0)
            margin = self._scores(log_feats, safe_pos) - self._scores(
                log_feats, safe_neg
            )
            if not torch.isfinite(margin[valid]).all():
                raise RuntimeError("nonfinite supervised diagnostic scores")
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Restrict breakdowns to audited masked positions
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
                ### >>> Self_EvolveRec-BLOCK-START: Represent missing supervised measurements explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Report supervision and unavailable-category audits
        breakdown = {
            "supervision_audit": audit,
            "category_metadata_available": False,
            "measured_positions": sum(int(m.numel()) for m in margins_all),
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

    ### >>> Self_EvolveRec-BLOCK-START: Mask-safe recent-history interventions on one fitted category checkpoint
    def _encode_events(self, events, length):
        mask = int(self.network.mask_token)
        if len(events) > length - 1:
            raise ValueError("freeze and truncate the prefix before intervention")
        sequence = np.zeros(length, dtype=np.int64)
        if events:
            sequence[length - len(events) - 1 : length - 1] = [
                item for _, item in events
            ]
        sequence[-1] = mask
        assert np.count_nonzero(sequence == mask) == 1
        assert np.count_nonzero(sequence) == len(events) + 1
        return sequence

    @staticmethod
    def _score_summary(scores, candidates):
        target = scores[0]
        # One-based average rank; deterministic item-ID tie breaking only for top10 sets.
        rank = 1.0 + float(np.sum(scores[1:] > target))
        rank += 0.5 * float(np.sum(scores[1:] == target))
        order = np.lexsort((candidates, -scores))
        boundary = float(np.sort(scores[1:])[-10])
        margin = float(target - boundary)
        scale = float(np.std(scores))
        return {
            "rank": rank,
            "hr10": float(rank <= 10.0),
            "raw_margin": margin,
            "standardized_margin": margin / scale if scale > 1e-12 else None,
            "zero_variance": scale <= 1e-12,
            "top10": set(int(i) for i in candidates[order[:10]]),
        }

    @staticmethod
    def _time_values(values):
        if values is None:
            return None
        try:
            result = np.asarray([float(v) for v in values], dtype=np.float64)
        except (TypeError, ValueError):
            try:
                dates = np.asarray(values, dtype="datetime64[ms]")
                if np.isnat(dates).any():
                    return None
                result = dates.astype(np.int64).astype(np.float64)
            except (TypeError, ValueError, OverflowError):
                return None
        return result if np.isfinite(result).all() else None

    def _prefix_interventions(self):
        histories = self.user_train.get("History", {})
        timestamps = self.user_train.get("Time", {})
        if not isinstance(histories, dict):
            raise ValueError("ModelView.user_train must contain a History mapping")
        if not isinstance(timestamps, dict):
            timestamps = {}
        itemnum = int(self.network.itemnum)
        length = int(self.network.maxlen)
        if length < 3:
            return {
                "available": False,
                "reason": "requires at least two visible events and MASK",
            }
        category = getattr(self.model, "dataset_name", None)
        if category is None:
            category = getattr(self.network, "dataset_name", None)
        category = (
            str(category) if category is not None else "current_category_unlabeled"
        )
        selection_rng = random.Random(42)
        candidate_rng = np.random.default_rng(42)
        edit_rng = np.random.default_rng(42)
        bootstrap_rng = np.random.default_rng(42)
        users = sorted(histories, key=lambda u: (str(type(u)), str(u)))
        sampled_users = selection_rng.sample(users, min(200, len(users)))
        counts = {
            "training_users": len(users),
            "sampled_users": len(sampled_users),
            "short_history": 0,
            "invalid_item_ids": 0,
            "insufficient_negatives": 0,
            "evaluated_users": 0,
            "identity_checks": 0,
            "identity_failures": 0,
        }
        records = []
        baseline_records = []
        catalog = np.arange(1, itemnum + 1, dtype=np.int64)
        for user in sampled_users:
            try:
                history = [int(item) for item in histories[user]]
            except (TypeError, ValueError, OverflowError):
                counts["invalid_item_ids"] += 1
                continue
            if len(history) < 3:
                counts["short_history"] += 1
                continue
            if any(item < 1 or item > itemnum for item in history):
                counts["invalid_item_ids"] += 1
                continue
            # Last training event is the target EVENT; matching earlier item IDs are valid.
            target_index = len(history) - 1
            start = max(0, target_index - (length - 1))
            visible = list(enumerate(history[start:target_index], start=start))
            if len(visible) < 2:
                counts["short_history"] += 1
                continue
            assert all(index < target_index for index, _ in visible)
            target = history[target_index]
            excluded = np.asarray(
                sorted({target} | {item for _, item in visible}), dtype=np.int64
            )
            # Catalog is already ordered and unique; avoid repeated sorting.
            eligible_mask = np.ones(itemnum, dtype=bool)
            eligible_mask[excluded - 1] = False
            eligible = catalog[eligible_mask]
            if len(eligible) < 99:
                counts["insufficient_negatives"] += 1
                continue
            candidates = np.concatenate(
                (
                    np.asarray([target]),
                    candidate_rng.choice(eligible, 99, replace=False),
                )
            )
            assert len(np.unique(candidates)) == 100
            assert not (set(candidates[1:]) & {item for _, item in visible})
            baseline_tensor = self._encode_events(visible, length)
            n = len(visible)
            k = min(3, n // 2)
            recent_indices = set(range(n - k, n))
            oldest_indices = set(range(k))
            branches = [("baseline", visible), ("identity", list(visible))]
            branches.append(
                (
                    "delete_recent_compact",
                    [e for i, e in enumerate(visible) if i not in recent_indices],
                )
            )
            branches.append(
                (
                    "delete_oldest_compact",
                    [e for i, e in enumerate(visible) if i not in oldest_indices],
                )
            )
            for _ in range(10):
                removed = set(int(i) for i in edit_rng.choice(n, k, replace=False))
                branches.append(
                    (
                        "delete_random_compact",
                        [e for i, e in enumerate(visible) if i not in removed],
                    )
                )
            window = min(5, n)
            for _ in range(10):
                order = edit_rng.permutation(window)
                edited = visible[:-window] + [
                    visible[n - window + int(i)] for i in order
                ]
                branches.append(("permute_recent_order_stress", edited))
            arrays = np.stack(
                [self._encode_events(events, length) for _, events in branches]
            )
            for _, events in branches:
                assert all(
                    index < target_index and history[index] == item
                    for index, item in events
                )
            sequence = torch.as_tensor(arrays, dtype=torch.long, device=self.device)
            features = self._features(sequence)
            positions = torch.arange(length, device=self.device).expand_as(sequence)
            last = positions.masked_fill(sequence == 0, -1).amax(dim=1)
            assert bool(
                torch.all(sequence.gather(1, last[:, None]) == self.network.mask_token)
            )
            if features.ndim != 3 or tuple(features.shape[:2]) != tuple(sequence.shape):
                raise RuntimeError("log2feats must return aligned B x L x H features")
            readout = features[torch.arange(len(branches), device=self.device), last]
            items = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
            logits = (
                self._scores(readout[:, None, :], items).detach().float().cpu().numpy()
            )
            if logits.shape != (len(branches), 100) or not np.isfinite(logits).all():
                raise RuntimeError(
                    "incompatible or nonfinite BERT4Rec intervention scores"
                )
            counts["identity_checks"] += 1
            if not np.allclose(logits[0], logits[1], rtol=1e-5, atol=1e-6):
                counts["identity_failures"] += 1
                raise RuntimeError("identity intervention failed score reproducibility")
            base = self._score_summary(logits[0], candidates)
            baseline_records.append(base)
            raw_times = timestamps.get(user, timestamps.get(str(user)))
            if raw_times is not None and len(raw_times) == len(history):
                times = self._time_values(raw_times)
            else:
                times = None
            user_record = {}
            for branch_index, (name, events) in enumerate(branches[1:], start=1):
                summary = self._score_summary(logits[branch_index], candidates)
                effective = not np.array_equal(arrays[branch_index], baseline_tensor)
                if name.startswith("delete_"):
                    edit_distance = n - len(events)
                else:
                    edit_distance = sum(a[0] != b[0] for a, b in zip(visible, events))
                inversion_count = None
                if name == "permute_recent_order_stress" and times is not None:
                    # Measured chronological availability violations among surviving events.
                    ordered_times = [times[index] for index, _ in events]
                    inversion_count = sum(
                        int(ordered_times[i] > ordered_times[j])
                        for i in range(len(events))
                        for j in range(i + 1, len(events))
                    )
                delta = {
                    "hr10": summary["hr10"] - base["hr10"],
                    "rank": summary["rank"] - base["rank"],
                    "raw_margin": summary["raw_margin"] - base["raw_margin"],
                    "standardized_margin": (
                        summary["standardized_margin"] - base["standardized_margin"]
                        if summary["standardized_margin"] is not None
                        and base["standardized_margin"] is not None
                        else None
                    ),
                }
                user_record.setdefault(name, []).append(
                    {
                        "delta": delta,
                        "effective": effective,
                        "zero_variance": summary["zero_variance"],
                        "overlap": len(base["top10"] & summary["top10"]) / 10.0,
                        "edit_distance": edit_distance,
                        "timestamp_inversions": inversion_count,
                        "baseline_correct": bool(base["hr10"]),
                        "history_length": n,
                    }
                )
            records.append(user_record)
        counts["evaluated_users"] = len(records)
        result = {
            "available": bool(records),
            "source": "training_prefix_in_sample_dependence",
            "category": category,
            "checkpoint_count": 1,
            "seed": 42,
            "counts": counts,
            "branches": {},
            "validation_usefulness": {
                "available": False,
                "reason": "No explicitly authorized pinned validation diagnostic protocol is supplied by ModelView/train_loader.",
            },
            "position_preserving_deletion": {
                "available": False,
                "reason": "Supplied BERT4Rec/ModelView has no explicit position_ids-capable diagnostic path; compact deletion only.",
            },
            "cross_category_macro": {
                "available": False,
                "reason": "This probe observes one category; four-category macro effects require all four category records.",
            },
            "bootstrap_replicates": 1000,
        }
        if not records:
            result["baseline"] = None
            return result
        result["baseline"] = {}
        for key in ("rank", "hr10", "raw_margin", "standardized_margin"):
            values = [row[key] for row in baseline_records if row[key] is not None]
            result["baseline"][key] = {
                "mean": float(np.mean(values)) if values else None,
                "users": len(values),
            }
        result["baseline"]["zero_variance_users"] = sum(
            int(row["zero_variance"]) for row in baseline_records
        )
        # One shared set of complete-user resamples preserves dependence across all draws/branches.
        resamples = bootstrap_rng.integers(0, len(records), size=(1000, len(records)))
        for name in records[0]:
            user_draws = [record[name] for record in records]
            draws = [draw for rows in user_draws for draw in rows]
            report = {
                "users": len(user_draws),
                "draws": len(draws),
                "no_op_draws": sum(int(not row["effective"]) for row in draws),
                "zero_variance_draws": sum(int(row["zero_variance"]) for row in draws),
                "top10_overlap": float(np.mean([row["overlap"] for row in draws])),
                "mean_event_edit_distance": float(
                    np.mean([row["edit_distance"] for row in draws])
                ),
                "paired_effects": {},
                "strata": {},
            }
            measured_times = [
                row["timestamp_inversions"]
                for row in draws
                if row["timestamp_inversions"] is not None
            ]
            report["temporal_availability"] = {
                "available": bool(measured_times),
                "draws": len(measured_times),
                "draws_with_inversions": sum(int(v > 0) for v in measured_times),
                "mean_inversions": (
                    float(np.mean(measured_times)) if measured_times else None
                ),
            }
            for key in ("hr10", "rank", "raw_margin", "standardized_margin"):
                means, effective_means = [], []
                for rows in user_draws:
                    valid = [
                        row["delta"][key]
                        for row in rows
                        if row["delta"][key] is not None
                    ]
                    effective = [
                        row["delta"][key]
                        for row in rows
                        if row["effective"] and row["delta"][key] is not None
                    ]
                    means.append(float(np.mean(valid)) if valid else np.nan)
                    if effective:
                        effective_means.append(float(np.mean(effective)))
                values = np.asarray(means, dtype=np.float64)
                sampled = values[resamples]
                finite_counts = np.isfinite(sampled).sum(axis=1)
                boot = np.divide(
                    np.nansum(sampled, axis=1),
                    finite_counts,
                    out=np.full(1000, np.nan),
                    where=finite_counts > 0,
                )
                boot = boot[np.isfinite(boot)]
                finite = values[np.isfinite(values)]
                report["paired_effects"][key] = {
                    "mean_delta": float(finite.mean()) if len(finite) else None,
                    "users": len(finite),
                    "exploratory_paired_user_95pct_interval": (
                        np.percentile(boot, [2.5, 97.5]).tolist() if len(boot) else None
                    ),
                    "valid_bootstrap_replicates": len(boot),
                    "effective_edit_mean_delta": (
                        float(np.mean(effective_means)) if effective_means else None
                    ),
                    "effective_edit_users": len(effective_means),
                }
            selectors = {
                "baseline_correct": lambda row: (
                    "correct" if row["baseline_correct"] else "incorrect"
                ),
                "history_length": lambda row: (
                    "2-5"
                    if row["history_length"] <= 5
                    else ("6-20" if row["history_length"] <= 20 else "21+")
                ),
                "event_edit_distance": lambda row: str(row["edit_distance"]),
            }
            for stratum, selector in selectors.items():
                groups = {}
                for rows in user_draws:
                    grouped_user = {}
                    for row in rows:
                        grouped_user.setdefault(selector(row), []).append(
                            row["delta"]["hr10"]
                        )
                    for label, values in grouped_user.items():
                        group = groups.setdefault(
                            label, {"users": 0, "draws": 0, "user_mean_deltas": []}
                        )
                        group["users"] += 1
                        group["draws"] += len(values)
                        group["user_mean_deltas"].append(float(np.mean(values)))
                for group in groups.values():
                    group["mean_delta_hr10"] = float(
                        np.mean(group.pop("user_mean_deltas"))
                    )
                report["strata"][stratum] = groups
            result["branches"][name] = report
        return result

    def run_full_diagnosis(self, train_loader):
        previous_modes = [
            (module, module.training) for module in self.network.modules()
        ]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        # Preserve explicit loader/sampler generators as well as global RNGs.
        generators = []
        seen_generators = set()
        for owner in (
            train_loader,
            getattr(train_loader, "sampler", None),
            getattr(getattr(train_loader, "batch_sampler", None), "sampler", None),
        ):
            generator = getattr(owner, "generator", None)
            if (
                isinstance(generator, torch.Generator)
                and id(generator) not in seen_generators
            ):
                seen_generators.add(id(generator))
                generators.append((generator, generator.get_state()))
        # Restrict CUDA state capture and seeding to the actual model device.
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                for device_index in cuda_devices:
                    torch.cuda.default_generators[device_index].manual_seed(42)
                for generator, _ in generators:
                    generator.manual_seed(42)
                self.network.eval()
                with torch.no_grad():
                    collapse_score = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048
                    )
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            train_loader, num_batches=10, eps=0.5, topk_categories=20
                        )
                    )
                    prefix = self._prefix_interventions()
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "recent_history_probe": prefix,
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "Mean sampled real-item pair cosine mapped to [0,1]; excludes PAD/MASK. Measures angular similarity, not recommendation diversity. Null when fewer than two embeddings can be sampled.",
                        "pos_neg_margin_mean": "Mean positive-minus-negative logit on real-label supervised positions containing reserved MASK; training separation, not validation performance. Null if no valid positions.",
                        "pos_neg_margin_p10": "Nearest-rank 10th percentile of audited supervised training margins; null if unavailable.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of audited masked positions with negative logit greater than positive logit.",
                        "pos_neg_small_margin_rate": "Fraction of audited masked positions with margin below 0.5.",
                        "pos_neg_margin_category_breakdown": "Counts and mask audit for up to ten training batches. Original-label/corruption provenance is unavailable; equality alone cannot establish leakage. Category lists remain empty because categories are unavailable.",
                        "recent_history_probe": {
                            "source": "At most 200 deterministically sampled training users in this category; final training event is target, only earlier events enter the frozen last L-1 prefix. Repeated item IDs are permitted. Descriptive fitted-model dependence only.",
                            "candidates": "One fixed target plus 99 unique uniform real-item negatives excluding target and frozen visible prefix IDs; eligibility never changes after edits. Insufficient-catalog examples are skipped and counted. This is not the pinned evaluator.",
                            "inference": "One actual fitted BERT4Rec module, eval/no_grad, reserved appended MASK, left padding, greatest nonpadding readout. Identity scores must agree at rtol=1e-5, atol=1e-6. No additional model fitting or score fusion.",
                            "interventions": "k=min(3,floor(visible_length/2)). Delete recent/oldest k or ten independently sampled uniform k-subsets without replacement within each draw; compact and left-pad. Ten permutations of the last min(5,length) events are order-corruption stress tests. No-ops remain in primary estimates. Position-preserving deletion is unavailable without an explicit positional path.",
                            "rank_hr10": "One-based average target rank under exact score ties; descriptive HR10 is 1 if rank<=10. Top10 sets use score then item-ID tie breaking; overlap is intersection size/10. No assertion of agreement with host tie handling.",
                            "margins": "Raw margin is target logit minus tenth-largest negative logit. Standardized margin divides by population standard deviation of all 100 candidate logits. Standardized measurements are null when scale<=1e-12; zero-variance counts are reported.",
                            "paired_effects": "Edited minus baseline metrics, averaged across assigned draws within user, then equally across users. Positive HR/margin delta indicates improvement; positive rank delta indicates degradation. Missing standardized observations are omitted with counts. Effective sensitivity uses only tensor-changing draws, then equal user weights.",
                            "uncertainty": "1000 seed-42 complete paired-user bootstrap resamples, keeping all interventions/draws together. Percentile 95% intervals are exploratory; valid replicate counts are reported. No four-category macro is inferred from one category.",
                            "strata": "Descriptive baseline-correct/incorrect, frozen visible history-length bins and event edit-distance strata; counts include users and draws. Permutation distance counts displaced event identities, while effective edits compare item tensors. Strata are not independent tests.",
                            "temporal_availability": "For order-corruption draws with parsable aligned timestamps, report pairwise chronological inversions in the resulting entire visible history. May include original inversions. These measure chronological disorder; item release availability and causal effects are unobserved. Otherwise unavailable.",
                            "availability_budget": "Validation intervention usefulness and four-category macro aggregation are unavailable through this interface. No hidden-file access. At most ten training batches and 200 prefix users/category; all execution consumes the shared search budget. Global RNGs, discovered loader/sampler generators and module modes are restored. Worker-local RNGs and iterator progress cannot be restored: pass a disposable diagnostic loader, without persistent workers, rather than an active training iterator.",
                        },
                    },
                }
        finally:
            for module, was_training in previous_modes:
                module.training = was_training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in generators:
                generator.set_state(state)

    ### <<< Self_EvolveRec-BLOCK-END
