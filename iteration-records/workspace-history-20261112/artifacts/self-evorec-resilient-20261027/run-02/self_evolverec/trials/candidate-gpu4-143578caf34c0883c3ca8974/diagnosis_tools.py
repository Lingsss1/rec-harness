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

    ### >>> Self_EvolveRec-BLOCK-START: Evidence-limited interpretation of paired temporal diagnostics
    Rules:
    - Keep core_findings and key_implications to at most three entries each.
    - Ground every finding in measured values, definitions, valid counts, and missing controls.
    - Report unavailable measurements as unavailable; missing evidence is not evidence of stability.
    - Distinguish score sensitivity from ranking quality and from unobserved preference intent.
    - These are fitted-model, in-sample training-prefix interventions, not validation or test results.
    - Repacking changes positions and context. Last-item deletion changes the final query.
    - Order shuffles preserve the final item, but timestamp plausibility can remain unverified.
    - Without the explicit mask/position hook, position-preserving removal is unavailable.
    - Even that hook changes attention context and does not establish preference causality.
    - Bootstrap intervals condition on the fitted models and fixed candidates. They exclude
      training variability, candidate variability, and adaptive validation selection bias.
    - Do not infer compatibility, ownership, fulfilled demand, dislike, review sentiment,
      simulator diversity/conformity traits, or generalization from these measurements.
    - Model-order agreement is unavailable with only one fitted model per route.
    - Auxiliary sampling with seed 42 does not reproduce the host evaluator.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Separate the compatibility view from the actual trained module
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Respect unavailable item-category metadata
    def _get_categories(self, item_id: int):
        # Dataset routes are available; item taxonomy is not available in this task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Use the compatibility embedding hook
            emb = self.view.item_emb.weight.detach()
            ### <<< Self_EvolveRec-BLOCK-END
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Undefined alignment has a null value
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute embedding alignment without a quadratic Gram matrix
            if k < 2:
                return None
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite diagnostic item embeddings")
            # Sum all pairwise dot products, then remove the diagonal.
            # This also handles zero-norm embeddings without assuming unit diagonals.
            x = x.to(torch.float64)
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
            return float((mean_cos + 1.0) * 0.5)
            ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        self.model.eval()

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

            seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            ### >>> Self_EvolveRec-BLOCK-START: Score through the compatibility encoder
            log_feats = self.view.log2feats(seq_np)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            ### >>> Self_EvolveRec-BLOCK-START: Score through the compatibility scorer
            all_logits = self.view.predict_score(log_feats, all_items)
            ### <<< Self_EvolveRec-BLOCK-END
            margin = all_logits[..., 0] - all_logits[..., 1]

            valid = pos_t != 0
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = pos_cpu != 0

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
                "mean": 0.0,
                "p10": 0.0,
                "neg_beats_pos_rate": 0.0,
                "small_margin_rate": 0.0,
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

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Validate temporal controls and report direct paired contrasts with isolated state
    @staticmethod
    def _histories(view):
        data = view.user_train
        return data.get("History", data)

    @staticmethod
    def _order_key(value):
        return (str(type(value)), str(value))

    def _sequence(self, items, maxlen):
        seq = torch.zeros((1, maxlen), dtype=torch.long, device=self.device)
        items = list(items)[-maxlen:]
        if items:
            seq[0, -len(items) :] = torch.tensor(items, device=self.device)
        return seq

    def _features(self, view, seq):
        # The initial view contract accepts NumPy sequence IDs. The actual
        # nn.Module is used only for mode management, never as a view substitute.
        features = view.log2feats(seq.detach().cpu().numpy())
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("log2feats must return B x L x D tensors")
        return features[:, -1:, :]

    def _scores(self, view, features, ids):
        items = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        scores = view.predict_score(features, items.reshape(1, 1, -1))
        values = scores.detach().float().cpu().numpy().reshape(-1)
        if len(values) != len(ids) or not np.isfinite(values).all():
            raise ValueError("Diagnostic scorer returned invalid scores")
        return values.astype(np.float64)

    @staticmethod
    def _top(ids, scores, k=10):
        ids = np.asarray(ids, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        order = np.lexsort((ids, -scores))[:k]
        return ids[order], scores[order]

    def _sampled_measure(self, view, feature, candidates, target):
        scores = self._scores(view, feature, candidates)
        ids = np.asarray(candidates, dtype=np.int64)
        target_index = int(np.flatnonzero(ids == target)[0])
        target_score = scores[target_index]
        precedes = (scores > target_score) | ((scores == target_score) & (ids < target))
        negative_scores = scores[ids != target]
        top, _ = self._top(ids, scores)
        rank = 1 + int(precedes.sum())
        return {
            "rank": rank,
            "hit10": float(rank <= 10),
            "margin": (
                float(target_score - negative_scores.mean())
                if len(negative_scores)
                else None
            ),
            "candidate_count": len(ids),
            "top10": top.tolist(),
        }

    def _catalog_measure(self, view, feature, itemnum, exclusions, target=None):
        target_score = None
        if target is not None:
            target_score = float(self._scores(view, feature, [target])[0])
        top_ids = np.empty(0, dtype=np.int64)
        top_scores = np.empty(0, dtype=np.float64)
        preceding, total, negative_count, negative_sum = 0, 0, 0, 0.0
        for start in range(1, itemnum + 1, 4096):
            ids = np.asarray(
                [
                    i
                    for i in range(start, min(itemnum + 1, start + 4096))
                    if i not in exclusions
                ],
                dtype=np.int64,
            )
            if not len(ids):
                continue
            scores = self._scores(view, feature, ids)
            # Use the same target score in every comparison, including exact ties.
            if target is not None:
                scores[ids == target] = target_score
                preceding += int(
                    (
                        (scores > target_score)
                        | ((scores == target_score) & (ids < target))
                    ).sum()
                )
                negative = scores[ids != target]
                negative_sum += float(negative.sum())
                negative_count += len(negative)
            total += len(ids)
            chunk_ids, chunk_scores = self._top(ids, scores)
            top_ids, top_scores = self._top(
                np.concatenate((top_ids, chunk_ids)),
                np.concatenate((top_scores, chunk_scores)),
            )
        return {
            "rank": 1 + preceding if target is not None else None,
            "hit10": float(preceding < 10) if target is not None else None,
            "margin": (
                float(target_score - negative_sum / negative_count)
                if target is not None and negative_count
                else None
            ),
            "candidate_count": total,
            "top10": top_ids.tolist(),
        }

    @staticmethod
    def _matched_removals(prefix, tertiles, rng):
        from itertools import combinations, combinations_with_replacement

        n = len(prefix)
        r = min(3, (n - 1) // 3)
        recent_pool = tuple(range(n - 1 - 2 * r, n - 1))
        older_pool = tuple(range(2 * r))

        def vector(positions):
            return tuple(
                sum(tertiles[prefix[p]] == t for p in positions) for t in range(3)
            )

        recent_options = [(p, vector(p)) for p in combinations(recent_pool, r)]
        older_options = [(p, vector(p)) for p in combinations(older_pool, r)]
        discrepancy, recent, older, recent_vector, older_vector = min(
            (sum(abs(a - b) for a, b in zip(rv, ov)), rp, op, rv, ov)
            for rp, rv in recent_options
            for op, ov in older_options
        )
        groups = [
            [p for p in range(n - 1) if tertiles[prefix[p]] == t] for t in range(3)
        ]
        feasible = []
        for labels in combinations_with_replacement(range(3), r):
            counts = tuple(labels.count(t) for t in range(3))
            if all(counts[t] <= len(groups[t]) for t in range(3)):
                distance = sum(abs(a - b) for a, b in zip(counts, recent_vector))
                feasible.append((distance, counts))
        random_distance, random_vector = min(feasible)
        random_positions = sorted(
            int(p)
            for t in range(3)
            if random_vector[t]
            for p in rng.choice(groups[t], size=random_vector[t], replace=False)
        )
        positions = {
            "recent": list(recent),
            "older": list(older),
            "random": random_positions,
        }
        return positions, {
            "r": r,
            "recent_pool": list(recent_pool),
            "older_pool": list(older_pool),
            "random_pool": list(range(n - 1)),
            "tertile_counts": {
                name: list(vector(pos)) for name, pos in positions.items()
            },
            "recent_older_L1": discrepancy,
            "random_recent_L1": random_distance,
        }

    @staticmethod
    def _timestamp(value):
        from datetime import datetime, timezone

        try:
            number = float(value)
            if not np.isfinite(number):
                return None
            # Normalize contemporary Unix milliseconds/microseconds/nanoseconds
            # to seconds so numeric and ISO timestamps can be compared.
            magnitude = abs(number)
            if magnitude >= 1e17:
                number /= 1e9
            elif magnitude >= 1e14:
                number /= 1e6
            elif magnitude >= 1e11:
                number /= 1e3
            return number
        except (TypeError, ValueError, OverflowError):
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except (TypeError, ValueError, OverflowError, OSError):
                return None

    def _availability(self, view, histories):
        time_data = view.user_train.get("Time", {})
        first_seen, usable_times = {}, {}
        for user, history in histories.items():
            raw_times = time_data.get(user, time_data.get(str(user), []))
            if len(raw_times) != len(history):
                continue
            parsed = [self._timestamp(t) for t in raw_times]
            if any(t is None for t in parsed):
                continue
            if any(a > b for a, b in zip(parsed, parsed[1:])):
                # Unsorted timestamps do not support chronological plausibility checks.
                continue
            usable_times[user] = parsed
            for item, timestamp in zip(history, parsed):
                item = int(item)
                first_seen[item] = min(first_seen.get(item, timestamp), timestamp)
        return usable_times, first_seen

    def _isolated_margin(self, view, histories, itemnum, maxlen):
        # This separate deterministic iterable never touches train_loader or its sampler.
        rng = np.random.default_rng(42)
        users = sorted(histories, key=self._order_key)
        users = [users[int(i)] for i in rng.permutation(len(users))]
        values, batches = [], []
        for user in users:
            history = [int(i) for i in histories[user]]
            if len(history) < 2 or any(i < 1 or i > itemnum for i in history):
                continue
            seen = set(history)
            if len(seen) >= itemnum:
                continue
            seq = np.zeros(maxlen, dtype=np.int64)
            pos = np.zeros(maxlen, dtype=np.int64)
            neg = np.zeros(maxlen, dtype=np.int64)
            prefix = history[:-1][-maxlen:]
            targets = history[-len(prefix) :]
            eligible = None
            for offset, (item, target) in enumerate(
                zip(prefix, targets), maxlen - len(prefix)
            ):
                seq[offset], pos[offset] = item, target
                for _ in range(128):
                    candidate = int(rng.integers(1, itemnum + 1))
                    if candidate not in seen:
                        break
                else:
                    if eligible is None:
                        eligible = [i for i in range(1, itemnum + 1) if i not in seen]
                    candidate = int(rng.choice(eligible))
                neg[offset] = candidate
            batches.append((seq, pos, neg))
            if len(batches) >= 320:
                break
        for start in range(0, len(batches), 32):
            seq, pos, neg = (np.stack(x) for x in zip(*batches[start : start + 32]))
            features = view.log2feats(seq)
            ids = torch.as_tensor(
                np.stack((pos, neg), axis=-1), dtype=torch.long, device=self.device
            )
            scores = view.predict_score(features, ids)
            margin = (scores[..., 0] - scores[..., 1]).detach().float().cpu().numpy()
            valid = (pos != 0) & (neg != 0) & np.isfinite(margin)
            values.extend(margin[valid].astype(float).tolist())
        return {
            "status": "available" if values else "unavailable",
            "reason": (
                None if values else "No valid isolated diagnostic margin positions"
            ),
            "valid_positions": len(values),
            "sampled_users": len(batches),
            "eps": 0.5,
            "mean": float(np.mean(values)) if values else None,
            "p10": float(np.quantile(values, 0.1)) if values else None,
            "neg_beats_pos_rate": (
                float(np.mean(np.asarray(values) < 0)) if values else None
            ),
            "small_margin_rate": (
                float(np.mean(np.asarray(values) < 0.5)) if values else None
            ),
            "item_category_breakdown": {
                "status": "unavailable",
                "value": None,
                "reason": "Item taxonomy is not available in this shared task",
            },
        }

    @staticmethod
    def _summaries(cases):
        variants = sorted({v for c in cases for v in c["interventions"]})
        routes = sorted({c["dataset"] for c in cases})
        output = {}
        for variant in variants:
            output[variant] = {}
            for setting in ("sampled", "catalog"):
                rows = []
                for case in cases:
                    changed = case["interventions"].get(variant)
                    if changed is None:
                        continue
                    base, current = case["baseline"][setting], changed[setting]
                    denominator = min(10, base["candidate_count"])
                    rows.append(
                        {
                            "dataset": case["dataset"],
                            "user": case["user"],
                            "length_slice": case["length_slice"],
                            "target_frequency_tertile": case[
                                "target_frequency_tertile"
                            ],
                            "repeated_target": case["target_in_prefix"],
                            "rank_delta": float(current["rank"] - base["rank"]),
                            "hit10_delta": current["hit10"] - base["hit10"],
                            "margin_delta": (
                                current["margin"] - base["margin"]
                                if current["margin"] is not None
                                and base["margin"] is not None
                                else None
                            ),
                            "top10_overlap": (
                                len(set(base["top10"]) & set(current["top10"]))
                                / denominator
                                if denominator
                                else None
                            ),
                        }
                    )

                def aggregate(selected):
                    result = {"users": len(selected)}
                    for metric in (
                        "rank_delta",
                        "hit10_delta",
                        "margin_delta",
                        "top10_overlap",
                    ):
                        by_route = {}
                        for route in routes:
                            values = [
                                row[metric]
                                for row in selected
                                if row["dataset"] == route and row[metric] is not None
                            ]
                            by_route[route] = {
                                "count": len(values),
                                "mean": float(np.mean(values)) if values else None,
                            }
                        valid_routes = [r for r in routes if by_route[r]["count"]]
                        means = [by_route[r]["mean"] for r in valid_routes]
                        interval = None
                        if means:
                            bootstrap_rng = np.random.default_rng(42)
                            draws = np.zeros(1000, dtype=np.float64)
                            for route in valid_routes:
                                values = np.asarray(
                                    [
                                        row[metric]
                                        for row in selected
                                        if row["dataset"] == route
                                        and row[metric] is not None
                                    ]
                                )
                                indices = bootstrap_rng.integers(
                                    0, len(values), size=(1000, len(values))
                                )
                                draws += values[indices].mean(axis=1) / len(
                                    valid_routes
                                )
                            interval = np.quantile(draws, [0.025, 0.975]).tolist()
                        result[metric] = {
                            "per_category": by_route,
                            "available_category_macro_mean": (
                                float(np.mean(means)) if means else None
                            ),
                            "paired_user_bootstrap_95_percentile": interval,
                            "contributing_categories": valid_routes,
                        }
                    return result

                slices = {}
                for field, levels in (
                    ("length_slice", ("8-15", "16-31", "32+")),
                    ("target_frequency_tertile", (0, 1, 2)),
                    ("repeated_target", (False, True)),
                ):
                    slices[field] = {
                        str(level): aggregate([r for r in rows if r[field] == level])
                        for level in levels
                    }
                output[variant][setting] = {
                    "overall": aggregate(rows),
                    "slices": slices,
                    "paired_users": rows,
                }
        return output

    def _route_diagnosis(self, name, view):
        histories = self._histories(view)
        itemnum = int(view.item_emb.weight.shape[0] - 1)
        maxlen = int(getattr(view.model, "maxlen", getattr(view, "maxlen", 128)))
        if maxlen < 1:
            raise ValueError("Diagnostic sequence limit must be positive")
        counts = Counter(int(i) for history in histories.values() for i in history)
        ordered_items = sorted(range(1, itemnum + 1), key=lambda i: (counts[i], i))
        tertiles = {
            item: min(2, 3 * index // max(itemnum, 1))
            for index, item in enumerate(ordered_items)
        }
        rng = np.random.default_rng(42)
        eligible_users = []
        invalid_histories = 0
        for user in sorted(histories, key=self._order_key):
            history = [int(i) for i in histories[user]]
            if any(i < 1 or i > itemnum for i in history):
                invalid_histories += 1
                continue
            if len(history[:-1][-maxlen:]) >= 8:
                eligible_users.append(user)
        chosen = [
            eligible_users[int(i)] for i in rng.permutation(len(eligible_users))[:50]
        ]
        times, first_seen = self._availability(view, histories)
        hook = getattr(view, "diagnostic_log2feats", None)
        if hook is None:
            hook = getattr(view.model, "diagnostic_log2feats", None)
        hook_contract = getattr(
            view,
            "diagnostic_mask_contract",
            getattr(view.model, "diagnostic_mask_contract", None),
        )
        supported = (
            callable(hook) and hook_contract == "keys_values_queries_every_layer"
        )
        missing_control = (
            None
            if supported
            else (
                "Requires diagnostic_log2feats(sequence, position_ids=..., keep_mask=...) "
                "and diagnostic_mask_contract='keys_values_queries_every_layer'; the hook must "
                "retain position IDs, mask removed keys/values in every layer, suppress removed "
                "query outputs, and apply the same keep mask to any residual history branch. "
                "The baseline log2feats contract alone does not guarantee this behavior."
            )
        )
        probe = DiagnosisProbe(view, self.device)
        embedding_score = probe.analyze_embeddings_collapse_cosine()
        route = {
            "dataset": name,
            "eligible_users": len(eligible_users),
            "invalid_histories": invalid_histories,
            "selected_users": [str(u) for u in chosen],
            "itemnum": itemnum,
            "maxlen": maxlen,
            "embedding_alignment": {
                "value": embedding_score,
                "sampled_items": min(2048, itemnum),
                "valid_pairs": min(2048, itemnum) * max(0, min(2048, itemnum) - 1),
            },
            "sampled_margin": self._isolated_margin(view, histories, itemnum, maxlen),
            "position_preserving_control": {
                "status": "available" if supported else "unavailable",
                "reason": missing_control,
            },
            "cases": [],
        }
        candidate_rng = np.random.default_rng(42)
        intervention_rng = np.random.default_rng(43)
        for user in chosen:
            history = [int(i) for i in histories[user]]
            target = history[-1]
            prefix = history[:-1][-maxlen:]
            seen = set(history)
            eligible_negatives = np.asarray(
                [i for i in range(1, itemnum + 1) if i not in seen], dtype=np.int64
            )
            sampled = candidate_rng.choice(
                eligible_negatives, size=min(99, len(eligible_negatives)), replace=False
            )
            candidates = [target] + [int(i) for i in sampled]
            # The complete original history exclusion is frozen for every variant.
            exclusions = seen - {target}
            positions, matching = self._matched_removals(
                prefix, tertiles, intervention_rng
            )
            n = len(prefix)
            seq = self._sequence(prefix, maxlen)
            baseline_feature = self._features(view, seq)
            if supported:
                position_ids = torch.arange(maxlen, device=self.device).reshape(1, -1)
                control_baseline = hook(
                    seq, position_ids=position_ids, keep_mask=(seq != 0)
                )
                if not isinstance(
                    control_baseline, torch.Tensor
                ) or control_baseline.shape != (1, maxlen, baseline_feature.shape[-1]):
                    raise ValueError(
                        "Diagnostic hook returned an invalid feature shape"
                    )
                # A hook that changes the unperturbed encoder is not a paired
                # position-preserving control for this fitted model.
                torch.testing.assert_close(
                    control_baseline[:, -1:, :],
                    baseline_feature,
                    rtol=1e-4,
                    atol=1e-5,
                    msg="Diagnostic hook must reproduce the unperturbed final representation",
                )

            def measure(feature):
                return {
                    "sampled": self._sampled_measure(view, feature, candidates, target),
                    "catalog": self._catalog_measure(
                        view, feature, itemnum, exclusions, target
                    ),
                }

            record = {
                "dataset": name,
                "user": str(user),
                "prefix_cutoff": len(history) - 1,
                "prefix_start": max(0, len(history) - 1 - maxlen),
                "prefix_items": prefix,
                "target": target,
                "prefix_length": n,
                "length_slice": "8-15" if n < 16 else "16-31" if n < 32 else "32+",
                "target_frequency_tertile": tertiles[target],
                "target_in_prefix": target in prefix,
                "candidate_ids": candidates,
                "original_seen_items": sorted(seen),
                "positions_are": "zero-based within truncated nonpadding prefix",
                "removal_positions": positions,
                "matching": matching,
                "removed_training_frequencies": {
                    key: [counts[prefix[p]] for p in value]
                    for key, value in positions.items()
                },
                "baseline": measure(baseline_feature),
                "interventions": {},
            }
            for kind, removed in positions.items():
                survivors = [item for p, item in enumerate(prefix) if p not in removed]
                feature = self._features(view, self._sequence(survivors, maxlen))
                record["interventions"][kind + "_repacked"] = measure(feature)
                if supported:
                    keep = seq != 0
                    keep[0, [maxlen - n + p for p in removed]] = False
                    position_ids = torch.arange(maxlen, device=self.device).reshape(
                        1, -1
                    )
                    feature = hook(seq, position_ids=position_ids, keep_mask=keep)
                    if not isinstance(feature, torch.Tensor) or feature.shape != (
                        1,
                        maxlen,
                        baseline_feature.shape[-1],
                    ):
                        raise ValueError(
                            "Diagnostic hook returned an invalid feature shape"
                        )
                    if not torch.isfinite(feature).all():
                        raise FloatingPointError(
                            "Nonfinite position-preserving features"
                        )
                    if torch.count_nonzero(feature[~keep]).item() != 0:
                        raise ValueError(
                            "Diagnostic hook did not suppress removed query outputs"
                        )
                    record["interventions"][kind + "_position_preserving"] = measure(
                        feature[:, -1:, :]
                    )
            record["last_item_removal_position"] = n - 1
            record["interventions"]["last_item_repacked"] = measure(
                self._features(view, self._sequence(prefix[:-1], maxlen))
            )
            local_positions = list(range(n - 1 - min(5, n - 1), n - 1))
            permutation = intervention_rng.permutation(len(local_positions))
            if np.array_equal(permutation, np.arange(len(local_positions))):
                permutation = np.roll(permutation, 1)
            shuffled = list(prefix)
            for destination, source in zip(local_positions, permutation):
                shuffled[destination] = prefix[local_positions[int(source)]]
            timestamp_status, violations = "unverified", None
            if user in times:
                prefix_times = times[user][:-1][-maxlen:]
                if len(prefix_times) == n:
                    violations = sum(
                        first_seen[shuffled[p]] > prefix_times[p]
                        for p in local_positions
                    )
                    timestamp_status = (
                        "checked_against_earliest_public_training_observation"
                    )
            record["shuffle"] = {
                "positions": local_positions,
                "source_permutation": permutation.tolist(),
                "item_sequence_changed": shuffled != prefix,
                "timestamp_plausibility": timestamp_status,
                "availability_violations": violations,
            }
            if violations:
                record["shuffle"][
                    "unavailable_reason"
                ] = "Permutation violates observed item-availability bounds"
            else:
                record["interventions"]["local_shuffle"] = measure(
                    self._features(view, self._sequence(shuffled, maxlen))
                )
            route["cases"].append(record)

        # Keep externally supplied identities; otherwise choose five independently.
        requested = getattr(view, "diagnostic_user_ids", None)
        source = (
            "externally_specified"
            if requested is not None
            else "deterministic_training_sample"
        )
        if requested is None:
            valid_users = [
                u
                for u in sorted(histories, key=self._order_key)
                if histories[u] and all(1 <= int(i) <= itemnum for i in histories[u])
            ]
            rec_rng = np.random.default_rng(42)
            requested = [
                valid_users[int(i)] for i in rec_rng.permutation(len(valid_users))[:5]
            ]
        lookup = {str(u): u for u in histories}
        recommendations = []
        for requested_user in requested:
            user = lookup.get(str(requested_user))
            if user is None or not histories[user]:
                recommendations.append(
                    {
                        "user": str(requested_user),
                        "status": "unavailable",
                        "reason": "Requested identity has no usable training history",
                    }
                )
                continue
            history = [int(i) for i in histories[user]]
            if any(i < 1 or i > itemnum for i in history):
                recommendations.append(
                    {
                        "user": str(requested_user),
                        "status": "unavailable",
                        "reason": "Requested history contains out-of-range item IDs",
                    }
                )
                continue
            feature = self._features(view, self._sequence(history, maxlen))
            result = self._catalog_measure(view, feature, itemnum, set(history))
            recommendations.append(
                {
                    "user": str(user),
                    "status": "available",
                    "history_cutoff": len(history),
                    **result,
                }
            )
        route["recommendations"] = {
            "identity_source": source,
            "requested_count": len(requested),
            "users": recommendations,
            "required_default_count": 5,
        }
        return route

    def run_full_diagnosis(self, train_loader):
        import os
        import time
        from pathlib import Path

        # Optional multi-route views let the adapter compute a four-category macro
        # directly. A single-route invocation reports only its observed category.
        expected = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        routes = getattr(self.view, "diagnostic_routes", None)
        if routes is None:
            name = getattr(
                self.view,
                "dataset_name",
                getattr(self.model, "dataset_name", "unlabeled_route"),
            )
            routes = {str(name): self.view}
        if not routes:
            raise ValueError("No diagnostic routes supplied")
        modules, module_ids = [], set()
        for view in routes.values():
            for module in view.model.modules():
                if id(module) not in module_ids:
                    module_ids.add(id(module))
                    modules.append((module, module.training))
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cuda_devices = []
        for view in routes.values():
            tensors = list(view.model.parameters()) + list(view.model.buffers())
            for tensor in tensors:
                if tensor.is_cuda and tensor.device.index not in cuda_devices:
                    cuda_devices.append(tensor.device.index)
        if self.device.type == "cuda":
            active_device = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            if active_device not in cuda_devices:
                cuda_devices.append(active_device)
        started = time.perf_counter()
        metrics = {
            "configuration": {
                "seed": 42,
                "intervention_seed": 43,
                "rng_streams": "Independent user selection, candidate sampling, and intervention generators",
                "max_users_per_category": 50,
                "minimum_prefix_items": 8,
                "target_rule": "latest training item; preceding prefix truncated to model sequence limit",
                "auxiliary_max_unique_negatives": 99,
                "catalog_chunk_size": 4096,
                "bootstrap_replicates": 1000,
                "recommendation_k": 10,
                "training_loader_consumed": False,
                "margin_loader": "independent deterministic histories; up to 320 users, batches of 32",
                "predeclared_subsequent_ablation": {
                    "name": "scalar_gated_recent_history_residual",
                    "status": "declared_only_not_executed_by_diagnostics",
                    "comparison": "same fitted-training protocol and seeds; change only the residual gate to one scalar",
                    "selection_metric": "unchanged host macro validation HR@10",
                },
            },
            "expected_routes": expected,
            "observed_routes": list(routes),
            "missing_routes": [r for r in expected if r not in routes],
            "model_order_agreement": {
                "status": "unavailable",
                "value": None,
                "reason": "Requires multiple fitted models on identical cases and candidates",
            },
            "route_results": {},
            "trial_outcome": "running",
        }
        definitions = {
            "embedding_alignment": "(mean off-diagonal sampled item-embedding cosine + 1)/2; directional alignment only, not evidence of preference diversity or rank collapse. Padding excluded; ordered valid-pair count reported.",
            "sampled_margin": "Independent training-only probe: positive minus one sampled unseen-negative score at each valid next-item position. Mean, interpolated p10, fraction <0, fraction <0.5, and valid count. Not held-out ranking quality.",
            "rank": "One plus the number of eligible candidates preceding target under score descending, item ID ascending. Sampled and catalog settings are distinct.",
            "margin": "Target score minus mean score of fixed eligible negatives; null when no negatives exist. Catalog margin uses all eligible non-target items.",
            "rank_delta": "Intervention rank minus baseline rank on identical candidates; positive is worse target rank.",
            "hit10_delta": "Intervention indicator(rank<=10) minus baseline indicator; an in-sample target-ranking change, not host HR@10.",
            "margin_delta": "Intervention margin minus baseline margin on fixed negatives; score-scale dependent.",
            "top10_overlap": "Intersection of baseline and intervention top-10 item sets divided by min(10, candidate_count), with deterministic item-ID tie breaks.",
            "matching": "r=min(3,floor((n-1)/3)); recent newest 2r nonfinal positions, older earliest 2r, random all nonfinal. Enumerated recent/older subsets minimize frequency-tertile-count L1, then position lexicographic order. Random subset uniformly sampled within a feasible minimum-discrepancy tertile vector; vector ties lexicographic.",
            "frequency_tertiles": "Category training occurrence counts, sorted ascending with item ID breaking ties, partitioned by min(2,floor(3*index/catalog_size)); labels 0/1/2. Unobserved catalog items have count zero.",
            "repacked": "Delete selected prefix positions, compact survivors, restore original right-aligned padding; changes evidence and absolute positions.",
            "last_item_repacked": "Separate one-item final-query deletion; not an equal-count recent/older comparison.",
            "position_preserving": "Optional explicitly declared architecture hook: original position IDs retained, removed keys/values masked at every layer, removed query outputs suppressed, and residual history branches masked. Still changes context; not preference causality.",
            "local_shuffle": "Nonidentity permutation of last min(5,n-1) nonfinal occupied positions, preserving final item and multiset. Equal repeated items can leave item sequence unchanged. Availability uses earliest observed public training timestamp only, not true release dates; unverified when timestamps missing.",
            "target_in_prefix": "Exact item repeat in truncated prefix; repeated targets remain eligible for this diagnostic and are reported separately. No ownership or demand interpretation.",
            "available_category_macro_mean": "Equal-weight average of nonempty observed category means, with contributing categories listed. It is a four-category macro only when all four contribute; empty values are null.",
            "paired_user_bootstrap_95_percentile": "1000 user-within-category resamples of paired differences, averaging nonempty category means equally. Conditional on fitted models, fixed cases/candidates; excludes training/candidate variability and adaptive selection bias.",
            "recommendations": "Exact chunked full-category top-10 excluding padding and complete original seen history, scoring each selected user's encoded history once. Chunk pruning and merging use score descending/item ID ascending. May return fewer than ten unseen items.",
            "candidate_ids": "Target plus up to 99 unique category negatives sampled without replacement, excluding complete original training history. Auxiliary seed 42 does not replicate host candidate sampling.",
            "catalog": "All category items except original training-history items other than target. Same exclusions for every intervention. Sampled/catalog ranks compare candidate settings of one model, not order agreement across models.",
            "direct_temporal_contrasts": "Paired recent-minus-older and random-minus-older comparisons on the same users and fixed candidates, reported separately for repacked and supported position-preserving controls. Rank, hit10, and margin differences use older removal as reference; overlap compares the two intervention top-10 sets. Bootstrap resamples paired user contrasts directly. Frequency matching does not establish preference causality.",
            "configuration": "Diagnostic-only configuration, counts, missing controls, and declared future ablation. Independent RNG streams derive deterministically from seed 42; intervention stream uses seed 43. No training changes, validation selection, or hidden test access occurs here.",
        }
        result = {"metrics": metrics, "metric_definitions": definitions}
        error = None
        try:
            with torch.random.fork_rng(devices=sorted(cuda_devices)):
                for view in routes.values():
                    view.model.eval()
                with torch.inference_mode():
                    all_cases = []
                    for name, view in routes.items():
                        route = self._route_diagnosis(str(name), view)
                        metrics["route_results"][str(name)] = route
                        all_cases.extend(route["cases"])
                    metrics["paired_summaries"] = self._summaries(all_cases)
                    # Direct temporal contrasts require paired recent-minus-older
                    # observations; subtracting marginal confidence bounds is invalid.
                    contrast_cases = []
                    for case in all_cases:
                        for mode in ("repacked", "position_preserving"):
                            older = case["interventions"].get("older_" + mode)
                            if older is None:
                                continue
                            alternatives = {
                                kind
                                + "_minus_older_"
                                + mode: case["interventions"][kind + "_" + mode]
                                for kind in ("recent", "random")
                                if kind + "_" + mode in case["interventions"]
                            }
                            contrast_cases.append(
                                {
                                    **case,
                                    "baseline": older,
                                    "interventions": alternatives,
                                }
                            )
                    metrics["direct_temporal_contrasts"] = self._summaries(
                        contrast_cases
                    )
                    metrics["case_count"] = len(all_cases)
                    metrics["case_status"] = {
                        "status": "available" if all_cases else "unavailable",
                        "reason": (
                            None
                            if all_cases
                            else "No training prefixes satisfy the minimum length"
                        ),
                    }
                    metrics["four_route_coverage_complete"] = all(
                        any(case["dataset"] == name for case in all_cases)
                        for name in expected
                    )
                    metrics["trial_outcome"] = "completed"
        except Exception as exc:
            error = exc
            metrics["trial_outcome"] = "failed"
            metrics["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            for module, training in modules:
                # Direct assignment preserves heterogeneous child-module flags.
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            metrics["runtime_seconds"] = time.perf_counter() - started
            output_dir = Path(__file__).resolve().parent / "working" / "diagnostics"
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"temporal_{os.getpid()}_{time.time_ns()}.json"
            metrics["artifact_path"] = str(path)
            with path.open("w") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
        if error is not None:
            raise error
        return result

    ### <<< Self_EvolveRec-BLOCK-END
