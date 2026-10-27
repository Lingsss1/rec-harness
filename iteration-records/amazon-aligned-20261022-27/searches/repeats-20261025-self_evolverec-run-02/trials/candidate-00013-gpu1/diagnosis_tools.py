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
    ### >>> Self_EvolveRec-BLOCK-START: Validate the actual module while preserving the scoring adapter
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.module = view.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.user_train = view.user_train
        self.dataset = getattr(view, "dataset_name", None)
        if self.dataset is None:
            self.dataset = getattr(self.module, "dataset_name", None)
        self.routes = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )

    def _get_categories(self, item_id: int):
        # Item categories are not available in this matched task.
        return []

    # Review summary: statically checked adapter call sites, bounded preprocessing,
    # independent B x L batches, exact policy cutoff, missingness, and state restoration.
    # Full matched-search orchestration remains unimplemented in this diagnostic file.
    # No execution, training, or validation improvement is claimed.
    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            ### >>> Self_EvolveRec-BLOCK-START: Report undefined embedding alignment without fabricating zeros
            n = emb.size(0)
            if n < 2 or sample_size < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Validate sampled embeddings and accumulate cosine statistics in float32
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite sampled item embeddings")
            x = F.normalize(x, p=2, dim=-1)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Compute identical mean cosine without a quadratic Gram matrix
            offdiag_sum = x.sum(0).square().sum() - x.square().sum()
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
        ### >>> Self_EvolveRec-BLOCK-START: Let the enclosing diagnostic own evaluation mode restoration
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

            ### >>> Self_EvolveRec-BLOCK-START: Respect the tensor adapter contract and validate probe shapes
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis batches must contain B x L sequence/positive/negative tensors"
                )
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding negatives and reject nonfinite valid scores
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if valid.any():
                if not torch.isfinite(margin[valid]).all():
                    raise ValueError("Nonfinite sampled positive-negative margins")
                margins_all.append(margin[valid].detach().float().cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same valid positions for all margin summaries
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
                ### >>> Self_EvolveRec-BLOCK-START: Distinguish unavailable margins from measured zeros
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

        ### >>> Self_EvolveRec-BLOCK-START: Explicitly report unavailable item-category breakdowns
        breakdown = {
            "available": False,
            "unavailable_reason": "Item categories are not provided in the matched task",
            "valid_positions": int(sum(m.numel() for m in margins_all)),
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

    ### >>> Self_EvolveRec-BLOCK-START: Coverage-verified guidance with bounded independent probes and explicit failures
    @staticmethod
    def _timestamp(value):
        from datetime import datetime, timezone

        try:
            result = float(value)
            # Reject infinity before unit conversion: infinity / 1000 stays infinite.
            if not np.isfinite(result):
                return None
            # Public timestamps may be seconds, milliseconds, microseconds or nanoseconds.
            while abs(result) >= 1e11:
                result /= 1000.0
            return result
        except (TypeError, ValueError):
            try:
                stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                return stamp.timestamp()
            except (TypeError, ValueError, OverflowError):
                return None

    def _training_histories(self):
        source = self.user_train
        if "History" in source:
            return source["History"], source.get("Time", {})
        return source, {}

    def _text_groups(self, observed, deadline):
        import time
        from sklearn.feature_extraction.text import TfidfVectorizer

        ids, documents = [], []
        for item in sorted(observed):
            if time.monotonic() >= deadline:
                raise TimeoutError
            info = self.meta_data.get(str(item), self.meta_data.get(item, {}))
            info = info if isinstance(info, dict) else {}
            fields = []
            for name in ("title", "description"):
                value = info.get(name, "")
                if isinstance(value, str):
                    fields.append(value)
                elif isinstance(value, list):
                    fields.extend(v for v in value if isinstance(v, str))
            ids.append(item)
            documents.append(" ".join(fields))
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
            min_df=1,
            max_features=20000,
            norm="l2",
            dtype=np.float32,
        )
        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            return (
                {},
                None,
                {
                    "available": False,
                    "reason": "No usable training-observed title/description text",
                },
            )
        if time.monotonic() >= deadline:
            raise TimeoutError
        return (
            {item: row for row, item in enumerate(ids)},
            matrix,
            {
                "available": True,
                "vocabulary_size": len(vectorizer.vocabulary_),
                "training_observed_items": len(ids),
                "items_with_lexical_vectors": int(
                    np.count_nonzero(matrix.getnnz(axis=1))
                ),
            },
        )

    def _features(self, history, maxlen):
        if maxlen < 1:
            raise ValueError("Diagnostic maxlen must be positive")
        history = tuple(history)[-maxlen:]
        sequence = torch.zeros((1, maxlen), device=self.device, dtype=torch.long)
        if history:
            sequence[0, -len(history) :] = torch.as_tensor(
                history, device=self.device, dtype=torch.long
            )
        features = self.model.log2feats(sequence)
        if features.ndim != 3 or features.shape[:2] != sequence.shape:
            raise ValueError("log2feats must return B x L x H features")
        final = features[:, -1, :]
        if not torch.isfinite(final).all():
            raise ValueError("Nonfinite diagnostic final features")
        return final.clone()

    def _scores(self, history, candidates, maxlen, deadline):
        import time

        if time.monotonic() >= deadline:
            raise TimeoutError("Diagnostic scoring deadline reached")
        if len(candidates) == 0:
            raise ValueError("Diagnostic candidate pool is empty")
        features = self._features(history, maxlen)
        output = []
        for start in range(0, len(candidates), 8192):
            if time.monotonic() >= deadline:
                raise TimeoutError
            items = torch.as_tensor(
                candidates[start : start + 8192].copy(),
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)
            scores = self.model.predict_score(features, items)
            if scores.shape != items.shape:
                raise ValueError(
                    "predict_score must return scores of shape 1 x candidates"
                )
            scores = scores.detach().float().reshape(-1).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite full-catalog diagnostic scores")
            output.append(scores)
        return np.concatenate(output)

    @staticmethod
    def _matched_controls(target, eligible, strata, rng):
        need = Counter(strata[i] for i in target)
        pools = {
            key: [i for i in eligible if strata[i] == key and i not in target]
            for key in need
        }
        if any(len(pools[key]) < count for key, count in need.items()):
            return None
        return [
            sorted(
                i for key, count in need.items() for i in rng.sample(pools[key], count)
            )
            for _ in range(5)
        ]

    @staticmethod
    def _rank_change(baseline, changed, candidates, baseline_order):
        order = np.lexsort((candidates, -changed))
        original_top = int(baseline_order[0])
        runner_up = int(baseline_order[1])
        return {
            "top10_disruption": 1.0
            - len(set(baseline_order[:10]) & set(order[:10])) / 10.0,
            "original_top_displacement": float(
                np.flatnonzero(order == original_top)[0]
            ),
            "original_pair_gap_change": float(
                (float(changed[original_top]) - float(changed[runner_up]))
                - (float(baseline[original_top]) - float(baseline[runner_up]))
            ),
        }

    @staticmethod
    def _summaries(records):
        # A user can contribute multiple records; collapse those before bootstrapping.
        grouped = {}
        for record in records:
            key = (record["contrast"], record["length_bin"])
            for bucket in (key, (record["contrast"], "all")):
                grouped.setdefault(bucket, {}).setdefault(record["user"], []).append(
                    record
                )
        result = {}
        rng = np.random.default_rng(42)
        for (contrast, length_bin), users in sorted(grouped.items()):
            summary = {"users": len(users)}
            fields = [
                k
                for k in next(iter(users.values()))[0]
                if k.startswith("target_") or k.startswith("excess_")
            ]
            for field in fields:
                values = np.asarray(
                    [
                        np.mean([record[field] for record in rows])
                        for rows in users.values()
                    ],
                    dtype=np.float64,
                )
                entry = {
                    "mean": float(values.mean()),
                    "user_bootstrap_95_percent": None,
                }
                if len(values) >= 2:
                    boot = values[
                        rng.integers(0, len(values), size=(500, len(values)))
                    ].mean(1)
                    entry["user_bootstrap_95_percent"] = np.quantile(
                        boot, [0.025, 0.975]
                    ).tolist()
                summary[field] = entry
            result.setdefault(contrast, {})[length_bin] = summary
        return result

    def _perturbations(self, deadline):
        import time

        histories, times = self._training_histories()
        maxlen = int(getattr(self.module, "maxlen", 128))
        itemnum = int(self.model.item_emb.num_embeddings - 1)
        eligible, frequency = [], Counter()
        for user in sorted(histories, key=str):
            if time.monotonic() >= deadline:
                raise TimeoutError
            history = [int(i) for i in histories[user]]
            if any(i < 1 or i > itemnum for i in history):
                raise ValueError("Training history contains out-of-catalog items")
            frequency.update(history)
            if len(history[-maxlen:]) >= 8:
                eligible.append(user)
        rng = random.Random(42)
        selected = rng.sample(eligible, min(32, len(eligible)))
        edges = (
            np.quantile(list(frequency.values()), [0.25, 0.5, 0.75]).tolist()
            if frequency
            else []
        )
        if not selected:
            return {
                "available": False,
                "reason": "No training users have at least eight interactions within model.maxlen",
                "eligible_users": len(eligible),
                "selected_users": 0,
                "completed_users": 0,
                "budget_stopped": False,
                "coverage": {},
                "summaries": {},
                "user_records": [],
            }
        text_index, text_matrix, lexical_info = self._text_groups(frequency, deadline)
        records, coverage = [], Counter()
        rows = []
        completed = 0
        budget_stopped = False
        for user in selected:
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                full = [int(i) for i in histories[user]]
                prefix = full[-maxlen:]
                n = len(prefix)
                candidates = np.setdiff1d(
                    np.arange(1, itemnum + 1, dtype=np.int64),
                    np.unique(full),
                    assume_unique=True,
                )
                if len(candidates) < 10:
                    coverage["insufficient_unseen_candidates"] += 1
                    continue
                recent = np.arange(n) >= n - int(np.ceil(0.25 * n))
                counts = Counter(prefix)
                id_recurrent = np.asarray([counts[item] >= 2 for item in prefix])
                lexical_recurrent = np.zeros(n, dtype=bool)
                if text_matrix is not None:
                    vectors = text_matrix[[text_index[item] for item in prefix]]
                    similarities = (vectors @ vectors.T).toarray()
                    np.fill_diagonal(similarities, 0.0)
                    lexical_recurrent = (similarities >= 0.5).any(axis=1)
                recurrent = id_recurrent | lexical_recurrent
                coverage["sampled_positions"] += n
                coverage["id_recurrent_positions"] += int(id_recurrent.sum())
                coverage["lexical_recurrent_positions"] += int(lexical_recurrent.sum())
                raw_times = times.get(user, times.get(str(user), []))
                raw_times = raw_times if isinstance(raw_times, (list, tuple)) else []
                stamps = (
                    [self._timestamp(t) for t in raw_times[-n:]]
                    if len(raw_times) == len(full)
                    else [None] * n
                )
                endpoint = stamps[-1]
                strata, recent_strata = [], []
                for i, item in enumerate(prefix):
                    elapsed = (
                        None
                        if endpoint is None or stamps[i] is None
                        else (endpoint - stamps[i]) / 86400.0
                    )
                    time_bin = (
                        "missing"
                        if elapsed is None or elapsed < 0
                        else int(np.searchsorted([7, 30, 180], elapsed, side="left"))
                    )
                    popularity_bin = int(
                        np.searchsorted(edges, frequency[item], side="right")
                    )
                    strata.append(
                        (min(3, 4 * i // n), time_bin, popularity_bin, i == n - 1)
                    )
                    recent_strata.append((popularity_bin, bool(recurrent[i])))
                groups = {
                    "recent_only": np.flatnonzero(recent & ~recurrent).tolist(),
                    "recurrent_only": np.flatnonzero(recurrent & ~recent).tolist(),
                    "overlap_joint": np.flatnonzero(recent & recurrent).tolist(),
                    "isolated_nonrecent": np.flatnonzero(~recent & ~recurrent).tolist(),
                }
                # Freeze targets and controls before scoring any intervention.
                plans = []
                for contrast, group in groups.items():
                    coverage[contrast + ":attempted"] += 1
                    if not group:
                        coverage[contrast + ":empty_target"] += 1
                        continue
                    target = sorted(rng.sample(group, min(2, len(group), n - 2)))
                    if contrast == "recent_only":
                        control_pool = np.flatnonzero(~recent & ~recurrent).tolist()
                        matching = recent_strata
                    elif contrast == "isolated_nonrecent":
                        control_pool = np.flatnonzero(~recent & recurrent).tolist()
                        matching = strata
                    else:
                        control_pool = np.flatnonzero(~recurrent).tolist()
                        matching = strata
                    controls = self._matched_controls(
                        target, control_pool, matching, rng
                    )
                    if controls is None:
                        coverage[contrast + ":infeasible_exact_matching"] += 1
                        continue
                    plans.append((contrast, target, controls))
                permutations = {}
                for name, fixed_last in (
                    ("shuffle_all", False),
                    ("shuffle_last_fixed", True),
                ):
                    indices = list(range(n - int(fixed_last)))
                    rng.shuffle(indices)
                    if fixed_last:
                        indices.append(n - 1)
                    permutations[name] = [prefix[i] for i in indices]
                baseline = self._scores(prefix, candidates, maxlen, deadline)
                order = np.lexsort((candidates, -baseline))
                # Verify repeatability on the same public prefix and item subset.
                identity = self._scores(prefix, candidates[:32], maxlen, deadline)
                np.testing.assert_allclose(
                    identity, baseline[:32], rtol=1e-5, atol=1e-5
                )
                scaled_order = np.lexsort(
                    (candidates, -(baseline.astype(np.float64) * 2.0))
                )
                if not np.array_equal(order, scaled_order):
                    raise ValueError(
                        "Positive score rescaling changed diagnostic ranking"
                    )
                length_bin = "8-31" if n < 32 else "32-63" if n < 64 else "64+"
                user_rows = []
                cache = {
                    (): {
                        "top10_disruption": 0.0,
                        "original_top_displacement": 0.0,
                        "original_pair_gap_change": 0.0,
                    }
                }
                for contrast, target, controls in plans:
                    changes = []
                    for removed in [target] + controls:
                        key = tuple(removed)
                        if key not in cache:
                            removed_set = set(removed)
                            altered = [
                                item
                                for i, item in enumerate(prefix)
                                if i not in removed_set
                            ]
                            scores = self._scores(altered, candidates, maxlen, deadline)
                            cache[key] = self._rank_change(
                                baseline, scores, candidates, order
                            )
                        changes.append(cache[key])
                    record = {
                        "user": str(user),
                        "contrast": contrast,
                        "length_bin": length_bin,
                    }
                    for field in changes[0]:
                        record["target_" + field] = changes[0][field]
                        record["excess_" + field] = changes[0][field] - float(
                            np.mean([c[field] for c in changes[1:]])
                        )
                    user_rows.append(record)
                for name, shuffled in permutations.items():
                    scores = self._scores(shuffled, candidates, maxlen, deadline)
                    change = self._rank_change(baseline, scores, candidates, order)
                    user_rows.append(
                        {
                            "user": str(user),
                            "contrast": name,
                            "length_bin": length_bin,
                            **{"target_" + key: value for key, value in change.items()},
                        }
                    )
                # Commit complete users only; keep their controls together.
                records.extend(user_rows)
                for record in user_rows:
                    coverage[record["contrast"] + ":completed_users"] += 1
                rows.append(
                    {
                        "user": str(user),
                        "prefix_length": n,
                        "partition_counts": {k: len(v) for k, v in groups.items()},
                    }
                )
                completed += 1
            except TimeoutError:
                budget_stopped = True
                break
        return {
            "available": bool(records),
            "dataset": str(getattr(self.view, "dataset_name", "current_dataset")),
            "eligible_users": len(eligible),
            "selected_users": len(selected),
            "completed_users": completed,
            "budget_stopped": budget_stopped,
            "coverage": dict(coverage),
            "frequency_quartile_edges": edges,
            "lexical_metadata": lexical_info,
            "summaries": self._summaries(records),
            "user_records": records,
            "sampled_user_partitions": rows,
            "fixed_slot_masking": {
                "available": False,
                "reason": "Internal zero tokens are not verified attention-key exclusions; compact deletion only",
            },
            "nomination": {
                "status": "inconclusive_pending_four_dataset_aggregation",
                "rule": "Positive mean excess top10 disruption for recurrent_only compact deletion in at least three datasets, each with at least 16 completed exactly matched users out of up to 32 sampled",
                "automatic_model_change": False,
            },
        }

    def _independent_margin_loader(self, deadline):
        """Training-only B x L batches; never inspect the live loader."""
        import time

        histories, _ = self._training_histories()
        itemnum = int(self.model.item_emb.num_embeddings - 1)
        maxlen = int(getattr(self.module, "maxlen", 128))
        if maxlen < 1 or itemnum < 1:
            raise ValueError("Margin probes require positive maxlen and itemnum")
        eligible = []
        self.margin_batches_completed = 0
        for user in sorted(histories, key=str):
            if time.monotonic() >= deadline:
                return
            history = tuple(int(i) for i in histories[user])
            if any(i < 1 or i > itemnum for i in history):
                raise ValueError("Out-of-catalog training history")
            if len(history) >= 2 and len(set(history)) < itemnum:
                eligible.append(history)
        if not eligible:
            return
        user_rng = random.Random(42)
        negative_rng = np.random.default_rng(42)
        for _ in range(10):
            if time.monotonic() >= deadline:
                return
            seq = np.zeros((32, maxlen), dtype=np.int64)
            pos = np.zeros_like(seq)
            neg = np.zeros_like(seq)
            for row in range(32):
                history = user_rng.choice(eligible)
                inputs = history[:-1][-maxlen:]
                targets = history[1:][-maxlen:]
                observed = np.asarray(sorted(set(history)), dtype=np.int64)
                shifted = observed - np.arange(len(observed), dtype=np.int64) - 1
                ranks = negative_rng.integers(
                    0, itemnum - len(observed), size=len(targets)
                )
                negatives = ranks + 1 + np.searchsorted(shifted, ranks, side="right")
                seq[row, -len(inputs) :] = inputs
                pos[row, -len(targets) :] = targets
                neg[row, -len(targets) :] = negatives
            if time.monotonic() >= deadline:
                return
            yield (
                None,
                torch.as_tensor(seq, device=self.device),
                torch.as_tensor(pos, device=self.device),
                torch.as_tensor(neg, device=self.device),
            )
            # Resumption occurs only after the consumer successfully scores this batch.
            self.margin_batches_completed += 1

    @staticmethod
    def _digest(value):
        import hashlib

        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _panel_recommendations(self, deadline):
        """Five real training users, with explicit coverage and frozen provenance."""
        import time

        if time.monotonic() >= deadline:
            raise TimeoutError("Panel allocation exhausted before preprocessing")
        histories, _ = self._training_histories()
        itemnum = int(self.model.item_emb.num_embeddings - 1)
        maxlen = int(getattr(self.module, "maxlen", 128))
        if maxlen < 1:
            raise ValueError("model.maxlen must be positive")
        normalized = {}
        for user in sorted(histories, key=str):
            if time.monotonic() >= deadline:
                raise TimeoutError("Panel allocation exhausted during preprocessing")
            key = str(user)
            if key in normalized:
                raise ValueError("Duplicate normalized training-user ID")
            history = tuple(int(i) for i in histories[user])
            if any(i < 1 or i > itemnum for i in history):
                raise ValueError("Out-of-catalog training history")
            normalized[key] = history
        eligible = sorted(
            user
            for user, history in normalized.items()
            if history and itemnum - len(set(history)) >= 10
        )
        panel, source = None, None
        for name in ("diagnostic_user_ids", "simulator_user_ids", "audit_user_ids"):
            supplied = getattr(self.view, name, None)
            if supplied is not None:
                source = name
                if isinstance(supplied, dict):
                    supplied = supplied.get(self.dataset)
                    if supplied is None:
                        panel = []
                        break
                if isinstance(supplied, (str, bytes)) or not isinstance(
                    supplied, (list, tuple, np.ndarray)
                ):
                    raise ValueError(
                        "Supplied panel must be an ordered sequence of user IDs"
                    )
                panel = [str(user) for user in supplied]
                break
        if panel is None:
            source = "sorted_eligible_training_users_local_seed_42"
            panel = random.Random(42).sample(eligible, min(5, len(eligible)))
        failures, records = [], []
        valid_panel = len(panel) == 5 and len(set(panel)) == 5
        if not valid_panel:
            failures.append("Panel must contain exactly five distinct users")
        if self.dataset not in self.routes:
            failures.append("Missing or invalid explicit dataset route")
        if valid_panel:
            for user in panel:
                if user not in eligible:
                    failures.append(
                        {"user": user, "reason": "ineligible_training_user"}
                    )
                    continue
                full = normalized[user]
                candidates = np.setdiff1d(
                    np.arange(1, itemnum + 1, dtype=np.int64),
                    np.asarray(sorted(set(full)), dtype=np.int64),
                    assume_unique=True,
                )
                try:
                    scores = self._scores(full[-maxlen:], candidates, maxlen, deadline)
                    order = np.lexsort((candidates, -scores))
                    # Reverse candidate order to verify item/score alignment.
                    subset = candidates[:32].copy()
                    repeated = self._scores(full[-maxlen:], subset, maxlen, deadline)
                    reversed_scores = self._scores(
                        full[-maxlen:], subset[::-1].copy(), maxlen, deadline
                    )
                    np.testing.assert_allclose(
                        repeated, scores[: len(subset)], rtol=1e-5, atol=1e-5
                    )
                    np.testing.assert_allclose(
                        reversed_scores[::-1], repeated, rtol=1e-5, atol=1e-5
                    )
                    records.append(
                        {
                            "user": user,
                            "history_length": len(full),
                            "history_sha256": self._digest(full),
                            "candidate_count": len(candidates),
                            "candidate_sha256": self._digest(candidates.tolist()),
                            "top10": candidates[order[:10]].tolist(),
                            "identity_scores_close": True,
                            "candidate_permutation_scores_close": True,
                        }
                    )
                except (TimeoutError, ValueError, RuntimeError, AssertionError) as exc:
                    failures.append(
                        {"user": user, "reason": type(exc).__name__, "detail": str(exc)}
                    )
        import hashlib

        checkpoint = hashlib.sha256()
        for name, tensor in sorted(self.module.state_dict().items()):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Panel allocation exhausted during checkpoint hashing"
                )
            checkpoint.update(name.encode("utf-8"))
            checkpoint.update(str((tensor.dtype, tuple(tensor.shape))).encode("utf-8"))
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            checkpoint.update(raw.numpy().tobytes())
        mapping = getattr(self.view, "item_mapping", None)
        return {
            "dataset": self.dataset,
            "source": source,
            "user_ids": panel,
            "eligible_users": len(eligible),
            "completed_users": len(records),
            "coverage_complete": valid_panel and len(records) == 5 and not failures,
            "records": records,
            "failures": failures,
            "training_history_sha256": self._digest(normalized),
            "checkpoint_sha256": checkpoint.hexdigest(),
            "item_mapping_sha256": (
                self._digest(sorted((str(k), str(v)) for k, v in mapping.items()))
                if isinstance(mapping, dict)
                else None
            ),
            "mapping_status": (
                "supplied" if isinstance(mapping, dict) else "unavailable"
            ),
        }

    def run_full_diagnosis(self, train_loader):
        import os
        import time

        # Exclude outstanding training kernels from diagnostic runtime accounting.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        module_modes = [(module, module.training) for module in self.module.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.get_rng_state().clone()
        cuda_state = (
            torch.cuda.get_rng_state(self.device).clone()
            if self.device.type == "cuda"
            else None
        )
        # Shared search accounting belongs to the host. Never infer remaining
        # budget from the original 43,200-second allocation after prior trials.
        supplied = getattr(self.view, "remaining_gpu_seconds", None)
        if supplied is None:
            supplied = os.environ.get("SELF_EVOLVEREC_REMAINING_GPU_SECONDS")
        remaining = None if supplied is None else float(supplied)
        if remaining is not None and (not np.isfinite(remaining) or remaining < 0):
            raise ValueError("remaining_gpu_seconds must be finite and nonnegative")
        # A bounded local allocation is not an assertion about the global balance.
        # The host must charge actual cost even when no balance was supplied.
        allocation = 60.0 if remaining is None else min(60.0, 0.05 * remaining)
        deadline = started + allocation
        try:
            self.module.eval()
            with torch.no_grad():
                try:
                    panel = self._panel_recommendations(deadline)
                except (TimeoutError, ValueError, RuntimeError, AssertionError) as exc:
                    panel = {
                        "dataset": self.dataset,
                        "coverage_complete": False,
                        "completed_users": 0,
                        "failures": [
                            {"reason": type(exc).__name__, "detail": str(exc)}
                        ],
                    }
                collapse_score = None
                if deadline is None or time.monotonic() < deadline:
                    collapse_score = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048
                    )
                eps = 0.5

                # train_loader and its workers remain completely untouched.
                def bounded_loader():
                    yield from self._independent_margin_loader(deadline)

                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        bounded_loader(), num_batches=10, eps=eps, topk_categories=20
                    )
                )
                if time.monotonic() >= deadline:
                    perturbations = {
                        "available": False,
                        "budget_stopped": True,
                        "reason": "Diagnostic allocation exhausted before perturbations",
                    }
                else:
                    try:
                        perturbations = self._perturbations(deadline)
                    except (
                        TimeoutError,
                        ValueError,
                        RuntimeError,
                        AssertionError,
                        ImportError,
                    ) as exc:
                        perturbations = {
                            "available": False,
                            "budget_stopped": isinstance(exc, TimeoutError),
                            "reason": type(exc).__name__,
                            "detail": str(exc),
                        }
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.monotonic() - started
            # Use integer counts rather than the rounded display metric at the cutoff.
            positions = breakdown["valid_positions"]
            wrong = breakdown["wrong_rank_total_positions"]
            outranking = wrong / positions if positions else None
            complete_margins = getattr(self, "margin_batches_completed", 0) == 10
            evidence_valid = (
                panel["coverage_complete"]
                and complete_margins
                and positions > 0
                and outranking is not None
                and np.isfinite(outranking)
            )
            priority = (
                ["sampling", "residual"]
                if evidence_valid and outranking > 0.15
                else ["residual", "sampling"] if evidence_valid else ["baseline"]
            )
            return {
                "metrics": {
                    "training_panel": panel,
                    "frozen_guidance": {
                        "manifest_version": "training_diagnostics_policy_v1",
                        "dataset": self.dataset,
                        "threshold": 0.15,
                        "threshold_operator": ">",
                        "observed_outranking_rate_unrounded": outranking,
                        "margin_batches_completed": getattr(
                            self, "margin_batches_completed", 0
                        ),
                        "required_margin_batches": 10,
                        "priority": priority,
                        "evidence_valid": bool(evidence_valid),
                        "requires_external_four_route_coverage": True,
                        "automatic_training_or_selection": False,
                        "configuration_manifest": {
                            "baseline": "Common prespecified SASRec incumbent",
                            "residual": {
                                "window": 5,
                                "pool": "causal nonpadding mean including current input",
                                "fusion": "h + sigmoid(b) * LayerNorm(pool)",
                                "initial_b": -4.0,
                                "layernorm_scale": 1.0,
                                "layernorm_bias": 0.0,
                                "mask_padding": True,
                            },
                            "sampling": {
                                "rho": 0.25,
                                "negative_count": "unchanged from common baseline",
                                "hard_pool": "training adjacent successors minus complete user training history, target and padding",
                                "hard_draw": "uniform",
                                "fallback": "uniform eligible catalog",
                                "empty_eligible": "skip and count",
                            },
                        },
                        "scope": "Diagnostic ordering proposal only; manifest actions require host-side training implementations and matched budgets before comparison",
                        "selection": "Only pinned host macro validation HR@10",
                    },
                    "live_training_loader_consumed": False,
                    "allocated_single_gpu_seconds": (
                        elapsed if self.device.type == "cuda" else 0.0
                    ),
                    "embedding_collapse_score": collapse_score,
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "history_perturbations": perturbations,
                    "diagnostic_runtime_seconds": elapsed,
                    "diagnostic_allocation_seconds": allocation,
                    "allocation_exceeded": (
                        None if allocation is None else elapsed > allocation
                    ),
                },
                "metric_definitions": {
                    "training_panel": "Five distinct real training users per explicit route; supplied ordered IDs take precedence over deterministic seed-42 fallback. Full-catalog candidates exclude complete training histories. Coverage requires five completed recommendations and identity/permutation checks. Verify all four routes externally before claiming twenty-user coverage.",
                    "frozen_guidance": "Ordering proposal only. Require complete panel coverage and all ten independent margin batches; otherwise abstain to baseline. Compare the unrounded wrong-rank count divided by valid-position count strictly against 0.15. Sampling and residual entries describe proposed actions; this diagnostic does not implement their training, random-search/placebo arms, matched ledgers or host evaluation. No comparative performance claim is supported by this report alone.",
                    "live_training_loader_consumed": "False: independent diagnostic batches use public training histories and local generators; the supplied live iterator is untouched.",
                    "allocated_single_gpu_seconds": "Synchronized diagnostic wall duration on CUDA, including CPU preparation; charge actual consumption and overruns in the host's global ledger.",
                    "embedding_collapse_score": "(1 + sampled mean off-diagonal item-embedding cosine)/2, accumulated in float32. Null when fewer than two items can be sampled or the allocation expires before this probe. Approximately 0.5 for orthogonal vectors; directional alignment alone does not measure recommendation diversity or establish representation collapse.",
                    "pos_neg_margin_mean": "Mean positive-minus-one-sampled-negative score over nonpadding sequence/positive/negative positions. Score-scale dependent; not full-catalog ranking quality.",
                    "pos_neg_margin_p10": "Nearest-rank 10th percentile of those sampled margins. Null when no valid positions were observed.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid sampled margins below zero; ties do not count. Up to ten independent batches of 32 eligible training users sampled uniformly with replacement; latest maxlen teacher-forced transitions; one uniform negative per position excluding the user's complete training history. Local seed-42 user and negative generators are independent. Allocation truncation is explicit through valid_positions. This is training fit, not validation HR@10.",
                    "pos_neg_small_margin_rate": "Fraction of valid sampled margins below eps=0.5; changes under positive score rescaling.",
                    "pos_neg_margin_category_breakdown": "Item categories are unavailable. Category lists remain empty; totals still count valid sampled positions.",
                    "history_perturbations": {
                        "sample": "Seed 42; up to 32 training users per dataset with at least eight interactions after truncating complete training histories to model.maxlen. No loader targets or held-out events reconstruct prefixes.",
                        "groups": "Recent means last ceil(n/4) positions. Recurrent means duplicate item ID or cosine >=0.5 to another prefix position using training-observed title/description TF-IDF. Lowercase Unicode-accent-stripped word unigrams of length >=2, min_df=1, max_features=20000, L2 normalization. Missing text provides no lexical match. Operational similarity does not establish shared intent.",
                        "partition": "Recent-only, recurrent-only, overlap, neither. At most two positions are deleted, retaining at least two interactions. Overlap is joint sensitivity; isolated_nonrecent compares nonrecent nonrecurrent targets with nonrecent recurrent controls.",
                        "matching": "Five independent control draws, without replacement inside each draw; reuse across draws is allowed. Recurrence, overlap and isolated contrasts exactly match relative-position quartile floor(4*i/n), elapsed-time bin <=7, >7-30, >30-180, >180 days or missing, training-frequency quartile and final-item status. Recent-only matches frequency quartile and recurrence status against nonrecent positions; recency, position and time remain inseparable. Infeasible strata are skipped without relaxation. Frequency uses all training interactions; saved quartile edges use right-sided assignment.",
                        "operations": "Compact deletion and left repadding using existing absolute model positions. Fixed-slot and final-position masking are unavailable without verified attention and readout semantics. Shuffle controls permute only nonpadding positions; one fixes the last item.",
                        "candidates": "One fixed full catalog minus ALL original observed training items per user, including truncated-away items. Removed items never become eligible. No validation/test events are consulted. Scores are chunked; descending score ties use ascending item ID.",
                        "top10_disruption": "1 minus intersection(original top10, changed top10)/10. Excess subtracts the mean of five matched-control disruptions. This is sensitivity, not accuracy or a causal preference effect.",
                        "original_top_displacement": "Changed zero-based rank of the original top item minus its original rank, zero. Excess subtracts the matched-control mean.",
                        "original_pair_gap_change": "Change in score(original top item)-score(original runner-up), always on those same two items. Excess subtracts matched-control changes. Score-scale dependent; rank diagnostics are invariant to positive rescaling.",
                        "aggregation": "Average within user, then across users within the current dataset, with prefix-length strata 8-31, 32-63, 64+. Seed-42 percentile bootstrap with 500 user resamples; all controls stay with their user. Fewer than two users gives no interval. Intervals are descriptive, conditional on diagnostic completion, and do not cover training-seed variation or adaptive validation reuse.",
                        "coverage": "Attempted, empty, infeasible and completed counts are separate. Budget-interrupted users contribute no partial contrasts. Fine exact strata, particularly final-item matching, can make coverage sparse.",
                        "selection": "Supplementary full-catalog diagnostics never select checkpoints or modify the model. Four-dataset nomination needs external aggregation; no experiment or comparative validation result is fabricated. Budget-matched baseline/comparator trials and pinned macro HR@10 selection remain host orchestration responsibilities.",
                    },
                    "diagnostic_runtime_seconds": "Measured synchronized wall time; the host must charge actual elapsed cost. Local allocation is min(60 seconds, 5% supplied remaining budget), or a bounded 60 seconds if the global balance is unknown. This fallback does not establish available global budget. Five-user full-catalog coverage runs first, followed by embedding, independent margin and optional perturbation probes. Checks occur between operations; overruns are reported and charged. No global search or equal-arm comparison is executed by this probe.",
                    "random_ranking_reference": "Independent continuous random scores give expected pinned HR@10=0.10 for 100 candidates. This is a theoretical reference, not an observed metric.",
                    "unobserved_information": "No review content, ratings, item categories, prices, stores, ownership or simulator diversity/conformity traits are used or inferred.",
                },
            }
        finally:
            for module, training in module_modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, self.device)

    ### <<< Self_EvolveRec-BLOCK-END
