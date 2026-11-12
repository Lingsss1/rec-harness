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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret controlled replacement effects and prospective hypotheses
    - Unavailable/null measurements are missing evidence, never healthy zeros.
    - Training-prefix results are in-sample; the reserved targets may have been used during fitting. These are not validation scores.
    - Interpret replacement results using coverage, matching failures, identity checks, overlap, actual changed positions, count imbalance, and cosine distributions.
    - Shared substitutes estimate effects on common replacement support, not each policy's full marginal distribution. Matching controls measured variables only.
    - Compare replacement policies using the reported common-user contrasts, not differences between policy-specific populations.
    - Swaps preserve the history multiset and anchor but generally change twice as many positions as replacements. They are descriptive order controls, not equal-severity replacements.
    - Sensitivity does not identify purchase intent, compatibility, simulator traits, or an isolated HSTU attention mechanism.
    - Bootstrap intervals condition on fitted weights, fixed candidates, and the corruption design; they exclude training-seed variation, candidate variation, and repeated-selection optimism.
    - Do not infer unavailable metadata or a four-category macro result from a single-category report. Raw margins cannot be compared across fitted models.
    - Include exactly one prospective training hypothesis as the final key_implications entry, explicitly labeled prospective, with one intervention inside the jointly trained HSTU, its matched control, and the predeclared host macro validation HR@10 comparison. Do not claim it has been tested.
    # Review checks: statically traced selection, candidate exclusion, replacement,
    # scoring, aggregation and RNG restoration. No execution or training performed.
    # Architecture/checkpoints unchanged; no measured validation gain claimed.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the host view and distinguish its trained module
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = device
        # DEBUG: Metadata may be unavailable; preserve the host interfaces while
        # accepting both direct user-history mappings and the History wrapper.
        from collections.abc import Mapping

        self.meta_data = view.meta_data if view.meta_data is not None else {}
        self.review_data = view.review_data
        self.user_train = view.user_train
        if not isinstance(self.meta_data, Mapping):
            raise TypeError("meta_data must be an item-ID mapping or None")
        if not isinstance(self.user_train, Mapping):
            raise TypeError("user_train must be a mapping")
        self.histories = self.user_train.get("History", self.user_train)
        if not isinstance(self.histories, Mapping):
            raise TypeError("Training histories must be a user-to-sequence mapping")
        if not self.histories:
            raise ValueError("Diagnosis requires nonempty public training histories")

    def _get_categories(self, item_id: int):
        # Item categories are explicitly unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Bounded raw cosine without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            n = min(int(self.network.itemnum), emb.shape[0] - 1)
            rng = np.random.default_rng(seed)
            ids = rng.choice(n, size=min(sample_size, n), replace=False) + 1
            indices = torch.as_tensor(ids, device=emb.device, dtype=torch.long)
            x = emb.index_select(0, indices).to(device="cpu", dtype=torch.float64)
            norms = x.norm(dim=-1)
            usable = torch.isfinite(x).all(dim=-1) & (norms > 1e-12)
            x = x[usable]
            if len(x) < 2:
                return None
            x = F.normalize(x, p=2, dim=-1, eps=1e-12)
            cosine = (x.sum(dim=0).square().sum() - len(x)) / (len(x) * (len(x) - 1))
            return float(cosine.clamp(-1, 1))

    ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnosis wrapper manage and restore module mode
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

            ### >>> Self_EvolveRec-BLOCK-START: HSTU tensor input compatibility
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding negatives and nonfinite margins
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0) & torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same valid positions for breakdowns
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize counts for unavailable item categories
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Empty margin samples are unavailable
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "valid_positions": 0,
                "unavailable_reason": "no_finite_positions_with_real_positive_and_negative",
                ### <<< Self_EvolveRec-BLOCK-END
            }
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Disclose margin sample size
                "valid_positions": int(N),
                "unavailable_reason": None,
                "mean": round(float(m.mean().item()), 4),
                ### <<< Self_EvolveRec-BLOCK-END
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            ### >>> Self_EvolveRec-BLOCK-START: Explicit unavailable metadata breakdown
            "category_breakdown_status": "unavailable",
            "category_breakdown_reason": "item_categories_not_available_in_matched_task",
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

    ### >>> Self_EvolveRec-BLOCK-START: Controlled HSTU replacements with isolated candidates and overlap-aware pairing
    CATEGORIES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )
    POLICIES = ("uniform", "frequency", "frequency_plus_semantic")
    EFFECTS = ("delta_hr", "delta_rank", "delta_margin", "top10_overlap")

    @staticmethod
    def _distribution(values):
        x = np.asarray([v for v in values if v is not None], dtype=np.float64)
        x = x[np.isfinite(x)]
        return {
            "n": int(len(x)),
            "mean": float(x.mean()) if len(x) else None,
            "p10": float(np.percentile(x, 10)) if len(x) else None,
            "p50": float(np.percentile(x, 50)) if len(x) else None,
            "p90": float(np.percentile(x, 90)) if len(x) else None,
        }

    @staticmethod
    def _bootstrap(values, indices):
        # Resample whole users, never individual corruption draws.
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        denominators = valid[indices].sum(axis=1)
        numerators = np.where(valid, values, 0.0)[indices].sum(axis=1)
        boot = np.full(numerators.shape, np.nan, dtype=np.float64)
        np.divide(numerators, denominators, out=boot, where=denominators > 0)
        estimates = {}
        for j, name in enumerate(DiagnosisProbe.EFFECTS):
            observed = values[valid[:, j], j]
            replicas = boot[np.isfinite(boot[:, j]), j]
            estimates[name] = {
                "mean": float(observed.mean()) if len(observed) else None,
                "ci95": (
                    np.percentile(replicas, [2.5, 97.5]).tolist()
                    if len(replicas)
                    else None
                ),
                "valid_bootstrap_replicates": int(len(replicas)),
            }
        return estimates, boot

    @classmethod
    def _effect_matrices(cls, records):
        # One row per selected user, including users with failed identity checks.
        # Policy contrasts use the same user AND draw for all three policies.
        keys = sorted({key for r in records for key in r["cells"]})
        matrices = {}
        for key in keys:
            values = np.full((len(records), len(cls.EFFECTS)), np.nan)
            for i, record in enumerate(records):
                cell = record["cells"].get(key)
                if cell and cell["effects"] is not None:
                    values[i] = cell["effects"]
            matrices[key] = values
        return matrices

    @classmethod
    def _paired_summary(cls, records):
        if not records:
            return {
                "selected_users": 0,
                "conditions": {},
                "reason": "no_eligible_users",
            }
        indices = np.random.default_rng(42).integers(
            0, len(records), (2000, len(records))
        )
        conditions = {}
        for key, values in cls._effect_matrices(records).items():
            estimates, _ = cls._bootstrap(values, indices)
            conditions[key] = {
                "eligible_users": int(np.isfinite(values).all(axis=1).sum()),
                "selected_users": len(records),
                "estimates": estimates,
            }
        return {
            "selected_users": len(records),
            "conditions": conditions,
            "reason": None,
        }

    @classmethod
    def aggregate_categories(cls, reports):
        """Consume only explicitly supplied, current-trial category reports."""
        missing = [c for c in cls.CATEGORIES if c not in reports]
        if missing:
            return {
                "value": None,
                "reason": "missing_category_reports",
                "missing": missing,
            }
        groups = {}
        for category in cls.CATEGORIES:
            probe = reports[category]["metrics"]["paired_recent_context"]
            if probe.get("category") != category:
                return {"value": None, "reason": "category_identity_not_verified"}
            groups[category] = probe["records"]
        if any(not rows for rows in groups.values()):
            return {"value": None, "reason": "category_without_eligible_users"}
        matrices = {c: cls._effect_matrices(rows) for c, rows in groups.items()}
        keys = sorted(set().union(*(set(v) for v in matrices.values())))
        rng = np.random.default_rng(42)
        indices = {
            c: rng.integers(0, len(rows), (2000, len(rows)))
            for c, rows in groups.items()
        }
        output = {}
        for key in keys:
            if any(
                key not in matrices[c]
                or not np.isfinite(matrices[c][key]).all(axis=1).any()
                for c in cls.CATEGORIES
            ):
                output[key] = {"value": None, "reason": "missing_category_cell_support"}
                continue
            means, boots, coverage = [], [], {}
            for category in cls.CATEGORIES:
                values = matrices[category][key]
                estimates, boot = cls._bootstrap(values, indices[category])
                means.append([estimates[e]["mean"] for e in cls.EFFECTS])
                boots.append(boot)
                coverage[category] = int(np.isfinite(values).all(axis=1).sum())
            mean = np.mean(means, axis=0)
            macro_boot = np.mean(boots, axis=0)
            estimates = {}
            for j, effect in enumerate(cls.EFFECTS):
                valid = macro_boot[np.isfinite(macro_boot[:, j]), j]
                estimates[effect] = {
                    "mean": float(mean[j]),
                    "ci95": (
                        np.percentile(valid, [2.5, 97.5]).tolist()
                        if len(valid)
                        else None
                    ),
                    "valid_bootstrap_replicates": int(len(valid)),
                }
            # Raw margins have model-dependent scales; do not macro-average them.
            estimates.pop("delta_margin")
            output[key] = {
                "estimates": estimates,
                "eligible_users": coverage,
                "reason": None,
            }
        return {"value": output, "reason": None, "scope": "in_sample_training_prefix"}

    def _training_catalog(self, histories):
        # DEBUG: Compute sparse TF-IDF using the standard library so diagnosis
        # does not require scikit-learn or SciPy in the host environment.
        import math
        import re

        itemnum = int(self.network.itemnum)
        counts = Counter()
        for history in histories.values():
            counts.update(int(i) for i in history if 1 <= int(i) <= itemnum)
        catalog = np.asarray(sorted(counts), dtype=np.int64)
        bins = {i: int(counts[i]).bit_length() - 1 for i in counts}
        pools = {
            b: np.asarray([i for i in catalog if bins[int(i)] == b], dtype=np.int64)
            for b in sorted(set(bins.values()))
        }
        texts = []
        for item in catalog:
            entry = self.meta_data.get(
                int(item), self.meta_data.get(str(int(item)), {})
            )
            entry = entry if isinstance(entry, dict) else {}
            parts = []
            for field in ("title", "description"):
                value = entry.get(field, "")
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, (list, tuple)):
                    parts.extend(v for v in value if isinstance(v, str))
            texts.append(" ".join(parts))
        # DEBUG: Match lowercase word unigram/bigram TF-IDF, smooth IDF,
        # sublinear term frequency and L2 normalization without dense matrices.
        vectors, text_reason = None, None
        has_text = np.zeros(len(catalog), dtype=bool)
        if len(catalog):
            token_pattern = re.compile(r"(?u)\b\w\w+\b")
            term_counts = []
            document_frequency = Counter()
            for text in texts:
                tokens = token_pattern.findall(text.lower())
                terms = Counter((token,) for token in tokens)
                terms.update(zip(tokens, tokens[1:]))
                term_counts.append(terms)
                document_frequency.update(terms.keys())
            if document_frequency:
                document_count = len(texts)
                idf = {
                    term: math.log((1.0 + document_count) / (1.0 + frequency)) + 1.0
                    for term, frequency in document_frequency.items()
                }
                documents, postings = [], defaultdict(list)
                for row, terms in enumerate(term_counts):
                    weights = {
                        term: (1.0 + math.log(count)) * idf[term]
                        for term, count in terms.items()
                    }
                    norm = math.sqrt(sum(value * value for value in weights.values()))
                    if norm > 0.0:
                        weights = {
                            term: value / norm for term, value in weights.items()
                        }
                        has_text[row] = True
                    documents.append(weights)
                    for term, value in weights.items():
                        postings[term].append((row, value))
                vectors = {"documents": documents, "postings": postings}
            else:
                text_reason = "empty_training_text_vocabulary"
        else:
            text_reason = "empty_training_catalog"
        rows = {int(item): j for j, item in enumerate(catalog)}
        return counts, catalog, bins, pools, rows, vectors, has_text, text_reason

    @torch.no_grad()
    def _score_sequences(self, sequences, candidates, anchor):
        output = []
        for start in range(0, len(sequences), 32):
            seq = torch.as_tensor(
                np.asarray(sequences[start : start + 32]),
                dtype=torch.long,
                device=self.device,
            )
            positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
            final = positions.masked_fill(seq == 0, -1).amax(dim=1)
            if not bool((final == anchor).all()):
                raise ValueError("Diagnostic corruption moved the nonpadding anchor")
            features = self.model.log2feats(seq)
            features = features[torch.arange(len(seq), device=seq.device), final]
            items = torch.as_tensor(candidates, dtype=torch.long, device=seq.device)
            scores = self.model.predict_score(
                features, items[None].expand(len(seq), -1)
            )
            if scores.shape != (len(seq), 100):
                raise ValueError("Diagnostic scorer must return [batch, 100]")
            output.append(scores.detach().double().cpu().numpy())
        return np.concatenate(output, axis=0)

    @staticmethod
    def _score_metrics(scores, candidates, clean=None):
        order = np.lexsort((np.asarray(candidates), -scores))
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        top = [int(candidates[j]) for j in order[:10]]
        result = {
            "rank": rank,
            "hit": int(rank <= 10),
            "margin": float(scores[0] - scores[1:].max()),
            "top10": top,
        }
        if clean is not None:
            result.update(
                delta_hr=result["hit"] - clean["hit"],
                delta_rank=rank - clean["rank"],
                delta_margin=result["margin"] - clean["margin"],
                top10_overlap=len(set(top) & set(clean["top10"])) / 10.0,
            )
        return result

    @torch.no_grad()
    def _paired_recent_context(self):
        import os
        import hashlib

        # DEBUG: Use the validated host history mapping; a direct mapping must
        # not silently produce an empty diagnostic and unavailable measurements.
        histories = self.histories
        counts, catalog, bins, bin_pools, rows, vectors, has_text, text_reason = (
            self._training_catalog(histories)
        )
        category = getattr(self.model, "dataset_name", None)
        if category not in self.CATEGORIES:
            category = getattr(self.network, "dataset_name", None)
        category = category if category in self.CATEGORIES else None
        maxlen, itemnum = int(self.network.maxlen), int(self.network.itemnum)
        users = sorted(histories, key=str)
        selection_rng = np.random.default_rng(42)
        candidate_rng = np.random.default_rng(42)
        corruption_seeds = np.random.SeedSequence(42).spawn(200)
        records, selected, reasons = [], [], Counter()
        considered = 0
        # Select structurally, before observing scores or replacement feasibility.
        for index in selection_rng.permutation(len(users)):
            if len(selected) == 200:
                break
            considered += 1
            user = users[int(index)]
            full = list(map(int, histories[user]))
            if any(i < 1 or i > itemnum for i in full):
                reasons["invalid_training_item_id"] += 1
                continue
            history = full[:-1][-maxlen:]
            if len(history) < 5:
                reasons["fewer_than_four_preceding_positions"] += 1
                continue
            eligible = np.setdiff1d(catalog, np.asarray(full, dtype=np.int64))
            if len(eligible) < 99:
                reasons["fewer_than_99_unseen_training_items"] += 1
                continue
            selected.append((user, full, history))

        for user_index, (user, full, history) in enumerate(selected):
            # Candidate sampling is independent of earlier corruption feasibility.
            rng = np.random.default_rng(corruption_seeds[user_index])
            eligible = np.setdiff1d(catalog, np.asarray(full, dtype=np.int64))
            candidates = [full[-1]] + sorted(
                map(int, candidate_rng.choice(eligible, 99, replace=False))
            )
            excluded = np.asarray(sorted(set(full) | set(candidates)), dtype=np.int64)
            # Cache eligible pools once per user, original item and policy.
            admissible_cache = {}

            def admissible(original, policy):
                if policy not in self.POLICIES:
                    raise ValueError(f"Unknown replacement policy: {policy}")
                key = (None if policy == "uniform" else original, policy)
                if key not in admissible_cache:
                    if policy == "uniform":
                        pool = catalog
                    elif policy == "frequency":
                        pool = bin_pools[bins[original]]
                    else:
                        pool = np.asarray([], dtype=np.int64)
                        if vectors is not None and has_text[rows[original]]:
                            # DEBUG: Accumulate exact sparse cosine products
                            # through an inverted index, without SciPy operators.
                            similarities = defaultdict(float)
                            document = vectors["documents"][rows[original]]
                            for term, weight in document.items():
                                for row, other_weight in vectors["postings"][term]:
                                    similarities[row] += weight * other_weight
                            matching_rows = np.asarray(
                                sorted(
                                    row
                                    for row, similarity in similarities.items()
                                    if similarity >= 0.2
                                ),
                                dtype=np.int64,
                            )
                            ids = catalog[matching_rows]
                            pool = np.intersect1d(
                                ids, bin_pools[bins[original]], assume_unique=True
                            )
                    admissible_cache[key] = np.setdiff1d(
                        pool, excluded, assume_unique=True
                    )
                return admissible_cache[key]

            def cosine(a, b):
                if vectors is None or not has_text[rows[a]] or not has_text[rows[b]]:
                    return None
                # DEBUG: Both documents are L2 normalized, so their sparse dot
                # product is the measured TF-IDF cosine used for matching.
                left = vectors["documents"][rows[a]]
                right = vectors["documents"][rows[b]]
                if len(left) > len(right):
                    left, right = right, left
                return float(
                    sum(weight * right.get(term, 0.0) for term, weight in left.items())
                )

            sequence = np.zeros(maxlen, dtype=np.int64)
            sequence[-len(history) :] = history
            anchor = int(np.flatnonzero(sequence)[-1])
            preceding = np.flatnonzero(sequence)[:-1]
            record = {
                "user_id": str(user),
                "target": full[-1],
                "history": history,
                "full_training_length": len(full),
                "repeat_target": full[-1] in history,
                "anchor": anchor,
                "candidates": candidates,
                "cells": {},
                "attempts": [],
            }
            records.append(record)
            initial = self._score_sequences(
                [sequence, sequence.copy()], candidates, anchor
            )
            if not np.isfinite(initial).all():
                record["failure"] = "nonfinite_clean_or_identity_scores"
                continue
            record["clean_scores"], record["identity_scores"] = initial.tolist()
            record["identity_max_abs_error"] = float(
                np.abs(initial[1] - initial[0]).max()
            )
            record["identity_pass"] = bool(
                np.allclose(initial[0], initial[1], atol=1e-5, rtol=1e-5)
            )
            if not record["identity_pass"]:
                record["failure"] = "identity_control_failed"
                continue
            clean = self._score_metrics(initial[0], candidates)
            record["clean"] = clean
            sequences, pending = [], []
            seen_k = set()
            for fraction in (0.1, 0.3):
                k = max(1, int(np.floor(fraction * len(preceding))))
                if k in seen_k:
                    continue
                seen_k.add(k)
                severity = f"f{int(fraction * 100)}"
                recent = preceding[-k:]
                for draw in range(5):
                    uniform = np.sort(rng.choice(preceding, k, replace=False))
                    matched_parts = []
                    for b, required in sorted(
                        Counter(bins[int(sequence[p])] for p in recent).items()
                    ):
                        available = np.asarray(
                            [p for p in preceding if bins[int(sequence[p])] == b]
                        )
                        if len(available) < required:
                            matched_parts = None
                            break
                        matched_parts.extend(
                            map(int, rng.choice(available, required, replace=False))
                        )
                    matched = None if matched_parts is None else np.sort(matched_parts)
                    for control_name, control in (
                        ("uniform_positions", uniform),
                        ("matched_positions", matched),
                    ):
                        base = f"{severity}/{control_name}"
                        if control is None:
                            record["attempts"].append(
                                {
                                    "base": base,
                                    "draw": draw,
                                    "failure": "position_bin_match_infeasible",
                                }
                            )
                            continue
                        # Shared positions receive the identical substitute in both
                        # arms. Pair remaining positions by bin for matched controls.
                        shared_positions = np.intersect1d(recent, control)
                        ordering = lambda p: (
                            (bins[int(sequence[p])], int(p))
                            if control_name == "matched_positions"
                            else (int(p),)
                        )
                        left = np.asarray(
                            shared_positions.tolist()
                            + sorted(
                                np.setdiff1d(recent, shared_positions), key=ordering
                            ),
                            dtype=np.int64,
                        )
                        right = np.asarray(
                            shared_positions.tolist()
                            + sorted(
                                np.setdiff1d(control, shared_positions), key=ordering
                            ),
                            dtype=np.int64,
                        )
                        originals_a, originals_b = sequence[left], sequence[right]
                        imbalance = [
                            abs(counts[int(a)] - counts[int(b)])
                            for a, b in zip(originals_a, originals_b)
                        ]
                        common = {
                            "base": base,
                            "draw": draw,
                            "k": k,
                            "fraction": fraction,
                            "actual_fraction": k / len(preceding),
                            "preceding_positions": len(preceding),
                            "recent_positions": left.tolist(),
                            "control_positions": right.tolist(),
                            "position_overlap": len(set(left) & set(right)) / k,
                            "original_count_a": [counts[int(a)] for a in originals_a],
                            "original_count_b": [counts[int(b)] for b in originals_b],
                            "original_count_abs_imbalance": imbalance,
                            "original_bin_match_fraction": float(
                                np.mean(
                                    [
                                        bins[int(a)] == bins[int(b)]
                                        for a, b in zip(originals_a, originals_b)
                                    ]
                                )
                            ),
                        }
                        for policy in self.POLICIES:
                            attempt = dict(
                                common, policy=policy, key=f"{base}/{policy}"
                            )
                            record["attempts"].append(attempt)
                            substitutes, used, sizes = [], set(), []
                            for a, b in zip(originals_a, originals_b):
                                pool = np.intersect1d(
                                    admissible(int(a), policy),
                                    admissible(int(b), policy),
                                    assume_unique=True,
                                )
                                if used:
                                    pool = np.setdiff1d(
                                        pool,
                                        np.asarray(sorted(used), dtype=np.int64),
                                        assume_unique=True,
                                    )
                                sizes.append(int(len(pool)))
                                if not len(pool):
                                    attempt["failure"] = (
                                        "empty_common_substitute_support"
                                    )
                                    break
                                substitute = int(rng.choice(pool))
                                substitutes.append(substitute)
                                used.add(substitute)
                            attempt["common_pool_sizes"] = sizes
                            if "failure" in attempt:
                                continue
                            changed_a, changed_b = sequence.copy(), sequence.copy()
                            changed_a[left], changed_b[right] = substitutes, substitutes
                            attempt.update(
                                substitutes=substitutes,
                                substitute_counts=[counts[s] for s in substitutes],
                                cosine_a=[
                                    cosine(int(a), s)
                                    for a, s in zip(originals_a, substitutes)
                                ],
                                cosine_b=[
                                    cosine(int(b), s)
                                    for b, s in zip(originals_b, substitutes)
                                ],
                            )
                            pending.append(
                                (attempt, len(sequences), changed_a, changed_b)
                            )
                            sequences.extend([changed_a, changed_b])
                        # Same disjoint partner positions for both within-history swap arms.
                        partners = np.setdiff1d(preceding, np.union1d(left, right))
                        attempt = dict(common, policy="swap", key=f"{base}/swap")
                        record["attempts"].append(attempt)
                        if len(partners) < k:
                            attempt["failure"] = "insufficient_disjoint_swap_partners"
                            continue
                        partners = np.sort(rng.choice(partners, k, replace=False))
                        changed_a, changed_b = sequence.copy(), sequence.copy()
                        changed_a[left], changed_a[partners] = (
                            sequence[partners],
                            sequence[left],
                        )
                        changed_b[right], changed_b[partners] = (
                            sequence[partners],
                            sequence[right],
                        )
                        attempt["swap_partner_positions"] = partners.tolist()
                        pending.append((attempt, len(sequences), changed_a, changed_b))
                        sequences.extend([changed_a, changed_b])
            if sequences:
                scores = self._score_sequences(sequences, candidates, anchor)
                for attempt, offset, changed_a, changed_b in pending:
                    pair = scores[offset : offset + 2]
                    if not np.isfinite(pair).all():
                        attempt["failure"] = "nonfinite_corruption_scores"
                        continue
                    a = self._score_metrics(pair[0], candidates, clean)
                    b = self._score_metrics(pair[1], candidates, clean)
                    attempt.update(
                        scores_a=pair[0].tolist(),
                        scores_b=pair[1].tolist(),
                        metrics_a=a,
                        metrics_b=b,
                        changed_positions_a=int(
                            np.count_nonzero(changed_a != sequence)
                        ),
                        changed_positions_b=int(
                            np.count_nonzero(changed_b != sequence)
                        ),
                        effects=[a[e] - b[e] for e in self.EFFECTS],
                    )
            # Average draws within a user. Failed draws are retained explicitly.
            keys = sorted({a["key"] for a in record["attempts"] if "key" in a})
            for key in keys:
                attempts = [a for a in record["attempts"] if a.get("key") == key]
                good = [a for a in attempts if "effects" in a]
                record["cells"][key] = {
                    "attempted_draws": len(attempts),
                    "successful_draws": len(good),
                    "effects": (
                        np.mean([a["effects"] for a in good], axis=0).tolist()
                        if good
                        else None
                    ),
                }
            # Stronger comparison: identical user/draw support across all policies.
            for base in sorted({a["base"] for a in record["attempts"]}):
                by_policy = {
                    p: {
                        a["draw"]: a
                        for a in record["attempts"]
                        if a["base"] == base and a.get("policy") == p and "effects" in a
                    }
                    for p in self.POLICIES
                }
                shared = sorted(set.intersection(*(set(v) for v in by_policy.values())))
                for policy in self.POLICIES:
                    effects = [by_policy[policy][d]["effects"] for d in shared]
                    record["cells"][f"{base}/common/{policy}"] = {
                        "successful_draws": len(shared),
                        "draw_ids": shared,
                        "effects": (
                            np.mean(effects, axis=0).tolist() if effects else None
                        ),
                    }
                for first, second in (
                    ("frequency", "uniform"),
                    ("frequency_plus_semantic", "frequency"),
                ):
                    effects = [
                        np.subtract(
                            by_policy[first][d]["effects"],
                            by_policy[second][d]["effects"],
                        )
                        for d in shared
                    ]
                    record["cells"][f"{base}/contrast/{first}_minus_{second}"] = {
                        "successful_draws": len(shared),
                        "draw_ids": shared,
                        "effects": (
                            np.mean(effects, axis=0).tolist() if effects else None
                        ),
                    }
        coverage = {}
        all_attempts = [a for r in records for a in r["attempts"]]
        for key in sorted({a["key"] for a in all_attempts if "key" in a}):
            attempts = [a for a in all_attempts if a.get("key") == key]
            good = [a for a in attempts if "effects" in a]
            flatten = lambda name, source: [v for a in source for v in a.get(name, [])]
            coverage[key] = {
                "attempted_draws": len(attempts),
                "successful_draws": len(good),
                "failures": dict(
                    Counter(a["failure"] for a in attempts if "failure" in a)
                ),
                "position_overlap": self._distribution(
                    [a["position_overlap"] for a in attempts]
                ),
                "original_bin_match_fraction": self._distribution(
                    [a["original_bin_match_fraction"] for a in attempts]
                ),
                "original_counts_a": self._distribution(
                    flatten("original_count_a", good)
                ),
                "original_counts_b": self._distribution(
                    flatten("original_count_b", good)
                ),
                "original_count_abs_imbalance": self._distribution(
                    flatten("original_count_abs_imbalance", good)
                ),
                "substitute_counts": self._distribution(
                    flatten("substitute_counts", good)
                ),
                "cosine_a": self._distribution(flatten("cosine_a", good)),
                "cosine_b": self._distribution(flatten("cosine_b", good)),
                "text_pairs_attempted_per_arm": sum(
                    len(a.get("substitutes", [])) for a in good
                ),
                "changed_positions_a": self._distribution(
                    [a["changed_positions_a"] for a in good]
                ),
                "changed_positions_b": self._distribution(
                    [a["changed_positions_b"] for a in good]
                ),
            }
        result = {
            "category": category,
            "scope": "in_sample_training_prefix",
            "seed": 42,
            "considered_users": considered,
            "selected_users": len(selected),
            "selected_user_ids": [str(u) for u, _, _ in selected],
            "scored_users": sum("clean" in r for r in records),
            "selection_failures": dict(reasons),
            "scoring_failures": dict(
                Counter(r["failure"] for r in records if "failure" in r)
            ),
            "identity_max_abs_error": self._distribution(
                [r.get("identity_max_abs_error") for r in records]
            ),
            "training_catalog_items": len(catalog),
            "text_covered_items": int(has_text.sum()),
            "text_unavailable_reason": text_reason,
            "coverage": coverage,
            "records": records,
            "summary": self._paired_summary(records),
            "four_category_macro": {
                "value": None,
                "reason": "requires_four_explicit_category_reports",
            },
            "authorized_validation": {
                "value": None,
                "reason": "host_cases_not_exposed",
            },
            "predeclared": {
                "prediction": "Recent replacements reduce target HR@10 more than paired random replacements, attenuating after popularity and semantic controls.",
                "prospective_intervention": "Inside the single jointly trained HSTU, apply training-only item replacement to 10 percent of non-anchor occupied input positions, using frequency-bin matched unseen training items; preserve existing causal labels and readout.",
                "matched_control": "Same replacement rate and substitute policy, uniformly selected non-anchor positions versus recent non-anchor positions; same seed, initialization, optimizer, epochs, batches, and checkpoint selection.",
                "validation_rule": "Only a future authorized host evaluation can accept a training change: compare equally weighted four-category validation HR@10 under pinned users/candidates. Retain every attempted configuration; do not select on diagnostic HR or claim a held-out result.",
            },
            "settings": {
                "maximum_users": 200,
                "draws_per_cell": 5,
                "fractions": [0.1, 0.3],
                "forward_batch_size": 32,
                "maximum_sequences_per_user": 162,
                "bootstrap_replicates": 2000,
                "semantic_threshold": 0.2,
                "position_policy": "replace_in_place_preserve_anchor_and_padding",
            },
        }
        fingerprint = hashlib.sha256()
        for user in users:
            fingerprint.update(
                json.dumps([str(user), list(map(int, histories[user]))]).encode()
            )
        fingerprint.update(str(itemnum).encode())
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
        )
        os.makedirs(directory, exist_ok=True)
        label = category or ("unidentified_" + fingerprint.hexdigest()[:16])
        # Never overwrite a prior successful or unsuccessful diagnostic attempt.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=label + "_paired_replacement_",
            suffix=".json",
            dir=directory,
            delete=False,
        ) as stream:
            result["artifact_path"] = stream.name
            json.dump(result, stream, allow_nan=False)
        return result

    def run_full_diagnosis(self, train_loader):
        from itertools import islice

        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        device = torch.device(self.device)
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                for index in cuda_devices:
                    torch.cuda.default_generators[index].manual_seed(42)
                self.network.eval()
                with torch.no_grad():
                    cosine = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048, seed=42
                    )
                    margin, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            islice(train_loader, 10),
                            num_batches=10,
                            eps=0.5,
                            topk_categories=20,
                        )
                    )
                    paired = self._paired_recent_context()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, mode in modes:
                module.training = mode
        return {
            "metrics": {
                "embedding_mean_offdiagonal_cosine": cosine,
                "embedding_cosine_unavailable_reason": (
                    "fewer_than_two_usable_rows" if cosine is None else None
                ),
                "pos_neg_margin_mean": margin["mean"],
                "pos_neg_margin_p10": margin["p10"],
                "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin["small_margin_rate"],
                "pos_neg_margin_valid_positions": margin["valid_positions"],
                "pos_neg_margin_unavailable_reason": margin["unavailable_reason"],
                "pos_neg_margin_category_breakdown": breakdown,
                "paired_recent_context": paired,
            },
            "metric_definitions": {
                "embedding_mean_offdiagonal_cosine": "Mean off-diagonal cosine of up to 2048 uniformly sampled real-item embeddings, seed 42, finite nonzero rows only, float64. Directional similarity is not a calibrated collapse or HR failure measure.",
                "pos_neg_margin_mean": "Mean positive-minus-negative raw score over finite, nonpadding sequence/positive/negative positions; null without usable positions.",
                "pos_neg_margin_p10": "Nearest-rank tenth percentile of the same margins.",
                "pos_neg_neg_beats_pos_rate": "Fraction of usable margins below zero.",
                "pos_neg_small_margin_rate": "Fraction of usable margins below 0.5; score-scale dependent.",
                "pos_neg_margin_valid_positions": "Number of finite nonpadding margin observations, not independent users.",
                "pos_neg_margin_category_breakdown": "Item categories unavailable; lists are empty, while overall position counts are measured.",
                "paired_recent_context": {
                    "selection": "Sorted string IDs and independent seed-42 permutation; at most 200 structurally eligible training users selected before scoring. Last training interaction is target, preceding history truncated once and left-padded. At least four occupied positions before anchor and 99 unseen training catalog items required. Failed scoring never triggers replacement users.",
                    "training_inputs": "Counts include all training interaction occurrences. Catalog includes positive-count training IDs only. Frequency bin=floor(log2(count)). TF-IDF is fitted solely to these IDs' public titles/descriptions, lowercase word unigrams/bigrams, sublinear TF and L2 normalization. No reviews, ratings, categories, prices, stores, external data, or validation/test labels are used.",
                    "design": "For m preceding occupied positions, distinct k=max(1,floor(f*m)), f=0.1,0.3; duplicate k omitted under its later fraction. Five draws. Recent=last k preceding positions. Uniform control samples k preceding positions without replacement; matched control samples exact recent frequency-bin multiplicities. Overlap allowed and measured; matched controls may equal recent positions. All replacements preserve length, padding, absolute positions, and greatest-nonzero anchor readout.",
                    "replacement_support": "Uniform policy admits training catalog items. Frequency requires original frequency bin; frequency_plus_semantic additionally requires TF-IDF cosine >=0.2 and nonzero vectors. Each paired position gets one shared uniform draw from the intersection of both originals' admissible pools after excluding entire original history, all candidates and substitutes already used in that arm pair. Draws are sequential without replacement; empty support fails the whole paired draw without relaxation or retry. This is common-support sensitivity, not either full marginal corruption policy.",
                    "candidates": "Target plus exactly 99 unique uniform training-item negatives outside entire original training history. Fixed for every condition/draw per user. Rank is one-based descending score with ascending item-ID ties. This diagnostic tie rule does not assert host tie behavior.",
                    "identity": "Clean and identity sequences are scored together. Max absolute error is recorded; np.allclose atol=rtol=1e-5 must pass or user effects are unavailable. Raw clean, identity, and successful corruption score vectors are cached in trial-local JSON.",
                    "effects": "Per arm: delta_hr=perturbed hit minus clean hit; delta_rank=perturbed rank minus clean rank; delta_margin=perturbed margin minus clean margin; overlap=perturbed/clean top10 intersection divided by 10. Margin=target score minus maximum negative score. Stored effect vector is recent arm minus control arm in EFFECTS order. Negative delta_hr and positive delta_rank indicate greater recent sensitivity. Raw margins are only within-model quantities, excluded from macro aggregation.",
                    "swaps": "Each selected position swaps with a distinct preceding position outside the union of recent/control selections. Both arms use the same partner positions. Anchor and history multiset stay fixed; actual changed-position counts may be below 2k for repeated IDs. Swap support/severity differs from replacement support and must not be treated as an equal-severity policy contrast.",
                    "coverage": "Attempted/successful draws and failure reasons are retained; distributions summarize count imbalance, shared substitute counts, cosine, bin matching, position overlap, and actual changed positions. Cosine n counts pairs with usable vectors; text_pairs_attempted_per_arm is its denominator for replacement arms. Failed cells are missing, never zero. Per-policy estimates average successful draws within each user, then users equally.",
                    "randomization_and_pairing": "Selection and candidate sampling use separate seed-42 generators. Each selected user has an independent corruption stream derived from SeedSequence(42); earlier users' matching failures cannot change later users' candidates or corruptions. Positions shared by recent/control selections pair with themselves and receive identical substitutes. Remaining matched-control positions pair deterministically within frequency bins; remaining uniform-control positions pair by index. Cached admissible pools exclude history and candidates before sequential substitute sampling.",
                    "common_policy_comparisons": "For each severity/control, retain identical successful user/draw tuples across all three replacement policies. Common cells average those draws per user. Contrast cells subtract each user's paired effects between policies before averaging/bootstrap. Policy-specific populations are reported separately and cannot substitute for these common-support contrasts. Separate policies do not necessarily share their sampled substitutes. Residual cosine/count imbalance must still be inspected; threshold matching does not ensure equal semantic disruption.",
                    "uncertainty": "2000 seed-42 user bootstrap replicates resample all selected user records, carrying every condition/draw together. Unavailable cells remain missing and each replicate averages its observed users. Report valid replicate counts; percentile 95% intervals condition on fitted weights, fixed candidates, selected users, and corruption design, excluding training/candidate variation and selection optimism.",
                    "macro": "aggregate_categories consumes four explicit current-trial reports and verifies category identity. Each replicate independently resamples users within category and equally averages four category estimates. Missing category support yields null for that cell; no zero imputation. Raw score margins are excluded. This is an in-sample diagnostic macro, not host validation HR@10.",
                    "limitations": "Training-prefix targets may already have been fitted. Matching coarse frequency bins and text cosine does not isolate an HSTU mechanism, intent, compatibility, or unobserved traits. Repeated targets and overlap are disclosed. Timestamp inputs are unavailable to this HSTU; no temporal or geometry-isolating causal claim is made.",
                    "cost_and_state": "At most 162 sequences per selected user in forward chunks of 32, plus at most ten margin batches and 2048 embedding rows per category. No checkpoint changes, independently fitted recommendation scores, or fusion. Python/NumPy global, Torch CPU/model-GPU RNG states and all module modes restore even on exceptions. Caller-owned loader generators/workers require a dedicated diagnostic loader. Every completed attempt writes a new trial-local artifact.",
                },
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
