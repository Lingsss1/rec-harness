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
    ### >>> Self_EvolveRec-BLOCK-START: Matched lexical masking with compatible scoring, batched interventions and explicit coverage
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.review_data = getattr(view, "review_data", {}) or {}
        self.user_train = view.user_train
        self.category = getattr(view, "dataset_name", None)
        self.settings = {
            "seed": 42,
            "max_users": 50,
            "recent_events": 20,
            "cosine_threshold": 0.3,
            "controls": 20,
            "bootstrap_replicates": 1000,
            "text_permutations": 20,
            "control_draw_limit": 20000,
            "position_bins": "min(3, floor(4 * original_slot / maxlen))",
            "popularity_bins": "training counts; inverted-CDF quartiles; ties go right",
            "repeat_bins": "exact item frequency in model-truncated history",
            "singleton_rule": "latest group containing exactly one event",
            "target_rule": "largest group, latest member, original positions",
            "selection": "unchanged host macro validation HR@10 only",
        }
        histories = self.user_train.get("History", self.user_train)
        self.histories = {
            str(u): [int(i) for i in h if int(i) > 0]
            for u, h in histories.items()
            if isinstance(h, (list, tuple, np.ndarray))
        }
        self.counts = Counter(i for h in self.histories.values() for i in h)
        self.maxlen = int(getattr(self.model, "maxlen", getattr(view, "maxlen", 128)))
        self.itemnum = int(view.item_emb.weight.shape[0] - 1)
        self._popularity_thresholds = (
            np.quantile(
                list(self.counts.values()), [0.25, 0.5, 0.75], method="inverted_cdf"
            )
            if self.counts
            else None
        )
        # DEBUG: Share the training category's deadline instead of allocating
        # uncharged diagnostic work after the six-configuration search.
        self._deadline = float(getattr(view, "diagnostic_deadline", float("inf")))
        self._cache = {}
        self._hook = getattr(view, "diagnostic_intervention", None)
        required = (
            "item_emb",
            "pos_emb",
            "emb_dropout",
            "attention_layernorms",
            "attention_layers",
            "forward_layernorms",
            "forward_layers",
            "last_layernorm",
            "hidden_units",
            "pooling_mode",
        )
        self._builtin = self.model.__class__.__name__ == "SASRec" and all(
            hasattr(self.model, name) for name in required
        )

    # DEBUG: Deadline checks report incomplete coverage explicitly; they never
    # replace unmeasured effects with zero or select models using diagnostics.
    def _budget_exhausted(self):
        import time

        return time.monotonic() >= self._deadline

    def _get_categories(self, item_id):
        # Item category labels are outside this task's public data boundary.
        return []

    def _features(self, sequence, removed):
        """Hook: (sequence, removed_bool) -> B x L x D, original slots/readout.

        An external hook must remove identity before encoding, forbid removed
        attention keys at every layer, and keep a removed readout as a neutral
        positional query. No deletion fallback is permitted here.
        """
        if self._hook is not None:
            return self._hook(sequence, removed)
        if not self._builtin:
            raise NotImplementedError("Architecture needs diagnostic_intervention")
        m = self.model
        batch, length = sequence.shape
        padding = sequence.eq(0)
        identity = sequence.masked_fill(removed, 0)
        emb = m.item_emb(identity) * (m.hidden_units**0.5)
        # Explicit zeroing also protects against a nonzero embedding row zero.
        emb = emb.masked_fill(removed.unsqueeze(-1), 0)
        positions = torch.arange(length, device=sequence.device)
        x = m.emb_dropout(emb + m.pos_emb(positions)[None])
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        causal = positions[None, :] <= positions[:, None]
        # Preserve baseline padding behavior; only intervened keys are removed.
        allowed = causal[None] & ~removed[:, None, :]
        no_keys = ~allowed.any(-1)
        safe_allowed = allowed.clone()
        safe_allowed[:, :, 0] |= no_keys
        for norm, attention, ffn_norm, ffn in zip(
            m.attention_layernorms,
            m.attention_layers,
            m.forward_layernorms,
            m.forward_layers,
        ):
            source = x.transpose(0, 1)
            query = norm(source)
            mask = (~safe_allowed).repeat_interleave(attention.num_heads, dim=0)
            out, _ = attention(
                query, source, source, attn_mask=mask, need_weights=False
            )
            # Avoid all-masked attention rows rather than multiplying NaNs by zero.
            out = out.masked_fill(no_keys.T.unsqueeze(-1), 0)
            x = ffn(ffn_norm((query + out).transpose(0, 1)))
            x = x.masked_fill(padding.unsqueeze(-1), 0)
        result = m.last_layernorm(x)
        if m.pooling_mode != "none":
            # Keep the pooling branch active at removed queries, especially the
            # original readout. Remove evidence keys, not the entire query pool.
            e = m.item_emb(identity).masked_fill(removed.unsqueeze(-1), 0)
            valid = identity.ne(0)
            age = positions[:, None] - positions[None, :]
            pool_mask = (~padding)[:, :, None] & valid[:, None, :] & age.ge(0)[None]
            with torch.autocast(device_type=sequence.device.type, enabled=False):
                weights = torch.ones(
                    batch, length, length, device=sequence.device, dtype=torch.float32
                )
                if m.pooling_mode in {"support", "combined"}:
                    z = F.normalize(e.detach().float(), dim=-1, eps=1e-8)
                    similarity = torch.bmm(z, z.transpose(1, 2)).clamp_min(0)
                    pairs = valid[:, :, None] & valid[:, None, :]
                    pairs &= ~torch.eye(
                        length, device=sequence.device, dtype=torch.bool
                    )[None]
                    similarity = similarity.masked_fill(~pairs, 0)
                    count = valid.cumsum(-1).float()
                    support = similarity.cumsum(-1).transpose(1, 2)
                    support = support / (count - 1).clamp_min(1)[:, :, None]
                    weights = weights * (1 + support)
                if m.pooling_mode in {"recency", "combined"}:
                    weights = (
                        weights * torch.exp(-0.05 * age.clamp_min(0).float())[None]
                    )
                weights = weights.masked_fill(~pool_mask, 0)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            pooled = torch.bmm(weights.to(e.dtype), e)
            result = result + m.pool_gate.sigmoid() * m.pool_projection(pooled)
        return result.masked_fill(padding.unsqueeze(-1), 0)

    def _validate_intervention(self):
        if self.maxlen < 1 or self.itemnum < 2:
            raise ValueError(
                "Intervention validation requires maxlen >= 1 and two real items"
            )
        n = min(self.maxlen, 4)
        seq = torch.zeros((1, self.maxlen), dtype=torch.long, device=self.device)
        seq[0, -n:] = torch.arange(n, device=self.device) % self.itemnum + 1
        absent = torch.zeros_like(seq, dtype=torch.bool)
        original = self.view.log2feats(seq)
        actual = self._features(seq, absent)
        torch.testing.assert_close(actual, original, atol=2e-5, rtol=2e-5)
        empty = torch.zeros_like(seq)
        torch.testing.assert_close(
            self._features(empty, absent),
            self.view.log2feats(empty),
            atol=2e-5,
            rtol=2e-5,
        )
        all_removed = seq.ne(0)
        substituted = seq.clone()
        substituted[all_removed] = substituted[all_removed] % self.itemnum + 1
        torch.testing.assert_close(
            self._features(seq, all_removed),
            self._features(substituted, all_removed),
            atol=2e-5,
            rtol=2e-5,
        )
        cases = [(empty, absent), (seq, all_removed)]
        removed = absent.clone()
        removed[:, -1] = True
        cases.append((seq, removed))
        for sequence, mask in cases:
            value = self._features(sequence, mask)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite intervention features")
        changed = seq.clone()
        changed[removed] = changed[removed] % self.itemnum + 1
        torch.testing.assert_close(
            self._features(seq, removed),
            self._features(changed, removed),
            atol=2e-5,
            rtol=2e-5,
        )
        # Probe identity invariance at every occupied slot, including the readout.
        for slot in range(self.maxlen - n, self.maxlen):
            mask = absent.clone()
            mask[:, slot] = True
            altered = seq.clone()
            altered[:, slot] = altered[:, slot] % self.itemnum + 1
            torch.testing.assert_close(
                self._features(seq, mask),
                self._features(altered, mask),
                atol=2e-5,
                rtol=2e-5,
            )
        return {
            "status": "available",
            "no_op_agreement": True,
            "empty_and_all_removed_finite": True,
            "identity_invariance": True,
            "implementation": "view hook" if self._hook else "SASRec adapter",
        }

    def _documents(self, field):
        ids = sorted(i for i in self.counts if 1 <= i <= self.itemnum)
        documents = []
        for item in ids:
            info = self.meta_data.get(str(item), self.meta_data.get(item, {}))
            info = info if isinstance(info, dict) else {}
            parts = []
            for key in (("title", "description") if field == "combined" else (field,)):
                value = info.get(key, "")
                if isinstance(value, list):
                    value = " ".join(v for v in value if isinstance(v, str))
                if isinstance(value, str):
                    parts.append(value)
            documents.append(" ".join(parts).strip())
        return ids, documents

    def _text_features(self, field, remove_common=False):
        # DEBUG: Check evidence availability before importing an optional text
        # dependency. Missing public text must not crash otherwise valid probes.
        ids, documents = self._documents(field)
        nonempty = [d for d in documents if d]
        if len(nonempty) < 2:
            return None, {
                "status": "unavailable",
                "reason": "Insufficient text documents",
                "documents": len(nonempty),
            }
        # DEBUG: Import only when there are actual text documents to analyze.
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        try:
            # Missing text is unavailable evidence, not an empty document that
            # should alter IDF or the max_df denominator.
            vectorizer.fit(nonempty)
            matrix = vectorizer.transform(documents)
        except ValueError as exc:
            return None, {
                "status": "unavailable",
                "reason": str(exc),
                "documents": len(nonempty),
            }
        terms = vectorizer.get_feature_names_out()
        removed_terms = []
        if remove_common:
            from scipy.sparse import diags

            # Denominator is available, nonempty training-item documents.
            df = np.asarray((matrix > 0).sum(axis=0)).ravel()
            keep = df < 0.5 * len(nonempty)
            removed_terms = terms[~keep].tolist()
            matrix = matrix[:, keep].tocsr()
            terms = terms[keep]
            if not len(terms):
                return None, {
                    "status": "unavailable",
                    "reason": "Empty filtered vocabulary",
                    "documents": len(nonempty),
                }
            norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
            inverse = np.zeros_like(norms)
            np.divide(1, norms, out=inverse, where=norms > 0)
            matrix = (diags(inverse) @ matrix).tocsr()
        importance = np.asarray(matrix.mean(axis=0)).ravel()
        top = sorted(range(len(terms)), key=lambda j: (-importance[j], terms[j]))[:10]
        report = {
            "status": "available",
            "documents": len(nonempty),
            "training_items": len(ids),
            "vocabulary_size": len(terms),
            "top_terms": [
                {"term": str(terms[j]), "mean_tfidf": float(importance[j])} for j in top
            ],
            "removed_common_phrase_count": len(removed_terms),
            "removed_common_phrase_examples": removed_terms[:20],
        }
        return (matrix, {item: row for row, item in enumerate(ids)}), report

    def _groups(self, history, features):
        matrix, rows = features
        offset = self.maxlen - len(history)
        events = []
        for j in range(max(0, len(history) - 20), len(history)):
            item = history[j]
            if item in rows and matrix[rows[item]].nnz:
                events.append((offset + j, item))
        groups = []
        duplicate_examples = []
        for event in reversed(events):
            for group in groups:
                similarities = [
                    float(matrix[rows[event[1]]].multiply(matrix[rows[other[1]]]).sum())
                    for other in group
                ]
                if all(s >= 0.3 for s in similarities):
                    for other, similarity in zip(group, similarities):
                        if similarity >= 0.95 and event[1] != other[1]:
                            duplicate_examples.append(
                                {"items": [event[1], other[1]], "cosine": similarity}
                            )
                    group.append(event)
                    break
            else:
                groups.append([event])
        eligible = [g for g in groups if len({item for _, item in g}) >= 2]
        eligible.sort(
            key=lambda g: (
                -len(g),
                -max(p for p, _ in g),
                tuple(sorted(p for p, _ in g)),
            )
        )
        singletons = sorted((g for g in groups if len(g) == 1), key=lambda g: -g[0][0])
        return (
            eligible[0] if eligible else None,
            singletons[0] if singletons else None,
            duplicate_examples[:10],
        )

    def _controls(self, history, target, seed):
        import math

        rng = np.random.default_rng(seed)
        thresholds = self._popularity_thresholds
        if thresholds is None:
            return None, {
                "reason": "No training popularity observations",
                "seed": int(seed),
            }
        frequency = Counter(history)
        offset = self.maxlen - len(history)
        bins = {}
        slot_bins = {}
        for j, item in enumerate(history):
            slot = offset + j
            key = (
                min(3, 4 * slot // self.maxlen),
                int(np.searchsorted(thresholds, self.counts[item], side="right")),
                frequency[item],
            )
            bins.setdefault(key, []).append(slot)
            slot_bins[slot] = key
        target = tuple(sorted(target))
        needs = Counter(slot_bins[p] for p in target)
        possibilities = (
            math.prod(math.comb(len(bins[k]), count) for k, count in needs.items()) - 1
        )
        if possibilities < 20:
            return None, {
                "reason": "Fewer than 20 distinct matched alternatives",
                "possible_alternatives": int(possibilities),
                "seed": int(seed),
                "target_slots": list(target),
            }
        found = set()
        for draw in range(self.settings["control_draw_limit"]):
            chosen = []
            for key, count in sorted(needs.items()):
                chosen.extend(rng.choice(bins[key], count, replace=False).tolist())
            chosen = tuple(sorted(chosen))
            if chosen != target:
                found.add(chosen)
            if len(found) == 20:
                controls = sorted(found)
                return controls, {
                    "seed": int(seed),
                    "draws": draw + 1,
                    "mean_target_overlap_fraction": float(
                        np.mean(
                            [len(set(c) & set(target)) / len(target) for c in controls]
                        )
                    ),
                    "target_slots": list(target),
                    "control_slots": [list(c) for c in controls],
                }
        return None, {
            "reason": "Distinct-control sampler exhausted its declared draw limit",
            "distinct_found": len(found),
            "seed": int(seed),
        }

    def _top10(self, features, candidates):
        # Candidate IDs are ascending, so stable sorting implements ID tie breaks.
        parts = []
        for start in range(0, len(candidates), 8192):
            ids = torch.tensor(candidates[start : start + 8192], device=self.device)
            # Use the same architecture-specific scoring contract as margins.
            items = ids.view(1, 1, -1).expand(features.shape[0], 1, -1)
            scores = self.view.predict_score(features[:, None, :], items)
            if scores.shape != items.shape:
                raise ValueError("predict_score must return B x L x K candidate scores")
            scores = scores[:, 0, :]
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite full-catalog scores")
            scores = scores.float().cpu().numpy()
            best = np.argsort(-scores, axis=1, kind="stable")[:, :10]
            parts.append(
                (
                    np.asarray(candidates[start : start + 8192])[best],
                    np.take_along_axis(scores, best, axis=1),
                )
            )
        results = []
        for row in range(features.shape[0]):
            ids = np.concatenate([p[0][row] for p in parts])
            scores = np.concatenate([p[1][row] for p in parts])
            order = np.lexsort((ids, -scores))[:10]
            results.append(tuple(int(i) for i in ids[order]))
        return results

    @torch.no_grad()
    def _recommend(self, user, history, masks, mode="mask"):
        if mode not in {"mask", "delete", "shuffle"}:
            raise ValueError(f"Unknown diagnostic intervention mode: {mode}")
        candidate_key = ("candidates", user)
        if candidate_key not in self._cache:
            observed = set(self.histories[user])
            self._cache[candidate_key] = [
                i for i in range(1, self.itemnum + 1) if i not in observed
            ]
        candidates = self._cache[candidate_key]
        if len(candidates) < 10:
            raise ValueError("Fewer than ten original unseen catalog items")
        keys = [(user, mode, tuple(slots)) for slots in masks]
        pending = list(dict.fromkeys(key for key in keys if key not in self._cache))
        # Batch encoder calls and catalog scoring while bounding activation memory.
        for start in range(0, len(pending), 8):
            chunk = pending[start : start + 8]
            sequences, removals = [], []
            for _, _, slots in chunk:
                sequence = [0] * (self.maxlen - len(history)) + history
                removed = [False] * self.maxlen
                if mode == "delete":
                    slot_set = set(slots)
                    retained = [
                        i for p, i in enumerate(sequence) if i and p not in slot_set
                    ]
                    sequence = [0] * (self.maxlen - len(retained)) + retained
                elif mode == "shuffle":
                    if sorted(slots) != list(range(self.maxlen)):
                        raise ValueError(
                            "Shuffle must supply a permutation of original slots"
                        )
                    sequence = [sequence[p] for p in slots]
                else:
                    for slot in slots:
                        removed[slot] = True
                sequences.append(sequence)
                removals.append(removed)
            seq = torch.tensor(sequences, dtype=torch.long, device=self.device)
            if mode == "mask":
                removed = torch.tensor(removals, dtype=torch.bool, device=self.device)
                feats = self._features(seq, removed)
            else:
                feats = self.view.log2feats(seq)
            rankings = self._top10(feats[:, -1], candidates)
            self._cache.update(zip(chunk, rankings))
        return [self._cache[key] for key in keys]

    def _effect(self, user, history, target, controls, mode="mask"):
        baseline = self._recommend(user, history, [()], "mask")[0]
        rankings = self._recommend(user, history, [target] + controls, mode)
        effects = [1 - len(set(baseline) & set(r)) / 10 for r in rankings]
        return {
            "user": user,
            "target_effect": float(effects[0]),
            "control_effects": [float(v) for v in effects[1:]],
            "control_mean": float(np.mean(effects[1:])),
            "matched_effect": float(effects[0] - np.mean(effects[1:])),
        }

    @staticmethod
    def _summary(records, seed=42):
        if not records:
            return {
                "status": "unavailable",
                "n_users": 0,
                "mean": None,
                "ci95": None,
                "reason": "No eligible matched users",
            }
        values = np.asarray([r["matched_effect"] for r in records])
        rng = np.random.default_rng(seed)
        bootstrap = values[rng.integers(0, len(values), (1000, len(values)))].mean(1)
        return {
            "status": "available",
            "n_users": len(records),
            "mean": float(values.mean()),
            "target_mean": float(np.mean([r["target_effect"] for r in records])),
            "control_mean": float(np.mean([r["control_mean"] for r in records])),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "records": records,
        }

    def _lexical_run(self, users, features, seed=42, auxiliary=False):
        records, singleton_records, deletion_records = [], [], []
        coverage, near_duplicates, shuffles = [], [], []
        for index, user in enumerate(users):
            # DEBUG: Bound the expensive full-catalog intervention loop.
            if self._budget_exhausted():
                break
            history = self.histories[user][-self.maxlen :]
            group, singleton, duplicates = self._groups(history, features)
            near_duplicates.extend(duplicates)
            row = {"user": user, "history_events": len(history)}
            unseen_count = self.itemnum - len(
                {i for i in self.histories[user] if 1 <= i <= self.itemnum}
            )
            if unseen_count < 10:
                row.update(
                    status="unavailable",
                    reason="Fewer than ten unseen catalog items",
                    unseen_candidates=unseen_count,
                )
                coverage.append(row)
                continue
            if group is None:
                row["status"] = "no_repeated_distinct_item_group"
            else:
                target = tuple(sorted(p for p, _ in group))
                controls, matching = self._controls(history, target, seed + index)
                row["matching"] = matching
                if controls is None:
                    row["status"] = "matched_controls_unavailable"
                else:
                    row["status"] = "available"
                    records.append(self._effect(user, history, target, controls))
                    if auxiliary:
                        deletion_records.append(
                            self._effect(user, history, target, controls, mode="delete")
                        )
            if auxiliary and singleton:
                target = (singleton[0][0],)
                controls, matching = self._controls(
                    history, target, seed + 10000 + index
                )
                row["singleton_matching"] = matching
                if controls is not None:
                    singleton_records.append(
                        self._effect(user, history, target, controls)
                    )
            if auxiliary and history:
                rng = np.random.default_rng(seed + 20000 + index)
                slots = list(range(self.maxlen))
                occupied = list(range(self.maxlen - len(history), self.maxlen - 1))
                permutation = rng.permutation(occupied).tolist()
                for dst, src in zip(occupied, permutation):
                    slots[dst] = src
                baseline = self._recommend(user, history, [()])[0]
                shuffled = self._recommend(user, history, [tuple(slots)], "shuffle")[0]
                shuffles.append(
                    {
                        "user": user,
                        "seed": seed + 20000 + index,
                        "source_slots": slots,
                        "effect": float(1 - len(set(baseline) & set(shuffled)) / 10),
                        "changed": permutation != occupied,
                    }
                )
            coverage.append(row)
        result = self._summary(records)
        result.update(
            sampled_users=len(users),
            # DEBUG: Distinguish the requested sample from processed coverage.
            processed_users=len(coverage),
            budget_exhausted=self._budget_exhausted(),
            coverage=coverage,
            near_duplicate_inspection={
                "scope": "grouped pairs in sampled recent training histories",
                "examples": near_duplicates[:10],
            },
        )
        if auxiliary:
            result["singleton_matched"] = self._summary(singleton_records)
            result["deletion_repacking_matched"] = self._summary(deletion_records)
            result["prefix_shuffle"] = {
                "n_users": len(shuffles),
                "records": shuffles,
                "mean_effect": (
                    float(np.mean([r["effect"] for r in shuffles]))
                    if shuffles
                    else None
                ),
            }
        return result

    def analyze_matched_evidence(self):
        import time

        started = time.monotonic()
        result = {
            "settings": dict(self.settings),
            "category": self.category,
            "target_metrics": {
                "status": "unavailable",
                "n_targets": 0,
                "reason": "No separate legal fixed-target diagnostic interface supplied",
            },
        }
        try:
            result["intervention_checks"] = self._validate_intervention()
        except (NotImplementedError, AssertionError, ValueError, RuntimeError) as exc:
            result.update(status="unavailable", reason=str(exc), n_users=0)
            result["runtime_seconds"] = time.monotonic() - started
            return result
        available = sorted(u for u, h in self.histories.items() if h)
        rng = np.random.default_rng(42)
        users = (
            [
                available[i]
                for i in rng.choice(
                    len(available), min(50, len(available)), replace=False
                )
            ]
            if available
            else []
        )
        result["sampled_user_ids"] = users
        result["exploratory_proxy_validity"] = {}
        primary_features = None
        for label, field, remove in (
            ("combined", "combined", False),
            ("title_only", "title", False),
            ("description_only", "description", False),
            ("common_phrases_removed", "combined", True),
        ):
            # DEBUG: Retain an explicit report for checks the budget prevented.
            if self._budget_exhausted():
                report = {
                    "status": "unavailable",
                    "reason": "Category diagnostic time budget exhausted",
                }
                if label == "combined":
                    result["primary"] = report
                else:
                    result["exploratory_proxy_validity"][label] = report
                continue
            begin = time.monotonic()
            features, report = self._text_features(field, remove)
            if features is not None:
                report["effects"] = self._lexical_run(
                    users, features, auxiliary=label == "combined"
                )
            report["runtime_seconds"] = time.monotonic() - begin
            if label == "combined":
                primary_features = features
                result["primary"] = report
            else:
                result["exploratory_proxy_validity"][label] = report
        permutations = []
        if primary_features is not None:
            matrix, rows = primary_features
            for index in range(20):
                # DEBUG: Permutations consume the same category budget.
                if self._budget_exhausted():
                    break
                begin = time.monotonic()
                seed = 42000 + index
                permutation = np.random.default_rng(seed).permutation(matrix.shape[0])
                report = self._lexical_run(users, (matrix[permutation], rows))
                # Full mappings make each exploratory permutation reproducible.
                report.update(
                    seed=seed,
                    item_to_text_row=permutation.tolist(),
                    runtime_seconds=time.monotonic() - begin,
                )
                permutations.append(report)
        result["exploratory_proxy_validity"]["item_to_text_permutations"] = permutations
        # A fitted vocabulary alone does not establish diagnostic availability.
        primary = result["primary"]
        effects = primary.get("effects", {})
        result["status"] = effects.get("status", "unavailable")
        result["n_users"] = effects.get("n_users", 0)
        if result["status"] != "available":
            result["reason"] = effects.get(
                "reason", primary.get("reason", "No matched observations")
            )
        result["runtime_seconds"] = time.monotonic() - started
        return result

    @staticmethod
    def aggregate_categories(results):
        """Combine category run_full_diagnosis results without dropping routes."""
        expected = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        arrays = []
        missing = []
        for category in expected:
            result = results.get(category, {})
            evidence = result.get("metrics", {}).get("matched_evidence_masking", {})
            if evidence.get("category") not in (None, category):
                raise ValueError(f"Diagnosis route mismatch for {category}")
            records = evidence.get("primary", {}).get("effects", {}).get("records", [])
            if not records:
                missing.append(category)
            else:
                arrays.append(np.asarray([r["matched_effect"] for r in records]))
        if missing:
            return {
                "status": "unavailable",
                "mean": None,
                "ci95": None,
                "unavailable_categories": missing,
                "reason": "Four-category macro requires observations in every category",
            }
        rng = np.random.default_rng(42)
        bootstrap = np.zeros(1000)
        for values in arrays:
            bootstrap += (
                values[rng.integers(0, len(values), (1000, len(values)))].mean(1) / 4
            )
        return {
            "status": "available",
            "mean": float(np.mean([v.mean() for v in arrays])),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "category_user_counts": dict(zip(expected, map(len, arrays))),
            "seed": 42,
            "bootstrap_replicates": 1000,
        }

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Read embeddings through the compatibility view
            emb = self.view.item_emb.weight.detach()
            ### <<< Self_EvolveRec-BLOCK-END
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable cosine estimates for insufficient samples
            if min(sample_size, n) < 2:
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
        ### >>> Self_EvolveRec-BLOCK-START: Preserve caller mode and keep compatibility margin inference
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

            ### >>> Self_EvolveRec-BLOCK-START: Use tensor sequences on the trained model device
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.view.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            ### >>> Self_EvolveRec-BLOCK-START: Route scoring through the compatibility view
            all_logits = self.view.predict_score(log_feats, all_items)
            if not torch.isfinite(all_logits).all():
                raise ValueError("Nonfinite diagnostic margin logits")
            ### <<< Self_EvolveRec-BLOCK-END
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Require real positive and negative margin observations
            valid = (pos_t != 0) & (neg_t != 0)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Align error counts with valid margin observations
            valid_cpu = (pos_cpu != 0) & (neg_cpu != 0)
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
            ### >>> Self_EvolveRec-BLOCK-START: Empty margin samples are unavailable rather than healthy zeros
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "n_positions": 0,
                "status": "unavailable",
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            ### >>> Self_EvolveRec-BLOCK-START: Attach observation counts to margin estimates
            margin_metrics = {
                "n_positions": int(N),
                "status": "available",
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

        ### >>> Self_EvolveRec-BLOCK-START: Explicitly disable unavailable item-category error labels
        breakdown = {
            "status": "unavailable",
            "reason": "Item category metadata is unavailable in the matched task",
            "eps": float(eps),
            "n_positions": margin_metrics["n_positions"],
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
        }
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: State-safe diagnosis with explicit coverage and local RNG seeding
    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        import time

        started = time.monotonic()
        # Save individual modes as well as RNGs, including loader-side sampling.
        modes = [(module, module.training) for module in self.model.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        self._cache.clear()
        try:
            self.model.eval()
            random.seed(42)
            np.random.seed(42)
            # Seed only generators whose states this call saves and restores.
            torch.random.default_generator.manual_seed(42)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(42)
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=0.5, topk_categories=20
                )
            )
            evidence = self.analyze_matched_evidence()
            result = {
                "metrics": {
                    "embedding_collapse_score": collapse_score,
                    "embedding_cosine_sample_items": min(2048, self.itemnum),
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_n_positions": margin_metrics["n_positions"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "matched_evidence_masking": evidence,
                    "four_category_macro_matched_effect": {
                        "status": "unavailable",
                        "mean": None,
                        "reason": "Use aggregate_categories with all four category diagnosis results",
                    },
                    "unobserved_traits": {
                        "simulator_diversity": None,
                        "simulator_conformity": None,
                        "reason": "Not observed in the public task data",
                    },
                    # DEBUG: Expose budget truncation alongside measured results.
                    "diagnostic_budget_exhausted": self._budget_exhausted(),
                    "diagnostic_runtime_seconds": time.monotonic() - started,
                },
                "metric_definitions": {
                    "embedding_collapse_score": (
                        "(mean off-diagonal item cosine + 1)/2; null with fewer than two "
                        "sampled items. A value near 0.5 means mean cosine near zero and "
                        "does not establish collapse. Directional similarity alone does not "
                        "measure catalog diversity, rank, or embedding variance."
                    ),
                    "embedding_cosine_sample_items": "Number of sampled nonpadding item embeddings.",
                    "pos_neg_margin_mean": "Mean positive-minus-negative score over real paired IDs.",
                    "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of observed score margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of observed margins below zero.",
                    "pos_neg_small_margin_rate": "Fraction of observed margins below the fixed epsilon 0.5.",
                    "pos_neg_margin_n_positions": "Number of valid paired training positions; empty statistics are null.",
                    "pos_neg_margin_category_breakdown": "Unavailable: item category labels are not public task inputs.",
                    "matched_evidence_masking": (
                        "Training-history diagnostic, not recommendation accuracy. Effect is "
                        "one minus original/perturbed full-catalog unseen top-10 overlap. "
                        "The original entire observed history is excluded in every condition; "
                        "ties use ascending item ID. Matched effect subtracts each user's "
                        "mean of 20 distinct same-user controls matched jointly on original "
                        "position quartile, training popularity quartile, and exact truncated "
                        "history repeat frequency. Partial overlap is permitted and reported. "
                        "Repeated lexical groups require at least two distinct item IDs. "
                        "User bootstrap intervals resample whole user intervention records. "
                        "Singleton masking, prefix shuffling, deletion/repacking and all text "
                        "sensitivity/permutation results are separate exploratory proxy checks. "
                        "No next-item labels or causal preference claims are manufactured."
                    ),
                    "four_category_macro_matched_effect": (
                        "Equal mean of all four category user means. Bootstrap users within "
                        "each category, then average category means equally; unavailable if "
                        "any category lacks matched observations. Call aggregate_categories "
                        "with complete category results. This diagnostic never selects models."
                    ),
                    "unobserved_traits": "Simulator conformity/diversity traits are explicitly unobserved.",
                    # DEBUG: Document partial diagnostic execution explicitly.
                    "diagnostic_budget_exhausted": "Whether the shared category deadline was reached; inspect processed coverage before interpreting partial results.",
                    "diagnostic_runtime_seconds": "Wall time of this call; diagnostic GPU time is charged by the host.",
                },
            }
            return result
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, self.device)
            self._cache.clear()

    # Review checks: traced compatibility scoring, no-op and identity invariance,
    # pooling at removed readouts, fixed candidates, matching coverage, category
    # routing, and RNG restoration. Execution and GPU timing remain host checks;
    # no measured accuracy improvement is claimed by these diagnostic changes.
    ### <<< Self_EvolveRec-BLOCK-END
