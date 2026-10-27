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


### >>> Self_EvolveRec-BLOCK-START: Validate replacement strata, matching settings, sparse controls, and diagnostic failures
class DiagnosisProbe:
    """Frozen-model diagnostics; never an accuracy objective or a user-trait estimator.

    Optional view adapter fields:
      dataset_name: the current category route;
      diagnostic_user_ids: the supplied five training users, or a route-to-IDs dict;
      item_mapping: the public item-ID mapping, if exposed.
    simulator_user_ids and audit_user_ids are accepted panel aliases. Missing panel
    information is reported explicitly; no substitute panel is sampled.
    Full CPU rank arrays remain available in probe.rank_cache after the call.
    """

    def __init__(self, view, device):
        import hashlib

        self.view = view
        self.model = view.model
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.item_emb = view.item_emb
        self._hashlib = hashlib
        route = getattr(view, "dataset_name", None)
        if route is None:
            route = getattr(self.model, "dataset_name", None)
        self.dataset = str(route) if route is not None else None
        self.maxlen = getattr(self.model, "maxlen", getattr(view, "maxlen", None))
        if self.maxlen is None and hasattr(self.model, "pos_emb"):
            self.maxlen = self.model.pos_emb.num_embeddings
        if self.maxlen is None or int(self.maxlen) < 1:
            raise ValueError("Adapter must expose the trained model's positive maxlen")
        self.maxlen = int(self.maxlen)
        self.itemnum = int(self.item_emb.weight.shape[0]) - 1
        self.rank_cache = {}
        self.settings = {
            "seed": 42,
            "draws": 5,
            "topk": 10,
            "chunk_size": 4096,
            "pair_limit": 4096,
            "margin_batches": 10,
            "margin_batch_size": 32,
            "margin_eps": 0.5,
            "bootstrap_replicates": 2000,
            "control_half_life": 10.0,
            "candidate_policy": "IDs 1..embedding_rows-1 minus complete original training history",
            "tie_policy": "ascending candidate ID, stable descending score",
            "position_policy": "replace IDs in fixed slots; preserve padding and final query; shuffling controls reported separately",
            "replacement_draw_seeds": [0, 1, 2, 3, 4],
            "transition_alpha": 10.0,
            "matching_caliper_sd": 0.5,
            "text_cosine_difference_caliper": 0.1,
            "audit_rtol": 1e-5,
            "audit_atol": 1e-6,
        }

    def _digest(self, value):
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return self._hashlib.sha256(payload).hexdigest()

    def _seed(self, user, edit, draw=0):
        return int(self._digest([42, self.dataset, str(user), edit, draw])[:16], 16) & (
            (1 << 63) - 1
        )

    def _sequence(self, history):
        tail = tuple(history)[-self.maxlen :]
        return torch.tensor(
            [[0] * (self.maxlen - len(tail)) + list(tail)],
            dtype=torch.long,
            device=self.device,
        )

    def _features(self, history):
        features = self.view.log2feats(self._sequence(history))
        if features.ndim != 3 or features.shape[:2] != (1, self.maxlen):
            raise ValueError("log2feats must return B x L x D features")
        final = features[:, -1, :]
        if not torch.isfinite(final).all():
            raise ValueError("Nonfinite encoder features")
        # Protect cached baseline features if an adapter reuses an output buffer.
        return final.clone()

    def _scores(self, features, candidates):
        if not len(candidates):
            raise ValueError("Cannot score an empty diagnostic candidate pool")
        chunks = []
        for start in range(0, len(candidates), self.settings["chunk_size"]):
            ids = torch.as_tensor(
                candidates[start : start + self.settings["chunk_size"]],
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)
            values = self.view.predict_score(features, ids)
            if not isinstance(values, torch.Tensor) or values.shape != ids.shape:
                raise ValueError("Candidate predict_score must return B x C logits")
            if not torch.isfinite(values).all():
                raise ValueError(
                    "Nonfinite candidate scores; candidate pool was not changed"
                )
            # Convert before NumPy export, including for bfloat16 adapters.
            chunks.append(
                values.detach()
                .reshape(-1)
                .to(device="cpu", dtype=torch.float64)
                .numpy()
            )
        return np.concatenate(chunks)

    @staticmethod
    def _ranks(scores):
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(len(scores))
        return order, ranks

    def _pairs(self, user, count):
        limit = min(self.settings["pair_limit"], count * (count - 1) // 2)
        if count * (count - 1) // 2 <= limit:
            pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
        else:
            rng = np.random.default_rng(self._seed(user, "candidate_pairs"))
            selected = set()
            while len(selected) < limit:
                i, j = rng.integers(0, count, size=2).tolist()
                if i != j:
                    selected.add((min(i, j), max(i, j)))
            pairs = sorted(selected)
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def _statistics(self, original, changed, pairs):
        baseline_order, baseline_ranks = self._ranks(original)
        order, ranks = self._ranks(changed)
        k = min(10, len(original))
        a = np.sign(original[pairs[:, 0]] - original[pairs[:, 1]])
        b = np.sign(changed[pairs[:, 0]] - changed[pairs[:, 1]])
        strict = (a != 0) & (b != 0)
        return (
            {
                "top10_overlap": float(
                    len(set(baseline_order[:k]) & set(order[:k])) / k
                ),
                "top10_mean_absolute_rank_displacement": float(
                    np.abs(
                        ranks[baseline_order[:k]] - baseline_ranks[baseline_order[:k]]
                    ).mean()
                ),
                "strict_pair_flip_rate": (
                    float((a[strict] != b[strict]).mean()) if strict.any() else None
                ),
                "strict_pair_count": int(strict.sum()),
                "baseline_tied_pairs": int((a == 0).sum()),
                "edited_tied_pairs": int((b == 0).sum()),
                "raw_mean_absolute_score_change": float(
                    np.abs(original - changed).mean()
                ),
            },
            order,
            ranks,
        )

    def _edits(self, user, history):
        yield "identity", 0, history, 0, "retained"
        if len(history) < 2:
            return
        yield "last_item_only", 0, history[-1:], len(history) - 1, "retained"
        for draw in self.settings["replacement_draw_seeds"]:
            rng = random.Random(self._seed(user, "prefix_shuffle", draw))
            prefix = list(history[:-1])
            rng.shuffle(prefix)
            yield "prefix_shuffle", draw, tuple(prefix) + history[-1:], 0, "retained"

    def _diagnostic_loader(self, snapshot, counts):
        """Independent zero-worker B x L batches; never touches train_loader."""
        eligible = []
        for user, history in sorted(snapshot.items()):
            if len(history) < 2:
                counts["margin_insufficient_history"] += 1
            elif len(set(history)) >= self.itemnum:
                counts["margin_no_negative"] += 1
            else:
                eligible.append(user)
        if not eligible:
            return
        # User selection and negative sampling have independent local streams.
        rng = random.Random(self._seed("margins", "users"))
        negative_rng = random.Random(self._seed("margins", "negatives"))
        complement_cache = {}
        for _ in range(self.settings["margin_batches"]):
            seqs, positives, negatives = [], [], []
            for _ in range(self.settings["margin_batch_size"]):
                user = rng.choice(eligible)
                history = snapshot[user]
                inputs, targets = (
                    history[:-1][-self.maxlen :],
                    history[1:][-self.maxlen :],
                )
                if user not in complement_cache:
                    observed_ids = np.asarray(sorted(set(history)), dtype=np.int64)
                    shifted = observed_ids - np.arange(len(observed_ids)) - 1
                    complement_cache[user] = (shifted, self.itemnum - len(observed_ids))
                shifted, available = complement_cache[user]
                # Vectorized exact uniform sampling from the unseen complement.
                indices = np.asarray(
                    [negative_rng.randrange(available) for _ in targets], dtype=np.int64
                )
                neg = (
                    indices + 1 + np.searchsorted(shifted, indices, side="right")
                ).tolist()
                pad = [0] * (self.maxlen - len(inputs))
                seqs.append(pad + list(inputs))
                positives.append(pad + list(targets))
                negatives.append(pad + neg)
            yield (
                None,
                torch.tensor(seqs, dtype=torch.long, device=self.device),
                torch.tensor(positives, dtype=torch.long, device=self.device),
                torch.tensor(negatives, dtype=torch.long, device=self.device),
            )

    def _margins(self, snapshot, counts):
        collected = []
        for _, sequence, positive, negative in self._diagnostic_loader(
            snapshot, counts
        ):
            features = self.view.log2feats(sequence)
            items = torch.stack((positive, negative), dim=-1)
            logits = self.view.predict_score(features, items)
            if logits.shape != items.shape:
                raise ValueError("Sequence predict_score must return B x L x 2 logits")
            values = (logits[..., 0] - logits[..., 1])[positive.ne(0)]
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite diagnostic margins")
            collected.append(values.detach().cpu().double().numpy())
        values = np.concatenate(collected) if collected else np.empty(0)
        return {
            "positions": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "p10": (
                float(
                    np.partition(values, max(0, int(np.ceil(0.1 * values.size)) - 1))[
                        max(0, int(np.ceil(0.1 * values.size)) - 1)
                    ]
                )
                if values.size
                else None
            ),
            "neg_beats_pos_rate": float((values < 0).mean()) if values.size else None,
            "small_margin_rate": (
                float((values < self.settings["margin_eps"]).mean())
                if values.size
                else None
            ),
            "category_breakdown": {},
            "distribution": (
                f"{self.settings['margin_batches'] * self.settings['margin_batch_size']} "
                "independent uniform eligible-user draws with replacement when eligible "
                "users exist; latest L teacher-forced transitions; negatives uniform "
                "over catalog minus full observed training history; independent "
                "user-selection and negative-sampling RNG streams"
            ),
        }

    def _embeddings(self):
        emb = self.item_emb.weight.detach()[1:]
        n = min(2048, len(emb))
        if n < 2:
            return {"status": "insufficient_embeddings", "sample_size": n}
        generator = torch.Generator(device=emb.device)
        generator.manual_seed(42)
        indices = torch.randperm(len(emb), generator=generator, device=emb.device)[:n]
        x = emb.index_select(0, indices).float()
        if not torch.isfinite(x).all():
            return {"status": "nonfinite_embeddings", "sample_size": n}
        norms = x.norm(dim=-1)
        unit = F.normalize(x, dim=-1)
        cosine = (
            (unit.sum(0).square().sum() - unit.square().sum()) / (n * (n - 1))
        ).clamp(-1, 1)
        singular = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
        mass = singular.sum()
        defined = bool(mass > 0)
        p = singular / mass if defined else singular
        entropy = -(p[p > 0] * p[p > 0].log()).sum()
        return {
            "status": "ok" if defined else "undefined_zero_centered_spectrum",
            "sample_size": n,
            "centering": "subtract sampled column means for SVD only",
            "mean_pairwise_cosine": float(cosine),
            "effective_rank_fraction": (
                float(entropy.exp() / min(n - 1, x.shape[1])) if defined else None
            ),
            "leading_singular_value_mass": (
                float(singular[0] / mass) if defined else None
            ),
            "embedding_norm_mean": float(norms.mean()),
            "zero_norm_count": int((norms == 0).sum()),
        }

    def _panel(self):
        for name in ("diagnostic_user_ids", "simulator_user_ids", "audit_user_ids"):
            panel = getattr(self.view, name, None)
            if isinstance(panel, dict):
                panel = panel.get(self.dataset)
            if panel is not None:
                ids = [str(user) for user in panel]
                if len(ids) != 5 or len(set(ids)) != 5:
                    return [], {
                        "status": "invalid_supplied_panel",
                        "source": name,
                        "user_ids": ids,
                    }
                if self.dataset is None:
                    return [], {
                        "status": "missing_dataset_route",
                        "source": name,
                        "user_ids": ids,
                    }
                return ids, {
                    "status": "supplied_exploratory_panel",
                    "source": name,
                    "user_ids": ids,
                }
        return [], {"status": "missing_supplied_panel", "user_ids": []}

    def _aggregate(self, records):
        result = {}
        fields = (
            "top10_overlap",
            "top10_mean_absolute_rank_displacement",
            "strict_pair_flip_rate",
            "raw_mean_absolute_score_change",
            "feature_l2_change",
        )
        for edit in sorted({row["edit"] for row in records}):
            rows = [row for row in records if row["edit"] == edit]
            summary = {
                "successful_draws": len(rows),
                "no_op_draws": sum(row["no_op"] for row in rows),
            }
            for field in fields:
                user_values = []
                for user in sorted({row["user"] for row in rows}):
                    values = [
                        row[field]
                        for row in rows
                        if row["user"] == user and row[field] is not None
                    ]
                    if values:
                        user_values.append(float(np.mean(values)))
                rng = np.random.default_rng(self._seed("bootstrap", edit + ":" + field))
                if user_values:
                    x = np.asarray(user_values)
                    boot = x[
                        rng.integers(
                            0,
                            len(x),
                            size=(self.settings["bootstrap_replicates"], len(x)),
                        )
                    ].mean(1)
                    interval = (
                        np.quantile(boot, [0.025, 0.975]).tolist()
                        if len(x) >= 2
                        else None
                    )
                    summary[field] = {
                        "mean": float(x.mean()),
                        "users": len(x),
                        "exploratory_user_bootstrap_95_interval": interval,
                    }
                else:
                    summary[field] = {
                        "mean": None,
                        "users": 0,
                        "exploratory_user_bootstrap_95_interval": None,
                    }
            result[edit] = summary
        return result

    def _replacement_audit(self, snapshot, panel):
        """Plan matches before scoring; never relax calipers after observing effects."""
        if not panel:
            return {"status": "missing_or_invalid_supplied_panel", "match_plans": []}
        frequency = np.zeros(self.itemnum + 1, dtype=np.float64)
        transitions = {}
        for history in snapshot.values():
            np.add.at(frequency, np.asarray(history, dtype=np.int64), 1)
            for a, b in zip(history, history[1:]):
                transitions.setdefault(a, Counter())[b] += 1
        catalog = np.flatnonzero(frequency > 0)
        if not len(catalog):
            return {"status": "empty_training_catalog"}
        prior = frequency / frequency.sum()
        logpop = np.zeros_like(frequency)
        logpop[catalog] = np.log(frequency[catalog])
        pop_sd = float(logpop[catalog].std())
        text, text_status = None, "unavailable"
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            metadata = {str(k): v for k, v in (self.view.meta_data or {}).items()}
            documents = []
            for item in catalog:
                entry = metadata.get(str(item), {})
                if not isinstance(entry, dict):
                    raise TypeError(f"Metadata for item {item} must be a dictionary")
                parts = []
                for field in ("title", "description"):
                    value = entry.get(field)
                    if isinstance(value, str):
                        parts.append(value)
                    elif isinstance(value, (list, tuple)):
                        parts.extend(v for v in value if isinstance(v, str))
                documents.append(" ".join(parts))
            text = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                norm="l2",
            ).fit_transform(documents)
            text_status = "training_catalog_titles_descriptions_only"
        except (AttributeError, TypeError, ValueError, ImportError) as error:
            text_status = "unavailable: " + str(error)
        locations = {int(item): i for i, item in enumerate(catalog)}
        plans, rows, contrasts, failures, audits, controls = [], [], [], [], {}, {}
        fields = (
            "top10_overlap",
            "top10_mean_absolute_rank_displacement",
            "strict_pair_flip_rate",
            "raw_mean_absolute_score_change",
            "feature_l2_change",
        )

        alpha = self.settings["transition_alpha"]
        caliper = self.settings["matching_caliper_sd"]
        rtol, atol = self.settings["audit_rtol"], self.settings["audit_atol"]
        text_available = (
            np.asarray(text.getnnz(axis=1)).ravel() > 0 if text is not None else None
        )

        def incoming(predecessor):
            raw = transitions.get(predecessor, {})
            counts = np.zeros(len(catalog), dtype=np.float64)
            for item, count in raw.items():
                counts[locations[item]] = count
            probability = (counts + alpha * prior[catalog]) / (
                sum(raw.values()) + alpha
            )
            return np.log(probability), counts

        def behavioral_score(items, weights):
            # Exact smoothed mixture, using sparse observed transitions.
            score = np.zeros_like(prior)
            for item, weight in zip(items, weights):
                raw = transitions.get(item, {})
                scale = weight / (sum(raw.values()) + alpha)
                score += scale * alpha * prior
                for target, count in raw.items():
                    score[target] += scale * count
            return score

        for user in panel:
            full = snapshot.get(user, ())
            history = full[-self.maxlen :]
            if not history:
                failures.append({"user": user, "reason": "missing_or_empty_history"})
                continue
            candidates = np.setdiff1d(np.arange(1, self.itemnum + 1), full)
            eligible = ~np.isin(catalog, full)
            selected = []
            n = len(history)
            split = (n - 1) // 2
            for seed in self.settings["replacement_draw_seeds"]:
                rng = random.Random(self._seed(user, "replacement", seed))
                strata = [("final", [n - 1])]
                if split and n - 1 - split:
                    strata.insert(
                        0,
                        ("paired", [rng.randrange(split, n - 1), rng.randrange(split)]),
                    )
                else:
                    failures.append(
                        {
                            "user": user,
                            "draw": seed,
                            "reason": "no_two_nonfinal_regions",
                        }
                    )
                for stratum, positions in strata:
                    feasible, details, similarities = eligible.copy(), [], []
                    for position in positions:
                        displaced = history[position]
                        predecessor = history[position - 1] if position else None
                        lp, raw = incoming(predecessor)
                        sd = float(lp.std())
                        target_index = locations[displaced]
                        pop_delta = logpop[catalog] - logpop[displaced]
                        trans_delta = lp - lp[target_index]
                        feasible &= np.abs(pop_delta) <= caliper * pop_sd
                        feasible &= np.abs(trans_delta) <= caliper * sd
                        cosine = None
                        if text is not None and text[target_index].nnz:
                            cosine = (text @ text[target_index].T).toarray().ravel()
                        similarities.append(cosine)
                        details.append(
                            (predecessor, pop_delta, trans_delta, sd, raw, target_index)
                        )
                    if len(similarities) == 2 and all(
                        v is not None for v in similarities
                    ):
                        feasible &= ~text_available | (
                            np.abs(similarities[0] - similarities[1])
                            <= self.settings["text_cosine_difference_caliper"]
                        )
                    support = np.flatnonzero(feasible)
                    plan = {
                        "user": user,
                        "draw": seed,
                        "stratum": stratum,
                        "predecessor_stratum": (
                            "missing_predecessor"
                            if 0 in positions
                            else "observed_predecessors"
                        ),
                        "positions": [self.maxlen - n + p for p in positions],
                        "displaced_items": [history[p] for p in positions],
                        "eligible_donors": int(eligible.sum()),
                        "feasible_donors": int(len(support)),
                        "matched": bool(len(support)),
                    }
                    plans.append(plan)
                    if not len(support):
                        continue
                    index = int(support[rng.randrange(len(support))])
                    donor = int(catalog[index])
                    plan.update({"donor": donor, "edits": []})
                    for arm, position, detail, cosine in zip(
                        (["recent", "older"] if stratum == "paired" else ["final"]),
                        positions,
                        details,
                        similarities,
                    ):
                        predecessor, pd, td, sd, raw, target_index = detail
                        detail_record = {
                            "arm": arm,
                            "position": self.maxlen - n + position,
                            "window_position": position,
                            "displaced": history[position],
                            "predecessor": predecessor,
                            "missing_predecessor": predecessor is None,
                            "log_frequency_delta": float(pd[index]),
                            "frequency_standardized_delta": (
                                float(pd[index] / pop_sd) if pop_sd else None
                            ),
                            "log_incoming_probability_delta": float(td[index]),
                            "transition_standardized_delta": (
                                float(td[index] / sd) if sd else None
                            ),
                            "displaced_transition_count": int(raw[target_index]),
                            "donor_transition_count": int(raw[index]),
                            "outgoing_transition_count": int(
                                sum(transitions.get(predecessor, {}).values())
                            ),
                            "text_cosine": (
                                float(cosine[index])
                                if cosine is not None and text[index].nnz
                                else None
                            ),
                        }
                        plan["edits"].append(detail_record)
                    selected.append(plan)
            donors = sorted({p["donor"] for p in selected})
            for pool_name, pool in (
                ("C", candidates),
                ("C_minus", np.setdiff1d(candidates, donors)),
            ):
                if not len(pool):
                    failures.append(
                        {"user": user, "pool": pool_name, "reason": "empty_pool"}
                    )
                    continue
                try:
                    feature = self._features(history)
                    original = self._scores(feature, pool)
                    repeated_feature = self._features(history)
                    repeated = self._scores(repeated_feature, pool)
                    identity = list(history)
                    identity[-1] = history[-1]
                    identity_feature = self._features(tuple(identity))
                    identity_scores = self._scores(identity_feature, pool)
                    order, ranks = self._ranks(original)
                    audit = {
                        "repeated_scores_close": bool(
                            np.allclose(original, repeated, rtol=rtol, atol=atol)
                        ),
                        "identity_scores_close": bool(
                            np.allclose(original, identity_scores, rtol=rtol, atol=atol)
                        ),
                        "repeated_features_close": bool(
                            torch.allclose(
                                feature, repeated_feature, rtol=rtol, atol=atol
                            )
                        ),
                        "identity_features_close": bool(
                            torch.allclose(
                                feature, identity_feature, rtol=rtol, atol=atol
                            )
                        ),
                        "repeated_ranks_equal": bool(
                            np.array_equal(order, self._ranks(repeated)[0])
                        ),
                        "identity_ranks_equal": bool(
                            np.array_equal(order, self._ranks(identity_scores)[0])
                        ),
                        "positive_rescaling_rank_invariant": bool(
                            np.array_equal(order, self._ranks(2 * original)[0])
                        ),
                    }
                    audits[user + ":" + pool_name] = audit
                    if not all(audit.values()):
                        raise ValueError(
                            "Identity/repeated-inference/rescaling audit failed"
                        )
                    pairs = self._pairs(user + ":" + pool_name, len(pool))
                    for plan in selected:
                        arm_rows = {}
                        try:
                            for edit in plan["edits"]:
                                changed_history = list(history)
                                changed_history[edit["window_position"]] = plan["donor"]
                                changed_feature = self._features(tuple(changed_history))
                                changed = self._scores(changed_feature, pool)
                                stats, _, changed_ranks = self._statistics(
                                    original, changed, pairs
                                )
                                stats.update(
                                    {
                                        "user": user,
                                        "draw": plan["draw"],
                                        "no_op": False,
                                        "edit": pool_name
                                        + ":"
                                        + edit["arm"]
                                        + ":"
                                        + plan["predecessor_stratum"],
                                        "feature_l2_change": float(
                                            (changed_feature - feature).norm()
                                        ),
                                        "donor": plan["donor"],
                                        "position": edit["position"],
                                    }
                                )
                                if pool_name == "C":
                                    index = int(np.searchsorted(pool, plan["donor"]))
                                    stats["donor_score_change"] = float(
                                        changed[index] - original[index]
                                    )
                                    stats["donor_rank_change"] = int(
                                        changed_ranks[index] - ranks[index]
                                    )
                                    stats["donor_entered_top10"] = bool(
                                        ranks[index] >= 10 and changed_ranks[index] < 10
                                    )
                                arm_rows[edit["arm"]] = stats
                            rows.extend(arm_rows.values())
                            if plan["stratum"] == "paired":
                                a, b = arm_rows["recent"], arm_rows["older"]
                                contrast = {
                                    "user": user,
                                    "draw": plan["draw"],
                                    "no_op": False,
                                    "edit": pool_name + ":recent_minus_older",
                                }
                                contrast.update(
                                    {
                                        f: (
                                            a[f] - b[f]
                                            if a[f] is not None and b[f] is not None
                                            else None
                                        )
                                        for f in fields
                                    }
                                )
                                contrasts.append(contrast)
                        except (ValueError, RuntimeError) as error:
                            failures.append(
                                {
                                    "user": user,
                                    "pool": pool_name,
                                    "draw": plan["draw"],
                                    "stratum": plan["stratum"],
                                    "error": str(error),
                                }
                            )
                    if pool_name == "C":
                        transition_score = behavioral_score(history[-1:], [1.0])
                        weights = np.exp2(
                            -np.arange(n - 1, -1, -1)
                            / self.settings["control_half_life"]
                        )
                        weights /= weights.sum()
                        weighted = behavioral_score(history, weights)
                        controls[user] = {}
                        for name, score in (
                            ("popularity", prior),
                            ("last_transition", transition_score),
                            ("recency_pairwise", weighted),
                        ):
                            stats, control_order, _ = self._statistics(
                                original, score[pool], pairs
                            )
                            # Probability scores and SASRec logits have unrelated scales.
                            stats.pop("raw_mean_absolute_score_change")
                            controls[user][name] = {
                                **stats,
                                "top10": pool[control_order[:10]].tolist(),
                            }
                except (ValueError, RuntimeError) as error:
                    failures.append(
                        {"user": user, "pool": pool_name, "error": str(error)}
                    )
        combined = rows + contrasts
        users = sorted({r["user"] for r in combined})
        return {
            "status": "observational_training_only",
            "match_plans": plans,
            "matched_plans": sum(p["matched"] for p in plans),
            "attempted_plans": len(plans),
            "unmatched_fraction": (
                float(np.mean([not p["matched"] for p in plans])) if plans else None
            ),
            "donor_reuse": dict(
                Counter(
                    (p["user"] + ":" + str(p["donor"])) for p in plans if p["matched"]
                )
            ),
            "matching_population": "observed training catalog; log interaction frequency; predecessor-specific log smoothed probability SD; missing predecessor uses the prior and is summarized separately",
            "text_status": text_status,
            "records": rows,
            "paired_records": contrasts,
            "summary": self._aggregate(combined),
            "per_user_summary": {
                u: self._aggregate([r for r in combined if r["user"] == u])
                for u in panel
            },
            "audits": audits,
            "failures": failures,
            "leave_one_user_out": {
                u: self._aggregate([r for r in combined if r["user"] != u])
                for u in users
            },
            "training_behavioral_controls": controls,
            "limitations": "Same donor in separate fixed-slot edits; final-position effects separate. Missing predecessor is marked. Timestamps remain attached to slots, producing synthetic item/time pairs. Text is a matching association with item identity, not model word use. Summaries condition on successful fixed matches; no rematching or training-seed uncertainty. Five users/category and sparse overlap limit conclusions. Aggregate four complete category means externally with equal weights; no accuracy improvement is claimed.",
        }

    def _checkpoint_hash(self):
        digest = self._hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("utf-8"))
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        return digest.hexdigest()

    def run_full_diagnosis(self, train_loader):
        import platform
        import time

        # train_loader is intentionally never iterated or inspected.
        module_modes = [(module, module.training) for module in self.model.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_torch_state = torch.get_rng_state().clone()
        cuda_torch_state = None
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            torch.cuda.synchronize(self.device)
            cuda_torch_state = torch.cuda.get_rng_state(devices[0]).clone()
        started = time.perf_counter()
        report = None
        self.rank_cache = {}
        try:
            with torch.random.fork_rng(devices=devices):
                self.model.eval()
                with torch.inference_mode():
                    source = self.view.user_train
                    histories = source.get("History", source)
                    times = source.get("Time", {}) if "History" in source else {}
                    snapshot, counts = {}, Counter()
                    invalid_histories = []
                    for user, history in histories.items():
                        try:
                            frozen = tuple(int(item) for item in history)
                            if any(item < 1 or item > self.itemnum for item in frozen):
                                raise ValueError("Out-of-catalog training history")
                            key = str(user)
                            if key in snapshot:
                                raise ValueError("Duplicate normalized user ID")
                            snapshot[key] = frozen
                        except (ValueError, TypeError, OverflowError) as error:
                            counts["invalid_training_history"] += 1
                            invalid_histories.append(
                                {"user": str(user), "error": str(error)}
                            )
                    frozen_times = {
                        str(user): tuple(str(t) for t in history)
                        for user, history in times.items()
                    }
                    panel, panel_info = self._panel()
                    records, failures, baseline_info = [], [], {}
                    for user in invalid_histories:
                        failures.append({"probe": "training_snapshot", **user})
                    try:
                        margins = self._margins(snapshot, counts)
                    except (ValueError, RuntimeError) as error:
                        failures.append(
                            {"probe": "training_margins", "error": str(error)}
                        )
                        margins = {"status": "failed", "error": str(error)}
                    try:
                        embeddings = self._embeddings()
                    except (ValueError, RuntimeError) as error:
                        failures.append(
                            {"probe": "embedding_geometry", "error": str(error)}
                        )
                        embeddings = {"status": "failed", "error": str(error)}
                    for user in panel:
                        history = snapshot.get(user)
                        if history is None:
                            counts["panel_missing_training_user"] += 1
                            continue
                        if not history:
                            counts["panel_empty_history"] += 1
                            continue
                        baseline = history[-self.maxlen :]
                        if len(baseline) < 2:
                            counts["panel_insufficient_edit_history"] += 1
                        if len(baseline) < 3:
                            counts["panel_ineligible_preserved_anchor_comparison"] += 1
                        candidates = np.setdiff1d(
                            np.arange(1, self.itemnum + 1, dtype=np.int64),
                            np.asarray(history, dtype=np.int64),
                            assume_unique=False,
                        )
                        if not len(candidates):
                            counts["panel_empty_candidate_pool"] += 1
                            continue
                        try:
                            feature = self._features(baseline)
                            original = self._scores(feature, candidates)
                            repeated = self._scores(
                                self._features(baseline), candidates
                            )
                            order, ranks = self._ranks(original)
                            pairs = self._pairs(user, len(candidates))
                            timestamps = frozen_times.get(user, ())
                            cutoff = (
                                timestamps[-1]
                                if len(timestamps) == len(history)
                                else None
                            )
                            baseline_info[user] = {
                                "history_sha256": self._digest(history),
                                "timestamp_history_sha256": (
                                    self._digest(timestamps) if timestamps else None
                                ),
                                "prediction_cutoff": cutoff,
                                "observed_training_interactions": len(history),
                                "window_interactions": len(baseline),
                                "candidate_count": len(candidates),
                                "candidate_sha256": self._digest(candidates.tolist()),
                                "repeated_inference_equal": bool(
                                    np.array_equal(original, repeated)
                                ),
                                "positive_rescaling_rank_invariant": bool(
                                    np.array_equal(
                                        order, self._ranks(2.0 * original)[0]
                                    )
                                ),
                                "baseline_top10": candidates[order[:10]].tolist(),
                            }
                            self.rank_cache[(user, "baseline", 0)] = {
                                "candidates": candidates,
                                "ranks": ranks,
                                "pairs": pairs,
                                "score_summary": {
                                    "min": float(original.min()),
                                    "max": float(original.max()),
                                    "mean": float(original.mean()),
                                    "std": float(original.std()),
                                },
                            }
                        except (ValueError, RuntimeError) as error:
                            failures.append(
                                {"user": user, "edit": "baseline", "error": str(error)}
                            )
                            continue
                        for edit, draw, edited, removed, final_policy in self._edits(
                            user, baseline
                        ):
                            try:
                                changed_feature = self._features(edited)
                                changed = self._scores(changed_feature, candidates)
                                stats, edited_order, edited_ranks = self._statistics(
                                    original, changed, pairs
                                )
                                stats.update(
                                    {
                                        "user": user,
                                        "edit": edit,
                                        "draw": draw,
                                        "removed_interactions": removed,
                                        "final_item_policy": final_policy,
                                        "no_op": edited == baseline,
                                        "feature_l2_change": float(
                                            (changed_feature - feature).norm()
                                        ),
                                        "top10": candidates[edited_order[:10]].tolist(),
                                    }
                                )
                                if edit == "identity":
                                    stats["identity_scores_equal"] = bool(
                                        np.array_equal(original, changed)
                                    )
                                    stats["identity_features_equal"] = bool(
                                        torch.equal(feature, changed_feature)
                                    )
                                records.append(stats)
                                self.rank_cache[(user, edit, draw)] = {
                                    "ranks": edited_ranks,
                                    "score_summary": {
                                        "min": float(changed.min()),
                                        "max": float(changed.max()),
                                        "mean": float(changed.mean()),
                                        "std": float(changed.std()),
                                    },
                                }
                            except (ValueError, RuntimeError) as error:
                                failures.append(
                                    {
                                        "user": user,
                                        "edit": edit,
                                        "draw": draw,
                                        "error": str(error),
                                    }
                                )
                    try:
                        replacement_audit = self._replacement_audit(snapshot, panel)
                    except (
                        ValueError,
                        RuntimeError,
                        TypeError,
                        AttributeError,
                        KeyError,
                    ) as error:
                        failures.append(
                            {"probe": "replacement_audit", "error": str(error)}
                        )
                        replacement_audit = {"status": "failed", "error": str(error)}
                    mapping = getattr(self.view, "item_mapping", None)
                    report = {
                        "metrics": {
                            "dataset": self.dataset,
                            "settings": dict(self.settings),
                            "panel": panel_info,
                            "baseline_audit": baseline_info,
                            "history_edits": self._aggregate(records),
                            "matched_history_edit_contrasts": replacement_audit,
                            "paired_draw_records": records,
                            "training_margins": margins,
                            "embedding_geometry": embeddings,
                            "counts": dict(counts),
                            "failures": failures,
                            "provenance": {
                                "training_history_sha256": self._digest(snapshot),
                                "checkpoint_state_sha256": self._checkpoint_hash(),
                                "item_mapping_sha256": (
                                    self._digest(mapping)
                                    if mapping is not None
                                    else None
                                ),
                                "item_mapping_status": (
                                    "adapter_mapping"
                                    if mapping is not None
                                    else "unavailable; numeric catalog IDs recorded"
                                ),
                                "catalog_itemnum": self.itemnum,
                                "python": platform.python_version(),
                                "numpy": np.__version__,
                                "torch": str(torch.__version__),
                                "cuda": torch.version.cuda,
                            },
                            "unsupported_or_unavailable": {
                                "lexical_arm": "not run: neighborhood annotation and timestamp-matched controls are not established; no semantic claim",
                                "slot_preserving_arm": "fixed-slot ID replacements implemented; explicit positional-ID/key-value interventions remain unavailable",
                                "fault_injection": "not run: bounded-copy corruption/repair and model-specific fault hooks are not supplied",
                                "paired_validation_hr10": "host-owned outcomes and candidate sets unavailable here; no validation surrogate computed",
                                "simulator_traits": "unobserved",
                                "category_breakdowns": {},
                                "trial_ledger": "host must log every inspected trial and add this call's diagnostic cost",
                            },
                        },
                        "metric_definitions": {
                            "history_edits": "Frozen-model sensitivity, not recommendation accuracy. Draws are averaged within user, then users equally. Five supplied users/category are exploratory, not representative. Aggregate four complete category reports externally with equal weights; never treat absent reports as zero.",
                            "matched_history_edit_contrasts": "Five predeclared draw seeds select older/recent nonfinal slots and one shared unseen donor meeting both popularity and predecessor-transition calipers. Require donor-to-displaced TF-IDF cosines to differ by at most 0.1 when both exist. No caliper relaxation. Separate final-slot matches. C excludes complete original history; C_minus additionally excludes all planned donors, with a fresh baseline. Recent-minus-older contrasts use only complete successful pairs. Average matched draws within user, then users equally; coverage and failures remain explicit. Donor rank change is edited minus baseline rank, so negative means promotion. Controls measure disagreement with SASRec, never accuracy. Externally average four complete category reports equally.",
                            "top10_overlap": "Intersection of original and edited top min(10,C), divided by min(10,C); higher is greater stability, not necessarily greater accuracy.",
                            "top10_mean_absolute_rank_displacement": "Mean absolute change in zero-based full-catalog rank over the original top min(10,C); stable descending score sorting breaks ties by ascending item ID.",
                            "strict_pair_flip_rate": "Fraction of a fixed deterministic sample of at most 4096 distinct unordered candidate pairs whose strict ordering reverses. Exclude pairs tied in either condition; report ties and denominator. Undefined denominators return null.",
                            "raw_mean_absolute_score_change": "Mean absolute paired logit change. Scale-dependent; cannot be compared as accuracy or interpreted as invariant sensitivity.",
                            "feature_l2_change": "L2 change in the encoder's final feature at position L-1, separately from score/rank effects; representation-scale-dependent.",
                            "training_margins": "Mean, lower empirical p10, P(negative>positive), and P(margin<0.5) under the stated independent training-only diagnostic distribution. Not directly comparable to live-loader margins; raw values depend on logit scale. Missing cases are null, not zero.",
                            "embedding_geometry": "At most 2048 nonpadding rows sampled with a local seed-42 generator. Mean cosine measures directional alignment only. Centered singular-value entropy gives effective rank, normalized by min(n-1,d); leading mass uses singular values, not squared energy. Neither proves collapse or user diversity.",
                            "edit_policy": "Primary paired replacements preserve sequence length, padding, positions and the final item. Divide nonfinal positions into contiguous halves, assigning any extra position to recent. Final-slot replacements are separate. Prefix shuffling and last-item-only inputs remain distribution-shifting complementary controls. No model retraining or benchmark input changes occur.",
                            "exploratory_user_bootstrap_95_interval": "2000 locally seeded within-category user bootstrap replicates after averaging draws within user, conditional on recorded matches. Undefined with fewer than two users. Excludes rematching, training-seed variation and adaptive search. Leave-one-user-out summaries expose panel sensitivity; five users/category provide weak uncertainty estimates. Full-catalog stability differs from pinned sampled-candidate HR@10.",
                            "selection_policy": "The unchanged host-pinned four-category macro validation HR@10 is the sole selection metric. One changed hit among 200 users/category changes macro HR@10 by 0.00125. No test outcomes, ratings, reviews, item categories, prices, stores, or simulator traits are used.",
                            "integrity": "Identity, repeated inference, and positive score-rescaling checks are observations, not assumed passes. All module flags and Python/NumPy/torch RNG state are restored; live training iterator and worker state are untouched.",
                            "diagnostic_cost": "Synchronized wall seconds for this call, including CPU preparation. On CUDA the same duration is reported as allocated single-GPU seconds for conservative search-budget accounting; this is not measured kernel-active time.",
                        },
                    }
        finally:
            for module, training in module_modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_torch_state)
            if cuda_torch_state is not None:
                torch.cuda.set_rng_state(cuda_torch_state, devices[0])
            if devices:
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            if report is not None:
                current_numpy = np.random.get_state()
                numpy_equal = (
                    numpy_state[0] == current_numpy[0]
                    and np.array_equal(numpy_state[1], current_numpy[1])
                    and numpy_state[2:] == current_numpy[2:]
                )
                report["metrics"]["integrity"] = {
                    "module_flags_restored": all(
                        module.training == flag for module, flag in module_modes
                    ),
                    "python_rng_restored": random.getstate() == python_state,
                    "numpy_rng_restored": bool(numpy_equal),
                    "torch_cpu_rng_restored": bool(
                        torch.equal(cpu_torch_state, torch.get_rng_state())
                    ),
                    "torch_active_cuda_rng_restored": (
                        bool(
                            torch.equal(
                                cuda_torch_state, torch.cuda.get_rng_state(devices[0])
                            )
                        )
                        if devices
                        else None
                    ),
                    "live_training_loader_consumed": False,
                }
                report["metrics"]["diagnostic_cost"] = {
                    "wall_seconds": elapsed,
                    "allocated_single_gpu_seconds": elapsed if devices else 0.0,
                }
        return report

    # Static review: checked fixed-slot edits, shared donors, separate predecessor
    # strata, settings call sites, sparse smoothing, failure reporting, and RNG
    # restoration. No execution, training, or HR@10 improvement is claimed.


### <<< Self_EvolveRec-BLOCK-END
