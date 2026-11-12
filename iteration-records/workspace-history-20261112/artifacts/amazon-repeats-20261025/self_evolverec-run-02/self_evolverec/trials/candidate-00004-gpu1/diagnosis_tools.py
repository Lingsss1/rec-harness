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


### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic scoring, expose failures, and compute matched user contrasts
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
            "deletion_budget": "min(3,max(1,floor(0.2*n)),n-1), n>=2",
            "candidate_policy": "IDs 1..embedding_rows-1 minus complete original training history",
            "tie_policy": "ascending candidate ID, stable descending score",
            "position_policy": "compact edits, retain latest L, left-pad; final query at L-1",
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
        n = len(history)
        yield "identity", 0, history, 0, "retained"
        if n < 2:
            return
        k = min(3, max(1, n // 5), n - 1)
        yield "recent_removal", 0, history[:-k], k, "removed"
        yield "old_removal", 0, history[k:], k, "retained"
        yield "last_item_only", 0, history[-1:], n - 1, "retained"
        preserved_k = min(k, n - 2)
        if preserved_k:
            yield "preanchor_removal", 0, history[: n - 1 - preserved_k] + history[
                -1:
            ], preserved_k, "retained"
            yield "early_anchor_preserved", 0, history[
                preserved_k:
            ], preserved_k, "retained"
        for draw in range(self.settings["draws"]):
            rng = random.Random(self._seed(user, "random_deletion", draw))
            removed = set(rng.sample(range(n), k))
            edited = tuple(item for i, item in enumerate(history) if i not in removed)
            yield "random_deletion", draw, edited, k, (
                "removed" if n - 1 in removed else "retained"
            )
            for name, fixed in (("shuffle", False), ("prefix_shuffle", True)):
                rng = random.Random(self._seed(user, name, draw))
                prefix = list(history[:-1] if fixed else history)
                rng.shuffle(prefix)
                edited = tuple(prefix) + (history[-1:] if fixed else ())
                yield name, draw, edited, 0, "retained" if fixed else "unrestricted"

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

    def _matched_contrasts(self, records):
        """Paired effects, using identical users and complete stochastic draws."""
        comparisons = (
            ("recent_minus_old", "recent_removal", "old_removal"),
            ("preanchor_minus_early", "preanchor_removal", "early_anchor_preserved"),
            ("shuffle_minus_prefix_shuffle", "shuffle", "prefix_shuffle"),
        )
        fields = (
            "top10_overlap",
            "top10_mean_absolute_rank_displacement",
            "strict_pair_flip_rate",
            "raw_mean_absolute_score_change",
            "feature_l2_change",
        )
        result = {}
        users = sorted({row["user"] for row in records})
        for name, left, right in comparisons:
            expected = self.settings["draws"] if left == "shuffle" else 1
            expected_ids = set(range(expected))
            paired, excluded = [], []
            for user in users:
                a = [r for r in records if r["user"] == user and r["edit"] == left]
                b = [r for r in records if r["user"] == user and r["edit"] == right]
                if (
                    len(a) != expected
                    or len(b) != expected
                    or {r["draw"] for r in a} != expected_ids
                    or {r["draw"] for r in b} != expected_ids
                ):
                    excluded.append(user)
                    continue
                row = {"user": user, "edit": name, "no_op": False}
                for field in fields:
                    av, bv = [r[field] for r in a], [r[field] for r in b]
                    row[field] = (
                        float(np.mean(av) - np.mean(bv))
                        if all(v is not None for v in av + bv)
                        else None
                    )
                paired.append(row)
            summary = self._aggregate(paired).get(name, {})
            summary.update(
                {
                    "direction": f"{left} minus {right}",
                    "paired_user_ids": [r["user"] for r in paired],
                    "excluded_incomplete_or_ineligible_user_ids": excluded,
                    "interpretation": (
                        "Difference in frozen-input sensitivity, not an accuracy effect. "
                        "Recent/old removal differ in final-item policy; preanchor/early "
                        "both preserve the anchor. Shuffle conditions differ in whether "
                        "the final slot is fixed. Compaction also changes positions."
                    ),
                }
            )
            result[name] = summary
        return result

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
                    mapping = getattr(self.view, "item_mapping", None)
                    report = {
                        "metrics": {
                            "dataset": self.dataset,
                            "settings": dict(self.settings),
                            "panel": panel_info,
                            "baseline_audit": baseline_info,
                            "history_edits": self._aggregate(records),
                            "matched_history_edit_contrasts": self._matched_contrasts(
                                records
                            ),
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
                                "slot_preserving_arm": "unsupported: adapter exposes no verified positional-ID and key/value mask intervention",
                                "fault_injection": "not run: bounded-copy corruption/repair and model-specific fault hooks are not supplied",
                                "paired_validation_hr10": "host-owned outcomes and candidate sets unavailable here; no validation surrogate computed",
                                "simulator_traits": "unobserved",
                                "category_breakdowns": {},
                                "trial_ledger": "host must log every inspected trial and add this call's diagnostic cost",
                            },
                        },
                        "metric_definitions": {
                            "history_edits": "Frozen-model sensitivity, not recommendation accuracy. Draws are averaged within user, then users equally. Five supplied users/category are exploratory, not representative. Aggregate four complete category reports externally with equal weights; never treat absent reports as zero.",
                            "matched_history_edit_contrasts": "Within-user differences between edit summaries, averaged equally across paired users. Require all preregistered draws in both conditions. Bootstrap paired user differences with 2000 seeded replicates. Negative overlap contrasts mean less overlap under the first edit; signs of other fields follow their definitions. These are sensitivity comparisons, not recommendation improvements. Users absent from every successful condition remain visible in panel metadata and failures.",
                            "top10_overlap": "Intersection of original and edited top min(10,C), divided by min(10,C); higher is greater stability, not necessarily greater accuracy.",
                            "top10_mean_absolute_rank_displacement": "Mean absolute change in zero-based full-catalog rank over the original top min(10,C); stable descending score sorting breaks ties by ascending item ID.",
                            "strict_pair_flip_rate": "Fraction of a fixed deterministic sample of at most 4096 distinct unordered candidate pairs whose strict ordering reverses. Exclude pairs tied in either condition; report ties and denominator. Undefined denominators return null.",
                            "raw_mean_absolute_score_change": "Mean absolute paired logit change. Scale-dependent; cannot be compared as accuracy or interpreted as invariant sensitivity.",
                            "feature_l2_change": "L2 change in the encoder's final feature at position L-1, separately from score/rank effects; representation-scale-dependent.",
                            "training_margins": "Mean, lower empirical p10, P(negative>positive), and P(margin<0.5) under the stated independent training-only diagnostic distribution. Not directly comparable to live-loader margins; raw values depend on logit scale. Missing cases are null, not zero.",
                            "embedding_geometry": "At most 2048 nonpadding rows sampled with a local seed-42 generator. Mean cosine measures directional alignment only. Centered singular-value entropy gives effective rank, normalized by min(n-1,d); leading mass uses singular values, not squared energy. Neither proves collapse or user diversity.",
                            "edit_policy": "Recent deletion removes the final item; old deletion retains it. Preanchor/early comparisons both retain the final item and use min(k,n-2) deletions. Random deletion has the same primary k and reports its final-item policy. Compaction changes positions as well as content; these are not retrained ablations.",
                            "exploratory_user_bootstrap_95_interval": "2000 seeded within-category user bootstrap replicates after averaging draws within user. Undefined with fewer than two users. Full-catalog stability differs from pinned sampled-candidate HR@10; adaptive-selection intervals remain exploratory.",
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

    # Static review: checked score shapes, complement sampling, paired contrasts,
    # explicit failure reporting, and RNG/mode restoration paths. No execution or
    # training was performed; predictive HR@10 improvements are not claimed.


### <<< Self_EvolveRec-BLOCK-END
