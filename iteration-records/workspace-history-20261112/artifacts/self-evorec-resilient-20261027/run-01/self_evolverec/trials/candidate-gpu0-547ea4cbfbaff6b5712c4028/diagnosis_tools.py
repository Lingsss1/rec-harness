import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Isolated and persisted diagnostic resources
import os
import hashlib
from datetime import datetime, timezone

### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Ground interpretations in measured training sensitivity
    - Status describes the available evidence; without calibrated thresholds, do not claim mathematical health, collapse, or critical failure.
    - Explicitly report null measurements, eligibility counts, unsupported controls, and scope limitations.
    - These are sampled in-sample training statistics, not validation accuracy, HR@10, or causal preference effects.
    - Neither larger perturbation effects nor smaller embedding alignment establishes better recommendations.
    - Do not infer ownership, satisfaction, compatibility, equipment, ratings, review sentiment, or simulator diversity/conformity traits.
    - Distinguish documented prerequisites from unknown equipment, possible redundancy from established ownership, and media similarity from present desirability.
    - Do not claim empirical gains. Benchmark selection remains independently measured macro validation HR@10.
    ### <<< Self_EvolveRec-BLOCK-END
    """


### >>> Self_EvolveRec-BLOCK-START: Paired diagnostics with efficient resampling and complete missing-evidence reporting
class DiagnosisProbe:
    CONDITIONS = (
        "delete_newest",
        "delete_oldest",
        "delete_random",
        "shuffle",
        "shuffle_keep_latest",
        "availability_shuffle",
        "availability_shuffle_keep_latest",
    )

    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.user_train = view.user_train
        self.maxlen = int(getattr(self.model, "maxlen", getattr(view, "maxlen", 0)))
        if self.maxlen < 1:
            raise ValueError("Diagnostic adapter requires the trained maxlen")
        self.itemnum = int(view.item_emb.weight.shape[0]) - 1
        self.dataset = getattr(
            view, "dataset_name", getattr(self.model, "dataset_name", None)
        )
        self.config = {
            "version": 1,
            "seed": 42,
            "seeds": [42, 43, 44, 45, 46],
            "batch_size": 128,
            "max_batches": 10,
            "maxlen": self.maxlen,
            "itemnum": self.itemnum,
            "dataset": self.dataset,
            "conditions": list(self.CONDITIONS),
            "deletion_fraction": 0.25,
            "transition": "final training transition; prefix excludes target",
        }

    @staticmethod
    def _timestamp(value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                result = parsed.timestamp() * 1000.0
            except (TypeError, ValueError, OverflowError):
                return None
        return result if np.isfinite(result) else None

    @staticmethod
    def _negative(history, itemnum, rng):
        excluded = set(history)
        for _ in range(64):
            item = rng.randint(1, itemnum)
            if item not in excluded:
                return item
        # Uniform rank in the complement avoids an unbounded rejection loop.
        ordered = sorted(excluded)
        rank = rng.randrange(itemnum - len(ordered)) + 1
        for item in ordered:
            if item <= rank:
                rank += 1
            else:
                break
        return rank

    def _panel(self):
        raw = self.user_train
        histories = raw.get("History", raw)
        times = raw.get("Time", {}) if "History" in raw else {}
        records, first = [], {}
        time_ok = True
        for user, values in sorted(histories.items(), key=lambda pair: str(pair[0])):
            history = [int(item) for item in values]
            if any(item < 1 or item > self.itemnum for item in history):
                raise ValueError("Training histories contain invalid item IDs")
            stamps = times.get(user, times.get(str(user), []))
            stamps = [self._timestamp(value) for value in stamps]
            valid_time = (
                len(stamps) == len(history)
                and all(value is not None for value in stamps)
                and all(a <= b for a, b in zip(stamps, stamps[1:]))
            )
            if history and not valid_time:
                time_ok = False
            if valid_time:
                for item, stamp in zip(history, stamps):
                    first[item] = min(first.get(item, stamp), stamp)
            records.append((str(user), history, stamps if valid_time else None))
        payload = {"config": self.config, "training": records}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"panel_{digest}.json")
        if os.path.isfile(path):
            with open(path) as handle:
                panel = json.load(handle)
            if panel["fingerprint"] != digest or panel["config"] != self.config:
                raise ValueError("Diagnostic panel configuration mismatch")
            return panel, path

        eligible = [
            row
            for row in records
            if len(row[1]) >= 2 and len(set(row[1])) < self.itemnum
        ]
        rng = random.Random(42)
        cap = self.config["batch_size"] * self.config["max_batches"]
        selected = rng.sample(eligible, cap) if len(eligible) > cap else eligible
        rows = []
        for user, history, stamps in selected:
            prefix = history[:-1][-self.maxlen :]
            prefix_times = stamps[:-1][-self.maxlen :] if stamps is not None else None
            row = {
                "user": user,
                "prefix": prefix,
                "times": prefix_times,
                "target": history[-1],
                "negative": self._negative(history, self.itemnum, rng),
                "variants": {},
            }
            for condition in self.CONDITIONS:
                draws = []
                for seed in self.config["seeds"]:
                    key = f"{user}:{condition}:{seed}"
                    local_seed = int.from_bytes(
                        hashlib.sha256(key.encode()).digest()[:8], "big"
                    )
                    draws.append(
                        self._variant(
                            prefix,
                            prefix_times,
                            first,
                            condition,
                            random.Random(local_seed),
                            time_ok,
                        )
                    )
                row["variants"][condition] = draws
            rows.append(row)
        panel = {
            "fingerprint": digest,
            "config": self.config,
            "rows": rows,
            "training_users": len(records),
            "eligible_users": len(eligible),
            "complete_training_timestamps": time_ok,
        }
        temporary = path + f".{os.getpid()}.tmp"
        with open(temporary, "w") as handle:
            json.dump(panel, handle, allow_nan=False)
        os.replace(temporary, path)
        return panel, path

    @staticmethod
    def _variant(prefix, stamps, first, condition, rng, time_ok):
        n = len(prefix)
        if n < 2:
            return {
                "supported": False,
                "reason": "Retained prefix has fewer than two items",
            }
        out = list(prefix)
        deleted, partners, nonself, attempts = [], [], 0, 0
        if condition.startswith("delete_"):
            k = min(n - 1, max(1, int(0.25 * n)))
            if condition == "delete_newest":
                deleted = list(range(n - k, n))
            elif condition == "delete_oldest":
                deleted = list(range(k))
            else:
                deleted = sorted(rng.sample(range(n), k))
            removed = set(deleted)
            out = [item for i, item in enumerate(prefix) if i not in removed]
        elif condition.startswith("availability_"):
            if not time_ok or stamps is None:
                return {
                    "supported": False,
                    "reason": "Complete aligned monotone training timestamps unavailable",
                }
            # Freeze each occurrence at its item's training minimum, including ties.
            frozen = {i for i, item in enumerate(prefix) if stamps[i] == first[item]}
            if condition.endswith("keep_latest"):
                frozen.add(n - 1)
            for i in range(n):
                if i in frozen:
                    continue
                allowed = [
                    j
                    for j in range(i, n)
                    if j not in frozen
                    and first[out[j]] <= stamps[i]
                    and first[out[i]] <= stamps[j]
                ]
                partners.append(len(allowed))
                if not allowed:
                    raise ValueError(
                        "Availability permutation lost its feasible self-swap"
                    )
                j = rng.choice(allowed)
                attempts += 1
                nonself += int(j != i)
                out[i], out[j] = out[j], out[i]
        else:
            end = n - 1 if condition.endswith("keep_latest") else n
            movable = out[:end]
            rng.shuffle(movable)
            out[:end] = movable
        # IDs and source positions are persisted; timestamps stay in original slots for shuffles.
        return {
            "supported": True,
            "prefix": out,
            "deleted_positions": deleted,
            "times": (
                [value for i, value in enumerate(stamps) if i not in set(deleted)]
                if stamps is not None
                else None
            ),
            "displaced_occurrence_fraction": (
                sum(a != b for a, b in zip(prefix, out)) / n if not deleted else None
            ),
            "eligible_partner_mean": float(np.mean(partners)) if partners else None,
            "nonself_swap_rate": nonself / attempts if attempts else None,
            "swap_attempts": attempts,
        }

    def _measure(self, prefixes, targets, negatives):
        sequence = torch.zeros(
            (len(prefixes), self.maxlen), dtype=torch.long, device=self.device
        )
        for i, prefix in enumerate(prefixes):
            sequence[i, -len(prefix) :] = torch.as_tensor(prefix, device=self.device)
        features = self.view.log2feats(sequence)
        if not isinstance(features, torch.Tensor):
            raise TypeError("log2feats must return a tensor")
        candidates = torch.tensor(
            list(zip(targets, negatives)), dtype=torch.long, device=self.device
        )
        if features.ndim == 3 and features.shape[:2] == sequence.shape:
            final = features[:, -1, :]
            # Preserve the compatibility contract's sequence and candidate axes.
            scores = self.view.predict_score(
                features[:, -1:, :], candidates[:, None, :]
            )
            expected = (len(prefixes), 1, 2)
            if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected:
                raise ValueError(f"predict_score must return {expected}")
            scores = scores[:, 0, :]
        elif features.ndim == 2 and features.shape[0] == len(prefixes):
            if getattr(self.view, "diagnostic_feature_contract", None) != "final_BD":
                raise ValueError(
                    "B x D features require diagnostic_feature_contract='final_BD'"
                )
            final = features
            scores = self.view.predict_score(final, candidates)
            if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != (
                len(prefixes),
                2,
            ):
                raise ValueError("Final-feature predict_score must return B x 2")
        else:
            raise ValueError(f"Unexpected log2feats shape: {tuple(features.shape)}")
        if not torch.isfinite(final).all() or not torch.isfinite(scores).all():
            raise ValueError("Nonfinite diagnostic features or scores")
        return final.float().cpu(), (scores[:, 0] - scores[:, 1]).float().cpu().numpy()

    def _embedding_alignment(self):
        embedding = self.view.item_emb.weight.detach()[1:]
        generator = torch.Generator(device="cpu").manual_seed(42)
        indices = torch.randperm(len(embedding), generator=generator)[:2048]
        sample = embedding.index_select(0, indices.to(embedding.device)).float()
        if not torch.isfinite(sample).all():
            raise ValueError("Nonfinite sampled item embeddings")
        valid = sample.norm(dim=-1) > 0
        sample = sample[valid]
        sample = sample / sample.norm(dim=-1, keepdim=True)
        count = len(sample)
        if count < 2:
            cosine = None
        else:
            # Exact off-diagonal average without allocating a Gram matrix.
            value = (sample.sum(0).square().sum() - sample.square().sum()) / (
                count * (count - 1)
            )
            cosine = float(value.clamp(-1, 1).item())
        return {
            "mean_cosine": cosine,
            "mapped_alignment": None if cosine is None else (cosine + 1) / 2,
            "sampled_items": len(indices),
            "nonzero_items": count,
            "reason": (
                "Fewer than two nonzero sampled embeddings" if cosine is None else None
            ),
        }

    @staticmethod
    def _summary(rows):
        result = {}
        keys = sorted({key for row in rows for key in row})
        for key in keys:
            values = np.asarray(
                [row[key] for row in rows if row.get(key) is not None], dtype=float
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite diagnostic observations for {key}")
            interval = None
            if len(values) >= 20:
                rng = np.random.default_rng(42)
                replicates = np.empty(1000)
                # Bound temporary memory while avoiding 1000 Python iterations
                # for every metric. Each row resamples entire user summaries.
                for start in range(0, 1000, 100):
                    stop = min(start + 100, 1000)
                    indices = rng.integers(
                        0, len(values), size=(stop - start, len(values))
                    )
                    replicates[start:stop] = values[indices].mean(axis=1)
                interval = np.quantile(replicates, [0.025, 0.975]).tolist()
            result[key] = {
                "mean": float(values.mean()) if len(values) else None,
                "eligible_users": len(values),
                "reason": (
                    None if len(values) else "No eligible non-null user observations"
                ),
                "user_bootstrap_95": interval,
                "interval_reason": (
                    None
                    if interval is not None
                    else "Requires at least 20 eligible users"
                ),
            }
        return result

    def _run(self):
        panel, path = self._panel()
        rows = panel["rows"]
        intact_rows, intact_margins = [], []
        paired = {condition: [] for condition in self.CONDITIONS}
        unavailable = {condition: {} for condition in self.CONDITIONS}
        batch_size = self.config["batch_size"]
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            targets = [row["target"] for row in batch]
            negatives = [row["negative"] for row in batch]
            reference, margin = self._measure(
                [row["prefix"] for row in batch], targets, negatives
            )
            # Identical complete intervention batches have identical outputs in
            # eval mode. Reuse deterministic deletion draws and identity shuffles.
            # This cache is local to one checkpoint and one fixed-candidate batch.
            measurement_cache = {
                (
                    tuple(range(len(batch))),
                    tuple(tuple(row["prefix"]) for row in batch),
                ): (reference, margin)
            }
            intact_margins.extend(margin.tolist())
            for value in margin:
                intact_rows.append(
                    {
                        "margin": float(value),
                        "below_zero": float(value < 0),
                        "below_0_5": float(value < 0.5),
                        "tie": float(value == 0),
                    }
                )
            for condition in self.CONDITIONS:
                user_draws = [[] for _ in batch]
                for draw in range(len(self.config["seeds"])):
                    selected = [
                        i
                        for i, row in enumerate(batch)
                        if row["variants"][condition][draw]["supported"]
                    ]
                    if not selected:
                        continue
                    variants = [batch[i]["variants"][condition][draw] for i in selected]
                    cache_key = (
                        tuple(selected),
                        tuple(tuple(variant["prefix"]) for variant in variants),
                    )
                    if cache_key not in measurement_cache:
                        measurement_cache[cache_key] = self._measure(
                            [variant["prefix"] for variant in variants],
                            [targets[i] for i in selected],
                            [negatives[i] for i in selected],
                        )
                    features, changed_margin = measurement_cache[cache_key]
                    original = reference[selected]
                    norm_ok = (original.norm(dim=-1) > 0) & (features.norm(dim=-1) > 0)
                    # Explicit normalization handles every positive norm consistently;
                    # cosine_similarity's default epsilon distorts tiny nonzero features.
                    original_norm = original.norm(dim=-1, keepdim=True)
                    changed_norm = features.norm(dim=-1, keepdim=True)
                    original_unit = original / torch.where(
                        original_norm > 0, original_norm, torch.ones_like(original_norm)
                    )
                    changed_unit = features / torch.where(
                        changed_norm > 0, changed_norm, torch.ones_like(changed_norm)
                    )
                    distance = 1 - (original_unit * changed_unit).sum(-1).clamp(-1, 1)
                    for j, i in enumerate(selected):
                        variant = variants[j]
                        user_draws[i].append(
                            {
                                "margin_delta": float(changed_margin[j] - margin[i]),
                                "ordering_flip": float(
                                    (changed_margin[j] > 0) != (margin[i] > 0)
                                ),
                                "perturbed_tie": float(changed_margin[j] == 0),
                                "feature_cosine_distance": (
                                    float(distance[j]) if norm_ok[j] else None
                                ),
                                "displaced_occurrence_fraction": variant[
                                    "displaced_occurrence_fraction"
                                ],
                                "eligible_partner_mean": variant[
                                    "eligible_partner_mean"
                                ],
                                "nonself_swap_rate": variant["nonself_swap_rate"],
                            }
                        )
                for i, draws in enumerate(user_draws):
                    if not draws:
                        reason = batch[i]["variants"][condition][0]["reason"]
                        unavailable[condition][reason] = (
                            unavailable[condition].get(reason, 0) + 1
                        )
                        continue
                    averaged = {}
                    for key in draws[0]:
                        values = [draw[key] for draw in draws if draw[key] is not None]
                        averaged[key] = float(np.mean(values)) if values else None
                    paired[condition].append(averaged)
        conditions = {}
        for condition in self.CONDITIONS:
            conditions[condition] = {
                "eligible_users": len(paired[condition]),
                "panel_coverage": len(paired[condition]) / len(rows) if rows else None,
                "metrics": (
                    self._summary(paired[condition]) if paired[condition] else None
                ),
                "unavailable_users_by_reason": unavailable[condition],
                "reason": "No eligible panel users" if not paired[condition] else None,
            }
        unsupported = {
            "position_preserving_deletion": "No verified adapter for masking holes as attention keys and states and gathering newest surviving features",
            "position_only": "No explicit verified position-ID interface",
        }
        conditions.update(
            {
                key: {"metrics": None, "eligible_users": 0, "reason": reason}
                for key, reason in unsupported.items()
            }
        )
        return {
            "metrics": {
                "dataset": self.dataset,
                "panel": {
                    "path": path,
                    "fingerprint": panel["fingerprint"],
                    "training_users": panel["training_users"],
                    "eligible_users": panel["eligible_users"],
                    "selected_users": len(rows),
                    "eligible_user_coverage": (
                        len(rows) / panel["eligible_users"]
                        if panel["eligible_users"]
                        else None
                    ),
                    "complete_training_timestamps": panel[
                        "complete_training_timestamps"
                    ],
                },
                "embedding_alignment": self._embedding_alignment(),
                "intact": {
                    "metrics": self._summary(intact_rows) if intact_rows else None,
                    "margin_p10": (
                        float(np.quantile(intact_margins, 0.1, method="inverted_cdf"))
                        if intact_margins
                        else None
                    ),
                    "eligible_users": len(intact_rows),
                    "reason": "No eligible training users" if not intact_rows else None,
                },
                "conditions": conditions,
            },
            "metric_definitions": {
                "configuration": self.config,
                "scope": "One final training transition per selected user; in-sample, fixed positive and one fixed uniform training-history-excluded negative. Historical repeats of the target are retained.",
                "features": "Oldest interactions truncated; survivors right-aligned with left padding; baseline absolute slots 0..maxlen-1; final slot scored. BxD output requires explicit final_BD declaration.",
                "embedding_alignment": "Mean off-diagonal normalized cosine of up to 2048 seed-42 sampled nonpadding embeddings, excluding zero norms. mapped_alignment=(cosine+1)/2; zero cosine maps to 0.5. Neither metric establishes collapse or recommendation diversity.",
                "intact": "User means of fixed positive-minus-negative margin, margin<0, margin<0.5, and exact ties; p10 is the inverse empirical CDF over users. These two-candidate statistics are not HR@10. The 0.5 cutoff is descriptive, not calibrated.",
                "margin_delta": "Perturbed minus intact margin for identical target and negative; draws averaged within user before averaging users.",
                "ordering_flip": "Fraction of user/draw pairs for which margin>0 changes truth value. Exact ties are nonpositive and reported separately.",
                "feature_cosine_distance": "1-cosine between intact and perturbed final features; zero-norm pairs excluded. eligible_users reports metric-specific coverage.",
                "deletion": "For n>=2 remove min(n-1,max(1,floor(0.25*n))) newest, oldest, or uniformly sampled positions. Compact survivors in order. Content, context, and absolute positions can change together; persisted deletion indices refer to retained-prefix positions.",
                "shuffle": "Permute nonpadding IDs only, preserving multiplicities, targets and candidate IDs. Timestamp slots stay fixed. keep_latest excludes the newest slot from both swap roles.",
                "availability_shuffle": "Window-restricted adaptation: minima computed over complete public training histories. Freeze all occurrences at their item's minimum timestamp, including ties. Scan unfrozen slots chronologically and uniformly choose feasible unfrozen partners j>=i with both swapped items available at destination times. Self-swaps allowed. Not a reproduction of complete-history author shuffling; training minima approximate availability, not release dates.",
                "swap_statistics": "Displacement is fraction of original slots with changed item ID; repeated IDs can hide swaps. eligible_partner_mean averages feasible partner counts over attempted slots. nonself_swap_rate divides nonself swaps by attempted slots. No attempted slots gives null, not zero. Deletion displacement is undefined.",
                "uncertainty": "1000 seed-42 user-cluster percentile bootstrap replicates of user-averaged measurements when >=20 eligible users. Conditional on fixed panel, candidates and fitted model; excludes training seed variation and search bias. No cross-checkpoint trajectory comparison or calibrated health threshold is claimed.",
                "aggregation": "This result is per dataset. aggregate_diagnoses requires all four named datasets and computes equal-weight macro means only where all four provide a measurement; it does not pool users or p10 values.",
                "unavailable": "Null plus reason denotes empty or unsupported evidence. Item categories, ratings, prices, stores, review text and simulator traits are unavailable and unused. No larger-candidate ranks, top-10 overlaps, full-catalog quality or simulator conclusions are measured.",
                "state_isolation": "Active loader is never iterated or inspected. Panel and permutations are persisted independently; local RNGs only. Global RNGs, explicit generators in view/model fields and standard containers, heterogeneous module modes and registered buffers are protected. Buffer mutation is restored and rejected. Identical intervention batches reuse measurements within a checkpoint, requiring deterministic eval forwards. Arbitrary Python-side mutations, custom stochastic eval behavior and generators hidden inside arbitrary objects are outside this adapter's guarantee.",
                "verification_limit": "Enabled-versus-disabled subsequent optimizer-step equivalence has not been empirically tested here. No training or validation improvement is claimed. Diagnostic calls are host scheduled; no checkpoint-trajectory inference is made.",
            },
        }

    def run_full_diagnosis(self, train_loader):
        # Deliberately never consume the active loader, sampler, or workers.
        modes = [(module, module.training) for module in self.model.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        buffers = [
            (name, buffer, buffer.detach().clone())
            for name, buffer in self.model.named_buffers()
        ]
        generators = []
        seen = set()

        def capture_generator(value):
            # Inspect explicit generator fields and their containers only.
            # Never traverse a loader, sampler, worker, or arbitrary object.
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Generator):
                generators.append((value, value.get_state()))
            elif isinstance(value, np.random.Generator):
                import copy

                generators.append((value, copy.deepcopy(value.bit_generator.state)))
            elif isinstance(value, np.random.RandomState):
                generators.append((value, value.get_state()))
            elif isinstance(value, random.Random):
                generators.append((value, value.getstate()))
            elif isinstance(value, dict):
                for child in value.values():
                    capture_generator(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    capture_generator(child)

        for owner in [self.view, *self.model.modules()]:
            for value in vars(owner).values():
                capture_generator(value)
        mutation = []
        try:
            self.model.eval()
            with torch.no_grad():
                result = self._run()
            return result
        finally:
            try:
                with torch.no_grad():
                    current = dict(self.model.named_buffers())
                    if set(current) != {name for name, _, _ in buffers}:
                        mutation.append("buffer registration")
                    for name, original, saved in buffers:
                        value = current.get(name)
                        equal = value is original and value.shape == saved.shape
                        if equal:
                            equal = torch.equal(value, saved) or (
                                value.is_floating_point()
                                and torch.allclose(
                                    value, saved, rtol=0, atol=0, equal_nan=True
                                )
                            )
                        if not equal:
                            mutation.append(name)
                            parent_name, _, leaf = name.rpartition(".")
                            parent = (
                                self.model.get_submodule(parent_name)
                                if parent_name
                                else self.model
                            )
                            # Preserve external references and shared-buffer aliases.
                            # Restore the original tensor before re-registering it.
                            if (
                                original.shape == saved.shape
                                and original.dtype == saved.dtype
                                and original.device == saved.device
                            ):
                                original.copy_(saved)
                            else:
                                original.data = saved.clone()
                            parent._buffers[leaf] = original
                    original_names = {name for name, _, _ in buffers}
                    for name in set(current) - original_names:
                        parent_name, _, leaf = name.rpartition(".")
                        parent = (
                            self.model.get_submodule(parent_name)
                            if parent_name
                            else self.model
                        )
                        del parent._buffers[leaf]
            finally:
                for module, mode in modes:
                    module.training = mode
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(cpu_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
                for generator, state in generators:
                    if isinstance(generator, torch.Generator):
                        generator.set_state(state)
                    elif isinstance(generator, np.random.Generator):
                        generator.bit_generator.state = state
                    elif isinstance(generator, np.random.RandomState):
                        generator.set_state(state)
                    else:
                        generator.setstate(state)
            if mutation:
                raise RuntimeError(
                    f"Diagnostic forward mutated registered buffers: {mutation}"
                )


def aggregate_diagnoses(results):
    """Combine separately produced category diagnoses; never substitute missing routes."""
    expected = {
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    }
    mapped = {}
    for result in results:
        name = result["metrics"]["dataset"]
        if name in mapped:
            raise ValueError("Provide exactly one checkpoint result per dataset")
        mapped[name] = result
    if set(mapped) != expected:
        return {
            "metrics": {"macro": None, "datasets": mapped},
            "metric_definitions": {
                "reason": "All four explicit dataset names are required for macro aggregation"
            },
        }

    def combine(nodes):
        names = sorted(expected)
        summaries = [
            isinstance(node, dict) and "mean" in node and "eligible_users" in node
            for node in nodes
        ]
        if any(summaries):
            values = [
                node["mean"] if valid else None for node, valid in zip(nodes, summaries)
            ]
            missing = [name for name, value in zip(names, values) if value is None]
            return {
                "mean": float(np.mean(values)) if not missing else None,
                "eligible_users_by_dataset": {
                    name: node["eligible_users"] if valid else 0
                    for name, node, valid in zip(names, nodes, summaries)
                },
                "reason": (
                    "Missing measurement in: " + ", ".join(missing) if missing else None
                ),
                "user_bootstrap_95": None,
                "interval_reason": "Macro bootstrap is not implemented",
            }
        dictionaries = [node for node in nodes if isinstance(node, dict)]
        # Use the union: intersection silently removed measurements whenever a
        # category had no eligible users. Missing category leaves remain null.
        keys = sorted(set().union(*(set(node) for node in dictionaries)))
        children = {}
        for key in keys:
            values = [
                node.get(key) if isinstance(node, dict) else None for node in nodes
            ]
            if any(isinstance(value, dict) for value in values):
                children[key] = combine(values)
        if children:
            return children
        return {
            "mean": None,
            "reason": "No aggregatable measurements across all four datasets",
        }

    ordered = [mapped[name]["metrics"] for name in sorted(expected)]
    macro = {
        "intact": combine([node["intact"] for node in ordered]),
        "conditions": combine([node["conditions"] for node in ordered]),
    }
    alignments = [node["embedding_alignment"]["mean_cosine"] for node in ordered]
    macro["embedding_alignment_mean_cosine"] = (
        float(np.mean(alignments))
        if all(value is not None for value in alignments)
        else None
    )
    return {
        "metrics": {"macro": macro, "datasets": mapped},
        "metric_definitions": {
            "macro": "Equal-weight average of four dataset means after averaging draws within users. Missing category measurements produce null. Category results retain counts, coverage, definitions and conditional intervals; p10 values are not averaged.",
            "uncertainty": "Macro intervals and cross-checkpoint trajectory comparisons unavailable; do not average category interval endpoints.",
        },
    }


### <<< Self_EvolveRec-BLOCK-END
