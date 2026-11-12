import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Architecture-aware diagnostic utilities
import random
import inspect
from contextlib import contextmanager
from collections import Counter
from itertools import islice

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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret only observed diagnostic evidence
    - Null values and zero measurement counts indicate unavailable evidence, not healthy behavior.
    - Use measurement counts, missingness, and limitations included in the metrics.
    - Training-pair separation and training-prefix perturbations are fitted-model diagnostics, not validation or test performance.
    - Embedding cosine alignment does not establish recommendation diversity, collapse, or user preferences by itself.
    - Perturbation differences describe model dependence and may involve out-of-distribution inputs; they do not identify causal user preferences.
    - Item categories, ratings, reviews, prices, stores, and simulator traits are unavailable. Do not infer them.
    - Diagnostic HR@10 uses training-prefix targets and diagnostic candidates; it is not the official benchmark HR@10.
    - Scalar gate magnitude alone does not identify branch usefulness because branch features may be correlated.
    # Review checks: traced feature/scoring calls, final-valid-state readout,
    # loader consumption, mode/RNG restoration, candidate exclusion, control
    # windows, and missing-comparison counts. No execution or performance
    # measurements were performed; no validation improvement is claimed.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the view contract and probe the trained scoring path
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.embedding_sample_count = 0
        self.embedding_excluded_count = 0
        self.feature_samples = []

    @contextmanager
    def _evaluation(self):
        modes = [(module, module.training) for module in self.module.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
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
            # Preserve training RNG streams, including draws made by the loader.
            # Consuming an iterator still advances its cursor and worker state.
            with torch.random.fork_rng(devices=cuda_devices):
                self.module.eval()
                with torch.no_grad():
                    yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training

    def _feature_function(self):
        fn = getattr(self.module, "log2feats", None)
        return fn if callable(fn) else self.model.log2feats

    def _features(self, sequence, position_ids=None):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.device)
        fn = self._feature_function()
        if position_ids is None:
            return fn(sequence)
        return fn(sequence, position_ids=position_ids)

    def _supports_position_ids(self):
        try:
            return (
                "position_ids" in inspect.signature(self._feature_function()).parameters
            )
        except (TypeError, ValueError):
            return False

    def _get_categories(self, item_id: int):
        # Category metadata is explicitly unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Measure embedding alignment without fabricating undefined values
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            self.embedding_sample_count = 0
            self.embedding_excluded_count = 0
            if n == 0 or sample_size < 2:
                return None
            rng = random.Random(seed)
            indices = rng.sample(range(n), min(sample_size, n))
            idx = torch.tensor(indices, dtype=torch.long, device=emb.device)
            x = emb.index_select(0, idx).float()
            valid = torch.isfinite(x).all(dim=-1) & (x.norm(dim=-1) > 0)
            self.embedding_excluded_count = int((~valid).sum().item())
            x = F.normalize(x[valid], p=2, dim=-1)
            k = x.size(0)
            if k < 2:
                return None
            self.embedding_sample_count = k
            # Equivalent to the off-diagonal Gram mean, without a k-by-k matrix.
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1)
            return float(((mean_cos + 1.0) * 0.5).item())

    ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Restore model mode for standalone training-pair probes
        with self._evaluation():
            return self._training_pair_margins(
                dataloader, num_batches, eps, topk_categories
            )

    def _training_pair_margins(self, dataloader, num_batches, eps, topk_categories):
        self.feature_samples = []
        feature_count = 0
        measured_batches = 0
        excluded_positions = 0
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and reject malformed diagnostic batches
        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnosis loader must yield (None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            shapes = [tuple(np.shape(value)) for value in (seq, pos, neg)]
            if len(shapes[0]) != 2 or any(shape != shapes[0] for shape in shapes):
                raise ValueError(
                    f"Diagnosis tensors must have matching B x L shapes; got {shapes}"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Score actual blended features and count valid training pairs
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self._features(seq_t)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]
            eligible = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            finite = torch.isfinite(margin) & torch.isfinite(all_logits).all(dim=-1)
            valid = eligible & finite
            excluded_positions += int((eligible & ~finite).sum().item())
            measured_batches += 1
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())
            if feature_count < 2048:
                feature_valid = (seq_t != 0) & torch.isfinite(log_feats).all(dim=-1)
                features = log_feats[feature_valid][: 2048 - feature_count]
                if features.numel():
                    self.feature_samples.append(features.detach().float().cpu())
                    feature_count += features.size(0)

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Keep unavailable category breakdowns empty
            low_total += int(low_mask.sum())
            wrong_total += int(wrong_mask.sum())
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Mark absent training-pair measurements as undefined
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

        ### >>> Self_EvolveRec-BLOCK-START: Attach training-pair coverage and missingness
        margin_metrics["count"] = sum(values.numel() for values in margins_all)
        margin_metrics["batches"] = measured_batches
        margin_metrics["nonfinite_positions"] = excluded_positions
        ### <<< Self_EvolveRec-BLOCK-END

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

    ### >>> Self_EvolveRec-BLOCK-START: Paired training-prefix perturbations with explicit positional controls
    def _gate_measurement(self):
        values = []
        for name, module in self.module.named_modules():
            gate = getattr(module, "gate_logit", None)
            alpha = getattr(module, "alpha", None)
            if isinstance(gate, torch.Tensor) and gate.numel() == 1:
                value = float(torch.sigmoid(gate.detach()).item())
                values.append({"module": name, "transform": "sigmoid", "value": value})
            elif isinstance(alpha, torch.Tensor) and alpha.numel() == 1:
                value = float(torch.tanh(alpha.detach()).item())
                values.append({"module": name, "transform": "tanh", "value": value})
        return values

    def _feature_geometry(self):
        if not self.feature_samples:
            return {"effective_rank": None, "mean_norm": None, "count": 0}
        x = torch.cat(self.feature_samples).double()
        count = x.size(0)
        result = {
            "effective_rank": None,
            "mean_norm": float(x.norm(dim=-1).mean().item()),
            "count": count,
        }
        if count < 2:
            return result
        singular = torch.linalg.svdvals(x - x.mean(dim=0, keepdim=True))
        tolerance = torch.finfo(x.dtype).eps * max(x.shape) * singular.max()
        singular = singular[singular > tolerance]
        if singular.numel():
            p = singular / singular.sum()
            result["effective_rank"] = float(torch.exp(-(p * p.log()).sum()).item())
        return result

    def _prefix_scores(self, events, candidates, maxlen, position_values=None):
        seq = torch.zeros((1, maxlen), dtype=torch.long, device=self.device)
        seq[0, -len(events) :] = torch.tensor(events, device=self.device)
        position_ids = None
        if position_values is not None:
            position_ids = torch.arange(maxlen, device=self.device).unsqueeze(0)
            position_ids[0, -len(events) :] = torch.tensor(
                position_values, device=self.device
            )
        features = self._features(seq, position_ids)
        valid_indices = seq[0].ne(0).nonzero(as_tuple=True)[0]
        if not valid_indices.numel():
            raise ValueError("A diagnostic prefix must contain a surviving interaction")
        final = features[:, valid_indices[-1], :]
        items = torch.tensor([candidates], dtype=torch.long, device=self.device)
        scores = self.model.predict_score(final, items)
        if scores.numel() != len(candidates):
            raise ValueError(
                "predict_score must return exactly one score per diagnostic candidate; "
                f"got shape {tuple(scores.shape)} for {len(candidates)} candidates"
            )
        return scores.reshape(-1).detach().float().cpu().numpy()

    @staticmethod
    def _outcomes(scores, original_top):
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        top = set(order[: min(10, len(order))].tolist())
        return np.array(
            [
                float(scores[0]),
                float(scores[0] - scores[1:].mean()),
                float(rank),
                float(rank <= 10),
                len(top & original_top) / len(original_top),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _paired_summary(rows):
        names = (
            "target_score",
            "target_minus_mean_negative",
            "target_rank",
            "diagnostic_hr10",
            "top10_overlap",
        )
        if not rows:
            return {
                "count": 0,
                "original_mean": None,
                "perturbed_mean": None,
                "mean_change": None,
                "change_ci95": None,
                "strata": {},
            }
        original = np.stack([row[0] for row in rows])
        perturbed = np.stack([row[1] for row in rows])
        change = perturbed - original
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(rows), size=(2000, len(rows)))
        boot = change[indices].mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975], axis=0)
        strata = {}
        for label_index, key in ((2, "history_length"), (3, "target_popularity")):
            strata[key] = {}
            for label in sorted({row[label_index] for row in rows}):
                selected = [
                    i for i, row in enumerate(rows) if row[label_index] == label
                ]
                strata[key][label] = {
                    "count": len(selected),
                    "mean_change": dict(
                        zip(names, change[selected].mean(axis=0).tolist())
                    ),
                }
        return {
            "count": len(rows),
            "original_mean": dict(zip(names, original.mean(axis=0).tolist())),
            "perturbed_mean": dict(zip(names, perturbed.mean(axis=0).tolist())),
            "mean_change": dict(zip(names, change.mean(axis=0).tolist())),
            "change_ci95": {
                name: interval[:, i].tolist() for i, name in enumerate(names)
            },
            "strata": strata,
        }

    def analyze_history_perturbations(self, max_users=20):
        with self._evaluation():
            return self._history_perturbations(max_users)

    def _history_perturbations(self, max_users):
        if not isinstance(max_users, int) or max_users < 0:
            raise ValueError("max_users must be a nonnegative integer")
        histories = self.user_train.get("History")
        if not isinstance(histories, dict):
            raise ValueError(
                "Diagnostic view.user_train must contain a History mapping"
            )
        itemnum = self.model.item_emb.num_embeddings - 1
        encoder = getattr(self.module, "encoder", self.module)
        maxlen = int(getattr(encoder, "maxlen", 0))
        positional_hook = self._supports_position_ids()
        names = [
            "delete_recent_two",
            "delete_older_two",
            "delete_random_two",
            "delete_recent_band_two",
            "delete_older_band_two",
            "replace_recent_content",
            "replace_older_content",
            "padding_recent_positions",
            "delete_final_interaction",
            "delete_recent_preserved_positions",
            "delete_older_preserved_positions",
            "shift_positions_minus_one",
        ]
        records = {name: [] for name in names}
        report = {
            "scope": str(getattr(self.model, "dataset_name", "current_view")),
            "requested_users": max_users,
            "eligible_users": 0,
            "selected_users": 0,
            "measured_users": 0,
            "invalid_histories": 0,
            "candidate_shortage_users": 0,
            "zero_negative_users": 0,
            "nonfinite_comparisons": 0,
            "position_id_hook_available": positional_hook,
            "unsupported_variants": [] if positional_hook else names[-3:],
            "candidate_counts": [],
            "replacement_shortage_users": 0,
            "position_shift_ineligible_users": 0,
            "measurement_count_definition": (
                "measured_users counts finite original prefixes; each comparison "
                "has its own count of finite paired measurements"
            ),
        }
        if maxlen < 10:
            report["unavailable_reason"] = "Encoder maxlen unavailable or below ten"
            report["comparisons"] = {name: self._paired_summary([]) for name in names}
            return report
        eligible = []
        popularity = Counter()
        for user, raw_history in histories.items():
            try:
                history = [int(item) for item in raw_history]
            except (TypeError, ValueError):
                report["invalid_histories"] += 1
                continue
            if any(item < 1 or item > itemnum for item in history):
                report["invalid_histories"] += 1
                continue
            popularity.update(history)
            if len(history) >= 11:
                eligible.append((str(user), history))
        eligible.sort(key=lambda pair: pair[0])
        report["eligible_users"] = len(eligible)
        threshold = float(np.median(list(popularity.values()))) if popularity else None
        report["popularity_threshold"] = threshold
        selection_rng = random.Random(42)
        selected = selection_rng.sample(eligible, min(max_users, len(eligible)))
        report["selected_users"] = len(selected)
        for user, history in selected:
            # Stable per-user streams prevent one user's missing comparisons
            # from changing later users' diagnostic candidates and controls.
            rng = random.Random(f"diagnosis:42:{user}")
            target = history[-1]
            prefix = history[:-1][-maxlen:]
            n = len(prefix)
            excluded = set(history)
            pool = [item for item in range(1, itemnum + 1) if item not in excluded]
            negatives = rng.sample(pool, min(99, len(pool)))
            report["candidate_counts"].append(1 + len(negatives))
            report["candidate_shortage_users"] += int(len(negatives) < 99)
            if not negatives:
                report["zero_negative_users"] += 1
                continue
            candidates = [target] + negatives
            base_scores = self._prefix_scores(prefix, candidates, maxlen)
            if not np.isfinite(base_scores).all():
                report["nonfinite_comparisons"] += 1
                continue
            original_top = set(np.argsort(-base_scores, kind="stable")[:10].tolist())
            original = self._outcomes(base_scores, original_top)
            recent = [n - 3, n - 2]
            older = [n - 5, n - 4]
            deletions = {
                "delete_recent_two": recent,
                "delete_older_two": older,
                "delete_random_two": rng.sample(range(n - 1), 2),
                "delete_recent_band_two": rng.sample(range(n - 5, n - 1), 2),
                # This four-event band contains the older primary window.
                "delete_older_band_two": rng.sample(range(n - 7, n - 3), 2),
                "delete_final_interaction": [n - 1],
            }
            variants = {}
            for name, removed in deletions.items():
                survivors = [i for i in range(n) if i not in removed]
                variants[name] = ([prefix[i] for i in survivors], None)
            # Do not inject a ranked negative candidate into the visible history:
            # that would confound content sensitivity with candidate exposure.
            candidate_set = set(candidates)
            replacement_pool = [item for item in pool if item not in candidate_set]
            replacements = (
                rng.sample(replacement_pool, 2) if len(replacement_pool) >= 2 else None
            )
            report["replacement_shortage_users"] += int(replacements is None)
            for label, removed in (("recent", recent), ("older", older)):
                if replacements is not None:
                    replaced = list(prefix)
                    for i, replacement in zip(removed, replacements):
                        replaced[i] = replacement
                    variants[f"replace_{label}_content"] = (replaced, None)
                if positional_hook:
                    survivors = [i for i in range(n) if i not in removed]
                    variants[f"delete_{label}_preserved_positions"] = (
                        [prefix[i] for i in survivors],
                        [maxlen - n + i for i in survivors],
                    )
            padded = list(prefix)
            for i in recent:
                padded[i] = 0
            variants["padding_recent_positions"] = (padded, None)
            if positional_hook:
                if n < maxlen:
                    variants["shift_positions_minus_one"] = (
                        prefix,
                        list(range(maxlen - n - 1, maxlen - 1)),
                    )
                else:
                    report["position_shift_ineligible_users"] += 1
            length_label = "at_most_20" if len(history) - 1 <= 20 else "over_20"
            pop_label = (
                "at_most_median" if popularity[target] <= threshold else "over_median"
            )
            for name, (events, positions) in variants.items():
                scores = self._prefix_scores(events, candidates, maxlen, positions)
                if not np.isfinite(scores).all():
                    report["nonfinite_comparisons"] += 1
                    continue
                outcomes = self._outcomes(scores, original_top)
                records[name].append((original, outcomes, length_label, pop_label))
            report["measured_users"] += 1
        report["comparisons"] = {
            name: self._paired_summary(rows) for name, rows in records.items()
        }
        return report

    def run_full_diagnosis(self, train_loader):
        with self._evaluation():
            alignment = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            margins, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                train_loader, num_batches=10, eps=0.5, topk_categories=20
            )
            geometry = self._feature_geometry()
            perturbations = self.analyze_history_perturbations(max_users=20)
            gates = self._gate_measurement()
        limitations = [
            "Training-pair and training-prefix diagnostics are not held-out validation metrics.",
            "Perturbation users are separate from the native five-users-per-category panel.",
            "This invocation covers only the supplied view; it does not imply four-category coverage.",
            "Targets are the final public training events; the model may have trained on them.",
            "Targets, candidates, and truncated visible prefixes are frozen within each comparison; no backfilling occurs.",
            "All primary two-event deletions retain the final interaction. Final-interaction deletion is separate.",
            "Compact deletion uses ordinary model positions; preserved-position comparisons require an explicit encoder hook.",
            "Padding replacement changes the padding pattern and recent-event membership; its effect on attention depends on the encoder's actual masking implementation.",
            "Replacement items exclude both the public history and diagnostic candidates; the same two replacements are used for recent and older windows. Comparisons are unavailable when fewer than two replacements exist.",
            "Four-event random-control bands contain their corresponding primary windows but overlap each other; they are broad recency controls, not exact-distance matches.",
            "Diagnostic RNG and module training modes are restored. Loader iterator cursors and worker-local RNG states cannot be restored; use a separate diagnostic loader when resuming training.",
            "This diagnostic patch does not implement the proposed readout, encoder position-ID hook, checkpoint architecture, predictor reconstruction, or host selection policy. Those require separate training-code changes.",
            "Position-hook support does not prove that an implementation honors supplied position IDs.",
            "Perturbations may be out of distribution and establish neither generalization nor causal preferences.",
            "Bootstrap intervals resample diagnostic users and condition on this model, sample, and candidates.",
            "No reviews, ratings, item categories, prices, stores, or simulator traits are measured.",
            "Unavailable comparisons return null measurements with count zero; they were not executed.",
        ]
        return {
            "metrics": {
                "embedding_collapse_score": alignment,
                "embedding_sample_count": self.embedding_sample_count,
                "embedding_excluded_count": self.embedding_excluded_count,
                "pos_neg_margin_mean": margins["mean"],
                "pos_neg_margin_p10": margins["p10"],
                "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margins["small_margin_rate"],
                "training_pair_count": margins["count"],
                "training_batch_count": margins["batches"],
                "training_pair_nonfinite_count": margins["nonfinite_positions"],
                "pos_neg_margin_category_breakdown": breakdown,
                "sequence_feature_geometry": geometry,
                "learned_scalar_gates": gates,
                "history_perturbations": perturbations,
                "limitations": limitations,
            },
            "metric_definitions": {
                "embedding_collapse_score": "Legacy key: (mean off-diagonal item-embedding cosine + 1)/2, excluding padding. Measures directional alignment only; it does not by itself establish collapse or recommendation diversity.",
                "embedding_sample_count": "Number of finite, nonzero sampled embeddings used in alignment; zero when fewer than two can be measured.",
                "embedding_excluded_count": "Sampled embeddings excluded for nonfinite coordinates or zero norm.",
                "pos_neg_margin_mean": "Mean positive-minus-negative logit on finite, nonpadding training pairs using the actual trained feature and scoring path.",
                "pos_neg_margin_p10": "Nearest-rank tenth percentile of finite training-pair margins.",
                "pos_neg_neg_beats_pos_rate": "Fraction of measured training pairs with margin below zero.",
                "pos_neg_small_margin_rate": "Fraction of measured training pairs with margin below 0.5.",
                "training_pair_count": "Number of finite training pairs used in margin summaries; zero means summaries are undefined.",
                "training_batch_count": "Number of structurally valid loader batches scored, up to ten.",
                "training_pair_nonfinite_count": "Eligible nonpadding training pairs excluded because logits or margins were nonfinite.",
                "pos_neg_margin_category_breakdown": "Position counts at eps=0.5; category lists remain empty because categories are unavailable.",
                "sequence_feature_geometry": "First up to 2048 finite valid blended training states: sample count, mean L2 norm, and centered effective rank exp(-sum(p log p)), with p proportional to numerically nonzero singular values. Rank is null for insufficient or constant samples. This position sample is not a user-level uncertainty estimate.",
                "learned_scalar_gates": "Measured scalar sigmoid(gate_logit) or tanh(alpha), with module names and the actual transform. An empty list means no supported scalar gate was observed; conditional gates are not reduced to a fictitious scalar.",
                "history_perturbations": (
                    "Local seed-42 sample of up to twenty users from the current view with at least ten prefix events and a public training next-item target. "
                    "Candidates contain that target and up to 99 unique negatives excluding the entire public training history. Shortages and coverage are reported. "
                    "For a prefix of n events, recent deletion removes indices n-3,n-2 and immediately older deletion removes n-5,n-4. "
                    "Random controls remove two events globally, within indices n-5..n-2, or within n-7..n-4. Each four-event band contains its corresponding primary deletion window; the bands overlap. "
                    "Content replacement uses two distinct unseen items excluded from all diagnostic candidates, at fixed positions and length. Identical replacements are used for recent and older windows; shortages are reported and those comparisons are skipped. "
                    "Preserved-position deletion assigns survivors their original absolute positions. Position shifts subtract one with content unchanged only when indices remain valid. "
                    "Readout always uses the largest surviving valid sequence index, including the separate final-event deletion. "
                    "Outcomes are observed-target score, target-minus-mean-negative score, one-based target rank, diagnostic HR@10, and top-ten candidate overlap with the original model. "
                    "Stable candidate ordering resolves ties with the target first. Changes equal perturbed minus original; higher rank changes mean worse ranks. "
                    "Each comparison reports paired means, measurement count, and percentile intervals from 2000 seed-42 user bootstrap replicates. "
                    "Strata use untruncated prefix length at most twenty versus longer, and target training frequency at most versus above the median frequency among observed training items. "
                    "These candidates and HR values never replace the pinned host evaluator."
                ),
                "limitations": "Explicit missingness, scope, positional controls, and limits on interpretation of fitted-model diagnostics.",
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
