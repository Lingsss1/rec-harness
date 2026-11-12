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
    ### >>> Self_EvolveRec-BLOCK-START: Evidence boundaries for BERT4Rec diagnosis
    - Missing metrics and unavailable checks are unknown, never evidence of health.
    - Structural failures must be distinguished from training-exposed recovery and sensitivity.
    - Training-history targets were exposed during training. Their recovery does not measure held-out generalization or prove memorization alone.
    - History interventions change content, positions or attention normalization. Do not infer causal preferences from them.
    - Auxiliary candidate draws do not replace or modify the host validation protocol.
    - Inspect sample counts, repeat/explore strata, candidate counts, bootstrap intervals and unavailable reasons before drawing conclusions.
    - No review text, ratings, item categories, prices, stores or simulator traits are available. Do not infer them.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family-specific ModelView boundary
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.network = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Category metadata is unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Require at least two sampled embeddings for cosine estimation
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute identical pairwise cosine statistic without a quadratic Gram matrix
            # sum_{i != j} dot(x_i, x_j) equals
            # ||sum_i x_i||^2 - sum_i ||x_i||^2, including zero-vector cases.
            # Double accumulation reduces cancellation near zero mean cosine.
            accumulated = x.to(torch.float64)
            offdiag_sum = (
                accumulated.sum(dim=0).square().sum() - accumulated.square().sum()
            )
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
        ### >>> Self_EvolveRec-BLOCK-START: Mask-aligned training margin counts
        margins_all = []
        inspected_batches = 0
        invalid_targets = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic sampling and reject malformed batches
        from itertools import islice

        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError("Expected (None, sequence, positive, negative)")
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Validate family-specific diagnostic tensors and scores
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative must have matching B x L shapes"
                )
            if bool(((seq_t < 0) | (seq_t > self.network.mask_token)).any()):
                raise ValueError("Diagnostic input contains an invalid token ID")
            if bool(((pos_t < 0) | (pos_t > self.network.itemnum)).any()):
                raise ValueError("Diagnostic positive contains a noncatalog target")
            if bool(((neg_t < 0) | (neg_t > self.network.itemnum)).any()):
                raise ValueError("Diagnostic negative contains a noncatalog item")
            seq_np = seq_t.detach().cpu().numpy()
            log_feats = self.view.log2feats(seq_np)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.view.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError("ModelView.predict_score must return B x L x 2 scores")
            if not bool(torch.isfinite(all_logits).all()):
                raise ValueError("Nonfinite training-margin diagnostic scores")
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Score only reserved-mask reconstruction targets
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            selected = pos_t != 0
            valid = (
                selected
                & (seq_t == self.network.mask_token)
                & (pos_t >= 1)
                & (pos_t <= self.network.itemnum)
                & (neg_t >= 1)
                & (neg_t <= self.network.itemnum)
            )
            inspected_batches += 1
            invalid_targets += int((selected & ~valid).sum().item())
            if valid.any():
                ### <<< Self_EvolveRec-BLOCK-END
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Match breakdown eligibility to margin eligibility
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
                ### >>> Self_EvolveRec-BLOCK-START: Report empty margin observations explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Record training exposure and structural exclusions
        margin_metrics["count"] = sum(int(m.numel()) for m in margins_all)
        margin_metrics["reason"] = (
            None if margins_all else "No valid reserved-mask training targets observed"
        )
        breakdown = {
            "exposure": "training-exposed masked-target margins",
            "inspected_batches": inspected_batches,
            "invalid_target_positions": invalid_targets,
            "category_metadata_available": False,
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

    ### >>> Self_EvolveRec-BLOCK-START: Observe inference tokens, gate sensitivity and vectorize user bootstrap
    def _pack_inference(self, history):
        length = int(self.network.maxlen)
        if length < 1:
            raise ValueError("BERT4Rec maxlen must be positive")
        retained = list(history)[-(length - 1) :] if length > 1 else []
        tokens = retained + [int(self.network.mask_token)]
        return [0] * (length - len(tokens)) + tokens

    def _integrity(self):
        model = self.network
        length, itemnum = int(model.maxlen), int(model.itemnum)
        result = {
            "sampler_provenance": None,
            "sampler_provenance_reason": (
                "The loader supplies corrupted inputs and labels, but no original "
                "event/position provenance. Same-position labels, corruption branches "
                "and loss weights cannot be independently certified."
            ),
            "fresh_process_predictor_parity": None,
            "fresh_process_predictor_parity_reason": (
                "No fresh-process predictor/checkpoint runner is supplied through ModelView. "
                "Live inference checks below do not establish checkpoint, mapping or writer parity."
            ),
            "attention_deletion": None,
            "attention_deletion_reason": (
                "The supplied encode_sequence interface has no diagnostic key/value mask hook."
            ),
        }
        if itemnum < 1 or length < 1:
            result["failure"] = "Invalid item vocabulary or sequence length"
            return result
        other = min(2, itemnum)
        histories = [[], [1], [1] * length, [1, other, 1]]
        raw = torch.zeros((4, length), dtype=torch.long, device=self.device)
        for row, history in enumerate(histories):
            retained = history[-length:]
            if retained:
                raw[row, -len(retained) :] = torch.tensor(retained, device=self.device)
        expected = torch.tensor(
            [self._pack_inference(h) for h in histories],
            dtype=torch.long,
            device=self.device,
        )
        shifted = torch.zeros_like(raw)
        shifted[:, :-1] = raw[:, 1:]
        shifted[:, -1] = model.mask_token
        candidates = torch.arange(1, min(itemnum, 32) + 1, device=self.device)
        # Match the live single-user batch shape to avoid attributing differences
        # between batched kernels to preprocessing or checkpoint inconsistency.
        reference = torch.stack(
            [
                model.encode_sequence(expected[i : i + 1])[0, -1]
                @ model.item_emb(candidates).T
                for i in range(4)
            ]
        )
        observed_inputs = []

        def capture_sequence(module, args):
            tokens = args[0]
            if tokens.ndim == 2 and tuple(tokens.shape) == (1, length):
                observed_inputs.append(tokens.detach().clone())

        handle = model.item_emb.register_forward_pre_hook(capture_sequence)
        try:
            live = torch.stack(
                [model.score_candidates(raw[i : i + 1], candidates) for i in range(4)]
            )
        finally:
            handle.remove()
        ids_match = len(observed_inputs) == 4 and all(
            torch.equal(observed_inputs[i], expected[i : i + 1]) for i in range(4)
        )
        finite = bool(torch.isfinite(reference).all() & torch.isfinite(live).all())
        result.update(
            {
                "reserved_mask_id_valid": int(model.mask_token) == itemnum + 1,
                "fixture_count": 4,
                "fixture_names": ["empty", "short", "full_length", "repeated_id"],
                "inference_ids_match": bool(ids_match),
                "constructed_shift_ids_match": bool(torch.equal(expected, shifted)),
                "observed_sequence_calls": len(observed_inputs),
                "readout_absolute_index": length - 1,
                "finite_scores": finite,
                "live_inference_allclose": finite
                and bool(torch.allclose(reference, live, rtol=1.3e-6, atol=1e-5)),
                "live_inference_max_abs_error": (
                    float((reference - live).abs().max()) if finite else None
                ),
                "rtol": 1.3e-6,
                "atol": 1e-5,
            }
        )
        # A synthetic provenance control rejects shifted labels by position,
        # even when repeated IDs make their item identities indistinguishable.
        selected = np.array([False, True, False])
        source_positions = np.arange(3)
        correct = source_positions.copy()
        shifted_labels = np.array([1, 2, 2])
        result["shifted_provenance_control_rejected"] = bool(
            np.array_equal(correct[selected], source_positions[selected])
            and not np.array_equal(shifted_labels[selected], source_positions[selected])
        )
        result["shifted_provenance_control_scope"] = (
            "Synthetic validator only; not sampler certification"
        )
        if length < 3:
            result["padding_and_bidirectionality_reason"] = (
                "Fixtures require maxlen >= 3"
            )
            return result
        fixture = torch.tensor(
            [self._pack_inference([1])], dtype=torch.long, device=self.device
        )
        original = model.encode_sequence(fixture)
        saved_padding = model.item_emb.weight[0].detach().clone()
        try:
            model.item_emb.weight[0].copy_(
                saved_padding
                + torch.arange(
                    model.hidden_units,
                    device=self.device,
                    dtype=saved_padding.dtype,
                )
                + 1
            )
            perturbed = model.encode_sequence(fixture)
        finally:
            model.item_emb.weight[0].copy_(saved_padding)
        valid = fixture != 0
        result["padding_embedding_invariant"] = bool(
            torch.allclose(original[valid], perturbed[valid], rtol=1.3e-6, atol=1e-5)
        )
        result["padding_max_abs_error"] = float(
            (original[valid] - perturbed[valid]).abs().max()
        )
        # Perturb a visible token right of an interior query. A causal encoder
        # cannot pass this influence check; degeneracy is reported as inconclusive.
        changed = fixture.clone()
        changed[:, -1] = 1
        changed_features = model.encode_sequence(changed)
        right_delta = float((original[:, -2] - changed_features[:, -2]).abs().max())
        result["right_context_max_abs_influence"] = right_delta
        result["right_context_influence_detected"] = right_delta > 1e-5
        result["right_context_scope"] = (
            "Observed right-token influence; zero influence is inconclusive or a structural failure"
        )
        omitted = fixture.clone()
        omitted[:, -1] = 0
        omitted_scores = (
            model.encode_sequence(omitted)[:, -1] @ model.item_emb(candidates).T
        )
        wrong_index = int((fixture[0] != 0).sum()) - 1
        wrong_scores = original[:, wrong_index] @ model.item_emb(candidates).T
        result["omitted_mask_control_rejected"] = not bool(
            torch.allclose(
                omitted_scores,
                original[:, -1] @ model.item_emb(candidates).T,
                rtol=1.3e-6,
                atol=1e-5,
            )
        )
        result["count_minus_one_control_rejected"] = (
            not bool(
                torch.allclose(
                    wrong_scores,
                    original[:, -1] @ model.item_emb(candidates).T,
                    rtol=1.3e-6,
                    atol=1e-5,
                )
            )
            if wrong_index != length - 1
            else None
        )
        return result

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {
                "mean": None,
                "count": 0,
                "ci95": None,
                "reason": "No eligible user observations",
            }
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite observations in diagnostic bootstrap")
        rng = np.random.default_rng(42)
        indices = rng.integers(0, values.size, size=(1000, values.size))
        means = values[indices].mean(axis=1)
        return {
            "mean": float(values.mean()),
            "count": int(values.size),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "reason": None,
        }

    def _history_sensitivity(self):
        histories = self.user_train.get("History", self.user_train)
        itemnum = int(self.network.itemnum)
        length = int(self.network.maxlen)
        eligible = []
        popularity = Counter()
        for user, events in histories.items():
            events = list(events)
            if any(int(x) < 1 or int(x) > itemnum for x in events):
                raise ValueError("Training history contains a noncatalog item ID")
            events = [int(x) for x in events]
            popularity.update(events)
            # One last training event per user; all later events are absent.
            prefix = events[:-1][-(length - 1) :] if length > 1 else []
            if len(prefix) >= 2:
                eligible.append((user, events, prefix, events[-1]))
        eligible.sort(key=lambda row: str(row[0]))
        if len(eligible) > 200:
            eligible = random.Random(42).sample(eligible, 200)
        length_cuts = (
            np.quantile([len(row[2]) for row in eligible], [0.25, 0.5, 0.75]).tolist()
            if eligible
            else []
        )
        pop_cuts = (
            np.quantile(
                [popularity[i] for i in range(1, itemnum + 1)], [0.25, 0.5, 0.75]
            ).tolist()
            if itemnum
            else []
        )
        rows = {seed: [] for seed in (42, 43, 44)}
        names = [
            "original",
            "compact_oldest",
            "compact_newest",
            "mask_oldest",
            "mask_newest",
        ]
        for user, events, prefix, target in eligible:
            k = min(3, len(prefix) // 2)
            base = self._pack_inference(prefix)
            old_mask, new_mask = base.copy(), base.copy()
            first = length - 1 - len(prefix)
            old_mask[first : first + k] = [self.network.mask_token] * k
            new_mask[length - 1 - k : length - 1] = [self.network.mask_token] * k
            inputs = [
                base,
                self._pack_inference(prefix[k:]),
                self._pack_inference(prefix[:-k]),
                old_mask,
                new_mask,
            ]
            seq = torch.tensor(inputs, dtype=torch.long, device=self.device)
            features = self.network.encode_sequence(seq)[:, -1]
            excluded = set(events) | {0, target, int(self.network.mask_token)}
            available_count = itemnum - sum(1 <= x <= itemnum for x in excluded)
            for seed in (42, 43, 44):
                # Stable user-specific RNG; candidate sets are frozen for all interventions.
                rng = random.Random(f"{seed}:{user}")
                count = min(99, available_count)
                if available_count < 198:
                    pool = [x for x in range(1, itemnum + 1) if x not in excluded]
                    negatives = rng.sample(pool, count)
                else:
                    negatives, used = [], set()
                    while len(negatives) < count:
                        candidate = rng.randint(1, itemnum)
                        if candidate not in excluded and candidate not in used:
                            used.add(candidate)
                            negatives.append(candidate)
                candidates = torch.tensor([target] + negatives, device=self.device)
                scores = (
                    (features @ self.network.item_emb(candidates).T)
                    .float()
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite diagnostic candidate scores")
                measured, tops = {}, {}
                for index, name in enumerate(names):
                    order = np.argsort(-scores[index], kind="stable")
                    rank = int(np.flatnonzero(order == 0)[0]) + 1
                    tops[name] = set(order[:10].tolist())
                    measured[name] = {
                        "target_score": float(scores[index, 0]),
                        "rank": rank,
                        "hr10": float(rank <= 10),
                        "margin": (
                            float(scores[index, 0] - scores[index, 1:].max())
                            if count
                            else None
                        ),
                    }
                for name in names[1:]:
                    measured[name]["top10_overlap"] = len(
                        tops[name] & tops["original"]
                    ) / min(len(tops[name]), len(tops["original"]))
                    for metric in ("target_score", "rank", "hr10", "margin"):
                        value = measured[name][metric]
                        measured[name]["delta_" + metric] = (
                            value - measured["original"][metric]
                            if value is not None
                            else None
                        )
                for mode in ("compact", "mask"):
                    measured[mode + "_recent_minus_older"] = {
                        metric: (
                            (
                                measured[mode + "_newest"][metric]
                                - measured[mode + "_oldest"][metric]
                            )
                            if measured[mode + "_newest"][metric] is not None
                            else None
                        )
                        for metric in (
                            "target_score",
                            "rank",
                            "hr10",
                            "margin",
                            "top10_overlap",
                        )
                    }
                rows[seed].append(
                    {
                        "user": str(user),
                        "repeat": target in prefix,
                        "length_quartile": int(
                            np.searchsorted(length_cuts, len(prefix), side="right")
                        ),
                        "popularity_quartile": int(
                            np.searchsorted(pop_cuts, popularity[target], side="right")
                        ),
                        "candidate_count": count + 1,
                        "measurements": measured,
                    }
                )

        # Keep metric keys and explicit null summaries in empty strata.
        metric_keys = {
            name + "." + metric
            for records in rows.values()
            for record in records
            for name, measurements in record["measurements"].items()
            for metric in measurements
        }
        if not metric_keys:
            for name in names:
                metrics = ["target_score", "rank", "hr10", "margin"]
                if name != "original":
                    metrics += [
                        "top10_overlap",
                        "delta_target_score",
                        "delta_rank",
                        "delta_hr10",
                        "delta_margin",
                    ]
                metric_keys.update(name + "." + metric for metric in metrics)
            for mode in ("compact", "mask"):
                metric_keys.update(
                    mode + "_recent_minus_older." + metric
                    for metric in (
                        "target_score",
                        "rank",
                        "hr10",
                        "margin",
                        "top10_overlap",
                    )
                )

        def aggregate(records):
            output = {key: [] for key in sorted(metric_keys)}
            for record in records:
                for name, measurements in record["measurements"].items():
                    for metric, value in measurements.items():
                        if value is not None:
                            output[name + "." + metric].append(value)
            return {key: self._summary(values) for key, values in output.items()}

        draws = {}
        for seed, records in rows.items():
            draws[str(seed)] = {
                "users": len(records),
                "candidate_counts": dict(
                    Counter(r["candidate_count"] for r in records)
                ),
                "summary": aggregate(records),
                "repeat_explore": {
                    label: {
                        "users": sum(r["repeat"] == repeat for r in records),
                        "summary": aggregate(
                            [r for r in records if r["repeat"] == repeat]
                        ),
                    }
                    for label, repeat in (("repeat", True), ("explore", False))
                },
                "strata": {
                    field: {
                        str(q): {
                            "users": sum(r[field] == q for r in records),
                            "summary": aggregate([r for r in records if r[field] == q]),
                        }
                        for q in range(4)
                    }
                    for field in ("length_quartile", "popularity_quartile")
                },
            }
        return {
            "exposure": "training-exposed recovery; not held-out generalization",
            "category": getattr(self.view, "dataset_name", None),
            "eligible_selected_users": len(eligible),
            "reason": (
                None
                if eligible
                else "No training prefix with at least two retained items"
            ),
            "selection_seed": 42,
            "bootstrap_seed": 42,
            "bootstrap_resamples": 1000,
            "length_quartile_cutpoints": length_cuts,
            "item_training_popularity_quartile_cutpoints": pop_cuts,
            "candidate_policy": (
                "Positive exactly once; up to 99 unique catalog negatives excluding all "
                "permitted training-history items, positive, padding and mask. Frozen across interventions."
            ),
            "draws": draws,
            "category_metadata_breakdowns": {},
            "equal_category_macro": None,
            "equal_category_macro_reason": "A single ModelView cannot certify eligibility of all four categories",
        }

    def run_full_diagnosis(self, train_loader):
        modes = [(module, module.training) for module in self.network.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.network.eval()
            with torch.inference_mode():
                integrity = self._integrity()
                collapse = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048, seed=42
                )
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                required_checks = (
                    "reserved_mask_id_valid",
                    "inference_ids_match",
                    "finite_scores",
                    "live_inference_allclose",
                    "padding_embedding_invariant",
                    "omitted_mask_control_rejected",
                    "count_minus_one_control_rejected",
                )
                failed_checks = [
                    key for key in required_checks if integrity.get(key) is False
                ]
                unavailable_checks = [
                    key for key in required_checks if integrity.get(key) is None
                ]
                if integrity.get("failure"):
                    failed_checks.append("failure")
                # Invalid supervision is a structural failure, even when other
                # positions provide enough observations for finite margins.
                integrity["masked_target_alignment_valid"] = (
                    breakdown["invalid_target_positions"] == 0
                    if breakdown["inspected_batches"] > 0
                    else None
                )
                if integrity["masked_target_alignment_valid"] is False:
                    failed_checks.append("masked_target_alignment_valid")
                elif integrity["masked_target_alignment_valid"] is None:
                    unavailable_checks.append("masked_target_alignment_valid")
                integrity["failed_required_checks"] = failed_checks
                integrity["unavailable_required_checks"] = unavailable_checks
                integrity["structural_status"] = (
                    "FAILED"
                    if failed_checks
                    else (
                        "INCOMPLETE" if unavailable_checks else "OBSERVED_CHECKS_PASSED"
                    )
                )
                integrity["certification_scope"] = (
                    "Observed checks only; sampler provenance, fresh-process predictor "
                    "parity and effective attention-mask inspection remain uncertified."
                )
                sensitivity = (
                    {
                        "exposure": "training-exposed recovery; not held-out generalization",
                        "result": None,
                        "reason": "Skipped because required structural checks failed",
                        "failed_checks": failed_checks,
                        "category_metadata_breakdowns": {},
                    }
                    if failed_checks
                    else self._history_sensitivity()
                )
                metrics = {
                    "bert4rec_integrity": integrity,
                    "embedding_collapse_score": collapse,
                    "embedding_collapse_sample_count": min(
                        2048, int(self.network.itemnum)
                    ),
                    "embedding_collapse_reason": (
                        None
                        if collapse is not None
                        else "Fewer than two item embeddings"
                    ),
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "pos_neg_margin_sample_count": margins["count"],
                    "pos_neg_margin_reason": margins["reason"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "training_history_sensitivity": sensitivity,
                }
                definitions = {
                    "bert4rec_integrity": (
                        "Independent appended-mask fixtures compare exact IDs and live candidate scores "
                        "at rtol=1.3e-6, atol=1e-5. Includes padding perturbation and negative controls. "
                        "False checks require investigation; unavailable checks are not passes. "
                        "Synthetic provenance controls do not certify the actual sampler. "
                        "masked_target_alignment_valid checks that observed supervised positions "
                        "contain reserved masks and valid catalog positive/negative IDs; an empty "
                        "loader makes this check unavailable. It does not certify event provenance."
                    ),
                    "embedding_collapse_score": (
                        "(Mean off-diagonal pairwise cosine + 1)/2 over up to 2048 seed-42 sampled "
                        "catalog embeddings, excluding padding and mask. Measures angular alignment, "
                        "not recommendation diversity or representation quality by itself."
                    ),
                    "embedding_collapse_sample_count": "Number of catalog embeddings sampled",
                    "embedding_collapse_reason": "Reason the collapse statistic is undefined, or null",
                    "pos_neg_margin_mean": "Training-exposed mean positive-minus-negative logit at valid masked targets",
                    "pos_neg_margin_p10": "Training-exposed nearest-rank tenth percentile of masked-target margins",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid masked targets with margin < 0",
                    "pos_neg_small_margin_rate": "Fraction of valid masked targets with margin < 0.5",
                    "pos_neg_margin_sample_count": "Number of valid supervised reserved-mask positions observed",
                    "pos_neg_margin_reason": "Reason no valid training margins were observed, or null",
                    "pos_neg_margin_category_breakdown": (
                        "Observed batch, invalid-target and low-margin counts. Category lists remain empty "
                        "because item category metadata is unavailable. The supplied sampler always masks selected targets."
                    ),
                    "training_history_sensitivity": (
                        "One last training target/user with strictly earlier prefix truncated to L-1; "
                        "earlier repeats are retained. Up to 200 users selected with seed 42. "
                        "Candidate draws 42/43/44 are reported separately. Rank is one-based with stable "
                        "candidate-order ties; HR10 is rank<=10; margin is target minus hardest negative. "
                        "Each delta is intervention minus original. Recent-minus-older is newest-block "
                        "intervention minus oldest-block intervention; k=min(3,floor(n/2)). "
                        "Overlap is intersection/minimum returned top-list length. Compact deletion shifts "
                        "survivor positions; mask replacement retains positions and introduces learned mask content. "
                        "Attention deletion is unavailable without a supported hook. Means have 1000-resample "
                        "seed-42 user-bootstrap percentile 95% intervals, conditional on the selected users and "
                        "candidate draw. These intervals exclude checkpoint-selection uncertainty. "
                        "Popularity quartiles use catalog training counts; length quartiles use selected retained "
                        "prefix lengths; tied cutpoints may produce empty strata. Results describe training-exposed "
                        "sensitivity, not causal preferences, memorization alone or held-out generalization. "
                        "Host validation is unchanged; no four-category macro is fabricated."
                    ),
                }
                return {"metrics": metrics, "metric_definitions": definitions}
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    ### <<< Self_EvolveRec-BLOCK-END
