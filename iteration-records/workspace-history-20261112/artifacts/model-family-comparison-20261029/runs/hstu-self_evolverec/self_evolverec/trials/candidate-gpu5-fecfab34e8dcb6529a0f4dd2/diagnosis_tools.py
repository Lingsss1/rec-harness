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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret actual-backbone key sensitivity with explicit unsupported factors
    - Unavailable/null measurements are missing evidence, never healthy zeros.
    - Training-prefix measurements are in-sample, including their reserved targets; do not call them validation scores.
    - Distinguish slot-preserving post-softmax value-access gates from separately labeled compacted deletions. Blocked keys still affect softmax denominators; these gates cannot isolate blocked embedding occurrences. Anchor removal changes the final observed item.
    - The supplied backbone uses softmax, additive position embeddings and no temporal input. Do not describe these probes as post-SiLU gates, position-bias removal or a completed factorial study.
    - Interpret geometry measurements only when their status is measured and correctness checks pass. Report nonfinite cases and inconclusive sensitivity controls.
    - Recent-minus-older HR below zero indicates greater recent-context dependence. Such dependence may be useful; it does not establish harmful overreliance or a benefit from retraining with regularization.
    - Intermediate queries can lose different counts of eligible keys. Do not claim distance-independent content effects or per-query cardinality matching.
    - Bootstrap intervals condition on fitted weights and fixed candidates. They exclude training-seed variation, candidate-sampling variation, and repeated-selection optimism.
    - Do not infer unavailable metadata, simulator traits, or a four-category macro result from a single-category report.
    - Raw cosine and representation rank alone do not establish HR@10 failure.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the host view and distinguish its trained module
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

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

    ### >>> Self_EvolveRec-BLOCK-START: Audit actual HSTU key access with fixed slots and explicit unsupported bias factors
    CONDITIONS = (
        "intact",
        "suffix",
        "remove_recent_context",
        "remove_old_context",
        "remove_anchor",
    )
    CATEGORIES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )

    @staticmethod
    def _timestamp_span(values):
        if not values or any(value is None for value in values):
            return {"value": None, "units": None, "reason": "missing_timestamps"}
        try:
            parsed = np.asarray([float(value) for value in values], dtype=np.float64)
            if not np.isfinite(parsed).all():
                raise ValueError("nonfinite timestamp")
            span = float(parsed.max()) - float(parsed.min())
            if not np.isfinite(span):
                raise ValueError("nonfinite timestamp span")
            return {
                "value": span,
                "units": "source_numeric_units",
                "reason": None,
            }
        except (ValueError, TypeError, OverflowError):
            try:
                from datetime import datetime

                parsed = [
                    datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                    for v in values
                ]
                span = (max(parsed) - min(parsed)).total_seconds()
                return {"value": float(span), "units": "seconds", "reason": None}
            except (ValueError, TypeError, OverflowError):
                return {
                    "value": None,
                    "units": None,
                    "reason": "unparseable_timestamps",
                }

    @classmethod
    def _paired_summary(cls, records):
        if not records:
            return {
                "eligible_users": 0,
                "conditions": {},
                "unavailable_reason": "no_eligible_users",
            }
        rng = np.random.default_rng(42)
        n = len(records)
        # Same resampled user indices for every condition and statistic.
        indices = rng.integers(0, n, size=(2000, n))
        summary = {}
        for name in cls.CONDITIONS:
            rows = [r["conditions"][name] for r in records]
            values = np.asarray(
                [
                    [
                        r["hit"],
                        r["rank"],
                        r["delta_hr"],
                        r["delta_rank"],
                        r["top10_overlap"],
                    ]
                    for r in rows
                ],
                dtype=np.float64,
            )
            boot = values[indices].mean(axis=1)
            means = values.mean(axis=0)
            intervals = np.percentile(boot, [2.5, 97.5], axis=0)
            names = ("hr10", "rank", "delta_hr", "delta_rank", "top10_overlap")
            summary[name] = {
                "eligible_users": n,
                "affected_users": sum(int(r["affected"]) for r in rows),
                "estimates": {
                    key: {
                        "mean": float(means[j]),
                        "ci95": [float(intervals[0, j]), float(intervals[1, j])],
                    }
                    for j, key in enumerate(names)
                },
            }
        return {"eligible_users": n, "conditions": summary, "unavailable_reason": None}

    @classmethod
    def aggregate_categories(cls, reports):
        """Explicit aggregation of four reports from this trial; never read other trials."""
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
        rng = np.random.default_rng(42)
        sampled = {
            c: rng.integers(0, len(rows), size=(2000, len(rows)))
            for c, rows in groups.items()
        }
        output = {}
        for condition in cls.CONDITIONS:
            means, bootstraps = [], []
            for category, rows in groups.items():
                values = np.asarray(
                    [
                        [
                            r["conditions"][condition][key]
                            for key in (
                                "hit",
                                "delta_hr",
                                "delta_rank",
                                "top10_overlap",
                            )
                        ]
                        for r in rows
                    ],
                    dtype=np.float64,
                )
                means.append(values.mean(axis=0))
                bootstraps.append(values[sampled[category]].mean(axis=1))
            mean = np.mean(means, axis=0)
            interval = np.percentile(np.mean(bootstraps, axis=0), [2.5, 97.5], axis=0)
            output[condition] = {
                key: {
                    "mean": float(mean[j]),
                    "ci95": [float(interval[0, j]), float(interval[1, j])],
                }
                for j, key in enumerate(
                    ("hr10", "delta_hr", "delta_rank", "top10_overlap")
                )
            }
        return {
            "value": output,
            "eligible_users": {c: len(v) for c, v in groups.items()},
            "scope": "in_sample_training_prefix",
            "reason": None,
        }

    @torch.no_grad()
    def _paired_recent_context(self):
        import os
        import hashlib

        histories = self.user_train.get("History", {})
        times = self.user_train.get("Time", {})
        category = getattr(self.model, "dataset_name", None)
        if category not in self.CATEGORIES:
            category = getattr(self.network, "dataset_name", None)
        if category not in self.CATEGORIES:
            category = None
        itemnum = int(self.network.itemnum)
        maxlen = int(self.network.maxlen)
        rng = np.random.default_rng(42)
        users = sorted(histories, key=lambda u: str(u))
        order = rng.permutation(len(users))
        reasons = Counter()
        records = []
        considered = 0
        selected_users = []
        unavailable_cases = []
        for index in order:
            # Cap attempted eligible cases, including nonfinite model outputs.
            # Otherwise failed scoring can trigger an unbounded replacement search
            # and make the selected population depend on the fitted model.
            if len(selected_users) >= 200:
                break
            considered += 1
            user = users[int(index)]
            full = [int(i) for i in histories[user]]
            if any(i < 1 or i > itemnum for i in full):
                reasons["invalid_training_item_id"] += 1
                continue
            history = full[:-1][-maxlen:]
            if len(history) < 3:
                reasons["insufficient_history"] += 1
                continue
            target = full[-1]
            excluded = set(full)
            available = itemnum - len(excluded)
            if available < 99:
                reasons["insufficient_catalog"] += 1
                continue
            # Rejection sampling is uniform without replacement and bounded.
            negatives = set()
            for _ in range(32):
                draws = rng.integers(1, itemnum + 1, size=256)
                for draw in draws:
                    item = int(draw)
                    if item not in excluded:
                        negatives.add(item)
                        if len(negatives) == 99:
                            break
                if len(negatives) == 99:
                    break
            if len(negatives) < 99:
                pool = np.asarray(
                    [
                        i
                        for i in range(1, itemnum + 1)
                        if i not in excluded and i not in negatives
                    ],
                    dtype=np.int64,
                )
                negatives.update(
                    map(int, rng.choice(pool, 99 - len(negatives), replace=False))
                )
            candidates = [target] + sorted(negatives)
            selected_users.append(str(user))
            n = len(history)
            k = min(5, (n - 1) // 2)
            positions = list(range(n))
            retained = {
                "intact": positions,
                "suffix": positions[-min(10, n) :],
                "remove_recent_context": positions[: n - 1 - k] + positions[-1:],
                "remove_old_context": positions[k:],
                "remove_anchor": positions[:-1],
            }
            raw_times = times.get(user, times.get(str(user), []))
            aligned_times = (
                list(raw_times[:-1][-maxlen:])
                if len(raw_times) == len(full)
                else [None] * n
            )
            sequence = np.zeros((len(self.CONDITIONS), maxlen), dtype=np.int64)
            for row, name in enumerate(self.CONDITIONS):
                items = [history[j] for j in retained[name]]
                sequence[row, -len(items) :] = items
            seq_t = torch.as_tensor(sequence, device=self.device, dtype=torch.long)
            features = self.model.log2feats(seq_t)
            positions_t = torch.arange(maxlen, device=seq_t.device).expand_as(seq_t)
            final = positions_t.masked_fill(seq_t == 0, -1).amax(dim=1)
            features = features[torch.arange(len(sequence), device=seq_t.device), final]
            candidate_t = torch.as_tensor(
                candidates, device=self.device, dtype=torch.long
            )
            candidate_t = candidate_t.unsqueeze(0).expand(len(sequence), -1)
            scores = (
                self.model.predict_score(features, candidate_t)
                .detach()
                .double()
                .cpu()
                .numpy()
            )
            if scores.shape != (len(self.CONDITIONS), 100):
                raise ValueError(
                    "Diagnostic predict_score must return [conditions, candidates]"
                )
            if not np.isfinite(scores).all():
                reasons["nonfinite_candidate_scores"] += 1
                unavailable_cases.append(
                    {
                        "user_id": str(user),
                        "target": target,
                        "history": history,
                        "timestamps": [
                            None if t is None else str(t) for t in aligned_times
                        ],
                        "candidates": candidates,
                        "reason": "nonfinite_candidate_scores",
                    }
                )
                continue
            candidate_ids = np.asarray(candidates)
            rankings = [np.lexsort((candidate_ids, -row)) for row in scores]
            intact_rank = int(np.flatnonzero(rankings[0] == 0)[0]) + 1
            intact_top = set(candidate_ids[rankings[0][:10]].tolist())
            conditions = {}
            for row, name in enumerate(self.CONDITIONS):
                keep = retained[name]
                rank = int(np.flatnonzero(rankings[row] == 0)[0]) + 1
                conditions[name] = {
                    "rank": rank,
                    "hit": int(rank <= 10),
                    "delta_rank": rank - intact_rank,
                    "delta_hr": int(rank <= 10) - int(intact_rank <= 10),
                    "top10_overlap": len(
                        intact_top & set(candidate_ids[rankings[row][:10]].tolist())
                    )
                    / 10.0,
                    "top10_item_ids": candidate_ids[rankings[row][:10]].tolist(),
                    "retained_indices": keep,
                    "retained_length": len(keep),
                    "timestamp_span": self._timestamp_span(
                        [aligned_times[j] for j in keep]
                    ),
                    "anchor_item": history[keep[-1]],
                    "anchor_preserved": keep[-1] == n - 1,
                    "repeat_target_present": target in [history[j] for j in keep],
                    "affected": keep != positions,
                }
            records.append(
                {
                    "category": category,
                    "user_id": str(user),
                    "target": target,
                    "original_length": n,
                    "full_training_length": len(full),
                    "history": history,
                    "timestamps": [
                        None if t is None else str(t) for t in aligned_times
                    ],
                    "timestamp_span": self._timestamp_span(aligned_times),
                    "repeat_target": target in history,
                    "k": k,
                    "candidates": candidates,
                    "conditions": conditions,
                }
            )
        result = {
            "category": category,
            "category_identity_available": category is not None,
            "scope": "in_sample_training_prefix",
            "seed": 42,
            "considered_users": considered,
            "selected_users": len(selected_users),
            "selected_user_ids": selected_users,
            "scored_users": len(records),
            "eligible_users": len(records),
            "unavailable_scoring_cases": unavailable_cases,
            "not_considered_users": len(users) - considered,
            "unavailable_users": int(sum(reasons.values())),
            "unavailable_reasons": dict(reasons),
            "records": records,
            "summary": self._paired_summary(records),
            "geometry_preserving_masks": {
                "value": None,
                "reason": "legacy_compacted_probe; see_separate_geometry_key_access_report_for_value_gates",
            },
            "authorized_validation": {
                "value": None,
                "reason": "exact_host_cases_and_pinned_candidates_not_exposed",
            },
            "four_category_macro": {
                "value": None,
                "reason": "single_view_report; aggregate_categories_requires_all_four_explicit_category_reports",
            },
            "settings": {
                "maximum_users": 200,
                "maximum_paired_forward_batches": 200,
                "selection_failure_policy": "record_nonfinite_scores_without_replacement",
                "maximum_history": maxlen,
                "negatives": 99,
                "bootstrap_replicates": 2000,
                "tie_rule": "descending_score_then_ascending_item_id",
                "deletion": "compacted_without_backfill; joint_evidence_and_geometry_intervention",
                "temporal_model": "supplied_HSTU_does_not_consume_timestamps",
                "strata": "not_computed",
                "spectra": "disabled_to_bound_diagnostic_cost",
            },
        }
        # Fingerprint identifies this public training input, without consulting other trials.
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
        path = os.path.join(directory, label + "_paired_context.json")
        result["artifact_path"] = path
        with open(path, "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return result

    def _geometry_key_audit(self, paired):
        """Inference-only post-softmax gates for the supplied MHA backbone.

        This is deliberately not described as post-SiLU attention: this model's
        attention uses softmax. Neither additive position embeddings nor absent
        timestamps supply the separable attention-bias factors in the proposal.
        """
        import os
        import math

        result = {
            "category": paired["category"],
            "scope": "in_sample_inference_intervention_on_one_checkpoint",
            "prespecified_contrast": "HR_recent_minus_HR_older; negative means greater recent-context sensitivity",
            "position_bias_factorial": {
                "supported": False,
                "reason": "backbone_has_additive_position_embeddings_not_separable_attention_position_bias",
            },
            "time_bias_removal": {
                "supported": False,
                "reason": "backbone_does_not_consume_timestamps",
            },
            "legacy_anchor_analysis": "separate_compacted_remove_anchor_cell_in_paired_recent_context",
            "records": [],
            "unavailable_cases": [],
            "correctness": {},
            "settings": {
                "seed": 42,
                "maximum_users": 200,
                "attention": "original_softmax_then_key_column_gate_without_renormalization",
                "blocking": "every_head_every_layer; preserve_width_slots_residuals_queries_diagonal_rule",
                "position_embeddings": "unchanged_in_every_cell",
                "timestamps": "not_consumed_by_backbone",
                "tie_rule": "stable_descending_score_with_fixed_seeded_candidate_permutation",
                "bootstrap_replicates": 2000,
                "interval_scope": "conditional_on_checkpoint_users_and_candidates",
                "host_evaluator": "unchanged; host_tie_policy_not_supplied",
            },
        }
        net = self.network
        blocks = getattr(net, "blocks", None)
        supported = (
            blocks is not None
            and len(blocks) > 0
            and hasattr(net, "item_emb")
            and all(
                isinstance(getattr(b, "attn", None), torch.nn.MultiheadAttention)
                and b.attn.batch_first
                and b.attn._qkv_same_embed_dim
                and b.attn.in_proj_weight is not None
                and b.attn.bias_k is None
                and b.attn.bias_v is None
                and not b.attn.add_zero_attn
                for b in blocks
            )
        )
        if not supported:
            result["status"] = "unsupported_attention_implementation"
            return result

        dtype = net.item_emb.weight.dtype
        atol, rtol = (
            (3e-3, 3e-3) if dtype in (torch.float16, torch.bfloat16) else (2e-5, 2e-4)
        )
        result["correctness"]["tolerance"] = {"atol": atol, "rtol": rtol}

        def forward(
            sequence,
            blocked,
            capture=None,
            perturb=None,
            value_capture=None,
            value_perturb=None,
        ):
            handles = []
            saved_forwards = []

            def attention_hook(module, args, kwargs, original):
                q, k, v = args[:3]
                if not (q is k and k is v):
                    raise ValueError("Audit requires the supplied self-attention path")
                if kwargs.get("key_padding_mask") is not None:
                    raise ValueError("Unexpected separate padding mask")
                batch, length, width = q.shape
                heads = module.num_heads
                dim = width // heads
                projected = F.linear(q, module.in_proj_weight, module.in_proj_bias)
                query, key, value = projected.chunk(3, dim=-1)

                def split(x):
                    return x.reshape(batch, length, heads, dim).transpose(1, 2)

                query, key, value = map(split, (query, key, value))
                if value_capture is not None:
                    value_capture.append(value)
                if value_perturb is not None:
                    value = value + (
                        blocked[:, None, :, None].to(value.dtype)
                        * value_perturb.reshape(1, heads, 1, dim)
                    )
                logits = (query / math.sqrt(dim)) @ key.transpose(-2, -1)
                mask = kwargs.get("attn_mask")
                if mask is None or mask.dtype != torch.bool:
                    raise ValueError(
                        "Expected the supplied boolean causal/padding mask"
                    )
                if mask.ndim == 2:
                    mask = mask[None, None]
                else:
                    mask = mask.reshape(batch, heads, length, length)
                weights = logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
                # Gate AFTER the original normalization. Blocked keys still
                # influence the softmax denominator: this removes value access,
                # not all dependence on blocked embedding occurrences.
                allowed_keys = ((sequence != 0) & ~blocked)[:, None, None, :]
                weights = weights * allowed_keys.to(weights.dtype)
                output = (weights @ value).transpose(1, 2).reshape(batch, length, width)
                output = F.linear(output, module.out_proj.weight, module.out_proj.bias)
                return output, None

            def embedding_hook(module, args, output):
                if capture is not None:
                    output = output.detach().requires_grad_(True)
                    capture.append(output)
                if perturb is not None:
                    output = output + perturb
                return output

            try:
                # A forward hook would compute and discard the original MHA
                # output, doubling attention work and retaining unused graphs.
                for block in blocks:
                    module = block.attn
                    had_override = "forward" in module.__dict__
                    saved_forwards.append(
                        (module, had_override, module.__dict__.get("forward"))
                    )

                    def replacement(*args, _module=module, **kwargs):
                        return attention_hook(_module, args, kwargs, None)

                    module.forward = replacement
                if capture is not None or perturb is not None:
                    handles.append(net.item_emb.register_forward_hook(embedding_hook))
                return net.log2feats(sequence)
            finally:
                for handle in handles:
                    handle.remove()
                for module, had_override, original_forward in reversed(saved_forwards):
                    if had_override:
                        module.forward = original_forward
                    else:
                        del module.forward

        def readout(features, sequence):
            positions = torch.arange(sequence.shape[1], device=sequence.device)
            final = positions.expand_as(sequence).masked_fill(sequence == 0, -1).amax(1)
            if (final < 0).any():
                raise ValueError("Audit requires a real readout item")
            return features[torch.arange(len(sequence), device=sequence.device), final]

        def agree(a, b):
            return bool(
                torch.isfinite(a).all()
                and torch.isfinite(b).all()
                and torch.allclose(a, b, atol=atol, rtol=rtol)
            )

        rng = np.random.default_rng(42)
        permutation_rng = np.random.default_rng(42)
        # Preserve selection order independently of legacy scoring outcomes.
        by_user = {
            p["user_id"]: p
            for p in list(paired["records"]) + list(paired["unavailable_scoring_cases"])
        }
        panels = [by_user[user] for user in paired["selected_user_ids"]]
        result["settings"][
            "intervention_scope"
        ] = "post_softmax_value_access_gate; blocked_keys_remain_in_normalizer"
        result["blocked_embedding_isolation"] = {
            "supported": False,
            "reason": "blocked_keys_affect_softmax_denominators_even_after_value_gating",
        }
        result["selected_users"] = len(panels)
        result["source_exclusions"] = paired["unavailable_reasons"]
        tested = False
        for panel in panels:
            history = panel["history"]
            length = int(net.maxlen)
            sequence = torch.zeros((1, length), dtype=torch.long, device=self.device)
            sequence[0, -len(history) :] = torch.as_tensor(history, device=self.device)
            valid = torch.nonzero(sequence[0] != 0, as_tuple=False).flatten().tolist()
            context, anchor = valid[:-1], valid[-1]
            k = min(5, len(context) // 2)
            if k < 1:
                result["unavailable_cases"].append(
                    {"user_id": panel["user_id"], "reason": "insufficient_context"}
                )
                continue
            sets = {
                "intact": [],
                "recent": context[-k:],
                "older": context[:k],
                "random": sorted(map(int, rng.choice(context, k, replace=False))),
            }
            candidates = np.asarray(panel["candidates"], dtype=np.int64)
            candidates = candidates[permutation_rng.permutation(len(candidates))]
            target_index = int(np.flatnonzero(candidates == panel["target"])[0])
            items = torch.as_tensor(candidates, device=self.device, dtype=torch.long)
            empty = torch.zeros_like(sequence, dtype=torch.bool)
            masks = {}
            for name, positions in sets.items():
                mask = empty.clone()
                mask[0, positions] = True
                masks[name] = mask
            with torch.no_grad():
                original = net.log2feats(sequence)
                baseline = forward(sequence, empty)
                fixed_embeddings = net.item_emb(items).detach().clone()
                original_scores = self.model.predict_score(
                    readout(original, sequence)[0], items
                )
                reconstructed_scores = (
                    readout(baseline, sequence)[0] @ fixed_embeddings.T
                )
                finite = all(
                    bool(torch.isfinite(x).all())
                    for x in (original, baseline, original_scores, reconstructed_scores)
                )
                agreement = (
                    finite
                    and agree(original, baseline)
                    and agree(original_scores, reconstructed_scores)
                )
            if not finite:
                result["unavailable_cases"].append(
                    {
                        "user_id": panel["user_id"],
                        "reason": "nonfinite_empty_intervention_features_or_scores",
                        "candidates": candidates.tolist(),
                    }
                )
                continue
            if not agreement:
                result["status"] = "failed_empty_intervention_agreement"
                result["correctness"]["empty_agreement"] = False
                result["records"] = []
                break
            if not tested:
                with torch.no_grad():
                    base = forward(sequence, masks["recent"])
                    direction = torch.linspace(
                        -0.2, 0.2, net.hidden_units, device=sequence.device, dtype=dtype
                    )
                    perturb = masks["recent"].unsqueeze(-1) * direction
                    changed = forward(sequence, masks["recent"], perturb=perturb)
                    isolation = agree(
                        readout(base, sequence), readout(changed, sequence)
                    )
                    score_isolation = agree(
                        readout(base, sequence) @ fixed_embeddings.T,
                        readout(changed, sequence) @ fixed_embeddings.T,
                    )
                    value_changed = forward(
                        sequence, masks["recent"], value_perturb=direction
                    )
                    value_isolation = agree(
                        readout(base, sequence), readout(value_changed, sequence)
                    ) and agree(
                        readout(base, sequence) @ fixed_embeddings.T,
                        readout(value_changed, sequence) @ fixed_embeddings.T,
                    )
                    # Several actual left-padding lengths at the original width.
                    padding_checks = []
                    causal_checks = []
                    for count in sorted({min(len(history), n) for n in (2, 3, 5)}):
                        test = torch.zeros_like(sequence)
                        test[0, -count:] = sequence[0, -count:]
                        feats = net.log2feats(test)
                        padding_checks.append(
                            agree(net.final_features(test), readout(feats, test))
                        )
                        earlier = length - 2
                        altered = test.clone()
                        altered[0, -1] = next(
                            int(item)
                            for item in candidates
                            if int(item) != int(test[0, -1])
                        )
                        causal_checks.append(
                            agree(feats[:, earlier], net.log2feats(altered)[:, earlier])
                        )
                        test_mask = torch.zeros_like(test, dtype=torch.bool)
                        test_mask[0, -count] = True
                        causal_checks.append(
                            agree(
                                forward(test, test_mask)[:, earlier],
                                forward(altered, test_mask)[:, earlier],
                            )
                        )
                    unblocked = forward(sequence, empty, perturb=perturb)
                    sensitivity = float(
                        (readout(unblocked, sequence) - readout(baseline, sequence))
                        .abs()
                        .max()
                    )
                with torch.enable_grad():
                    captured, captured_values = [], []
                    feats = forward(
                        sequence,
                        masks["recent"],
                        capture=captured,
                        value_capture=captured_values,
                    )
                    # Embedding dependence through normalization is measured,
                    # while blocked projected-value derivatives must be zero.
                    # autograd.grad leaves parameter .grad buffers untouched.
                    jacobian_max = value_jacobian_max = 0.0
                    jacobian_finite = True
                    final_features = readout(feats, sequence)[0]
                    for feature in range(net.hidden_units):
                        gradients = torch.autograd.grad(
                            final_features[feature],
                            (captured[0], *captured_values),
                            retain_graph=feature + 1 < net.hidden_units,
                        )
                        jacobian_finite &= all(
                            bool(torch.isfinite(g).all()) for g in gradients
                        )
                        jacobian_max = max(
                            jacobian_max,
                            float(gradients[0][masks["recent"]].abs().max().detach()),
                        )
                        for gradient in gradients[1:]:
                            selected = masks["recent"][:, None, :, None].expand_as(
                                gradient
                            )
                            value_jacobian_max = max(
                                value_jacobian_max,
                                float(gradient[selected].abs().max().detach()),
                            )
                # Source-policy check, not an independent observation of MHA.
                causal = torch.triu(
                    torch.ones(
                        (length, length), dtype=torch.bool, device=sequence.device
                    ),
                    diagonal=1,
                )
                mask = causal[None] | (sequence == 0)[:, None, :]
                mask.diagonal(dim1=-2, dim2=-1).fill_(False)
                diagonal_ok = bool((~mask.diagonal(dim1=-2, dim2=-1)).all())
                passed = (
                    value_isolation
                    and jacobian_finite
                    and value_jacobian_max == 0.0
                    and all(padding_checks)
                    and all(causal_checks)
                    and diagonal_ok
                )
                result["correctness"].update(
                    {
                        "empty_agreement": True,
                        "blocked_occurrence_feature_isolation": isolation,
                        "blocked_occurrence_fixed_candidate_score_isolation": score_isolation,
                        "blocked_occurrence_readout_jacobian_max_abs": (
                            jacobian_max if jacobian_finite else None
                        ),
                        "blocked_projected_value_isolation": value_isolation,
                        "blocked_projected_value_jacobian_max_abs": (
                            value_jacobian_max if jacobian_finite else None
                        ),
                        "jacobians_finite": jacobian_finite,
                        "embedding_isolation_required": False,
                        "embedding_isolation_reason": "softmax_normalization_retains_key_dependence",
                        "unblocked_perturbation_feature_max_abs_change": sensitivity,
                        "unblocked_sensitivity_observed": sensitivity > atol,
                        "future_token_invariance": all(causal_checks),
                        "greatest_nonpadding_readout": all(padding_checks),
                        "expected_source_diagonal_allowed": diagonal_ok,
                        "padding_width_invariance": "not_required_or_tested",
                        "tested_user_id": panel["user_id"],
                    }
                )
                if not passed or sensitivity <= atol:
                    result["status"] = (
                        "correctness_failed_or_sensitivity_control_inconclusive"
                    )
                    break
                tested = True
            cells = {}
            with torch.no_grad():
                for name, blocked in masks.items():
                    feats = baseline if name == "intact" else forward(sequence, blocked)
                    scores = (
                        (readout(feats, sequence)[0] @ fixed_embeddings.T)
                        .double()
                        .cpu()
                        .numpy()
                    )
                    if not np.isfinite(scores).all():
                        break
                    order = np.argsort(-scores, kind="stable")
                    rank = int(np.flatnonzero(order == target_index)[0]) + 1
                    cutoff = scores[order[9]]
                    counts = [sum(j <= query for j in sets[name]) for query in valid]
                    cells[name] = {
                        "rank": rank,
                        "hit": int(rank <= 10),
                        "margin": float(
                            scores[target_index]
                            - np.max(np.delete(scores, target_index))
                        ),
                        "positive_cutoff_tie": bool(
                            scores[target_index] == cutoff
                            and np.count_nonzero(scores == cutoff) > 1
                        ),
                        "top10_item_ids": candidates[order[:10]].tolist(),
                        "blocked_positions": sets[name],
                        "blocked_distances_from_anchor": [
                            anchor - j for j in sets[name]
                        ],
                        "causally_eligible_blocked_counts_by_valid_query": counts,
                        "surviving_key_counts_by_valid_query": [
                            i + 1 - counts[i] for i in range(len(valid))
                        ],
                    }
            if len(cells) != len(sets):
                result["unavailable_cases"].append(
                    {
                        "user_id": panel["user_id"],
                        "reason": "nonfinite_candidate_scores",
                        "candidates": candidates.tolist(),
                    }
                )
                continue
            for cell in cells.values():
                cell["delta_hr"] = cell["hit"] - cells["intact"]["hit"]
                cell["delta_rank"] = cell["rank"] - cells["intact"]["rank"]
                cell["top10_overlap"] = (
                    len(
                        set(cell["top10_item_ids"])
                        & set(cells["intact"]["top10_item_ids"])
                    )
                    / 10.0
                )
            result["records"].append(
                {
                    "user_id": panel["user_id"],
                    "target": panel["target"],
                    "sequence": sequence[0].cpu().tolist(),
                    "anchor_position": anchor,
                    "timestamps": panel["timestamps"],
                    "k": k,
                    "intervening_context_positions": context[k:-k],
                    "candidates": candidates.tolist(),
                    "target_candidate_index": target_index,
                    "conditions": cells,
                }
            )
        result.setdefault(
            "status", "measured" if result["records"] else "no_eligible_users"
        )
        if result["status"] != "measured":
            result["records"] = []
        result["scored_users"] = len(result["records"])
        result["summary"] = self._geometry_summary(
            {result["category"] or "unidentified_category": result["records"]}
        )
        directory = os.path.dirname(paired["artifact_path"])
        path = os.path.join(
            directory,
            os.path.basename(paired["artifact_path"]).replace(
                "_paired_context.json", "_geometry_keys.json"
            ),
        )
        result["artifact_path"] = path
        with open(path, "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return result

    @staticmethod
    def _geometry_summary(groups):
        if not groups or any(not rows for rows in groups.values()):
            return {"value": None, "reason": "missing_eligible_records"}
        rng = np.random.default_rng(42)
        names, category_means, category_bootstraps = None, [], []
        for rows in groups.values():
            values = []
            for row in rows:
                cells = row["conditions"]
                metrics, labels = [], []
                for condition in ("intact", "recent", "older", "random"):
                    for key in (
                        "hit",
                        "rank",
                        "margin",
                        "delta_hr",
                        "delta_rank",
                        "top10_overlap",
                        "positive_cutoff_tie",
                    ):
                        labels.append(condition + "." + key)
                        metrics.append(cells[condition][key])
                labels.append("recent_minus_older_hr")
                metrics.append(cells["recent"]["hit"] - cells["older"]["hit"])
                values.append(metrics)
            names = labels
            values = np.asarray(values, dtype=np.float64)
            indices = rng.integers(0, len(rows), size=(2000, len(rows)))
            category_means.append(values.mean(0))
            category_bootstraps.append(values[indices].mean(1))
        mean = np.mean(category_means, axis=0)
        bounds = np.percentile(
            np.mean(category_bootstraps, axis=0), [2.5, 97.5], axis=0
        )
        return {
            "value": {
                name: {
                    "mean": float(mean[j]),
                    "ci95": [float(bounds[0, j]), float(bounds[1, j])],
                }
                for j, name in enumerate(names)
            },
            "users": {category: len(rows) for category, rows in groups.items()},
            "reason": None,
            "position_bias_interaction": {
                "value": None,
                "reason": "unsupported_factor",
            },
        }

    @classmethod
    def aggregate_geometry_categories(cls, reports):
        missing = [category for category in cls.CATEGORIES if category not in reports]
        if missing:
            return {
                "value": None,
                "reason": "missing_category_reports",
                "missing": missing,
            }
        groups = {}
        for category in cls.CATEGORIES:
            report = reports[category]["metrics"].get("geometry_key_access", {})
            if report.get("category") != category or report.get("status") != "measured":
                return {"value": None, "reason": "unverified_or_unavailable_category"}
            groups[category] = report["records"]
        return cls._geometry_summary(groups)

    def run_full_diagnosis(self, train_loader):
        from itertools import islice

        previous_modes = [
            (module, module.training) for module in self.network.modules()
        ]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
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
            # Diagnosis must not change subsequent training samples or dropout.
            # fork_rng restores CPU and the single model GPU even on exceptions.
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
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            islice(train_loader, 10),
                            num_batches=10,
                            eps=0.5,
                            topk_categories=20,
                        )
                    )
                    paired = self._paired_recent_context()
                    geometry = self._geometry_key_audit(paired)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, mode in previous_modes:
                module.training = mode
        return {
            "metrics": {
                "embedding_mean_offdiagonal_cosine": cosine,
                "embedding_cosine_unavailable_reason": (
                    "fewer_than_two_finite_nonzero_sampled_rows"
                    if cosine is None
                    else None
                ),
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_valid_positions": margin_metrics["valid_positions"],
                "pos_neg_margin_unavailable_reason": margin_metrics[
                    "unavailable_reason"
                ],
                "pos_neg_margin_category_breakdown": breakdown,
                "paired_recent_context": paired,
                "geometry_key_access": geometry,
            },
            "metric_definitions": {
                "embedding_mean_offdiagonal_cosine": (
                    "Raw mean off-diagonal cosine over up to 2048 uniformly sampled real-item rows, "
                    "seed 42; computed in float64 after excluding nonfinite rows or norms<=1e-12. "
                    "Uses (||sum(unit_rows)||^2-N)/(N*(N-1)); requires N>=2. "
                    "Directional similarity, not a calibrated collapse probability or evidence of HR failure."
                ),
                "pos_neg_margin_mean": "Mean positive-minus-negative score over finite nonpadding sequence/positive/negative positions.",
                "pos_neg_margin_p10": "Nearest-rank tenth percentile of those margins; null when no valid positions exist.",
                "pos_neg_neg_beats_pos_rate": "Fraction of valid margins below zero.",
                "pos_neg_small_margin_rate": "Fraction of valid margins below eps=0.5; score-scale dependent.",
                "pos_neg_margin_category_breakdown": "Item categories unavailable; category lists intentionally empty. Position counts remain measured.",
                "geometry_key_access": {
                    "implementation": "Actual supplied backbone uses MHA softmax attention. Reconstruct its attention and apply key-column gates after softmax at every layer/head without renormalization. Default model and predictor are unchanged. Empty-gate features and scores must agree with original calls for every included user.",
                    "population": "Reuse the structurally selected training-user panel and its fixed negatives, including original scoring failures. Last training item is target; this is in-sample. Seeded candidate permutation is fixed across cells. Failed correctness checks suppress all estimates; nonfinite scoring users are recorded without replacement.",
                    "conditions": "Intact, k most recent, k oldest, and uniform random k context positions blocked, k=min(5,floor(context_count/2)). Context excludes the final observed anchor. Positions, padded width, additive positions, residuals and query paths remain fixed. Intermediate-query eligible blocked counts can differ and are recorded.",
                    "metrics": "Rank is one-based stable descending-score rank in the saved candidate permutation. Hit is rank<=10. Margin is target minus highest negative. Deltas are cell minus intact. Top10 overlap is intersection/10. Positive cutoff ties require the target score to equal the tenth score and at least one other candidate score. Nonfinite scores exclude the paired user and are reported.",
                    "unsupported": "No separable attention position bias or temporal bias exists in this source. Position-bias-off cells, time-bias removal and the factorial interaction are unavailable; removing additive position embeddings would be a different intervention.",
                    "correctness": "Require empty reconstruction agreement, blocked projected-value perturbation isolation and zero projected-value readout Jacobians at every layer, future-token invariance and greatest-nonpadding readout agreement. Blocked embedding perturbations and Jacobians are measured but need not vanish because softmax denominators retain blocked-key dependence. The diagonal check verifies the expected source policy rather than independently observing attention execution. Gradient probes use autograd.grad without changing parameter gradient buffers. Unblocked embedding perturbation is a sensitivity control. No invariance across padded widths is asserted.",
                    "uncertainty": "2000 seed-42 user bootstrap resamples with shared indices across cells/statistics. Macro aggregation requires four identified measured reports and averages categories equally, resampling users independently within categories. Intervals condition on fitted checkpoints and fixed candidates; training and negative-sampling uncertainty are excluded.",
                    "interpretation": "Prespecified recent-minus-older hit contrast: negative means greater recent-context dependence. Useful dependence can produce large damage. These inference interventions do not establish harmful overreliance, beneficial regularization or retrained-model performance. Legacy compacted anchor removal remains separate.",
                },
                "paired_recent_context": {
                    "selection": "Sorted string user IDs, seed-42 permutation; select at most 200 structurally eligible training users per category before inspecting scores. Nonfinite scoring cases are recorded without replacement and excluded from paired estimates; report selected and scored counts separately. Last training item is target; preceding history truncated once to model.maxlen. Requires at least three items.",
                    "scope": "All prefix results are in-sample: fitting may already have used the reserved target. Earlier target occurrences remain and are flagged. No validation reproduction without exact authorized host candidates.",
                    "conditions": "n=bounded length; k=min(5,(n-1)//2). Intact=history; suffix=last min(10,n); recent deletion=history[:n-1-k]+history[-1:]; old deletion=history[k:]; anchor deletion=history[:-1]. No backfill. Recent/old remove equal counts and preserve anchor; suffix is descriptive, and unchanged suffixes are unaffected. Anchor deletion changes readout item.",
                    "geometry": "These legacy cells compact histories and read the greatest nonzero index, changing positions and evidence jointly. The separate geometry_key_access audit preserves slots with post-softmax value gates, retaining blocked-key normalization dependence. Original aligned timestamps are recorded; this HSTU has no temporal input.",
                    "candidates": "99 unique uniform real-item negatives without replacement, excluding every training item, padding and reserved IDs. Identical 100 candidates across conditions. Rank=1+preceding count under descending score, ascending item ID ties; this diagnostic tie rule is not a claim about host ties.",
                    "effects": "Hit=rank<=10; delta_rank=perturbed-intact rank; delta_hr=perturbed-intact hit. Top10 overlap=intersection with intact top10 divided by 10. All paired estimates use identical eligible users.",
                    "uncertainty": "2000 paired user bootstrap replicates, seed 42, same user indices across conditions; percentile 95% intervals. Conditional on fitted model, selected users and fixed candidates; excludes training-seed/candidate variation and repeated-selection optimism.",
                    "macro": "aggregate_categories accepts four explicitly identified reports from this trial, resamples within each category and equally averages category means. Missing or empty categories yield unavailable, never zero imputation. Single-view reports contain no purported macro estimate.",
                    "timestamps": "Span=max-min of surviving original timestamps. Numeric values retain source units; parseable ISO values use seconds. Missing or unparseable timestamps yield null spans.",
                    "cost": "Legacy probe: at most 200 batches of five sequences. Additional value-gate audit: at most 200 users, five forward calls per ordinary user plus bounded correctness forwards and one hidden-width Jacobian sweep on the first finite case. At most ten margin batches and 2048 embeddings. Nonfinite users are recorded without replacement. No auxiliary predictor, checkpoint change or score fusion.",
                    "rng_isolation": "Python, NumPy global, Torch CPU and the model GPU RNG states are restored after diagnosis, including exceptions. Explicit independent loader generators or persistent worker state remain owned by the caller; use a dedicated diagnostic loader for those cases.",
                },
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
