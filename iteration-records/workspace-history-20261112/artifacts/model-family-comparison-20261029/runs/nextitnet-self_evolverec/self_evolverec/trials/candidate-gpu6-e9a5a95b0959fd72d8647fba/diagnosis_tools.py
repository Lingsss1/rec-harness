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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret catalog competition as conditional training fit
    - Catalog ranks describe training-event fit, never validation HR@10, generalization, or improvement.
    - Distinguish repeat-capable and unseen-target populations and their denominators.
    - Report sampling allocation, identifiable users/events, exclusions, numerical configuration, and missing measurements.
    - Rank intervals reflect exact computed-score ties; they are not confidence intervals or the host tie policy.
    - Supplied-negative margins and full-catalog ranks measure different competition; compare only disclosed paired populations.
    - Stratified selected-event averages do not estimate the natural training-event distribution.
    - Do not infer intent, dislike, exposure propensity, satisfaction, simulator traits, or unavailable metadata.
    - Missing or failed audits are unknown. Do not classify missing measurements as healthy behavior.
    - Model selection remains exclusively pinned four-category macro validation HR@10.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family view and unavailable metadata contract
    def __init__(self, view, device):
        self.model = view
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        return []  # Item categories are unavailable in this benchmark.

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
            if n <= 2:
                return 0.0

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
            log_feats = self.model.log2feats(seq_np)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
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

    ### >>> Self_EvolveRec-BLOCK-START: Bounded catalog audit with exact reference checks and explicit failures
    def run_full_diagnosis(self, train_loader):
        import os
        import time

        started = time.perf_counter()
        actual = self.model.model
        modes = [(module, module.training) for module in actual.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        previous = {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        }
        counts = Counter()
        rng = np.random.default_rng(42)
        vocab = int(actual.itemnum)
        width = 4096
        cap = 128
        records = []
        allocation = []
        metrics = {
            "scope": "Training-event fit; not host validation or held-out accuracy.",
            "catalog_audit": None,
            "catalog_audit_missing_reason": None,
            "category_breakdown_missing_reason": "Item categories are unavailable.",
        }
        definitions = {
            "scope": "Only public training histories and aligned training targets are used. Targets may have influenced fitting. No additional predictor is fitted.",
            "identity": "Within this category invocation, identity is (training user ID, zero-based target endpoint). All matching endpoints are searched; ambiguous encoded-prefix/target matches are excluded. The category is supplied by the host's per-category invocation.",
            "sampling": "At most ten loader batches and 128 unique events. Target-frequency quintiles keep equal frequencies together. Length cells are 1-5, 6-15, 16-31, 32+. Even allocation over nonempty cells, capacity capped, followed by fixed cell-order redistribution; seed 42 sampling without replacement. Summaries describe selected events with equal event weights, not the natural training population.",
            "repeat_capable": "All IDs 1..itemnum are eligible, including previously consumed items. Padding is excluded. Repeat means the target occurs in complete strictly pre-target training history.",
            "unseen_target": "Repeat-target events are excluded. For first-time targets, every item in complete strictly pre-target training history is excluded from candidates.",
            "rank_bounds": "Excluding the target, g counts computed scores strictly greater than the target and t counts exactly equal scores. One-based bounds are [1+g,1+g+t]. HR@10 lower uses the upper rank; HR@10 upper uses the lower rank. These are training diagnostic tie bounds, not the host tie policy.",
            "rank_id_ascending": "Diagnostic total order: descending computed score, then ascending integer item ID. Exact equality is used, without tolerance.",
            "normalized_rank_bounds": "(rank-1)/(eligible_candidates-1), missing for singleton candidate sets.",
            "tenth_competitor_margin": "Target score minus the tenth highest eligible alternative score. Competitors use descending score then ascending ID. Missing with fewer than ten alternatives. Unobserved items are not confirmed negatives.",
            "supplied_negative": "The first cached negative for each uniquely identified event is retained; repeated draws are not averaged. It must be a valid catalog ID distinct from the target. original_supplied_negative_events records selected events with finite features and target scores before catalog rank exclusions. Missing margins indicate invalid or nonfinite negatives. Margins use target and negative scores extracted from the same first-pass catalog calls, with subtraction in Python floating point. Protocol summaries condition on valid catalog results; paired summaries additionally require an eligible finite supplied negative. These populations are distinct from the original cached-row diagnostic population.",
            "numerics": "FP32 features and scoring, autocast disabled, TF32 disabled, deterministic algorithms requested. Each target is extracted from its actual catalog chunk before masking. A second pass repeats identical feature/chunk shapes and calls. Ranks are exact for these computed scores and eligibility only; shape invariance is not claimed.",
            "validation": "Streaming counts, stable competitor identities and scores are checked against materialized identical second-pass chunk scores for up to two selected events. Synthetic checks cover ties, exclusions, repeats and fewer than ten alternatives. This does not certify another GEMM shape.",
            "uncertainty": "No confidence intervals are computed. Results condition on this selected stratified sample, checkpoint and numerical configuration; training, search and event-sampling uncertainty are not estimated.",
            "counts": "Counts distinguish cached rows, unique eligible event identities, sampling and numerical exclusions. Missing measurements are unknown, not zero or evidence of health.",
        }

        def length_band(n):
            return 0 if n <= 5 else 1 if n <= 15 else 2 if n <= 31 else 3

        def summary(rows):
            output = {"events": len(rows), "users": len({r["user"] for r in rows})}
            for name in (
                "rank_lower",
                "rank_upper",
                "rank_id_ascending",
                "normalized_rank_lower",
                "normalized_rank_upper",
                "tenth_competitor_margin",
                "supplied_negative_margin",
            ):
                values = [r[name] for r in rows if r.get(name) is not None]
                output[name] = {
                    "observations": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "quantiles_10_50_90": (
                        np.quantile(values, [0.1, 0.5, 0.9]).tolist()
                        if values
                        else None
                    ),
                    "missing_reason": (
                        None if values else "No eligible finite observations."
                    ),
                }
            output["hr10_lower"] = (
                float(np.mean([r["rank_upper"] <= 10 for r in rows])) if rows else None
            )
            output["hr10_upper"] = (
                float(np.mean([r["rank_lower"] <= 10 for r in rows])) if rows else None
            )
            paired = [
                r
                for r in rows
                if r.get("negative_eligible")
                and r.get("supplied_negative_margin") is not None
            ]
            output["eligible_negative_paired"] = {
                "events": len(paired),
                "negative_win_rate": (
                    float(np.mean([r["supplied_negative_margin"] < 0 for r in paired]))
                    if paired
                    else None
                ),
                "mean_margin": (
                    float(np.mean([r["supplied_negative_margin"] for r in paired]))
                    if paired
                    else None
                ),
                "catalog_hr10_lower": (
                    float(np.mean([r["rank_upper"] <= 10 for r in paired]))
                    if paired
                    else None
                ),
                "catalog_hr10_upper": (
                    float(np.mean([r["rank_lower"] <= 10 for r in paired]))
                    if paired
                    else None
                ),
            }
            return output

        def rank_result(ids, scores, target, target_score):
            alternatives = ids != target
            a_ids, a_scores = ids[alternatives], scores[alternatives]
            greater = int(np.sum(a_scores > target_score))
            ties = int(np.sum(a_scores == target_score))
            order = np.lexsort((a_ids, -a_scores))[:10]
            n = len(ids)
            return {
                "eligible_candidates": n,
                "greater": greater,
                "ties": ties,
                "rank_lower": 1 + greater,
                "rank_upper": 1 + greater + ties,
                "rank_id_ascending": 1
                + greater
                + int(np.sum((a_scores == target_score) & (a_ids < target))),
                "normalized_rank_lower": greater / (n - 1) if n > 1 else None,
                "normalized_rank_upper": (greater + ties) / (n - 1) if n > 1 else None,
                "tenth_competitor_margin": (
                    float(target_score) - float(a_scores[order[9]])
                    if len(order) == 10
                    else None
                ),
                "top_competitor_ids": a_ids[order].tolist(),
            }

        try:
            if vocab < 1:
                raise ValueError("Catalog audit requires a positive item vocabulary")
            if any(
                p.is_floating_point() and p.dtype != torch.float32
                for p in actual.parameters()
            ):
                raise ValueError(
                    "FP32 audit requires an FP32 checkpoint; weights were not converted"
                )
            actual.eval()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            device = actual.item_emb.weight.device
            metrics["numerical_configuration"] = {
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "device": str(device),
                "hardware": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "dtype": "float32",
                "autocast": False,
                "tf32": False,
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cublas_configuration_note": "Recorded only; adapter does not configure CUDA environment after initialization.",
                "user_batch_size": 1,
                "chunk_width": width,
                "chunk_shapes": [
                    [1, min(width, vocab - start + 1)]
                    for start in range(1, vocab + 1, width)
                ],
                "shape_sensitivity": {
                    "value": None,
                    "missing_reason": "Additional GEMM shapes were not evaluated in this bounded audit.",
                },
            }
            histories = self.user_train.get("History", self.user_train)
            histories = {str(u): [int(i) for i in h] for u, h in histories.items()}
            frequencies = np.zeros(vocab + 1, dtype=np.int64)
            valid_histories = {}
            for user, history in histories.items():
                valid_ids = [i for i in history if 1 <= i <= vocab]
                np.add.at(frequencies, valid_ids, 1)
                if len(valid_ids) == len(history):
                    valid_histories[user] = history
                else:
                    counts["invalid_stored_histories"] += 1
            thresholds = np.quantile(frequencies[1:], [0.2, 0.4, 0.6, 0.8])
            bands = np.searchsorted(thresholds, frequencies, side="left")
            wanted = {}
            for _, batch in zip(range(10), train_loader):
                counts["loader_batches"] += 1
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    counts["malformed_batches"] += 1
                    continue
                seq, pos, neg = [
                    torch.as_tensor(x).detach().cpu().numpy() for x in batch[1:4]
                ]
                if (
                    seq.ndim != 2
                    or seq.shape[1] == 0
                    or pos.shape != seq.shape
                    or neg.shape != seq.shape
                ):
                    counts["malformed_batches"] += 1
                    continue
                for s, p, n in zip(seq, pos, neg):
                    counts["cached_rows"] += 1
                    if any(
                        not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all()
                        for a in (s, p, n)
                    ):
                        counts["noninteger_or_nonfinite_rows"] += 1
                        continue
                    s, p, n = s.astype(np.int64), p.astype(np.int64), n.astype(np.int64)
                    occupied = np.flatnonzero(s != 0)
                    if not len(occupied):
                        counts["empty_rows"] += 1
                        continue
                    last = int(occupied[-1])
                    if np.any((s < 0) | (s > vocab)) or not np.array_equal(
                        occupied, np.arange(occupied[0], len(s))
                    ):
                        counts["invalid_or_non_left_padded_rows"] += 1
                        continue
                    target = int(p[last])
                    if not 1 <= target <= vocab:
                        counts["invalid_aligned_targets"] += 1
                        continue
                    key = (len(s), tuple(s[occupied].tolist()), target)
                    if key in wanted:
                        counts["repeated_encoded_rows"] += 1
                        continue
                    wanted[key] = {
                        "sequence": s.copy(),
                        "last": last,
                        "target": target,
                        "negative": int(n[last]),
                    }
            # Hash complete request keys instead of comparing every request
            # sharing a target. Preserve the same strict prefix matching rule.
            widths_by_target = {}
            for input_width, encoded, target in wanted:
                widths_by_target.setdefault(target, set()).add(input_width)
            matches = {key: [] for key in wanted}
            for user, history in valid_histories.items():
                for endpoint in range(1, len(history)):
                    target = history[endpoint]
                    for input_width in sorted(widths_by_target.get(target, ())):
                        encoded = tuple(
                            history[max(0, endpoint - input_width) : endpoint]
                        )
                        key = (input_width, encoded, target)
                        candidates = matches.get(key)
                        if candidates is not None and len(candidates) < 2:
                            candidates.append((user, endpoint))
            events = {}
            for key, candidates in matches.items():
                if len(candidates) != 1:
                    counts[
                        "ambiguous_events" if candidates else "unidentified_events"
                    ] += 1
                    continue
                user, endpoint = candidates[0]
                identity = (user, endpoint)
                if identity in events:
                    counts["duplicate_event_identities"] += 1
                    continue
                event = dict(wanted[key])
                event.update(
                    user=user,
                    endpoint=endpoint,
                    history=valid_histories[user][:endpoint],
                )
                event["frequency_band"] = int(bands[event["target"]])
                event["length_band"] = length_band(len(key[1]))
                events[identity] = event
            counts["eligible_unique_event_pool"] = len(events)
            cells = {(f, l): [] for f in range(5) for l in range(4)}
            for identity, event in sorted(events.items()):
                cells[(event["frequency_band"], event["length_band"])].append(identity)
            nonempty = [cell for cell in cells if cells[cell]]
            quota = {cell: 0 for cell in cells}
            if nonempty:
                initial = cap // len(nonempty)
                for cell in nonempty:
                    quota[cell] = min(initial, len(cells[cell]))
                remaining = min(cap, len(events)) - sum(quota.values())
                while remaining:
                    for cell in nonempty:
                        if quota[cell] < len(cells[cell]) and remaining:
                            quota[cell] += 1
                            remaining -= 1
            selected = []
            for cell, identities in cells.items():
                k = quota[cell]
                allocation.append(
                    {
                        "frequency_band": cell[0],
                        "length_band": cell[1],
                        "pool_events": len(identities),
                        "selected_events": k,
                        "inclusion_probability": (
                            k / len(identities) if identities else None
                        ),
                    }
                )
                if k:
                    for index in rng.choice(len(identities), k, replace=False):
                        selected.append(events[identities[int(index)]])
            counts["selected_events"] = len(selected)

            # Synthetic reference checks do not use model outputs or fit a predictor.
            test_ids = np.arange(1, 5)
            test_scores = np.array([2.0, 2.0, 1.0, 0.0], dtype=np.float32)
            test = rank_result(test_ids, test_scores, 2, np.float32(2.0))
            assert (
                test["rank_lower"],
                test["rank_upper"],
                test["rank_id_ascending"],
            ) == (1, 2, 2)
            assert test["tenth_competitor_margin"] is None
            assert (
                rank_result(test_ids[1:], test_scores[1:], 2, np.float32(2.0))["ties"]
                == 0
            )
            # Exercise complete-history filtering and repeat-event exclusion.
            synthetic_seen = {1, 2}
            for synthetic_target in (2, 3):
                repeat_target = synthetic_target in synthetic_seen
                protocols = ["repeat_capable"] + (
                    [] if repeat_target else ["unseen_target"]
                )
                assert ("unseen_target" in protocols) == (not repeat_target)
                if not repeat_target:
                    eligible = ~np.isin(test_ids, sorted(synthetic_seen))
                    filtered = rank_result(
                        test_ids[eligible],
                        test_scores[eligible],
                        synthetic_target,
                        test_scores[synthetic_target - 1],
                    )
                    assert filtered["eligible_candidates"] == 2
                    assert filtered["rank_lower"] == 1
                    assert filtered["tenth_competitor_margin"] is None
            singleton = rank_result(
                np.array([3]),
                np.array([1.0], dtype=np.float32),
                3,
                np.float32(1.0),
            )
            assert singleton["normalized_rank_lower"] is None
            assert singleton["rank_upper"] == 1
            # Use more than ten alternatives to check tenth-score arithmetic.
            many_ids = np.arange(1, 13)
            many_scores = np.linspace(0.1, 1.2, 12, dtype=np.float32)
            many = rank_result(many_ids, many_scores, 1, many_scores[0])
            assert many["tenth_competitor_margin"] == (
                float(many_scores[0]) - float(many_scores[2])
            )
            assert len({("u", 3), ("u", 3), ("u", 4)}) == 2
            metrics["synthetic_checks_passed"] = True

            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=False
            ):
                for event_index, event in enumerate(selected):
                    sequence = event["sequence"][None, :]
                    feats = self.model.log2feats(sequence)
                    if (
                        not isinstance(feats, torch.Tensor)
                        or feats.ndim != 3
                        or tuple(feats.shape[:2]) != tuple(sequence.shape)
                    ):
                        raise ValueError(
                            "ModelView.log2feats must preserve B x L positions"
                        )
                    feature = feats[:, event["last"], :].detach()
                    if (
                        feature.dtype != torch.float32
                        or not torch.isfinite(feature).all()
                    ):
                        counts["invalid_feature_events"] += 1
                        continue
                    target, negative = event["target"], event["negative"]
                    valid_negative = 1 <= negative <= vocab and negative != target
                    if not valid_negative:
                        counts["invalid_supplied_negative_events"] += 1

                    def score_chunk(start):
                        ids = torch.arange(
                            start, min(start + width, vocab + 1), device=feature.device
                        )
                        result = self.model.predict_score(feature, ids)
                        # Singleton user batches avoid the host adapter's extra broadcasting axes.
                        if (
                            not isinstance(result, torch.Tensor)
                            or result.numel() != ids.numel()
                        ):
                            raise ValueError(
                                "ModelView catalog score shape is incompatible with singleton scoring"
                            )
                        if result.dtype != torch.float32:
                            raise ValueError("Catalog score dtype is not FP32")
                        return (
                            ids.cpu().numpy(),
                            result.reshape(-1).detach().cpu().numpy(),
                        )

                    target_score = None
                    negative_score = None
                    for start in range(1, vocab + 1, width):
                        ids, values = score_chunk(start)
                        if ids[0] <= target <= ids[-1]:
                            target_score = values[target - start]
                        if valid_negative and ids[0] <= negative <= ids[-1]:
                            negative_score = values[negative - start]
                    if target_score is None or not np.isfinite(target_score):
                        counts["nonfinite_target_events"] += 1
                        continue
                    # Subtract as Python floats to avoid FP32 overflow and to
                    # use identical arithmetic in streaming/reference margins.
                    margin = (
                        float(target_score) - float(negative_score)
                        if valid_negative
                        and negative_score is not None
                        and np.isfinite(negative_score)
                        else None
                    )
                    if valid_negative and margin is None:
                        counts["nonfinite_supplied_negative_events"] += 1
                    seen = set(event["history"])
                    seen_ids = np.asarray(sorted(seen), dtype=np.int64)
                    repeat = target in seen
                    # Preserve supplied-negative observations even when a later
                    # catalog numerical check excludes this event's ranks.
                    metrics.setdefault("original_supplied_negative_events", []).append(
                        {
                            "user": event["user"],
                            "target_endpoint": event["endpoint"],
                            "target": target,
                            "negative": negative,
                            "valid_distinct_negative": valid_negative,
                            "supplied_negative_margin": margin,
                            "repeat": repeat,
                            "repeat_capable_negative_eligible": valid_negative,
                            "unseen_target_negative_eligible": (
                                valid_negative and not repeat and negative not in seen
                            ),
                        }
                    )
                    counts["selected_repeat_events"] += int(repeat)
                    counts["unseen_protocol_excluded_repeat_events"] += int(repeat)
                    protocols = ["repeat_capable"] + (
                        [] if repeat else ["unseen_target"]
                    )
                    state = {
                        p: {
                            "n": 0,
                            "g": 0,
                            "t": 0,
                            "id_ties": 0,
                            "bad": False,
                            "top": [],
                        }
                        for p in protocols
                    }
                    materialized = []
                    replay_ok = True
                    for start in range(1, vocab + 1, width):
                        ids, values = score_chunk(start)
                        if ids[0] <= target <= ids[-1]:
                            replay_ok &= bool(values[target - start] == target_score)
                        if (
                            valid_negative
                            and ids[0] <= negative <= ids[-1]
                            and np.isfinite(negative_score)
                        ):
                            replay_ok &= bool(
                                values[negative - start] == negative_score
                            )
                        if event_index < 2:
                            materialized.append(values.copy())
                        for protocol, counter in state.items():
                            eligible = np.ones(len(ids), dtype=bool)
                            if protocol == "unseen_target":
                                eligible &= ~np.isin(ids, seen_ids)
                            counter["bad"] |= bool(
                                (~np.isfinite(values[eligible])).any()
                            )
                            counter["n"] += int(eligible.sum())
                            alternative = eligible & (ids != target)
                            a_ids, a_values = ids[alternative], values[alternative]
                            counter["g"] += int((a_values > target_score).sum())
                            counter["t"] += int((a_values == target_score).sum())
                            counter["id_ties"] += int(
                                ((a_values == target_score) & (a_ids < target)).sum()
                            )
                            finite = np.isfinite(a_values)
                            order = np.lexsort((a_ids[finite], -a_values[finite]))[:10]
                            new_top = [
                                (float(s), int(i))
                                for s, i in zip(
                                    a_values[finite][order], a_ids[finite][order]
                                )
                            ]
                            counter["top"] = sorted(
                                counter["top"] + new_top,
                                key=lambda pair: (-pair[0], pair[1]),
                            )[:10]
                    if not replay_ok:
                        counts["target_or_negative_replay_mismatch_events"] += 1
                        continue
                    for protocol, counter in state.items():
                        if counter["bad"]:
                            counts[protocol + ":nonfinite_eligible_score_events"] += 1
                            continue
                        n, g, t = counter["n"], counter["g"], counter["t"]
                        top = counter["top"]
                        row = {
                            "user": event["user"],
                            "target_endpoint": event["endpoint"],
                            "target": target,
                            "target_frequency": int(frequencies[target]),
                            "frequency_band": event["frequency_band"],
                            "length_band": event["length_band"],
                            "repeat": repeat,
                            "protocol": protocol,
                            "eligible_candidates": n,
                            "greater": g,
                            "ties": t,
                            "rank_lower": 1 + g,
                            "rank_upper": 1 + g + t,
                            "rank_id_ascending": 1 + g + counter["id_ties"],
                            "normalized_rank_lower": g / (n - 1) if n > 1 else None,
                            "normalized_rank_upper": (
                                (g + t) / (n - 1) if n > 1 else None
                            ),
                            "tenth_competitor_margin": (
                                float(target_score - top[9][0])
                                if len(top) == 10
                                else None
                            ),
                            "top_competitor_ids": [i for _, i in top],
                            "supplied_negative_margin": margin,
                            "negative_eligible": valid_negative
                            and (protocol == "repeat_capable" or negative not in seen),
                        }
                        if event_index < 2:
                            all_values = np.concatenate(materialized)
                            all_ids = np.arange(1, vocab + 1)
                            eligible = np.ones(vocab, dtype=bool)
                            if protocol == "unseen_target":
                                eligible &= ~np.isin(all_ids, seen_ids)
                            reference = rank_result(
                                all_ids[eligible],
                                all_values[eligible],
                                target,
                                target_score,
                            )
                            for name, value in reference.items():
                                assert row[name] == value, (
                                    "Streaming reference mismatch: " + name
                                )
                            counts["materialized_protocol_checks_passed"] += 1
                        records.append(row)
            if not selected:
                metrics["catalog_audit_missing_reason"] = (
                    "No uniquely identifiable eligible training events."
                )
            elif not records:
                metrics["catalog_audit_missing_reason"] = (
                    "No events passed numerical checks."
                )
            else:
                output = {}
                for protocol in ("repeat_capable", "unseen_target"):
                    rows = [r for r in records if r["protocol"] == protocol]
                    output[protocol] = {
                        "overall": summary(rows),
                        "repeat": summary([r for r in rows if r["repeat"]]),
                        "first_time": summary([r for r in rows if not r["repeat"]]),
                        "frequency_strata": {
                            str(f): summary(
                                [r for r in rows if r["frequency_band"] == f]
                            )
                            for f in range(5)
                        },
                        "length_strata": {
                            str(l): summary([r for r in rows if r["length_band"] == l])
                            for l in range(4)
                        },
                    }
                metrics["catalog_audit"] = output
            metrics["events"] = records
            metrics["sampling_cells"] = allocation
            metrics["target_frequency_quintile_boundaries"] = thresholds.tolist()
            metrics["catalog_frequency_bin_sizes"] = {
                str(f): int((bands[1:] == f).sum()) for f in range(5)
            }
            metrics["length_band_labels"] = ["1-5", "6-15", "16-31", "32+"]
            metrics["synthetic_checks_passed"] = True
        except RuntimeError as error:
            # Only a specifically identified deterministic-operation failure is
            # an unavailable audit. Propagate OOM, shape and other runtime bugs.
            message = str(error)
            deterministic_failure = "deterministic" in message.lower() and (
                "implementation" in message.lower()
                or "CUBLAS_WORKSPACE_CONFIG" in message
            )
            if not deterministic_failure:
                raise
            import warnings

            warnings.warn(
                "Catalog audit unavailable under deterministic FP32 settings: "
                + message,
                RuntimeWarning,
            )
            metrics["catalog_audit"] = None
            metrics["events"] = []
            metrics["catalog_audit_missing_reason"] = message
            metrics["partial_catalog_results_discarded"] = True
        finally:
            for module, mode in modes:
                module.training = mode
            torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
            torch.backends.cudnn.benchmark = previous["benchmark"]
            torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
            torch.use_deterministic_algorithms(
                previous["deterministic"], warn_only=previous["warn_only"]
            )
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            metrics["counts"] = dict(counts)
            metrics["elapsed_diagnostic_seconds"] = time.perf_counter() - started
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
