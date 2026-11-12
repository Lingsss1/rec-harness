import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Local diagnostic persistence and measured runtime
import random
import os
import time
from collections import Counter, defaultdict

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
    ### >>> Self_EvolveRec-BLOCK-START: Evidence-limited temporal interpretation
    - Every statement must be grounded in the provided metric values/definitions.
    - Cite controls, counts, paired effects and intervals when interpreting temporal reliance.
    - If controls fail, withhold temporal interpretation. Missing measurements are unavailable.
    - Distinguish earlier-order sensitivity, recent-context dependence and final-item dependence.
    - Training-prefix outcomes are in-sample, not held-out evaluation or evidence of improvement.
    - Intervals condition on fitted checkpoints and sampled diagnostic histories; they omit training randomness, search selection and in-sample bias.
    - Never infer causal intent, ratings, review content, item categories, prices, stores, compatibility, ownership, mood or simulator traits.
    - Diagnostic candidates differ from the host evaluator; do not claim benchmark equivalence.
    - Include at most two evidence-linked, testable single-model hypotheses in key_implications; these are hypotheses, not demonstrated improvements or action plans.
    - Any behavioral hypothesis must be testable within one jointly trained GRU4Rec model per category; do not suggest independent predictor fusion.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the ModelView contract without unsupported metadata
    def __init__(self, view, device):
        self.model = view
        # DEBUG: Allocate diagnostic tensors on the actual trained model's device.
        # The caller's device may be unspecified CUDA or differ from the module.
        self.device = view.model.item_emb.weight.device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this task, including in routed datasets.
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
            ### >>> Self_EvolveRec-BLOCK-START: Represent insufficient embedding observations as unavailable
            if n <= 2:
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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained module
        self.model.model.eval()
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

            # DEBUG: Run the trained architecture directly. A host ModelView feature
            # wrapper may implement baseline transformations rather than this model's
            # log2feats method. Normalize inputs on the actual embedding device.
            actual_model = self.model.model
            seq_t = torch.as_tensor(
                seq, dtype=torch.long, device=actual_model.item_emb.weight.device
            )
            log_feats = actual_model.log2feats(seq_t)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            # DEBUG: Score through the trained GRU's architecture-aware method.
            # Updating GRU4RecModel does not update a host ModelView scoring wrapper.
            # Aligned scoring also avoids allocating B x L x 2 x H embeddings.
            scorer = self.model.model.predict_score
            pos_logits = scorer(log_feats, pos_t)
            neg_logits = scorer(log_feats, neg_t)
            margin = pos_logits - neg_logits

            ### >>> Self_EvolveRec-BLOCK-START: Measure margins only at valid observed and sampled positions
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Reuse the complete valid-position mask
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
            ### >>> Self_EvolveRec-BLOCK-START: Do not fabricate margins for absent observations
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

        ### >>> Self_EvolveRec-BLOCK-START: Expose measured counts without unsupported category breakdowns
        return margin_metrics, {
            "valid_positions": sum(int(m.numel()) for m in margins_all),
            "eps": float(eps),
            "item_category_breakdown": None,
            "reason": "Item categories are unavailable in the shared task.",
        }
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Controlled temporal diagnostics with validated scoring and complete sampling records
    def _histories(self):
        source = self.user_train.get("History", self.user_train)
        return {
            str(user): [int(item) for item in items]
            for user, items in source.items()
            if isinstance(items, (list, tuple, np.ndarray))
        }

    def _category(self, histories):
        categories = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        # Identify routing using public training histories only. Never inspect other trials.
        matches = []
        for category in categories:
            path = os.path.join(os.environ["GAGC_DATA_DIR"], category + "_train.txt")
            if not os.path.isfile(path):
                continue
            observed = defaultdict(list)
            with open(path) as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) == 2:
                        observed[str(int(fields[0]))].append(int(fields[1]))
            if dict(observed) == histories:
                matches.append(category)
        return matches[0] if len(matches) == 1 else None

    def _readout(self, history, candidates, width):
        if not history:
            raise ValueError("Empty diagnostic histories must be skipped")
        if width < len(history):
            raise ValueError("Diagnostic width must accommodate the entire input")
        sequence = np.zeros((1, width), dtype=np.int64)
        sequence[0, -len(history) :] = history
        # DEBUG: Measure the same feature computation used in training and prediction,
        # without depending on a potentially baseline-specific ModelView wrapper.
        actual_model = self.model.model
        sequence_t = torch.as_tensor(
            sequence, dtype=torch.long, device=actual_model.item_emb.weight.device
        )
        features = actual_model.log2feats(sequence_t)
        if features.ndim != 3 or features.shape[:2] != sequence_t.shape:
            raise RuntimeError("Trained model.log2feats must preserve B x L alignment")
        seq_t = torch.as_tensor(sequence, device=features.device)
        positions = torch.arange(width, device=features.device).expand_as(seq_t)
        last = positions.masked_fill(seq_t.eq(0), -1).max(1).values
        final = features[torch.arange(1, device=features.device), last]
        items = torch.as_tensor(candidates, dtype=torch.long, device=features.device)
        # DEBUG: Use the trained model's shared-candidate scoring contract directly.
        # Keeping the sequence axis prevents a singleton candidate list from being
        # mistaken for one aligned item per batch row.
        logits = actual_model.predict_score(final[:, None, :], items)
        if tuple(logits.shape) != (1, 1, len(candidates)):
            raise RuntimeError(
                "Trained model.predict_score must return B x L x C logits"
            )
        return final.detach(), logits[0, 0].detach()

    @staticmethod
    def _outcome(scores, candidates, target):
        values = scores.double().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite diagnostic candidate scores")
        order = np.lexsort((np.asarray(candidates), -values))
        ordered = np.asarray(candidates)[order]
        rank = int(np.flatnonzero(ordered == target)[0]) + 1
        return {
            "rank": float(rank),
            "reciprocal_rank": 1.0 / rank,
            "hr10": float(rank <= 10),
            "top10": set(int(x) for x in ordered[:10]),
        }

    def _temporal(self):
        rng = np.random.default_rng(42)
        histories = self._histories()
        itemnum = int(self.model.model.itemnum)
        width = int(
            getattr(self.model, "maxlen", os.environ.get("GRU4REC_MAXLEN", "128"))
        )
        if itemnum <= 0 or width < 3:
            raise ValueError("Temporal diagnostics require real items and maxlen >= 3")
        if any(
            item < 1 or item > itemnum
            for history in histories.values()
            for item in history
        ):
            raise ValueError("Training histories must contain only real item IDs")
        category = self._category(histories)
        counts = Counter(item for items in histories.values() for item in items)
        popularity = np.asarray([counts[item] for item in range(1, itemnum + 1)])
        boundaries = np.quantile(popularity, [1.0 / 3.0, 2.0 / 3.0]).tolist()
        specification = {
            "seed": 42,
            "maximum_prefixes_per_user": 4,
            "shuffle_replicates": 5,
            "negative_count": 99,
            "bootstrap_replicates": 1000,
            "atol": 1e-6,
            "rtol": 1e-5,
            "width": width,
            "length_strata": ["3-5", "6-10", "11-20", "21+"],
            "popularity_tertile_boundaries": boundaries,
            "popularity_boundary_rule": "low <= first; middle <= second; otherwise high",
            "candidate_policy": "target plus 99 uniform unique real items excluding original prefix and target; ascending item-ID order and tie break",
            "endpoint_policy": "up to four uniformly sampled endpoints per user; endpoint is the target index in the full training history",
            "context_policy": "last maxlen observed items before each endpoint; perturbations and length strata refer to this model-visible context",
            "rng_policy": "local NumPy seed 42; diagnostic execution preserves caller Python, NumPy and active-device Torch RNG states",
            "host_eligibility_seen_exclusions_repeated_target_ties": "unspecified here; host unchanged",
            "scope": "in-sample training prefixes; no benchmark equivalence or improvement claimed",
        }
        selected = []
        eligible_users = eligible_prefixes = 0
        for user in sorted(histories):
            items = histories[user]
            endpoints = np.arange(3, len(items))
            if not len(endpoints):
                continue
            eligible_users += 1
            eligible_prefixes += len(endpoints)
            for endpoint in sorted(
                rng.choice(
                    endpoints, size=min(4, len(endpoints)), replace=False
                ).tolist()
            ):
                selected.append(
                    (
                        user,
                        endpoint,
                        items[max(0, endpoint - width) : endpoint],
                        items[endpoint],
                    )
                )
        controls = {
            "passed": True,
            "tested_prefixes": 0,
            "maximum_feature_error": 0.0,
            "maximum_logit_error": 0.0,
            "comparisons": ["unchanged_repeat", "extra_left_padding", "unpadded"],
            "atol": 1e-6,
            "rtol": 1e-5,
        }
        records = []
        identifiers = [
            {
                "user": user,
                "target_index": endpoint,
                "context_start_index": endpoint - len(history),
                "observed_length": len(history),
                "target_item": target,
                "status": "not_scored",
            }
            for user, endpoint, history, target in selected
        ]
        skipped = 0
        unchanged = permutations = 0
        names = (
            "older_delete",
            "recent_delete_keep_final",
            "final_delete",
            "prefix_shuffles",
        )
        for (user, endpoint, history, target), identifier in zip(selected, identifiers):
            excluded = set(history) | {target}
            available_count = itemnum - len(excluded)
            if available_count < 99:
                identifier["status"] = "insufficient_candidates"
                identifier["skip"] = "fewer than 99 eligible unique negatives"
                skipped += 1
                continue
            if available_count < 198 or available_count < itemnum // 4:
                # Avoid slow rejection sampling when most catalog items are excluded.
                pool = np.setdiff1d(
                    np.arange(1, itemnum + 1, dtype=np.int64),
                    np.fromiter(excluded, dtype=np.int64),
                    assume_unique=True,
                )
                negatives = rng.choice(pool, size=99, replace=False).tolist()
            else:
                negatives = set()
                while len(negatives) < 99:
                    item = int(rng.integers(1, itemnum + 1))
                    if item not in excluded:
                        negatives.add(item)
            candidates = sorted([target] + list(negatives))
            identifier["candidates"] = candidates
            baseline_feature, baseline_scores = self._readout(
                history, candidates, width
            )
            controls["tested_prefixes"] += 1
            for alternative_width in (width, width + 7, len(history)):
                feature, scores = self._readout(history, candidates, alternative_width)
                for actual, expected, key in (
                    (feature, baseline_feature, "maximum_feature_error"),
                    (scores, baseline_scores, "maximum_logit_error"),
                ):
                    finite = bool(
                        torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    )
                    error = (
                        float((actual - expected).abs().max().item())
                        if finite
                        else None
                    )
                    if not finite:
                        controls[key] = None
                    elif controls[key] is not None:
                        controls[key] = max(controls[key], error)
                    if not finite or not torch.allclose(
                        actual, expected, atol=1e-6, rtol=1e-5
                    ):
                        controls["passed"] = False
                        controls.setdefault("failures", []).append(
                            {
                                "user": user,
                                "target_index": endpoint,
                                "alternative_width": alternative_width,
                                "quantity": key,
                                "maximum_absolute_error": error,
                                "finite": finite,
                            }
                        )
            if not controls["passed"]:
                identifier["status"] = "control_failed"
                break
            original = self._outcome(baseline_scores, candidates, target)
            n = len(history)
            k = max(1, (n - 1) // 4)
            variants = {
                "older_delete": [history[k:]],
                "recent_delete_keep_final": [history[: -k - 1] + history[-1:]],
                "final_delete": [history[:-1]],
                "prefix_shuffles": [],
            }
            identifier["shuffle_indices"] = []
            for _ in range(5):
                indices = rng.permutation(n - 1).tolist()
                shuffled = [history[i] for i in indices] + history[-1:]
                identifier["shuffle_indices"].append(indices)
                unchanged += int(shuffled == history)
                permutations += 1
                variants["prefix_shuffles"].append(shuffled)
            length_group = (
                "3-5"
                if n <= 5
                else "6-10" if n <= 10 else "11-20" if n <= 20 else "21+"
            )
            target_count = counts[target]
            popularity_group = (
                "low"
                if target_count <= boundaries[0]
                else "middle" if target_count <= boundaries[1] else "high"
            )
            row = {
                "user": user,
                "length": length_group,
                "popularity": popularity_group,
                "original": {
                    key: original[key] for key in ("rank", "reciprocal_rank", "hr10")
                },
                "effects": {},
            }
            for name in names:
                repetitions = []
                for edited in variants[name]:
                    _, scores = self._readout(edited, candidates, width)
                    result = self._outcome(scores, candidates, target)
                    union = original["top10"] | result["top10"]
                    repetitions.append(
                        {
                            "rank_change": result["rank"] - original["rank"],
                            "reciprocal_rank_change": result["reciprocal_rank"]
                            - original["reciprocal_rank"],
                            "hr10_change": result["hr10"] - original["hr10"],
                            "top10_jaccard": len(original["top10"] & result["top10"])
                            / len(union),
                            "rank": result["rank"],
                            "reciprocal_rank": result["reciprocal_rank"],
                            "hr10": result["hr10"],
                        }
                    )
                row["effects"][name] = {
                    key: float(np.mean([entry[key] for entry in repetitions]))
                    for key in repetitions[0]
                }
            records.append(row)
            identifier["status"] = "scored"
        attempted_scored_prefixes = len(records)
        if not controls["tested_prefixes"]:
            controls["passed"] = None
        if controls["passed"] is not True:
            records = []
        result = {
            "category": category,
            "specification": specification,
            "controls": controls,
            "counts": {
                "eligible_users": eligible_users,
                "eligible_prefixes": eligible_prefixes,
                "sampled_users": len({entry[0] for entry in selected}),
                "sampled_prefixes": len(selected),
                "scored_prefixes": len(records),
                "scored_users": len({row["user"] for row in records}),
                "completed_prefixes_before_control_gating": attempted_scored_prefixes,
                "not_scored_prefixes": sum(
                    entry["status"] == "not_scored" for entry in identifiers
                ),
                "insufficient_candidate_skips": skipped,
                "shuffle_replicates": permutations,
                "unchanged_shuffle_replicates": unchanged,
                "unchanged_shuffle_fraction": (
                    unchanged / permutations if permutations else None
                ),
            },
            "summary": _temporal_summary(records) if records else None,
            "strata": (
                {
                    axis: {
                        label: _temporal_summary(
                            [r for r in records if r[axis] == label]
                        )
                        for label in labels
                    }
                    for axis, labels in (
                        ("length", specification["length_strata"]),
                        ("popularity", ["low", "middle", "high"]),
                    )
                }
                if records
                else None
            ),
            "records": records,
            "sampled_identifiers": identifiers,
        }
        return result

    def run_full_diagnosis(self, train_loader):
        actual_model = self.model.model
        previous_mode = actual_model.training
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = (
            [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if self.device.type == "cuda"
            else []
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(42)
                actual_model.eval()
                collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
                margins, counts = self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=0.5
                )
                temporal = self._temporal()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                result = {
                    "metrics": {
                        "embedding_collapse_score": collapse,
                        "pos_neg_margin_mean": margins["mean"],
                        "pos_neg_margin_p10": margins["p10"],
                        "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                        "pos_neg_small_margin_rate": margins["small_margin_rate"],
                        "margin_observations": counts,
                        "temporal": temporal,
                        "diagnostic_elapsed_seconds": time.perf_counter() - started,
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "(mean off-diagonal item-embedding cosine + 1)/2; geometric alignment only, not recommendation diversity or proof of collapse. Unavailable with insufficient items.",
                        "pos_neg_margin_mean": "Mean positive-minus-negative logit at positions with nonzero sequence, positive and negative IDs; score-scale dependent and in-sample.",
                        "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of valid-position margins; score-scale dependent.",
                        "pos_neg_neg_beats_pos_rate": "Valid-position fraction with negative margin; sampled-negative training diagnostic.",
                        "pos_neg_small_margin_rate": "Valid-position fraction with margin below 0.5; threshold depends on score scale.",
                        "temporal": "Training-prefix outcomes with fixed paired candidates. Rank is one-based; positive rank change is deterioration. Negative HR or reciprocal-rank change is deterioration. Jaccard compares original and edited top ten. Five shuffle outcomes are averaged before prefixes within users, then users equally within category. Older deletion tests early-context removal; recent deletion retains the final item; final deletion is a separate recency stress test; prefix shuffling tests earlier-order sensitivity. Effects are descriptive and in-sample.",
                        "controls": "Eval-mode repeats, extra left padding and unpadded execution must agree in final features and logits at atol=1e-6, rtol=1e-5. A failed or absent control withholds temporal effects.",
                        "intervals": "1000 paired user-cluster bootstrap replicates, seed 42; each selected user's matched prefixes and averaged shuffles stay together. Percentile 95% intervals condition on fitted checkpoints and sampled diagnostic histories, excluding training randomness, search selection and in-sample bias.",
                        "macro": "Equal average of exactly four category effects, recomputed after independent within-category user resampling in every replicate. Unavailable until all four matching diagnostic specifications are present in this trial.",
                        "unavailable": "Missing observations are null. Item categories, ratings, review content, prices, stores and simulator traits are unobserved. Dataset routing does not supply item categories. Host evaluator semantics beyond the supplied benchmark settings are unspecified and unchanged.",
                        "diagnostic_elapsed_seconds": "Measured synchronized diagnostic wall time, including GPU probes; must be charged by the host to the shared search budget.",
                    },
                }
                directory = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
                )
                os.makedirs(directory, exist_ok=True)
                category = temporal["category"]
                result["metrics"]["macro_temporal"] = None
                if category is None:
                    result["metrics"][
                        "category_identification"
                    ] = "Unavailable: training histories did not uniquely match a public category split."
                    destination = os.path.join(
                        directory, f"unidentified_{time.time_ns()}.json"
                    )
                else:
                    destination = os.path.join(directory, category + ".json")
                temporary = destination + ".tmp"
                with open(temporary, "w") as handle:
                    json.dump(result, handle, allow_nan=False)
                os.replace(temporary, destination)
                if category is not None:
                    result["metrics"]["macro_temporal"] = _four_category_macro(
                        directory
                    )
                result["metrics"]["diagnostic_elapsed_seconds"] = (
                    time.perf_counter() - started
                )
                with open(temporary, "w") as handle:
                    json.dump(result, handle, allow_nan=False)
                os.replace(temporary, destination)
                return result
        finally:
            actual_model.train(previous_mode)
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Paired user-cluster summaries and explicit four-category aggregation


def _user_vectors(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["user"]].append(row)
    if not grouped:
        return [], np.empty((0, 0), dtype=np.float64)
    first = records[0]["effects"]
    keys = [(name, metric) for name in first for metric in first[name]]
    vectors = []
    for user in sorted(grouped):
        rows = grouped[user]
        vectors.append(
            [
                np.mean([row["effects"][name][metric] for row in rows])
                for name, metric in keys
            ]
        )
    return keys, np.asarray(vectors, dtype=np.float64)


def _pack_effects(keys, means, lower, upper):
    packed = {}
    for index, (name, metric) in enumerate(keys):
        packed.setdefault(name, {})[metric] = {
            "mean": float(means[index]),
            "percentile95": [float(lower[index]), float(upper[index])],
        }
    return packed


def _temporal_summary(records):
    keys, vectors = _user_vectors(records)
    if not len(vectors):
        return {"users": 0, "prefixes": 0, "effects": None, "original": None}
    rng = np.random.default_rng(42)
    bootstrap = np.asarray(
        [
            vectors[rng.integers(0, len(vectors), size=len(vectors))].mean(0)
            for _ in range(1000)
        ]
    )
    lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    users = defaultdict(list)
    for row in records:
        users[row["user"]].append(row)
    original = {
        metric: float(
            np.mean(
                [
                    np.mean([row["original"][metric] for row in rows])
                    for rows in users.values()
                ]
            )
        )
        for metric in ("rank", "reciprocal_rank", "hr10")
    }
    return {
        "users": len(vectors),
        "prefixes": len(records),
        "original": original,
        "effects": _pack_effects(keys, vectors.mean(0), lower, upper),
    }


def _four_category_macro(directory):
    categories = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )
    loaded = []
    for category in categories:
        path = os.path.join(directory, category + ".json")
        if not os.path.isfile(path):
            return None
        with open(path) as handle:
            temporal = json.load(handle)["metrics"]["temporal"]
        if (
            temporal["category"] != category
            or temporal["controls"]["passed"] is not True
            or not temporal["records"]
        ):
            return None
        loaded.append(temporal)
    # Popularity boundaries are deliberately category specific; other settings must match.
    specifications = [
        {
            key: value
            for key, value in entry["specification"].items()
            if key != "popularity_tertile_boundaries"
        }
        for entry in loaded
    ]
    if any(spec != specifications[0] for spec in specifications[1:]):
        return None
    extracted = [_user_vectors(entry["records"]) for entry in loaded]
    keys = extracted[0][0]
    if any(entry[0] != keys for entry in extracted):
        return None
    arrays = [entry[1] for entry in extracted]
    rng = np.random.default_rng(42)
    bootstrap = []
    for _ in range(1000):
        category_means = [
            values[rng.integers(0, len(values), size=len(values))].mean(0)
            for values in arrays
        ]
        bootstrap.append(np.mean(category_means, axis=0))
    lower, upper = np.percentile(np.asarray(bootstrap), [2.5, 97.5], axis=0)
    result = {
        "categories": {
            category: entry["summary"] for category, entry in zip(categories, loaded)
        },
        "equal_four_category_macro": _pack_effects(
            keys, np.mean([values.mean(0) for values in arrays], axis=0), lower, upper
        ),
        "scope": "In-sample diagnostic effects only; not host validation metrics or evidence of improvement.",
    }
    with open(os.path.join(directory, "four_category_macro.json"), "w") as handle:
        json.dump(result, handle, allow_nan=False)
    return result


### <<< Self_EvolveRec-BLOCK-END
