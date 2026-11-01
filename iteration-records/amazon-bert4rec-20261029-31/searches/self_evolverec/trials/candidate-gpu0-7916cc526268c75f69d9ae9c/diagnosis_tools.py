import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and persist trial-local diagnostic provenance
from collections import Counter
from itertools import islice
from pathlib import Path

### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Ground interpretation in diagnostic provenance and availability
def diagnosis_interpreter_prompt(raw_diagnosis):
    return _diagnosis_interpreter_prompt(raw_diagnosis) + """
    Additional interpretation requirements:
    Distinguish supervised training separation, descriptive in-sample prefix
    dependence, and held-out predictive usefulness. Prefix diagnostics are never
    host validation performance. Events in these prefixes may have contributed
    to fitting. Interventions are not causal user counterfactuals.
    Cite measured values, counts, availability and definitions for every finding.
    Missing measurements and provenance are unknown, not evidence of health.
    Embedding cosine similarity does not establish recommendation diversity.
    Input-target item equality does not establish leakage: repeated events and
    legitimate corruption policies can share item IDs.
    Interpret frequency-matched substitution as training-exposed content
    disruption and block swaps as arrangement sensitivity. Neither identifies
    causal recency, utility, compatibility or user preferences. Frequency bins
    do not ensure contextual plausibility or exchangeability. Compare substitution
    and swap effects on their common matchable-user subset; all-eligible swap
    results are separately descriptive. Inspect excluded-user characteristics,
    empty pools, candidate counts, donor reuse, balance and fixture failures.
    User bootstrap intervals condition on the checkpoint, selected users,
    candidates and realized donors. Donor Monte Carlo errors quantify a separate
    source of variation and must not be added mechanically to these intervals.
    Ordinary 95% intervals are exploratory. The separately reported Bonferroni
    percentile intervals use a predeclared family of 48 and approximate coverage;
    20000 bootstrap draws give limited resolution at their extreme tails.
    A live identity check does not establish fresh-process predictor parity.
    Interpret bootstrap intervals and strata as exploratory. Account for no-ops,
    effective edits, small samples and unrealistic order-corruption interventions.
    Timestamp inversions describe chronological disorder, not proven violations
    of item release availability, which is unobserved.
    Do not invent ratings, reviews, categories, ownership or simulator traits.
    Validation usefulness requires explicitly authorized pinned diagnostic inputs;
    do not infer it from this loader or request hidden files.
    Selection remains host macro validation HR@10; acknowledge validation adaptation
    and the shared compute budget. Do not recommend independent predictor fusion.
    If required evidence is unavailable, use NEEDS_IMPROVEMENT and explain the
    measurement limitation without claiming the fitted model itself has failed.
    These requirements take precedence over any default suggestion of STABLE.
    """


def _diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and use the actual BERT4Rec module
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are not available in this matched task.
        return []

    def _features(self, sequence):
        # The supplied BERT4Rec path accepts tensors; retain the view fallback.
        method = getattr(self.network, "log2feats", None)
        if callable(method):
            return method(sequence)
        return self.model.log2feats(sequence.detach().cpu().numpy())

    def _scores(self, features, items):
        method = getattr(self.network, "predict_score", None)
        if callable(method):
            return method(features, items)
        return self.model.predict_score(features, items)

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable pairwise similarity instead of fabricated zero collapse
            n = emb.size(0)
            if min(int(sample_size), n) < 2:
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
        ### >>> Self_EvolveRec-BLOCK-START: Audit selected masked supervision without inferring event provenance
        margins_all = []
        audit = {
            "selected_positions": 0,
            "selected_with_reserved_mask": 0,
            "selected_without_reserved_mask": 0,
            "invalid_target_or_negative": 0,
            "original_labels_and_corruption_provenance": "unavailable",
            "leakage_conclusion": "unknown; item-ID equality is not a leakage test",
        }
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume exactly the requested number of diagnostic batches
        for batch in islice(dataloader, max(0, int(num_batches))):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError("Expected (None, sequence, positive, negative)")

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score aligned BERT4Rec masked targets through compatible interfaces
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or seq_t.shape != pos_t.shape
                or seq_t.shape != neg_t.shape
            ):
                raise ValueError(
                    "diagnostic batches must contain aligned B x L tensors"
                )
            itemnum = int(self.network.itemnum)
            if bool(((seq_t < 0) | (seq_t > self.network.mask_token)).any()):
                raise ValueError(
                    "Diagnostic sequence contains a noncatalog/nonmask token"
                )
            selected = pos_t != 0
            real_labels = (pos_t >= 1) & (pos_t <= itemnum)
            real_negatives = (neg_t >= 1) & (neg_t <= itemnum)
            masked = seq_t == int(self.network.mask_token)
            audit["selected_positions"] += int(selected.sum().item())
            audit["selected_with_reserved_mask"] += int(
                (selected & masked).sum().item()
            )
            audit["selected_without_reserved_mask"] += int(
                (selected & ~masked).sum().item()
            )
            audit["invalid_target_or_negative"] += int(
                (selected & ~(real_labels & real_negatives)).sum().item()
            )
            valid = selected & masked & real_labels & real_negatives
            log_feats = self._features(seq_t)
            # Separate aligned calls avoid B x L x 2 broadcasting ambiguity.
            safe_pos = pos_t.masked_fill(~valid, 0)
            safe_neg = neg_t.masked_fill(~valid, 0)
            margin = self._scores(log_feats, safe_pos) - self._scores(
                log_feats, safe_neg
            )
            if not torch.isfinite(margin[valid]).all():
                raise RuntimeError("nonfinite supervised diagnostic scores")
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Restrict breakdowns to audited masked positions
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
                ### >>> Self_EvolveRec-BLOCK-START: Represent missing supervised measurements explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Report supervision and unavailable-category audits
        breakdown = {
            "supervision_audit": audit,
            "category_metadata_available": False,
            "measured_positions": sum(int(m.numel()) for m in margins_all),
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

    ### >>> Self_EvolveRec-BLOCK-START: Audit matched substitutions, preserve failed-draw provenance and avoid artifact collisions
    def _encode_events(self, events, length):
        mask = int(self.network.mask_token)
        if len(events) > length - 1:
            raise ValueError("freeze and truncate the prefix before intervention")
        sequence = np.zeros(length, dtype=np.int64)
        if events:
            sequence[length - len(events) - 1 : length - 1] = [
                item for _, item in events
            ]
        sequence[-1] = mask
        assert np.count_nonzero(sequence == mask) == 1
        assert np.count_nonzero(sequence) == len(events) + 1
        return sequence

    @staticmethod
    def _score_summary(scores, candidates):
        # Stable sorting uses the frozen candidate-array order to break ties.
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        return {
            "target_score": float(scores[0]),
            "rank": float(rank),
            "hr10": float(rank <= 10),
            "margin": (
                float(scores[0] - scores[1:].max()) if len(scores) > 1 else None
            ),
            "zero_variance": bool(np.std(scores) <= 1e-12),
            "top10": set(int(i) for i in candidates[order[:10]]),
        }

    @staticmethod
    def _time_values(values):
        if values is None:
            return None
        try:
            result = np.asarray([float(v) for v in values], dtype=np.float64)
        except (TypeError, ValueError):
            try:
                dates = np.asarray(values, dtype="datetime64[ms]")
                if np.isnat(dates).any():
                    return None
                result = dates.astype(np.int64).astype(np.float64)
            except (TypeError, ValueError, OverflowError):
                return None
        return result if np.isfinite(result).all() else None

    def _prefix_interventions(self):
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise ValueError("ModelView.user_train must supply a history mapping")
        itemnum, length = int(self.network.itemnum), int(self.network.maxlen)
        if itemnum < 1 or length < 3:
            return {"available": False, "reason": "requires a catalog and L >= 3"}
        category = str(
            getattr(self.model, "dataset_name", None)
            or getattr(self.network, "dataset_name", None)
            or "current_category_unlabeled"
        )
        catalog = list(range(1, itemnum + 1))
        frequency = Counter()
        eligible_users, failures = [], []
        counts = Counter(
            dict.fromkeys(
                (
                    "invalid_item_ids",
                    "short_history",
                    "attempted_users",
                    "fewer_than_99_negatives",
                    "empty_pool_users",
                    "matched_users",
                    "identity_checks",
                    "identity_failures",
                    "attempted_substitution_replicates",
                    "successful_substitution_replicates",
                ),
                0,
            )
        )
        counts["training_users"] = len(histories)
        for user in sorted(histories, key=lambda u: (str(type(u)), str(u))):
            try:
                history = [int(i) for i in histories[user]]
                if any(i < 1 or i > itemnum for i in history):
                    raise ValueError("noncatalog ID")
            except (TypeError, ValueError, OverflowError):
                counts["invalid_item_ids"] += 1
                failures.append({"user": str(user), "reason": "invalid item IDs"})
                continue
            frequency.update(history)
            if len(history) < 3:
                counts["short_history"] += 1
            else:
                eligible_users.append((user, history))
        counts["eligible_users"] = len(eligible_users)
        selected = random.Random(42).sample(
            eligible_users, min(200, len(eligible_users))
        )
        counts["sampled_users"] = len(selected)
        # Integer bit lengths implement the declared bins exactly at boundaries.
        bins = {i: int(frequency[i]).bit_length() for i in catalog}
        bin_items = {}
        for i in catalog:
            bin_items.setdefault(bins[i], []).append(i)
        records, baseline_records, assignments, support, all_swaps = [], [], [], [], []
        realized_assignments = []
        identity_errors = []
        bootstrap_rng = np.random.default_rng(42)
        outcome_keys = ("target_score", "margin", "rank", "hr10")

        def score_arrays(arrays, candidates):
            sequence = torch.as_tensor(arrays, dtype=torch.long, device=self.device)
            features = self._features(sequence)
            if features.ndim != 3 or features.shape[:2] != sequence.shape:
                raise RuntimeError("log2feats must return B x L x H")
            positions = torch.arange(length, device=self.device).expand_as(sequence)
            last = positions.masked_fill(sequence == 0, -1).amax(dim=1)
            if not bool(
                (
                    sequence.gather(1, last[:, None]) == int(self.network.mask_token)
                ).all()
            ):
                raise RuntimeError("readout must select the appended MASK")
            readout = features[torch.arange(len(arrays), device=self.device), last]
            items = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
            # The shared-candidate contract returns B x 1 x K here.
            scores = self._scores(readout[:, None, :], items)
            if scores.ndim == 3 and scores.shape[1] == 1:
                scores = scores[:, 0, :]
            # Accumulate differences in float64 so finite float32 logits do not
            # overflow during margin or intervention-effect subtraction.
            values = scores.detach().to(dtype=torch.float64).cpu().numpy()
            if values.shape != (len(arrays), len(candidates)):
                raise RuntimeError("incompatible candidate score shape")
            return values

        for user, history in selected:
            counts["attempted_users"] += 1
            visible = list(enumerate(history[:-1][-(length - 1) :]))
            n, target = len(visible), history[-1]
            k = min(3, n // 2)
            seed_tag = f"42:{type(user).__name__}:{user}"
            candidate_rng = random.Random(seed_tag + ":candidates")
            edit_rng = random.Random(seed_tag + ":donors")
            history_items = set(history)
            unseen = [i for i in catalog if i not in history_items]
            candidates = np.asarray(
                [target] + candidate_rng.sample(unseen, min(99, len(unseen))),
                dtype=np.int64,
            )
            counts["fewer_than_99_negatives"] += int(len(unseen) < 99)
            excluded = history_items | set(candidates.tolist())
            blocks = {"older": list(range(k)), "recent": list(range(n - k, n))}
            pools = {
                p: [i for i in bin_items[bins[visible[p][1]]] if i not in excluded]
                for block in blocks.values()
                for p in block
            }
            matched = all(pools.values())
            info = {
                "user": str(user),
                "seed_tag": seed_tag,
                "history_length": len(history),
                "retained_length": n,
                "target_frequency": frequency[target],
                "repeat": target in [i for _, i in visible],
                "candidate_count": len(candidates),
                "candidates": candidates.tolist(),
                "matchable": bool(matched),
                "scored": False,
                "pool_sizes": {str(p): len(pool) for p, pool in pools.items()},
            }
            support.append(info)
            counts["attempted_substitution_replicates"] += 10
            if not matched:
                counts["empty_pool_users"] += 1
                failures.append(
                    {
                        "user": str(user),
                        "reason": "empty same-bin donor pool",
                        "positions": [p for p in pools if not pools[p]],
                    }
                )
            swapped = visible[-k:] + visible[k:-k] + visible[:k]
            assert Counter(i for _, i in swapped) == Counter(i for _, i in visible)
            branches = [
                ("baseline", visible),
                ("identity", list(visible)),
                ("swap-baseline", swapped),
            ]
            pending_assignments = []
            if matched:
                for replicate in range(10):
                    for name, block in blocks.items():
                        edited = list(visible)
                        for p in block:
                            donor = edit_rng.choice(pools[p])
                            original = visible[p][1]
                            assert (
                                donor not in excluded and bins[donor] == bins[original]
                            )
                            edited[p] = (visible[p][0], donor)
                            pending_assignments.append(
                                {
                                    "user": str(user),
                                    "replicate": replicate,
                                    "block": name,
                                    "position": p,
                                    "absolute_position": length - n - 1 + p,
                                    "original": original,
                                    "donor": donor,
                                    "pool_size": len(pools[p]),
                                    "seed_tag": seed_tag,
                                }
                            )
                        branches.append((name + "-baseline", edited))
            # Preserve every realized draw, including fixtures later rejected.
            realized_assignments.extend(pending_assignments)
            arrays = np.stack([self._encode_events(e, length) for _, e in branches])
            assert np.all((arrays == 0) == (arrays[0] == 0))
            assert np.all(arrays[:, -1] == int(self.network.mask_token))
            logits = score_arrays(arrays, candidates)
            counts["identity_checks"] += 1
            finite = bool(np.isfinite(logits).all())
            error = float(np.max(np.abs(logits[0] - logits[1]))) if finite else None
            identity_errors.append(error)
            if not finite or not np.allclose(
                logits[0], logits[1], rtol=1.3e-6, atol=1e-5
            ):
                counts["identity_failures"] += 1
                failures.append(
                    {
                        "user": str(user),
                        "reason": "nonfinite or identity failure",
                        "max_abs_error": error,
                    }
                )
                continue
            info["scored"] = True
            base = self._score_summary(logits[0], candidates)
            user_record = {}
            for b, (name, events) in enumerate(branches[2:], start=2):
                summary = self._score_summary(logits[b], candidates)
                denominator = min(len(base["top10"]), len(summary["top10"]))
                row = {
                    "delta": {
                        key: (
                            summary[key] - base[key]
                            if summary[key] is not None and base[key] is not None
                            else None
                        )
                        for key in outcome_keys
                    },
                    "effective": not np.array_equal(arrays[b], arrays[0]),
                    "zero_variance": summary["zero_variance"],
                    "overlap": (
                        len(base["top10"] & summary["top10"]) / denominator
                        if denominator
                        else None
                    ),
                    "edit_distance": sum(a[1] != c[1] for a, c in zip(visible, events)),
                    "timestamp_inversions": None,
                    "baseline_correct": bool(base["hr10"]),
                    "history_length": n,
                    "target_frequency": frequency[target],
                    "repeat": info["repeat"],
                }
                user_record.setdefault(name, []).append(row)
            all_swaps.append(user_record["swap-baseline"][0])
            if not matched:
                continue
            assignments.extend(pending_assignments)
            counts["matched_users"] += 1
            counts["successful_substitution_replicates"] += 10
            contrasts = []
            for recent, older in zip(
                user_record["recent-baseline"], user_record["older-baseline"]
            ):
                contrast = dict(recent)
                # These describe a single edited sequence and have no natural
                # interpretation for a difference between two interventions.
                contrast.update(overlap=None, edit_distance=None, zero_variance=None)
                contrast["delta"] = {
                    key: (
                        recent["delta"][key] - older["delta"][key]
                        if recent["delta"][key] is not None
                        and older["delta"][key] is not None
                        else None
                    )
                    for key in outcome_keys
                }
                contrast["effective"] = recent["effective"] or older["effective"]
                contrasts.append(contrast)
            user_record["recent-older"] = contrasts
            records.append(user_record)
            baseline_records.append(base)
        counts["evaluated_users"] = len(records)
        result = {
            "available": bool(records),
            "source": "training_prefix_in_sample_dependence",
            "category": category,
            "checkpoint_count": 1,
            "seed": 42,
            "counts": counts,
            "branches": {},
            "validation_usefulness": {
                "available": False,
                "reason": "No explicitly authorized pinned validation diagnostic protocol is supplied by ModelView/train_loader.",
            },
            "position_offset_probe": {
                "available": False,
                "reason": "No supported explicit position-ID interface; identity substitutions preserve the existing absolute slots.",
            },
            "fresh_process_predictor_parity": {
                "available": False,
                "reason": "No host reload runner is supplied; identity checks inspect only the live module.",
            },
            "cross_category_macro": {
                "available": False,
                "reason": "This probe observes one category; four-category macro effects require all four category records.",
            },
            "bootstrap_replicates": 20000,
            "planned_inferential_family_size": 48,
            "identity_rtol": 1.3e-6,
            "identity_atol": 1e-5,
            "identity_max_abs_error": max(
                (e for e in identity_errors if e is not None), default=None
            ),
            "support": support,
            "failures": failures,
            "category_metadata_breakdowns": {},
        }
        counts["excluded_substitution_users"] = len(selected) - len(records)
        counts["excluded_substitution_replicates"] = (
            counts["attempted_substitution_replicates"]
            - counts["successful_substitution_replicates"]
        )
        result["counts"] = dict(counts)
        result["support_rejection_fraction"] = (
            1.0 - len(records) / len(selected) if selected else None
        )
        result["all_eligible_swap_descriptive"] = {
            "users": len(all_swaps),
            "no_ops": sum(not r["effective"] for r in all_swaps),
            "effects": {
                key: {
                    "mean_delta": (float(np.mean(values)) if values else None),
                    "users": len(values),
                    "reason": None if values else "No defined observations",
                }
                for key in outcome_keys
                for values in [
                    [r["delta"][key] for r in all_swaps if r["delta"][key] is not None]
                ]
            },
        }
        # Save full provenance locally rather than depending on host metric truncation.
        directory = Path(__file__).resolve().parent / "diagnostic_artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        safe_category = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in category
        )
        # Unique files preserve repeated diagnostic calls and unlabeled categories
        # without consuming Python, NumPy or Torch training RNG streams.
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=safe_category + "_matched_substitution_",
            suffix=".json",
            delete=False,
        ) as handle:
            artifact = Path(handle.name)
            json.dump(
                {
                    "specification": "frequency_only_v1_seed42_R10_B20000_family48",
                    "category": category,
                    "catalog_order": catalog,
                    "frequency": {str(i): frequency[i] for i in catalog},
                    "bins": {
                        str(b): {
                            "lower_inclusive": 0 if b == 0 else 2 ** (b - 1),
                            "upper_exclusive": 1 if b == 0 else 2**b,
                            "size": len(items),
                            "items": items,
                        }
                        for b, items in bin_items.items()
                    },
                    "counts": dict(counts),
                    "support": support,
                    "failures": failures,
                    "assignments_in_draw_order": realized_assignments,
                    "assignment_analysis_rule": "Only users with matchable pools and passing finite identity fixtures enter balance, reuse and effect summaries; all realized assignments are retained here.",
                    "matched_user_records": records,
                    "all_eligible_swap_records": all_swaps,
                },
                handle,
                allow_nan=False,
            )
        result["provenance_artifact"] = str(artifact)
        reuse = Counter(a["donor"] for a in assignments)
        total = sum(reuse.values())
        result["donor_reuse"] = {
            "draws": total,
            "unique_donors": len(reuse),
            "reuse_counts": dict(reuse),
            "maximum_draw_share": max(reuse.values()) / total if total else None,
            "effective_donor_diversity": (
                total**2 / sum(v * v for v in reuse.values()) if total else None
            ),
            "reason": None if total else "No successful matched assignments",
        }
        result["frequency_balance"] = {}
        for block in ("older", "recent"):
            rows = [a for a in assignments if a["block"] == block]
            result["frequency_balance"][block] = {}
            for transform in ("raw", "log1p"):
                x = np.asarray([frequency[a["original"]] for a in rows], dtype=float)
                y = np.asarray([frequency[a["donor"]] for a in rows], dtype=float)
                if transform == "log1p":
                    x, y = np.log1p(x), np.log1p(y)
                pooled_sd = float(np.sqrt((x.var() + y.var()) / 2)) if len(x) else 0.0
                grid = np.unique(np.concatenate([x, y]))
                ecdf = (
                    float(
                        np.max(
                            np.abs(
                                np.searchsorted(np.sort(x), grid, side="right") / len(x)
                                - np.searchsorted(np.sort(y), grid, side="right")
                                / len(y)
                            )
                        )
                    )
                    if len(x)
                    else None
                )
                result["frequency_balance"][block][transform] = {
                    "occurrences": len(x),
                    "original_mean": float(x.mean()) if len(x) else None,
                    "donor_mean": float(y.mean()) if len(y) else None,
                    "donor_minus_original": float((y - x).mean()) if len(x) else None,
                    "ecdf_max_discrepancy": ecdf,
                    "standardized_mean_difference": (
                        float((y.mean() - x.mean()) / pooled_sd)
                        if pooled_sd > 0
                        else None
                    ),
                    "undefined_reason": (
                        "No assignments"
                        if not len(x)
                        else (
                            "Zero pooled variance for standardized difference"
                            if pooled_sd == 0
                            else None
                        )
                    ),
                }
        cutpoints = {
            key: (
                np.quantile([r[key] for r in support], [0.25, 0.5, 0.75]).tolist()
                if support
                else []
            )
            for key in ("history_length", "target_frequency")
        }
        result["support_characteristics"] = {"cutpoints": cutpoints, "groups": {}}
        for label, keep in (("retained", True), ("excluded", False)):
            group = [r for r in support if bool(r["matchable"] and r["scored"]) == keep]
            result["support_characteristics"]["groups"][label] = {
                "users": len(group),
                "repeat_users": sum(r["repeat"] for r in group),
                "quartile_counts": {
                    key: dict(
                        Counter(
                            int(np.searchsorted(cutpoints[key], r[key], side="right"))
                            for r in group
                        )
                    )
                    for key in cutpoints
                },
                "means": {
                    key: float(np.mean([r[key] for r in group])) if group else None
                    for key in cutpoints
                },
            }
        if not records:
            result["baseline"] = None
            result["reason"] = (
                "No matchable users with passing finite identity fixtures"
            )
            return result
        result["baseline"] = {}
        for key in ("rank", "hr10", "margin", "target_score"):
            values = [row[key] for row in baseline_records if row[key] is not None]
            result["baseline"][key] = {
                "mean": float(np.mean(values)) if values else None,
                "users": len(values),
            }
        result["baseline"]["zero_variance_users"] = sum(
            int(row["zero_variance"]) for row in baseline_records
        )
        # One shared set of complete-user resamples preserves dependence across all draws/branches.
        resamples = bootstrap_rng.integers(0, len(records), size=(20000, len(records)))
        for name in records[0]:
            user_draws = [record[name] for record in records]
            draws = [draw for rows in user_draws for draw in rows]
            report = {
                "users": len(user_draws),
                "draws": len(draws),
                "no_op_draws": sum(int(not row["effective"]) for row in draws),
                "zero_variance_draws": (
                    sum(int(row["zero_variance"]) for row in draws)
                    if name != "recent-older"
                    else None
                ),
                "top10_overlap": (
                    float(np.mean([row["overlap"] for row in draws]))
                    if name != "recent-older"
                    else None
                ),
                "mean_event_edit_distance": (
                    float(np.mean([row["edit_distance"] for row in draws]))
                    if name != "recent-older"
                    else None
                ),
                "sequence_summary_reason": (
                    "A paired effect contrast has no single edited sequence"
                    if name == "recent-older"
                    else None
                ),
                "paired_effects": {},
                "strata": {},
            }
            measured_times = [
                row["timestamp_inversions"]
                for row in draws
                if row["timestamp_inversions"] is not None
            ]
            report["temporal_availability"] = {
                "available": bool(measured_times),
                "draws": len(measured_times),
                "draws_with_inversions": sum(int(v > 0) for v in measured_times),
                "mean_inversions": (
                    float(np.mean(measured_times)) if measured_times else None
                ),
            }
            for key in ("hr10", "rank", "margin", "target_score"):
                means, effective_means = [], []
                for rows in user_draws:
                    valid = [
                        row["delta"][key]
                        for row in rows
                        if row["delta"][key] is not None
                    ]
                    effective = [
                        row["delta"][key]
                        for row in rows
                        if row["effective"] and row["delta"][key] is not None
                    ]
                    means.append(float(np.mean(valid)) if valid else np.nan)
                    if effective:
                        effective_means.append(float(np.mean(effective)))
                values = np.asarray(means, dtype=np.float64)
                sampled = values[resamples]
                finite_counts = np.isfinite(sampled).sum(axis=1)
                boot = np.divide(
                    np.nansum(sampled, axis=1),
                    finite_counts,
                    out=np.full(20000, np.nan),
                    where=finite_counts > 0,
                )
                boot = boot[np.isfinite(boot)]
                finite = values[np.isfinite(values)]
                report["paired_effects"][key] = {
                    "mean_delta": float(finite.mean()) if len(finite) else None,
                    "users": len(finite),
                    "exploratory_paired_user_95pct_interval": (
                        np.percentile(boot, [2.5, 97.5]).tolist()
                        if len(boot) and key != "target_score"
                        else None
                    ),
                    "approximate_familywise_bonferroni_interval": (
                        np.quantile(
                            boot, [0.05 / (2 * 48), 1 - 0.05 / (2 * 48)]
                        ).tolist()
                        if len(boot) and key != "target_score"
                        else None
                    ),
                    "interval_reason": (
                        "Target scores are descriptive, outside the planned family"
                        if key == "target_score"
                        else (None if len(boot) else "No defined observations")
                    ),
                    "valid_bootstrap_replicates": len(boot),
                    "effective_edit_mean_delta": (
                        float(np.mean(effective_means)) if effective_means else None
                    ),
                    "effective_edit_users": len(effective_means),
                }
            report["donor_monte_carlo_standard_errors"] = {}
            if name != "swap-baseline":
                for key in ("margin", "rank", "hr10", "target_score"):
                    replicate_means = [
                        float(np.mean(values)) if values else None
                        for replicate in range(10)
                        for values in [
                            [
                                rows[replicate]["delta"][key]
                                for rows in user_draws
                                if rows[replicate]["delta"][key] is not None
                            ]
                        ]
                    ]
                    valid_means = [v for v in replicate_means if v is not None]
                    report["donor_monte_carlo_standard_errors"][key] = {
                        "replicate_category_means": replicate_means,
                        "standard_error": (
                            float(np.std(valid_means, ddof=1) / np.sqrt(10))
                            if len(valid_means) == 10
                            else None
                        ),
                        "reason": (
                            "Ten-draw precision is limited"
                            if len(valid_means) == 10
                            else "Outcome undefined for one or more replicate means"
                        ),
                    }
            selectors = {
                "repeat_status": lambda row: "repeat" if row["repeat"] else "explore",
                "target_frequency": lambda row: str(
                    int(
                        np.searchsorted(
                            cutpoints["target_frequency"],
                            row["target_frequency"],
                            side="right",
                        )
                    )
                ),
                "baseline_correct": lambda row: (
                    "correct" if row["baseline_correct"] else "incorrect"
                ),
                "history_length": lambda row: (
                    "2-5"
                    if row["history_length"] <= 5
                    else ("6-20" if row["history_length"] <= 20 else "21+")
                ),
                "event_edit_distance": lambda row: str(row["edit_distance"]),
            }
            if name == "recent-older":
                selectors.pop("event_edit_distance")
            for stratum, selector in selectors.items():
                groups = {}
                for rows in user_draws:
                    grouped_user = {}
                    for row in rows:
                        grouped_user.setdefault(selector(row), []).append(
                            row["delta"]["hr10"]
                        )
                    for label, values in grouped_user.items():
                        group = groups.setdefault(
                            label, {"users": 0, "draws": 0, "user_mean_deltas": []}
                        )
                        group["users"] += 1
                        group["draws"] += len(values)
                        group["user_mean_deltas"].append(float(np.mean(values)))
                for group in groups.values():
                    group["mean_delta_hr10"] = float(
                        np.mean(group.pop("user_mean_deltas"))
                    )
                report["strata"][stratum] = groups
            result["branches"][name] = report
        return result

    def run_full_diagnosis(self, train_loader):
        previous_modes = [
            (module, module.training) for module in self.network.modules()
        ]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        # Preserve explicit loader/sampler generators as well as global RNGs.
        generators = []
        seen_generators = set()
        for owner in (
            train_loader,
            getattr(train_loader, "sampler", None),
            getattr(getattr(train_loader, "batch_sampler", None), "sampler", None),
        ):
            generator = getattr(owner, "generator", None)
            if (
                isinstance(generator, torch.Generator)
                and id(generator) not in seen_generators
            ):
                seen_generators.add(id(generator))
                generators.append((generator, generator.get_state()))
        # Restrict CUDA state capture and seeding to the actual model device.
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
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                for device_index in cuda_devices:
                    torch.cuda.default_generators[device_index].manual_seed(42)
                for generator, _ in generators:
                    generator.manual_seed(42)
                self.network.eval()
                with torch.no_grad():
                    collapse_score = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048
                    )
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            train_loader, num_batches=10, eps=0.5, topk_categories=20
                        )
                    )
                    prefix = self._prefix_interventions()
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "recent_history_probe": prefix,
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "Mean sampled real-item pair cosine mapped to [0,1]; excludes PAD/MASK. Measures angular similarity, not recommendation diversity. Null when fewer than two embeddings can be sampled.",
                        "pos_neg_margin_mean": "Mean positive-minus-negative logit on real-label supervised positions containing reserved MASK; training separation, not validation performance. Null if no valid positions.",
                        "pos_neg_margin_p10": "Nearest-rank 10th percentile of audited supervised training margins; null if unavailable.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of audited masked positions with negative logit greater than positive logit.",
                        "pos_neg_small_margin_rate": "Fraction of audited masked positions with margin below 0.5.",
                        "pos_neg_margin_category_breakdown": "Counts and mask audit for up to ten training batches. Original-label/corruption provenance is unavailable; equality alone cannot establish leakage. Category lists remain empty because categories are unavailable.",
                        "recent_history_probe": {
                            "source": "One independently trained category checkpoint. Up to 200 seed-42 sampled eligible training users after catalog-ID validation and length filtering. Target is the final training event; prefix is its strictly earlier last L-1 events. Earlier target repeats remain valid. Training-exposed recovery only; no host validation claim.",
                            "candidates": "Target once plus up to 99 unique uniform negatives excluding the entire supplied training history. Small eligible pools use all available items and report actual counts. Candidate arrays and order are frozen before donor draws and reused across interventions. These auxiliary candidates do not change the host benchmark.",
                            "matching": "Occurrence frequency counts all valid supplied training events including repeats. Bin 0 is f=0; positive frequency bin b=1+floor(log2(f)) has [2^(b-1),2^b). Catalog IDs and donor pools are sorted. Each original occurrence is replaced uniformly from its bin excluding the full user history and every candidate. Draws use replacement across occurrences and ten paired replicates; repeats may be introduced. Empty pools are never widened. No additional plausibility or repeat-multiplicity matching is performed.",
                            "inference": "Reserved appended MASK, unchanged absolute slots and padding, greatest nonpadding absolute readout, one module in evaluation mode without gradients. Live identity-copy scores require rtol=1.3e-6 and atol=1e-5; failed or nonfinite user fixtures are excluded and recorded. Live parity does not certify fresh-process predictor/checkpoint parity, which remains unavailable.",
                            "interventions": "Disjoint oldest and newest blocks have k=min(3,floor(n/2)). Ten paired identity-substitution replicates preserve length, visibility, padding and final MASK. Swap exchanges the two blocks preserving internal order, middle tokens and exact item multiset. Four branches report recent-baseline, older-baseline, recent-older and swap-baseline on common matchable users. All-eligible swap effects are separately descriptive. No-op swaps remain included.",
                            "outcomes": "Target score, target-minus-hardest-negative margin, one-based stable candidate-order target rank and descriptive HR10=rank<=10. Margin is null with no negatives. Top10 overlap is intersection/minimum returned list length; empty lists make it undefined. Temporal inversion fields are unavailable for this specification, not evidence of chronological health.",
                            "aggregation": "Average the ten replicates within each user then weight users equally. Recent-older subtracts paired older effects from recent effects. Positive rank delta means worse recovery; positive margin or HR delta means improved recovery. Effective-edit summaries are separately descriptive. No matching multiplicity weights are applied.",
                            "uncertainty": "20000 seed-42 paired-user bootstrap resamples share indices across contrasts within category, conditional on fitted checkpoint, selected users, fixed candidates and realized donors. Ordinary percentile 95% intervals are exploratory. Bonferroni percentile tails are 0.05/(2*48) and 1-0.05/(2*48), retaining the planned four contrasts x three outcomes x four categories denominator. Approximate coverage; empirical tail resolution is 1/20000, about 10.4 draws per extreme tail. Target scores, overlap and strata are descriptive and outside the inferential family. Undefined outcomes report null and counts.",
                            "donor_monte_carlo": "For each paired replicate compute its equal-user category mean; report across-ten-replicate sample SD/sqrt(10) separately from user intervals. Ten draws give limited precision. This concerns donor sampling conditional on users and candidates and is not added to bootstrap intervals. Donor concentration and 1/sum(p_d^2) are draw-diversity summaries, not independent-user sample sizes.",
                            "support_balance": "Per-position pool sizes, attempted/matched users and replicates, empty-pool reasons, candidate counts and complete assignments are recorded. Rejection fraction includes fixture failures. Retained versus excluded sampled users are compared by full history length, target training frequency and retained-prefix repeat status. Quartile cutpoints come from all sampled support rows; ties use searchsorted side=right. Balance uses occurrence-weighted raw/log1p means, donor-minus-original differences, maximum empirical-CDF discrepancy and standardized differences using sqrt((var_original+var_donor)/2); zero variance makes standardized differences null. Coarse balance does not imply contextual plausibility, exchangeability or absence of selection bias.",
                            "provenance": "Trial-local diagnostic_artifacts JSON records category, catalog order, occurrence counts, exact bin edges and sizes, fixed candidates, per-user seeds, per-token donor pools sizes, assignments in draw order, effects and failures. No model architecture, training objective, predictor writer or checkpoint configuration changes. No extra fitted scorer or score fusion.",
                            "limitations": "Substitution damage mixes content and context disruption; swaps mix arrangement, local context and item-position associations. Neither identifies causal recency, preferences or utility. Position-offset and fresh-process parity hooks remain unavailable. Results do not generalize automatically beyond matchable users. No unavailable reviews, ratings, item categories, commercial attributes or simulator traits are used.",
                            "availability_budget": "Only host-supported four-category macro validation HR10 selects revisions. No validation diagnostic protocol or reload runner is supplied here. No candidate sensitivity seeds or adaptive bin selection are added. This bounded diagnostic specification uses ten loader batches and at most 200 users/category. All execution consumes the shared search budget. Module modes, global RNGs and discovered loader generators are restored; iterator progress and worker-local RNGs cannot generally be restored. Supply a disposable diagnostic loader without persistent workers. Training, four-category routing, full-catalog simulator recommendations and one checkpoint/category remain unchanged.",
                        },
                    },
                }
        finally:
            for module, was_training in previous_modes:
                module.training = was_training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in generators:
                generator.set_state(state)

    ### <<< Self_EvolveRec-BLOCK-END
