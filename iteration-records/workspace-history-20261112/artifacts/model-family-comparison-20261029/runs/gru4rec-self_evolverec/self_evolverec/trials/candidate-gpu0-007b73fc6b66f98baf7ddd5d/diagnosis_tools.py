import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Local deterministic diagnostic sampling
from collections import Counter
import hashlib
import os

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
    ### >>> Self_EvolveRec-BLOCK-START: Require exposure and uncertainty labels in diagnostic interpretations
    - Every statement must be grounded in the provided metric values/definitions.
    - Missing measurements are unavailable, never evidence of healthy behavior.
    - Report padding failures before interpreting temporal effects. Temporal effects are unavailable when padding verification fails.
    - Training-prefix probes do not estimate unseen-interaction generalization. Preserve their interaction-exposure labels, including unknown exposure.
    - Bootstrap intervals condition on the checkpoint, candidates and perturbations; they exclude retraining, negative-sampling and repeated-search uncertainty.
    - Directional embedding alignment alone does not establish representation collapse. Raw margins depend on score scale.
    - Do not infer simulator traits, preferences, rejection reasons or causal effects from these measurements.
    - Distinguish sampled-candidate overlap, full-catalog overlap and the pinned validation objective.
    - Do not describe category confidence intervals as four-category macro intervals. A macro result requires all four category records.
    - If measured metrics look healthy, status may remain STABLE; unavailable metrics must still be identified.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and restrict diagnostics to observed task data
    def __init__(self, view, device):
        self.view = view
        self.model = view  # Preserve the existing supplementary-probe interface.
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        configured = getattr(view, "maxlen", None)
        if configured is None:
            configured = getattr(view.model, "maxlen", None)
        if configured is None:
            configured = getattr(getattr(view, "args", None), "maxlen", None)
        self.maxlen_source = "ModelView/model configuration"
        if configured is None:
            configured = int(os.environ.get("GRU4REC_MAXLEN", "128"))
            self.maxlen_source = "GRU4REC_MAXLEN or supplied baseline default"
        self.maxlen = int(configured)
        if self.maxlen < 1:
            raise ValueError("Diagnostic context length must be positive")

    def _get_categories(self, item_id: int):
        # Item category metadata is unavailable in this matched task.
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
            ### >>> Self_EvolveRec-BLOCK-START: Missing alignment measurements are null
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Reject nonfinite embedding measurements explicitly
            x = emb.index_select(0, idx).float()
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError(
                    "Nonfinite sampled item embeddings in alignment diagnostic"
                )
            x = F.normalize(x, p=2, dim=-1)
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: The full diagnostic owns evaluation mode and restores it
        # run_full_diagnosis controls the actual nn.Module's mode.
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

            seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            log_feats = self.model.log2feats(seq_np)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic shapes and score through the family interface
            if pos_t.shape != neg_t.shape or pos_t.ndim != 2:
                raise ValueError(
                    "Diagnostic positives and negatives must both have shape B x L"
                )
            pos_logits = self.view.predict_score(log_feats, pos_t)
            neg_logits = self.view.predict_score(log_feats, neg_t)
            if pos_logits.shape != pos_t.shape or neg_logits.shape != neg_t.shape:
                raise ValueError(
                    "ModelView must return one score per diagnostic item ID"
                )
            margin = pos_logits - neg_logits
            observed = (pos_t != 0) & (neg_t != 0)
            if not bool(torch.isfinite(margin[observed]).all()):
                raise FloatingPointError(
                    "Nonfinite scores in observed diagnostic margin comparisons"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and unavailable negative comparisons
            valid = (pos_t != 0) & (neg_t != 0)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Match category counts to valid margin comparisons
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
            ### >>> Self_EvolveRec-BLOCK-START: Report absent margin observations explicitly
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "count": 0,
                "reason": "No positions with both a positive and an eligible negative",
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
                ### >>> Self_EvolveRec-BLOCK-START: Retain observed margin sample counts
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
                "count": int(N),
                "reason": None,
                ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Padding-gated paired GRU temporal audit with conditional uncertainty
    def _histories(self):
        histories = self.user_train.get("History", self.user_train)
        return sorted(histories.items(), key=lambda pair: str(pair[0]))

    @staticmethod
    def _rng(user, purpose):
        key = json.dumps([42, str(user), purpose], ensure_ascii=True)
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        return random.Random(seed)

    def _sequence(self, events, width=None):
        width = self.maxlen if width is None else int(width)
        events = list(events)[-width:]
        sequence = torch.zeros((1, width), dtype=torch.long, device=self.device)
        if events:
            sequence[0, -len(events) :] = torch.as_tensor(
                events, dtype=torch.long, device=self.device
            )
        return sequence

    def _last_features(self, sequence):
        # ModelView's established interface accepts numpy sequences.
        features = self.view.log2feats(sequence.detach().cpu().numpy())
        positions = torch.arange(sequence.size(1), device=sequence.device)
        last = positions.expand_as(sequence).masked_fill(sequence == 0, -1)
        last = last.max(dim=1).values.clamp_min(0)
        result = features[torch.arange(sequence.size(0), device=self.device), last]
        return result.masked_fill(~sequence.ne(0).any(dim=1, keepdim=True), 0)

    def _scores(self, features, candidates):
        items = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        # B x C item IDs avoid confusing pairwise and candidate scoring paths.
        items = items.unsqueeze(0).expand(features.size(0), -1)
        return self.view.predict_score(features, items)

    def verify_padding(self, atol=1e-5, rtol=1e-4):
        actual = self.view.model
        result = {
            "status": "unavailable",
            "passed": False,
            "atol": atol,
            "rtol": rtol,
            "comparisons": 0,
            "history_count": 0,
            "max_feature_abs_error": None,
            "max_score_abs_error": None,
            "reason": None,
            "backend": {
                "torch": str(torch.__version__),
                "device": str(self.device),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cudnn_enabled": bool(torch.backends.cudnn.enabled),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "dtype": str(self.view.item_emb.weight.dtype),
                "model_class": type(actual).__name__,
            },
            "reference": "Zero initial states; explicit GRU state masking; identical weights and event retention",
            "tested_layouts": [
                "unpadded",
                "baseline_left_padding",
                "extra_left_padding",
                "mixed_batch",
            ],
        }
        if not isinstance(getattr(actual, "gru", None), torch.nn.GRU):
            result["reason"] = (
                "No supported GRU reference; adapt the reference to this backbone"
            )
            return result
        if actual.gru.bidirectional or self.view.item_emb.weight.dtype != torch.float32:
            result["reason"] = "This reference requires a unidirectional float32 GRU"
            return result
        histories = []
        for _, history in self._histories():
            events = [int(item) for item in history][-self.maxlen :]
            if events and all(1 <= item <= actual.itemnum for item in events):
                histories.append(events)
            if len(histories) == 4:
                break
        if not histories:
            result["reason"] = "No nonempty valid training histories"
            return result
        # Include short prefixes to exercise substantially different leading padding.
        examples = []
        for events in histories:
            for length in sorted({1, min(4, len(events)), len(events)}):
                examples.append(events[-length:])
        result["history_count"] = len(examples)
        candidates = list(range(1, min(actual.itemnum, 100) + 1))
        all_passed = True
        max_feature_error = max_score_error = 0.0

        def reference(events):
            hidden = self.view.item_emb.weight.new_zeros(
                actual.gru.num_layers, 1, actual.gru.hidden_size
            )
            rho = 1.0
            if getattr(actual, "event_decay", False):
                rho = torch.exp(-F.softplus(actual.retention_logit))
            # Compact valid events: no leading padding enters the recurrent state.
            for item in events:
                token = torch.tensor([[item]], dtype=torch.long, device=self.device)
                embedding = actual.item_emb(token)
                embedding = actual.dropout(embedding)
                _, proposed = actual.gru(embedding, hidden * rho)
                hidden = proposed
            return hidden[-1]

        def compare(features, expected):
            nonlocal all_passed, max_feature_error, max_score_error
            scores = self._scores(features, candidates)
            expected_scores = self._scores(expected, candidates)
            finite = bool(
                torch.isfinite(features).all() and torch.isfinite(scores).all()
            )
            finite = finite and bool(
                torch.isfinite(expected).all() and torch.isfinite(expected_scores).all()
            )
            result["comparisons"] += 1
            if not finite:
                all_passed = False
                result["reason"] = "Nonfinite padding-reference features or scores"
                return
            max_feature_error = max(
                max_feature_error, float((features - expected).abs().max().item())
            )
            max_score_error = max(
                max_score_error, float((scores - expected_scores).abs().max().item())
            )
            all_passed = (
                all_passed
                and torch.allclose(features, expected, atol=atol, rtol=rtol)
                and torch.allclose(scores, expected_scores, atol=atol, rtol=rtol)
            )

        try:
            for index, events in enumerate(examples):
                expected = reference(events)
                for width in sorted({len(events), self.maxlen, self.maxlen + 3}):
                    sequence = self._sequence(events, width)
                    compare(self._last_features(sequence), expected)
                    companion = histories[index % len(histories)][-width:]
                    mixed = torch.cat(
                        [
                            self._sequence(companion, width),
                            sequence,
                            self._sequence([], width),
                        ],
                        dim=0,
                    )
                    compare(self._last_features(mixed)[1:2], expected)
                # Check the raw feature output on empty rows; do not hide failures
                # behind the explicit empty readout used by diagnostic helpers.
                empty = self._sequence([])
                raw = self.view.log2feats(empty.detach().cpu().numpy())[:, -1]
                compare(raw, reference([]))
            result.update(
                {
                    "status": "passed" if all_passed else "failed",
                    "passed": bool(all_passed),
                    "max_feature_abs_error": max_feature_error,
                    "max_score_abs_error": max_score_error,
                }
            )
            if not all_passed and result["reason"] is None:
                result["reason"] = (
                    "At least one feature or candidate-score comparison exceeded tolerance"
                )
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
        ) as error:
            result.update(
                {
                    "status": "unavailable",
                    "passed": False,
                    "max_feature_abs_error": (
                        max_feature_error if result["comparisons"] else None
                    ),
                    "max_score_abs_error": (
                        max_score_error if result["comparisons"] else None
                    ),
                    "reason": (
                        f"Padding verification could not complete: {type(error).__name__}: {error}"
                    ),
                }
            )
        return result

    @staticmethod
    def _effect_summary(records):
        if not records:
            return None
        names = sorted(records[0]["effects"])
        values = np.asarray(
            [[record["effects"][name] for name in names] for record in records],
            dtype=np.float64,
        )
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(records), size=(2000, len(records)))
        boot = values[indices].mean(axis=1)
        return {
            name: {
                "mean": float(values[:, index].mean()),
                "ci95": np.quantile(boot[:, index], [0.025, 0.975]).tolist(),
            }
            for index, name in enumerate(names)
        }

    def analyze_temporal_sensitivity(self):
        itemnum = int(self.view.model.itemnum)
        records = []
        skips = Counter()
        eligible_examined = 0
        for user, history in self._histories():
            history = [int(item) for item in history]
            context = history[:-1][-self.maxlen :]
            if len(context) < 4:
                skips["short_context"] += 1
                continue
            if eligible_examined >= 200:
                break
            eligible_examined += 1
            if any(item < 1 or item > itemnum for item in history):
                skips["invalid_item_id"] += 1
                continue
            target = history[-1]
            excluded = set(history)
            if itemnum - len(excluded) < 99:
                skips["insufficient_negatives"] += 1
                continue
            rng = self._rng(user, "negative_candidates")
            available = itemnum - len(excluded)
            if available <= 198 or len(excluded) > itemnum // 2:
                negatives = rng.sample(
                    [item for item in range(1, itemnum + 1) if item not in excluded], 99
                )
            else:
                negatives, selected = [], set()
                while len(negatives) < 99:
                    candidate = rng.randint(1, itemnum)
                    if candidate not in excluded and candidate not in selected:
                        negatives.append(candidate)
                        selected.add(candidate)
            candidates = [target] + negatives
            prefix, latest = context[:-1], context[-1]
            k = max(1, len(prefix) // 4)
            variants = {
                "original": context,
                "last_only": [latest],
                "recent_deleted": prefix[:-k] + [latest],
                "oldest_deleted": prefix[k:] + [latest],
            }
            unchanged = 0
            for repeat in range(5):
                shuffled = list(prefix)
                self._rng(user, f"prefix_shuffle_{repeat}").shuffle(shuffled)
                unchanged += int(shuffled == prefix)
                variants[f"prefix_shuffle_{repeat}"] = shuffled + [latest]
            outcomes, top_sets = {}, {}
            invalid_scores = False
            # Nine input interventions share one checkpoint and one candidate set.
            # Batch execution reduces recurrent launches without averaging scores.
            variant_names = list(variants)
            sequences = torch.cat(
                [self._sequence(variants[name]) for name in variant_names], dim=0
            )
            features = self._last_features(sequences)
            score_tensor = self._scores(features, candidates)
            expected_shape = (len(variant_names), len(candidates))
            if tuple(score_tensor.shape) != expected_shape:
                raise ValueError(
                    f"Temporal candidate scores must have shape {expected_shape}; "
                    f"received {tuple(score_tensor.shape)}"
                )
            score_rows = score_tensor.detach().float().cpu().numpy()
            for name, scores in zip(variant_names, score_rows):
                if not np.isfinite(scores).all():
                    invalid_scores = True
                    break
                ordered = np.lexsort((np.asarray(candidates, dtype=np.int64), -scores))
                rank = int(np.flatnonzero(ordered == 0)[0]) + 1
                outcomes[name] = {
                    "rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "HR@10": float(rank <= 10),
                }
                top_sets[name] = {candidates[int(index)] for index in ordered[:10]}
            if invalid_scores:
                skips["invalid_candidate_scores"] += 1
                continue
            original_top = top_sets["original"]
            for name, top in top_sets.items():
                outcomes[name]["top10_jaccard"] = len(top & original_top) / len(
                    top | original_top
                )
            outcomes["prefix_shuffle"] = {
                metric: float(
                    np.mean(
                        [
                            outcomes[f"prefix_shuffle_{repeat}"][metric]
                            for repeat in range(5)
                        ]
                    )
                )
                for metric in outcomes["original"]
            }
            effects = {}
            for name in (
                "last_only",
                "prefix_shuffle",
                "recent_deleted",
                "oldest_deleted",
            ):
                for metric in ("rank", "HR@10", "reciprocal_rank"):
                    effects[f"{name}.{metric}_delta"] = (
                        outcomes[name][metric] - outcomes["original"][metric]
                    )
            for metric in ("rank", "HR@10", "reciprocal_rank"):
                effects[f"recent_minus_oldest_deleted.{metric}"] = (
                    outcomes["recent_deleted"][metric]
                    - outcomes["oldest_deleted"][metric]
                )
            records.append(
                {
                    "user": str(user),
                    "target": target,
                    "context_events": len(context),
                    "deleted_events": k,
                    "exposure": "unknown",
                    "exposure_reason": "The diagnostic has no realized training-sampler exposure ledger",
                    "target_available_during_fitting": True,
                    "target_also_in_context": target in context,
                    "unchanged_shuffle_permutations": unchanged,
                    "candidates": candidates,
                    "outcomes": outcomes,
                    "effects": effects,
                }
            )
        variant_means = None
        if records:
            variant_means = {
                name: {
                    metric: float(
                        np.mean([r["outcomes"][name][metric] for r in records])
                    )
                    for metric in records[0]["outcomes"][name]
                }
                for name in records[0]["outcomes"]
            }
        return {
            "status": "available" if records else "unavailable",
            "reason": None if records else "No eligible finite paired observations",
            "source": "training-prefix fallback; targets available during fitting",
            "selection": "Stable lexicographic string user-ID order; first 200 context-eligible users",
            "maxlen": self.maxlen,
            "maxlen_source": self.maxlen_source,
            "eligible_examined": eligible_examined,
            "count": len(records),
            "skip_counts": dict(skips),
            "exposure_counts": {"confirmed_in_sample": 0, "unknown": len(records)},
            "unchanged_shuffle_permutations": sum(
                r["unchanged_shuffle_permutations"] for r in records
            ),
            "variant_means": variant_means,
            "paired_effects": self._effect_summary(records),
            "records": records,
            "four_category_macro": {
                "status": "unavailable",
                "estimate": None,
                "reason": "A single category probe cannot observe all four checkpoints; use aggregate_temporal_audits with all four category results",
            },
        }

    @staticmethod
    def aggregate_temporal_audits(by_category):
        # Optional host-side aggregation of supplied results only; no file reads.
        required = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        audits, missing = {}, []
        for category in required:
            raw = by_category.get(category)
            if raw is None:
                missing.append(category)
                continue
            metrics = raw.get("metrics", raw)
            audit = metrics.get("temporal_audit", {})
            padding = metrics.get("padding_verification", {})
            if (
                padding.get("passed") is not True
                or audit.get("status") != "available"
                or not audit.get("records")
            ):
                missing.append(category)
                continue
            audits[category] = audit
        counts = {
            category: len(audits[category]["records"]) if category in audits else 0
            for category in required
        }
        if missing:
            return {
                "status": "unavailable",
                "paired_effects": None,
                "category_counts": counts,
                "missing_categories": missing,
                "reason": "All four padding-verified category summaries are required",
            }
        names = sorted(audits[required[0]]["records"][0]["effects"])
        rng = np.random.default_rng(42)
        macro_point = np.zeros(len(names), dtype=np.float64)
        macro_boot = np.zeros((2000, len(names)), dtype=np.float64)
        category_effects = {}
        for category in required:
            records = audits[category]["records"]
            if any(sorted(record["effects"]) != names for record in records):
                return {
                    "status": "unavailable",
                    "paired_effects": None,
                    "category_counts": counts,
                    "reason": "Inconsistent diagnostic panels",
                }
            values = np.asarray(
                [[record["effects"][name] for name in names] for record in records],
                dtype=np.float64,
            )
            indices = rng.integers(0, len(records), size=(2000, len(records)))
            boot = values[indices].mean(axis=1)
            point = values.mean(axis=0)
            macro_point += point / 4.0
            macro_boot += boot / 4.0
            category_effects[category] = {
                name: {
                    "mean": float(point[index]),
                    "ci95": np.quantile(boot[:, index], [0.025, 0.975]).tolist(),
                }
                for index, name in enumerate(names)
            }
        return {
            "status": "available",
            "category_counts": counts,
            "skip_counts": {
                category: audits[category]["skip_counts"] for category in required
            },
            "per_category_effects": category_effects,
            "paired_effects": {
                name: {
                    "mean": float(macro_point[index]),
                    "ci95": np.quantile(macro_boot[:, index], [0.025, 0.975]).tolist(),
                }
                for index, name in enumerate(names)
            },
            "bootstrap": {
                "replicates": 2000,
                "seed": 42,
                "unit": "users resampled independently within each category",
                "aggregation": "equal-weight four-category macro",
            },
        }

    def run_full_diagnosis(self, train_loader):
        # Preserve mixed submodule modes and global RNG streams: consuming the
        # diagnostic loader must not change subsequent training sampling/dropout.
        original_modes = [
            (module, module.training) for module in self.view.model.modules()
        ]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        self.view.model.eval()
        try:
            with torch.no_grad():
                padding = self.verify_padding()
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                if padding["passed"]:
                    temporal = self.analyze_temporal_sensitivity()
                else:
                    temporal = {
                        "status": "unavailable",
                        "count": 0,
                        "records": [],
                        "paired_effects": None,
                        "reason": "Padding verification did not pass. Resolve training and predictor computation consistently, retrain if changed, then repeat the fixed-checkpoint audit.",
                    }
                item_count = int(self.view.model.itemnum)
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "embedding_alignment_sample_count": min(2048, item_count),
                        "embedding_alignment_reason": (
                            None
                            if collapse_score is not None
                            else "Fewer than two real item embeddings"
                        ),
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_count": margin_metrics["count"],
                        "pos_neg_margin_reason": margin_metrics["reason"],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "padding_verification": padding,
                        "temporal_audit": temporal,
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "Legacy name: (mean off-diagonal cosine + 1)/2 for up to 2048 sampled real-item embeddings, seed 42. Measures directional alignment; alone it does not establish collapse or recommendation diversity.",
                        "embedding_alignment_sample_count": "Number of sampled real-item embeddings; alignment is null when fewer than two exist.",
                        "pos_neg_margin_mean": "Mean positive-minus-negative logit over positions where both item IDs are nonzero. Scale-sensitive training-sample diagnostic, not validation performance.",
                        "pos_neg_margin_p10": "Nearest-rank tenth percentile of positive-minus-negative logit over the same valid positions.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of valid margins strictly below zero.",
                        "pos_neg_small_margin_rate": "Fraction of valid margins strictly below eps=0.5; scale-sensitive.",
                        "pos_neg_margin_count": "Observed valid positive/negative comparisons from at most ten diagnostic batches. Missing measurements are null with a reason.",
                        "pos_neg_margin_category_breakdown": "Legacy fields retained. Category lists are empty because item-category metadata is unavailable; position counts remain measured.",
                        "padding_verification": "Float32 eval-mode, identical-weight feature and candidate-score comparisons against a zero-initial-state compact-event GRU reference. Tests unpadded execution, baseline and extra leading padding, batch companions and empty rows. atol=1e-5, rtol=1e-4. Temporal results are gated on passing every comparison. No diagnostic-only architecture correction is applied.",
                        "temporal_audit": {
                            "scope": "Deterministic training-prefix fallback: target=history[-1], context=history[:-1] truncated to the configured baseline cap, at least four context events. These targets were available during fitting; realized exact user-target interaction exposure remains unknown without a sampler ledger. Repeated target occurrences in context are recorded separately. This does not estimate unseen-interaction generalization.",
                            "candidates": "Target plus 99 distinct real-item negatives excluding the complete training history, sampled with a local SHA256-derived seed-42 user RNG. Identical candidates across all variants; insufficient catalogs are skipped and counted. These are diagnostic candidates, not reconstructed host validation candidates.",
                            "variants": "Original; latest item only; five deterministic prefix shuffles preserving multiplicities, length and latest item; recent-prefix deletion prefix[:-k]+[latest]; oldest-prefix deletion prefix[k:]+[latest], k=max(1,len(prefix)//4). Unchanged shuffles are retained and counted. Shuffles average outcomes within user, never scores or checkpoints.",
                            "outcomes": "One-based target rank with ascending item-ID tie breaking; reciprocal rank=1/rank; HR@10=1[rank<=10]; sampled-candidate top-10 Jaccard against original. Every variant uses left padding and the greatest nonzero input index for readout.",
                            "paired_effects": "Variant-minus-original rank, HR@10 and reciprocal-rank differences. Matched deletion effects are recent-deleted minus oldest-deleted; positive rank or negative HR differences mean greater damage from recent deletion. Effects describe input perturbation sensitivity, not causal preferences.",
                            "uncertainty": "2000 paired user bootstrap replicates, local numpy seed 42, percentile 95% intervals. Category intervals are separate from macro intervals. aggregate_temporal_audits requires all four padding-verified category results, independently resamples users within categories and averages four category effects equally. Missing categories make macro results unavailable. Intervals condition on fitted checkpoints, realized candidates and perturbations and exclude retraining, negative-sampling and repeated-search uncertainty.",
                            "limitations": "No ratings, review text, prices, stores, item categories or simulator traits are observed. Sampled-candidate overlap is distinct from native full-catalog unseen-item overlap and the pinned validation objective. No full-catalog overlap is measured by this panel.",
                        },
                    },
                }
        finally:
            for module, training in original_modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)

    # Review checks: traced ModelView scoring shapes, paired candidate reuse,
    # padding-failure gating, finite measurements, mode/RNG restoration and
    # single-checkpoint batched interventions. No execution was performed.
    # Training configuration and generated predictor are unchanged.
    ### <<< Self_EvolveRec-BLOCK-END
