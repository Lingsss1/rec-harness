### >>> Self_EvolveRec-BLOCK-START: Bounded paired HSTU sensitivity diagnostics
import json
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Ground interpretation in sensitivity controls and missingness
def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
Analyze these measurements from one fitted HSTU model. Return concise JSON with
status (CRITICAL, NEEDS_IMPROVEMENT, STABLE, or INSUFFICIENT_EVIDENCE),
core_findings, key_implications, evidence, limitations, and hypotheses.

Measured values, counts, eligibility, and missingness:
{json.dumps(raw_diagnosis['metrics'], indent=2, allow_nan=False)}

Metric definitions:
{json.dumps(raw_diagnosis['metric_definitions'], indent=2, allow_nan=False)}

Rules:
- Ground every finding in measured values and their definitions.
- Report identity failures and missingness first. Null is unavailable, never zero.
  No checked identity rows means insufficient evidence of identity parity.
- These are training-input sensitivity measurements conditional on one fitted
  model and fixed target pairs. Do not infer user intent, causal preferences,
  time sensitivity, model compatibility, or expected validation improvement.
- Ordinary deletion combines information removal and coordinate repacking.
  Keep final-event removal, penultimate-event removal, and prefix deletion separate.
- Embedding cosine geometry does not measure recommendation diversity or accuracy.
  Margin < 0.5 is a score-scale-dependent threshold.
- Rows and replicates are dependent and are not verified unique users. Omit
  user-level confidence intervals and significance claims without verified IDs.
- Loader diagnostics cannot measure validation HR@10 or target-rank changes.
  Do not forecast numerical uplift or invoke unavailable metadata or simulator traits.
- Include at most two explicitly untested hypotheses. Any suggested follow-up must
  follow the predeclared sequence; avoid repeated adaptive validation mining.
- Preserve the HSTU family, one trained checkpoint per category, and unchanged host
  four-category macro validation HR@10 for selecting any future training change.
- Keep core_findings and key_implications to at most three entries each.
"""


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Identity-gated fixed-pair HSTU diagnostics with bounded geometry computation
class DiagnosisProbe:
    CONDITIONS = (
        "prefix_delete_oldest",
        "prefix_delete_recent",
        "prefix_delete_random",
        "delete_final",
        "delete_penultimate",
        "shuffle_prefix",
        "swap_adjacent_prefix",
        "shuffle_before_suffix",
    )

    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        # Preserve the public adapter references without consuming unavailable fields.
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    @staticmethod
    def _unavailable(reason):
        return {"value": None, "reason": reason}

    @staticmethod
    def _last(sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        return positions.expand_as(sequence).masked_fill(sequence.eq(0), -1).amax(-1)

    @classmethod
    def _final(cls, features, sequence):
        last = cls._last(sequence)
        rows = torch.arange(sequence.shape[0], device=sequence.device)
        return features[rows, last.clamp_min(0)].masked_fill(last.lt(0)[:, None], 0)

    def _scores(self, features, candidates):
        # ModelView's paired-candidate contract is B x 1 x D / B x 1 x 2.
        scores = self.view.predict_score(features[:, None, :], candidates[:, None, :])
        expected = (features.shape[0], 1, 2)
        if tuple(scores.shape) != expected:
            raise ValueError(
                f"ModelView returned {tuple(scores.shape)}; expected {expected}"
            )
        return scores[:, 0, :].float()

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        return {
            "count": int(finite.size),
            "nonfinite_count": int(values.size - finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "p10": float(np.quantile(finite, 0.1)) if finite.size else None,
            "reason": None if finite.size else "No finite eligible observations",
        }

    @classmethod
    def _margin_summary(cls, values):
        result = cls._summary(values)
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        result.update(
            {
                "negative_wins_rate": (
                    float((values < 0).mean()) if values.size else None
                ),
                "small_margin_rate": (
                    float((values < 0.5).mean()) if values.size else None
                ),
                "small_margin_threshold": 0.5,
            }
        )
        return result

    def _audit(self):
        supplied = (
            type(self.model).__name__ == "HSTURec"
            and hasattr(self.model, "pos_emb")
            and hasattr(self.model, "blocks")
            and all(type(b).__name__ == "HSTUBlock" for b in self.model.blocks)
        )
        return {
            "actual_module": type(self.model).__name__,
            "documentation_scope": (
                "Supplied HSTURec/HSTUBlock source; structural name checks only, "
                "not an instrumented proof of an arbitrary adapter's implementation"
            ),
            "supplied_structure_matches": supplied,
            "timestamp_inputs": (
                (
                    "Supplied forward(seq) consumes item IDs only; no timestamp inputs "
                    "or next-position temporal query anchors"
                )
                if supplied
                else None
            ),
            "absolute_positions": (
                ("Learned pos_emb at physical slots 1..L, zero on padding")
                if supplied
                else None
            ),
            "relative_positions": (
                "No explicit relative-position bias" if supplied else None
            ),
            "attention": (
                (
                    "Causal MultiheadAttention; padding keys excluded, padding queries "
                    "have a self-key and their outputs are erased after each block"
                )
                if supplied
                else None
            ),
            "aggregation_scaling": (
                (
                    "Input item embeddings scaled by sqrt(hidden_units); standard "
                    "scaled dot-product softmax attention, then value * SiLU(gate)"
                )
                if supplied
                else None
            ),
            "normalization": (
                (
                    "Pre-attention LayerNorm, pre-FFN LayerNorm, final LayerNorm; "
                    "candidate scores use normalized features/items divided by temperature"
                )
                if supplied
                else None
            ),
            "coordinate_control": self._unavailable(
                "Sequence-only ModelView has no verified API for preserving original "
                "absolute/relative coordinates and temporal query anchors while fully "
                "excluding deleted keys, queries, and normalization contributions; "
                "instrumented identity parity is unverified"
            ),
            "time_specific_probe": self._unavailable(
                "No verified timestamp-aware diagnostic path; do not fabricate timestamps"
            ),
        }

    @torch.no_grad()
    def analyze_embeddings_collapse_cosine(self, sample_size=2048, seed=42):
        emb = self.view.item_emb.weight.detach()[1 : self.model.itemnum + 1]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(emb), size=min(sample_size, len(emb)), replace=False)
        x = emb[torch.as_tensor(idx, device=emb.device, dtype=torch.long)].float()
        norms = x.norm(dim=-1)
        valid = torch.isfinite(x).all(-1) & torch.isfinite(norms) & norms.gt(1e-12)
        x = F.normalize(x[valid], dim=-1).double()
        k = len(x)
        score = None
        if k >= 2:
            # Sum all off-diagonal dot products without constructing a k x k Gram matrix.
            off_diagonal_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (off_diagonal_sum / (k * (k - 1))).clamp(-1, 1)
            score = float((mean_cos + 1) / 2)
        return {
            "value": score,
            "sampled_items": len(idx),
            "usable_items": k,
            "excluded_items": len(idx) - k,
            "ordered_pair_denominator": k * (k - 1),
            "reason": (
                None if score is not None else "Fewer than two finite nonzero vectors"
            ),
        }

    @staticmethod
    def _alter(sequence, condition, replicate, batch_id):
        original = sequence.detach().cpu().numpy()
        altered = np.zeros_like(original)
        eligible = np.zeros(len(original), dtype=bool)
        changed = np.zeros(len(original), dtype=bool)
        for row, padded in enumerate(original):
            items = padded[padded != 0]
            n = len(items)
            # Independent local streams; global training RNG state is untouched.
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        42,
                        batch_id,
                        row,
                        replicate,
                        DiagnosisProbe.CONDITIONS.index(condition),
                    ]
                )
            )
            result = items.copy()
            if condition.startswith("prefix_delete_") and n >= 3:
                k = max(1, (n - 1) // 5)
                if condition == "prefix_delete_oldest":
                    remove = np.arange(k)
                elif condition == "prefix_delete_recent":
                    remove = np.arange(n - 1 - k, n - 1)
                else:
                    remove = rng.choice(n - 1, size=k, replace=False)
                result = np.delete(items, remove)
                eligible[row] = True
            elif condition in ("delete_final", "delete_penultimate") and n >= 2:
                result = np.delete(
                    items, n - 1 if condition == "delete_final" else n - 2
                )
                eligible[row] = True
            elif condition == "shuffle_prefix" and n >= 3:
                result[:-1] = rng.permutation(result[:-1])
                eligible[row] = True
            elif condition == "swap_adjacent_prefix" and n >= 3:
                # Nonoverlapping adjacent swaps, with a locally randomized offset.
                offset = 0 if n == 3 else int(rng.integers(0, 2))
                for j in range(offset, n - 2, 2):
                    result[j], result[j + 1] = result[j + 1], result[j]
                eligible[row] = True
            elif condition == "shuffle_before_suffix":
                suffix = min(3, n - 1) if n else 0
                width = n - suffix
                if width >= 2:
                    result[:width] = rng.permutation(result[:width])
                    eligible[row] = True
            if len(result):
                altered[row, -len(result) :] = result
            changed[row] = not np.array_equal(altered[row], padded)
        return (
            torch.as_tensor(altered, device=sequence.device, dtype=sequence.dtype),
            torch.as_tensor(eligible, device=sequence.device),
            torch.as_tensor(changed, device=sequence.device),
        )

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        was_training = self.model.training
        # Preserve any deliberately mixed submodule modes as well.
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            cache, batches = [], []
            # Clone immediately in case the loader reuses its backing buffers.
            for batch_id, batch in enumerate(islice(train_loader, 10)):
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    batches.append(
                        {"batch": batch_id, "reason": "Expected four-element batch"}
                    )
                    continue
                tensors = tuple(
                    torch.as_tensor(x).detach().cpu().clone() for x in batch[1:]
                )
                if (
                    any(x.ndim != 2 for x in tensors)
                    or any(x.shape != tensors[0].shape for x in tensors)
                    or tensors[0].shape[1] == 0
                ):
                    batches.append(
                        {
                            "batch": batch_id,
                            "reason": "Expected matching B x L tensors, L > 0",
                        }
                    )
                    continue
                cache.append((batch_id, tensors))
            records = {condition: [] for condition in self.CONDITIONS}
            identities, training_margins = [], []
            for batch_id, tensors in cache:
                seq, pos, neg = (x.to(self.device, dtype=torch.long) for x in tensors)
                b, length = seq.shape
                if b == 0:
                    batches.append(
                        {"batch": batch_id, "rows": 0, "reason": "Empty batch"}
                    )
                    continue
                rows = torch.arange(b, device=self.device)
                last = self._last(seq)
                candidates = torch.stack(
                    (pos[rows, last.clamp_min(0)], neg[rows, last.clamp_min(0)]), -1
                )
                base_eligible = last.ge(0) & candidates.ne(0).all(-1)
                features = self.view.log2feats(seq)
                original_features = self._final(features, seq)
                scores = self._scores(original_features, candidates)
                margin = scores[:, 0] - scores[:, 1]
                finite_base = torch.isfinite(scores).all(-1) & torch.isfinite(margin)
                eligible = base_eligible & finite_base
                valid_positions = seq.ne(0) & pos.ne(0) & neg.ne(0)
                logits = self.view.predict_score(features, torch.stack((pos, neg), -1))
                if tuple(logits.shape) != (b, length, 2):
                    raise ValueError(
                        "ModelView position scores must have shape B x L x 2"
                    )
                position_margins = (logits[..., 0] - logits[..., 1]).float()
                training_margins.extend(
                    position_margins[valid_positions].cpu().tolist()
                )
                repeat_features = self._final(self.view.log2feats(seq.clone()), seq)
                repeated = self._scores(repeat_features, candidates)
                checked = base_eligible
                close = torch.isclose(scores, repeated, atol=1e-5, rtol=1e-4).all(-1)
                identity_finite = torch.isfinite(scores).all(-1) & torch.isfinite(
                    repeated
                ).all(-1)
                differences = (scores - repeated).abs()[checked & identity_finite]
                identity_failed = int((checked & ~(close & identity_finite)).sum())
                identities.append(
                    {
                        "batch": batch_id,
                        "checked_rows": int(checked.sum()),
                        "failed_rows": identity_failed,
                        "nonfinite_rows": int((checked & ~identity_finite).sum()),
                        "max_absolute_score_difference": (
                            float(differences.max()) if differences.numel() else None
                        ),
                    }
                )
                batches.append(
                    {
                        "batch": batch_id,
                        "rows": b,
                        "tensor_positions": b * length,
                        "nonpadding_positions": int(seq.ne(0).sum()),
                        "valid_target_positions": int(valid_positions.sum()),
                        "nonfinite_position_margins": int(
                            (valid_positions & ~torch.isfinite(position_margins)).sum()
                        ),
                        "empty_rows": int(last.lt(0).sum()),
                        "nonempty_zero_target_rows": int(
                            (last.ge(0) & ~candidates.ne(0).all(-1)).sum()
                        ),
                        "nonfinite_final_score_or_margin_rows": int(
                            (base_eligible & ~finite_base).sum()
                        ),
                        "eligible_final_rows": int(eligible.sum()),
                    }
                )
                for condition in self.CONDITIONS:
                    for replicate in range(5):
                        altered, condition_eligible, changed = self._alter(
                            seq, condition, replicate, batch_id
                        )
                        eligible_before_identity = eligible & condition_eligible
                        # A failed batch-level control invalidates its intervention contrasts.
                        selected = eligible_before_identity & (identity_failed == 0)
                        record = {
                            "identity_excluded_rows": (
                                int(eligible_before_identity.sum())
                                if identity_failed
                                else 0
                            ),
                            "batch": batch_id,
                            "replicate": replicate,
                            "eligible_rows": int(selected.sum()),
                            "length_ineligible_rows": int(
                                (eligible & ~condition_eligible).sum()
                            ),
                            "identity_control_passed": identity_failed == 0,
                        }
                        if not bool(selected.any()):
                            record.update(
                                {
                                    "pairs": [],
                                    "reason": (
                                        "Batch identity control failed; intervention not evaluated"
                                        if identity_failed
                                        else "No eligible rows"
                                    ),
                                }
                            )
                            records[condition].append(record)
                            continue
                        altered_features = self._final(
                            self.view.log2feats(altered), altered
                        )
                        altered_scores = self._scores(altered_features, candidates)
                        altered_margin = altered_scores[:, 0] - altered_scores[:, 1]
                        finite = torch.isfinite(altered_scores).all(
                            -1
                        ) & torch.isfinite(altered_margin)
                        usable = selected & finite
                        record["nonfinite_score_or_margin_rows"] = int(
                            (selected & ~finite).sum()
                        )
                        cosine_valid = (
                            torch.isfinite(original_features).all(-1)
                            & torch.isfinite(altered_features).all(-1)
                            & original_features.float().norm(dim=-1).gt(1e-12)
                            & altered_features.float().norm(dim=-1).gt(1e-12)
                        )
                        cosine = F.cosine_similarity(
                            original_features.float(), altered_features.float(), dim=-1
                        )
                        pairs = []
                        for row in (
                            usable.nonzero(as_tuple=False).flatten().cpu().tolist()
                        ):
                            before, after = float(margin[row]), float(
                                altered_margin[row]
                            )
                            cos_ok = bool(cosine_valid[row]) and bool(
                                torch.isfinite(cosine[row])
                            )
                            pairs.append(
                                {
                                    "row": row,
                                    "original_margin": before,
                                    "altered_margin": after,
                                    "margin_change": after - before,
                                    "negative_wins": after < 0,
                                    "pairwise_order_flip": int(np.sign(before))
                                    != int(np.sign(after)),
                                    "feature_cosine": (
                                        float(cosine[row]) if cos_ok else None
                                    ),
                                    "cosine_reason": (
                                        None
                                        if cos_ok
                                        else "Nonfinite or zero-norm feature"
                                    ),
                                    "input_changed": bool(changed[row]),
                                }
                            )
                        record["pairs"] = pairs
                        records[condition].append(record)
            interventions = {}
            for condition, condition_records in records.items():
                pairs = [
                    p for record in condition_records for p in record.get("pairs", [])
                ]
                cosines = [
                    p["feature_cosine"]
                    for p in pairs
                    if p["feature_cosine"] is not None
                ]
                interventions[condition] = {
                    "replicates_requested": 5,
                    "replicates_with_eligible_rows": len(
                        {
                            r["replicate"]
                            for r in condition_records
                            if r["eligible_rows"]
                        }
                    ),
                    "eligible_row_replicates": sum(
                        r["eligible_rows"] for r in condition_records
                    ),
                    "identity_excluded_row_replicates": sum(
                        r["identity_excluded_rows"] for r in condition_records
                    ),
                    "reason": (
                        None
                        if pairs
                        else (
                            "No finite eligible pairs after length and identity controls; "
                            "see raw_paired_records for exclusions"
                        )
                    ),
                    "finite_pair_denominator": len(pairs),
                    "original_margins": self._margin_summary(
                        [p["original_margin"] for p in pairs]
                    ),
                    "altered_margins": self._margin_summary(
                        [p["altered_margin"] for p in pairs]
                    ),
                    "margin_changes": self._summary(
                        [p["margin_change"] for p in pairs]
                    ),
                    "pairwise_order_flip_rate": (
                        sum(p["pairwise_order_flip"] for p in pairs) / len(pairs)
                        if pairs
                        else None
                    ),
                    "unchanged_input_rate": (
                        sum(not p["input_changed"] for p in pairs) / len(pairs)
                        if pairs
                        else None
                    ),
                    "feature_cosines": self._summary(cosines),
                    "undefined_cosine_count": len(pairs) - len(cosines),
                    "raw_paired_records": condition_records,
                }
            metrics = {
                "backbone_audit": self._audit(),
                "seed": 42,
                "maximum_cached_batches": 10,
                "cached_batches": len(cache),
                "batch_counts": batches,
                "embedding_collapse_score": self.analyze_embeddings_collapse_cosine(),
                "training_pair_separation": self._margin_summary(training_margins),
                "identity": {
                    "atol": 1e-5,
                    "rtol": 1e-4,
                    "batches": identities,
                    "checked_rows": sum(x["checked_rows"] for x in identities),
                    "failed_rows": sum(x["failed_rows"] for x in identities),
                    "passed": (
                        all(x["failed_rows"] == 0 for x in identities)
                        if any(x["checked_rows"] for x in identities)
                        else None
                    ),
                    "reason": (
                        None
                        if any(x["checked_rows"] for x in identities)
                        else "No eligible rows checked; identity parity is unmeasured"
                    ),
                },
                "interventions": interventions,
                "item_category_breakdown": self._unavailable(
                    "Item categories are unavailable in this task"
                ),
                "user_confidence_intervals": self._unavailable(
                    "Loader identities are None; row-to-user mapping is unverified. "
                    "Repeated rows and replicates are dependent; summaries are descriptive only"
                ),
                "paired_validation": self._unavailable(
                    "Training loader cannot calculate validation ranks, HR@10, or top-10 overlap; "
                    "requires separately authorized fixed-candidate host evaluation"
                ),
                "predeclared_follow_up_sequence": [
                    "First resolve identity failures or missingness without changing training.",
                    "If subsequently authorized, evaluate only prefix deletion contrasts under the pinned host protocol.",
                    "Then evaluate only prefix order contrasts under the same fixed users and candidates; no automatic training changes.",
                ],
            }
            return {
                "metrics": metrics,
                "metric_definitions": {
                    "scope": "Conditional on one fitted model and sampled training histories; no retraining or predictor changes.",
                    "counts": "Batch-local row IDs are not user IDs. Exclusions report empty rows, zero targets, length eligibility, and nonfinite values separately.",
                    "embedding_collapse_score": "(Mean sampled off-diagonal embedding cosine + 1)/2; finite nonzero catalog vectors only. A directional geometry statistic, not accuracy or recommendation diversity.",
                    "training_pair_separation": "Untransformed positive-minus-negative scores at nonpadding positions with both targets nonzero. Summary count excludes nonfinite margins; small means margin < 0.5, a scale-dependent threshold.",
                    "fixed_pairs": "Each intervention uses the original final nonpadding position's positive/negative pair. Readout uses masked physical position indices for both original and altered histories, never count-1.",
                    "identity": "Repeated ordinary inference on an unchanged cloned sequence in eval/no_grad mode. Any nonfinite eligible score or tolerance failure invalidates all intervention contrasts from that batch; excluded row-replicates are counted separately. With no checked rows, passed is null with a reason.",
                    "prefix_deletion": "For n >= 3, remove k=max(1,floor(0.2*(n-1))) oldest, most recent, or uniformly sampled prefix events, preserving the final event. Same k within each row.",
                    "endpoint_deletion": "For n >= 2, final-event and penultimate-event removal are separately labeled. Both retain the original candidate pair; final-event removal also changes the readout event.",
                    "order_probes": "Preserve length, padding, final item, and multiset. Shuffle the full prefix, swap disjoint adjacent prefix pairs, or shuffle before a suffix of min(3,n-1) events. At least two permutable events required. Identity draws and repeated item IDs may leave inputs unchanged; these remain in the denominator and are reported.",
                    "coordinate_confounding": "Deletion repacks retained events with left padding at fixed tensor length. It combines information removal with changes in coordinate distances and attention normalization. The supplied absolute embeddings use physical slots. Without instrumented controls, differences do not isolate temporal, positional, or causal preference effects.",
                    "replicates": "Five local seed-42 plans per condition and batch. Deterministic deletions repeat identical subsets; random plans may also repeat. Replicates are not independent fitted models, users, or observations for inference.",
                    "raw_paired_records": "Batch/row/replicate-indexed original and altered raw margins, altered-minus-original change, strict negative-wins indicator, sign-change indicator including ties, and feature cosine. Only finite eligible score pairs are included; missing scores and eligibility counts remain explicit.",
                    "denominators": "Margin and flip summaries use finite_pair_denominator. Cosine summaries use their own finite nonzero-feature count, with null/reason for undefined vectors. Empty measurements have null values and reasons. P10 uses NumPy's interpolated 10th percentile.",
                    "dependence": "Summaries weight sampled row-replicates, not unique users. No user-level bootstrap is justified without verified identity mapping. If later supplied, first aggregate histories/replicates within users, then bootstrap users within categories using 1000 seed-42 resamples; intervals are conditional on this fitted model and fixed candidates, not training-seed uncertainty.",
                    "validation": "Only unchanged host evaluation selects models using four-category macro HR@10, with its 200 users/category, target, 99 unique negatives, eligibility, and seed 42 fixed across any authorized paired comparisons. Training-pair flips are not validation rank changes.",
                    "unavailable_attributes": "No ratings, review text, item categories, prices, stores, or simulator traits are consumed. Public metadata/timestamps are not synthesized.",
                },
            }
        finally:
            self.model.train(was_training)
            for module, training in modes:
                module.training = training


### <<< Self_EvolveRec-BLOCK-END
