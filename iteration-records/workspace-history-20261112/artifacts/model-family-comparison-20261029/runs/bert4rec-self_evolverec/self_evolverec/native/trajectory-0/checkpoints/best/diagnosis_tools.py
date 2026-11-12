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
    - Separate consumed-pipeline consistency, synthetic calibration, training recovery, deployment parity and host validation.
    - Inspect assertion denominators, verified injections, equivalent and unresolved cases, clean false positives and unavailable reasons.
    - Passing bounded replay does not certify the entire pipeline, gradients, convergence, deployed checkpoints or held-out gains.
    - Repeated item IDs do not establish exposure of the same target event. Surviving mutations do not establish equivalence.
    - Retain exactly one independently trained BERT4Rec checkpoint per category; do not propose independent predictor fusion.
    - Separate training catalog recovery, exact expected sampled HR10, realized calibration draws, filtering effects, deployment parity and host validation.
    - Inspect eligibility, N/K/m, repeat exclusions, completed users, precision, failures and diagnostic cost before interpreting catalog results.
    - Expected sampled hits on catalog misses are mathematically possible and are not themselves implementation bugs.
    - A better full-history-filtered rank follows from removing competitors; it does not demonstrate a modeling improvement.
    - Partial or failed catalog audits do not certify the selected cohort; unavailable fixtures remain unknown.
    ### <<< Self_EvolveRec-BLOCK-END
    """


### >>> Self_EvolveRec-BLOCK-START: Bounded consumed-event replay with explicit availability and verified restoration


def audit_consumed_records(records, histories, itemnum):
    """CPU-only checks; never call the producer or run another model forward."""
    if records is None:
        return {
            "available": False,
            "reason": "No explicit consumed-record side channel supplied",
        }
    if not isinstance(records, (list, tuple)):
        raise TypeError("consumed_records must be an explicit bounded list or tuple")
    histories = histories.get("History", histories)
    counts, failures, loss_errors = Counter(), [], []
    record_metadata = []

    def check(record):
        checks = {}
        try:
            sequence = np.asarray(record["sequence"], dtype=np.int64)
            positive = np.asarray(record["positive"], dtype=np.int64)
            negative = np.asarray(record["negative"], dtype=np.int64)
            if negative.ndim == 2:
                negative = negative[..., None]
            if (
                sequence.ndim != 2
                or positive.shape != sequence.shape
                or negative.shape[:2] != sequence.shape
                or negative.ndim != 3
                or negative.shape[-1] < 1
                or len(record["trace"]) != len(sequence)
            ):
                raise ValueError("unaligned record shapes")
            length = sequence.shape[1]
            expected_seq = np.zeros_like(sequence)
            expected_pos = np.zeros_like(positive)
            expected_neg = np.zeros_like(negative)
            for row, trace in enumerate(record["trace"]):
                user = trace["user"]
                history = histories[user] if user in histories else histories[str(user)]
                history = [int(i) for i in history]
                start, stop = trace["start"], trace["stop"]
                if not (
                    0 <= start < stop <= len(history)
                    and stop >= 2
                    and start == max(0, stop - length)
                ):
                    raise ValueError("invalid crop bounds")
                if any(i < 1 or i > itemnum for i in history):
                    raise ValueError("invalid public history IDs")
                selected = trace["selected"]
                offset = length - (stop - start)
                source = [start + p - offset for p in selected]
                if (
                    not selected
                    or len(set(selected)) != len(selected)
                    or any(p < offset or p >= length for p in selected)
                    or source != trace["source_events"]
                    or trace["original_labels"] != [history[i] for i in source]
                    or len(trace["negative_seeds"]) != len(selected)
                    or trace["policy"] != "selected_targets_always_reserved_mask"
                ):
                    raise ValueError("invalid selected-event trace")
                branch = trace["branch"]
                if branch not in {"terminal", "random"}:
                    raise ValueError("unknown corruption branch")
                if branch == "terminal" and selected != [length - 1]:
                    raise ValueError("terminal target must be final source event")
                expected_seq[row, offset:] = history[start:stop]
                excluded = set(history)
                if len(excluded) >= itemnum:
                    raise ValueError("no eligible negatives")
                for p, event, seed in zip(selected, source, trace["negative_seeds"]):
                    expected_pos[row, p] = history[event]
                    expected_seq[row, p] = itemnum + 1
                    rng = random.Random(seed)
                    for j in range(negative.shape[-1]):
                        candidate = rng.randint(1, itemnum)
                        while candidate in excluded:
                            candidate = rng.randint(1, itemnum)
                        expected_neg[row, p, j] = candidate
                # Event identities, rather than item-ID inequality, establish
                # which occurrences remain visible. Earlier repeats are legal.
                visible_events = set(range(start, stop)) - set(source)
                if branch == "terminal" and any(i >= source[0] for i in visible_events):
                    raise ValueError("terminal target or suffix event exposed")
            checks["event_trace"] = True
            checks["sequence_replay"] = bool(np.array_equal(sequence, expected_seq))
            checks["same_event_labels"] = bool(np.array_equal(positive, expected_pos))
            checks["negative_replay"] = bool(np.array_equal(negative, expected_neg))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            checks["event_trace"] = False
            checks["reason"] = str(error)
        return checks

    def passed(checks):
        return all(
            checks.get(k) is True
            for k in (
                "event_trace",
                "sequence_replay",
                "same_event_labels",
                "negative_replay",
            )
        )

    calibration = {
        name: Counter() for name in ("shift_label_index", "restore_masked_target")
    }
    controls = Counter()
    for index, record in enumerate(records[:4]):
        if not isinstance(record, dict):
            raise TypeError("Each consumed record must be a dictionary")
        record_metadata.append(
            {
                key: record.get(key)
                for key in (
                    "category",
                    "epoch",
                    "batch_index",
                    "forward_step",
                    "update_outcome",
                )
            }
        )
        checks = check(record)
        for name, value in checks.items():
            if name == "reason":
                continue
            counts[name + ":checked"] += 1
            counts[name + ":passed"] += int(value)
            if not value:
                failures.append({"record": index, "assertion": name})
        payload = record.get("loss")
        if payload is None:
            counts["actual_loss_wiring:unavailable"] += 1
        else:
            try:
                p = torch.tensor(payload["positive_scores"], dtype=torch.float64)
                n = torch.tensor(payload["negative_scores"], dtype=torch.float64)
                selected = np.argwhere(np.asarray(record["positive"]) != 0).tolist()
                if (
                    p.ndim != 1
                    or n.ndim != 2
                    or n.shape[0] != len(p)
                    or n.shape[1] < 1
                    or not len(p)
                    or selected != payload["selected_indices"]
                    or len(selected) != len(p)
                    or payload["reduction"] != "pooled_valid_target_mean"
                    or payload["auxiliary_loss"] != 0
                    or payload["regularization_in_scalar"] != 0
                ):
                    raise ValueError("unsupported or misaligned loss specification")
                if not bool(torch.isfinite(p).all() & torch.isfinite(n).all()):
                    raise ValueError("Nonfinite captured same-forward logits")
                differences = n - p[:, None]
                if (
                    payload["objective"] == "ce"
                    and payload["negative_reduction"] == "logsumexp"
                ):
                    terms = torch.logsumexp(
                        torch.cat((torch.zeros_like(p[:, None]), differences), 1), 1
                    )
                elif (
                    payload["objective"] == "bpr"
                    and payload["negative_reduction"] == "mean"
                ):
                    terms = torch.logaddexp(
                        torch.zeros_like(differences), differences
                    ).mean(1)
                else:
                    raise ValueError("unknown objective")
                recorded_terms = torch.tensor(
                    payload["target_losses"], dtype=torch.float64
                )
                actual = float(payload["actual_loss"])
                expected = float(terms.mean())
                dtype = payload["dtype"]
                tolerance = {
                    "torch.float64": 1e-10,
                    "torch.float32": 1e-5,
                    "torch.float16": 5e-3,
                    "torch.bfloat16": 5e-2,
                }[dtype]
                agreement = (
                    recorded_terms.shape == terms.shape
                    and bool(torch.isfinite(terms).all())
                    and np.isfinite(actual)
                    and bool(
                        torch.allclose(
                            terms, recorded_terms, rtol=tolerance, atol=tolerance
                        )
                    )
                    and abs(expected - actual) <= tolerance * (1 + abs(expected))
                )
                loss_errors.append(
                    {
                        "record": index,
                        "absolute": abs(expected - actual),
                        "relative": abs(expected - actual) / max(abs(expected), 1e-12),
                        "rtol": tolerance,
                        "atol": tolerance,
                        "targets": len(p),
                    }
                )
                counts["actual_loss_wiring:checked"] += 1
                counts["actual_loss_wiring:passed"] += int(agreement)
                # This verifies the captured selection metadata. It cannot
                # independently certify absent terms in the executed loss graph.
                counts["loss_selection_metadata:checked"] += 1
                counts["loss_selection_metadata:passed"] += int(
                    selected == payload["selected_indices"]
                )
                counts["ignored_positions_zero_direct_terms:unavailable"] += 1
                if not agreement:
                    failures.append(
                        {"record": index, "assertion": "actual_loss_wiring"}
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                counts["actual_loss_wiring:unavailable"] += 1
                failures.append(
                    {
                        "record": index,
                        "assertion": "loss_record_invalid",
                        "reason": str(error),
                    }
                )
        # Predeclared calibration subset: first retained record only.
        if index != 0:
            continue
        import copy

        if not passed(checks):
            controls["unresolved_clean_records"] += 1
            continue
        controls["verified_clean_controls"] += 1
        controls["false_positives"] += int(not passed(check(record)))
        # Restrict this control to event/tensor assertions. Historical loss
        # payload ordering is deliberately excluded from the control's scope.
        harmless = copy.deepcopy(record)
        harmless.pop("loss", None)
        keys = ("sequence", "positive", "negative", "trace")
        for key in keys:
            harmless[key] = list(reversed(harmless[key]))
        inverse = copy.deepcopy(harmless)
        for key in keys:
            inverse[key] = list(reversed(inverse[key]))
        if not all(inverse[key] == record[key] for key in keys):
            controls["invalid_row_permutation_controls"] += 1
        else:
            controls["verified_row_permutation_controls"] += 1
            controls["false_positives"] += int(not passed(check(harmless)))
        for operator, report in calibration.items():
            report["attempted"] += 1
            mutant = copy.deepcopy(record)
            trace = mutant["trace"][0]
            position = trace["selected"][0]
            original = record["positive"][0][position]
            if operator == "restore_masked_target":
                if record["sequence"][0][position] != itemnum + 1:
                    report["inapplicable"] += 1
                    continue
                mutant["sequence"][0][position] = original
                verified = original != itemnum + 1 and 1 <= original <= itemnum
                assertion = "sequence_replay"
            else:
                user = trace["user"]
                history = histories[user] if user in histories else histories[str(user)]
                event = trace["source_events"][0]
                other_event = event - 1 if event > 0 else event + 1
                if not 0 <= other_event < len(history):
                    report["inapplicable"] += 1
                    continue
                changed = int(history[other_event])
                if changed == original:
                    report["proven_equivalent"] += 1
                    continue
                mutant["positive"][0][position] = changed
                verified = 1 <= changed <= itemnum and changed != original
                assertion = "same_event_labels"
            if not verified:
                report["unresolved"] += 1
                continue
            field = "sequence" if operator == "restore_masked_target" else "positive"
            before = record[field][0][position]
            after = mutant[field][0][position]
            # Verify an isolated, reversible change without consulting the
            # detector. Equality with a copy of the original proves nothing.
            restored = copy.deepcopy(mutant)
            restored[field][0][position] = before
            if before == after or restored != record:
                report["invalid"] += 1
                continue
            report.setdefault("injections", []).append(
                {
                    "record": index,
                    "row": 0,
                    "position": position,
                    "field": field,
                    "before": before,
                    "after": after,
                    "violated_assertion": assertion,
                    "restoration_exact": True,
                }
            )
            report["verified_corruption"] += 1
            outcome = check(mutant).get(assertion)
            if outcome is None:
                report["detector_unavailable"] += 1
            detected = outcome is False
            report["detected"] += int(detected)
            report["surviving"] += int(not detected)
            report["restoration_exact"] += 1
    result_calibration = {}
    for name, report in calibration.items():
        summary = {
            key: int(report[key])
            for key in (
                "attempted",
                "inapplicable",
                "invalid",
                "proven_equivalent",
                "unresolved",
                "verified_corruption",
                "detected",
                "surviving",
            )
        }
        summary["restoration_exact"] = int(report["restoration_exact"])
        summary["detector_unavailable"] = int(report["detector_unavailable"])
        summary["injections"] = report.get("injections", [])
        denominator = report["verified_corruption"]
        summary["detection_coverage"] = (
            report["detected"] / denominator if denominator else None
        )
        summary["assertion"] = (
            "same_event_labels" if name == "shift_label_index" else "sequence_replay"
        )
        result_calibration[name] = summary
    denominator = (
        controls["verified_clean_controls"]
        + controls["verified_row_permutation_controls"]
    )
    return {
        "available": bool(records),
        "records": min(len(records), 4),
        "reason": None if records else "Explicit consumed-record buffer is empty",
        "record_metadata": record_metadata,
        "capture_schedule": "first two batches of first and final executed epochs; deduplicated",
        "scope": "bounded consumed batches; trace consistency, not authenticity or exhaustive certification",
        "control_scope": "event/tensor assertions only; clean validity is conditional on public-history replay, not an independent correctness oracle",
        "ignored_positions_zero_direct_terms_reason": "Selection metadata alone does not independently certify every contribution to the executed scalar loss",
        "assertion_counts": dict(counts),
        "failures": failures,
        "loss_errors": loss_errors,
        "calibration": result_calibration,
        "controls": dict(controls),
        "clean_false_positive_rate": (
            controls["false_positives"] / denominator if denominator else None
        ),
        "unsupported_edge_cases": [
            "empty training windows",
            "singleton training windows",
        ],
        "deployment_parity": None,
        "host_validation_gain": None,
    }


### <<< Self_EvolveRec-BLOCK-END


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
            ### >>> Self_EvolveRec-BLOCK-START: Undefined collapse is not a healthy zero
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Linear-memory embedding alignment with double accumulation
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
        # Retained for compatibility; catalog recovery uses the same final
        # training-event prefixes without changing the trained network.
        return self._catalog_recovery()

    def _catalog_recovery(self):
        import time
        from itertools import combinations

        # DEBUG: Use standard-library exact combinations; SciPy is unavailable.
        from math import comb

        started = time.perf_counter()
        device_type = self.network.item_emb.weight.device.type
        report = {
            "available": False,
            "reason": None,
            "exposure": "training-exposed catalog recovery; not host validation",
            "category": getattr(self.view, "dataset_name", None),
            "user_limit": 50,
            "time_limit_seconds": 30.0,
            "chunk_size": 4096,
            "completed_users": 0,
            "catalog_chunks": 0,
            "rows": [],
            "failures": [],
            "equal_category_macro": None,
            "equal_category_macro_reason": "Only one category ModelView is available",
            "fresh_process_predictor_parity": None,
            "fresh_process_predictor_parity_reason": "This audit exercises live ModelView scoring only",
        }
        old_precision = torch.get_float32_matmul_precision()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        if device_type == "cuda":
            torch.cuda.synchronize(self.network.item_emb.weight.device)
        gpu_start = time.perf_counter()
        rows = report["rows"]
        try:
            if device_type not in {"cpu", "cuda"}:
                raise ValueError("Unsupported precision-control device")
            if any(
                p.is_floating_point() and p.dtype != torch.float32
                for p in self.network.parameters()
            ):
                raise ValueError(
                    "FP32 encoder and scorer parameters are required; no parameter casting performed"
                )
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report["precision"] = {
                "encoder_parameter_dtype": str(self.network.item_emb.weight.dtype),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "autocast": False,
                "device": str(self.network.item_emb.weight.device),
                "scorer": "ModelView.predict_score",
                "final_chunk": "unmodified remainder",
                "tie_rule": "descending score, ascending mapped item ID; exact equality",
            }
            if report["precision"]["matmul_tf32"]:
                raise ValueError("IEEE FP32 matmul control was not honored")

            def expectation(N, K, m):
                if not 0 <= K <= N or not 0 <= m <= N:
                    raise ValueError("Invalid hypergeometric counts")
                # DEBUG: Compute P(X <= 9) without SciPy using exact integer
                # hypergeometric weights. Convert only the final ratio to float
                # to avoid overflow or cancellation in intermediate probabilities.
                lower = max(0, m - (N - K))
                upper = min(m, K)
                if lower > 9:
                    value = 0.0
                elif upper <= 9:
                    # Includes N=0, m=0 and every guaranteed sampled hit.
                    value = 1.0
                else:
                    numerator = sum(
                        comb(K, x) * comb(N - K, m - x) for x in range(lower, 10)
                    )
                    value = numerator / comb(N, m)
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid hypergeometric probability")
                return value

            # Exhaustive finite-population fixtures include nontrivial misses.
            fixture_count = 0
            for N, K, m in ((0, 0, 0), (5, 3, 4), (12, 10, 10), (12, 12, 10)):
                subsets = list(combinations(range(N), m))
                exact = sum(sum(i < K for i in s) <= 9 for s in subsets) / len(subsets)
                if abs(expectation(N, K, m) - exact) > 1e-12:
                    raise ValueError("Hypergeometric enumeration fixture failed")
                fixture_count += 1
            report["fixtures"] = {
                "hypergeometric_enumerations_passed": fixture_count,
                "live_padding_and_mask_fixtures": "reported separately in bert4rec_integrity",
                "cross_chunk_score_and_rank_fixtures": [],
                "cross_chunk_score_and_rank_reason": "Pending first completed encoder fixture; bounded checks do not certify universal numerical invariance",
            }
            itemnum, length = int(self.network.itemnum), int(self.network.maxlen)
            if itemnum < 1 or length < 2:
                raise ValueError("Require a real-item catalog and maxlen >= 2")
            histories = self.user_train.get("History", self.user_train)
            eligible, frequency = [], Counter()
            invalid_users = short_users = 0
            for user, events in histories.items():
                events = [int(x) for x in events]
                if any(x < 1 or x > itemnum for x in events):
                    invalid_users += 1
                    continue
                frequency.update(events)
                if len(events) < 2:
                    short_users += 1
                else:
                    eligible.append((str(user), events))
            eligible.sort(key=lambda pair: pair[0])
            if len({u for u, _ in eligible}) != len(eligible):
                raise ValueError("Canonical string user IDs are not unique")
            report.update(
                {
                    "eligible_users": len(eligible),
                    "invalid_users": invalid_users,
                    "short_users": short_users,
                    "canonical_order": "lexicographic string user ID",
                    "selection_seed": 42,
                }
            )
            selected = random.Random(42).sample(eligible, min(50, len(eligible)))
            length_cuts = (
                np.quantile(
                    [min(len(h) - 1, length - 1) for _, h in selected],
                    [0.25, 0.5, 0.75],
                ).tolist()
                if selected
                else []
            )
            frequency_cuts = np.quantile(
                [frequency[i] for i in range(1, itemnum + 1)], [0.25, 0.5, 0.75]
            ).tolist()
            report["quartiles"] = {
                "retained_length": length_cuts,
                "training_frequency": frequency_cuts,
                "boundary": "numpy.searchsorted(side='right'); tied cuts retained",
            }
            report["selected_users"] = [u for u, _ in selected]
            report["interrupted"] = False
            with torch.autocast(device_type=device_type, enabled=False):
                for user, events in selected:
                    if time.perf_counter() - started >= 30.0:
                        report["interrupted"] = True
                        break
                    H, target = events[:-1], events[-1]
                    W = H[-(length - 1) :]
                    tokens = np.asarray([self._pack_inference(W)], dtype=np.int64)
                    features = self.view.log2feats(tokens)[:, length - 1, :]
                    if features.dtype != torch.float32 or not bool(
                        torch.isfinite(features).all()
                    ):
                        raise ValueError("Encoder did not return finite FP32 features")
                    report["encoder_output_dtype"] = str(features.dtype)
                    if not report["fixtures"]["cross_chunk_score_and_rank_fixtures"]:
                        # Exercise the actual scorer without changing parameters.
                        # Ascending IDs define exact-tie ordering independently
                        # of sort implementation and candidate input order.
                        fixture_ids = torch.arange(
                            1,
                            min(itemnum, 17) + 1,
                            device=features.device,
                            dtype=torch.long,
                        )

                        def fixture_scores(ids):
                            values = self.view.predict_score(features, ids[None, :])
                            if (
                                values.shape != (1, len(ids))
                                or values.dtype != torch.float32
                                or not bool(torch.isfinite(values).all())
                            ):
                                raise ValueError("Invalid live scorer fixture output")
                            return values[0].detach().cpu().numpy()

                        reference = fixture_scores(fixture_ids)
                        mapped_ids = fixture_ids.cpu().numpy()
                        reference_order = np.lexsort((mapped_ids, -reference))
                        reference_ranks = np.empty(len(mapped_ids), dtype=np.int64)
                        reference_ranks[reference_order] = np.arange(
                            1, len(mapped_ids) + 1
                        )
                        for width in (1, 4, 8):
                            chunked = np.concatenate(
                                [
                                    fixture_scores(fixture_ids[a : a + width])
                                    for a in range(0, len(fixture_ids), width)
                                ]
                            )
                            order = np.lexsort((mapped_ids, -chunked))
                            ranks = np.empty(len(mapped_ids), dtype=np.int64)
                            ranks[order] = np.arange(1, len(mapped_ids) + 1)
                            tolerance = 1e-5 + 1.3e-6 * np.abs(reference)
                            gaps = np.abs(reference[:, None] - reference[None, :])
                            separated = gaps > (tolerance[:, None] + tolerance[None, :])
                            reversed_pairs = (
                                (reference[:, None] > reference[None, :])
                                != (chunked[:, None] > chunked[None, :])
                            ) & separated
                            record = {
                                "items": len(mapped_ids),
                                "chunk_size": width,
                                "final_chunk_size": (len(mapped_ids) - 1) % width + 1,
                                "score_allclose": bool(
                                    np.allclose(
                                        reference, chunked, rtol=1.3e-6, atol=1e-5
                                    )
                                ),
                                "max_abs_score_error": float(
                                    np.abs(reference - chunked).max()
                                ),
                                "exact_rank_agreement": bool(
                                    np.array_equal(reference_ranks, ranks)
                                ),
                                "exact_top10_agreement": bool(
                                    np.array_equal(reference_order[:10], order[:10])
                                ),
                                "changed_rank_count": int(
                                    np.count_nonzero(reference_ranks != ranks)
                                ),
                                "separated_pair_reversals": int(reversed_pairs.sum()),
                                "rtol": 1.3e-6,
                                "atol": 1e-5,
                            }
                            report["fixtures"][
                                "cross_chunk_score_and_rank_fixtures"
                            ].append(record)
                            if (
                                not record["score_allclose"]
                                or record["separated_pair_reversals"]
                            ):
                                raise ValueError("Live chunk scorer fixture failed")
                        report["fixtures"]["cross_chunk_score_and_rank_reason"] = (
                            "Measured on first selected training prefix and up to 17 real IDs; "
                            "near-tie rank changes are reported, not hidden by score tolerance. "
                            "Synthetic duplicate-embedding and injected-nonfinite fixtures remain unmeasured."
                        )
                    exclusions = {
                        "full_history_target_restored": set(H) - {target},
                        "window_target_restored": set(W) - {target},
                    }
                    # Small draw storage only; no complete score vector retained.
                    draws = {}
                    for policy, excluded in exclusions.items():
                        N = itemnum - len(excluded) - 1
                        m = min(99, N)
                        for seed in (42, 43, 44):
                            rng = random.Random(f"{seed}:{user}:{policy}")
                            if N < 198:
                                pool = [
                                    i
                                    for i in range(1, itemnum + 1)
                                    if i != target and i not in excluded
                                ]
                                chosen = rng.sample(pool, m)
                            else:
                                chosen = set()
                                while len(chosen) < m:
                                    candidate = rng.randint(1, itemnum)
                                    if (
                                        candidate != target
                                        and candidate not in excluded
                                    ):
                                        chosen.add(candidate)
                                chosen = sorted(chosen)
                            draws[(policy, seed)] = {i: None for i in chosen}
                    # Preindex exclusions and calibration IDs once per user.
                    # This avoids scanning full histories and all draws for
                    # every catalog chunk while preserving identical scoring.
                    excluded_ids = {
                        policy: np.asarray(sorted(excluded), dtype=np.int64)
                        for policy, excluded in exclusions.items()
                    }
                    draw_ids = {
                        key: np.asarray(sorted(draw), dtype=np.int64)
                        for key, draw in draws.items()
                    }
                    target_score = None
                    K = {policy: 0 for policy in exclusions}
                    complete = True
                    for pass_index in range(2):
                        for start in range(1, itemnum + 1, 4096):
                            if time.perf_counter() - started >= 30.0:
                                complete = False
                                break
                            ids = torch.arange(
                                start,
                                min(start + 4096, itemnum + 1),
                                device=features.device,
                                dtype=torch.long,
                            )
                            # B x K IDs avoid the one-item scoring ambiguity.
                            scores = self.view.predict_score(features, ids.unsqueeze(0))
                            report["catalog_chunks"] += 1
                            if (
                                scores.shape != (1, len(ids))
                                or scores.dtype != torch.float32
                            ):
                                raise ValueError(
                                    "Scorer must return FP32 scores with shape 1 x K"
                                )
                            scores = scores[0].detach().cpu().numpy()
                            if not np.isfinite(scores).all():
                                raise ValueError(
                                    "Nonfinite catalog score before filtering"
                                )
                            if start <= target < start + len(ids):
                                observed = float(scores[target - start])
                                if pass_index == 0:
                                    target_score = observed
                                elif observed != target_score:
                                    raise ValueError(
                                        "Target score changed between identical catalog passes"
                                    )
                            if pass_index == 0:
                                continue
                            chunk_ids = np.arange(start, start + len(scores))
                            if target_score is None:
                                raise ValueError(
                                    "First catalog pass did not capture the target"
                                )
                            ahead = (scores > target_score) | (
                                (scores == target_score) & (chunk_ids < target)
                            )
                            ahead &= chunk_ids != target
                            stop = start + len(scores)
                            for policy, excluded in excluded_ids.items():
                                filtered = ahead.copy()
                                lo, hi = np.searchsorted(excluded, [start, stop])
                                filtered[excluded[lo:hi] - start] = False
                                K[policy] += int(filtered.sum())
                            for key, selected_ids in draw_ids.items():
                                lo, hi = np.searchsorted(selected_ids, [start, stop])
                                for item in selected_ids[lo:hi]:
                                    draws[key][int(item)] = float(scores[item - start])
                        if not complete:
                            break
                    if not complete:
                        report["interrupted"] = True
                        break
                    if K["full_history_target_restored"] > K["window_target_restored"]:
                        raise ValueError("History/window rank subset invariant failed")
                    stratum = (
                        "repeat_within_window"
                        if target in W
                        else (
                            "repeat_only_outside_window"
                            if target in H
                            else "unseen_in_history"
                        )
                    )
                    row = {
                        "user": user,
                        "target_event_index": len(events) - 1,
                        "target": target,
                        "repeat_stratum": stratum,
                        "length_quartile": int(
                            np.searchsorted(length_cuts, len(W), side="right")
                        ),
                        "frequency_quartile": int(
                            np.searchsorted(
                                frequency_cuts, frequency[target], side="right"
                            )
                        ),
                        "policies": {},
                    }
                    for policy, excluded in exclusions.items():
                        N, k = itemnum - len(excluded) - 1, K[policy]
                        m = min(99, N)
                        probability = expectation(N, k, m)
                        hit = int(k <= 9)
                        realized = {}
                        for seed in (42, 43, 44):
                            draw = draws[(policy, seed)]
                            if any(score is None for score in draw.values()):
                                raise ValueError(
                                    "Incomplete retained calibration scores"
                                )
                            rank = 1 + sum(
                                score > target_score
                                or (score == target_score and item < target)
                                for item, score in draw.items()
                            )
                            if hit and rank > 10:
                                raise ValueError(
                                    "Catalog hit lost under subset sampling"
                                )
                            realized[str(seed)] = {
                                "rank": rank,
                                "hr10": int(rank <= 10),
                                "candidate_count": m + 1,
                            }
                        row["policies"][policy] = {
                            "N": N,
                            "K": k,
                            "m": m,
                            "rank": k + 1,
                            "support": [max(0, m - (N - k)), min(m, k)],
                            "catalog_hr10": hit,
                            "expected_sampled_hr10": probability,
                            "sampled_minus_catalog": probability - hit,
                            "expected_sampled_hit_catalog_miss": (1 - hit)
                            * probability,
                            "draws": realized,
                        }
                    rows.append(row)
            report["completed_users"] = len(rows)
            fields = (
                "catalog_hr10",
                "expected_sampled_hr10",
                "sampled_minus_catalog",
                "expected_sampled_hit_catalog_miss",
                "rank",
                "N",
                "m",
            )

            def summarize(cohort):
                return {
                    policy: {
                        field: self._summary(
                            [r["policies"][policy][field] for r in cohort]
                        )
                        for field in fields
                    }
                    for policy in (
                        "full_history_target_restored",
                        "window_target_restored",
                    )
                }

            # Each row retains both policies and every calibration draw.
            # Applying the same seeded bootstrap indices preserves pairing.
            report["bootstrap"] = {
                "seed": 42,
                "resamples": 1000,
                "unit": "completed user; policies and draws kept together",
                "scope": "conditional on completed cohort, fitted model and eligibility policies",
            }
            report["summary_scope"] = (
                "completed subset only" if report["interrupted"] else "selected cohort"
            )
            report["selected_users_count"] = len(selected)
            report["uncompleted_users_count"] = len(selected) - len(rows)
            report["draw_summaries"] = (
                {
                    policy: {
                        str(seed): {
                            metric: self._summary(
                                [
                                    r["policies"][policy]["draws"][str(seed)][metric]
                                    for r in rows
                                ]
                            )
                            for metric in ("rank", "hr10", "candidate_count")
                        }
                        for seed in (42, 43, 44)
                    }
                    for policy in exclusions
                    if rows
                }
                if rows
                else {}
            )
            report["calibration_variability"] = {
                policy: {
                    "rank_range": self._summary(
                        [
                            max(
                                d["rank"]
                                for d in r["policies"][policy]["draws"].values()
                            )
                            - min(
                                d["rank"]
                                for d in r["policies"][policy]["draws"].values()
                            )
                            for r in rows
                        ]
                    ),
                    "hit_flip_rate": self._summary(
                        [
                            float(
                                len(
                                    {
                                        d["hr10"]
                                        for d in r["policies"][policy]["draws"].values()
                                    }
                                )
                                > 1
                            )
                            for r in rows
                        ]
                    ),
                }
                for policy in ("full_history_target_restored", "window_target_restored")
            }
            report["summary"] = summarize(rows)
            report["repeat_strata"] = {
                s: summarize([r for r in rows if r["repeat_stratum"] == s])
                for s in (
                    "unseen_in_history",
                    "repeat_within_window",
                    "repeat_only_outside_window",
                )
            }
            report["repeat_stratum_counts"] = {
                name: sum(r["repeat_stratum"] == name for r in rows)
                for name in (
                    "unseen_in_history",
                    "repeat_within_window",
                    "repeat_only_outside_window",
                )
            }
            report["strict_unseen_excluded_repeated_targets"] = sum(
                r["repeat_stratum"] != "unseen_in_history" for r in rows
            )
            report["strict_unseen_denominator"] = report["repeat_stratum_counts"][
                "unseen_in_history"
            ]
            report["quartile_summaries"] = {
                field: {
                    str(q): summarize([r for r in rows if r[field] == q])
                    for q in range(4)
                }
                for field in ("length_quartile", "frequency_quartile")
            }
            report["paired_policy_rank_difference"] = self._summary(
                [
                    r["policies"]["window_target_restored"]["rank"]
                    - r["policies"]["full_history_target_restored"]["rank"]
                    for r in rows
                ]
            )
            report["cohort_complete"] = len(rows) == len(selected) and bool(selected)
            report["available"] = bool(rows) and report["cohort_complete"]
            report["completed_subset_available"] = bool(rows)
            report["reason"] = (
                "Partial cohort: time limit reached"
                if report["interrupted"]
                else None if rows else "No eligible completed users"
            )
        except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as error:
            report["available"] = False
            report["reason"] = str(error)
            report["failures"].append(str(error))
            report["completed_users"] = len(rows)
            report["cohort_complete"] = False
            report["completed_subset_available"] = bool(rows)
            report["summary_scope"] = (
                "failed audit; retained rows are incomplete observations"
            )
        finally:
            torch.set_float32_matmul_precision(old_precision)
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            if device_type == "cuda":
                torch.cuda.synchronize(self.network.item_emb.weight.device)
            report["synchronized_device_interval_seconds"] = (
                time.perf_counter() - gpu_start
            )
            report["wall_seconds"] = time.perf_counter() - started
            report["cost_scope"] = (
                "Synchronized wall interval including CPU work; not isolated kernel time"
            )
        return report

    def _legacy_history_sensitivity(self):
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
                records = getattr(train_loader, "consumed_records", None)
                if records is None:
                    records = getattr(self.view, "consumed_records", None)
                if records is None:
                    records = getattr(self.network, "consumed_records", None)
                # Mined negatives are selected pool slots, not direct uniform
                # draws. Use the supplied production-path replay for that schema.
                if records and any("pool_ids" in r for r in records[:4]):
                    forward = getattr(self.network.forward, "__func__", None)
                    replay = getattr(forward, "__globals__", {}).get(
                        "audit_mining_records"
                    )
                    if callable(replay):
                        consumed_audit = replay(
                            records,
                            self.user_train.get("History", self.user_train),
                            int(self.network.itemnum),
                        )
                    else:
                        consumed_audit = {
                            "available": False,
                            "reason": "Mined-record replay unavailable; uniform replay is inapplicable",
                        }
                else:
                    consumed_audit = audit_consumed_records(
                        records, self.user_train, int(self.network.itemnum)
                    )
                # Never advance an arbitrary production iterator. Missing
                # capture transport means margins are unavailable for this audit.
                train_loader = []
                if records:
                    # Replay CPU copies only. The diagnostic contract remains
                    # four fields with B x L tensors; margins use negative #1.
                    train_loader = [
                        (
                            None,
                            torch.tensor(r["sequence"]),
                            torch.tensor(r["positive"]),
                            (lambda n: n[..., 0] if n.ndim == 3 else n)(
                                torch.tensor(r["negative"])
                            ),
                        )
                        for r in records[:4]
                    ]
                integrity["sampler_provenance"] = consumed_audit
                integrity["sampler_provenance_reason"] = (
                    None
                    if consumed_audit["available"]
                    else consumed_audit.get("reason", "No captured records")
                )
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
                if breakdown["invalid_target_positions"]:
                    failed_checks.append("masked_target_alignment")
                elif breakdown["inspected_batches"] == 0:
                    unavailable_checks.append("masked_target_alignment")
                if consumed_audit.get("failures"):
                    failed_checks.append("consumed_pipeline_audit")
                if not consumed_audit.get("available"):
                    unavailable_checks.append("consumed_pipeline_audit")
                elif consumed_audit.get("assertion_counts", {}).get(
                    "actual_loss_wiring:unavailable", 0
                ):
                    unavailable_checks.append("actual_loss_wiring")
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
                    "Observed live fixtures and explicitly captured consumed-batch consistency only. "
                    "Missing records or same-forward loss payloads remain unavailable. "
                    "Fresh-process predictor parity, complete gradient wiring and effective "
                    "attention-mask inspection remain uncertified."
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
                    "training_catalog_recovery": sensitivity,
                    "training_history_sensitivity": {
                        "result": None,
                        "reason": "Replaced by bounded catalog recovery; no history interventions executed",
                    },
                    "consumed_pipeline_audit": consumed_audit,
                }
                definitions = {
                    "training_catalog_recovery": (
                        "At most 50 seed-42 selected training users, ordered lexicographically before sampling. "
                        "The final training event is recovered from strictly earlier events, retaining at most L-1 "
                        "events and appending MASK at absolute index L-1. Full-history and retained-window "
                        "filtering both restore the target exactly once. Catalog ranks use descending FP32 "
                        "ModelView scores and ascending mapped IDs for exact ties, under fixed 4096-item "
                        "chunks with an unmodified final remainder. Two passes extract and verify the target "
                        "score and count preceding eligible competitors. N excludes the target, K=rank-1, "
                        "m=min(99,N). Expected sampled HR10 is Hypergeometric(N,K,m).cdf(9), with N=0 "
                        "defined as one. This assumes uniform distinct negatives under the reported eligibility, "
                        "not necessarily the host policy. Sampled-minus-catalog and sampled-hit/catalog-miss "
                        "are paired user quantities. Each summary uses 1000 seed-42 user bootstrap replicates; "
                        "intervals condition on selected users, fitted weights and policies. Optional draws "
                        "42/43/44 reuse catalog-pass scores and are reported separately per user. Repeated "
                        "targets and strict-unseen exclusions are explicit. Partial cohorts, failures, effective "
                        "precision, actual counts and synchronized diagnostic cost are reported. Partial "
                        "cohorts have available=false; any retained summaries describe completed users only. "
                        "Calibration summaries retain seeds separately and report three-draw rank ranges and hit flips. "
                        "Bounded live cross-chunk fixtures compare scores, exact ranks and ordered top10; "
                        "near-tie rank changes remain visible even when scores pass tolerance. Universal "
                        "cross-shape invariance and fresh-process predictor parity remain unmeasured here. "
                        "This is training recovery, not held-out quality, user preference, or a selection metric."
                    ),
                    "consumed_pipeline_audit": (
                        "At most four explicitly supplied production records. Independently reconstructs crops, "
                        "source events, reserved-mask corruption, labels and seeded negative draws from public "
                        "training histories. Same-forward detached logits check pooled CE/BPR scalar reduction "
                        "without a second forward; tolerances are recorded per dtype. No gradient certification. "
                        "Calibration uses only the first retained record, verified changed-label and restored-mask "
                        "faults, plus clean and consistent row-permutation controls. Coverage is detected divided "
                        "by verified non-equivalent corruptions; zero denominators are null. Repeated-ID shifts "
                        "are equivalent for the label-only operator. These bounded controls do not estimate real "
                        "fault prevalence. Captures need not belong to the selected checkpoint. Missing transport, "
                        "deployment parity and held-out gains remain unavailable. No production iterator is "
                        "advanced when captured replay is supplied; worker state and exact restart are not certified."
                    ),
                    "bert4rec_integrity": (
                        "Independent appended-mask fixtures compare exact IDs and live candidate scores "
                        "at rtol=1.3e-6, atol=1e-5. Includes padding perturbation and negative controls. "
                        "False checks require investigation; unavailable checks are not passes. "
                        "Synthetic provenance controls do not certify the actual sampler."
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
                        "Unavailable in this workflow: no history interventions were executed. "
                        "The following describes the retained legacy probe only, not measured results: "
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
                # Review summary: statically traced catalog invocation, scorer
                # shapes, target extraction, exact ties, eligibility subsets,
                # paired summaries, partial-cohort flags and state restoration.
                # No execution, training or validation was performed here.
                # Training, checkpoint architecture and predictor are unchanged.
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
