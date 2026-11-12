import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Reproducible conditional temporal sensitivity diagnostics
import random
import time
from collections import Counter

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
    ### >>> Self_EvolveRec-BLOCK-START: Restrict interpretation to measured conditional training-prefix effects
    - Every statement must be grounded in the provided metric values/definitions and include relevant counts and correctness checks.
    - Temporal effects are conditional in-sample sensitivity of fitted checkpoints to selected training prefixes and fixed candidates. They do not establish generalization, causal user intent, or validation gains.
    - Failed correctness checks invalidate temporal interpretation. Null metrics mean unavailable evidence, not zero effects. Missing categories invalidate four-category estimates.
    - State movement without ranking damage establishes representation sensitivity only.
    - Distinguish measurements from alternative explanations: memorization, distribution shift, redundant histories, and scoring bottlenecks.
    - Do not infer exposure, compatibility, dislikes, current intent, mood, ratings, review text, or simulator traits.
    - Confidence intervals condition on the checkpoint, prefixes, and candidates; they omit training-seed variability and validation-search selection optimism.
    - Only independently measured official macro validation HR@10 selects models. Do not claim an improvement from these diagnostics.
    - Candidate-panel results measure conditional training-prefix candidate-distribution sensitivity. Frequency matching is not an unbiased evaluation correction or proof of popularity bias; frequency is not exposure and unseen candidates may be relevant.
    - Keep fresh five-draw estimates, previous frozen realized sets, optional exact expectations, uniform-only populations and the existing temporal population separate. Missing measurements are not zero effects.
    - Candidate MCSE and selected-user reweighting intervals describe different variation; do not add them. Deterministic user selection supplies no population-coverage guarantee. No retraining or search-selection uncertainty is captured.
    # Review checks: inspected shapes, final-position readout, predictor parity,
    # finite outcomes, RNG restoration, stale-state handling and GPU batch caps.
    # Static review only; no execution or validation improvement is claimed.
    # Four-category aggregation still requires all four labeled category records;
    # a single ModelView cannot produce that measurement independently.
    # Static review summary: checked adapter shapes, maximum-nonzero readout,
    # actual-module score parity, native-dtype ranking, complete five-draw
    # reductions, independent-stream MCSE, explicit missing results, trial-local
    # artifacts and finally-based restoration of modes and global RNG states.
    # Optional exact scoring remains deferred. Training and prediction are
    # unchanged; no execution success or validation improvement is claimed.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and use only available training information
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._observed_maxlen = None
        self._margin_count = 0
        self._margin_frequency = {}

    def _get_categories(self, item_id: int):
        # Item-category metadata is unavailable in this matched task.
        return []

    def _histories(self):
        histories = self.user_train.get("History", self.user_train)
        return {
            u: [int(i) for i in history if int(i) != 0]
            for u, history in histories.items()
        }

    def _features(self, sequence):
        # Preserve the ModelView adapter and native feature dtype.
        features = torch.as_tensor(
            self.model.log2feats(sequence.detach().cpu().numpy()),
            device=sequence.device,
        )
        if features.ndim != 3 or features.shape[:2] != sequence.shape:
            raise ValueError("ModelView.log2feats must return B x L x H")
        if not torch.isfinite(features).all():
            raise FloatingPointError("ModelView returned nonfinite features")
        return features

    # DEBUG: No traceback was supplied, so the original failure is unconfirmed.
    # Normalize candidate scoring to the B x L x K ModelView contract already
    # used by the sampled-margin probe. Do not require the host adapter to
    # implement additional vector or B x K broadcasting conventions.
    def _candidate_scores(self, features, items):
        if features.ndim not in (1, 2):
            raise ValueError("Candidate features must have shape H or B x H")
        single = features.ndim == 1
        states = features.unsqueeze(0) if single else features
        ids = torch.as_tensor(items, dtype=torch.long, device=states.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0).expand(states.size(0), -1)
        if ids.ndim != 2 or ids.size(0) != states.size(0):
            raise ValueError("Candidate IDs must have shape K or B x K")
        position_ids = ids.unsqueeze(1)
        scores = torch.as_tensor(
            self.model.predict_score(states.unsqueeze(1), position_ids),
            device=states.device,
        )
        if scores.shape != position_ids.shape:
            raise ValueError("ModelView candidate scoring must return B x L x K")
        if not torch.isfinite(scores).all():
            raise FloatingPointError("ModelView returned nonfinite candidate scores")
        scores = scores[:, 0, :]
        return scores[0] if single else scores

    def _last(self, features, sequence):
        if sequence.size(1) == 0:
            return features.new_zeros(sequence.size(0), features.size(-1))
        positions = torch.arange(sequence.size(1), device=sequence.device)
        last = positions.expand_as(sequence).masked_fill(sequence.eq(0), -1).amax(1)
        states = features[
            torch.arange(sequence.size(0), device=sequence.device), last.clamp_min(0)
        ]
        return torch.where(last[:, None].ge(0), states, torch.zeros_like(states))

    def _pad(self, histories, width):
        sequence = torch.zeros(
            len(histories), width, dtype=torch.long, device=self.device
        )
        for row, history in enumerate(histories):
            if history:
                sequence[row, -len(history) :] = torch.as_tensor(
                    history, dtype=torch.long, device=self.device
                )
        return sequence

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            # DEBUG: Access architecture attributes through the actual module
            # retained from view.model, without another adapter dereference.
            emb = emb[
                1 : self.module.itemnum + 1
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
        ### >>> Self_EvolveRec-BLOCK-START: Measure actual recurrent model with frequency proxy counts
        self.module.eval()
        frequency = Counter(i for h in self._histories().values() for i in h)
        frequency_margins = {"1-4": [], "5-19": [], "20+": []}
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

            ### >>> Self_EvolveRec-BLOCK-START: Record existing window and preserve family-specific feature adapter
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            if seq_t.ndim != 2 or seq_t.size(1) < 1:
                raise ValueError("Training sequences must have shape B x L with L >= 1")
            self._observed_maxlen = int(seq_t.size(1))
            log_feats = torch.cat(
                [self._features(chunk) for chunk in seq_t.split(64)], dim=0
            )
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Bound sampled-margin scoring and reject invalid measurements
            if pos_t.shape != seq_t.shape or neg_t.shape != seq_t.shape:
                raise ValueError(
                    "Sequence, positive and negative tensors must share B x L shape"
                )
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = torch.cat(
                [
                    self.model.predict_score(features, items)
                    for features, items in zip(log_feats.split(64), all_items.split(64))
                ],
                dim=0,
            )
            if all_logits.shape != all_items.shape:
                raise ValueError("Per-position candidate scoring must return B x L x K")
            if not torch.isfinite(all_logits).all():
                raise ValueError("Sampled-margin scoring returned nonfinite values")
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count valid sampled margins and stratify by training frequency
            if all_logits.shape != all_items.shape:
                raise ValueError("Per-position candidate scoring must return B x L x K")
            valid = (pos_t != 0) & (seq_t != 0) & (neg_t != 0)
            if valid.any():
                values = margin[valid].detach().float().cpu()
                margins_all.append(values)
                ids = pos_t[valid].detach().cpu().tolist()
                for item, value in zip(ids, values.tolist()):
                    count = frequency[item]
                    label = "1-4" if count < 5 else "5-19" if count < 20 else "20+"
                    frequency_margins[label].append(value)
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Apply the same valid-position mask to all margin summaries
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
                ### >>> Self_EvolveRec-BLOCK-START: Missing sampled margins are unavailable measurements
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

        ### >>> Self_EvolveRec-BLOCK-START: Report sampled-margin denominators and frequency strata
        self._margin_count = sum(m.numel() for m in margins_all)
        self._margin_frequency = {
            label: {
                "positions": len(values),
                "mean": float(np.mean(values)) if values else None,
                "neg_beats_pos_rate": (
                    float(np.mean(np.asarray(values) < 0)) if values else None
                ),
            }
            for label, values in frequency_margins.items()
        }
        breakdown["valid_positions"] = self._margin_count
        breakdown["metadata_status"] = (
            "Item-category metadata unavailable; breakdown lists intentionally empty"
        )
        if not self._margin_count:
            breakdown["unavailable_reason"] = (
                "No valid sampled positive/negative positions"
            )
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Audit actual GRU scoring and measure separate candidate and temporal sensitivity with explicit missing evidence
    def _correctness_checks(self, width):
        report = {
            "passed": False,
            "atol": 1e-5,
            "rtol": 1e-4,
            "max_valid_feature_error": 0.0,
            "max_final_feature_error": 0.0,
            "max_candidate_score_error": 0.0,
            "max_position_score_error": 0.0,
        }
        try:
            if width < 1:
                raise ValueError("Existing model input window is unavailable")
            itemnum = int(self.module.itemnum)
            if itemnum < 1:
                raise ValueError("Empty item vocabulary")
            history = [1 + i % itemnum for i in range(min(5, width))]
            histories = [history, [history[-1]], []]
            sequence = self._pad(histories, width)
            features = self._features(sequence)
            if features.shape[:2] != sequence.shape or features.ndim != 3:
                raise ValueError("log2feats must return B x L x H")
            report["dtype"] = str(features.dtype)
            states = self._last(features, sequence)
            candidates = torch.arange(1, min(3, itemnum) + 1, device=self.device)
            ids = candidates[None, :].expand(len(histories), -1)
            # DEBUG: Exercise the host's supported per-position scoring shape.
            scores = self._candidate_scores(states, ids)
            if scores.shape != ids.shape:
                raise ValueError("Batched candidate scoring must return B x K")

            def compare(a, b, key):
                if (
                    a.shape != b.shape
                    or not torch.isfinite(a).all()
                    or not torch.isfinite(b).all()
                ):
                    raise ValueError(f"Nonfinite values or shape mismatch: {key}")
                error = float((a - b).abs().max().item()) if a.numel() else 0.0
                report[key] = max(report.get(key, 0.0), error)
                if not torch.allclose(a, b, atol=1e-5, rtol=1e-4):
                    raise ValueError(f"Execution consistency failed: {key}")

            actual_features = self.module.log2feats(sequence)
            compare(features, actual_features, "max_modelview_feature_error")
            actual_readout = self.module._readout(features, sequence)
            compare(states, actual_readout, "max_same_tensor_readout_error")
            if not torch.equal(states, actual_readout):
                raise ValueError("Same-tensor final-position readouts disagree")
            compare(
                scores,
                self.module.predict_score(states, ids),
                "max_modelview_score_error",
            )
            compare(features, self._features(sequence), "max_repeated_feature_error")
            order = torch.arange(sequence.size(0) - 1, -1, -1, device=self.device)
            compare(
                features.index_select(0, order),
                self._features(sequence.index_select(0, order)),
                "max_batch_order_feature_error",
            )
            for row, h in enumerate(histories):
                if h:
                    # Compare aligned valid features at every supported leading-pad width.
                    widths = sorted({len(h), min(width, len(h) + 1), width})
                    reference = self._features(self._pad([h], len(h)))
                    for test_width in widths:
                        test_seq = self._pad([h], test_width)
                        test_features = self._features(test_seq)
                        compare(
                            reference[0],
                            test_features[0, -len(h) :],
                            "max_valid_feature_error",
                        )
                        compare(
                            states[row],
                            self._last(test_features, test_seq)[0],
                            "max_final_feature_error",
                        )
                    compare(
                        reference[0],
                        features[row, -len(h) :],
                        "max_valid_feature_error",
                    )
                    # DEBUG: Preserve singleton output shape through ModelView.
                    individual = self._candidate_scores(reference[0, -1], candidates)
                else:
                    compare(
                        features[row],
                        torch.zeros_like(features[row]),
                        "max_valid_feature_error",
                    )
                    # DEBUG: Score empty-history features through the same adapter.
                    individual = self._candidate_scores(
                        torch.zeros_like(states[row]), candidates
                    )
                compare(scores[row], individual, "max_candidate_score_error")
            compare(
                features[sequence.eq(0)],
                torch.zeros_like(features[sequence.eq(0)]),
                "max_valid_feature_error",
            )
            # Explicit broadcasting of a shared list; verify every selected position.
            position_ids = candidates[None, None, :].expand(sequence.size(0), width, -1)
            position_scores = self.model.predict_score(features, position_ids)
            if position_scores.shape != position_ids.shape:
                raise ValueError("Per-position scoring must return B x L x K")
            for row in range(sequence.size(0)):
                for position in sorted({0, width // 2, width - 1}):
                    # DEBUG: Avoid requiring a separate H x K adapter interface.
                    reference = self._candidate_scores(
                        features[row, position], candidates
                    )
                    compare(
                        position_scores[row, position],
                        reference,
                        "max_position_score_error",
                    )
            # Compare against the actual prediction entry point as well as the
            # ModelView adapter. This detects a wrong readout or scoring adapter.
            predictor = getattr(self.module, "score_candidates", None)
            if not callable(predictor):
                raise ValueError(
                    "GRU4Rec prediction entry point score_candidates is unavailable"
                )
            for row in range(sequence.size(0)):
                predicted = predictor(sequence[row : row + 1], candidates)
                if predicted.shape != candidates.shape:
                    raise ValueError(
                        "Prediction entry point must return one score per candidate"
                    )
                compare(scores[row], predicted, "max_prediction_score_error")
            # Also execute a standalone all-padding batch: its path can differ
            # from an empty row sharing a batch with nonempty histories.
            empty_sequence = self._pad([[]], width)
            empty_features = self._features(empty_sequence)
            compare(
                empty_features,
                torch.zeros_like(empty_features),
                "max_valid_feature_error",
            )
            # DEBUG: Use the same canonical scorer for standalone empty batches.
            empty_scores = self._candidate_scores(
                self._last(empty_features, empty_sequence), candidates[None, :]
            )
            compare(empty_scores[0], scores[-1], "max_candidate_score_error")
            exported = getattr(self.model, "diagnostic_exported_predictor", None)
            report["exported_predictor_parity"] = None
            report["exported_predictor_reason"] = (
                "Exported callable adapter unavailable"
            )
            if callable(exported):
                exported_scores = torch.as_tensor(
                    exported(sequence.cpu().numpy(), candidates.cpu().tolist()),
                    device=scores.device,
                )
                compare(scores, exported_scores, "max_exported_score_error")
                report["exported_predictor_parity"] = True
                report["exported_predictor_reason"] = None
            # Success covers the executed module/View checks. Missing export
            # coverage remains unavailable and is never counted as a pass.
            report["passed"] = True
        except Exception as exc:
            report["reason"] = f"{type(exc).__name__}: {exc}"
        return report

    def _temporal_sensitivity(self, width):
        category_names = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        category = getattr(self.model, "dataset_name", None)
        if category is None:
            category = getattr(self.module, "dataset_name", None)
        if category not in category_names:
            category = None
        result = {
            "population": "Selected training-prefix targets; conditional in-sample sensitivity",
            "dataset_category": category,
            "category_label_reason": (
                None
                if category
                else "Dataset routing label not exposed by this ModelView"
            ),
            "seed": 42,
            "user_cap": 200,
            "bootstrap_replicates": 2000,
            "eligible_users": 0,
            "sampled_users": 0,
            "retained_users": 0,
            "exclusions": {},
            "unique_keys": True,
            "user_outcomes": [],
            "category_results": {},
            "available_category_summary": None,
            "four_category_mean": None,
            "four_category_interval": None,
            "four_category_unavailable_reason": "This probe receives one category ModelView; all four independent category records are required",
        }
        if width < 5:
            result["unavailable_reason"] = (
                "Existing input window has fewer than five positions"
            )
            return result
        histories = self._histories()
        eligible = sorted(
            (u for u, h in histories.items() if len(h) >= 6), key=lambda u: str(u)
        )
        rng = np.random.default_rng(42)
        result["eligible_users"] = len(eligible)
        selected = rng.choice(
            len(eligible), size=min(200, len(eligible)), replace=False
        )
        users = [eligible[int(i)] for i in selected]
        result["sampled_users"] = len(users)
        itemnum = int(self.module.itemnum)
        excluded = Counter()
        keys = set()
        effects = []
        for user in users:
            original = histories[user]
            target_index = int(rng.integers(5, len(original)))
            target = original[target_index]
            window = original[:target_index][-width:]
            if len(window) < 5:
                excluded["short_retained_prefix"] += 1
                continue
            if not all(1 <= item <= itemnum for item in original):
                excluded["invalid_training_item_id"] += 1
                continue
            # The unchanged full training history defines exclusions, just as in
            # the supplied validation sampler. Perturbations never change them.
            forbidden = set(original) | {0, target}
            available = itemnum - sum(1 <= item <= itemnum for item in forbidden)
            if available < 99:
                excluded["fewer_than_99_unique_negatives"] += 1
                continue
            candidates = [target]
            used = set(forbidden)
            # Complement sampling avoids rejection blowups for dense histories.
            if available * 5 < itemnum:
                pool = np.asarray(
                    [i for i in range(1, itemnum + 1) if i not in forbidden]
                )
                candidates.extend(int(i) for i in rng.choice(pool, 99, replace=False))
            else:
                while len(candidates) < 100:
                    item = int(rng.integers(1, itemnum + 1))
                    if item not in used:
                        candidates.append(item)
                        used.add(item)
            n = len(window)
            k = max(1, (n - 1) // 4)
            variants = [
                ("unchanged", list(window)),
                ("oldest_delete", window[k:]),
                ("recent_delete", window[: n - 1 - k] + window[-1:]),
                ("final_item_remove", window[:-1]),
                ("last_item_only", window[-1:]),
            ]
            for _ in range(5):
                removed = set(int(i) for i in rng.choice(n - 1, k, replace=False))
                variants.append(
                    (
                        "random_delete",
                        [item for i, item in enumerate(window) if i not in removed],
                    )
                )
                shuffled = []
                for start in range(0, n - 1, 3):
                    block = window[start : min(start + 3, n - 1)]
                    shuffled.extend(int(i) for i in rng.permutation(block))
                variants.append(("block_shuffle", shuffled + window[-1:]))
            sequence = self._pad([h for _, h in variants], width)
            features = self._features(sequence)
            states = self._last(features, sequence)
            ids = torch.tensor(candidates, device=self.device, dtype=torch.long)
            # Fifteen examples, one hundred candidates: below both GPU caps.
            # DEBUG: Temporal diagnostics must use the same ModelView scoring
            # contract as the execution checks and sampled-margin probe.
            scores = self._candidate_scores(
                states, ids[None, :].expand(len(variants), -1)
            )
            if scores.shape != (len(variants), 100) or not torch.isfinite(scores).all():
                raise ValueError(
                    "Nonfinite temporal scores or invalid B x K score shape"
                )
            helper = getattr(self.model, "diagnostic_rank_helper", None)
            rule = getattr(self.model, "diagnostic_ranking_rule", None)
            helper = (
                helper if callable(helper) and isinstance(rule, str) and rule else None
            )
            result["ranking_rule"] = rule if helper else None
            result["ranking_unavailable_reason"] = (
                None if helper else "Pinned ranking/tie adapter unavailable"
            )
            native_rows = scores.detach().cpu()
            ranks, hits = [], []
            for row in native_rows:
                rank = hit = None
                if helper is not None:
                    rank, hit = helper(row.clone(), target, list(candidates), k=10)
                    if (
                        not np.isfinite(rank)
                        or int(rank) != rank
                        or not 1 <= rank <= 100
                        or hit not in (0, 1)
                        or hit != int(rank <= 10)
                    ):
                        raise ValueError("Pinned helper returned invalid rank/hit")
                    rank, hit = int(rank), int(hit)
                ranks.append(rank)
                hits.append(hit)
            if not torch.isfinite(states).all():
                raise ValueError("Temporal features contain nonfinite values")
            state_array = states.detach().float().cpu().numpy().astype(np.float64)
            base = state_array[0]
            base_norm = float(np.linalg.norm(base))
            condition_values = {}
            for index, (condition, _) in enumerate(variants):
                h = state_array[index]
                norm = float(np.linalg.norm(h))
                cosine = (
                    None
                    if min(norm, base_norm) < 1e-8
                    else float(
                        1.0 - np.clip(np.dot(h, base) / (norm * base_norm), -1.0, 1.0)
                    )
                )
                values = {
                    "rank_change": (
                        None if ranks[index] is None else float(ranks[index] - ranks[0])
                    ),
                    "hr10_change": (
                        None if hits[index] is None else float(hits[index] - hits[0])
                    ),
                    "cosine_distance": cosine,
                    "normalized_displacement": float(
                        np.linalg.norm(h - base) / max(base_norm, 1e-8)
                    ),
                }
                condition_values.setdefault(condition, []).append(values)
            averaged = {}
            for condition, repeats in condition_values.items():
                averaged[condition] = {}
                for metric in repeats[0]:
                    values = [
                        entry[metric] for entry in repeats if entry[metric] is not None
                    ]
                    averaged[condition][metric] = (
                        float(np.mean(values))
                        if len(values) == len(repeats) and np.isfinite(values).all()
                        else None
                    )
                    averaged[condition][metric + "_valid_repeats"] = len(values)
            key = (category, str(user), target_index)
            if key in keys:
                raise ValueError("Duplicate category/user/target-index diagnostic key")
            keys.add(key)
            length_slice = "5-9" if n < 10 else "10-19" if n < 20 else "20+"
            effects.append(
                {
                    "user": str(user),
                    "target_index": target_index,
                    "history_length": n,
                    "length_slice": length_slice,
                    "target_status": (
                        "repeat" if target in original[:target_index] else "novel"
                    ),
                    "intact_rank": ranks[0],
                    "intact_hr10": hits[0],
                    "effects": averaged,
                }
            )
        result["exclusions"] = dict(excluded)
        result["retained_users"] = len(effects)
        result["user_outcomes"] = effects
        summary = self._summarize_users(effects)
        result["available_category_summary"] = summary
        if category:
            result["category_results"][category] = summary
        if not effects:
            result["unavailable_reason"] = (
                "No retained eligible training-prefix examples"
            )
        return result

    @staticmethod
    def _summarize_users(records):
        bootstrap_rng = np.random.default_rng(42)

        def summarize(rows):
            report = {
                "users": len(rows),
                "sparse": len(rows) < 30,
                "intact_hr10": (
                    float(np.mean([r["intact_hr10"] for r in rows]))
                    if rows and all(r["intact_hr10"] is not None for r in rows)
                    else None
                ),
                "conditions": {},
            }
            if not rows:
                report["unavailable_reason"] = "No users in this population"
                return report
            # The same resampled users are used for all paired outcome columns.
            draws = (
                bootstrap_rng.integers(0, len(rows), size=(2000, len(rows)))
                if len(rows) >= 30
                else None
            )
            for condition in rows[0]["effects"]:
                report["conditions"][condition] = {}
                for metric in (
                    "rank_change",
                    "hr10_change",
                    "cosine_distance",
                    "normalized_displacement",
                ):
                    values = np.asarray(
                        [
                            (
                                np.nan
                                if r["effects"][condition][metric] is None
                                else r["effects"][condition][metric]
                            )
                            for r in rows
                        ],
                        dtype=np.float64,
                    )
                    finite = np.isfinite(values)
                    count = int(finite.sum())
                    entry = {
                        "users": count,
                        "unavailable_users": len(rows) - count,
                        "mean": float(values[finite].mean()) if count else None,
                        "constant": (
                            bool(np.ptp(values[finite]) == 0) if count else None
                        ),
                        "interval": None,
                    }
                    if draws is not None and count == len(rows):
                        means = values[draws].mean(1)
                        entry["interval"] = [
                            float(v) for v in np.percentile(means, [2.5, 97.5])
                        ]
                    else:
                        entry["interval_reason"] = (
                            "Requires at least 30 unique users and complete outcomes in this population"
                        )
                    report["conditions"][condition][metric] = entry
            return report

        result = summarize(records)
        result["slices"] = {}
        for field, labels in (
            ("length_slice", ["5-9", "10-19", "20+"]),
            ("target_status", ["repeat", "novel"]),
        ):
            result["slices"][field] = {
                label: summarize([r for r in records if r[field] == label])
                for label in labels
            }
        return result

    def _candidate_panel(self, width):
        import os
        import tempfile

        categories = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        category = getattr(self.model, "dataset_name", None) or getattr(
            self.module, "dataset_name", None
        )
        if category not in categories or width < 5:
            raise ValueError("Candidate panel requires category routing and width >= 5")
        category_index = categories.index(category)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        panel_started = time.perf_counter()
        histories = self._histories()
        if len({str(u) for u in histories}) != len(histories):
            raise ValueError("Serialized training user IDs collide")
        itemnum = int(self.module.itemnum)
        if any(i < 1 or i > itemnum for h in histories.values() for i in h):
            raise ValueError("Training histories contain out-of-catalog item IDs")
        frequency = Counter(i for h in histories.values() for i in h)

        def band(count):
            return (
                "0"
                if count == 0
                else "1-4" if count < 5 else "5-19" if count < 20 else "20+"
            )

        catalog = np.arange(1, itemnum + 1, dtype=np.int64)
        bands = np.asarray([band(frequency[int(i)]) for i in catalog])
        result = {
            "protocol": "candidate_panel_v1",
            "dataset_category": category,
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "generator": "default_rng/PCG64",
            "draws_per_condition": 5,
            "selection": "First 200 eligible serialized user IDs; final training target; latest width prefix",
            "case_protocol_matches_existing_temporal_probe": False,
            "protocol_difference": "Existing temporal probe samples users and target indices; cases are not pooled",
            "records": [],
            "uniform_only_records": [],
            "exclusions": [],
            "failures": [],
            "frozen_sets": {
                "records": [],
                "reason": "No protocol-matching prior frozen sets supplied",
            },
            "exact_reference": {
                "records": [],
                "reason": "Optional full-pool scoring deferred to bound diagnostic cost",
            },
            "macro": None,
            "macro_reason": "Requires external aggregation of all four categories",
        }
        helper = getattr(self.model, "diagnostic_rank_helper", None)
        rule = getattr(self.model, "diagnostic_ranking_rule", None)
        helper = helper if callable(helper) and isinstance(rule, str) and rule else None
        result["ranking_rule"] = rule if helper else None
        result["ranking_unavailable_reason"] = (
            None if helper else "Pinned ranking/tie adapter unavailable"
        )
        eligible = []
        for user in sorted(histories, key=str):
            history = histories[user]
            reason = (
                "invalid_item"
                if any(i < 1 or i > itemnum for i in history)
                else "short_prefix" if len(history) < 6 else None
            )
            if reason:
                result["exclusions"].append({"user": str(user), "reason": reason})
            else:
                eligible.append(user)
        result["eligible_users"] = len(eligible)
        result["selected_users"] = min(200, len(eligible))
        for stable_index, user in enumerate(eligible[:200]):
            history = histories[user]
            target, prefix = history[-1], history[:-1][-width:]
            uniform = catalog[~np.isin(catalog, np.asarray(history, dtype=np.int64))]
            target_band = band(frequency[target])
            matched = uniform[bands[uniform - 1] == target_band]
            record = {
                "user": str(user),
                "stable_user_index": stable_index,
                "target": target,
                "target_index": len(history) - 1,
                "raw_length": len(history),
                "retained_length": len(prefix),
                "truncated_items": len(history) - 1 - len(prefix),
                "target_band": target_band,
                "repeat_retained": target in prefix,
                "repeat_complete_prefix": target in history[:-1],
                "length_slice": (
                    "5-9"
                    if len(prefix) < 10
                    else "10-19" if len(prefix) < 20 else "20+"
                ),
                "uniform_pool": int(uniform.size),
                "matched_pool": int(matched.size),
                "zero_frequency_uniform_items": int(np.sum(bands[uniform - 1] == "0")),
                "conditions": {},
            }
            if min(uniform.size, matched.size) < 99:
                result["exclusions"].append(
                    {
                        **{
                            k: record[k]
                            for k in (
                                "user",
                                "target_band",
                                "uniform_pool",
                                "matched_pool",
                            )
                        },
                        "reason": "insufficient_pool",
                    }
                )
            if uniform.size < 99:
                continue
            try:
                sequence = self._pad([prefix], width)
                record["sequence"] = sequence[0].cpu().tolist()
                features = self._features(sequence)
                if (
                    features.ndim != 3
                    or features.shape[:2] != sequence.shape
                    or not torch.isfinite(features).all()
                ):
                    raise ValueError("Invalid candidate-panel features")
                state = self._last(features, sequence)
                for condition_id, name, pool in (
                    (300, "uniform", uniform),
                    (301, "matched", matched),
                ):
                    if pool.size < 99:
                        continue
                    draws = record["conditions"][name] = []
                    for draw_index in range(5):
                        seed = [
                            42,
                            category_index,
                            stable_index,
                            condition_id,
                            draw_index,
                        ]
                        rng = np.random.default_rng(np.random.SeedSequence(seed))
                        candidates = [target] + rng.choice(
                            pool, 99, replace=False
                        ).tolist()
                        ids = torch.tensor(
                            candidates, device=self.device, dtype=torch.long
                        )
                        # Preserve the verified per-position ModelView scorer contract.
                        score_features = state[:, None, :].expand(-1, 2, -1)
                        score_ids = ids[None, None, :].expand(1, 2, -1)
                        scored = self.model.predict_score(score_features, score_ids)
                        if (
                            scored.shape != score_ids.shape
                            or not torch.isfinite(scored).all()
                        ):
                            raise ValueError(
                                "Invalid candidate-panel score shape or values"
                            )
                        # Exercise the exact panel shape against the actual
                        # trained scorer, without introducing another model.
                        actual = self.module.predict_score(score_features, score_ids)
                        torch.testing.assert_close(scored, actual, atol=1e-5, rtol=1e-4)
                        torch.testing.assert_close(
                            scored[:, 0], scored[:, 1], atol=1e-5, rtol=1e-4
                        )
                        native = scored[0, 0].detach().cpu()
                        rank = hit = None
                        if helper:
                            rank, hit = helper(
                                native.clone(), target, list(candidates), k=10
                            )
                            if (
                                not np.isfinite(rank)
                                or int(rank) != rank
                                or not 1 <= rank <= 100
                                or hit not in (0, 1)
                                or hit != int(rank <= 10)
                            ):
                                raise ValueError(
                                    "Pinned helper returned invalid rank/hit"
                                )
                            rank, hit = int(rank), int(hit)
                        scores = native.double().numpy()
                        draws.append(
                            {
                                "seed": seed,
                                "candidate_ids": candidates,
                                "scores": scores.tolist(),
                                "dtype": str(native.dtype),
                                "margin": float(scores[0] - scores[1:].max()),
                                "negative_win_fraction": float(
                                    np.mean(scores[1:] > scores[0])
                                ),
                                "exact_target_ties": int(
                                    np.sum(scores[1:] == scores[0])
                                ),
                                "rank": rank,
                                "hr10": hit,
                            }
                        )
                destination = (
                    "records" if matched.size >= 99 else "uniform_only_records"
                )
                result[destination].append(record)
            except Exception as exc:
                print(
                    f"Diagnosis candidate panel user {user}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                result["failures"].append(
                    {
                        "user": str(user),
                        "reason": f"{type(exc).__name__}: {exc}",
                        "partial_record": record,
                    }
                )

        def summarize(rows):
            output = {"users": len(rows), "sparse": len(rows) < 30, "metrics": {}}
            rng = np.random.default_rng(
                np.random.SeedSequence([42, category_index, 400])
            )
            indices = (
                rng.integers(len(rows), size=(2000, len(rows)))
                if len(rows) >= 30
                else None
            )
            for metric in (
                "margin",
                "negative_win_fraction",
                "exact_target_ties",
                "rank",
                "hr10",
            ):
                arrays = {}
                for condition in ("uniform", "matched"):
                    if not rows or any(condition not in r["conditions"] for r in rows):
                        continue
                    values = [
                        [d[metric] for d in r["conditions"][condition]] for r in rows
                    ]
                    if any(len(v) != 5 or any(x is None for x in v) for v in values):
                        continue
                    array = np.asarray(values, dtype=np.float64)
                    if not np.isfinite(array).all():
                        raise FloatingPointError("Nonfinite candidate-panel summary")
                    arrays[condition] = array
                entries = output["metrics"][metric] = {
                    name: {
                        "mean": None,
                        "users": 0,
                        "candidate_mcse": None,
                        "interval": None,
                        "unavailable_reason": "Requires five finite draws per user and the applicable ranking adapter",
                    }
                    for name in ("uniform", "matched", "damage")
                }
                for condition, array in arrays.items():
                    means = array.mean(1)
                    entries[condition] = {
                        "mean": float(means.mean()),
                        "users": len(rows),
                        "candidate_mcse": float(
                            np.sqrt(array.var(1, ddof=1).sum() / (5 * len(rows) ** 2))
                        ),
                        "interval": (
                            np.percentile(means[indices].mean(1), [2.5, 97.5]).tolist()
                            if indices is not None
                            else None
                        ),
                    }
                if len(arrays) == 2:
                    sign = -1 if metric in ("margin", "hr10") else 1
                    difference = sign * (
                        arrays["matched"].mean(1) - arrays["uniform"].mean(1)
                    )
                    entries["damage"] = {
                        "mean": float(difference.mean()),
                        "users": len(rows),
                        "candidate_mcse": float(
                            np.sqrt(
                                sum(a.var(1, ddof=1).sum() for a in arrays.values())
                                / (5 * len(rows) ** 2)
                            )
                        ),
                        "interval": (
                            np.percentile(
                                difference[indices].mean(1), [2.5, 97.5]
                            ).tolist()
                            if indices is not None
                            else None
                        ),
                    }
            return output

        result["summary"] = summarize(result["records"])
        result["uniform_only_summary"] = summarize(result["uniform_only_records"])
        result["slices"] = {
            field: {
                str(label): summarize(
                    [r for r in result["records"] if r[field] == label]
                )
                for label in labels
            }
            for field, labels in (
                ("target_band", ["1-4", "5-19", "20+"]),
                ("length_slice", ["5-9", "10-19", "20+"]),
                ("repeat_retained", [False, True]),
            )
        }
        result["complete_users"] = len(result["records"])
        result["uniform_only_users"] = len(result["uniform_only_records"])
        result["exclusion_counts"] = dict(
            Counter(r["reason"] for r in result["exclusions"])
        )
        result["excluded_pool_counts_by_target_band"] = {
            label: sum(r.get("target_band") == label for r in result["exclusions"])
            for label in ("1-4", "5-19", "20+")
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result["diagnostic_seconds_before_serialization"] = (
            time.perf_counter() - panel_started
        )
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{category}_candidate_panel.json")
        result["artifact_path"] = path
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=directory, delete=False
            ) as handle:
                temporary = handle.name
                json.dump(result, handle, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return result

    def run_full_diagnosis(self, train_loader):
        previous_modes = [(module, module.training) for module in self.module.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        metrics = {}
        try:
            # Reset per-run state so a failed or empty loader cannot expose
            # counts or a window retained from a previous diagnostic call.
            self._observed_maxlen = None
            self._margin_count = 0
            self._margin_frequency = {}
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(42)
            self.module.eval()
            with torch.inference_mode():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                try:
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            train_loader, num_batches=10, eps=eps, topk_categories=20
                        )
                    )
                except Exception as exc:
                    margin_metrics = dict.fromkeys(
                        ["mean", "p10", "neg_beats_pos_rate", "small_margin_rate"]
                    )
                    self._margin_count = 0
                    self._margin_frequency = {}
                    reason = f"{type(exc).__name__}: {exc}"
                    print(
                        f"Diagnosis sampled-margin probe unavailable: {reason}",
                        flush=True,
                    )
                    breakdown = {
                        "unavailable_reason": reason,
                        "valid_positions": 0,
                        "low_margin_top_categories": {"pos_item": [], "neg_item": []},
                        "wrong_rank_top_categories": {"pos_item": [], "neg_item": []},
                    }
                metrics.update(
                    {
                        "embedding_collapse_score": collapse_score,
                        "embedding_sample_count": min(2048, int(self.module.itemnum)),
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "pos_neg_margin_frequency_breakdown": self._margin_frequency,
                    }
                )
                width = self._observed_maxlen
                if width is None:
                    width = getattr(self.model, "maxlen", None)
                if width is None:
                    width = getattr(getattr(self.model, "args", None), "maxlen", 0)
                width = int(width or 0)
                checks = self._correctness_checks(width)
                metrics["execution_checks"] = checks
                # Export parity is separate from the available module/View checks.
                checks.setdefault("exported_predictor_parity", None)
                checks.setdefault(
                    "exported_predictor_reason",
                    "No exported callable adapter supplied to this probe",
                )
                metrics["candidate_panel"] = None
                if checks["passed"]:
                    try:
                        metrics["candidate_panel"] = self._candidate_panel(width)
                    except Exception as exc:
                        metrics["candidate_panel_unavailable_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        print(
                            f"Diagnosis candidate panel unavailable: {metrics['candidate_panel_unavailable_reason']}",
                            flush=True,
                        )
                else:
                    metrics["candidate_panel_unavailable_reason"] = (
                        "Required execution checks failed or remain unresolved"
                    )
                if checks["passed"]:
                    try:
                        metrics["temporal_sensitivity"] = self._temporal_sensitivity(
                            width
                        )
                    except Exception as exc:
                        metrics["temporal_sensitivity"] = None
                        metrics["temporal_unavailable_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        print(
                            f"Diagnosis temporal probe unavailable: {metrics['temporal_unavailable_reason']}",
                            flush=True,
                        )
                else:
                    metrics["temporal_sensitivity"] = None
                    metrics["temporal_unavailable_reason"] = (
                        "Execution checks failed; original checkpoint semantics were not modified"
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["diagnostic_inference_seconds"] = float(
                time.perf_counter() - started
            )
        finally:
            for module, training in previous_modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return {
            "metrics": metrics,
            "metric_definitions": {
                "embedding_collapse_score": "Sample up to 2048 real-item embeddings, seed 42, excluding padding/reserved tokens. Map mean off-diagonal pairwise cosine to [0,1]; denominator k*(k-1). Higher indicates directional alignment, not measured recommendation diversity. Existing convention returns zero for at most two items.",
                "embedding_sample_count": "Number of real-item embeddings sampled for the collapse probe; pairwise denominator is k*(k-1).",
                "pos_neg_margin_mean": "Mean positive-minus-negative model score across valid training-loader positions in at most ten batches; positive, negative and sequence IDs must be nonzero. Larger means stronger sampled separation; denominator is valid_positions in the breakdown.",
                "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of the same valid training-position margins, index ceil(0.1*N); lower indicates weaker sampled separation. Null if N=0.",
                "pos_neg_neg_beats_pos_rate": "Fraction of the same N valid training positions whose sampled negative score exceeds positive score; lower is better. Null if N=0.",
                "pos_neg_small_margin_rate": "Fraction of the same N valid training positions with positive-minus-negative score below eps=0.5; lower indicates fewer small margins. Null if N=0.",
                "pos_neg_margin_category_breakdown": "Counts low-margin and wrong-rank valid training positions and their denominator. Item-category metadata is unavailable, so category lists remain empty; these lists are unrelated to dataset routing.",
                "pos_neg_margin_frequency_breakdown": "Stratify the same sampled training positions by positive-item training-interaction frequency: 1-4, 5-19, 20+. Each stratum reports its position denominator, mean margin and negative-beats-positive fraction. Frequency is not exposure.",
                "execution_checks": "At atol=1e-5, rtol=1e-4, compare individual unpadded and batched left-padded aligned valid features, maximum-nonzero-position readouts and fixed candidate scores, including singleton/empty rows, a standalone all-padding batch and added leading zeros within the existing window. Compare ModelView scores with the actual trained module's score_candidates prediction entry point for every checked row. Empty sequence features must be zero; trained score biases remain included. Check BxLxK scoring against loop references. Report dtype, maximum discrepancies and failures; no checkpoint repair occurs here.",
                "temporal_sensitivity": {
                    "population": "At most 200 uniformly sampled eligible training users without replacement in sorted ID order, RNG 42, per actual dataset ModelView. One uniformly selected training target index with at least five preceding interactions; truncate its prefix once to the existing model window. Require five retained items. Report eligible, sampled, retained, exclusion and unique-key counts. These targets may have been used in training.",
                    "candidates": "Target plus 99 unique negatives drawn independently of official evaluation. Exclude the original full training history and target; preserve earlier target occurrences in inputs. Candidates remain fixed across interventions. Rank and HR require the pinned diagnostic helper and documented ranking rule; otherwise both remain null. No substitute tie ordering is assumed.",
                    "interventions": "For retained length n, k=max(1,floor((n-1)/4)). Unchanged, oldest-k deletion, most-recent-k deletion among the first n-1 positions, five uniform random-k deletions among those positions, five independent within-block permutations of the first n-1 positions with block size three, final-item removal, and last-item-only. Preserve survivor chronology and the final item except in explicit removal. Never refill; left-pad every variant to the same width. Average repeated scalar effects within each user before aggregation. Local shuffling can create implausible chronology.",
                    "rank_change": "Perturbed rank minus intact rank per retained user; positive means ranking damage. Means divide by reported contributing user counts.",
                    "hr10_change": "Perturbed indicator(rank<=10) minus intact indicator(rank<=10) per retained user; negative means ranking damage. Means divide by reported contributing user counts.",
                    "cosine_distance": "One minus cosine of perturbed and intact final states; larger means state movement. Null when either norm is below 1e-8. Report valid-repeat and user denominators. Representation movement alone does not establish utility damage.",
                    "normalized_displacement": "L2 state difference divided by max(intact-state L2 norm,1e-8), averaged over repeats then users; larger means relative state movement.",
                    "summaries": "Descriptive category means, intact diagnostic HR@10 and history-length 5-9/10-19/20+ and repeat/novel slices. Repeat means target occurred before its training index, even outside the retained window. Denominators and sparse/constant flags accompany each result. user_outcomes caches paired scalar outcomes for host aggregation; it contains no validation or test labels.",
                    "uncertainty": "2000 CPU bootstrap replicates, separate RNG 42, resampling users with replacement within this category and preserving paired conditions. Percentile 2.5th-97.5th intervals require at least 30 unique users with complete outcomes. Four-category point estimates require all four category records and equal category weights; four-category intervals additionally require at least 30 contributing users in every category, including within each slice. This single-view API cannot manufacture missing categories, so macro fields are null with reasons. An available-category summary is explicitly descriptive and separate.",
                    "limits": "Conditional in-sample sensitivity given fitted checkpoints, selected prefixes and candidates; intervals omit seed variability, new-category generalization and validation-search selection optimism. Simulator users and their required full-catalog recommendations remain separate. No inference about intent, exposure, ratings, reviews or simulator traits is supported. Official macro validation HR@10 alone selects models.",
                },
                "candidate_panel": {
                    "population": "New protocol, separate from existing temporal cases: first at most 200 eligible training users in lexicographic serialized-ID order, collisions rejected. Final training interaction is target; latest existing-width preceding items form the unchanged input, requiring five items. Targets may have participated in fitting.",
                    "pools": "Real IDs 1..itemnum excluding the complete unchanged training history, including target. Uniform includes zero-frequency catalog items; matched restricts to target training-occurrence band 1-4, 5-19 or 20+. Both require 99 eligible items. No replacement, band broadening or held-out exclusions. Pool sizes, exclusions and failures are recorded; uniform-only results use a separate population.",
                    "sampling": "Five fresh independent sets of 99 unique negatives per condition; target first solely for storage. Seeds are SeedSequence([42, category_index, stable_user_index, 300 or 301, draw_index]); fixed category order is Movies_and_TV, Industrial_and_Scientific, Electronics, CDs_and_Vinyl. Sets may overlap. Features are computed once per case; no global RNG is consumed by candidate draws. Failed cases retain partial records but contribute no complete-case summaries.",
                    "statistics": "Margin is target minus largest negative score; strict negative-win fraction divides strictly greater negative scores by 99. Exact target ties are counted separately. Native scores and dtype are persisted; summaries use float64. Rank and diagnostic HR require the pinned helper and documented rule, otherwise remain unavailable. Diagnostic HR is not official validation HR.",
                    "damage": "Matched minus uniform for negative-win fraction and rank; uniform minus matched for margin and HR, so positive means weaker separation under matched candidates. Tie-count damage is only a descriptive matched-minus-uniform count difference.",
                    "uncertainty": "Average five draws per user before equally weighting users. Condition candidate MCSE=sqrt(sum user draw variances/(5*U^2)); independent-stream contrast MCSE includes both variances. Sample variance uses ddof=1. Five replicates give noisy variance estimates. Separate 2000-replicate CPU bootstrap, namespace 400, resamples whole paired user records only when U>=30; intervals are empirical selected-user reweighting conditional on realized draws and checkpoint. MCSE and bootstrap variance are not added. Each slice reports its denominator and sparse flag.",
                    "separate_results": "Previous frozen sets are unavailable, never inserted into fresh replicates. Optional exact finite-pool reference is explicitly deferred. Category-local macro remains null; external macro aggregation requires all four categories and equal weights. Independent-category macro candidate MCSE=sqrt(sum category MCSE squared)/4.",
                    "limits": "Conditional training-prefix candidate-distribution sensitivity only. Frequency is not exposure; matching is not unbiased evaluation or proof of popularity bias. Unobserved negatives may be relevant and training targets memorized. No causal intent, generalization, simulator-trait or official validation improvement claim is supported. Architecture, checkpoints, training, predictor and native full-catalog simulator workflow are unchanged.",
                },
                "candidate_panel_execution": "The executed module/View checks gate the panel. Same-tensor production readout must agree exactly; feature and score comparisons use atol=1e-5 and rtol=1e-4. Every candidate draw additionally checks its per-position ModelView scores against the actual trained module. Export parity remains explicitly unavailable without its callable adapter; executed module checks do not establish export parity.",
                "candidate_panel_runtime": "Panel-local synchronized wall seconds before artifact serialization are persisted in the panel artifact. The overall diagnostic timer includes successful panel serialization; all diagnostic compute consumes the shared search budget.",
                "candidate_panel_unavailable_reason": "Panel failure or unresolved required execution checks; missing is not zero sensitivity.",
                "temporal_unavailable_reason": "Reason the entire temporal probe is null; failed execution checks or missing measurements must not be interpreted as zero effects.",
                "diagnostic_inference_seconds": "Elapsed wall seconds for diagnostic probes, including CPU summaries and synchronized GPU work; excludes model training. Diagnostics consume the shared search budget.",
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
