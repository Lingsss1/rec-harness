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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measured training sensitivity conservatively
    - Every statement must be grounded in the provided metric values/definitions.
    - History interventions measure sensitivity of a frozen fitted model on training prefixes, not validation performance, user intent, shortcut learning, or generalization.
    - Null statistics are unavailable, not zero. Do not infer elapsed-time effects: this supplied HSTU consumes item IDs and learned absolute positions only.
    - Confidence intervals condition on the fitted checkpoint, retained users/prefixes, primary replacement bank and candidate sets. They do not include training uncertainty; small-cluster coverage can be poor.
    - Report matching coverage, residual imbalance, text availability and control support alongside effects. Different sensitivity protocols or banks may retain different prefixes; do not interpret support changes as pure Monte Carlo variation.
    - The paired recent-versus-older comparison changes content at different locations. It does not isolate absolute position, recency, or an individual HSTU component. Score differences remain category-specific.
    - Replacement banks and diagnostic variants are inputs to one frozen checkpoint, never ensembles or independent predictors. Diagnostic outcomes must not replace macro validation HR@10 for model selection.
    - Do not infer diversity, conformity, exposure-adjusted category effects, or unavailable review/metadata attributes.
    # Review checks: traced HSTU feature/scoring shapes, validated IDs before
    # lookups, checked final-nonpadding readout and matched candidates, removed
    # unnecessary category loops, and checked undefined interval reporting.
    # Four-category macro statistics remain explicitly unavailable in this
    # single-category invocation. No training, checkpoint, or predictor changes.
    # Static review only; execution and benchmark performance were not measured.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the HSTU ModelView contract and unavailable metadata boundaries
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = getattr(view, "user_train", {})
        self.itemnum = int(self.network.itemnum)
        self.maxlen = int(self.network.maxlen)
        self.datasets = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        names = [
            getattr(owner, key, None)
            for owner in (view, self.network)
            for key in ("dataset_name", "dataset", "category_name")
        ]
        self.category = next(
            (name for name in names if isinstance(name, str) and name in self.datasets),
            None,
        )
        self.category_index = (
            self.datasets.index(self.category) if self.category is not None else 4
        )

    def _get_categories(self, item_id: int):
        # Item-category attributes are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Sample actual effective HSTU embeddings without materializing the catalog
        with torch.inference_mode():
            n = self.itemnum
            if n < 2:
                return None
            rng = np.random.default_rng(seed)
            k = min(int(sample_size), n)
            if k < 2:
                return None
            idx = torch.as_tensor(
                rng.choice(n, size=k, replace=False) + 1,
                dtype=torch.long,
                device=self.device,
            )
            x = self.model.item_emb(idx).detach().float()
            if not torch.isfinite(x).all() or (x.norm(dim=-1) == 0).any():
                return None
            ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Manage the actual module mode at the diagnostic boundary
        # run_full_diagnosis owns evaluation mode and restores it in finally.
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

            ### >>> Self_EvolveRec-BLOCK-START: Validate loader tensors before HSTU embedding lookups
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic loader must supply matching B x L tensors")
            if not 1 <= seq_t.size(1) <= self.maxlen:
                raise ValueError(
                    "Diagnostic sequence length exceeds the HSTU position table"
                )
            for name, tensor in (
                ("sequence", seq_t),
                ("positive", pos_t),
                ("negative", neg_t),
            ):
                if ((tensor < 0) | (tensor > self.itemnum)).any():
                    raise ValueError(
                        f"Diagnostic {name} contains IDs outside 0..itemnum"
                    )
            log_feats = self.model.log2feats(seq_t.detach().cpu().numpy())
            # Two aligned calls use the same scoring path as HSTU training.
            pos_logits = self.model.predict_score(log_feats, pos_t)
            neg_logits = self.model.predict_score(log_feats, neg_t)
            if pos_logits.shape != seq_t.shape or neg_logits.shape != seq_t.shape:
                raise ValueError("Aligned HSTU scores must have shape B x L")
            margin = pos_logits - neg_logits
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Restrict pair diagnostics to finite ordinary-item training positions
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            valid = (
                (seq_t != 0)
                & (pos_t > 0)
                & (pos_t <= self.itemnum)
                & (neg_t > 0)
                & (neg_t <= self.itemnum)
                & torch.isfinite(margin)
            )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same validity mask for pair summaries
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize position counts when item categories are unavailable
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Represent undefined sampled-pair statistics explicitly
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

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Paired HSTU replacements with bounded feature batches, explicit support, and user-cluster uncertainty
    def _audit_prefixes(self, batches):
        histories = (
            self.user_train.get("History", {})
            if isinstance(self.user_train, dict)
            else {}
        )
        if not isinstance(histories, dict):
            raise ValueError(
                "ModelView.user_train['History'] must map user IDs to histories"
            )
        exclusions = Counter()
        records = []
        coverage = {
            "source": "identified_training_users" if histories else "training_rows",
            "eligible_users": 0,
            "selected_users": 0,
            "unique_users": 0 if histories else None,
            "selected_rows": 0,
            "prefixes_requested": 0,
            "user_cap": 32,
            "targets_per_user_or_row": 2,
        }
        if histories:
            ordered = sorted(histories, key=lambda key: (str(key), type(key).__name__))
            eligible = []
            for index, uid in enumerate(ordered):
                try:
                    h = [int(item) for item in histories[uid]]
                except (TypeError, ValueError):
                    exclusions["malformed_history_user"] += 1
                    continue
                if any(item < 1 or item > self.itemnum for item in h):
                    exclusions["noncatalog_history_user"] += 1
                    continue
                if len(h) < 7 or self.maxlen < 5:
                    exclusions["history_too_short_user"] += 1
                    continue
                eligible.append((index, uid, h))
            coverage["eligible_users"] = len(eligible)
            rng = np.random.default_rng(
                np.random.SeedSequence([42, self.category_index])
            )
            chosen = sorted(
                rng.choice(
                    len(eligible), min(32, len(eligible)), replace=False
                ).tolist()
            )
            coverage["selected_users"] = len(chosen)
            for picked in chosen:
                index, uid, h = eligible[picked]
                rng = np.random.default_rng(
                    np.random.SeedSequence([42, self.category_index, index])
                )
                positions = sorted(
                    rng.choice(np.arange(5, len(h)), 2, replace=False).tolist()
                )
                for j in positions:
                    records.append((index, j, h[max(0, j - self.maxlen) : j], h[j]))
        else:
            # Rows have no reliable user identity. Never infer one from sequences.
            row_index = 0
            for batch in batches:
                if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                    continue
                seq = torch.as_tensor(batch[1]).detach().cpu().numpy()
                pos = torch.as_tensor(batch[2]).detach().cpu().numpy()
                for sequence, positive in zip(seq, pos):
                    if coverage["selected_rows"] >= 32:
                        break
                    candidates = []
                    for t in range(len(sequence)):
                        h = [int(x) for x in sequence[: t + 1] if int(x) != 0][
                            -self.maxlen :
                        ]
                        target = int(positive[t])
                        if (
                            len(h) >= 5
                            and 1 <= target <= self.itemnum
                            and all(1 <= x <= self.itemnum for x in h)
                        ):
                            candidates.append((t, h, target))
                    if not candidates:
                        exclusions["no_eligible_target_row"] += 1
                    else:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([42, self.category_index, row_index])
                        )
                        selected = sorted(
                            rng.choice(
                                len(candidates), min(2, len(candidates)), replace=False
                            ).tolist()
                        )
                        for selected_index in selected:
                            t, h, target = candidates[selected_index]
                            records.append((row_index, t, h, target))
                        coverage["selected_rows"] += 1
                    row_index += 1
                if coverage["selected_rows"] >= 32:
                    break
            coverage["rows_examined"] = row_index
        coverage["prefixes_requested"] = len(records)
        return records, coverage, exclusions

    def _build_matching_state(self):
        histories = self.user_train.get("History", {})
        if not isinstance(histories, dict):
            raise ValueError(
                "ModelView.user_train['History'] must map user IDs to histories"
            )
        counts, edges, outgoing = Counter(), Counter(), Counter()
        for history in histories.values():
            try:
                h = [int(item) for item in history]
            except (TypeError, ValueError):
                continue
            if any(not 1 <= item <= self.itemnum for item in h):
                continue
            counts.update(h)
            edges.update(zip(h[:-1], h[1:]))
            outgoing.update(h[:-1])
        ids = np.asarray(sorted(counts), dtype=np.int64)
        pop = np.zeros(self.itemnum + 1, dtype=np.float64)
        if ids.size:
            values = np.log1p([counts[int(item)] for item in ids])
            pop[ids] = (values - values.mean()) / max(float(values.std()), 1e-12)
        logp = lambda a, b: np.log(
            (edges[(int(a), int(b))] + 1.0) / (outgoing[int(a)] + self.itemnum)
        )
        total = sum(edges.values())
        mean = sum(count * logp(*edge) for edge, count in edges.items()) / max(total, 1)
        variance = sum(
            count * (logp(*edge) - mean) ** 2 for edge, count in edges.items()
        ) / max(total, 1)
        texts = []
        metadata = self.meta_data if isinstance(self.meta_data, dict) else {}
        for item in ids:
            record = metadata.get(str(int(item)), metadata.get(int(item), {}))
            fields = []
            for key in ("title", "description"):
                value = record.get(key, "") if isinstance(record, dict) else ""
                if isinstance(value, (list, tuple)):
                    value = " ".join(part for part in value if isinstance(part, str))
                fields.append(value if isinstance(value, str) else "")
            texts.append(" ".join(fields))
        text, text_reason = None, None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            if not ids.size or not any(value.strip() for value in texts):
                text_reason = "No training-item title/description text"
            else:
                text = TfidfVectorizer(
                    max_features=20000, dtype=np.float32
                ).fit_transform(texts)
        except (ImportError, ValueError) as error:
            text_reason = str(error)
        return {
            "ids": ids,
            "pop": pop,
            "edges": edges,
            "logp": logp,
            "transition_sd": float(np.sqrt(variance)),
            "text": text,
            "text_items": (
                set(map(int, ids[np.asarray(text.getnnz(axis=1)).ravel() > 0]))
                if text is not None
                else set()
            ),
            "lookup": {int(item): row for row, item in enumerate(ids)},
            "text_reason": text_reason,
            "training_edge_count": total,
        }

    def _score_prefix(
        self,
        index,
        position,
        history,
        target,
        counters,
        bank=0,
        tolerance=1.0,
        window_k=None,
    ):
        state = self._audit_state
        rng = np.random.default_rng(
            np.random.SeedSequence([42, self.category_index, index, position, bank])
        )
        pop, logp = state["pop"], state["logp"]
        n = len(history)
        k = min(2, (n - 1) // 2) if window_k is None else window_k
        older, recent = np.arange(k), np.arange(n - k - 1, n - 1)
        original = np.asarray(history, dtype=np.int64)
        failures, balance, draws = Counter(), [], []
        if np.any(
            np.abs(pop[original[older]] - pop[original[recent]]) > 0.5 * tolerance
        ):
            return None, "original_window_popularity_mismatch"
        pool = np.setdiff1d(state["ids"], list(set(history) | {target}))
        pools = [
            pool[
                (np.abs(pop[pool] - pop[original[a]]) <= 0.5 * tolerance)
                & (np.abs(pop[pool] - pop[original[b]]) <= 0.5 * tolerance)
            ]
            for a, b in zip(older, recent)
        ]
        if any(not len(items) for items in pools):
            return None, "empty_common_popularity_support"
        text, lookup = state["text"], state["lookup"]

        def has_text(item):
            return item in state["text_items"]

        text_mode = all(has_text(int(item)) for item in original[np.r_[older, recent]])
        if text_mode:
            pools = [
                np.asarray(
                    [item for item in items if has_text(int(item))], dtype=np.int64
                )
                for items in pools
            ]
            if any(not len(items) for items in pools):
                return None, "empty_common_text_support"

        def variant(slots, replacement):
            value = original.copy()
            value[slots] = replacement
            return value

        def transition(value, slots):
            changed = sorted(
                {
                    edge
                    for slot in slots
                    for edge in (int(slot) - 1, int(slot))
                    if 0 <= edge < n - 1
                }
            )
            return float(np.mean([logp(value[e], value[e + 1]) for e in changed]))

        def similarity(slots, replacement):
            return float(
                np.mean(
                    [
                        text[lookup[int(original[slot])]]
                        .multiply(text[lookup[int(item)]])
                        .sum()
                        for slot, item in zip(slots, replacement)
                    ]
                )
            )

        def matches(replacement):
            a, b = variant(recent, replacement), variant(older, replacement)
            delta = abs(transition(a, recent) - transition(b, older))
            if delta > 0.5 * tolerance * state["transition_sd"] + 1e-12:
                failures["transition_match_rejection"] += 1
                return None
            text_delta = (
                abs(similarity(recent, replacement) - similarity(older, replacement))
                if text_mode
                else None
            )
            if text_delta is not None and text_delta > 0.10 * tolerance:
                failures["text_match_rejection"] += 1
                return None
            return a, b, delta, text_delta

        for draw in range(5):
            accepted = None
            for attempt in range(100):
                counters["replacement_attempts"] += 1
                replacement = np.asarray([rng.choice(items) for items in pools])
                if len(set(replacement.tolist())) != k:
                    failures["duplicate_replacement"] += 1
                    continue
                accepted = matches(replacement)
                if accepted is not None:
                    break
            if accepted is None:
                failures["draw_matching_failed"] += 1
                continue
            a, b, transition_delta, text_delta = accepted
            variants = {
                "original": original.copy(),
                "identity": original.copy(),
                "recent": a,
                "older": b,
            }
            # Optional random-window control uses this same ordered replacement.
            slots = np.sort(rng.choice(n - 1, k, replace=False))
            control = variant(slots, replacement)
            if (
                not np.array_equal(slots, older)
                and not np.array_equal(slots, recent)
                and np.all(
                    np.abs(pop[replacement] - pop[original[slots]]) <= 0.5 * tolerance
                )
                and abs(transition(control, slots) - transition(a, recent))
                <= 0.5 * tolerance * state["transition_sd"] + 1e-12
                and (
                    not text_mode
                    or (
                        all(has_text(int(item)) for item in original[slots])
                        and abs(
                            similarity(slots, replacement)
                            - similarity(recent, replacement)
                        )
                        <= 0.10 * tolerance
                    )
                )
            ):
                variants["random_nonfinal"] = control
            # Descriptive corruption stress: lowest recent-window transition score
            # among up to sixteen distinct-ID proposals from the popularity bank.
            stress = None
            for _ in range(16):
                proposed = np.asarray([rng.choice(items) for items in pools])
                if len(set(proposed.tolist())) == k:
                    value = variant(recent, proposed)
                    score = transition(value, recent)
                    if score < transition(original, recent) and (
                        stress is None or score < stress[0]
                    ):
                        stress = score, value
            if stress is not None:
                variants["low_transition_stress"] = stress[1]
            union = set().union(
                *(set(value.tolist()) for value in variants.values())
            ) | {target}
            negative_pool = np.setdiff1d(np.arange(1, self.itemnum + 1), list(union))
            if not len(negative_pool):
                failures["no_eligible_negative"] += 1
                continue
            candidates = np.r_[
                target,
                rng.choice(negative_pool, min(999, len(negative_pool)), replace=False),
            ]
            sequences = np.zeros((len(variants), self.maxlen), dtype=np.int64)
            for row, value in enumerate(variants.values()):
                sequences[row, -n:] = value
            assert np.all((sequences != 0) == (sequences[0] != 0))
            assert np.all(sequences[:, -1] == original[-1])
            draws.append((variants, sequences, candidates))
            changed_edges = [
                (value[e], value[e + 1])
                for value, slots_used in ((a, recent), (b, older))
                for e in sorted(
                    {
                        edge
                        for slot in slots_used
                        for edge in (int(slot) - 1, int(slot))
                        if 0 <= edge < n - 1
                    }
                )
            ]
            balance.append(
                {
                    "transition_difference": transition_delta,
                    "text_similarity_difference": text_delta,
                    "max_replacement_popularity_difference": float(
                        max(
                            np.abs(pop[replacement] - pop[original[recent]]).max(),
                            np.abs(pop[replacement] - pop[original[older]]).max(),
                        )
                    ),
                    "observed_changed_edge_fraction": float(
                        np.mean(
                            [
                                state["edges"][tuple(map(int, edge))] > 0
                                for edge in changed_edges
                            ]
                        )
                    ),
                    "excluded_catalog_items": len(union),
                }
            )
        counters.update(
            {key: counters.get(key, 0) + value for key, value in failures.items()}
        )
        if not draws:
            return None, "no_retained_replacement_draw"
        # Retain only readouts between bounded feature batches. This avoids
        # materializing attention masks for every draw/control simultaneously.
        sequences = np.concatenate([draw[1] for draw in draws])
        readouts = []
        for start in range(0, len(sequences), 8):
            chunk = sequences[start : start + 8]
            features = self.model.log2feats(chunk)
            if features.ndim != 3 or tuple(features.shape[:2]) != chunk.shape:
                raise ValueError(
                    "HSTU log2feats must return [variants, maxlen, hidden]"
                )
            seq_t = torch.as_tensor(chunk, device=features.device)
            last = (
                torch.arange(self.maxlen, device=features.device)
                .expand_as(seq_t)
                .masked_fill(seq_t == 0, -1)
                .amax(1)
            )
            if (last < 0).any():
                raise ValueError("Matched diagnostic histories must be nonempty")
            readouts.append(
                features[torch.arange(len(chunk), device=features.device), last].clone()
            )
            counters["sequence_evaluations"] += len(chunk)
            del features
        readout = torch.cat(readouts, dim=0)
        outcomes, effect_rows, counts, ties, offset = [], {}, [], [], 0
        scored_balance = []
        for draw_index, (variants, sequences, candidates) in enumerate(draws):
            rows = readout[offset : offset + len(variants)]
            offset += len(variants)
            pieces = []
            for start in range(0, len(candidates), 256):
                items = torch.as_tensor(
                    candidates[start : start + 256],
                    device=self.device,
                    dtype=torch.long,
                )
                pieces.append(
                    self.model.predict_score(rows, items).detach().float().cpu().numpy()
                )
                counters["candidate_score_calls"] += 1
            scores = np.concatenate(pieces, axis=-1)
            counters["candidate_scores"] += len(variants) * len(candidates)
            if scores.shape != (len(variants), len(candidates)):
                raise ValueError("HSTU scores must have shape [variants, candidates]")
            if not np.isfinite(scores).all():
                counters["nonfinite_score_draws"] = (
                    counters.get("nonfinite_score_draws", 0) + 1
                )
                continue
            lo = 1 + (scores[:, 1:] > scores[:, :1]).sum(1)
            equal = (scores[:, 1:] == scores[:, :1]).sum(1)
            hi = lo + equal
            harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, len(candidates) + 1))]
            rr = (harmonic[hi] - harmonic[lo - 1]) / (equal + 1)
            hit = np.maximum(0, np.minimum(hi, 10) - lo + 1) / (equal + 1)
            # Entry/exit indicators require membership to be certain under ties.
            entry = ((lo[0] > 10) & (hi <= 10)).astype(float)
            exit_ = ((hi[0] <= 10) & (lo > 10)).astype(float)
            values = np.stack(
                [lo, hi, (lo + hi) / 2, rr, hit, scores[:, 0], entry, exit_], axis=1
            )
            named = dict(zip(variants, values))
            outcomes.append(values[0])
            scored_balance.append(balance[draw_index])
            for name in list(variants)[1:]:
                effect_rows.setdefault(name, []).append(named[name] - values[0])
            effect_rows.setdefault("recent_minus_older", []).append(
                named["recent"] - named["older"]
            )
            counts.append(len(candidates))
            ties.append(float((equal > 0).mean()))
        if not outcomes:
            return None, "all_draws_nonfinite"
        return {
            "prefix_endpoint": int(position),
            "original": np.mean(outcomes, axis=0),
            "effects": {
                name: np.mean(values, axis=0) for name, values in effect_rows.items()
            },
            "effect_draw_counts": {
                name: len(values) for name, values in effect_rows.items()
            },
            "candidate_count": float(np.mean(counts)),
            "candidate_counts": counts,
            "tie_frequency": float(np.mean(ties)),
            "matching": scored_balance,
            "matched_draws_before_score_filter": len(draws),
            "analysis": (
                "text_popularity_transition"
                if text_mode
                else "popularity_transition_only"
            ),
            "window_distances_from_final": {
                "recent": (n - 1 - recent).tolist(),
                "older": (n - 1 - older).tolist(),
            },
            "retained_draws": len(outcomes),
            "failures": dict(failures),
        }, None

    def _history_audit(self, batches):
        records, coverage, exclusions = self._audit_prefixes(batches)
        # Matching statistics require complete identified training histories.
        if not isinstance(
            self.user_train.get("History", {}), dict
        ) or not self.user_train.get("History", {}):
            records = []
            exclusions["training_histories_unavailable_for_matching"] += 1
        self._audit_state = self._build_matching_state()
        counters = {
            "sequence_evaluations": 0,
            "candidate_score_calls": 0,
            "candidate_scores": 0,
            "replacement_attempts": 0,
        }
        grouped, alternatives = {}, {}
        counts, ties = [], []
        labels = (
            "rank_optimistic",
            "rank_pessimistic",
            "expected_rank",
            "reciprocal_rank",
            "hit10",
            "target_score",
            "definite_top10_entry",
            "definite_top10_exit",
        )

        def summarize(groups):
            users = []
            for uid, rows in sorted(groups.items()):
                names = sorted(set.union(*(set(row["effects"]) for row in rows)))
                effects = {
                    name: np.mean(
                        [
                            row["effects"][name]
                            for row in rows
                            if name in row["effects"]
                        ],
                        axis=0,
                    )
                    for name in names
                }
                users.append(
                    {
                        "user_index": int(uid),
                        "prefix_count": len(rows),
                        "prefix_endpoints": [row["prefix_endpoint"] for row in rows],
                        "effect_prefix_endpoints": {
                            name: [
                                row["prefix_endpoint"]
                                for row in rows
                                if name in row["effects"]
                            ]
                            for name in names
                        },
                        "effects": {
                            name: dict(zip(labels, map(float, value)))
                            for name, value in effects.items()
                        },
                        "effect_prefix_counts": {
                            name: sum(name in row["effects"] for row in rows)
                            for name in names
                        },
                    }
                )
            names = sorted({name for user in users for name in user["effects"]})
            return {
                "users": len(users),
                "prefixes": sum(map(len, groups.values())),
                "user_summaries": users,
                "mean_effects": {
                    name: {
                        label: float(
                            np.mean(
                                [
                                    user["effects"][name][label]
                                    for user in users
                                    if name in user["effects"]
                                ]
                            )
                        )
                        for label in labels
                    }
                    for name in names
                },
            }

        for index, position, history, target in records:
            result, reason = self._score_prefix(
                index, position, history, target, counters
            )
            if reason:
                exclusions[reason] += 1
                continue
            grouped.setdefault(index, []).append(result)
            counts.extend(result["candidate_counts"])
            ties.append(result["tie_frequency"])
        # Fixed descriptive protocols; never select tolerances by rank effects.
        protocols = (
            ("bank_1", 1, 1.0, None),
            ("bank_2", 2, 1.0, None),
            ("half_tolerances", 0, 0.5, None),
            ("double_tolerances", 0, 2.0, None),
            ("k_1", 0, 1.0, 1),
        )
        for name, bank, tolerance, window_k in protocols:
            groups, failed = {}, Counter()
            for index, position, history, target in records:
                value, reason = self._score_prefix(
                    index,
                    position,
                    history,
                    target,
                    counters,
                    bank,
                    tolerance,
                    window_k,
                )
                if reason:
                    failed[reason] += 1
                else:
                    groups.setdefault(index, []).append(value)
            alternatives[name] = {
                **summarize(groups),
                "exclusions": dict(failed),
                "bank_seed_components": [42, self.category_index, bank],
                "tolerance_multiplier": tolerance,
                "window_k": window_k,
            }
        reliable = coverage["source"] == "identified_training_users"
        coverage["prefixes_scored"] = sum(map(len, grouped.values()))
        coverage["scored_draws"] = len(counts)
        coverage["unique_users"] = len(grouped) if reliable else None
        coverage["rows_scored"] = len(grouped) if not reliable else None
        coverage["exclusions"] = dict(exclusions)
        # The labels above include exact tie bounds and checkpoint-scale scores.

        def as_metrics(values):
            return {label: float(value) for label, value in zip(labels, values)}

        baseline = None
        effects = {}
        if grouped:
            baseline = as_metrics(
                np.mean(
                    [
                        np.mean([record["original"] for record in rows], axis=0)
                        for rows in grouped.values()
                    ],
                    axis=0,
                )
            )
            names = sorted(
                {
                    name
                    for rows in grouped.values()
                    for record in rows
                    for name in record["effects"]
                }
            )
            for name in names:
                # Optional controls use available prefixes within each user,
                # then weight supported users equally, as user_summaries does.
                supported = {
                    uid: [record for record in rows if name in record["effects"]]
                    for uid, rows in grouped.items()
                }
                supported = {uid: rows for uid, rows in supported.items() if rows}
                clusters = np.asarray(
                    [
                        np.mean([record["effects"][name] for record in rows], axis=0)
                        for rows in supported.values()
                    ]
                )
                interval = None
                if reliable and len(supported) >= 2:
                    rng = np.random.default_rng(42)
                    bootstrap_indices = rng.integers(
                        0, len(supported), size=(10000, len(supported))
                    )
                    boot = clusters[bootstrap_indices].mean(axis=1)
                    lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
                    interval = {
                        label: [float(lo), float(hi)]
                        for label, lo, hi in zip(labels, lower, upper)
                    }
                effects[name] = {
                    "supported_users": len(supported),
                    "supported_prefixes": sum(map(len, supported.values())),
                    "supported_draws": sum(
                        record["effect_draw_counts"][name]
                        for rows in supported.values()
                        for record in rows
                    ),
                    "mean": as_metrics(clusters.mean(axis=0)),
                    "user_cluster_ci95": interval,
                    "ci_null_reason": (
                        "Loader rows have no verified user IDs"
                        if not reliable
                        else (
                            "At least two identified users are required for a user-cluster interval"
                            if len(supported) < 2
                            else None
                        )
                    ),
                    "cluster_effect_p10": as_metrics(
                        np.quantile(clusters, 0.1, axis=0)
                    ),
                    "cluster_effect_p90": as_metrics(
                        np.quantile(clusters, 0.9, axis=0)
                    ),
                }
        result = {
            "category": self.category,
            "user_summaries": summarize(grouped)["user_summaries"],
            "descriptive_protocols": {"bank_0": summarize(grouped), **alternatives},
            "replacement_bank_variability": {
                "scope": "Three independently seeded input-replacement banks; fixed checkpoint. Bank means can differ through retained support as well as replacement draws.",
                "bank_mean_effects": {
                    "bank_0": summarize(grouped)["mean_effects"],
                    **{
                        name: alternatives[name]["mean_effects"]
                        for name in ("bank_1", "bank_2")
                    },
                },
                "common_user_mean_effects": {
                    name: {
                        user["user_index"]: user["effects"]
                        for user in summary["user_summaries"]
                        if user["user_index"]
                        in set.intersection(
                            *[
                                {
                                    entry["user_index"]
                                    for entry in candidate["user_summaries"]
                                }
                                for candidate in (
                                    summarize(grouped),
                                    alternatives["bank_1"],
                                    alternatives["bank_2"],
                                )
                            ]
                        )
                    }
                    for name, summary in (
                        ("bank_0", summarize(grouped)),
                        ("bank_1", alternatives["bank_1"]),
                        ("bank_2", alternatives["bank_2"]),
                    )
                },
            },
            "text_strata": {
                mode: summarize(
                    {
                        uid: selected
                        for uid, rows in grouped.items()
                        if (
                            selected := [row for row in rows if row["analysis"] == mode]
                        )
                    }
                )
                for mode in ("text_popularity_transition", "popularity_transition_only")
            },
            "matching_details": [
                {
                    "user_index": int(uid),
                    "prefix_index_within_user": prefix,
                    "prefix_endpoint": row["prefix_endpoint"],
                    "analysis": row["analysis"],
                    "matching": row["matching"],
                    "retained_draws": row["retained_draws"],
                    "matched_draws_before_score_filter": row[
                        "matched_draws_before_score_filter"
                    ],
                    "effect_draw_counts": row["effect_draw_counts"],
                    "failures": row["failures"],
                    "window_distances_from_final": row["window_distances_from_final"],
                }
                for uid, rows in grouped.items()
                for prefix, row in enumerate(rows)
            ],
            "matching_statistics": {
                "training_items": len(self._audit_state["ids"]),
                "training_items_with_text": len(self._audit_state["text_items"]),
                "training_edge_count": self._audit_state["training_edge_count"],
                "training_edge_log_probability_sd": self._audit_state["transition_sd"],
                "text_null_reason": self._audit_state["text_reason"],
            },
            "leave_one_user_out": (
                {
                    str(uid): summarize(
                        {other: rows for other, rows in grouped.items() if other != uid}
                    )["mean_effects"]
                    for uid in grouped
                }
                if 2 <= len(grouped) <= 5
                else None
            ),
            "category_null_reason": (
                None
                if self.category
                else "ModelView does not identify a matched category"
            ),
            "coverage": coverage,
            "cost_counts": counters,
            "original": baseline,
            "effects": effects,
            "null_reason": None if grouped else "No complete eligible matched prefixes",
            "candidate_counts": {
                "min": int(min(counts)) if counts else None,
                "max": int(max(counts)) if counts else None,
                "mean": float(np.mean(counts)) if counts else None,
                "fraction_at_most_10": (
                    float(np.mean(np.asarray(counts) <= 10)) if counts else None
                ),
            },
            "target_tie_frequency": (
                float(
                    np.mean(
                        [
                            np.mean([record["tie_frequency"] for record in rows])
                            for rows in grouped.values()
                        ]
                    )
                )
                if grouped
                else None
            ),
            "undefined_effects": {
                name: "No finite scored draw supports this optional control"
                for name in ("random_nonfinal", "low_transition_stress")
                if name not in effects
            },
            "protocol": {
                "seed": 42,
                "maxlen": self.maxlen,
                "negative_cap": 999,
                "replacement_draws_per_prefix": 5,
                "replacement_banks": 3,
                "attempt_cap_per_draw": 100,
                "bootstrap_replicates": 10000,
                "primary_uncertainty_bank": 0,
                "feature_batch_cap": 8,
                "effect_support": "Available draws within prefixes, available prefixes within users, then supported users equally; each effect reports user/prefix/draw counts. Bootstrap resamples supported whole users. Optional-control intervals condition on their own retained support.",
                "input_channels": "Actual HSTU ModelView item-ID sequence interface",
                "timestamp_control": "Unavailable: log2feats exposes item-ID sequences only",
                "position_policy": "Fixed occupied slots, padding and final item; older first k versus recent k immediately before final, k=min(2,floor((n-1)/2))",
                "popularity": "log1p(training interaction count), standardized across training-observed items; originals and each replacement must differ by at most 0.5 SD",
                "transition": "Add-one P(b|a)=(count(a,b)+1)/(outgoing(a)+itemnum); average log probability across unique changed edges, including both boundaries; paired difference <=0.5 training-edge SD",
                "text": "Training-observed items only; local normalized title/description TF-IDF; paired mean cosine difference <=0.10; missing original text is a separate popularity/transition-only stratum",
                "threshold_status": "Preregistered descriptive choices, not established optima; half/double tolerances and k=1 are reported without outcome-based selection",
                "identity_scope": "Category-local identified training users; simulator cohort is unchanged",
                "interval_scope": "Whole-user bootstrap conditional on one frozen checkpoint, bank 0, retained prefixes and fixed candidates; small-cluster coverage may be poor",
                "selection_metric": "Host macro validation HR@10 remains unchanged",
            },
        }
        # A probe receives one actual trained category model. Do not manufacture
        # a four-category estimate from one category or pool local user IDs.
        return {
            "per_category": {
                name: result if name == self.category else None
                for name in self.datasets
            },
            "unidentified_category_result": result if self.category is None else None,
            "macro_four_category_mean": None,
            "macro_four_category_ci95": None,
            "macro_null_reason": "This invocation contains one category checkpoint; all four category user-cluster records are required to recompute a macro estimate and bootstrap",
        }

    def run_full_diagnosis(self, train_loader):
        import time
        from itertools import islice

        print(">>> Running frozen HSTU training-prefix diagnosis...")
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        generators = []
        for owner in (train_loader, getattr(train_loader, "sampler", None)):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator) and all(
                generator is not old for old, _ in generators
            ):
                generators.append((generator, generator.get_state()))
        # Save all CUDA RNG states, including when the model itself is on CPU.
        rng_devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        devices = [self.device] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                for generator, _ in generators:
                    generator.manual_seed(42)
                self.network.eval()
                if devices:
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(torch.cuda.current_stream(self.device))
                start = time.perf_counter()
                batches = list(islice(iter(train_loader), 10))
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=10, eps=eps, topk_categories=20
                    )
                )
                audit = self._history_audit(batches)
                gpu_seconds, peak = None, None
                if devices:
                    end_event.record(torch.cuda.current_stream(self.device))
                    torch.cuda.synchronize(self.device)
                    gpu_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
                    peak = int(torch.cuda.max_memory_allocated(self.device))
                wall_seconds = time.perf_counter() - start
                breakdown["null_reason"] = (
                    "Item-category attributes are unavailable; raw counts would not adjust for exposure"
                )
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "embedding_null_reason": (
                            None
                            if collapse_score is not None
                            else "Fewer than two items, zero-norm embeddings, or nonfinite embeddings"
                        ),
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_null_reason": (
                            None
                            if margin_metrics["mean"] is not None
                            else "No finite valid sampled training pairs"
                        ),
                        "pos_neg_margin_category_breakdown": breakdown,
                        "history_interventions": audit,
                        "diagnostic_cost": {
                            "wall_seconds": float(wall_seconds),
                            "cuda_event_seconds": gpu_seconds,
                            "peak_allocated_bytes": peak,
                            "gpu_null_reason": None if devices else "CPU execution",
                        },
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "(Mean off-diagonal cosine + 1)/2 for up to 2048 effective ordinary-item embeddings sampled with local seed 42. Approximately 0.5 means mean cosine zero. This measures angular alignment, not recommendation diversity or representation rank. Null for undefined inputs.",
                        "pos_neg_margin_mean": "Mean positive-minus-loader-negative score over finite, nonpadding, ordinary-item training positions in at most ten batches. Supplied HSTU features are causal: position t sees sequence[:t+1], not later events.",
                        "pos_neg_margin_p10": "Nearest-order-statistic 10th percentile of sampled pair margins.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of valid sampled pairs with negative score strictly greater than positive score.",
                        "pos_neg_small_margin_rate": "Fraction of valid sampled pairs with positive-minus-negative score below eps=0.5; depends on score scale.",
                        "pos_neg_margin_category_breakdown": "Empty category lists because item categories are unavailable. Position counts remain descriptive; no exposure-adjusted conclusion is supported.",
                        "history_interventions": "Frozen fitted-HSTU sensitivity on training targets only; not benchmark validation. Deterministically select up to 32 training users with at least seven events and two distinct endpoints with at least five occupied context events. Training-only counts, transitions and optional title/description TF-IDF define matching; replacement IDs are training-observed, unique, and exclude padding, target and prefix items. Five paired draws reuse identical ordered replacements at disjoint older/recent nonfinal slots. Padding, occupied positions, final item and greatest-nonpadding readout remain fixed. Matching thresholds and sensitivity protocols are stated in protocol; failures and residual imbalance are reported. Each draw uses one shared target plus up to 999 unique uniform negatives excluding the union of all compared histories; this conditional diagnostic catalog never changes the pinned evaluator. With g strictly higher-scoring negatives and e exactly tied negatives, optimistic rank=1+g, pessimistic rank=1+g+e; expected_rank is their midpoint, reciprocal_rank is the exact mean of 1/r over all e+1 positions, and hit10 is the fraction of those positions <=10. Reciprocal midpoint rank is not used. Effects are perturbed-minus-original; positive rank is worse and positive reciprocal rank/hit10 is better. Target-score differences depend on checkpoint scale and must remain category-specific. Definite top10 entry requires original optimistic rank>10 and perturbed pessimistic rank<=10; definite exit requires original pessimistic rank<=10 and perturbed optimistic rank>10. Tie-ambiguous transitions are not counted as definite entry/exit. Identity differences should be zero. Optional random control uses the same replacement list and additional matching checks. Stress control chooses the lowest recent-window transition likelihood among up to sixteen popularity-matched proposals, strictly below original; it does not establish semantic corruption. Draws average within prefixes, then prefixes within users, then users equally. Primary effects use controls present in every retained primary prefix; optional available-case support is explicit in user summaries. Whole-user percentile bootstrap uses 10000 resamples, seed 42, bank 0, retaining overlapping prefixes together; fewer than two users yields null intervals. These descriptive intervals omit training, candidate-generation, matching-selection and replacement-bank uncertainty. Three bank summaries report Monte Carlo/support variability separately; common-user summaries still need not share retained prefixes. Text-complete and missing-text strata are separately reported. Half/double thresholds and k=1 are descriptive and may change support. A four-category macro estimate requires all four identified categories and independent within-category user resampling by the host; do not average interval endpoints, pool user IDs, or macro-average checkpoint-scale target scores. Model sensitivity under these content/location interventions does not identify absolute position, elapsed-time effects, intent, any HSTU component, or generalization. Diagnostic users do not alter the simulator cohort.",
                        "history_intervention_support": "Effect support follows protocol.effect_support: effects are reported whenever at least one finite draw supports them, including optional controls absent from other prefixes. This replaces the all-prefix control requirement in the history_interventions description. Intervals require at least two supported users and condition on effect-specific retained support. Prefix endpoints identify actual training targets for comparing support across replacement banks; common users alone do not establish common prefixes. Matching details include finite scored draws only; matched_draws_before_score_filter also counts draws subsequently excluded for nonfinite scores. Matching rejection counters count proposal attempts, whereas coverage exclusions count prefixes. Definite entry/exit fields for recent_minus_older are differences between each intervention's entry/exit rates relative to original, not direct older-to-recent transition rates.",
                        "diagnostic_cost": "Wall time includes loader access and all probes. CUDA-event elapsed time measures the diagnostic stream interval, including host-induced idle gaps, not kernel-active GPU time. Peak allocated memory is the absolute PyTorch allocation high-water mark after resetting it at audit entry. History cost_counts record actual sequence evaluations, candidate-score calls and scalar candidate scores. CPU GPU statistics are null. No claim of inexpensive auditing is made.",
                    },
                }
        finally:
            self.__dict__.pop("_audit_state", None)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in generators:
                generator.set_state(state)
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END
