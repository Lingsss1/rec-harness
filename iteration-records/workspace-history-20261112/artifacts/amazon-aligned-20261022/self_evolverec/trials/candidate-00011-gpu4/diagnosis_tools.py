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
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Bind the actual trained module and exclude unavailable category evidence
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.review_data = getattr(view, "review_data", {}) or {}
        self.user_train = view.user_train
        self._counts = {}

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this benchmark.
        return []

    def _features(self, seq):
        return self.view.log2feats(seq)

    def _scores(self, features, items, seq, positions=None):
        import inspect

        # Preserve architecture-specific residuals when the actual module exposes
        # explicit history-aware scoring. Otherwise use the compatibility hook.
        scorer = getattr(self.model, "predict_score", self.view.predict_score)
        parameters = inspect.signature(scorer).parameters
        if "history_ids" in parameters:
            kwargs = {"history_ids": seq}
            if "position_ids" in parameters:
                kwargs["position_ids"] = positions
            elif positions is not None and not bool(
                positions.eq(seq.shape[1] - 1).all()
            ):
                raise ValueError("Scorer cannot select the requested history positions")
            values = scorer(features, items, **kwargs)
        else:
            # DEBUG: The compatibility hook supports aligned B x L targets;
            # score each candidate column separately for B x L x C inputs.
            if features.ndim == 3 and items.ndim == 3:
                values = torch.stack(
                    [
                        self.view.predict_score(features, column)
                        for column in items.unbind(dim=-1)
                    ],
                    dim=-1,
                )
            else:
                values = self.view.predict_score(features, items)
        expected = (
            (features.shape[0], items.shape[-1])
            if features.ndim == 2
            else tuple(items.shape)
        )
        if not isinstance(values, torch.Tensor) or tuple(values.shape) != expected:
            raise ValueError(f"Diagnostic scorer must return shape {expected}")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite diagnostic scores")
        return values

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Use the public embedding compatibility hook
            emb = self.view.item_emb.weight.detach()
            ### <<< Self_EvolveRec-BLOCK-END
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Report embedding sample coverage and unavailable estimates
            self._counts["embedding_sample_items"] = min(sample_size, n)
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute sampled mean cosine without a quadratic Gram matrix
            if not bool(torch.isfinite(x).all()):
                raise ValueError("Nonfinite sampled item embeddings")
            # Sum of all pairwise dot products equals the squared norm of the sum.
            # Subtract actual diagonal norms, including any zero embeddings.
            x = x.double()
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
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
        ### >>> Self_EvolveRec-BLOCK-START: Collect final valid-position comparisons without changing model mode
        margins_all = []
        final_margins = []
        self._counts.update({"batches_seen": 0, "batches_skipped": 0})
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume exactly the requested diagnostic batches
        from itertools import islice

        for batch in islice(dataloader, max(0, num_batches)):
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Score training pairs through the trained architecture with explicit histories
            self._counts["batches_seen"] += 1
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                self._counts["batches_skipped"] += 1
                continue
            seq_t, pos_t, neg_t = [
                torch.as_tensor(x, device=self.device, dtype=torch.long)
                for x in batch[1:4]
            ]
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnosis batches must contain aligned B x L tensors")
            if seq_t.shape[0] == 0 or seq_t.shape[1] == 0:
                self._counts["batches_skipped"] += 1
                continue
            log_feats = self._features(seq_t)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self._scores(log_feats, all_items, seq_t)
            margin = all_logits[..., 0] - all_logits[..., 1]
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())
            indices = torch.arange(seq_t.shape[1], device=self.device).expand_as(seq_t)
            last = indices.masked_fill(~seq_t.ne(0), -1).amax(dim=1)
            rows = torch.arange(seq_t.shape[0], device=self.device)
            eligible = last.ge(0) & valid[rows, last.clamp_min(0)]
            if eligible.any():
                final_margins.append(
                    margin[rows[eligible], last[eligible]].float().cpu()
                )
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Reuse the complete training-pair eligibility mask
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
                ### >>> Self_EvolveRec-BLOCK-START: Distinguish empty samples from zero-valued measurements
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

        ### >>> Self_EvolveRec-BLOCK-START: Preserve legacy outputs while recording final-position evidence
        self._counts["valid_pair_positions"] = sum(x.numel() for x in margins_all)
        final = torch.cat(final_margins) if final_margins else torch.empty(0)
        self._final_pairs = {
            "count": final.numel(),
            "mean_margin": float(final.mean()) if final.numel() else None,
            "positive_win_rate": (
                float((final > 0).float().mean()) if final.numel() else None
            ),
            "tie_rate": float((final == 0).float().mean()) if final.numel() else None,
        }
        breakdown["category_evidence_available"] = False
        breakdown["unavailable_reason"] = (
            "Item categories are not observed in this benchmark"
        )
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Audit frozen training histories with anchored interventions and matched controls
    def _history_audit(self):
        import time
        import math
        import re

        # DEBUG: Implement the declared word unigram/bigram TF-IDF proxy using
        # standard-library sparse dictionaries; scikit-learn is not required.
        def build_tfidf(documents):
            rows = []
            document_frequency = Counter()
            for document in documents:
                tokens = re.findall(r"(?u)\b\w\w+\b", document.lower())
                terms = tokens + [
                    left + " " + right for left, right in zip(tokens, tokens[1:])
                ]
                row = Counter(terms)
                rows.append(row)
                document_frequency.update(row.keys())
            if not document_frequency:
                raise ValueError("Empty vocabulary in observed titles/descriptions")
            count = len(documents)
            idf = {
                term: math.log((1.0 + count) / (1.0 + frequency)) + 1.0
                for term, frequency in document_frequency.items()
            }
            normalized = []
            for row in rows:
                weighted = {
                    term: frequency * idf[term] for term, frequency in row.items()
                }
                norm = math.sqrt(sum(value * value for value in weighted.values()))
                normalized.append(
                    {term: value / norm for term, value in weighted.items()}
                    if norm
                    else {}
                )
            return normalized

        started = time.monotonic()
        # DEBUG: A checkpoint diagnostic must respect the enclosing trial cap.
        deadline = min(
            started + 120.0,
            float(getattr(self.view, "diagnosis_deadline", float("inf"))),
        )
        raw = self.user_train.get("History", self.user_train)
        if not isinstance(raw, dict):
            raise ValueError("Expected a training-history mapping")
        itemnum = self.view.item_emb.num_embeddings - 1
        histories = {}
        for user, values in raw.items():
            history = [int(x) for x in values]
            if any(x < 1 or x > itemnum for x in history):
                raise ValueError("Training history contains an out-of-catalog item")
            histories[str(user)] = tuple(history)
        nonempty = [h for h in histories.values() if h]
        events = sum(map(len, nonempty))
        repeated = sum(len(set(h)) < len(h) for h in nonempty)
        popularity = Counter(x for h in nonempty for x in h)
        catalog = np.arange(1, itemnum + 1, dtype=np.int64)
        cuts = (
            np.quantile([popularity[int(x)] for x in catalog], [0.25, 0.5, 0.75])
            if itemnum
            else []
        )
        length = int(getattr(self.model, "maxlen", 128))
        if length <= 0:
            raise ValueError("Diagnostic history length must be positive")
        # Proxy selection uses training recurrence coverage only, never validation.
        use_exact = bool(nonempty) and repeated / len(nonempty) >= 0.1
        text_ids, texts = [], []
        for item in sorted(popularity):
            info = self.meta_data.get(str(item), self.meta_data.get(item, {}))
            if not isinstance(info, dict):
                continue
            pieces = []
            for field in ("title", "description"):
                value = info.get(field, "")
                if isinstance(value, list):
                    value = " ".join(x for x in value if isinstance(x, str))
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
            text = " ".join(pieces)
            if text:
                text_ids.append(item)
                texts.append(text)
        matrix, text_index = None, {}
        lexical_error = None
        if not use_exact and texts:
            try:
                # DEBUG: Preserve smooth IDF, raw term frequency and L2 cosine.
                matrix = build_tfidf(texts)
                text_index = {item: i for i, item in enumerate(text_ids)}
            except ValueError as exc:
                lexical_error = str(exc)
        elif not use_exact:
            lexical_error = "No observed nonempty titles or descriptions"

        route = getattr(
            self.view, "dataset_name", getattr(self.model, "dataset_name", None)
        )
        categories = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        report = {
            "dataset_name": route if route in categories else None,
            "route_status": (
                "observed"
                if route in categories
                else "unavailable; no category inferred from item metadata"
            ),
            "category_coverage": {
                c: "current_view" if c == route else "not_observed_by_this_view"
                for c in categories
            },
            "training_users": len(histories),
            "nonempty_users": len(nonempty),
            "repeated_user_fraction": repeated / len(nonempty) if nonempty else None,
            "duplicate_event_fraction": (
                sum(len(h) - len(set(h)) for h in nonempty) / events if events else None
            ),
            "text_coverage": len(text_ids) / len(popularity) if popularity else None,
            "proxy": "exact" if use_exact else "training_catalog_tfidf_cosine",
            "exact_proxy_minimum_repeated_user_fraction": 0.1,
            "lexical_error": lexical_error,
            "mask_fixed_positions": {
                "available": False,
                "reason": "Internal-padding attention safety is not certified; model semantics unchanged",
            },
            "controls": {
                "seeds": list(range(42, 47)),
                "strata": "joint relative-position quartile and training-frequency quartile; anchor excluded",
            },
            "records": [],
            "ineligible": Counter(),
            "infeasible_control_seeds": 0,
            "threshold_coverage": {
                str(t): {"users_with_targets": 0, "users_examined": 0}
                for t in (0.4, 0.5, 0.6)
            },
            "lexical_examples": [],
            "determinism_checks": [],
            "time_bound_reached": False,
            "interrupted_users": 0,
            "threshold_coverage_population": "Users with at least two distinct observed-text items in the visible history",
            "lexical_match_rule": "Distinct item IDs only; repeated occurrences of the same item are not cross-item lexical evidence",
            "native_simulator_panel": "Separate exploratory panel; not sampled or modified here",
            "timestamp_matching": "Not used; no missing timestamps imputed",
        }
        users = sorted(histories)
        random.Random(42).shuffle(users)
        records = report["records"]
        eligible_users = examined = 0

        def score(history, candidates):
            if not history:
                raise ValueError("Empty intervention history")
            seq = torch.zeros((1, length), dtype=torch.long, device=self.device)
            tail = history[-length:]
            seq[0, -len(tail) :] = torch.tensor(tail, device=self.device)
            indices = torch.arange(length, device=self.device)
            last = indices.masked_fill(seq[0].eq(0), -1).max()
            features = self._features(seq)[:, last, :]
            values = []
            for start in range(0, len(candidates), 4096):
                if time.monotonic() >= deadline:
                    raise TimeoutError("History audit time bound reached")
                ids = torch.as_tensor(
                    candidates[start : start + 4096], device=self.device
                ).unsqueeze(0)
                values.append(
                    self._scores(features, ids, seq, last.reshape(1))
                    .reshape(-1)
                    .float()
                    .cpu()
                    .numpy()
                )
            return np.concatenate(values)

        def order(values, candidates):
            return np.lexsort((candidates, -values))

        def changes(baseline_order, altered, candidates, k):
            altered_order = order(altered, candidates)
            ranks = np.empty(len(altered_order), dtype=np.int64)
            ranks[altered_order] = np.arange(len(altered_order))
            displacement = ranks[baseline_order[:k]] - np.arange(k)
            return np.array(
                [
                    len(set(baseline_order[:k]) & set(altered_order[:k])) / k,
                    displacement.mean(),
                    np.abs(displacement).mean(),
                ],
                dtype=float,
            )

        try:
            for user in users:
                if time.monotonic() >= deadline:
                    report["time_bound_reached"] = True
                    break
                if eligible_users >= 20 or examined >= 1000:
                    break
                examined += 1
                original = histories[user]
                history = original[-length:]
                if not history:
                    report["ineligible"]["empty_original_history"] += 1
                    continue
                anchor = len(history) - 1
                if use_exact:
                    counts = Counter(history)
                    target = [i for i in range(anchor) if counts[history[i]] > 1]
                else:
                    supported = [i for i, x in enumerate(history) if x in text_index]
                    target = []
                    if matrix is not None and len({history[i] for i in supported}) >= 2:
                        # DEBUG: Compute the small visible-history Gram matrix
                        # directly from standard-library sparse TF-IDF rows.
                        sub = [matrix[text_index[history[i]]] for i in supported]
                        similarities = np.zeros((len(sub), len(sub)), dtype=float)
                        for left in range(len(sub)):
                            for right in range(left, len(sub)):
                                a, b = sub[left], sub[right]
                                if len(a) > len(b):
                                    a, b = b, a
                                cosine = sum(
                                    value * b.get(term, 0.0)
                                    for term, value in a.items()
                                )
                                similarities[left, right] = cosine
                                similarities[right, left] = cosine
                        supported_ids = np.asarray([history[i] for i in supported])
                        # Exclude both self-pairs and duplicate-item pairs. The
                        # lexical fallback measures recurrence across items.
                        similarities[
                            supported_ids[:, None] == supported_ids[None, :]
                        ] = -1
                        for threshold in (0.4, 0.5, 0.6):
                            positions = [
                                supported[j]
                                for j in range(len(supported))
                                if supported[j] != anchor
                                and similarities[j].max() >= threshold
                            ]
                            coverage = report["threshold_coverage"][str(threshold)]
                            coverage["users_examined"] += 1
                            coverage["users_with_targets"] += bool(positions)
                            if threshold == 0.5:
                                target = positions
                        if target and len(report["lexical_examples"]) < 8:
                            i = target[0]
                            j = supported[
                                int(similarities[supported.index(i)].argmax())
                            ]
                            report["lexical_examples"].append(
                                {
                                    "item_ids": [history[i], history[j]],
                                    "observed_text": [
                                        texts[text_index[history[i]]][:300],
                                        texts[text_index[history[j]]][:300],
                                    ],
                                    "cosine": float(
                                        similarities[
                                            supported.index(i), supported.index(j)
                                        ]
                                    ),
                                    "interpretation": "Lexical match only; boilerplate and edition matches require human inspection",
                                }
                            )
                    else:
                        report["ineligible"]["insufficient_observed_text"] += 1
                        continue
                if not target:
                    report["ineligible"]["no_proxy_evidence"] += 1
                    continue
                excluded = set(original)
                candidates = catalog[~np.isin(catalog, list(excluded))]
                if not len(candidates):
                    report["ineligible"]["empty_candidate_catalog"] += 1
                    continue
                target_set = set(target)

                def stratum(i):
                    return (
                        min(3, 4 * i // len(history)),
                        int(
                            np.searchsorted(cuts, popularity[history[i]], side="right")
                        ),
                    )

                needed = Counter(stratum(i) for i in target)
                pools = {}
                for i in range(anchor):
                    if i not in target_set:
                        pools.setdefault(stratum(i), []).append(i)
                controls = []
                for seed in range(42, 47):
                    if any(
                        len(pools.get(key, [])) < count for key, count in needed.items()
                    ):
                        report["infeasible_control_seeds"] += 1
                        continue
                    rng = random.Random(seed)
                    selected = []
                    for key, count in sorted(needed.items()):
                        selected.extend(rng.sample(pools[key], count))
                    selected = set(selected)
                    if (
                        anchor in selected
                        or selected & target_set
                        or len(selected) != len(target_set)
                        or Counter(stratum(i) for i in selected) != needed
                    ):
                        raise ValueError("Invalid matched intervention control")
                    controls.append((seed, selected))
                if not controls:
                    report["ineligible"]["no_feasible_controls"] += 1
                    continue
                baseline = score(history, candidates)
                baseline_order = order(baseline, candidates)
                k = min(10, len(candidates))
                user_records = []
                for operation in ("delete_repack", "retain_repack"):

                    def intervene(positions):
                        return tuple(
                            x
                            for i, x in enumerate(history)
                            if (
                                i not in positions
                                if operation == "delete_repack"
                                else i in positions or i == anchor
                            )
                        )

                    altered = intervene(target_set)
                    expected_length = (
                        len(history) - len(target_set)
                        if operation == "delete_repack"
                        else len(target_set) + 1
                    )
                    if (
                        not altered
                        or altered[-1] != history[anchor]
                        or len(altered) != expected_length
                    ):
                        raise ValueError("Invalid anchored history intervention")
                    target_scores = score(altered, candidates)
                    target_change = changes(
                        baseline_order, target_scores, candidates, k
                    )
                    control_changes = [
                        changes(
                            baseline_order,
                            score(intervene(p), candidates),
                            candidates,
                            k,
                        )
                        for _, p in controls
                    ]
                    control_array = np.stack(control_changes)
                    if len(report["determinism_checks"]) < 2:
                        rerun = score(altered, candidates)
                        identical_ranking = bool(
                            np.array_equal(
                                order(rerun, candidates),
                                order(target_scores, candidates),
                            )
                        )
                        check = {
                            "user": user,
                            "operation": operation,
                            "max_abs_score_error": float(
                                np.max(np.abs(rerun - target_scores))
                            ),
                            "identical_ranking": identical_ranking,
                            "scores_close": bool(
                                np.allclose(rerun, target_scores, rtol=1e-5, atol=1e-6)
                            ),
                        }
                        report["determinism_checks"].append(check)
                        if not check["scores_close"] or not identical_ranking:
                            import warnings

                            warnings.warn(
                                "Frozen history audit rerun failed reproducibility checks",
                                RuntimeWarning,
                            )
                    user_records.append(
                        {
                            "user": user,
                            "operation": operation,
                            "k": k,
                            "candidate_count": len(candidates),
                            "history_length": len(history),
                            "target_positions": target,
                            "anchor_position": anchor,
                            "control_positions": {
                                str(seed): sorted(p) for seed, p in controls
                            },
                            "target": target_change.tolist(),
                            "controls": control_array.tolist(),
                            "target_minus_control_mean": (
                                target_change - control_array.mean(axis=0)
                            ).tolist(),
                            "control_seed_std": control_array.std(axis=0).tolist(),
                        }
                    )
                # Commit only users with both primary operations completed.
                records.extend(user_records)
                eligible_users += 1
        except TimeoutError:
            report["time_bound_reached"] = True
            report["interrupted_users"] += 1
        report["determinism_status"] = (
            "unavailable"
            if not report["determinism_checks"]
            else (
                "passed"
                if all(
                    c["scores_close"] and c["identical_ranking"]
                    for c in report["determinism_checks"]
                )
                else "failed"
            )
        )
        report["users_examined"] = examined
        report["users_completed"] = eligible_users
        report["ineligible"] = dict(report["ineligible"])
        report["runtime_seconds"] = time.monotonic() - started
        report["scan_limit"] = 1000
        # DEBUG: Report the actual audit allowance under the shared deadline.
        report["time_limit_seconds"] = max(0.0, deadline - started)
        report["deadline_policy"] = (
            "Soft wall-time deadline checked between users and scoring chunks; "
            "metadata processing, TF-IDF fitting and individual kernels are not preempted"
        )
        report["summary"] = {}
        for operation in ("delete_repack", "retain_repack"):
            for k in sorted({r["k"] for r in records if r["operation"] == operation}):
                subset = [
                    r for r in records if r["operation"] == operation and r["k"] == k
                ]
                report["summary"][f"{operation}@{k}"] = {
                    "count": len(subset),
                    "mean_target_minus_control": np.mean(
                        [r["target_minus_control_mean"] for r in subset], axis=0
                    ).tolist(),
                }
        return report

    def run_full_diagnosis(self, train_loader):
        import time

        started = time.monotonic()
        # Diagnosis must not perturb subsequent training draws, including those
        # consumed by a loader using the global random generators.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        # Preserve mixed child-module modes as well as the root training flag.
        modes = [(module, module.training) for module in self.model.modules()]
        self._counts = {}
        try:
            self.model.eval()
            with torch.no_grad():
                collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
                margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=0.5, topk_categories=20
                )
                audit = self._history_audit()
            metrics = {
                "embedding_collapse_score": collapse,
                "pos_neg_margin_mean": margin["mean"],
                "pos_neg_margin_p10": margin["p10"],
                "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "sample_counts": dict(self._counts),
                "final_position_training_pairs": self._final_pairs,
                "history_intervention_audit": audit,
                "runtime_seconds": time.monotonic() - started,
            }
            definitions = {
                "embedding_collapse_score": "(Mean off-diagonal cosine + 1)/2 for up to 2048 nonpadding item embeddings sampled with seed 42. Directional similarity, not behavioral diversity; null for fewer than two items.",
                "pos_neg_margin_mean": "Mean positive-minus-negative logit over nonpadding sequence positions with nonzero targets; null for no samples.",
                "pos_neg_margin_p10": "Nearest-rank lower decile of the same training-pair margins; null for no samples.",
                "pos_neg_neg_beats_pos_rate": "Fraction of valid training pairs with margin < 0; null for no samples.",
                "pos_neg_small_margin_rate": "Fraction of valid training pairs with margin < 0.5; null for no samples.",
                "pos_neg_margin_category_breakdown": "Legacy low/wrong-margin position counts retained; item-category breakdown unavailable because categories are unobserved.",
                "sample_counts": "Actual embedding, batch and valid-pair counts. Empty evidence is not reported as a measured zero.",
                "final_position_training_pairs": "At each row's greatest nonzero sequence index: margin, positive win rate and tie rate against its supplied training negative. Not validation HR@10 or a catalog ranking metric.",
                "history_intervention_audit": (
                    "One current-category trained checkpoint, unchanged in eval mode. Seed-42 shuffled training users; scan at most 1000 and complete at most 20 under a soft 120-second deadline checked between users and scoring chunks. Preprocessing and individual operations are not preempted. "
                    "Exact recurrence is used if at least 10% of nonempty full training histories repeat an item; otherwise use training-catalog title/description TF-IDF word unigrams/bigrams, lowercase, L2 cosine >=0.5, with prespecified 0.4/0.6 coverage sensitivity. "
                    "Exact recurrence targets earlier repeated-item positions. Lexical recurrence requires a different item ID in the visible history, including the anchor as possible evidence; identical-item pairs are excluded. Threshold coverage uses users with at least two distinct observed-text items. Missing text is unavailable evidence. "
                    "Duplicate-event fraction is sum(length-unique_count)/sum(length); repeated-user fraction uses nonempty histories. Deduplicated histories do not establish absent recurring interests. "
                    "Both operations preserve the original last-item anchor and chronologically repack with left padding. Removal/retention controls are disjoint from targets and match joint position/popularity quartiles exactly under seeds 42-46; infeasible controls are recorded without relaxation. "
                    "Every original full-history item remains excluded from the fixed ascending full-category candidate catalog. Vectors contain overlap@k, mean signed rank displacement and mean absolute rank displacement for baseline top-k items; positive displacement means demotion. Ties use ascending item ID. "
                    "Differences are targeted minus mean control; control standard deviation uses ddof=0. Reduced k groups are reported separately. Position/spacing changes limit causal interpretation. "
                    "No fixed-position masking is performed without attention-safety certification. Route coverage is explicit; this probe never applies one category model to another category. "
                    "Lexical matches do not establish ownership, compatibility, intent, diversity or conformity. Simulator traits, reviews and item categories/ratings/prices/stores are unobserved. "
                    "The native simulator panel remains separate. No validation or test labels enter this audit; external equal-weight four-category validation HR@10 remains the selection metric."
                ),
                "runtime_seconds": "Diagnostic wall time including all probes; contributes to the shared search budget.",
            }
            return {"metrics": metrics, "metric_definitions": definitions}
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    ### <<< Self_EvolveRec-BLOCK-END
