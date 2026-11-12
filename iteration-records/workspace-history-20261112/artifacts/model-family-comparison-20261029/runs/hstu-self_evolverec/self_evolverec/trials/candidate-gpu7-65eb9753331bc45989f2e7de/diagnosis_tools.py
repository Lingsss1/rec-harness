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
    ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic interpretation to observed evidence
    - Every statement must be grounded in the provided metric values/definitions.
    - Report failed, unsupported and unexercised contracts first. Separate independent
      scoring-oracle evidence, consistency relations and deployment parity.
    - Passing checks cover only exercised properties; they neither diagnose nor rule
      out shortcuts, preferences, generalization or validation improvement.
    - A surviving exercised mutant is not automatically equivalent. State the tested
      observable and denominator for any mutation summary.
    - Training-prefix probes are in-sample diagnostics, not validation estimates.
    - Null means unavailable, never zero. Failed controls invalidate perturbation interpretations.
    - Do not infer causal preference, compatibility, current demand, fulfilled needs,
      simulator traits, or expected validation gains from these measurements.
    - Candidate-set overlap measures ranking stability; it is not catalog diversity.
    - Mean embedding cosine is not effective rank or a relevance measurement.
    - Do not recommend ensembles or combine independently fitted predictors.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the HSTU ModelView and public-data boundary
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            # Sample IDs before lookup: lexical weight properties may materialize
            # the complete effective catalog table.
            n = int(self.network.itemnum)
            ### >>> Self_EvolveRec-BLOCK-START: Do not fabricate an empty cosine measurement
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            k = min(int(sample_size), n)
            if k < 2:
                return None
            rng = np.random.default_rng(seed)
            idx = torch.tensor(
                rng.choice(n, size=k, replace=False) + 1,
                device=self.device,
                dtype=torch.long,
            )
            x = self.network.item_emb(idx).detach().float()
            if not torch.isfinite(x).all() or x.norm(dim=-1).eq(0).any():
                return None
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute the original mean cosine without a quadratic Gram matrix
            x = x.double()
            # Sum_{i != j} dot(x_i, x_j), including the actual diagonal
            # correction so zero embedding rows retain the original definition.
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
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnostic runner control actual module mode
        # run_full_diagnosis saves and restores every module's training flag.
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and reject malformed diagnostic batches
        from itertools import islice

        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnosis loader must yield (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Use family-compatible tensor inputs and separate target scoring
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis batches must contain B x L sequence/positive/negative tensors"
                )
            if not 1 <= seq_t.size(1) <= int(self.network.maxlen):
                raise ValueError("Diagnostic sequence length outside model bounds")
            for tensor in (seq_t, pos_t, neg_t):
                if ((tensor < 0) | (tensor > int(self.network.itemnum))).any():
                    raise ValueError("Diagnostic IDs outside ordinary catalog bounds")
            # Score through the actual trained lexical HSTU. A host
            # ModelView scorer may otherwise reconstruct ID-only dot products.
            log_feats = self.network.log2feats(seq_t)
            margin = self.network.predict_score(
                log_feats, pos_t
            ) - self.network.predict_score(log_feats, neg_t)
            if margin.shape != seq_t.shape:
                raise ValueError(
                    f"Expected B x L diagnostic margins, got {tuple(margin.shape)}"
                )
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            invalid_scores = valid & ~torch.isfinite(margin)
            if invalid_scores.any():
                raise ValueError(
                    "Nonfinite margins at "
                    f"{int(invalid_scores.sum().item())} valid training positions"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Match reported counts to valid scored transitions
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
                ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing margin measurements from zero
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

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Isolated bounded HSTU sensitivity, independent scoring oracle and complete-logit contracts
    @staticmethod
    def _summarize_pairs(rows):
        if not rows:
            return None
        values = np.asarray(rows, dtype=np.float64)
        rng = np.random.default_rng(42)
        # Chunk CPU resampling to avoid materializing a 10000 x users x 3 array.
        boot = np.empty((10000, values.shape[1]), dtype=np.float64)
        for start in range(0, 10000, 250):
            stop = min(start + 250, 10000)
            draws = rng.integers(0, len(values), size=(stop - start, len(values)))
            boot[start:stop] = values[draws].mean(axis=1)
        names = ("delta_hr10", "delta_target_rank", "overlap10")
        return {
            "users": len(values),
            **{
                name: {
                    "mean": float(values[:, i].mean()),
                    "paired_user_bootstrap_95pct": (
                        [
                            float(np.percentile(boot[:, i], 2.5)),
                            float(np.percentile(boot[:, i], 97.5)),
                        ]
                        if len(values) >= 2
                        else None
                    ),
                    "interval_null_reason": (
                        "At least two users required" if len(values) < 2 else None
                    ),
                }
                for i, name in enumerate(names)
            },
        }

    @staticmethod
    def _ranking(scores, candidates):
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite diagnostic candidate scores")
        order = np.lexsort((np.asarray(candidates, dtype=np.int64), -scores))
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        negatives = scores[1:]
        return {
            "target_rank": rank,
            "hr10": float(rank <= 10),
            "top10": [int(candidates[i]) for i in order[:10]],
            "pairwise_win_rate": float(
                ((scores[0] > negatives) + 0.5 * (scores[0] == negatives)).mean()
            ),
            "target_negative_tie_frequency": float((scores[0] == negatives).mean()),
            "any_candidate_tie": bool(len(np.unique(scores)) < len(scores)),
            "cutoff_tie": bool(scores[order[9]] == scores[order[10]]),
        }

    @torch.no_grad()
    def _temporal_probe(self):
        itemnum = int(self.network.itemnum)
        maxlen = int(self.network.maxlen)
        if maxlen < 2:
            raise ValueError("Controlled deletion probes require maxlen >= 2")
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError(
                "ModelView.user_train must expose public training histories"
            )
        eligible = []
        exclusions = Counter()
        for user in sorted(histories, key=str):
            history = [int(item) for item in histories[user]]
            if any(item < 1 or item > itemnum for item in history):
                exclusions["invalid_or_reserved_history_ids"] += 1
            elif len(history) < 5:
                exclusions["fewer_than_five_events"] += 1
            else:
                eligible.append((user, history))
        rng = random.Random(42)
        selected = rng.sample(eligible, min(8, len(eligible)))
        variants = (
            "identity",
            "no_op",
            "recent_k_delete",
            "oldest_k_delete",
            "final_item_delete",
            "random_k_delete",
            "shuffle_older_keep_last",
            "shuffle_all",
        )
        stochastic = {"random_k_delete", "shuffle_older_keep_last", "shuffle_all"}
        seeds = [42, 43, 44, 45, 46]
        records = []
        for user, history in selected:
            target = history[-1]
            window = history[:-1][-maxlen:]
            n = len(window)
            k = max(1, int(np.floor(0.2 * n)))
            known = set(history)
            if itemnum - len(known) < 99:
                reason = "fewer_than_99_eligible_negatives"
                exclusions[reason] += 1
                records.append(
                    {
                        "user": str(user),
                        "target": target,
                        "window": window,
                        "history_length": n,
                        "k": k,
                        "eligible_negative_count": itemnum - len(known),
                        "excluded": reason,
                    }
                )
                continue
            # Bound rejection sampling, with an exact uniform fallback for dense users.
            negatives, used = [], set()
            for _ in range(9900):
                item = rng.randint(1, itemnum)
                if item not in known and item not in used:
                    negatives.append(item)
                    used.add(item)
                    if len(negatives) == 99:
                        break
            if len(negatives) < 99:
                pool = [
                    i for i in range(1, itemnum + 1) if i not in known and i not in used
                ]
                negatives.extend(rng.sample(pool, 99 - len(negatives)))
            candidates = [target] + negatives
            sequences, labels = [], []
            for variant in variants:
                for seed in seeds if variant in stochastic else [42]:
                    local = random.Random(seed)
                    changed = list(window)
                    if variant == "recent_k_delete":
                        changed = changed[:-k]
                    elif variant == "oldest_k_delete":
                        changed = changed[k:]
                    elif variant == "final_item_delete":
                        changed = changed[:-1]
                    elif variant == "random_k_delete":
                        removed = set(local.sample(range(n), k))
                        changed = [
                            item for i, item in enumerate(changed) if i not in removed
                        ]
                    elif variant == "shuffle_older_keep_last":
                        older = changed[:-1]
                        local.shuffle(older)
                        changed = older + changed[-1:]
                    elif variant == "shuffle_all":
                        local.shuffle(changed)
                    if not changed:
                        raise ValueError(
                            "Diagnostic intervention unexpectedly emptied a history"
                        )
                    sequences.append([0] * (maxlen - len(changed)) + changed)
                    labels.append((variant, seed))
            sequence = torch.tensor(sequences, device=self.device, dtype=torch.long)
            candidates_t = torch.tensor(
                candidates, device=self.device, dtype=torch.long
            )
            # This is a batch of interventions to one checkpoint, never score fusion.
            # DEBUG: Keep perturbation scoring consistent with training and the
            # generated predictor, including the jointly trained text residual.
            features = self.network.log2feats(sequence)
            final = self.network.final_features(features, sequence)
            scores = self.network.predict_score(final, candidates_t)
            if scores.shape != (len(sequences), 100):
                raise ValueError(
                    f"Unexpected HSTU diagnostic score shape: {tuple(scores.shape)}"
                )
            # Float64 conversion preserves the computed score values and also
            # supports bfloat16 outputs, which NumPy cannot consume directly.
            scores = scores.detach().double().cpu().numpy()
            if not np.isfinite(scores).all():
                exclusions["nonfinite_scores"] += 1
                records.append(
                    {
                        "user": str(user),
                        "target": target,
                        "window": window,
                        "history_length": n,
                        "k": k,
                        "candidates": candidates,
                        "excluded": "nonfinite_scores",
                        "nonfinite_score_count": int((~np.isfinite(scores)).sum()),
                    }
                )
                continue
            control_error = float(np.abs(scores[0] - scores[1]).max())
            identity_rank = self._ranking(scores[0], candidates)
            noop_rank = self._ranking(scores[1], candidates)
            score_control_ok = bool(
                np.allclose(scores[0], scores[1], rtol=1e-5, atol=1e-6)
            )
            ranking_control_ok = (
                identity_rank["target_rank"] == noop_rank["target_rank"]
                and identity_rank["top10"] == noop_rank["top10"]
            )
            control_ok = score_control_ok and ranking_control_ok
            outcomes = [
                {
                    "variant": variant,
                    "seed": seed,
                    "sequence": sequences[i],
                    **self._ranking(scores[i], candidates),
                }
                for i, (variant, seed) in enumerate(labels)
            ]
            records.append(
                {
                    "user": str(user),
                    "target": target,
                    "candidates": candidates,
                    "history_length": n,
                    "k": k,
                    "window": window,
                    "control_ok": control_ok,
                    "score_control_ok": score_control_ok,
                    "ranking_control_ok": ranking_control_ok,
                    "control_max_absolute_score_error": control_error,
                    "outcomes": outcomes,
                }
            )
            if not control_ok:
                exclusions["identity_no_op_disagreement"] += 1
        valid = [r for r in records if r.get("control_ok", False)]
        effects = {}
        for variant in variants[1:]:
            paired, strata = [], {"1-9": [], "10-19": [], ">=20": []}
            by_seed = {
                str(seed): [] for seed in (seeds if variant in stochastic else [42])
            }
            baseline_hits, baseline_ranks = [], []
            for record in valid:
                baseline = record["outcomes"][0]
                realizations = []
                for outcome in record["outcomes"]:
                    if outcome["variant"] != variant:
                        continue
                    row = [
                        outcome["hr10"] - baseline["hr10"],
                        outcome["target_rank"] - baseline["target_rank"],
                        len(set(baseline["top10"]) & set(outcome["top10"])) / 10.0,
                    ]
                    realizations.append(row)
                    by_seed[str(outcome["seed"])].append(row)
                row = np.mean(realizations, axis=0).tolist()
                paired.append(row)
                n = record["history_length"]
                strata["1-9" if n < 10 else "10-19" if n < 20 else ">=20"].append(row)
                baseline_hits.append(baseline["hr10"])
                baseline_ranks.append(baseline["target_rank"])
            seed_means = {
                seed: np.mean(rows, axis=0).tolist() if rows else None
                for seed, rows in by_seed.items()
            }
            available_seed_means = [
                value for value in seed_means.values() if value is not None
            ]
            effects[variant] = {
                "paired": self._summarize_pairs(paired),
                "history_length_strata": {
                    name: self._summarize_pairs(rows) for name, rows in strata.items()
                },
                "baseline_hr10": (
                    float(np.mean(baseline_hits)) if baseline_hits else None
                ),
                "baseline_target_rank": (
                    float(np.mean(baseline_ranks)) if baseline_ranks else None
                ),
                "seed_means_delta_hr10_delta_rank_overlap10": seed_means,
                "across_seed_std": (
                    np.std(available_seed_means, axis=0).tolist()
                    if variant in stochastic and available_seed_means
                    else None
                ),
            }
        return {
            "eligible_users": len(eligible),
            "selected_users": len(selected),
            "measured_users": len(valid),
            "exclusions": dict(exclusions),
            "effects": effects,
            "records": records,
        }

    def _isolated_batches(self):
        # Never iterate, clone, or reset the active training loader. Public
        # histories provide a separate immutable diagnostic dataset.
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            return (), {"reason": "Public training histories unavailable"}
        n, length = int(self.network.itemnum), int(self.network.maxlen)
        eligible = []
        excluded = Counter()
        for uid in sorted(histories, key=lambda x: (str(x), type(x).__name__)):
            try:
                history = tuple(int(x) for x in histories[uid])
            except (TypeError, ValueError):
                excluded["malformed_history"] += 1
                continue
            if len(history) < 2 or any(x < 1 or x > n for x in history):
                excluded["short_or_invalid_history"] += 1
                continue
            eligible.append((str(uid), history))
        rng = np.random.default_rng(42)
        # Descriptive length stratification, without examining model outcomes.
        groups = [[], [], []]
        for record in eligible:
            occupied = min(length, len(record[1]) - 1)
            groups[0 if occupied < 10 else 1 if occupied < 20 else 2].append(record)
        for group in groups:
            rng.shuffle(group)
        selected = []
        while len(selected) < 8 and any(groups):
            for group in groups:
                if group and len(selected) < 8:
                    selected.append(group.pop())
        sequences, positives, negatives, records = [], [], [], []
        for uid, history in selected:
            source, target = history[:-1][-length:], history[1:][-length:]
            pool = np.setdiff1d(np.arange(1, n + 1), np.asarray(history))
            if not len(pool):
                excluded["no_unseen_negative"] += 1
                continue
            seq = np.zeros(length, dtype=np.int64)
            pos, neg = seq.copy(), seq.copy()
            seq[-len(source) :], pos[-len(target) :] = source, target
            neg[-len(source) :] = rng.choice(pool, size=len(source), replace=True)
            sequences.append(seq)
            positives.append(pos)
            negatives.append(neg)
            records.append({"user": uid, "target": int(target[-1])})
        coverage = {
            "source": "isolated immutable public-training-history arrays",
            "active_loader_consumed": False,
            "eligible_users": len(eligible),
            "selected_users": len(records),
            "maximum_users": 8,
            "records": records,
            "exclusions": dict(excluded),
            "mapping_scope": "Supplied ordinary integer IDs; no external mapping exposed",
        }
        if not sequences:
            coverage["reason"] = "No eligible isolated diagnostic histories"
            return (), coverage
        arrays = [np.stack(values) for values in (sequences, positives, negatives)]
        for array in arrays:
            array.setflags(write=False)
        # Copies avoid PyTorch views over read-only NumPy storage.
        return ((None, *(torch.tensor(a) for a in arrays)),), coverage

    def _contract_audit(self, batches):
        from types import SimpleNamespace

        checks, mutants = {}, {}
        executions = {"feature_calls": 0, "score_calls": 0, "scalar_scores": 0}
        atol, rtol = 1e-5, 1e-4

        def compare(name, actual, expected):
            actual, expected = actual.detach().float(), expected.detach().float()
            if actual.shape != expected.shape:
                value = {
                    "passed": False,
                    "actual_shape": list(actual.shape),
                    "expected_shape": list(expected.shape),
                }
            else:
                delta = (actual - expected).abs()
                value = {
                    "passed": bool(
                        torch.isfinite(actual).all()
                        and torch.isfinite(expected).all()
                        and torch.allclose(actual, expected, atol=atol, rtol=rtol)
                    ),
                    "max_absolute_error": float(delta.max()) if delta.numel() else 0.0,
                    "max_relative_error": (
                        float((delta / expected.abs().clamp_min(1e-12)).max())
                        if delta.numel()
                        else 0.0
                    ),
                }
            checks[name] = value
            return value["passed"]

        def features(seq):
            executions["feature_calls"] += 1
            return self.network.log2feats(seq)

        def score(feat, ids):
            executions["score_calls"] += 1
            value = self.network.predict_score(feat, ids)
            executions["scalar_scores"] += value.numel()
            return value

        def mutation(name, actual=None, expected=None, reason=None):
            if reason is not None:
                mutants[name] = {"status": "unsupported", "reason": reason}
                return
            exercised = actual.shape != expected.shape or not torch.allclose(
                actual, expected, atol=atol, rtol=rtol
            )
            if not exercised:
                mutants[name] = {
                    "status": "unexercised",
                    "reason": "No measurable exposure under fixed tolerance",
                }
                return
            probe = "mutant/" + name
            killed = not compare(probe, actual, expected)
            mutants[name] = {
                "status": "killed" if killed else "survived",
                "observable": "complete raw scores or readout features",
                "detection_matrix": {probe: killed},
            }

        # A hand-specified unfitted fixture invokes the production score method.
        # Expected values below use literal external-ID mappings and equations.
        identity = torch.nn.Embedding(4, 2, padding_idx=0).to(self.device)
        with torch.no_grad():
            identity.weight.copy_(
                torch.tensor(
                    [[0.0, 0.0], [1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]],
                    device=self.device,
                )
            )
        fixture_embedding = identity
        residual_coverage = {
            "covered": False,
            "reason": "Lexical fixture interface unavailable",
        }
        if type(self.network.item_emb).__name__ == "JointLexicalEmbedding" and hasattr(
            self.network.item_emb, "alpha"
        ):
            title = torch.zeros(4, 16, dtype=torch.long)
            description = torch.zeros(4, 64, dtype=torch.long)
            title[1, 0] = 2
            title[2, 0] = 3
            fixture_embedding = type(self.network.item_emb)(
                identity, 4, title, description
            ).to(self.device)
            with torch.no_grad():
                fixture_embedding.tokens.weight.zero_()
                fixture_embedding.tokens.weight[2, 0] = 1.0
                fixture_embedding.tokens.weight[3, 1] = 1.0
                fixture_embedding.projection.weight.zero_()
                fixture_embedding.projection.weight[0, 0] = 1.0
                fixture_embedding.projection.weight[1, 1] = 1.0
                fixture_embedding.alpha.fill_(float(np.arctanh(0.5)))
            residual_coverage = {
                "covered": True,
                "equation": "ID + 0.5 * norm(ID) * unit_text_direction",
            }
        literal_rows = [[0.0, 0.0], [1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]]
        if residual_coverage["covered"]:
            literal_rows[1][0] += 0.5 * np.sqrt(5.0)
            literal_rows[2][1] += 0.5 * np.sqrt(5.0)
        external_to_row = {701: 2, 903: 1, 211: 3}
        external_order = [701, 903, 211]
        ids = torch.tensor(
            [external_to_row[x] for x in external_order], device=self.device
        )
        known = torch.tensor([[2.0, -1.0], [-1.0, 3.0]], device=self.device)
        expected = torch.tensor(
            [
                [
                    sum(
                        float(f[d]) * literal_rows[external_to_row[x]][d]
                        for d in range(2)
                    )
                    for x in external_order
                ]
                for f in known.tolist()
            ],
            device=self.device,
        )
        production = type(self.network).predict_score
        fixture = SimpleNamespace(item_emb=fixture_embedding)
        compare("oracle/candidate_matrix", production(fixture, known, ids), expected)
        aligned_ids = torch.tensor([[2, 1], [3, 2]], device=self.device)
        aligned_feats = known[:, None, :].expand(-1, 2, -1)
        aligned_expected = torch.tensor(
            [
                [
                    sum(
                        float(known[b, d]) * literal_rows[int(aligned_ids[b, j])][d]
                        for d in range(2)
                    )
                    for j in range(2)
                ]
                for b in range(2)
            ],
            device=self.device,
        )
        compare(
            "oracle/aligned_training_scores",
            production(fixture, aligned_feats, aligned_ids),
            aligned_expected,
        )
        compare(
            "oracle/padding_zero",
            fixture_embedding(torch.tensor([0], device=self.device)),
            torch.zeros(1, 2, device=self.device),
        )
        mutation("off_by_one_ids", production(fixture, known, (ids % 3) + 1), expected)
        swapped = torch.tensor([1, 2, 3], device=self.device)
        mutation(
            "swapped_embedding_rows", production(fixture, known, swapped), expected
        )
        mutation(
            "padding_id_aliasing",
            production(fixture, known, torch.tensor([2, 0, 3], device=self.device)),
            expected,
        )

        seq = batches[0][1].to(self.device)
        # Include a short, heavily padded history and the explicit empty case.
        short = torch.zeros_like(seq[:1])
        short[0, -1] = seq[0, -1]
        seq = torch.cat([seq, short, torch.zeros_like(short)])
        feat = features(seq)
        positions = torch.arange(seq.size(1), device=self.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(1)
        expected_readout = feat[
            torch.arange(len(seq), device=self.device), last.clamp_min(0)
        ]
        expected_readout = expected_readout.masked_fill(last.lt(0)[:, None], 0.0)
        readout = getattr(self.network, "final_features", None)
        if not callable(readout):
            checks["production_readout"] = {
                "passed": None,
                "reason": "No production final_features callable",
            }
            return {
                "checks": checks,
                "mutants": mutants,
                "residual_oracle": residual_coverage,
            }
        actual_readout = readout(feat, seq)
        compare("production_readout", actual_readout, expected_readout)
        candidates = [int(x) for x in batches[0][2][:, -1].tolist()]
        candidates = list(dict.fromkeys(candidates))
        rng = np.random.default_rng(42)
        remaining = np.setdiff1d(
            np.arange(1, int(self.network.itemnum) + 1), candidates
        )
        candidates += rng.choice(
            remaining, min(100 - len(candidates), len(remaining)), replace=False
        ).tolist()
        candidate_ids = torch.tensor(candidates, device=self.device, dtype=torch.long)
        original = score(actual_readout, candidate_ids)
        compare(
            "identity_complete_logits",
            score(readout(features(seq), seq), candidate_ids),
            original,
        )
        perm = torch.tensor(rng.permutation(len(candidates)), device=self.device)
        permuted = score(actual_readout, candidate_ids[perm])
        compare("candidate_order", permuted[:, torch.argsort(perm)], original)
        mutation("candidate_sort_without_inverse", permuted, original)
        for batch_size in (1, 3):
            pieces = []
            for start in range(0, len(seq), batch_size):
                part = seq[start : start + batch_size]
                pieces.append(score(readout(features(part), part), candidate_ids))
            compare(f"batch_size_{batch_size}", torch.cat(pieces), original)
        for width in (1, 7, len(candidates)):
            chunks = [
                score(actual_readout, candidate_ids[j : j + width])
                for j in range(0, len(candidates), width)
            ]
            compare(f"candidate_chunk_{width}", torch.cat(chunks, -1), original)
        mutation("batch_row_broadcast", original[:1].expand_as(original), original)
        if len(candidates) % 7:
            mutation(
                "final_chunk_omission",
                original[:, : len(candidates) // 7 * 7],
                original,
            )
        else:
            mutants["final_chunk_omission"] = {
                "status": "unexercised",
                "reason": "No partial chunk",
            }
        wrong_index = seq.ne(0).sum(1).sub(1).clamp_min(0)
        wrong = feat[torch.arange(len(seq), device=self.device), wrong_index]
        exposed = last.ge(0) & wrong_index.ne(last)
        if exposed.any():
            mutation(
                "count_minus_one_readout", wrong[exposed], expected_readout[exposed]
            )
        else:
            mutants["count_minus_one_readout"] = {
                "status": "unexercised",
                "reason": "Indices coincide",
            }
        causal_positions = []
        for row in range(len(seq)):
            occupied = seq[row].ne(0).nonzero(as_tuple=True)[0]
            if len(occupied) < 2 or int(self.network.itemnum) < 2:
                continue
            t = int(occupied[0])
            changed = seq[row : row + 1].clone()
            future = occupied[occupied > t]
            changed[0, future] = changed[0, future] % int(self.network.itemnum) + 1
            altered = features(changed)
            compare(
                f"causal_features/{row}",
                altered[:, : t + 1],
                feat[row : row + 1, : t + 1],
            )
            compare(
                f"causal_complete_logits/{row}",
                score(altered[:, t], candidate_ids),
                score(feat[row : row + 1, t], candidate_ids),
            )
            causal_positions.append(
                {"row": row, "position": t, "future_slots": future.tolist()}
            )
        mutation(
            "disabled_causal_mask",
            reason="No isolated supported causal-mask mutation hook; trained network is not patched",
        )
        mutation(
            "wrong_category_routing",
            reason="Category routing belongs to generated predictor, not ModelView",
        )
        exercised = [
            m for m in mutants.values() if m["status"] in ("killed", "survived")
        ]
        return {
            "protocol_version": "hstu-contracts-v1",
            "atol": atol,
            "rtol": rtol,
            "checks": checks,
            "mutants": mutants,
            "mutation_summary": {
                "killed": sum(m["status"] == "killed" for m in exercised),
                "exercised_non_equivalent": len(exercised),
                "scope": "Listed diagnostic outputs only; no checkpoints mutated",
            },
            "residual_oracle": residual_coverage,
            "fixtures": {
                "sequences": seq.cpu().tolist(),
                "ordered_candidates": candidates,
                "causal_positions": causal_positions,
                "complete_reference_logits": original.cpu().tolist(),
            },
            "executions": executions,
            "prefix_only": {
                "passed": None,
                "reason": "Compacting prefixes shifts learned absolute positions",
            },
            "deployment_parity": {
                "passed": None,
                "reason": "ModelView exposes no pre-save capture or fresh-process generated-predictor integration; no deployment parity claimed",
            },
            "mapping_parity": {
                "passed": None,
                "reason": "Synthetic external mapping is tested; real public mapping and checkpoint/category metadata are not exposed",
            },
            "tie_rule": "Auxiliary diagnostics use descending score then ascending item ID; host tie rule unavailable",
            "empty_history_policy": "Production final_features is compared to explicit zero readout",
            "scoring_contract": "1D items -> shared candidate axis; B x L items with B x L x D features -> aligned scores",
        }

    def run_full_diagnosis(self, train_loader):
        import os
        import re
        import time

        modes = [(module, module.training) for module in self.network.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        stage = "isolated_fixture_construction"
        started = time.perf_counter()
        cuda_start = cuda_end = None
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                self.network.eval()
                batches, fixture_coverage = self._isolated_batches()
                if not batches:
                    return {
                        "metrics": {
                            "audit": None,
                            "reason": fixture_coverage,
                            "active_loader_consumed": False,
                        },
                        "metric_definitions": {
                            "audit": "Unavailable isolated training fixtures; no active loader was consumed and no correctness result inferred."
                        },
                    }
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                        cuda_start.record()
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type, enabled=False
                ):
                    stage = "effective_embedding_probe"
                    collapse_score = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048
                    )
                    stage = "isolated_training_margins"
                    margins, counts = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            batches, num_batches=1, eps=0.5, topk_categories=0
                        )
                    )
                    stage = "complete_logit_contracts"
                    parameter_dtypes = {
                        p.dtype
                        for p in self.network.parameters()
                        if p.is_floating_point()
                    }
                    if parameter_dtypes == {torch.float32}:
                        contract_audit = self._contract_audit(batches)
                    else:
                        contract_audit = {
                            "checks": None,
                            "reason": "Primary tolerance protocol requires float32 parameters; trained model dtype was not changed",
                        }
                    stage = "bounded_temporal_probe"
                    temporal = self._temporal_probe()
                stage = "report_and_artifact"
                if cuda_end is not None:
                    cuda_end.record()
                    cuda_end.synchronize()
                blocks = list(getattr(self.network, "blocks", []))
                dataset = getattr(
                    self.model,
                    "dataset_name",
                    getattr(self.network, "dataset_name", None),
                )
                configuration = {
                    "dataset_name": str(dataset) if dataset is not None else None,
                    "scope": "one supplied category ModelView; public training-prefix targets; in-sample",
                    "model_class": type(self.network).__module__
                    + "."
                    + type(self.network).__qualname__,
                    "parameter_count": sum(
                        p.numel() for p in self.network.parameters()
                    ),
                    "checkpoint_path": None,
                    "checkpoint_identity_reason": "No checkpoint path is exposed by the required ModelView; scoring uses its actual current model without fitting or loading another model.",
                    "itemnum": int(self.network.itemnum),
                    "maxlen": int(self.network.maxlen),
                    "parameter_dtype": str(next(self.network.parameters()).dtype),
                    "device": str(self.device),
                    "position_module": type(
                        getattr(self.network, "pos_emb", None)
                    ).__name__,
                    "attention_modules": [
                        type(getattr(block, "attn", None)).__name__ for block in blocks
                    ],
                    "supplied_hstu_mechanisms": "Supplied HSTU uses absolute slot embeddings, causal MultiheadAttention, SiLU gates and residual FFNs; no timestamp input or explicit relative-position bias is present in the supplied forward path.",
                    "timestamp_effects": None,
                    "timestamp_reason": "Required log2feats(sequence) scoring path accepts item IDs only; timestamp effects were not tested.",
                    "user_sampling_seed": 42,
                    "max_users": 8,
                    "fixture_coverage": fixture_coverage,
                    "torch_version": str(torch.__version__),
                    "cuda_version": torch.version.cuda,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "autocast_enabled": False,
                    "training_policy_modified": False,
                    "minimum_events": 5,
                    "perturbation_seeds": [42, 43, 44, 45, 46],
                    "bootstrap_seed": 42,
                    "bootstrap_replicates": 10000,
                    "candidate_count": 100,
                    "legacy_margin_max_batches": 1,
                    "margin_source": "Isolated deterministic training-history transitions with locally sampled unseen negatives; not active-loader samples",
                    "identity_no_op_tolerance": {"rtol": 1e-5, "atol": 1e-6},
                    "history_policy": "Truncate prefix once; perturb without backfill; compact surviving events; left pad; greatest nonzero position readout.",
                    "training_changes": False,
                }
                metrics = {
                    "configuration": configuration,
                    "implementation_contract_audit": contract_audit,
                    "embedding_collapse_score": collapse_score,
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "legacy_margin_counts": {
                        "low_margin_total_positions": counts[
                            "low_margin_total_positions"
                        ],
                        "wrong_rank_total_positions": counts[
                            "wrong_rank_total_positions"
                        ],
                    },
                    "pos_neg_margin_category_breakdown": None,
                    "category_breakdown_reason": "Item categories are unavailable; no category values were consumed or synthesized.",
                    "training_prefix_sensitivity": temporal,
                    "validation_effects": None,
                    "validation_reason": "run_full_diagnosis(train_loader) exposes no authorized validation targets, pinned candidates, filtering, or tie rules.",
                    "four_category_macro_effect": None,
                    "macro_reason": "This call measures one category ModelView; all four category estimates are not exposed together. No other trials or checkpoints are read.",
                    "full_catalog_stability": None,
                    "full_catalog_reason": "Optional catalog audit omitted to bound diagnostic cost; candidate-set effects are not full-catalog estimates.",
                    "cost": {
                        "wall_seconds_before_artifact_write": time.perf_counter()
                        - started,
                        "cuda_stream_elapsed_seconds": (
                            cuda_start.elapsed_time(cuda_end) / 1000.0
                            if cuda_end is not None
                            else None
                        ),
                        "cost_note": "CUDA event interval includes gaps between queued work, not a GPU utilization measurement; host budget accounting remains authoritative.",
                    },
                }
                definitions = {
                    "implementation_contract_audit": "Fixed v1 float32 complete-logit consistency checks at atol=1e-5, rtol=1e-4, without changing backend policy. Independent literal scoring expectations exercise the production score callable and, when supported, a nonzero synthetic lexical residual. Synthetic modules are unfitted test objects, never recommendation sources. Mutation detection is a violation of an explicit expected value or relation; unsupported and unexercised defects are excluded from the stated denominator. No trained checkpoint, parameter, hook or recommendation is mutated. Real mapping and fresh-process generated-predictor parity remain unavailable unless measured through host integration. Passing does not establish general correctness or validation quality. Execution counts cover this contract suite only; wall/CUDA cost covers all probes. Fixture IDs, candidates and reference logits are persisted in this report.",
                    "embedding_collapse_score": "(Mean off-diagonal cosine + 1)/2 over at most 2048 effective ordinary-item embeddings, including the jointly trained lexical contribution. Null with fewer than two items, nonfinite values or zero norms. Approximately 0.5 denotes zero mean cosine, not effective rank, catalog diversity or relevance.",
                    "legacy_margins": "Mean and nearest-rank 10th percentile of positive-minus-negative logits over nonpadding sampled training transitions; wrong rate is margin<0, small rate is margin<0.5. Scale- and sampler-dependent; null if no valid positions. Legacy margins do not estimate benchmark HR@10.",
                    "target_rank": "1 + number of candidates ahead, sorting descending score then ascending integer item ID. Auxiliary diagnostic tie rule only; host evaluation is unchanged.",
                    "hr10": "Indicator that the training-prefix target has candidate rank <=10; in-sample and based on frozen diagnostic negatives, not official validation.",
                    "pairwise_win_rate": "Mean over 99 negatives of I(target_score>negative_score)+0.5*I(equal). Target-negative tie frequency counts exact equality; any-candidate and cutoff ties are also saved.",
                    "candidate_policy": "Target plus 99 unique uniform negatives excluding padding, reserved tokens, target and every public training interaction for that user. Frozen across all interventions. Users with insufficient candidates are excluded and counted.",
                    "deletions": "k=max(1,floor(0.2*n)) for the once-truncated prefix. Recent, oldest and random deletion use the same users/k and retain at least one event. Combined content/length/absolute-position interventions do not isolate semantic recency. Final-item removal is separate.",
                    "shuffles": "Shuffle item identities in the fixed valid slots; older shuffle preserves the final item. No timestamp intervention is supported. Stochastic outcomes use seeds 42..46, without additional training or score averaging.",
                    "paired_effects": "For each user, average repeated outcome differences, never scores. Delta HR@10 and target rank are perturbed minus identity. Overlap10 is intersection size of fixed-candidate top10 IDs divided by 10. Negative delta HR@10 or positive delta rank indicates degradation.",
                    "controls": "Identity and no-op candidate scores must agree with rtol=1e-5, atol=1e-6, and their canonical target ranks and ordered top10 IDs must agree exactly. Numerical closeness alone cannot validate ranking effects near ties. Failed/nonfinite users are excluded from effects and counted; individual outcomes are retained when finite.",
                    "uncertainty": "10000 seed-42 user bootstrap replicates within this category; identical sampled indices for paired metrics. Report observed means and percentile 95% intervals. Conditions on checkpoint, candidates and realized perturbations; excludes training-seed, candidate-sampling and model-selection uncertainty. Across-seed standard deviation is descriptive, not an independent-user confidence interval.",
                    "strata": "History length bins 1-9, 10-19, >=20 use the fixed baseline window; empty strata are null. Eligibility requires at least five public training events.",
                    "availability": "Null means unavailable, with a reason; zero means measured zero. Item ratings/categories/prices/stores, review content and simulator traits are unobserved. Validation transition/suffix strata require authorized pinned validation examples and are not inferred here.",
                    "interpretation_limits": "Training-only effects may reflect memorization or distribution shift. They do not establish causality, current demand, compatibility, generalization or expected validation gains. Only the unchanged host four-category validation macro HR@10 selects models. No independent predictor is fitted, fused or deployed by this diagnostic.",
                }
                result = {"metrics": metrics, "metric_definitions": definitions}
                trial_dir = os.path.dirname(os.path.abspath(__file__))
                artifact_dir = os.path.join(trial_dir, "working", "diagnostics")
                os.makedirs(artifact_dir, exist_ok=True)
                label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(dataset or "category"))
                # Unique artifact name avoids overwriting a category when the host
                # does not expose its routing label through ModelView.
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix=label + "_",
                    suffix=".json",
                    dir=artifact_dir,
                    delete=False,
                ) as stream:
                    metrics["artifact_path"] = stream.name
                    json.dump(result, stream, indent=2, allow_nan=False)
                return result
        except Exception as error:
            import traceback

            return {
                "metrics": {
                    "diagnostic_failure": {
                        "stage": stage,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    "active_loader_consumed": False,
                    "elapsed_wall_seconds": time.perf_counter() - started,
                },
                "metric_definitions": {
                    "diagnostic_failure": "Observed diagnostic exception and failing stage. No pass rate, behavioral conclusion or validation result is inferred from failure. Module modes and RNG states are restored in finally."
                },
            }
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    ### <<< Self_EvolveRec-BLOCK-END
