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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret temporal diagnostics within their measured scope
    - Separate training fit, intervention dependence, padding artifacts, and official validation accuracy.
    - Mean embedding cosine measures directional alignment; it does not establish collapse or recommendation diversity.
    - Training targets and training-fitted descriptive controls do not measure generalization.
    - Missing controls are unavailable evidence. Do not infer recency preference from deletion when positional and padding controls are unsupported.
    - Do not infer causal preferences, ownership satisfaction, compatibility, or unobserved simulator diversity/conformity traits.
    - Sampled-candidate overlap and full-catalog overlap are different quantities.
    - Bootstrap intervals condition on the fixed model, users, and candidate policy; they exclude training-seed uncertainty and repeated-selection optimism.
    - Do not invent official validation results or validation confidence intervals.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Bind diagnostics to the trained module and available public histories
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.device = torch.device(device)
        self.actual_model = view.model
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item-category metadata is unavailable in this benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            ### >>> Self_EvolveRec-BLOCK-START: Compute directional alignment without a quadratic Gram matrix
            n = emb.size(0)
            if sample_size < 2:
                raise ValueError("Embedding alignment requires sample_size >= 2")
            if n < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite sampled item embeddings")
            if (x.norm(dim=-1) == 0).any():
                # A zero vector has no defined direction.
                return None
            x = F.normalize(x, p=2, dim=-1).double()
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
            return round(float((mean_cos + 1.0) * 0.5), 4)
            ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Leave module mode management to the diagnosis entry point
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

            ### >>> Self_EvolveRec-BLOCK-START: Use device tensors with the trained feature encoder
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.view.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Reject nonfinite training margins and exclude padding pairs
            valid = (pos_t != 0) & (neg_t != 0) & (seq_t != 0)
            if not torch.isfinite(margin[valid]).all():
                raise FloatingPointError("Nonfinite diagnostic training margins")
            if valid.any():
                ### <<< Self_EvolveRec-BLOCK-END
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Apply the same valid-pair mask to descriptive counts
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
                ### >>> Self_EvolveRec-BLOCK-START: Represent absent training margins as unavailable
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

    ### >>> Self_EvolveRec-BLOCK-START: Validate paired scoring shapes and isolate reproducible candidate and intervention sampling
    def _sequence(self, history, length):
        seq = torch.zeros((1, length), dtype=torch.long, device=self.device)
        history = list(history)[-length:]
        if history:
            seq[0, -len(history) :] = torch.tensor(history, device=self.device)
        return seq

    def _features(self, seq):
        features = self.view.log2feats(seq)
        if features.ndim != 3 or features.shape[:2] != seq.shape:
            raise ValueError("Temporal probe requires B x L x D features")
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite diagnostic features")
        # Preserve the trained SASRec readout, including for sequences with holes.
        return features[:, -1, :]

    def _scores(self, features, candidates):
        items = torch.as_tensor(
            candidates, dtype=torch.long, device=self.device
        ).reshape(1, -1)
        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError("Candidate scoring requires one B x D readout")
        # Use the same B x L x D / B x L x K contract as the margin probe.
        # Keeping L=1 avoids accidental candidate/feature-axis broadcasting.
        scores = self.view.predict_score(features.unsqueeze(1), items.unsqueeze(1))
        if tuple(scores.shape) != (1, 1, items.shape[1]):
            raise ValueError(
                f"Expected diagnostic scores shaped (1, 1, {items.shape[1]}), "
                f"received {tuple(scores.shape)}"
            )
        scores = scores.reshape(-1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite diagnostic candidate scores")
        return scores.detach().float().cpu().numpy()

    @staticmethod
    def _outcome(scores, candidates, target):
        scores = np.asarray(scores, dtype=np.float64)
        candidates = np.asarray(candidates, dtype=np.int64)
        if scores.ndim != 1 or scores.shape != candidates.shape:
            raise ValueError(
                "Diagnostic scores and candidates must be matching vectors"
            )
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite model or descriptive-control scores")
        if len(np.unique(candidates)) != len(candidates):
            raise ValueError("Diagnostic candidates must be unique")
        if np.count_nonzero(candidates == target) != 1:
            raise ValueError("Diagnostic target must occur exactly once")
        order = np.lexsort((candidates, -scores))
        ranked = candidates[order]
        rank = int(np.flatnonzero(ranked == target)[0]) + 1
        return {
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "hr10": float(rank <= 10),
        }, ranked[:10].tolist()

    def _catalog_recommendations(self, history, length, itemnum):
        features = self._features(self._sequence(history, length))
        excluded = set(history) | {0}
        best_ids = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        for start in range(1, itemnum + 1, 8192):
            ids = np.arange(start, min(itemnum + 1, start + 8192))
            ids = ids[
                np.fromiter(
                    (int(i) not in excluded for i in ids), dtype=bool, count=len(ids)
                )
            ]
            if not len(ids):
                continue
            scores = self._scores(features, ids)
            local = np.lexsort((ids, -scores))[:10]
            best_ids = np.concatenate((best_ids, ids[local]))
            best_scores = np.concatenate((best_scores, scores[local]))
            keep = np.lexsort((best_ids, -best_scores))[:10]
            best_ids, best_scores = best_ids[keep], best_scores[keep]
        return {
            "items": best_ids.tolist(),
            "scores": best_scores.tolist(),
            "exclusion_policy": "complete_training_history_and_padding",
            "catalog_size": itemnum,
            "encoded_context_length": min(length, len(history)),
        }

    @staticmethod
    def _summarize(records):
        groups = {}
        for record in records:
            groups.setdefault(record["intervention"], {}).setdefault(
                record["user"], []
            ).append(record)
        summaries = {}
        for name, users in sorted(groups.items()):
            rows = []
            for repeats in users.values():
                rows.append(
                    [
                        np.mean([r["baseline"]["rank"] for r in repeats]),
                        np.mean([r["changed"]["rank"] for r in repeats]),
                        np.mean([r["baseline"]["reciprocal_rank"] for r in repeats]),
                        np.mean([r["changed"]["reciprocal_rank"] for r in repeats]),
                        np.mean([r["baseline"]["hr10"] for r in repeats]),
                        np.mean([r["changed"]["hr10"] for r in repeats]),
                        np.mean([r["sampled_top10_overlap"] for r in repeats]),
                    ]
                )
            values = np.asarray(rows, dtype=float)
            differences = values[:, [1, 3, 5]] - values[:, [0, 2, 4]]
            rng = np.random.default_rng(42)
            bootstrap = differences[
                rng.integers(len(values), size=(1000, len(values)))
            ].mean(axis=1)
            names = (
                "baseline_rank",
                "changed_rank",
                "baseline_reciprocal_rank",
                "changed_reciprocal_rank",
                "baseline_hr10",
                "changed_hr10",
                "sampled_top10_overlap",
            )
            summary = dict(zip(names, values.mean(axis=0).tolist()))
            summary["users"] = len(users)
            summary["evaluations"] = sum(map(len, users.values()))
            for column, metric in enumerate(("rank", "reciprocal_rank", "hr10")):
                summary[metric + "_difference"] = float(differences[:, column].mean())
                summary[metric + "_difference_ci95"] = np.percentile(
                    bootstrap[:, column], [2.5, 97.5]
                ).tolist()
            summaries[name] = summary
        return summaries

    def run_full_diagnosis(self, train_loader):
        import os
        import time
        from pathlib import Path

        started = time.monotonic()
        # Reserve 150 wall seconds per dataset for planning optional work.
        # This is not a hard cap: required scoring and in-flight work can overrun.
        # GPU usage remains charged by the host's shared search-budget accounting.
        optional_deadline = started + 150.0
        previous_modes = [
            (module, module.training) for module in self.actual_model.modules()
        ]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        try:
            self.actual_model.eval()
            with torch.inference_mode():
                return self._run_temporal_diagnosis(
                    train_loader, started, optional_deadline, Path, os, time
                )
        finally:
            for module, mode in previous_modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _run_temporal_diagnosis(self, train_loader, started, deadline, Path, os, time):
        histories = self.user_train.get("History", self.user_train)
        histories = {str(u): [int(i) for i in h] for u, h in histories.items()}
        itemnum = self.view.item_emb.num_embeddings - 1
        if itemnum < 1:
            raise ValueError("Diagnostic catalog contains no real items")
        for user, history in histories.items():
            if any(item < 1 or item > itemnum for item in history):
                raise ValueError(f"Training history {user} contains an invalid item ID")
        length = int(self.actual_model.maxlen)
        if length < 5:
            raise ValueError("Temporal matched-removal probe needs maxlen >= 5")
        dataset = str(getattr(self.view, "dataset_name", "unspecified"))
        # Always stay beneath the trial's source directory; never use DATA_DIR.
        root = Path(__file__).resolve().parent / "working" / "diagnostics"
        root.mkdir(parents=True, exist_ok=True)
        safe_dataset = "".join(c if c.isalnum() or c in "_-" else "_" for c in dataset)
        rng = np.random.default_rng(42)
        eligible = sorted(u for u, h in histories.items() if len(h) >= 6)
        order = rng.permutation(len(eligible))[:50]
        selected = [eligible[int(i)] for i in order]
        simulator_users = sorted(u for u, h in histories.items() if h)
        simulator_rng = np.random.default_rng(42)
        simulator_users = [
            simulator_users[int(i)]
            for i in simulator_rng.permutation(len(simulator_users))[:5]
        ]
        recommendations = {}
        for user in simulator_users:
            recommendations[user] = self._catalog_recommendations(
                histories[user], length, itemnum
            )
        omissions, records, pools, descriptions = [], [], {}, {}
        frequency = Counter(i for h in histories.values() for i in h)
        transitions = {}
        for history in histories.values():
            for source, dest in zip(history[:-1], history[1:]):
                transitions.setdefault(source, Counter())[dest] += 1
        totals = {i: sum(counts.values()) for i, counts in transitions.items()}
        all_ids = np.arange(1, itemnum + 1)
        counts = np.asarray([frequency[int(i)] for i in all_ids])
        frequency_order = np.lexsort((all_ids, counts))
        deciles = np.empty(itemnum + 1, dtype=np.int64)
        deciles[0] = -1
        deciles[all_ids[frequency_order]] = np.minimum(
            9, np.arange(itemnum) * 10 // max(1, itemnum)
        )
        buckets = [all_ids[deciles[all_ids] == d] for d in range(10)]
        scoring_contract = {
            "readout": "features[:, -1, :]",
            "padding": "left padding to model.maxlen; holes preserve array positions",
            "position_ids": "unmodified model.log2feats implementation",
            "padding_key_exclusion": "unverified; zero embeddings do not prove exclusion",
            "preserved_position_packing": "unsupported: no explicit position-ID contract",
            "padded_key_perturbation": "unsupported: no explicit attention-mask contract",
            "holes_vs_preserved_packing": None,
            "official_predict_score_max_abs_error": None,
        }
        for user in selected:
            if time.monotonic() >= deadline:
                omissions.append(
                    {
                        "reason": "optional time allowance exhausted",
                        "remaining_users": len(selected) - len(pools),
                    }
                )
                break
            history = histories[user]
            target, prefix = history[-1], history[:-1][-length:]
            # Stable per-user streams keep pools independent of earlier users,
            # intervention count, donor availability, and optional omissions.
            import hashlib

            seed_bytes = hashlib.sha256(
                json.dumps([42, dataset, user], ensure_ascii=True).encode("utf-8")
            ).digest()
            streams = np.random.SeedSequence(
                int.from_bytes(seed_bytes[:16], "little")
            ).spawn(3)
            candidate_rng, intervention_rng, donor_rng = [
                np.random.default_rng(stream) for stream in streams
            ]
            excluded = set(prefix) | {target, 0}
            available_mask = np.ones(itemnum + 1, dtype=bool)
            available_mask[list(excluded)] = False
            available = all_ids[available_mask[1:]]
            if len(available) < 99:
                omissions.append(
                    {"user": user, "reason": "fewer than 99 eligible negatives"}
                )
                continue
            candidates = [target] + candidate_rng.choice(
                available, 99, replace=False
            ).tolist()
            pools[user] = {
                "target": target,
                "prefix": prefix,
                "candidates": candidates,
                "policy": "target_once_uniform_unique99_exclude_original_prefix",
            }
            descriptions[user] = {
                "training_history_length": len(history),
                "prefix_length": len(prefix),
                "repeated_item_fraction": 1.0 - len(set(prefix)) / len(prefix),
            }
            sequence = self._sequence(prefix, length)
            features = self._features(sequence)
            base_scores = self._scores(features, candidates)
            if callable(getattr(self.actual_model, "predict", None)):
                direct = self.actual_model.predict(
                    sequence, torch.tensor([candidates], device=self.device)
                )
                direct = direct.detach().float().cpu().numpy().reshape(-1)
                if not np.isfinite(direct).all():
                    raise FloatingPointError("Nonfinite actual-model prediction")
                if direct.shape != base_scores.shape:
                    raise ValueError("Actual-model and adapter score shapes disagree")
                error = float(np.max(np.abs(direct - base_scores)))
                previous_error = scoring_contract[
                    "official_predict_score_max_abs_error"
                ]
                scoring_contract["official_predict_score_max_abs_error"] = max(
                    error, previous_error if previous_error is not None else 0.0
                )
                if not np.allclose(direct, base_scores, rtol=1e-4, atol=1e-5):
                    raise ValueError(
                        "Diagnostic scoring disagrees with actual model.predict"
                    )
            baseline, baseline_top = self._outcome(base_scores, candidates, target)
            n = len(prefix)
            m = max(1, int(np.floor(0.2 * (n - 1))))
            interventions = []
            for name, indices in (
                ("oldest", list(range(m))),
                ("recent", list(range(n - 1 - m, n - 1))),
            ):
                interventions.append((name, 0, indices, None))
            for repeat in range(3):
                indices = sorted(
                    intervention_rng.choice(n - 1, m, replace=False).tolist()
                )
                interventions.append(("random", repeat, indices, None))
                for name, stop in (("shuffle_earlier", n - 1), ("shuffle_full", n)):
                    permutation = intervention_rng.permutation(stop).tolist() + list(
                        range(stop, n)
                    )
                    interventions.append((name, repeat, [], permutation))
            interventions.append(("delete_final", 0, [n - 1], None))
            for name, repeat, indices, permutation in interventions:
                variants = []
                if permutation is not None:
                    changed = [prefix[i] for i in permutation]
                    variants.append(
                        (
                            name,
                            self._sequence(changed, length),
                            sum(i != j for i, j in enumerate(permutation)) / n,
                            {},
                        )
                    )
                else:
                    removed = set(indices)
                    packed = [item for i, item in enumerate(prefix) if i not in removed]
                    variants.append(
                        (
                            name + "_packed",
                            self._sequence(packed, length),
                            len(indices) / n,
                            {},
                        )
                    )
                    if name != "delete_final":
                        holes = sequence.clone()
                        holes[0, [length - n + i for i in indices]] = 0
                        variants.append(
                            (name + "_holes_unverified", holes, len(indices) / n, {})
                        )
                        substituted = list(prefix)
                        donors = {}
                        # Donors must be unseen in the complete training history,
                        # including events outside the truncated model context.
                        donor_allowed = np.ones(itemnum + 1, dtype=bool)
                        donor_allowed[list(set(history) | {0})] = False
                        for index in indices:
                            decile = int(deciles[prefix[index]])
                            donor = None
                            for d in sorted(
                                range(10), key=lambda d: (abs(d - decile), d)
                            ):
                                options = buckets[d][donor_allowed[buckets[d]]]
                                if options.size:
                                    donor = int(donor_rng.choice(options))
                                    donors[str(index)] = {
                                        "item": donor,
                                        "decile": d,
                                        "source_decile": decile,
                                    }
                                    donor_allowed[donor] = False
                                    break
                            if donor is None:
                                break
                            substituted[index] = donor
                        if len(donors) == len(indices):
                            variants.append(
                                (
                                    name + "_frequency_substitution",
                                    self._sequence(substituted, length),
                                    len(indices) / n,
                                    {"donors": donors},
                                )
                            )
                        else:
                            omissions.append(
                                {
                                    "user": user,
                                    "intervention": name,
                                    "reason": "insufficient unseen substitution donors",
                                }
                            )
                for variant, seq, strength, extra in variants:
                    changed_scores = self._scores(self._features(seq), candidates)
                    changed, top = self._outcome(changed_scores, candidates, target)
                    records.append(
                        {
                            "user": user,
                            "intervention": variant,
                            "repeat": repeat,
                            "indices": indices,
                            "permutation": permutation,
                            "effective_strength": strength,
                            "baseline": baseline,
                            "changed": changed,
                            "sampled_top10_overlap": len(set(top) & set(baseline_top))
                            / 10.0,
                            **extra,
                        }
                    )
            embedding = self.view.item_emb
            candidate_embeddings = embedding(
                torch.tensor(candidates, device=self.device)
            )
            history_embeddings = embedding(torch.tensor(prefix, device=self.device))
            control_scores = {
                "last_item_dot": candidate_embeddings @ history_embeddings[-1],
                "mean_normalized_history_similarity": F.normalize(
                    candidate_embeddings, dim=-1
                )
                @ F.normalize(history_embeddings, dim=-1).mean(dim=0),
            }
            transition_scores = np.zeros(len(candidates), dtype=float)
            weights = np.asarray([frequency[i] for i in prefix], dtype=float)
            weights /= weights.sum()
            for weight, source in zip(weights, prefix):
                outgoing = transitions.get(source, {})
                transition_scores += weight * np.asarray(
                    [
                        (outgoing.get(int(dest), 0) + 1.0)
                        / (totals.get(source, 0) + itemnum)
                        for dest in candidates
                    ]
                )
            control_scores["training_transition_add_one"] = transition_scores
            for name, scores in control_scores.items():
                if isinstance(scores, torch.Tensor):
                    scores = scores.float().cpu().numpy()
                changed, top = self._outcome(scores, candidates, target)
                records.append(
                    {
                        "user": user,
                        "intervention": name,
                        "repeat": 0,
                        "baseline": baseline,
                        "changed": changed,
                        "sampled_top10_overlap": len(set(top) & set(baseline_top))
                        / 10.0,
                    }
                )
        optional_available = time.monotonic() < deadline
        alignment = (
            self.analyze_embeddings_collapse_cosine(sample_size=2048)
            if optional_available
            else None
        )
        if time.monotonic() < deadline:
            margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                train_loader, num_batches=10, eps=0.5
            )
        else:
            margin = dict.fromkeys(
                ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate")
            )
            breakdown = {
                "available": False,
                "reason": "optional time allowance exhausted",
            }
        summaries = self._summarize(records)
        strata = {}
        for label, predicate in (
            ("prefix_length_5_to_19", lambda d: d["prefix_length"] < 20),
            ("prefix_length_20_plus", lambda d: d["prefix_length"] >= 20),
            ("no_repeated_ids", lambda d: d["repeated_item_fraction"] == 0),
            ("repeated_ids", lambda d: d["repeated_item_fraction"] > 0),
        ):
            strata[label] = self._summarize(
                [r for r in records if predicate(descriptions[r["user"]])]
            )
        metrics = {
            "dataset": dataset,
            "embedding_collapse_score": alignment,
            "pos_neg_margin_mean": margin["mean"],
            "pos_neg_margin_p10": margin["p10"],
            "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
            "pos_neg_small_margin_rate": margin["small_margin_rate"],
            "pos_neg_margin_category_breakdown": breakdown,
            "scoring_contract": scoring_contract,
            "temporal_training_pairs": summaries,
            "descriptive_strata": strata,
            "counts": {
                "training_users": len(histories),
                "eligible_users": len(eligible),
                "excluded_short_prefix_users": len(histories) - len(eligible),
                "selected_users": len(selected),
                "evaluated_users": len(pools),
                "simulator_users": len(recommendations),
            },
            "four_dataset_macro_difference": None,
            "official_validation_hr10": None,
            "checkpoint_validation_associations": None,
            "omissions": omissions,
            "elapsed_wall_seconds": time.monotonic() - started,
            "budget": {
                "optional_deadline_seconds_per_dataset": 150,
                "required_catalog_scoring_completed_first": True,
                "hard_gpu_time_enforced": False,
            },
        }
        definitions = {
            "dataset": "Dataset identifier exposed by view; unspecified means routing identity was unavailable.",
            "embedding_collapse_score": "Legacy name: (sampled mean off-diagonal item-embedding cosine + 1)/2, rounded to four decimals. Computed by the squared-sum identity without a Gram matrix. Directional alignment only; not proof of collapse or diversity. Null if omitted, fewer than two items exist, or a sampled vector has zero norm.",
            "pos_neg_margin_mean": "Mean positive-minus-negative training logit over nonpadding sequence/positive/negative positions in up to ten loader batches; null if absent.",
            "pos_neg_margin_p10": "Empirical lower tenth percentile of the same training margins; null if absent.",
            "pos_neg_neg_beats_pos_rate": "Fraction of valid training pairs with margin < 0; null if absent.",
            "pos_neg_small_margin_rate": "Fraction of valid training pairs with margin < 0.5; null if absent.",
            "pos_neg_margin_category_breakdown": "Category lists stay empty because item categories are unavailable. Counts describe training positions only.",
            "scoring_contract": "Actual view encoder/scorer with final array-position readout. Candidate scoring uses B x 1 x D features and B x 1 x K items, matching the sequence margin contract. When model.predict exists, every evaluated baseline must agree at rtol=1e-4, atol=1e-5; the recorded error is the maximum across checked users. Position/mask controls remain unsupported, so holes cannot identify pure recency dependence.",
            "temporal_training_pairs": "Final training event is target, preceding context is prefix, including earlier target occurrences. Fixed target plus 99 unique uniform negatives excludes original prefix. Ties use descending score then ascending item ID. Ranks are one-based, reciprocal rank=1/rank, hr10=1(rank<=10), sampled_top10_overlap=intersection size/10. Differences are changed minus baseline. Repeats average within user before means and 1000 seed-42 user bootstrap percentile intervals. Intervals condition on this fixed trained model and candidate pools. Targets participated in optimization. Packed deletion changes positions; holes have unverified padding behavior. Substitutions preserve positions with unseen same-frequency-decile items or deterministic nearest-decile fallback; these can be out of distribution. Training-fitted dot-product, normalized-similarity and frequency-weighted add-one transition controls are descriptive, not generalization evidence.",
            "descriptive_strata": "Same paired summaries split by truncated prefix length and repeated-ID fraction. Overlapping descriptive groups; repeated IDs do not establish semantic purpose.",
            "counts": "Eligibility requires five preceding training events. Simulator users are independently selected from nonempty complete training histories, up to five per dataset.",
            "four_dataset_macro_difference": "Unavailable in a single-view call. Never substitute a subset-category average. Requires all four dataset outcomes and independent within-dataset user resampling with equal category weights.",
            "official_validation_hr10": "Unavailable: this adapter reads training histories only and does not call or reconstruct the host evaluator.",
            "checkpoint_validation_associations": "Unavailable without externally logged checkpoint identities and official validation aggregates. No Spearman correlations, direction agreement, category consistency or validation intervals are fabricated.",
            "omissions": "Explicit optional-budget and candidate-availability omissions; unsupported positional controls appear in scoring_contract.",
            "elapsed_wall_seconds": "Measured elapsed wall time through metric construction; not a measured GPU-time total.",
            "budget": "Optional work stops at user boundaries after a 150-second per-dataset wall allowance. Required full-catalog scoring runs first and may exceed it; this is not a hard GPU-time cap. Four dataset calls allocate 600 optional wall seconds in total.",
        }
        artifact = {
            "protocol": "paired_temporal_v2_seed42_user_streams_complete_history_donors",
            "metrics": metrics,
            "metric_definitions": definitions,
            "candidate_pools": pools,
            "user_descriptions": descriptions,
            "paired_records": records,
            "full_catalog_recommendations": recommendations,
            "checkpoint_identity": getattr(self.view, "checkpoint_id", None),
            "cache_policy": "no cross-call representation caches",
        }
        # Separate invocations retain their own pools and paired records.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=safe_dataset + "_",
            suffix=".json",
            dir=str(root),
            delete=False,
        ) as handle:
            json.dump(artifact, handle, indent=2, allow_nan=False)
            artifact_path = handle.name
        metrics["artifact_path"] = artifact_path
        definitions["artifact_path"] = (
            "Trial-local JSON containing fixed pools, intervention indices, paired outcomes, and required unseen full-catalog recommendations. No hidden labels are used."
        )
        # Static review checks: scorer dimensions, finite values, unique targets,
        # RNG isolation, complete-history donor exclusion, and inference parity.
        # No training or runtime checks were executed in this code-only review.
        # Four-dataset aggregation, checkpoint associations, and verified padding
        # controls remain explicitly unavailable; this patch claims no HR gain.
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
