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
    - Separate synthetic oracle correctness, metamorphic consistency, actual-model temporal sensitivity, and official predictive performance.
    - Report each mutant and relation with its denominator; surviving a relation is not proof of correctness.
    - Internal model.predict agreement is not verification against the generated production predictor.
    - Unavailable production fidelity or restored buffer mutations limit temporal interpretation.
    - The declared two-model predictive comparison is unexecuted unless actual host results are supplied.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Bind diagnostics to the actual trained module
    def __init__(self, view, device):
        self.view = view
        self.actual_model = view.model
        if not isinstance(self.actual_model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.model = self.actual_model
        self.device = torch.device(device)
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

    ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic work, audit scores safely, and persist verified isolation status
    def _sequence(self, history, length):
        seq = torch.zeros((1, length), dtype=torch.long, device=self.device)
        history = list(history)[-length:]
        if history:
            seq[0, -len(history) :] = torch.tensor(history, device=self.device)
        return seq

    def _features(self, seq):
        # Candidate-conditioned models bind evidence to tensor identity. Their
        # selection API must create the final readout and its matching context;
        # slicing and then unsqueezing would discard that binding.
        if callable(getattr(self.actual_model, "encode", None)) and callable(
            getattr(self.actual_model, "select_positions", None)
        ):
            features, context = self.actual_model.encode(seq)
            final, _ = self.actual_model.select_positions(features, context, -1)
        else:
            features = self.view.log2feats(seq)
            final = features[:, -1:, :]
        if features.ndim != 3 or features.shape[:2] != seq.shape:
            raise ValueError("Temporal probe requires B x L x D features")
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite diagnostic features")
        return final

    def _scores(self, features, candidates):
        items = torch.as_tensor(
            candidates, dtype=torch.long, device=self.device
        ).reshape(1, -1)
        if features.ndim != 3 or tuple(features.shape[:2]) != (1, 1):
            raise ValueError("Candidate scoring requires one B x 1 x D readout")
        scorer = (
            self.actual_model.predict_score
            if callable(getattr(self.actual_model, "select_positions", None))
            else self.view.predict_score
        )
        scores = scorer(features, items.unsqueeze(1))
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
    def _audit(reference, observed):
        reference = np.asarray(reference, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float32)
        if reference.shape != observed.shape:
            return {
                "attempted": 1,
                "failures": 1,
                "shape_mismatch": True,
                "nonfinite": None,
                "max_abs": None,
                "max_scaled": None,
            }
        finite = np.isfinite(reference) & np.isfinite(observed)
        # Preserve float32 audit inputs, but avoid overflow in subtraction.
        reference_finite = reference[finite].astype(np.float64)
        observed_finite = observed[finite].astype(np.float64)
        residual = np.abs(observed_finite - reference_finite)
        scaled = residual / (1e-5 + 1e-4 * np.abs(reference_finite))
        return {
            "attempted": 1,
            "failures": int(not finite.all() or bool((scaled > 1).any())),
            "shape_mismatch": False,
            "nonfinite": int((~finite).sum()),
            "max_abs": float(residual.max()) if residual.size else None,
            "max_scaled": float(scaled.max()) if scaled.size else None,
        }

    @classmethod
    def _known_score_calibration(cls):
        # Expected values and route tables are independently specified literals.
        # These tiny fixtures exercise mapping logic, not real catalog mappings.
        tables = {
            "Movies_and_TV": {"history": 1, "a": 2, "b": 3},
            "Industrial_and_Scientific": {"history": 1, "a": 2, "b": 3},
            "Electronics": {"history": 1, "a": 2, "b": 3},
            "CDs_and_Vinyl": {"history": 1, "a": 2, "b": 3},
        }
        rows = {
            "Movies_and_TV": [[0, 0], [1, 2], [3, 0], [0, 4]],
            "Industrial_and_Scientific": [[0, 0], [2, 1], [5, 0], [0, 6]],
            "Electronics": [[0, 0], [1, 3], [7, 0], [0, 2]],
            "CDs_and_Vinyl": [[0, 0], [3, 1], [2, 0], [0, 9]],
        }
        expected = {
            "Movies_and_TV": [3, 8],
            "Industrial_and_Scientific": [10, 6],
            "Electronics": [7, 6],
            "CDs_and_Vinyl": [6, 9],
        }
        expected_ranking = {
            "Movies_and_TV": ["b", "a"],
            "Industrial_and_Scientific": ["a", "b"],
            "Electronics": ["a", "b"],
            "CDs_and_Vinyl": ["b", "a"],
        }
        routes = list(tables)
        mutants = (
            "correct",
            "constant",
            "ignored_history",
            "candidate_position",
            "sorted_unrestored",
            "off_by_one",
            "swapped_rows",
            "wrong_route",
            "stale_mapping",
            "reversed_columns",
        )

        def score(route, candidates, mutant, history="history"):
            active = (
                routes[(routes.index(route) + 1) % 4]
                if mutant == "wrong_route"
                else route
            )
            table = dict(tables[active])
            embedding = np.asarray(rows[active], dtype=np.float32).copy()
            if mutant == "stale_mapping":
                table["a"], table["b"] = table["b"], table["a"]
            if mutant == "swapped_rows":
                embedding[[2, 3]] = embedding[[3, 2]]
            if mutant == "constant":
                return np.zeros(len(candidates), dtype=np.float32)
            if mutant == "candidate_position":
                return np.arange(len(candidates), dtype=np.float32)
            if mutant == "ignored_history":
                query = np.asarray([1, 0], dtype=np.float32)
            else:
                query = embedding[table[history]]
            ids = [table[item] for item in candidates]
            if mutant == "off_by_one":
                ids = [(item + 1) % len(embedding) for item in ids]
            result = embedding[ids] @ query
            if mutant == "sorted_unrestored":
                result = result[np.argsort(ids)]
            if mutant == "reversed_columns":
                result = result[::-1]
            return result

        matrix, correct_failures = {}, 0
        for route in routes:
            matrix[route] = {}
            for mutant in mutants:
                original = score(route, ["a", "b"], mutant)
                reordered = score(route, ["b", "a"], mutant)[::-1]
                oracle = cls._audit(expected[route], original)
                equivariance = cls._audit(original, reordered)
                # An independent history-dependent oracle: querying row a
                # yields [9,0], [25,0], [49,0], [4,0], respectively.
                alternate_expected = {
                    "Movies_and_TV": [9, 0],
                    "Industrial_and_Scientific": [25, 0],
                    "Electronics": [49, 0],
                    "CDs_and_Vinyl": [4, 0],
                }
                history_oracle = cls._audit(
                    alternate_expected[route],
                    score(route, ["a", "b"], mutant, history="a"),
                )
                checks = {
                    "known_scores": oracle,
                    "candidate_equivariance": equivariance,
                    "history_oracle": history_oracle,
                }
                matrix[route][mutant] = {
                    name: {
                        "outcome": (
                            ("false_alarm" if result["failures"] else "passed")
                            if mutant == "correct"
                            else ("detected" if result["failures"] else "survived")
                        ),
                        **result,
                    }
                    for name, result in checks.items()
                }
                if mutant == "correct":
                    correct_failures += sum(r["failures"] for r in checks.values())
                    ranked = ["a", "b"] if original[0] > original[1] else ["b", "a"]
                    if ranked != expected_ranking[route]:
                        raise AssertionError("Known-score ranking fixture failed")
        return {
            "fixture_kind": "independent synthetic route and mapping oracles",
            "matrix": matrix,
            "correct_control_checks": len(routes) * 3,
            "correct_control_false_alarms": correct_failures,
            "real_mapping_certification": False,
            "fixture_serialization": {
                "status": "unavailable",
                "reason": "No fixture checkpoint round trip implemented",
            },
            "joint_relabeling": {
                "status": "unavailable",
                "reason": "No production mapping relabeling interface exposed",
            },
        }

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
        # Four calls allocate at most 120 seconds of optional wall allowance.
        # A wall deadline is not a reservation in the host GPU budget ledger.
        optional_deadline = started + 30.0
        previous_modes = [
            (module, module.training) for module in self.actual_model.modules()
        ]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        import copy

        buffers = {
            name: (buffer, buffer.detach().clone())
            for name, buffer in self.actual_model.named_buffers()
        }
        generators, seen = [], set()

        def capture(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Generator):
                generators.append((value, value.get_state()))
            elif isinstance(value, np.random.Generator):
                generators.append((value, copy.deepcopy(value.bit_generator.state)))
            elif isinstance(value, np.random.RandomState):
                generators.append((value, value.get_state()))
            elif isinstance(value, random.Random):
                generators.append((value, value.getstate()))
            elif isinstance(value, dict):
                for child in value.values():
                    capture(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    capture(child)

        for owner in [self.view, *self.actual_model.modules()]:
            for value in vars(owner).values():
                capture(value)
        result = None
        mutations = []
        try:
            self.actual_model.eval()
            with torch.inference_mode():
                # The active loader is a compatibility argument only.
                result = self._run_temporal_diagnosis(
                    None, started, optional_deadline, Path, os, time
                )
            return result
        finally:
            try:
                with torch.no_grad():
                    current = dict(self.actual_model.named_buffers())
                    for name in set(current) | set(buffers):
                        parent_name, _, leaf = name.rpartition(".")
                        parent = (
                            self.actual_model.get_submodule(parent_name)
                            if parent_name
                            else self.actual_model
                        )
                        if name not in buffers:
                            mutations.append(name)
                            del parent._buffers[leaf]
                            continue
                        original, saved = buffers[name]
                        value = current.get(name)
                        equal = value is original and value.shape == saved.shape
                        if equal:
                            equal = torch.equal(value, saved)
                            if not equal and value.is_floating_point():
                                equal = torch.allclose(
                                    value, saved, rtol=0, atol=0, equal_nan=True
                                )
                        if not equal:
                            mutations.append(name)
                            original.data = saved.clone()
                            parent._buffers[leaf] = original
            finally:
                for module, mode in previous_modes:
                    module.training = mode
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.random.set_rng_state(torch_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
                for generator, state in generators:
                    if isinstance(generator, torch.Generator):
                        generator.set_state(state)
                    elif isinstance(generator, np.random.Generator):
                        generator.bit_generator.state = state
                    elif isinstance(generator, np.random.RandomState):
                        generator.set_state(state)
                    else:
                        generator.setstate(state)
            if result is not None:
                result["metrics"]["state_isolation"] = {
                    "registered_buffer_mutations_restored": mutations,
                    "active_loader_consumed": False,
                    "subsequent_optimizer_step_equivalence": "unmeasured",
                }
                result["metric_definitions"]["state_isolation"] = (
                    "Modes, global RNGs, accessible explicit generators and registered buffers restored in finally. "
                    "Arbitrary Python-side model mutations and hidden generators are outside this guarantee. "
                    "A restored buffer mutation invalidates the affected diagnostic interpretation."
                )
                if mutations:
                    result["metrics"]["temporal_interpretation_verified"] = False
                # The artifact was initially written before finally restored state.
                # Persist the same finalized evidence returned to the caller.
                artifact_path = result["metrics"]["artifact_path"]
                with open(artifact_path) as handle:
                    artifact = json.load(handle)
                artifact["metrics"] = result["metrics"]
                artifact["metric_definitions"] = result["metric_definitions"]
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", dir=str(Path(artifact_path).parent), delete=False
                ) as handle:
                    temporary = handle.name
                    json.dump(artifact, handle, indent=2, allow_nan=False)
                os.replace(temporary, artifact_path)

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
        import hashlib

        dataset = getattr(self.view, "dataset_name", None) or getattr(
            self.actual_model, "dataset_name", None
        )
        required_routes = {
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        }
        if dataset not in required_routes:
            raise ValueError(
                "Diagnostics require an explicit valid dataset_name on view or model"
            )
        checkpoint = hashlib.sha256()
        for name, tensor in sorted(self.actual_model.state_dict().items()):
            checkpoint.update(name.encode())
            checkpoint.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            checkpoint.update(
                tensor.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        calibration = self._known_score_calibration()
        relation_records, training_margins = [], []
        production_checks = []
        # Always stay beneath the trial's source directory; never use DATA_DIR.
        root = Path(__file__).resolve().parent / "working" / "diagnostics"
        root.mkdir(parents=True, exist_ok=True)
        safe_dataset = "".join(c if c.isalnum() or c in "_-" else "_" for c in dataset)

        def user_key(user):
            return hashlib.sha256(
                json.dumps([42, dataset, str(user)]).encode()
            ).digest()

        eligible = sorted(
            (u for u, h in histories.items() if len(h) >= 6), key=user_key
        )
        selected = eligible[:64]
        simulator_users = sorted((u for u, h in histories.items() if h), key=user_key)[
            :5
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
            candidate_permutation = candidate_rng.permutation(len(candidates))
            pools[user] = {
                "target": target,
                "prefix": prefix,
                "candidates": candidates,
                "candidate_permutation": candidate_permutation.tolist(),
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
            training_margins.append(float(base_scores[0] - base_scores[1]))
            embedding = self.view.item_emb
            candidate_embeddings = (
                embedding(torch.tensor(candidates, device=self.device)).detach().clone()
            )
            latest_scores = (
                (
                    candidate_embeddings
                    @ embedding(torch.tensor(prefix[-1], device=self.device)).detach()
                )
                .float()
                .cpu()
                .numpy()
            )
            frequency_scores = np.log1p(
                np.asarray([frequency[i] for i in candidates], dtype=np.float32)
            )
            permuted_candidates = np.asarray(candidates)[candidate_permutation].tolist()
            inverse = np.argsort(candidate_permutation)
            for backend, original, permuted in (
                ("trained", base_scores, self._scores(features, permuted_candidates)),
                (
                    "last_item_dot",
                    latest_scores,
                    (
                        embedding(
                            torch.tensor(permuted_candidates, device=self.device)
                        ).detach()
                        @ embedding(
                            torch.tensor(prefix[-1], device=self.device)
                        ).detach()
                    )
                    .float()
                    .cpu()
                    .numpy(),
                ),
                (
                    "training_frequency",
                    frequency_scores,
                    np.log1p(
                        np.asarray(
                            [frequency[i] for i in permuted_candidates],
                            dtype=np.float32,
                        )
                    ),
                ),
            ):
                relation_records.append(
                    {
                        "user": user,
                        "backend": backend,
                        "relation": "candidate_equivariance",
                        **self._audit(original, permuted[inverse]),
                    }
                )
            production = getattr(self.view, "production_predict", None)
            if callable(production):
                production_scores = production(
                    user, prefix, candidates, dataset_name=dataset
                )
                if isinstance(production_scores, torch.Tensor):
                    production_scores = production_scores.detach().float().cpu().numpy()
                check = self._audit(base_scores, production_scores)
                check["user"] = user
                if not check["failures"]:
                    _, production_top = self._outcome(
                        production_scores, candidates, target
                    )
                    _, adapter_top = self._outcome(base_scores, candidates, target)
                    check["top10_equal"] = production_top == adapter_top
                production_checks.append(check)
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
                    if time.monotonic() >= deadline:
                        omissions.append(
                            {
                                "user": user,
                                "intervention": variant,
                                "repeat": repeat,
                                "reason": "optional time allowance exhausted before scoring",
                            }
                        )
                        continue
                    changed_scores = self._scores(self._features(seq), candidates)
                    changed_prefix = seq[0][seq[0].ne(0)].cpu().tolist()
                    relation_records.append(
                        {
                            "user": user,
                            "backend": "training_frequency",
                            "relation": "history_invariance",
                            "intervention": variant,
                            **self._audit(
                                frequency_scores,
                                np.log1p(
                                    np.asarray(
                                        [frequency[i] for i in candidates],
                                        dtype=np.float32,
                                    )
                                ),
                            ),
                        }
                    )
                    if changed_prefix and changed_prefix[-1] == prefix[-1]:
                        changed_latest = (
                            (
                                candidate_embeddings
                                @ embedding(
                                    torch.tensor(changed_prefix[-1], device=self.device)
                                ).detach()
                            )
                            .float()
                            .cpu()
                            .numpy()
                        )
                        relation_records.append(
                            {
                                "user": user,
                                "backend": "last_item_dot",
                                "relation": "latest_preserving_history_invariance",
                                "intervention": variant,
                                **self._audit(latest_scores, changed_latest),
                            }
                        )
                    # Trained-model invariance is not required: zero or nonzero
                    # temporal sensitivity is an empirical observation.
                    changed, top = self._outcome(changed_scores, candidates, target)
                    records.append(
                        {
                            "user": user,
                            "intervention": variant,
                            "repeat": repeat,
                            "indices": indices,
                            "permutation": permutation,
                            "effective_strength": strength,
                            "sequence_ids": seq[0].cpu().tolist(),
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
        # Never iterate the active loader. Reuse one fixed pair from each
        # independently selected final-training transition instead.
        values = np.asarray(training_margins, dtype=np.float32)
        margin = {
            "mean": float(values.mean()) if values.size else None,
            "p10": (
                float(np.quantile(values, 0.1, method="inverted_cdf"))
                if values.size
                else None
            ),
            "neg_beats_pos_rate": float((values < 0).mean()) if values.size else None,
            "small_margin_rate": float((values < 0.5).mean()) if values.size else None,
        }
        breakdown = {
            "available": False,
            "reason": "Item categories are unavailable",
            "measured_users": int(values.size),
        }
        panel_payload = {
            "dataset": dataset,
            "seed": 42,
            "maximum_users": 64,
            "draws": 3,
            "candidate_pools": pools,
            "edits": [
                {
                    k: r[k]
                    for k in (
                        "user",
                        "intervention",
                        "repeat",
                        "indices",
                        "permutation",
                        "sequence_ids",
                    )
                }
                for r in records
                if "sequence_ids" in r
            ],
        }
        panel_fingerprint = hashlib.sha256(
            json.dumps(panel_payload, sort_keys=True).encode()
        ).hexdigest()
        production_status = {
            "status": "measured" if production_checks else "unavailable",
            "reason": (
                None
                if production_checks
                else (
                    "No independently generated production_predict callable exposed by view; "
                    "model.predict agreement is only an internal interface check"
                )
            ),
            "checks": production_checks,
            "checkpoint_and_mapping_identity": "caller must supply the identical routed production artifact",
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
            "adapter_version": "bounded_temporal_calibration_v3",
            "checkpoint_sha256": checkpoint.hexdigest(),
            "panel_fingerprint": panel_fingerprint,
            "item_map_fingerprint": getattr(self.view, "item_map_fingerprint", None),
            "item_map_fingerprint_reason": (
                None
                if getattr(self.view, "item_map_fingerprint", None)
                else "External item mapping is not exposed; contiguous IDs are not a mapping fingerprint"
            ),
            "dtype": str(self.view.item_emb.weight.dtype),
            "device": str(self.device),
            "synthetic_calibration": calibration,
            "metamorphic_relations": relation_records,
            "production_fidelity": production_status,
            "production_score_checks_passed": (
                all(not check["failures"] for check in production_checks)
                if production_checks
                else None
            ),
            # Score agreement alone does not establish checkpoint/map identity
            # or verify the unsupported positional and padding controls.
            "temporal_interpretation_verified": False,
            "predictive_comparison": {
                "status": "unexecuted",
                "reason": "Diagnostic adapter cannot reserve host budget, fit the two models, or request official evaluations",
                "configurations": ["baseline", "baseline_plus_recency_residual"],
                "half_life_positions": 10,
                "alpha_initial_value": 0,
                "checkpoint_rule": "final scheduled epoch only",
                "minimum_useful_absolute_macro_gain": 0.005,
                "maximum_comparison_gpu_seconds": 8000,
                "validation_driven_variants": 0,
            },
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
                "optional_deadline_seconds_per_dataset": 30,
                "maximum_declared_calibration_gpu_seconds": 120,
                "host_budget_reservation": "unavailable to this adapter",
                "required_catalog_scoring_completed_first": True,
                "hard_gpu_time_enforced": False,
            },
        }
        definitions = {
            "dataset": "Dataset identifier exposed by view; unspecified means routing identity was unavailable.",
            "embedding_collapse_score": "Legacy name: (sampled mean off-diagonal item-embedding cosine + 1)/2, rounded to four decimals. Computed by the squared-sum identity without a Gram matrix. Directional alignment only; not proof of collapse or diversity. Null if omitted, fewer than two items exist, or a sampled vector has zero norm.",
            "pos_neg_margin_mean": "Mean final-training target score minus the first fixed diagnostic negative score; one pair per evaluated user. Active loader is never consumed. Negative excludes the retained original prefix and target; it may occur elsewhere in the complete training history. Not validation HR@10.",
            "pos_neg_margin_p10": "Inverse empirical CDF at 0.1 over the same user-level fixed-pair margins; null if absent.",
            "pos_neg_neg_beats_pos_rate": "Fraction of these final-training user pairs with margin < 0; null if absent.",
            "pos_neg_small_margin_rate": "Fraction of these final-training user pairs with margin < 0.5; null if absent. Cutoff is descriptive and uncalibrated.",
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
            "budget": "Optional work stops at user boundaries after 30 wall seconds per category. Required full-catalog scoring and in-flight work can overrun. Four calls allocate 120 optional wall seconds, not a hard GPU cap or host-ledger reservation. The host must enforce remaining shared allowance; no budget compliance is inferred from this timer.",
            "synthetic_calibration": "Four literal route tables, distinct embedding rows, independently specified raw-score and ranking oracles, and one fault at a time. Matrix reports each relation separately; surviving permutation checks do not certify correctness. No universal mutation detection rate or real mapping certification is claimed. Serialization and relabeling checks are explicitly unavailable.",
            "metamorphic_relations": "Float32 audit with atol=1e-5 and rtol=1e-4. max_scaled=max(abs(observed-reference)/(atol+rtol*abs(reference))) on finite entries. Nonfinite entries fail. Candidate permutations are inverted before every per-column comparison. Latest-item control uses detached raw shared embeddings without normalization; its history invariant applies only to nonempty histories retaining the latest item. Frequency control is log1p(public routed training occurrence count), independent of history. Neither is a trained predictive reference.",
            "production_fidelity": "Independent production_predict is used only if exposed by view. Internal actual_model.predict parity is not production verification. Missing production access, checkpoint identity checks or mapping evidence limit temporal interpretation; this does not invalidate independently measured host validation.",
            "production_score_checks_passed": "Null when no production comparisons ran; otherwise whether every attempted raw-score audit passed. Does not verify checkpoint, item-map, device or dtype identity.",
            "temporal_interpretation_verified": "False: this adapter does not independently verify production checkpoint/map identity or implement verified positional and padding controls. Actual-model sensitivity remains descriptive; passing score checks alone cannot establish causal temporal effects.",
            "predictive_comparison": "Unexecuted declaration only. This diagnostic-only patch does not train, change the recommendation architecture, launch variants, or fabricate official validation results. The requested two-model final-checkpoint comparison requires separate host execution and budget reservation.",
            "panel_fingerprint": "SHA256 of routed candidate pools, exact prefixes, candidate permutations and actual persisted edits. Stable seed-42 category/user digests select at most 64 users; stochastic interventions have three draws. Deadline omissions remain explicit. This is a new bounded artifact and does not modify prior panels.",
        }
        artifact = {
            "protocol": "bounded_temporal_calibration_v3",
            "panel": panel_payload,
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
