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

    ### >>> Self_EvolveRec-BLOCK-START: Evidence-bounded interpretation of descriptive training audits
    Rules:
    - Keep core_findings and key_implications to at most three entries each.
    - Ground every statement in measured values, definitions, eligibility counts,
      uncertainty intervals, controls, and missingness. Null means unavailable.
    - These targets were previously observed during model fitting. This is a
      descriptive training-example audit, not an auxiliary generalization test.
    - Compact deletion changes surviving positions. Position-preserving results
      are unavailable unless a verified architecture-aware adapter was used.
    - Neither greater nor smaller deletion sensitivity is inherently desirable.
      Explain the signs of recent-minus-control differences before interpreting them.
    - Ranks and overlap depend on the fixed auxiliary candidate set and tie rule;
      they are not official validation metrics or full-catalog recommendation metrics.
    - Bootstrap intervals condition on this fitted model and sampled users. They
      exclude training-seed variation, adaptive selection bias, and uncertainty
      beyond the empirical within-category user bootstrap.
    - Do not infer causal preferences, simulator traits, dislikes, semantic
      incompatibility, or validation gains from these measurements. Functional fit
      requires explicit public text evidence and otherwise remains unknown.
    - Missing probes cannot establish health. If evidence is insufficient, use
      NEEDS_IMPROVEMENT and state the evidence limitation without diagnosing failure.
    - Auxiliary fresh-initialization suffix experiments and frozen gain-prediction
      experiments have not been run; describe neither as established evidence.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Validated routed audit with compatibility scoring and bounded negative sampling
    DATASETS = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )
    EFFECTS = (
        "recent_rank_delta",
        "control_rank_delta",
        "rank_difference",
        "recent_hr10_delta",
        "control_hr10_delta",
        "hr10_difference",
        "recent_overlap",
        "control_overlap",
        "overlap_difference",
    )

    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.user_train = view.user_train

    @staticmethod
    def _seed(*parts):
        import hashlib

        payload = json.dumps([42, *map(str, parts)], ensure_ascii=True)
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

    @staticmethod
    def _histories(view):
        data = view.user_train
        return data.get("History", data)

    def _routes(self):
        # Optional explicit multi-route adapter; never guess a category from IDs.
        routes = getattr(self.model, "dataset_views", None)
        if isinstance(routes, dict):
            return {ds: routes.get(ds) for ds in self.DATASETS}
        ds = getattr(self.model, "dataset_name", None)
        if ds is None:
            ds = getattr(self.network, "dataset_name", None)
        return {name: self.model if name == ds else None for name in self.DATASETS}

    @torch.no_grad()
    def _score_windows(self, view, windows, candidates, maxlen):
        network = view.model
        device = next(network.parameters()).device
        seq = torch.zeros((len(windows), maxlen), dtype=torch.long, device=device)
        for row, window in enumerate(windows):
            # DEBUG: -0 selects the entire row; skip empty histories and bound context.
            window = list(window)[-maxlen:]
            if window:
                seq[row, -len(window) :] = torch.as_tensor(
                    window, dtype=torch.long, device=device
                )
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        items = items.unsqueeze(0).expand(len(windows), -1)
        # The compatibility hooks have a prescribed signature; an arbitrary
        # network.predict method may require user IDs or other arguments.
        features = view.log2feats(seq)[:, -1, :]
        scores = view.predict_score(features, items)
        scores = scores.detach().float().cpu().numpy()
        if scores.shape != (len(windows), len(candidates)):
            raise ValueError("Scoring adapter returned an incompatible shape")
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite diagnostic candidate scores")
        return scores

    def _audit_route(self, ds, view):
        network = view.model
        itemnum = int(view.item_emb.weight.shape[0] - 1)
        maxlen = int(network.maxlen)
        if maxlen < 1:
            raise ValueError(f"Invalid model context length for {ds}: {maxlen}")
        declared_route = getattr(view, "dataset_name", None)
        if declared_route is None:
            declared_route = getattr(network, "dataset_name", None)
        if declared_route is not None and declared_route != ds:
            raise ValueError(
                f"Diagnostic route {ds!r} contains a view for {declared_route!r}"
            )
        histories = self._histories(view)
        if not isinstance(histories, dict):
            raise TypeError(f"Expected a training-history dictionary for {ds}")
        frequency = Counter()
        eligible = []
        counts = {
            "total_users": len(histories),
            "invalid_histories": 0,
            "short_history_exclusions": 0,
            "insufficient_negatives": 0,
        }
        for uid in sorted(histories, key=lambda x: (str(x), type(x).__name__)):
            history = [int(x) for x in histories[uid]]
            if any(x < 1 or x > itemnum for x in history):
                counts["invalid_histories"] += 1
                continue
            frequency.update(history)
            window = history[:-1][-maxlen:]
            if len(window) < 3:
                counts["short_history_exclusions"] += 1
                continue
            if itemnum - len(set(history)) < 99:
                counts["insufficient_negatives"] += 1
                continue
            eligible.append(uid)
        counts["eligible_users"] = len(eligible)
        rng = random.Random(self._seed(ds, "user_selection"))
        selected = rng.sample(eligible, min(200, len(eligible)))
        counts["selected_users"] = len(selected)
        counts["eligible_not_sampled"] = len(eligible) - len(selected)
        if not selected:
            return {
                "status": "unavailable",
                "reason": "No eligible training users",
                "counts": counts,
                "users": [],
                "mean": None,
                "ci95": None,
            }

        records, vectors = [], []
        for uid in selected:
            history = [int(x) for x in histories[uid]]
            # Freeze truncation before deleting any occurrence; never refill.
            target = history[-1]
            window = history[:-1][-maxlen:]
            terminal = len(window) - 1
            treatment = terminal - 1
            pool = list(range(treatment))
            draws = [
                random.Random(self._seed(ds, uid, "control", draw)).choice(pool)
                for draw in range(5)
            ]
            excluded = set(history)
            neg_rng = random.Random(self._seed(ds, uid, "candidates"))
            # Sample ranks in the unseen-item complement, then map them to
            # actual IDs. This is uniform without replacement and avoids
            # rejection loops for users covering most of a large catalog.
            available = itemnum - len(excluded)
            sampled_ranks = neg_rng.sample(range(available), 99)
            blocked_ids = np.asarray(sorted(excluded), dtype=np.int64)
            blocked_offsets = blocked_ids - np.arange(len(blocked_ids), dtype=np.int64)
            desired_ids = np.asarray(sampled_ranks, dtype=np.int64) + 1
            negatives = (
                desired_ids
                + np.searchsorted(blocked_offsets, desired_ids, side="right")
            ).tolist()
            candidates = [target] + negatives
            removed = [treatment] + draws
            windows = [window] + [window[:i] + window[i + 1 :] for i in removed]
            scores = self._score_windows(view, windows, candidates, maxlen)
            # Stable descending order: candidate order breaks ties, target first.
            orders = np.argsort(-scores, axis=1, kind="stable")
            ranks = [int(np.flatnonzero(order == 0)[0]) + 1 for order in orders]
            hits = [int(rank <= 10) for rank in ranks]
            base_top = set(orders[0, :10].tolist())
            overlaps = [
                len(base_top.intersection(order[:10].tolist())) / 10.0
                for order in orders[1:]
            ]
            recent_rank = ranks[1] - ranks[0]
            control_rank = float(np.mean(ranks[2:])) - ranks[0]
            recent_hit = hits[1] - hits[0]
            control_hit = float(np.mean(hits[2:])) - hits[0]
            recent_overlap = overlaps[0]
            control_overlap = float(np.mean(overlaps[1:]))
            vector = [
                recent_rank,
                control_rank,
                recent_rank - control_rank,
                recent_hit,
                control_hit,
                recent_hit - control_hit,
                recent_overlap,
                control_overlap,
                recent_overlap - control_overlap,
            ]
            vectors.append(vector)
            records.append(
                {
                    "user_id": str(uid),
                    "target": target,
                    "target_previously_used_in_fitting": True,
                    "original_training_length": len(history),
                    "frozen_window": window,
                    "terminal_index": terminal,
                    "treatment_index": treatment,
                    "control_indices": draws,
                    "control_duplicate_draws": 5 - len(set(draws)),
                    "window_repeat_occurrences": len(window) - len(set(window)),
                    "target_training_frequency": frequency[target],
                    "target_window_occurrences": window.count(target),
                    "removed_items": [window[i] for i in removed],
                    "removed_training_frequencies": [
                        frequency[window[i]] for i in removed
                    ],
                    "removed_window_occurrences": [
                        window.count(window[i]) for i in removed
                    ],
                    "candidates": candidates,
                    "ranks": ranks,
                    "hr10": hits,
                    "top10_items": [
                        [candidates[i] for i in order[:10]] for order in orders
                    ],
                    "compact_overlaps": overlaps,
                    "zero_overlap": [overlap == 0 for overlap in overlaps],
                    "effects": dict(zip(self.EFFECTS, vector)),
                    "position_preserving": None,
                    "position_preserving_reason": (
                        "No verified adapter preserving absolute positions and blocking "
                        "the deleted key/value at every layer, including the recent branch"
                    ),
                }
            )
        values = np.asarray(vectors, dtype=np.float64)
        boot_rng = np.random.default_rng(self._seed(ds, "bootstrap"))
        boot = np.stack(
            [
                values[boot_rng.integers(0, len(values), size=len(values))].mean(axis=0)
                for _ in range(1000)
            ]
        )
        bounds = np.percentile(boot, [2.5, 97.5], axis=0)
        result = {
            "status": "measured",
            "counts": counts,
            "users": records,
            "mean": dict(zip(self.EFFECTS, values.mean(axis=0).tolist())),
            "ci95": {
                name: bounds[:, i].tolist() for i, name in enumerate(self.EFFECTS)
            },
            "item_training_frequency_distribution": dict(
                sorted(Counter(frequency.values()).items())
            ),
            "zero_overlap_rates": {
                "recent": float(np.mean([r["zero_overlap"][0] for r in records])),
                "control": float(np.mean([r["zero_overlap"][1:] for r in records])),
            },
        }
        return result, boot

    def analyze_history_removal(self, routes):
        categories, bootstraps = {}, {}
        for ds in self.DATASETS:
            view = routes[ds]
            if view is None:
                categories[ds] = {
                    "status": "unavailable",
                    "reason": "No explicit view for this dataset route",
                    "counts": None,
                    "users": [],
                    "mean": None,
                    "ci95": None,
                }
                continue
            result = self._audit_route(ds, view)
            if isinstance(result, tuple):
                categories[ds], bootstraps[ds] = result
            else:
                categories[ds] = result
        macro = None
        if len(bootstraps) == len(self.DATASETS):
            means = np.array(
                [
                    [categories[ds]["mean"][key] for key in self.EFFECTS]
                    for ds in self.DATASETS
                ]
            ).mean(axis=0)
            macro_boot = np.stack([bootstraps[ds] for ds in self.DATASETS]).mean(axis=0)
            bounds = np.percentile(macro_boot, [2.5, 97.5], axis=0)
            macro = {
                "mean": dict(zip(self.EFFECTS, means.tolist())),
                "ci95": {
                    key: bounds[:, i].tolist() for i, key in enumerate(self.EFFECTS)
                },
            }
        return {
            "categories": categories,
            "four_category_macro": macro,
            "macro_missing_reason": (
                None if macro is not None else "All four measured routes are required"
            ),
            "seed": 42,
            "control_draws": 5,
            "bootstrap_replicates": 1000,
            "position_preserving": None,
            "auxiliary_generalization_experiment": "not_run",
            "predictive_validity_experiment": "not_run",
        }

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Linear-memory mean cosine with explicit geometry validation
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            if n < 2 or sample_size < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite sampled item embeddings")
            if (torch.linalg.vector_norm(x, dim=-1) == 0).any():
                # Cosine similarity is undefined for zero vectors.
                return None
            x = F.normalize(x, p=2, dim=-1).double()

            # Sum of all pairwise dot products equals the squared norm of
            # the vector sum. Subtract self-products without forming k x k.
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Tensor-compatible sampled margins without unavailable item categories
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches=10,
        eps=0.5,
        topk_categories=20,
    ):
        margins = []
        device = next(self.network.parameters()).device
        if dataloader is not None:
            iterator = iter(dataloader)
            for _ in range(num_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (list, tuple)) or len(batch) < 4:
                    continue
                seq, pos, neg = [
                    torch.as_tensor(x, dtype=torch.long, device=device)
                    for x in batch[1:4]
                ]
                features = self.model.log2feats(seq)
                logits = self.model.predict_score(features, torch.stack([pos, neg], -1))
                margin = logits[..., 0] - logits[..., 1]
                valid = (seq != 0) & (pos != 0) & (neg != 0)
                if valid.any():
                    measured = margin[valid].detach().float().cpu()
                    if not torch.isfinite(measured).all():
                        raise ValueError("Nonfinite sampled diagnostic margins")
                    margins.append(measured)
        result = dict.fromkeys(
            ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate")
        )
        m = torch.cat(margins) if margins else torch.empty(0)
        if m.numel():
            result = {
                "mean": float(m.mean()),
                "p10": float(
                    torch.kthvalue(m, max(1, int(np.ceil(0.1 * m.numel())))).values
                ),
                "neg_beats_pos_rate": float((m < 0).float().mean()),
                "small_margin_rate": float((m < eps).float().mean()),
            }
        return result, {
            "eps": float(eps),
            "valid_positions": int(m.numel()),
            "low_margin_total_positions": int((m < eps).sum()) if m.numel() else None,
            "wrong_rank_total_positions": int((m < 0).sum()) if m.numel() else None,
            "item_category_breakdown": None,
            "reason": "Item categories are unavailable in this task",
        }

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Restore module modes and global RNG state after descriptive diagnostics
    def run_full_diagnosis(self, train_loader):
        routes = self._routes()
        networks = [self.network] + [v.model for v in routes.values() if v is not None]
        saved = {}
        for network in networks:
            for module in network.modules():
                saved.setdefault(module, module.training)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for network in networks:
                network.eval()
            with torch.no_grad():
                collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5
                    )
                )
                audit = self.analyze_history_removal(routes)
        finally:
            # Direct assignment preserves heterogeneous child modes exactly.
            for module, training in saved.items():
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return {
            "metrics": {
                "embedding_collapse_score": collapse,
                "pos_neg_margin_mean": margins["mean"],
                "pos_neg_margin_p10": margins["p10"],
                "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margins["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "history_removal_audit": audit,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "(Mean off-diagonal cosine + 1)/2 for at most 2048 uniformly sampled "
                    "nonpadding item embeddings, seed 42. Random near-orthogonal vectors "
                    "score about 0.5. Directional similarity alone does not establish "
                    "representation collapse, recommendation diversity, or model quality. "
                    "Measured on the supplied primary view, not a four-category macro. "
                    "Null when fewer than two embeddings are sampled or any sampled "
                    "embedding has zero norm; nonfinite embeddings raise an error."
                ),
                "pos_neg_margin_mean": (
                    "Mean positive-minus-sampled-negative logit on valid training positions "
                    "in at most ten supplied batches. Depends on sampler and score scale; "
                    "not a held-out ranking or calibrated confidence measure."
                ),
                "pos_neg_margin_p10": "Nearest-rank lower 10th percentile of the same sampled margins.",
                "pos_neg_neg_beats_pos_rate": "Fraction of sampled valid positions with margin < 0.",
                "pos_neg_small_margin_rate": "Fraction of sampled valid positions with margin < 0.5.",
                "pos_neg_margin_category_breakdown": (
                    "Reports counts and threshold; item-category analysis is null because "
                    "those metadata are unavailable. Empty sampled measurements are null."
                ),
                "history_removal_audit": {
                    "scope": (
                        "Descriptive measurements on final prescribed-training occurrences "
                        "previously observed during fitting. No official split, fitting, "
                        "production inference, or model selection is changed. Full-catalog "
                        "recommendations for the twenty simulator users remain separate."
                    ),
                    "routing": (
                        "All four required dataset names are retained. Explicit dataset_views "
                        "may supply routed compatibility views; otherwise dataset_name identifies "
                        "the supplied view. Missing routes remain null, never pooled or guessed."
                    ),
                    "selection": (
                        "From users sorted by stable ID, sample uniformly without replacement "
                        "at most 200 eligible users per route. SHA256-derived independent RNG "
                        "streams use seed 42, dataset, user and purpose. Eligibility requires "
                        "three events in the frozen preceding context and 99 unseen items. "
                        "Counts distinguish invalid, short, negative-ineligible and unsampled users."
                    ),
                    "intervention": (
                        "Freeze history[:-1][-maxlen:] before perturbation. Delete one occurrence "
                        "by index and left-pad without refilling from older history. Treatment is "
                        "the latest nonterminal occurrence; five independent uniform controls "
                        "are drawn with replacement excluding treatment and terminal indices. "
                        "Compact deletion shifts surviving earlier absolute positions and thus "
                        "confounds content removal with position changes. No frequency matching "
                        "is claimed; draws, repetitions and frequencies are recorded."
                    ),
                    "position_preserving": (
                        "Null: no verified architecture-aware adapter preserving surviving "
                        "positions, blocking deleted keys/values in every layer and branch, "
                        "and retaining the terminal query. Zero embeddings alone do not suffice."
                    ),
                    "candidates": (
                        "Target first, followed by 99 unique uniformly sampled negatives from "
                        "IDs 1..itemnum excluding the entire prescribed training history. "
                        "Candidates are fixed across every intervention for that user. Descending "
                        "stable score order breaks ties by candidate order. Ranks are one-based."
                    ),
                    "record_order": (
                        "ranks, hr10 and top10_items: baseline, recent deletion, five controls. "
                        "removed fields, compact_overlaps and zero_overlap: recent, five controls. "
                        "Indices are zero-based within frozen_window. Training frequencies count "
                        "all occurrences in valid prescribed histories; their distribution maps "
                        "frequency to number of observed items. Window occurrence counts reveal "
                        "repeat imbalance; controls are not matched on popularity or repetition."
                    ),
                    "effects": (
                        "Rank_delta = perturbed target rank minus baseline rank; positive means "
                        "worse target rank. HR10_delta = perturbed hit minus baseline hit; positive "
                        "means improved HR@10. Overlap = intersection size of candidate top-10 sets "
                        "divided by 10; zero_overlap flags disjoint sets. Control quantities average "
                        "five draws within user. Each difference is recent minus mean control, "
                        "with no predetermined desirable sensitivity direction."
                    ),
                    "aggregation": (
                        "Means weight users equally within category; the macro weights all four "
                        "category means equally and is null if any category is unavailable. "
                        "Percentile 95% intervals use 1000 seed-42-derived bootstrap replicates, "
                        "resampling complete paired user vectors within category. Macro replicates "
                        "average independent category replicates. Intervals exclude training-seed "
                        "variation, uncertainty beyond the empirical user bootstrap and adaptive "
                        "selection bias. No multiplicity correction is applied."
                    ),
                    "unperformed_experiments": (
                        "Genuine auxiliary generalization requires predeclared training-only "
                        "suffixes or cutoff, fresh diagnostic models, and reserved targets excluded "
                        "from all fitting supervision and interaction statistics. A global future "
                        "claim requires a common cutoff and vocabulary eligibility reporting. "
                        "Predictive validity additionally requires a frozen gain-prediction rule "
                        "tested on unseen variant families against pinned validation HR@10 changes "
                        "and popularity, last-item, recency-pairwise and rank-margin controls. "
                        "Neither experiment was run; neither is evidence of expected gains."
                    ),
                },
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
