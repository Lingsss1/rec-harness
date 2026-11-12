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
    ### >>> Self_EvolveRec-BLOCK-START: Validate explicit scoring shapes and preserve the trained model scoring path
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.review_data = getattr(view, "review_data", {}) or {}
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Category metadata is explicitly unavailable in this matched task.
        return []

    def _score(self, features, items):
        if features.ndim == 3:
            if items.ndim not in (2, 3) or items.shape[:2] != features.shape[:2]:
                raise ValueError(
                    "Sequence features require aligned B x L or B x L x K item IDs"
                )
            expected = tuple(items.shape)
        elif features.ndim == 2:
            if items.ndim == 1:
                expected = (features.shape[0], items.shape[0])
            elif items.ndim == 2 and items.shape[0] == features.shape[0]:
                expected = tuple(items.shape)
            else:
                raise ValueError(
                    "Final features require shared K or per-user B x K item IDs"
                )
        else:
            raise ValueError("Features must have shape B x d or B x L x d")
        if items.dtype != torch.long or items.device != features.device:
            raise ValueError("Item IDs must be long tensors on the feature device")
        # Prefer architecture-aware scoring when the trained module exposes it;
        # retain the required compatibility-view path otherwise.
        scorer = getattr(self.model, "predict_score", None)
        if not callable(scorer):
            scorer = self.view.predict_score
        scores = scorer(features, items)
        if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected:
            actual = getattr(scores, "shape", type(scores).__name__)
            raise ValueError(f"predict_score returned {actual}, expected {expected}")
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite diagnostic logits")
        return scores

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Measure actual embedding geometry and distinguish missing measurements
            emb = self.view.item_emb.weight.detach()[1:]
            n = emb.size(0)
            self.embedding_sample_count = min(sample_size, n)
            if self.embedding_sample_count < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnosis entry point manage and restore model mode
        margins_all = []
        batches_used = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and reject malformed diagnostic batches
        from itertools import islice

        if num_batches < 1:
            raise ValueError("num_batches must be positive")
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnosis batches must be (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Use tensor encoder inputs and validate the aligned diagnostic contract
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis batches must contain B x L sequence, positive and negative tensors"
                )
            log_feats = self.view.log2feats(seq_t)
            if log_feats.ndim != 3 or log_feats.shape[:2] != seq_t.shape:
                raise ValueError("log2feats must return B x L x d fused features")
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self._score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0) & (pos_t != neg_t)
            batches_used += 1
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Keep margin summaries and breakdown eligibility identical
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
            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable margins as null rather than healthy zero values
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

        ### >>> Self_EvolveRec-BLOCK-START: Include denominators and explicit category unavailability
        breakdown = {
            "category_metadata_available": False,
            "sampled_batches": batches_used,
            "batch_limit": int(num_batches),
            "valid_positions": sum(int(m.numel()) for m in margins_all),
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

    ### >>> Self_EvolveRec-BLOCK-START: Reproducible paired fitted-reliance probes with isolated randomness and explicit limitations
    @torch.no_grad()
    def analyze_fitted_reliance(self, max_users=200, seed=42):
        if not isinstance(max_users, int) or max_users < 1:
            raise ValueError("max_users must be a positive integer")
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError(
                "user_train must provide a dictionary of training histories"
            )
        itemnum = int(self.view.item_emb.weight.shape[0]) - 1
        maxlen = int(getattr(self.model, "maxlen", 128))
        # User selection, candidate draws and interventions must not consume
        # each other's random streams. Candidate sets then remain identical
        # when perturbation repetitions or intervention implementations change.
        selection_rng = np.random.RandomState(seed)
        candidate_rng = np.random.RandomState(seed)
        intervention_rng = np.random.RandomState(seed)
        eligible = []
        popularity = Counter()
        for user, values in histories.items():
            history = [int(i) for i in values]
            if any(i < 1 or i > itemnum for i in history):
                raise ValueError("Training histories contain out-of-catalog items")
            popularity.update(history)
            if len(history) >= 7 and maxlen >= 6:
                eligible.append((user, history))
        eligible.sort(key=lambda pair: str(pair[0]))
        eligible_count = len(eligible)
        if len(eligible) > max_users:
            selected = sorted(
                selection_rng.choice(len(eligible), max_users, replace=False).tolist()
            )
            eligible = [eligible[i] for i in selected]
        conditions = (
            "delete_recent_compact",
            "delete_oldest_compact",
            "delete_random_compact",
            "shuffle_earlier",
        )
        records = []
        insufficient_catalog = 0
        metadata_counts = {"items": 0, "with_title": 0, "with_description": 0}
        if popularity:
            popularity_edges = np.quantile(list(popularity.values()), [1 / 3, 2 / 3])
        else:
            popularity_edges = [0, 0]

        def packed(events):
            result = [0] * maxlen
            events = events[-maxlen:]
            if events:
                result[-len(events) :] = events
            return result

        for _, history in eligible:
            target = history[-1]
            prefix = history[:-1][-maxlen:]
            excluded = set(history)
            available = itemnum - len(excluded)
            if available < 99:
                insufficient_catalog += 1
                continue
            negatives, chosen = [], set()
            if available <= 396 or len(excluded) * 2 > itemnum:
                pool = np.asarray(
                    [i for i in range(1, itemnum + 1) if i not in excluded]
                )
                negatives = candidate_rng.choice(pool, 99, replace=False).tolist()
            else:
                while len(negatives) < 99:
                    value = int(candidate_rng.randint(1, itemnum + 1))
                    if value not in excluded and value not in chosen:
                        negatives.append(value)
                        chosen.add(value)
            candidates = [target] + negatives
            k = min(3, len(prefix) // 2)
            removed = set(
                intervention_rng.choice(len(prefix), k, replace=False).tolist()
            )
            variants = [
                prefix,
                prefix[:-k],
                prefix[k:],
                [value for index, value in enumerate(prefix) if index not in removed],
            ]
            for _ in range(5):
                earlier = intervention_rng.permutation(prefix[:-1]).tolist()
                variants.append(earlier + prefix[-1:])
            sequences = torch.tensor(
                [packed(v) for v in variants], dtype=torch.long, device=self.device
            )
            items = torch.tensor(candidates, dtype=torch.long, device=self.device)
            items = items.unsqueeze(0).expand(len(variants), -1)
            features = self.view.log2feats(sequences)
            scores = self._score(features[:, -1, :], items).float().cpu().numpy()
            # Stable sorting preserves fixed candidate order when logits tie.
            orders = np.argsort(-scores, axis=1, kind="stable")
            ranks = np.argmax(orders == 0, axis=1)
            base_top = set(orders[0, :10].tolist())
            overlaps = np.asarray(
                [len(base_top.intersection(row[:10])) / 10 for row in orders]
            )
            deltas = np.stack(
                (
                    ranks - ranks[0],
                    (ranks < 10).astype(float) - float(ranks[0] < 10),
                    overlaps,
                ),
                axis=1,
            )
            paired = np.stack(
                (deltas[1], deltas[2], deltas[3], deltas[4:].mean(axis=0))
            )
            length_bin = (
                "6-10"
                if len(prefix) <= 10
                else ("11-50" if len(prefix) <= 50 else "51+")
            )
            pop_bin = ("low", "middle", "high")[
                int(np.searchsorted(popularity_edges, popularity[target], side="left"))
            ]
            records.append((paired, length_bin, pop_bin))
            for item in set(history):
                info = self.meta_data.get(str(item), self.meta_data.get(item, {}))
                info = info if isinstance(info, dict) else {}
                metadata_counts["items"] += 1
                metadata_counts["with_title"] += int(bool(info.get("title")))
                metadata_counts["with_description"] += int(
                    bool(info.get("description"))
                )

        def summarize(subset, bootstrap_seed):
            if not subset:
                return {"users": 0, "conditions": None}
            values = np.stack([record[0] for record in subset])
            means = values.mean(axis=0)
            boot_rng = np.random.RandomState(bootstrap_seed)
            # Bound temporary memory while retaining 2,000 paired user replicates.
            bootstrap = []
            for _ in range(20):
                indices = boot_rng.randint(0, len(values), size=(100, len(values)))
                bootstrap.append(values[indices].mean(axis=1))
            lower, upper = np.percentile(np.concatenate(bootstrap), [2.5, 97.5], axis=0)
            names = ("target_rank_delta", "hr10_delta", "top10_overlap")
            return {
                "users": len(values),
                "conditions": {
                    condition: {
                        name: {
                            "mean": float(means[i, j]),
                            "ci95": [float(lower[i, j]), float(upper[i, j])],
                        }
                        for j, name in enumerate(names)
                    }
                    for i, condition in enumerate(conditions)
                },
            }

        return {
            "dataset": getattr(self.view, "dataset_name", None),
            "seed": seed,
            "eligible_users": eligible_count,
            "sampled_users": len(eligible),
            "user_limit": max_users,
            "insufficient_negative_catalog_users": insufficient_catalog,
            "negative_count": 99,
            "shuffle_repetitions": 5,
            "bootstrap_replicates": 2000,
            "overall": summarize(records, seed),
            "by_prefix_length": {
                key: summarize([r for r in records if r[1] == key], seed)
                for key in ("6-10", "11-50", "51+")
            },
            "by_target_training_popularity": {
                key: summarize([r for r in records if r[2] == key], seed)
                for key in ("low", "middle", "high")
            },
            "training_popularity_tertile_edges": [float(x) for x in popularity_edges],
            "metadata_user_item_pairs": metadata_counts,
            "position_preserving_removal": None,
            "position_preserving_removal_available": False,
            "position_preserving_removal_reason": "The compatibility interface does not expose verified query and attention key/value masks.",
            "four_category_macro": None,
        }

    def run_full_diagnosis(self, train_loader):
        # Loader sampling may use global RNGs. Diagnosis must not change the
        # subsequent training minibatches, negatives or dropout randomness.
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        prior_modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            with torch.no_grad():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=eps, topk_categories=20
                    )
                )
                reliance = self.analyze_fitted_reliance(max_users=200, seed=42)
                gate_parameter = getattr(self.model, "recent_gate", None)
                fitted_gain = (
                    float(gate_parameter.detach().sigmoid().cpu())
                    if isinstance(gate_parameter, torch.Tensor)
                    and gate_parameter.numel() == 1
                    else None
                )
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "embedding_sample_count": self.embedding_sample_count,
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "fitted_recent_gate": fitted_gain,
                        "fitted_reliance": reliance,
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": (
                            "(Mean off-diagonal cosine + 1)/2 among up to 2048 nonpadding item embeddings, sampled with seed 42. "
                            "Values near 0.5 correspond to mean cosine near zero; higher values indicate greater directional alignment. "
                            "This is fitted embedding geometry, not evidence about user diversity, preference or recommendation quality. Null if fewer than two items."
                        ),
                        "embedding_sample_count": "Number of nonpadding embeddings sampled for the cosine statistic.",
                        "pos_neg_margin_mean": "Mean fitted positive-minus-negative logit over nonpadding, distinct positive/negative pairs; null without valid pairs.",
                        "pos_neg_margin_p10": "Nearest-rank 10th percentile of the same fitted margins; null without valid pairs.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of valid pairs with margin below zero; null without valid pairs.",
                        "pos_neg_small_margin_rate": "Fraction of valid pairs with margin below 0.5; logit-scale dependent and null without valid pairs.",
                        "pos_neg_margin_category_breakdown": (
                            "Reports valid-position and batch counts and thresholds. Category lists remain empty because category metadata is unavailable."
                        ),
                        "fitted_recent_gate": (
                            "Sigmoid of an exposed scalar recent_gate parameter, otherwise null. A fitted coefficient is not a causal reliance estimate or validation improvement."
                        ),
                        "fitted_reliance": (
                            "Training-only fitted-model probe, separate from host validation: each sampled user's final training event is the target, "
                            "and only preceding events enter the input. The target still participated in fitting. Users need at least six observed prefix events. "
                            "The effective prefix is truncated to model.maxlen. Seed 42 selects up to 200 users and 99 unique negatives per user, excluding "
                            "the target and original complete training history. Fixed candidate order breaks score ties identically across interventions. "
                            "Delete k=min(3,floor(prefix_length/2)) recent events including the final event, oldest events, or uniformly selected events; "
                            "all deletions compact survivors through ordinary left padding and thus change positions and adjacency. "
                            "Five earlier-event permutations preserve the last event, multiset and length and are averaged within user. "
                            "Rank delta is intervention minus baseline (positive means worse); HR@10 delta is intervention minus baseline; "
                            "top10_overlap is intersection size divided by 10. Within this dataset, 2000 seed-42 paired user bootstrap replicates yield "
                            "95% percentile intervals after averaging shuffle repetitions. Prefix-length and training-frequency tertile groups are descriptive. "
                            "Intervals condition on the fitted model and fixed candidates; they exclude training-seed variability and do not remove target "
                            "contamination or validation-selection optimism. Four-category macro results require paired aggregation across all four views "
                            "and are unavailable here. Position-preserving removal is unavailable without verified encoder masks. Metadata availability "
                            "counts sampled user-item pairs from training histories; titles/descriptions alone establish neither compatibility nor need. "
                            "No ratings, review text, category metadata, prices, stores or simulator traits are used. These probes do not estimate held-out performance."
                        ),
                    },
                }
        finally:
            for module, training in prior_modes:
                module.training = training
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    # Review checks: scoring dimensions, actual-module dispatch, loader bounds,
    # malformed-batch errors, fixed candidate randomness and mode/RNG restoration.
    # Reviewed statically; no training, timing or benchmark improvement measured.
    # RNG restoration covers this process; persistent loader-worker RNGs require
    # a separately seeded diagnostic loader supplied by the caller.
    # These changes repair diagnostics only. The supplied training model still
    # uses window pooling and optional candidate CE; it does not implement the
    # proposed normalized geometric residual with unchanged baseline BCE.

    ### <<< Self_EvolveRec-BLOCK-END
