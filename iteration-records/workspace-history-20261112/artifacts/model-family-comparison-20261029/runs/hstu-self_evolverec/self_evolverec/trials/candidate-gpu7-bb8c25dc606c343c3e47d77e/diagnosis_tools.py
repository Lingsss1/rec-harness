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
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Do not fabricate an empty cosine measurement
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
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
            # DEBUG: Score through the actual trained lexical HSTU. A host
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

    ### >>> Self_EvolveRec-BLOCK-START: Controlled in-sample HSTU history sensitivity with paired uncertainty
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
                    "paired_user_bootstrap_95pct": [
                        float(np.percentile(boot[:, i], 2.5)),
                        float(np.percentile(boot[:, i], 97.5)),
                    ],
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
        selected = rng.sample(eligible, min(100, len(eligible)))
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

    def run_full_diagnosis(self, train_loader):
        import os
        import re
        import time

        modes = [(module, module.training) for module in self.network.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = (
            [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if self.device.type == "cuda"
            else []
        )
        started = time.perf_counter()
        cuda_start = cuda_end = None
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                self.network.eval()
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                        cuda_start.record()
                with torch.no_grad():
                    collapse_score = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048
                    )
                    margins, counts = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            train_loader, num_batches=10, eps=0.5, topk_categories=0
                        )
                    )
                    temporal = self._temporal_probe()
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
                    "max_users": 100,
                    "minimum_events": 5,
                    "perturbation_seeds": [42, 43, 44, 45, 46],
                    "bootstrap_seed": 42,
                    "bootstrap_replicates": 10000,
                    "candidate_count": 100,
                    "legacy_margin_max_batches": 10,
                    "identity_no_op_tolerance": {"rtol": 1e-5, "atol": 1e-6},
                    "history_policy": "Truncate prefix once; perturb without backfill; compact surviving events; left pad; greatest nonzero position readout.",
                    "training_changes": False,
                }
                metrics = {
                    "configuration": configuration,
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
                    "embedding_collapse_score": "Legacy (mean off-diagonal cosine + 1)/2 over at most 2048 sampled ID embeddings excluding padding/reserved tokens. Null with fewer than two items. Not effective rank, catalog diversity or relevance.",
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
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    ### <<< Self_EvolveRec-BLOCK-END
