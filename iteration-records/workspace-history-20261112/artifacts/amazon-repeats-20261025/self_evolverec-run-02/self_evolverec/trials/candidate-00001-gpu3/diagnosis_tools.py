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
    ### >>> Self_EvolveRec-BLOCK-START: Bind diagnostics to the actual model and respect unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._replay_cache = None

    def _get_categories(self, item_id: int):
        # Item categories are not exposed by the matched benchmark adapter.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Do not fabricate embedding alignment for an insufficient catalog
            if n < 2 or sample_size < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute exact sampled alignment without a quadratic Gram matrix
            x = x.float()
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite sampled item embeddings")
            # Sum of all pairwise dot products equals the squared norm of the sum.
            # Subtract measured diagonal values, including any zero embeddings.
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
        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption while the entry point manages module mode
        from itertools import islice

        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Reject malformed diagnostic batches without consuming an extra batch
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise TypeError("Expected (None, sequence, positive, negative)")

            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate sequence and supervision shapes before scoring
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            if seq_t.ndim != 2 or seq_t.shape[1] == 0:
                raise ValueError(
                    "Diagnostic sequences must have shape B x L with L > 0"
                )
            if tuple(np.shape(pos)) != tuple(seq_t.shape) or tuple(
                np.shape(neg)
            ) != tuple(seq_t.shape):
                raise ValueError("Sequence, positive and negative shapes must match")
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Measure only finite scores at valid supervised positions
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            finite = torch.isfinite(margin)
            if (valid & ~finite).any():
                raise FloatingPointError(
                    "Nonfinite positive/negative diagnostic margin"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Keep diagnostic counts aligned with measured positions
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
            ### >>> Self_EvolveRec-BLOCK-START: Represent unmeasured margins explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Report sampled position counts without unavailable category evidence
        breakdown = {
            "eps": float(eps),
            "valid_positions": int(sum(m.numel() for m in margins_all)),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": None,
            "wrong_rank_top_categories": None,
            "category_metadata_available": False,
            "category_missing_reason": "Item categories are unavailable in this benchmark",
            "missing_margin_reason": (
                None if margins_all else "No valid supervised positions were measured"
            ),
        }
        ### <<< Self_EvolveRec-BLOCK-END

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Frozen final-prefix replay and paired recency attribution with explicit limitations
    def _freeze_replay(self):
        if self._replay_cache is not None:
            return self._replay_cache
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError("user_train must contain a user-to-history dictionary")
        backbone = getattr(self.module, "backbone", self.module)
        maxlen_value = getattr(backbone, "maxlen", None)
        if maxlen_value is None:
            raise AttributeError("Replay requires the trained backbone's actual maxlen")
        maxlen = int(maxlen_value)
        itemnum = int(self.model.item_emb.num_embeddings - 1)
        if maxlen < 1 or itemnum < 1:
            raise ValueError("Replay requires a positive maxlen and nonempty catalog")
        users = sorted(histories, key=lambda uid: str(uid))
        eligible, skipped = [], 0
        for uid in users:
            history = [int(item) for item in histories[uid]]
            if any(item < 1 or item > itemnum for item in history):
                raise ValueError(
                    "Training history contains an item outside the catalog"
                )
            if len(history) < 2:
                skipped += 1
                continue
            eligible.append((uid, history))
        selection_rng = random.Random(42)
        if len(eligible) > 50:
            selected_indices = sorted(selection_rng.sample(range(len(eligible)), 50))
            selected = [eligible[index] for index in selected_indices]
        else:
            selected = eligible
        examples = []
        for index, (uid, history) in enumerate(selected):
            target = history[-1]
            # Sampling a compact integer range avoids catalog-sized allocations.
            candidate_rng = random.Random(42 + index)
            negatives = candidate_rng.sample(range(1, itemnum), min(99, itemnum - 1))
            negatives = [item + (item >= target) for item in negatives]
            prefix = history[:-1][-maxlen:]
            examples.append(
                {
                    "user_id": str(uid),
                    "prefix": prefix,
                    "target": target,
                    "repeat_target": target in history[:-1],
                    "candidates": [target] + negatives,
                    "maxlen": maxlen,
                }
            )
        self._replay_cache = {
            "examples": examples,
            "eligible_users": len(eligible),
            "skipped_without_prefix": skipped,
            "scope": str(getattr(self.model, "dataset_name", "current_dataset_view")),
        }
        return self._replay_cache

    @torch.no_grad()
    def _measure_replay(self, example, prefix):
        maxlen = example["maxlen"]
        padded = [0] * (maxlen - len(prefix)) + list(prefix)
        sequence = torch.tensor([padded], device=self.device, dtype=torch.long)
        features = self.model.log2feats(sequence)
        candidates = torch.tensor(
            example["candidates"], device=self.device, dtype=torch.long
        )
        # Preserve complete feature tensors: some adapters attach sequence context
        # that would be lost by slicing before candidate scoring.
        items = candidates.view(1, 1, -1).expand(1, maxlen, -1)
        scores = self.model.predict_score(features, items)[0, -1]
        if not torch.isfinite(scores).all() or not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite replay features or candidate scores")
        scores = scores.detach().float().cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == 0)[0])
        return {
            "rank": rank,
            "hit": int(rank < 10),
            "top10": set(order[:10].tolist()),
            "feature": features[0, -1].detach().float().cpu(),
        }

    @staticmethod
    def _paired_replay(original, changed):
        cosine = F.cosine_similarity(
            original["feature"].unsqueeze(0), changed["feature"].unsqueeze(0), dim=-1
        )
        undefined_cosine = (
            original["feature"].norm().item() == 0.0
            or changed["feature"].norm().item() == 0.0
        )
        return {
            "rank_change": changed["rank"] - original["rank"],
            "hit10_change": changed["hit"] - original["hit"],
            "top10_overlap": len(original["top10"] & changed["top10"])
            / max(len(original["top10"]), 1),
            "feature_cosine": None if undefined_cosine else float(cosine.item()),
        }

    @torch.no_grad()
    def analyze_final_prefix_replay(self):
        frozen = self._freeze_replay()
        records = []
        skipped_deletion = 0
        for index, example in enumerate(frozen["examples"]):
            prefix = example["prefix"]
            original = self._measure_replay(example, prefix)
            n = len(prefix)
            k = min(3, (n - 1) // 2)
            record = {
                "user_id": example["user_id"],
                "history_length": n,
                "length_stratum": "1-5" if n <= 5 else "6-20" if n <= 20 else "21+",
                "repeat_target": example["repeat_target"],
                "target": example["target"],
                "prefix": prefix,
                "candidate_ids": example["candidates"],
                "original_rank_zero_based": original["rank"],
                "original_hit10": original["hit"],
                "original_feature_norm": float(original["feature"].norm().item()),
                "deletion_count": k,
                "perturbations": [],
            }
            if k == 0:
                skipped_deletion += 1
            recent_change = None
            if k:
                recent_positions = set(range(n - 1 - k, n - 1))
                recent_prefix = [
                    item for p, item in enumerate(prefix) if p not in recent_positions
                ]
                recent_change = self._paired_replay(
                    original, self._measure_replay(example, recent_prefix)
                )
            for seed in (42, 43, 44):
                rng = random.Random(seed * 1000003 + index)
                comparison = {"seed": seed}
                if k:
                    positions = set(rng.sample(range(n - 1), k))
                    control = [
                        item for p, item in enumerate(prefix) if p not in positions
                    ]
                    random_change = self._paired_replay(
                        original, self._measure_replay(example, control)
                    )
                    comparison["compact_recent_deletion"] = recent_change
                    comparison["compact_random_deletion"] = random_change
                    comparison["recent_minus_random_rank_change"] = (
                        recent_change["rank_change"] - random_change["rank_change"]
                    )
                    comparison["recent_minus_random_hit10_change"] = (
                        recent_change["hit10_change"] - random_change["hit10_change"]
                    )
                else:
                    comparison["compact_recent_deletion"] = None
                    comparison["compact_random_deletion"] = None
                    comparison["deletion_missing_reason"] = "k=0 for this prefix length"
                shuffled = list(prefix[:-1])
                random.Random(seed * 1000033 + index).shuffle(shuffled)
                shuffled.append(prefix[-1])
                comparison["last_item_preserving_shuffle"] = self._paired_replay(
                    original, self._measure_replay(example, shuffled)
                )
                record["perturbations"].append(comparison)
            records.append(record)

        def aggregate(group):
            result = {
                "users": len(group),
                "diagnostic_HR@10": (
                    float(np.mean([r["original_hit10"] for r in group]))
                    if group
                    else None
                ),
                "mean_rank_zero_based": (
                    float(np.mean([r["original_rank_zero_based"] for r in group]))
                    if group
                    else None
                ),
                "paired_perturbations": {},
            }
            measures = (
                "rank_change",
                "hit10_change",
                "top10_overlap",
                "feature_cosine",
            )
            for intervention in (
                "compact_recent_deletion",
                "compact_random_deletion",
                "last_item_preserving_shuffle",
            ):
                summary = {}
                for measure in measures:
                    # Average seeds within each user before averaging users.
                    user_means = []
                    comparisons = 0
                    for record in group:
                        values = []
                        for perturbation in record["perturbations"]:
                            changed = perturbation.get(intervention)
                            if changed is not None and changed.get(measure) is not None:
                                values.append(changed[measure])
                        if values:
                            user_means.append(float(np.mean(values)))
                            comparisons += len(values)
                    summary[measure] = {
                        "mean": float(np.mean(user_means)) if user_means else None,
                        "users": len(user_means),
                        "comparisons": comparisons,
                        "missing_users": len(group) - len(user_means),
                        "missing_reason": (
                            "Deletion unavailable when k=0; cosine undefined for zero vectors; "
                            "or no eligible users in this stratum"
                            if len(user_means) < len(group) or not group
                            else None
                        ),
                    }
                result["paired_perturbations"][intervention] = summary
            for measure in (
                "rank_change",
                "hit10_change",
                "top10_overlap",
                "feature_cosine",
            ):
                user_means = []
                for record in group:
                    differences = []
                    for perturbation in record["perturbations"]:
                        recent = perturbation.get("compact_recent_deletion")
                        control = perturbation.get("compact_random_deletion")
                        if recent is None or control is None:
                            continue
                        if recent.get(measure) is None or control.get(measure) is None:
                            continue
                        differences.append(recent[measure] - control[measure])
                    if differences:
                        user_means.append(float(np.mean(differences)))
                result["paired_perturbations"]["recent_minus_random_" + measure] = {
                    "mean": float(np.mean(user_means)) if user_means else None,
                    "users": len(user_means),
                    "missing_users": len(group) - len(user_means),
                    "missing_reason": (
                        "No eligible paired deletions or undefined feature cosine"
                        if not user_means
                        else None
                    ),
                }
            return result

        return {
            "scope": frozen["scope"],
            "eligible_users": frozen["eligible_users"],
            "skipped_without_prefix": frozen["skipped_without_prefix"],
            "skipped_deletion_users": skipped_deletion,
            "summary": aggregate(records),
            "repeat_target": aggregate([r for r in records if r["repeat_target"]]),
            "novel_target": aggregate([r for r in records if not r["repeat_target"]]),
            "by_prefix_length": {
                label: aggregate([r for r in records if r["length_stratum"] == label])
                for label in ("1-5", "6-20", "21+")
            },
            "records": records,
            "sampling_policy": (
                "Up to 50 eligible users in this dataset view, sorted by string user ID, "
                "subsampled using random.Random(42). Withhold the final training item; "
                "retain the preceding maxlen items. Target first, then up to 99 unique "
                "uniform non-target catalog IDs without replacement using seed 42+user_index; "
                "prefix items remain eligible negatives. Stable descending rank preserves "
                "candidate order on ties. Candidates and examples are cached on this probe."
            ),
            "in_place_masking": {
                "value": None,
                "reason": (
                    "The supplied baseline attention has no padding-key mask. Scoring internal "
                    "holes as ordinary zero tokens would not implement the requested intervention. "
                    "This audit requires an architecture-level safe key-padding-mask implementation."
                ),
                "unmeasured_users": len(records),
            },
            "official_validation": {
                "value": None,
                "reason": "Official per-user validation hits are not exposed to this probe",
                "available_users": 0,
            },
            "limitations": (
                "Training-prefix replay targets were used in fitting; these are descriptive "
                "audits, not validation or generalization scores. Compact deletion changes "
                "positions as well as context. Perturbations do not identify intent or causal "
                "effects. Seeds are repeated interventions on the same users, not independent users. "
                "No simulator traits, reviews, ratings, prices, stores or item categories are observed."
            ),
        }

    def run_full_diagnosis(self, train_loader):
        prior_modes = [(module, module.training) for module in self.module.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_state = None
        if self.device.type == "cuda":
            cuda_state = torch.cuda.get_rng_state(self.device)
        try:
            self.module.eval()
            with torch.no_grad():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
                replay = self.analyze_final_prefix_replay()
            return {
                "metrics": {
                    "embedding_collapse_score": collapse_score,
                    "embedding_alignment_missing_reason": (
                        "Fewer than two catalog embeddings"
                        if collapse_score is None
                        else None
                    ),
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "final_prefix_replay": replay,
                    "implementation_audit": {
                        "trained_module": type(self.module).__name__,
                        "recency_residual_verified": None,
                        "reason": (
                            "The supplied training code fits SASRec and deploys FrozenTransition; "
                            "it does not implement the proposed learned recency-pooling residual. "
                            "Diagnosis changes alone cannot add or validate that trained branch."
                        ),
                        "official_macro_selection_verified": None,
                        "selection_reason": (
                            "The supplied trainer selects category checkpoints independently and "
                            "selects transition fusion using local validation. This probe cannot "
                            "verify a common-epoch selection using official host scores."
                        ),
                        "checks": [
                            "Reviewed tensor shapes, finite margins, loader consumption and replay bounds",
                            "Preserved frozen replay candidates, stable ranking, mode and RNG restoration",
                            "Reduced embedding alignment memory and added user-weighted paired summaries",
                            "No execution performed; safe in-place masking and recency training remain unimplemented",
                        ],
                    },
                },
                "metric_definitions": {
                    "embedding_collapse_score": (
                        "Mean off-diagonal cosine among up to 2048 uniformly sampled item "
                        "embeddings, seed 42, mapped by (cosine+1)/2. Approximately orthogonal "
                        "embeddings score 0.5. Directional alignment alone does not establish "
                        "representation collapse, recommendation diversity or model quality."
                    ),
                    "pos_neg_margin_mean": (
                        "Mean positive-minus-sampled-negative logit at valid positions in up to "
                        "10 loader batches. Depends on logit scale and the loader sampling policy; "
                        "training transitions are descriptive, not held-out evidence."
                    ),
                    "pos_neg_margin_p10": "Empirical nearest-rank tenth percentile of the measured margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of measured margins strictly below zero.",
                    "pos_neg_small_margin_rate": "Fraction of measured margins below the scale-dependent threshold 0.5.",
                    "pos_neg_margin_category_breakdown": (
                        "Valid, low-margin and wrong-rank position counts. Category evidence is "
                        "unavailable; empty measurements are null rather than zero."
                    ),
                    "implementation_audit": (
                        "Static compatibility findings for the supplied training code; null means "
                        "unverified or unavailable, never a measured model-quality result. "
                        "No learned-gate benefit or official validation improvement is established."
                    ),
                    "final_prefix_replay": (
                        "Frozen auxiliary training-prefix audit with explicit candidates and paired "
                        "per-user perturbations. Positive rank change means worsening; positive "
                        "hit10 change means improvement. Top10 overlap is shared candidate indices "
                        "divided by the original top-list length. Feature cosine is null for zero "
                        "vectors. Repeat status refers to the full preceding training history. "
                        "Each probe covers its current dataset view; records expose length/repeat "
                        "strata for aggregation without treating intervention seeds as users. "
                        "This does not replace host validation or the required twenty-user simulator."
                    ),
                },
            }
        finally:
            # Restore mixed submodule modes too, including frozen eval backbones.
            for module, training in prior_modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, self.device)

    ### <<< Self_EvolveRec-BLOCK-END
