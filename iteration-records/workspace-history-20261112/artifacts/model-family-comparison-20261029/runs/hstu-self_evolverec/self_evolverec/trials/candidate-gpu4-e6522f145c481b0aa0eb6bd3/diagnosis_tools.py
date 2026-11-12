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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and identify the actual HSTU module
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._recent_history_cache = []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Keep unavailable item-category evidence empty
    def _get_categories(self, item_id: int):
        # Item-category metadata is unavailable in this matched benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Report insufficient embedding evidence explicitly
            # DEBUG: Diagnose the shared fused item representation actually used
            # by HSTU inputs and scoring, rather than only its raw ID component.
            n = int(self.module.itemnum)
            if sample_size < 2:
                raise ValueError("Embedding collapse requires sample_size >= 2")
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            embedding_device = self.module.item_emb.weight.device
            g = torch.Generator(device=embedding_device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            ids = torch.randperm(n, generator=g, device=embedding_device)[:k] + 1
            x = self.module.encode_items(ids).detach()
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite fused item embeddings in diagnosis")
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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained module
        self.module.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            # DEBUG: Validate the host's B x L contract before forward, stacking,
            # or broadcasting can fail with an unrelated tensor error.
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative tensors must have shape B x L"
                )
            log_feats = self.module.log2feats(seq_t)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            ### >>> Self_EvolveRec-BLOCK-START: Measure margins with the trained HSTU scoring function
            all_logits = self.module.predict_score(log_feats, all_items)
            ### <<< Self_EvolveRec-BLOCK-END
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Validate measured margins before aggregation
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative tensors must have shape B x L"
                )
            valid = pos_t.ne(0) & seq_t.ne(0) & neg_t.ne(0)
            if valid.any():
                measured = margin[valid].detach().float()
                if not torch.isfinite(measured).all():
                    raise ValueError("Nonfinite positive-negative diagnostic margins")
                margins_all.append(measured.cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical validity masks for metrics and breakdowns
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

        ### >>> Self_EvolveRec-BLOCK-START: Preserve missing margin measurements as null
        if len(margins_all) == 0:
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

        ### >>> Self_EvolveRec-BLOCK-START: Expose margin sample size and missingness
        measured_positions = sum(values.numel() for values in margins_all)
        breakdown = {
            "measured_positions": int(measured_positions),
            "missing_reason": (
                None
                if measured_positions
                else "No valid sampled margin positions were supplied"
            ),
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

    ### >>> Self_EvolveRec-BLOCK-START: Validated frozen-HSTU sensitivity with paired control contrasts and bounded bootstrap memory
    @staticmethod
    def _ranking_statistics(scores, candidates, baseline_top=None):
        scores = np.asarray(scores, dtype=np.float64)
        candidates = np.asarray(candidates, dtype=np.int64)
        if scores.shape != (100,) or candidates.shape != (100,):
            raise ValueError(
                "Diagnostic cases require one target and exactly 99 negatives"
            )
        if len(np.unique(candidates)) != 100 or np.any(candidates < 1):
            raise ValueError("Diagnostic candidates must be unique positive item IDs")
        if baseline_top is not None:
            if len(baseline_top) != 10 or len(set(baseline_top)) != 10:
                raise ValueError("Reference top-10 must contain ten unique item IDs")
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite diagnostic scores")
        # Candidate zero is the target. Item IDs provide deterministic ties.
        order = np.lexsort((candidates, -scores))
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        optimistic = 1 + int(np.sum(scores[1:] > scores[0]))
        pessimistic = 1 + int(np.sum(scores[1:] >= scores[0]))
        top = candidates[order[:10]].tolist()
        reference = top if baseline_top is None else baseline_top
        overlap = len(set(top) & set(reference)) / 10.0
        p = 0.9
        frbo = (
            (1 - p)
            / (1 - p**10)
            * sum(
                p ** (depth - 1)
                * len(set(top[:depth]) & set(reference[:depth]))
                / depth
                for depth in range(1, 11)
            )
        )
        return {
            "rank": float(rank),
            "rr": 1.0 / rank,
            "hr10": float(rank <= 10),
            "rank_optimistic": float(optimistic),
            "rank_pessimistic": float(pessimistic),
            "rr_lower": 1.0 / pessimistic,
            "rr_upper": 1.0 / optimistic,
            "hr10_lower": float(pessimistic <= 10),
            "hr10_upper": float(optimistic <= 10),
            "tied_target": float(pessimistic > optimistic),
            "overlap10": overlap,
            "frbo10": float(frbo),
        }, top

    @staticmethod
    def _stratum(length):
        if length < 10:
            return "4-9"
        if length < 20:
            return "10-19"
        if length < 50:
            return "20-49"
        return "50+"

    def _cache_training_panel(self):
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError("user_train must contain a history mapping")
        width = int(self.module.maxlen)
        itemnum = int(self.module.itemnum)
        rng = np.random.default_rng(42)
        exclusions = Counter()
        eligible = []
        for user in sorted(histories, key=str):
            history = [int(item) for item in histories[user]]
            if any(item < 1 or item > itemnum for item in history):
                exclusions["invalid_item_id"] += 1
                continue
            if len(history) < 5 or min(width, len(history) - 1) < 4:
                exclusions["fewer_than_four_visible_prefix_events"] += 1
                continue
            # Exclude observations available at this prediction point, and
            # explicitly exclude the target from negative candidates.
            excluded = set(history[:-1]) | {history[-1]}
            if itemnum - len(excluded) < 99:
                exclusions["fewer_than_99_eligible_negatives"] += 1
                continue
            eligible.append((user, history))
        selected = rng.permutation(len(eligible))[:200]
        panel = []
        for index in selected:
            user, history = eligible[int(index)]
            seed = int(rng.integers(0, 2**32 - 1))
            local = np.random.default_rng(seed)
            excluded = set(history[:-1]) | {history[-1]}
            if len(excluded) > itemnum // 2:
                pool = np.asarray(
                    [i for i in range(1, itemnum + 1) if i not in excluded],
                    dtype=np.int64,
                )
                negatives = local.choice(pool, 99, replace=False).tolist()
            else:
                negatives, seen = [], set(excluded)
                while len(negatives) < 99:
                    for item in local.integers(1, itemnum + 1, size=256):
                        item = int(item)
                        if item not in seen:
                            seen.add(item)
                            negatives.append(item)
                            if len(negatives) == 99:
                                break
            panel.append(
                {
                    "user": str(user),
                    "target": history[-1],
                    "history": history[:-1][-width:],
                    "original_prefix_length": len(history) - 1,
                    "truncated": len(history) - 1 > width,
                    "candidates": [history[-1]] + negatives,
                    "seed": seed,
                }
            )
        self._recent_history_cache = panel
        return panel, {
            "total_users": len(histories),
            "eligible_users_before_cap": len(eligible),
            "selected_users": len(panel),
            "eligible_not_selected": max(0, len(eligible) - len(panel)),
            "exclusions": dict(exclusions),
            "truncated_selected_users": sum(case["truncated"] for case in panel),
        }

    @staticmethod
    def _variants(case):
        history = case["history"]
        n = len(history)
        k = min(max(1, n // 4), n - 2)
        rng = np.random.default_rng(case["seed"])
        variants = [("intact", history), ("identity", history.copy())]
        specifications = []

        def delete(name, positions):
            positions = sorted(int(position) for position in positions)
            removed = set(positions)
            variants.append(
                (name, [item for j, item in enumerate(history) if j not in removed])
            )
            specifications.append({"condition": name, "deleted_positions": positions})

        delete("recent_delete", range(n - k, n))
        delete("oldest_delete", range(k))
        delete("prefinal_delete", range(n - k - 1, n - 1))
        for _ in range(5):
            delete("random_delete", rng.choice(n, k, replace=False))
            delete("random_nonfinal_delete", rng.choice(n - 1, k, replace=False))
            prefix_order = rng.permutation(n - 1).tolist() + [n - 1]
            full_order = rng.permutation(n).tolist()
            for name, order in (
                ("prefix_shuffle_keep_last", prefix_order),
                ("full_shuffle", full_order),
            ):
                variants.append((name, [history[j] for j in order]))
                specifications.append({"condition": name, "permutation": order})
        case["interventions"] = specifications
        case["k"] = k
        return variants

    @torch.no_grad()
    def _recent_history_diagnosis(self):
        panel, counts = self._cache_training_panel()
        rows = []
        width = int(self.module.maxlen)
        for case in panel:
            variants = self._variants(case)
            sequences = torch.zeros(
                (len(variants), width), dtype=torch.long, device=self.device
            )
            for index, (_, history) in enumerate(variants):
                sequences[index, -len(history) :] = torch.as_tensor(
                    history, dtype=torch.long, device=self.device
                )
            features = self.module.log2feats(sequences)
            positions = torch.arange(width, device=self.device).expand_as(sequences)
            last = positions.masked_fill(sequences.eq(0), -1).amax(dim=1)
            selected = features[torch.arange(len(variants), device=self.device), last]
            candidates = torch.tensor(
                case["candidates"], dtype=torch.long, device=self.device
            )
            # Expand candidates explicitly to avoid ambiguous broadcast scoring.
            scores = self.module.predict_score(
                selected, candidates.unsqueeze(0).expand(len(variants), -1)
            )
            torch.testing.assert_close(scores[0], scores[1], rtol=1e-5, atol=1e-6)
            scores = scores.detach().float().cpu().numpy()
            baseline, top = self._ranking_statistics(scores[0], case["candidates"])
            grouped = {}
            for index, (condition, _) in enumerate(variants):
                measured, _ = self._ranking_statistics(
                    scores[index], case["candidates"], top
                )
                if condition == "identity" and measured != baseline:
                    raise AssertionError(
                        "Identity intervention changed diagnostic ranks"
                    )
                values = dict(measured)
                for key in ("rank", "rr", "hr10"):
                    values["delta_" + key] = measured[key] - baseline[key]
                grouped.setdefault(condition, []).append(values)
            # Random repetitions are averaged before user aggregation.
            averaged = {
                condition: {
                    key: float(np.mean([value[key] for value in repeats]))
                    for key in repeats[0]
                }
                for condition, repeats in grouped.items()
            }
            # Compute contrasts before bootstrapping so each interval retains
            # the covariance between interventions on the same user.
            for name, treatment, control in (
                ("recent_vs_random", "recent_delete", "random_delete"),
                ("recent_vs_oldest", "recent_delete", "oldest_delete"),
                (
                    "prefinal_vs_random_nonfinal",
                    "prefinal_delete",
                    "random_nonfinal_delete",
                ),
                ("prefinal_vs_oldest", "prefinal_delete", "oldest_delete"),
            ):
                averaged[name] = {
                    "contrast_" + key: averaged[treatment][key] - averaged[control][key]
                    for key in ("rank", "rr", "hr10")
                }
            case["measurements"] = averaged
            rows.append(
                {
                    "user": case["user"],
                    "stratum": self._stratum(len(case["history"])),
                    "conditions": averaged,
                }
            )

        # Verify the list similarity definitions independently of model output.
        check, _ = self._ranking_statistics(np.arange(100), np.arange(1, 101))
        disjoint, _ = self._ranking_statistics(
            np.arange(100), np.arange(1, 101), list(range(101, 111))
        )
        if not (
            np.isclose(check["frbo10"], 1.0)
            and check["overlap10"] == 1.0
            and disjoint["frbo10"] == 0.0
            and disjoint["overlap10"] == 0.0
        ):
            raise AssertionError("List similarity self-check failed")

        def summarize(subset, seed):
            if not subset:
                return {"users": 0, "examples": 0, "conditions": {}}
            local = np.random.default_rng(seed)
            # Exactly one example/user here; retain every paired condition.
            indices = local.integers(0, len(subset), size=(2000, len(subset)))
            # Multiplicity weights are exactly equivalent to indexing sampled
            # users, without materializing a replicates x users x metrics array.
            weights = np.zeros((2000, len(subset)), dtype=np.float64)
            np.add.at(weights, (np.arange(2000)[:, None], indices), 1.0)
            weights /= len(subset)
            del indices
            conditions = {}
            for condition in subset[0]["conditions"]:
                keys = list(subset[0]["conditions"][condition])
                matrix = np.asarray(
                    [
                        [row["conditions"][condition][key] for key in keys]
                        for row in subset
                    ],
                    dtype=np.float64,
                )
                samples = weights @ matrix
                conditions[condition] = {
                    "mean": dict(zip(keys, matrix.mean(axis=0).tolist())),
                    "paired_user_bootstrap_95_percentile": {
                        key: np.percentile(samples[:, j], [2.5, 97.5]).tolist()
                        for j, key in enumerate(keys)
                    },
                }
            return {
                "users": len(subset),
                "examples": len(subset),
                "conditions": conditions,
            }

        category = getattr(self.model, "dataset_name", None)
        if category is None:
            category = getattr(self.module, "dataset_name", None)
        expected = {
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        }
        if category not in expected:
            category = None
        return {
            "panel": "in_sample_training_final_interaction",
            "category": category,
            "category_missing_reason": (
                None if category else "ModelView supplies no recognized dataset_name"
            ),
            "counts": counts,
            "all_lengths": summarize(rows, 42),
            "history_length_strata": {
                label: summarize([row for row in rows if row["stratum"] == label], 42)
                for label in ("4-9", "10-19", "20-49", "50+")
            },
            "identity_checked_users": len(rows),
            "held_out_panel": None,
            "held_out_panel_reason": "No explicit host validation cases with pinned candidates are supplied by this interface",
            "four_category_macro": None,
            "four_category_macro_reason": "This probe receives one category checkpoint; all four paired user panels are required",
            "coordinate_preserving_removal": None,
            "coordinate_preserving_removal_reason": "The supplied HSTU sequence interface has no retained-coordinate/event-mask arguments",
            "architecture": {
                "attention": "Supplied HSTU causal attention with padding-key masking and safe padded query rows",
                "positions": "Learned absolute positions at fixed padded width; compaction changes surviving coordinates",
                "normalization": "Block LayerNorm and final normalized sequence features; shared cosine item scoring",
                "timestamps": "Sequence-only interface; timestamp interventions and prediction-time controls unavailable",
            },
            "retrained_robustness": None,
            "retrained_robustness_reason": "No retraining is performed by this frozen-checkpoint probe",
        }

    def run_full_diagnosis(self, train_loader):
        module_states = [(module, module.training) for module in self.module.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        old_cache = getattr(self.module, "_item_cache", None)
        try:
            self.module.eval()
            if hasattr(self.module, "_item_cache"):
                self.module._item_cache = None
            with torch.no_grad():
                result = self._run_original_diagnosis(train_loader)
                result["metrics"][
                    "recent_history_sensitivity"
                ] = self._recent_history_diagnosis()
            result["metric_definitions"][
                "pos_neg_margin_category_breakdown"
            ] = "Position counts for low/wrong margins, measured_positions and missing_reason; null margin summaries mean no measurements. Item-category lists remain empty because category metadata is unavailable."
            # DEBUG: Keep the metric definition consistent with encode_items.
            result["metric_definitions"]["embedding_collapse_score"] = (
                "Mean off-diagonal cosine similarity of sampled jointly trained fused item "
                "representations returned by encode_items, mapped to [0,1]. Padding is excluded. "
                "Null means fewer than two items. A value near 0.5 means mean cosine near zero, "
                "not substantial directional collapse. This is an alignment measurement, "
                "not a measurement of recommendation diversity."
            )
            result["metric_definitions"]["recent_history_sensitivity"] = {
                "scope": "Frozen-model input sensitivity on observed training targets, not held-out accuracy, causal intent, or robustness after retraining. No official selection metric is replaced.",
                "panel": "Seed-42 selection of at most 200 eligible users from this checkpoint's route; one final training target/user, preceding visible prefix, fixed target plus 99 unique unseen negatives. Repeated targets remain legitimate. Full prefix observations define exclusions before truncation.",
                "cache": "User IDs, targets, original/visible lengths, visible histories, candidate IDs, seeds, deletion indices and permutations are retained in probe._recent_history_cache for this run.",
                "controls": "k=min(max(1,n//4),n-2). recent_delete may remove the readout item; compare with random_delete and oldest_delete. prefinal_delete preserves the latest item; compare with random_nonfinal_delete and oldest_delete. Five repeats for random deletions and each shuffle. Full shuffles may move the final item; prefix shuffles retain it.",
                "readout": "Fixed left-padded width; greatest nonpadding index. Features are recomputed for all interventions with dropout disabled and no item cache.",
                "ranking": "One-based sampled target ranks use descending scores then ascending item ID. These diagnostic ties do not modify official evaluator tie handling. Optimistic rank counts strictly greater competitors; pessimistic rank counts greater-or-equal competitors. RR/HR bounds follow those ranks; tied_target measures any competitor tied with target.",
                "paired_changes": "delta_rank, delta_rr and delta_hr10 are intervention minus intact. Positive delta_rank is deterioration; positive delta_rr/hr10 is improvement. Rank/RR/HR fields are absolute diagnostic measurements. Conditions named *_vs_* contain contrast_rank, contrast_rr and contrast_hr10: the first named intervention minus its control, computed within user after averaging random repetitions. Their intervals bootstrap these paired differences directly.",
                "list_similarity": "overlap10=intersection size/10. frbo10=(1-p)/(1-p^10)*sum(d=1..10,p^(d-1)*prefix_intersection/d), p=.9; finite normalized overlap, not extrapolated RBO.",
                "aggregation": "Average five random repetitions within each example, then examples within user (one here). Report route and visible-length strata separately. Empty strata contain no estimates.",
                "uncertainty": "2000 seed-42 paired user bootstrap replicates; all conditions share each resample. Percentile intervals condition on this checkpoint and cached candidates, exclude training uncertainty, and do not establish causal intent. Four-category macro requires all four category panels and is unavailable in a single-checkpoint call.",
                "limitations": "Compaction changes positions. Timestamps and coordinate-preserving controls are unsupported. Sampled metrics have candidate-sampling bias and do not replace the host's twenty full-catalog qualitative cases. No unavailable reviews, traits or metadata are inferred. Validation-guided associations remain exploratory.",
            }
            return result
        finally:
            self.module.train(module_states[0][1])
            for module, training in module_states:
                module.training = training
            if hasattr(self.module, "_item_cache"):
                self.module._item_cache = old_cache
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _run_original_diagnosis(self, train_loader):
        print(">>> Running HSTU embedding, margin and paired history diagnostics...")
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Range ~[0,1]. Estimated by mean pairwise cosine similarity of sampled item embeddings "
                    "(mapped to [0,1]). Higher means embeddings point in similar directions (worse diversity / more collapse)."
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                "pos_neg_margin_category_breakdown": (
                    "For low-margin (margin<eps) and wrong-rank (margin<0) cases, counts categories "
                    "from meta_data[str(item_id)]['categories'] separately for pos and neg items."
                ),
            },
        }
