import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Bounded training-only catalog diagnostic dependencies
import random
import inspect
import os
import time
import copy
from itertools import islice
from collections import Counter, defaultdict
from scipy.stats import hypergeom

### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret catalog competition as conditional training fit
    - Report support, missing evidence, truncation and undefined reasons.
    - These are training-fit diagnostics, potentially affected by memorization, not host validation HR@10 or generalization measurements.
    - The analytic sampled/full-catalog hit gap is an expected consequence of uniform candidate sampling; its size alone is not a model defect.
    - The declared training-catalog population differs from host eligibility and simulator unseen-item filtering. Competitors are not verified dislikes.
    - Anonymous rows are prefixes, not identified independent users; do not invent user uncertainty intervals.
    - Directional embedding concentration does not measure recommendation diversity.
    - Ratings, reviews, item categories, prices, stores and simulator traits are unavailable. Do not infer intent, compatibility, ownership needs, mood or causal mechanisms.
    - Module roundtrip checks do not establish generated-predictor or category-routing parity.
    - Failed correctness checks are CRITICAL. Missing checks are unresolved evidence. Use STABLE only within the scope of measured evidence.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and explicitly unavailable attributes
    def __init__(self, view, device):
        if not isinstance(view.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.model = view
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
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

            ### >>> Self_EvolveRec-BLOCK-START: Finite bounded embedding concentration without a Gram matrix
            n = emb.size(0)
            if n < 2:
                return None
            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            idx = torch.randperm(n, generator=g, device=emb.device)[:sample_size]
            x = emb.index_select(0, idx).float()
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError("Nonfinite diagnostic item embeddings")
            x = x[x.norm(dim=-1).gt(0)]
            k = x.size(0)
            if k < 2:
                return None
            x = F.normalize(x, p=2, dim=-1)
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
            return float((mean_cos + 1.0) * 0.5)
            ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained module
        self.model.model.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Reuse isolated cached GRU features and aligned finite pairs
        for step, batch in enumerate(islice(dataloader, num_batches)):
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError(
                    "Diagnostic batches must be (None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic tensors must have matching B x L shapes")
            cached = getattr(self, "_diagnostic_features", None)
            log_feats = (
                cached[step] if cached is not None else self.model.log2feats(seq_t)
            )
            expected = (*seq_t.shape, self.model.item_emb.embedding_dim)
            if tuple(log_feats.shape) != expected:
                raise RuntimeError("GRU diagnostic features lost B x L alignment")
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0].float() - all_logits[..., 1].float()
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if margin.shape != valid.shape or not bool(
                torch.isfinite(margin[valid]).all()
            ):
                raise RuntimeError("Invalid aligned diagnostic margins")
            if bool(valid.any()):
                margins_all.append(margin[valid].detach().cpu())
            margin_cpu = margin.detach().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
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
            ### >>> Self_EvolveRec-BLOCK-START: Undefined margins are not zero observations
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

        ### >>> Self_EvolveRec-BLOCK-START: Declare margin support and unavailable category metadata
        breakdown = {
            "valid_positions": int(sum(m.numel() for m in margins_all)),
            "undefined_reason": (
                None if margins_all else "No valid aligned training pairs"
            ),
            "category_availability": "Unavailable in this shared task; category lists are empty",
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

    ### >>> Self_EvolveRec-BLOCK-START: Isolated bounded training-catalog competition audit
    def _isolated_cache(self):
        # Explicit integration with the actual family's sampler, never the supplied
        # loader iterator, sampler, workers, generators or prefetch queues.
        module = inspect.getmodule(type(self.model.model))
        sampler = getattr(module, "sample_batch", None) if module is not None else None
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            return [], {"unavailable_reason": "Training histories are not a mapping"}
        if not callable(sampler):
            return [], {
                "unavailable_reason": "Actual model module exposes no family sample_batch"
            }
        width = getattr(module, "MAXLEN", None)
        if not isinstance(width, int) or width < 1:
            return [], {
                "unavailable_reason": "Family module exposes no positive MAXLEN"
            }
        users = [u for u, h in histories.items() if len(h) >= 2]
        if not users:
            return [], {"unavailable_reason": "No eligible training histories"}
        args = (histories, int(self.model.model.itemnum), 16, width)
        try:
            inspect.signature(sampler).bind(*args)
        except (TypeError, ValueError):
            return [], {
                "unavailable_reason": "Family sampler lacks the four-argument diagnostic interface"
            }
        # Temporarily installed diagnostic RNG streams are restored by the outer
        # finally. Default family calls are unaugmented, using training data only.
        random.seed(42)
        np.random.seed(42)
        torch.random.default_generator.manual_seed(42)
        cache = []
        for _ in range(10):
            result = sampler(*args)
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                raise ValueError(
                    "Family sample_batch must return sequence, positive, negative"
                )
            tensors = tuple(
                torch.as_tensor(x, dtype=torch.long).clone().cpu() for x in result
            )
            if tensors[0].shape != (16, width) or any(
                x.shape != tensors[0].shape for x in tensors
            ):
                raise ValueError("Family sample_batch returned invalid B x L tensors")
            cache.append((None, *tensors))
        return cache, {
            "unavailable_reason": None,
            "source": f"{module.__name__}.sample_batch",
            "sampling_order": "ten sequential independent sampler calls, sixteen rows each",
            "seed": 42,
            "num_workers": 0,
            "rows": 160,
            "batches": 10,
            "max_batches": 10,
            "max_prefixes": 256,
            "identifiers": "Unavailable; rows remain anonymous prefixes",
            "alignment": "Actual family sampler's observed sequence and aligned training positive",
            "active_training_iterator_consumed": False,
            "explicit_training_generators_touched": False,
        }

    def _correctness_audit(self):
        module = self.model.model
        a, b = 1, min(2, int(module.itemnum))
        checks = {}

        def sequence(history, width):
            result = torch.zeros(1, width, dtype=torch.long, device=self.device)
            if history:
                result[0, -len(history) :] = torch.tensor(history, device=self.device)
            return result

        def check(name, actual, expected):
            if actual.dtype != torch.float32 or expected.dtype != torch.float32:
                raise RuntimeError("Correctness audit requires actual FP32 execution")
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            checks[name] = float((actual - expected).abs().max())

        wide = sequence([a, b, a], 10)
        reference = self.model.log2feats(wide)
        for width in (3, 6):
            check(
                f"left_padding_{width}",
                self.model.log2feats(sequence([a, b, a], width))[:, -3:],
                reference[:, -3:],
            )
        check("padding_zero", reference[:, :-3], torch.zeros_like(reference[:, :-3]))
        empty = sequence([], 10)
        check("empty_history", self.model.log2feats(empty), torch.zeros_like(reference))
        single = sequence([a], 10)
        check(
            "single_item",
            self.model.log2feats(single)[:, -1],
            self.model.log2feats(single[:, -1:])[:, -1],
        )
        batch = torch.cat((wide, empty, single))
        expected = torch.cat(
            (reference, torch.zeros_like(reference), self.model.log2feats(single))
        )
        check("independent_batch_states", self.model.log2feats(batch), expected)
        for end in (1, 2, 3):
            prefix = sequence([a, b, a][:end], 10)
            check(
                f"causal_alignment_{end}",
                reference[:, 6 + end],
                self.model.log2feats(prefix)[:, -1],
            )
        candidates = torch.arange(
            1, min(7, int(module.itemnum)) + 1, device=self.device
        )
        scores = self.model.predict_score(reference[0, -1], candidates)
        check("module_features", module.log2feats(wide), reference)
        check("module_score_path", module.score_candidates(wide, candidates), scores)
        # A temporary same-weight replica checks state-dict loading only. It is
        # neither independently fitted nor saved as a category checkpoint.
        replica = copy.deepcopy(module).eval()
        try:
            replica.load_state_dict(module.state_dict(), strict=True)
            check(
                "state_dict_roundtrip",
                replica.score_candidates(wide, candidates),
                scores,
            )
        finally:
            del replica
        return {
            "status": "passed_module_checks",
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_abs_differences": checks,
            "generated_predictor_parity": "Unavailable: no executable generated-predictor interface supplied; routing remains unverified",
        }

    @staticmethod
    def _summary(rows):
        keys = (
            "rank",
            "full_diagnostic_hit",
            "expected_sampled_diagnostic_hit",
            "expected_sampled_hit_full_miss",
            "target_minus_tenth_competitor",
        )
        result = {
            "prefix_count": len(rows),
            "aggregation": "anonymous_prefix_means",
            "user_count": None,
            "user_bootstrap_interval": None,
        }
        for key in keys:
            values = [row[key] for row in rows if row[key] is not None]
            result[key] = {
                "mean": float(np.mean(values)) if values else None,
                "support": len(values),
                "undefined_reason": None if values else "No defined observations",
            }
        result["rank_distribution"] = dict(
            sorted(Counter(row["rank"] for row in rows).items())
        )
        if rows:
            ranks = [row["rank"] for row in rows]
            result["rank_quantiles"] = dict(
                zip(
                    ("min", "p50", "p90", "max"),
                    np.quantile(ranks, [0, 0.5, 0.9, 1]).tolist(),
                )
            )
        else:
            result["rank_quantiles"] = None
        return result

    def _catalog_audit(self, cache, deadline):
        itemnum = int(self.model.model.itemnum)
        histories = self.user_train.get("History", self.user_train)
        frequency = Counter(
            int(i) for h in histories.values() for i in h if int(i) != 0
        )
        if any(i < 1 or i > itemnum for i in frequency):
            raise ValueError("Training catalog contains an out-of-vocabulary item")
        catalog = sorted(frequency)
        ordered = sorted(catalog, key=lambda i: (-frequency[i], i))
        first, second = (len(catalog) + 2) // 3, (2 * len(catalog) + 2) // 3
        bands = {
            i: "head" if j < first else "middle" if j < second else "tail"
            for j, i in enumerate(ordered)
        }
        band_support = {
            name: {
                "items": sum(bands[i] == name for i in catalog),
                "interaction_mass": sum(
                    frequency[i] for i in catalog if bands[i] == name
                ),
            }
            for name in ("head", "middle", "tail")
        }
        records, counts, seen = [], Counter(), Counter()
        parity = {
            "status": "unavailable",
            "reason": "No eligible catalog-scoring prefix",
        }
        completed = 0
        for batch_index, (_, sequence, positive, _) in enumerate(cache):
            if time.perf_counter() >= deadline:
                counts["budget_truncation_batches"] = len(cache) - batch_index
                break
            features = self._diagnostic_features[batch_index]
            for row in range(sequence.size(0)):
                counts["visited_prefixes"] += 1
                positions = sequence[row].ne(0).nonzero(as_tuple=True)[0]
                if not positions.numel():
                    counts["empty_history"] += 1
                    continue
                last = int(positions.max())
                target = int(positive[row, last])
                if target == 0:
                    counts["missing_target"] += 1
                    continue
                if target not in frequency:
                    counts["target_outside_training_catalog_protocol_error"] += 1
                    continue
                history = sequence[row, positions].tolist()
                identity = (tuple(history), target)
                counts["duplicate_prefix_rows"] += int(seen[identity] > 0)
                seen[identity] += 1
                excluded = set(history) | {0, target}
                eligible = [i for i in catalog if i not in excluded]
                feature = features[row, last]
                target_items = torch.tensor(
                    [target], device=self.device, dtype=torch.long
                )
                target_score = float(
                    self.model.predict_score(feature, target_items).reshape(-1)[0]
                )
                finite, preceding, ties = np.isfinite(target_score), 0, 0
                top = []
                for start in range(0, len(eligible), 4096):
                    items = torch.tensor(
                        eligible[start : start + 4096],
                        device=self.device,
                        dtype=torch.long,
                    )
                    scores = self.model.predict_score(feature, items).reshape(-1)
                    if scores.dtype != torch.float32 or scores.numel() != items.numel():
                        raise RuntimeError(
                            "Catalog audit requires aligned actual FP32 scores"
                        )
                    if not bool(torch.isfinite(scores).all()):
                        finite = False
                        break
                    if parity["status"] == "unavailable":
                        # Check different chunk shapes on a deterministic bounded slice.
                        sample = items[: min(32, items.numel())]
                        direct = scores[: sample.numel()]
                        split = torch.cat(
                            [
                                self.model.predict_score(feature, x).reshape(-1)
                                for x in sample.split(7)
                            ]
                        )
                        torch.testing.assert_close(direct, split, rtol=1e-5, atol=1e-6)
                        parity = {
                            "status": "passed",
                            "items": sample.numel(),
                            "max_abs_difference": float((direct - split).abs().max()),
                            "rtol": 1e-5,
                            "atol": 1e-6,
                        }
                    equal = scores.eq(target_score)
                    preceding += int(
                        (scores.gt(target_score) | (equal & items.lt(target))).sum()
                    )
                    ties += int(equal.sum())
                    # Stable sorting preserves ascending input IDs for exact score
                    # ties. Merge eleven competitors to measure boundary ties too.
                    order = torch.argsort(scores, descending=True, stable=True)[:11]
                    pairs = zip(
                        scores[order].cpu().tolist(), items[order].cpu().tolist()
                    )
                    top = sorted(top + list(pairs), key=lambda x: (-x[0], x[1]))[:11]
                if not finite:
                    counts["nonfinite_score_invalidated_prefix"] += 1
                    continue
                c, m = len(eligible), preceding
                n = min(99, c)
                probability = (
                    1.0 if c == 0 else float(hypergeom.cdf(min(9, n), c, m, n))
                )
                if not np.isfinite(probability):
                    raise RuntimeError(
                        "Nonfinite hypergeometric diagnostic probability"
                    )
                full_hit = int(m <= 9)
                overall = sorted(
                    top + [(target_score, target)], key=lambda x: (-x[0], x[1])
                )
                boundary_tie = (
                    bool(overall[9][0] == overall[10][0]) if c >= 10 else None
                )
                records.append(
                    {
                        "batch_index": batch_index,
                        "row_index": row,
                        "user_id": None,
                        "prefix_id": f"batch_{batch_index}_row_{row}",
                        "history": history,
                        "target": target,
                        "target_score": target_score,
                        "competitors": c,
                        "preceding": m,
                        "rank": 1 + m,
                        "target_score_ties": ties,
                        "top10_boundary_tie": boundary_tie,
                        "full_diagnostic_hit": full_hit,
                        "diagnostic_candidate_count": 1 + n,
                        "expected_sampled_diagnostic_hit": probability,
                        "expected_sampled_hit_full_miss": probability * (1 - full_hit),
                        "target_minus_tenth_competitor": (
                            target_score - top[9][0] if c >= 10 else None
                        ),
                        "margin_undefined_reason": (
                            None if c >= 10 else "fewer_than_ten_competitors"
                        ),
                        "training_popularity_band": bands[target],
                        "history_length": len(history),
                        "length_bin": (
                            "1"
                            if len(history) == 1
                            else (
                                "2-5"
                                if len(history) <= 5
                                else "6-20" if len(history) <= 20 else "21+"
                            )
                        ),
                        "repeat_target": target in history,
                    }
                )
            completed += 1
        strata = defaultdict(list)
        for row in records:
            key = (
                row["training_popularity_band"],
                row["length_bin"],
                row["repeat_target"],
                row["diagnostic_candidate_count"],
            )
            strata[key].append(row)
        result = {
            "status": "completed" if completed == len(cache) else "truncated",
            "catalog_size": len(catalog),
            "popularity_bands": band_support,
            "completed_batches": completed,
            "counts": dict(counts),
            "summary": self._summary(records),
            "chunk_parity": parity,
            "joint_strata": [
                {
                    "popularity_band": key[0],
                    "length_bin": key[1],
                    "repeat_target": key[2],
                    "candidate_count": key[3],
                    "summary": self._summary(rows),
                }
                for key, rows in sorted(strata.items())
            ],
        }
        return result, records

    def run_full_diagnosis(self, train_loader):
        module = self.model.model
        modes = [(child, child.training) for child in module.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_device = None
        cuda_state = None
        if self.device.type == "cuda":
            cuda_device = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            cuda_state = torch.cuda.get_rng_state(cuda_device)
            torch.cuda.synchronize(cuda_device)
        started = time.perf_counter()
        # Fixed optional-audit guard; checked between complete sixteen-prefix
        # batches. A single batch may finish beyond this soft time limit.
        deadline = started + 30.0
        try:
            module.eval()
            cache, cache_protocol = self._isolated_cache()
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, enabled=False
            ):
                collapse = self.analyze_embeddings_collapse_cosine()
                correctness = (
                    self._correctness_audit()
                    if cache
                    else {
                        "status": "unavailable",
                        "reason": cache_protocol["unavailable_reason"],
                    }
                )
                self._diagnostic_features = [
                    self.model.log2feats(batch[1].to(self.device)) for batch in cache
                ]
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(cache)
                )
                if cache:
                    competition, records = self._catalog_audit(cache, deadline)
                else:
                    competition, records = {
                        "status": "unavailable",
                        "reason": cache_protocol["unavailable_reason"],
                        "summary": self._summary([]),
                    }, []
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            elapsed = time.perf_counter() - started
            protocol = {
                "seed": 42,
                "cache": cache_protocol,
                "precision": "FP32, autocast disabled",
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "device": str(self.device),
                "chunk_size": 4096,
                "chunk_size_reductions": [],
                "soft_wall_time_guard_seconds": 30.0,
                "elapsed_synchronized_wall_seconds": elapsed,
                "gpu_time_note": "Wall time includes synchronized GPU work, CPU sampling and artifact preparation; not exclusive GPU kernel time",
                "process_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(cuda_device))
                    if cuda_device is not None
                    else None
                ),
                "memory_note": "Process CUDA high-water mark; not reset, so may include earlier training",
                "catalog": "Sorted unique nonzero real IDs in training histories; embedding bounds checked",
                "competitors": "Catalog minus original observed prefix, target and padding; other full-history positives remain eligible",
                "target": "Included exactly once, including repeat targets; absent catalog targets are protocol errors",
                "ranking": "Descending actual FP32 score, ascending item ID for exact ties; no tie tolerance",
                "sampling": "Uniform n=min(99,C) unique competitors without replacement",
                "popularity": "Training interaction counts; descending count then ascending item ID; boundaries ceil(V/3), ceil(2V/3)",
                "aggregation": "Anonymous prefix means, no user-bootstrap intervals; duplicate rows retained and counted",
                "limits": "Training fit includes memorization and sampler weighting. Eligibility differs from host and simulator. Expected sampling gaps alone are not defects. No generalization, preference, compatibility, diversity or causal claims.",
            }
            result = {
                "metrics": {
                    "correctness_audit": correctness,
                    "embedding_collapse_score": collapse,
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "catalog_competition": competition,
                    "diagnostic_protocol": protocol,
                },
                "metric_definitions": {
                    "correctness_audit": "Measured FP32 padding, causal alignment, empty-history, batch-state, ModelView/module and state-dict roundtrip checks; failure raises. Generated writer and routing parity remain unverified.",
                    "embedding_collapse_score": "(Mean off-diagonal cosine+1)/2 over up to 2048 nonzero-norm sampled item embeddings excluding padding. Directional concentration, not recommendation diversity; null with insufficient support.",
                    "pos_neg_margin_mean": "Mean positive-minus-negative score over aligned nonzero input, positive and negative training positions; null without support.",
                    "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of valid training margins; null without support.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid training margins strictly below zero.",
                    "pos_neg_small_margin_rate": "Fraction of valid training margins below 0.5.",
                    "pos_neg_margin_category_breakdown": "Valid-position and low/wrong-margin counts. Item categories are unavailable, so category lists are empty.",
                    "diagnostic_protocol": "Measured execution details and fixed training-only population; no supplied training iterator or explicit training generator is consumed.",
                    "catalog_competition": {
                        "rank": "Exact training-catalog target rank 1+M; M counts C eligible competitors preceding target under declared tie rules.",
                        "full_diagnostic_hit": "Indicator M<=9 on the declared training-catalog population; not host validation HR@10.",
                        "expected_sampled_diagnostic_hit": "P(X<=9), X~Hypergeometric(C,M,min(99,C)); C=0 gives one. Analytic conditional probability, not an observed benchmark hit.",
                        "expected_sampled_hit_full_miss": "P(X<=9)*1[M>=10]; its prefix mean equals expected sampled-hit rate minus full diagnostic-hit rate. Expected sampling behavior, not evidence of a defect.",
                        "target_minus_tenth_competitor": "Target score minus tenth competitor score; null with fewer than ten competitors. Exact ties require item-ID ordering to determine hits.",
                        "rank_distribution": "Complete empirical rank counts over completed finite prefixes; duplicate sampled prefixes retained.",
                        "joint_strata": "Training item-rank popularity thirds crossed with observed length, repeat-target status and actual sampled candidate count; descriptive prefix means with support, no user intervals.",
                        "counts": "Visited/skipped/protocol-error/nonfinite/duplicate prefixes and unfinished batches. Nonfinite scores invalidate the entire prefix, never individual competitors.",
                        "chunk_parity": "Measured bounded comparison of candidate scoring with different chunk shapes; does not establish cross-platform bitwise reproducibility.",
                    },
                },
            }
            artifact_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
            )
            os.makedirs(artifact_dir, exist_ok=True)
            category = str(getattr(self.model, "dataset_name", "category_unreported"))
            safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in category)
            artifact = os.path.join(
                artifact_dir, f"{safe}_catalog_{time.time_ns()}.json"
            )
            with open(artifact, "w") as handle:
                json.dump(
                    {"result": result, "prefix_records": records},
                    handle,
                    indent=2,
                    allow_nan=False,
                )
            result["metrics"]["diagnostic_artifact"] = artifact
            result["metric_definitions"][
                "diagnostic_artifact"
            ] = "Bounded trial-local JSON containing protocol, measured summaries and completed prefix records; no other-trial access."
            return result
        finally:
            self.__dict__.pop("_diagnostic_features", None)
            for child, training in modes:
                child.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, cuda_device)

    ### <<< Self_EvolveRec-BLOCK-END
