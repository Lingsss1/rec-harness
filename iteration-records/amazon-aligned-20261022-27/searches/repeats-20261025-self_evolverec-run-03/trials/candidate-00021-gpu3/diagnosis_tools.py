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
    ### >>> Self_EvolveRec-BLOCK-START: Separate the compatibility view from the trained module
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._probe_cache = None

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Do not infer unavailable item categories
    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Missing embedding evidence is unavailable
            if n < 2 or sample_size < 2 or not torch.isfinite(emb).all():
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute exact sampled mean cosine without a quadratic Gram matrix
            norms = x.norm(dim=-1)
            if not torch.isfinite(x).all() or (norms == 0).any():
                return None
            # Sum all pairwise products algebraically, accumulating in float64.
            # Subtract the actual diagonal rather than assuming unit norms.
            x64 = x.double()
            offdiag_sum = x64.sum(dim=0).square().sum() - x64.square().sum()
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
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnostic transaction manage module modes
        margins_all = []
        valid_count = 0
        nonfinite_count = 0
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

            ### >>> Self_EvolveRec-BLOCK-START: Validate cached batch shapes and skip empty batches before encoding
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            if seq_t.ndim != 2:
                raise ValueError("Diagnostic sequences must have shape B x L")
            if tuple(np.shape(pos)) != tuple(seq_t.shape) or tuple(
                np.shape(neg)
            ) != tuple(seq_t.shape):
                raise ValueError(
                    "Diagnostic positives and negatives must match sequence shape B x L"
                )
            if seq_t.numel() == 0:
                continue
            # Encoding occurs once through the architecture-aware scorer below.
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic scores and avoid margin subtraction overflow
            _, all_logits = self._score_probe(seq_t, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError("predict_score must return shape B x L x 2")
            margin = all_logits[..., 0].double() - all_logits[..., 1].double()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count eligible and nonfinite training margins
            eligible = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            valid_count += int(eligible.sum().item())
            nonfinite_count += int((eligible & ~torch.isfinite(margin)).sum().item())
            valid = eligible & torch.isfinite(margin)
            if valid.any():
                margins_all.append(margin[valid].detach().cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical eligibility for breakdown counts
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
                ### >>> Self_EvolveRec-BLOCK-START: Empty margins are unavailable
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

        ### >>> Self_EvolveRec-BLOCK-START: Expose sample counts and absent category metadata
        breakdown = {
            "available": False,
            "reason": "Item-category metadata is unavailable in this task",
            "eligible_positions": valid_count,
            "finite_positions": sum(m.numel() for m in margins_all),
            "nonfinite_positions": nonfinite_count,
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

    ### >>> Self_EvolveRec-BLOCK-START: Isolated paired history probes with explicit failures and finite double-precision margins
    def _training_cache(self):
        """Build once from public training histories, never from a live iterator.

        At most 32 users are sampled uniformly with a private RNG. Each cached
        row uses the final maxlen training transitions and one independently
        uniform, full-training-history-excluding negative per occupied position.
        No sampler cursor, worker, validation record or test record is accessed.
        """
        if self._probe_cache is not None:
            return self._probe_cache
        histories = self.user_train.get("History", self.user_train)
        itemnum = self.view.item_emb.weight.shape[0] - 1
        maxlen = int(getattr(self.module, "maxlen", 128))
        if maxlen < 1:
            raise ValueError("Diagnostic maxlen must be positive")
        rng = np.random.default_rng(42)
        users = sorted(histories, key=str)
        users = [u for u in users if len(histories[u]) >= 2]
        if users:
            order = rng.permutation(len(users))[:32]
            users = [users[int(i)] for i in order]
        rows = []
        for user in users:
            history = np.asarray(histories[user], dtype=np.int64)
            if np.any((history < 1) | (history > itemnum)):
                raise ValueError("Invalid item ID in public training history")
            forbidden = np.unique(history)
            eligible = itemnum - len(forbidden)
            if eligible < 1:
                continue
            length = min(len(history) - 1, maxlen)
            seq = np.zeros(maxlen, dtype=np.int64)
            pos = np.zeros_like(seq)
            neg = np.zeros_like(seq)
            seq[-length:] = history[-length - 1 : -1]
            pos[-length:] = history[-length:]
            ranks = rng.integers(1, eligible + 1, size=length)
            neg[-length:] = ranks + np.searchsorted(
                forbidden - np.arange(len(forbidden)), ranks, side="right"
            )
            rows.append((seq, pos, neg))
        if rows:
            arrays = [torch.from_numpy(np.stack(x)) for x in zip(*rows)]
        else:
            arrays = [torch.empty((0, maxlen), dtype=torch.long) for _ in range(3)]
        self._probe_cache = (None, *arrays)
        return self._probe_cache

    @torch.no_grad()
    def _history_dependence(self, batch):
        _, seqs, positives, negatives = batch
        maxlen = seqs.shape[1]
        seeds = (42, 43, 44, 45, 46)
        conditions = (
            ("baseline", "recent_delete", "oldest_delete")
            + tuple(f"random_delete_{s}" for s in seeds)
            + tuple(f"shuffle_{s}" for s in seeds)
        )
        examples, rows, items = [], [], []
        counts = Counter()
        popularity = Counter()
        histories = self.user_train.get("History", self.user_train)
        for history in histories.values():
            popularity.update(map(int, history))
        imbalances = {name: [] for name in conditions[1:8]}
        for row_index, (seq, pos, neg) in enumerate(zip(seqs, positives, negatives)):
            positions = torch.where((seq != 0) & (pos != 0) & (neg != 0))[0].tolist()
            # Up to two independently constructed causal prefixes per cached user.
            eligible = [t for t in positions if int((seq[: t + 1] != 0).sum()) >= 3]
            if not eligible:
                continue
            chooser = np.random.default_rng(42 + row_index)
            chosen = sorted(
                chooser.choice(eligible, min(2, len(eligible)), replace=False).tolist()
            )
            for t in chosen:
                prefix = seq[: t + 1][seq[: t + 1] != 0].tolist()
                n = len(prefix)
                k = max(1, (n - 1) // 5)
                stratum = "3-5" if n <= 5 else "6-20" if n <= 20 else "21+"
                counts[stratum] += 1
                examples.append(stratum)
                variants = {"baseline": prefix}
                deletions = {
                    "recent_delete": list(range(n - 1 - k, n - 1)),
                    "oldest_delete": list(range(k)),
                }
                for seed in seeds:
                    # The same seed schedule is independent of global training RNGs.
                    rng = np.random.default_rng([seed, row_index, t])
                    deletions[f"random_delete_{seed}"] = rng.choice(
                        n - 1, k, replace=False
                    ).tolist()
                    order = rng.permutation(n - 1)
                    variants[f"shuffle_{seed}"] = [
                        prefix[int(j)] for j in order
                    ] + prefix[-1:]
                for name, deleted in deletions.items():
                    removed = set(deleted)
                    variants[name] = [
                        x for j, x in enumerate(prefix) if j not in removed
                    ]
                    imbalances[name].append(
                        [
                            float(
                                np.mean(
                                    [np.log1p(popularity[prefix[j]]) for j in deleted]
                                )
                            ),
                            float(
                                np.mean([prefix.count(prefix[j]) > 1 for j in deleted])
                            ),
                            float(np.mean([j / (n - 1) for j in deleted])),
                        ]
                    )
                for name in conditions:
                    history = variants[name]
                    rows.append([0] * (maxlen - len(history)) + history)
                    items.append([int(pos[t]), int(neg[t])])
        if not examples:
            return {
                "available": False,
                "eligible_prefixes": 0,
                "reason": "No cached training prefix has at least three occupied positions",
                "original_length_counts": dict(counts),
                "conditions": {},
            }
        scores = []
        for start in range(0, len(rows), 64):
            sequence = torch.tensor(
                rows[start : start + 64], device=self.device, dtype=torch.long
            )
            candidates = torch.tensor(
                items[start : start + 64], device=self.device, dtype=torch.long
            )
            _, logits = self._score_probe(sequence, candidates.unsqueeze(1))
            if logits.shape != candidates.unsqueeze(1).shape:
                raise ValueError(
                    "History probe predict_score must return shape B x 1 x 2"
                )
            scores.append(logits.reshape(-1, 2).double().cpu())
        logits = torch.cat(scores).reshape(len(examples), len(conditions), 2)
        finite = torch.isfinite(logits).all(dim=2).all(dim=1)
        result = {
            "available": bool(finite.any()),
            "eligible_prefixes": len(examples),
            "paired_finite_prefixes": int(finite.sum()),
            "nonfinite_prefixes": int((~finite).sum()),
            "original_length_counts": dict(counts),
            "finite_length_counts": dict(
                Counter(label for label, keep in zip(examples, finite.tolist()) if keep)
            ),
            "conditions": {},
            "removed_item_controls": {},
        }
        for name, values in imbalances.items():
            averages = np.asarray(values).mean(axis=0)
            result["removed_item_controls"][name] = {
                "mean_log1p_training_popularity": float(averages[0]),
                "repeated_in_original_prefix_share": float(averages[1]),
                "mean_normalized_original_position": float(averages[2]),
                "prefix_count": len(values),
            }
        if not finite.any():
            return result
        margins = logits[finite, :, 0] - logits[finite, :, 1]
        base = margins[:, 0]
        for j, name in enumerate(conditions):
            current = margins[:, j]
            result["conditions"][name] = {
                "count": len(current),
                "mean_margin": float(current.mean()),
                "mean_margin_delta": float((current - base).mean()),
                "target_win_rate": float((current > 0).float().mean()),
                "target_win_rate_delta": float(
                    ((current > 0).float() - (base > 0).float()).mean()
                ),
                "mean_absolute_margin_change": float((current - base).abs().mean()),
                "length_strata": {
                    label: {
                        "count": int(mask.sum()),
                        "mean_margin_delta": float((current[mask] - base[mask]).mean()),
                    }
                    for label in ("3-5", "6-20", "21+")
                    for mask in [
                        torch.tensor(
                            [
                                x == label
                                for x, keep in zip(examples, finite.tolist())
                                if keep
                            ]
                        )
                    ]
                    if mask.any()
                },
            }
        return result

    def run_full_diagnosis(self, train_loader):
        import copy

        definitions = {
            "embedding_collapse_score": (
                "Range ~[0,1]. Estimated by mean pairwise cosine similarity of sampled item embeddings "
                "(mapped to [0,1]). Higher means embeddings point in similar directions. "
                "This is not a measurement of recommendation diversity; null means unavailable."
            ),
            "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
            "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
            "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
            "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
            "pos_neg_margin_category_breakdown": (
                "Low-margin and wrong-rank counts are training diagnostics. Category lists are empty "
                "because category metadata is unavailable; eligible, finite and nonfinite counts are reported."
            ),
            "training_history_dependence": (
                "Paired training-prefix diagnostics, not validation HR@10. Each prefix through t predicts "
                "the supplied positive at t against the same cached training negative. For n>=3 remove "
                "k=max(1,floor(0.2*(n-1))) recent, oldest, or uniformly sampled nonanchor positions. "
                "Keep the latest item, compact left padding and score the last retained state. Five "
                "fixed seeds separately shuffle only pre-anchor slots. Positive margin deltas mean "
                "the changed history improved separation. Every condition uses the same finite prefixes. "
                "Deletion conflates evidence removal, packing and position changes; shuffle measures "
                "item-position sensitivity. Removed-item statistics expose unmatched popularity, repetition "
                "and position imbalance. Prefixes from one user are dependent; no significance is claimed."
            ),
            "probe_availability": (
                "Explicit availability and finite sample counts. Null estimates are missing evidence. "
                "An immutable CPU cache is built once from up to 32 public training histories with private "
                "seed 42, without reading or advancing train_loader. No validation/test data are used."
            ),
            "probe_protocol": (
                "Diagnostic-only fixed protocol; no architecture or predictor changes, no checkpoint "
                "selection by diagnostics. Host macro validation HR@10 remains the selection criterion. "
                "No recency-residual improvement, validation confidence interval or unobserved user trait "
                "is inferred from these training measurements."
            ),
        }
        metrics = {
            "embedding_collapse_score": None,
            "pos_neg_margin_mean": None,
            "pos_neg_margin_p10": None,
            "pos_neg_neg_beats_pos_rate": None,
            "pos_neg_small_margin_rate": None,
            "pos_neg_margin_category_breakdown": {
                "available": False,
                "low_margin_top_categories": {"pos_item": [], "neg_item": []},
                "wrong_rank_top_categories": {"pos_item": [], "neg_item": []},
            },
            "training_history_dependence": {"available": False, "eligible_prefixes": 0},
            "probe_availability": {"available": False},
            "probe_protocol": {
                "scope": "public_training_histories_only",
                "seed": 42,
                "control_seeds": [42, 43, 44, 45, 46],
                "max_users": 32,
                "max_prefixes_per_user": 2,
                "validation_audit": "not supplied; training loader cannot provide validation HR@10",
                "position_preserving_deletion": "unavailable; compacted deletion only",
                "category_metadata": "unavailable",
                "ratings_reviews_and_simulator_traits": "unobserved",
            },
        }
        module = self.module
        snapshot_hook = getattr(module, "diagnostic_snapshot_state", None)
        restore_hook = getattr(module, "diagnostic_restore_state", None)
        has_hooks = callable(snapshot_hook) and callable(restore_hook)
        # The supplied SASRec has no forward caches or train() overrides. Other
        # architectures must explicitly attest to that contract or supply hooks.
        known_sasrec = type(module).__name__ == "SASRec" and all(
            type(m).__module__.startswith("torch.nn")
            or type(m).__name__ in {"SASRec", "PointWiseFeedForward"}
            for m in module.modules()
        )
        safe = known_sasrec or bool(
            getattr(module, "diagnostic_stateless_forward", False)
        )
        if not safe and not has_hooks:
            metrics["probe_availability"]["reason"] = (
                "Unsupported mutable-forward behavior: provide diagnostic state hooks "
                "or attest diagnostic_stateless_forward=True"
            )
            return {"metrics": metrics, "metric_definitions": definitions}
        py_state = random.getstate()
        np_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_device = self.device if self.device.type == "cuda" else None
        cuda_state = (
            torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
        )
        flags = [(m, m.training) for m in module.modules()]
        # Registries contain explicit generators used by the view/model; probe
        # generators themselves are private, freshly seeded and never shared.
        np_generators, torch_generators = [], []
        seen = set()
        for owner in (self.view, module):
            for generator in getattr(owner, "diagnostic_numpy_generators", ()):
                if id(generator) not in seen:
                    seen.add(id(generator))
                    np_generators.append(
                        (generator, copy.deepcopy(generator.bit_generator.state))
                    )
            for generator in getattr(owner, "diagnostic_torch_generators", ()):
                if id(generator) not in seen:
                    seen.add(id(generator))
                    torch_generators.append((generator, generator.get_state().clone()))
        extra_state = None
        hook_ready = False
        buffers = []
        try:
            if has_hooks:
                extra_state = snapshot_hook()
                hook_ready = True
            buffers = [
                (m, name, value, None if value is None else value.detach().clone())
                for m in module.modules()
                for name, value in m._buffers.items()
            ]
            module.eval()
            with torch.no_grad():
                batch = self._training_cache()
                collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
                margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                    [batch], num_batches=1, eps=0.5, topk_categories=20
                )
                dependence = self._history_dependence(batch)
                metrics.update(
                    {
                        "embedding_collapse_score": collapse,
                        "pos_neg_margin_mean": margin["mean"],
                        "pos_neg_margin_p10": margin["p10"],
                        "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                        "pos_neg_small_margin_rate": margin["small_margin_rate"],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "training_history_dependence": dependence,
                        "probe_availability": {
                            "available": collapse is not None
                            or margin["mean"] is not None,
                            "embedding_available": collapse is not None,
                            "embedding_sample_count": min(
                                2048, self.view.item_emb.weight.shape[0] - 1
                            ),
                            "margin_available": margin["mean"] is not None,
                            "margin_finite_positions": breakdown["finite_positions"],
                            "cached_users": int(batch[1].shape[0]),
                            "history_dependence_available": dependence["available"],
                        },
                    }
                )
        except Exception as error:
            # The finally block restores state before this error reaches the host.
            # Empty samples already have explicit unavailable results; unexpected
            # implementation failures must remain visible to the caller.
            raise RuntimeError(
                f"DiagnosisProbe failed: {type(error).__name__}: {error}"
            ) from error
        finally:
            try:
                if hook_ready:
                    restore_hook(extra_state)
            finally:
                try:
                    with torch.no_grad():
                        for owner, name, original, saved in buffers:
                            owner._buffers[name] = original
                            if original is not None:
                                original.copy_(saved)
                finally:
                    for child, training in flags:
                        child.training = training
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, cuda_device)
                    for generator, state in np_generators:
                        generator.bit_generator.state = state
                    for generator, state in torch_generators:
                        generator.set_state(state)
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Architecture-aware causal controls and explicitly observational residual geometry
    _original_history_dependence = _history_dependence
    _original_run_full_diagnosis = run_full_diagnosis

    @torch.no_grad()
    def _score_probe(self, sequence, items):
        """Score explicit causal positions using the actual trained scorer.

        items is B x L x C for all positions or B x 1 x C for
        the final position. Never substitute tied dot products for an
        architecture that exposes candidate-conditioned scoring.
        """
        if sequence.ndim != 2 or items.ndim != 3:
            raise ValueError("Expected sequences B x L and candidates B x S x C")
        batch_size, length = sequence.shape
        if length < 1 or items.shape[0] != batch_size:
            raise ValueError("Invalid probe sequence length or candidate batch size")
        if items.shape[1] not in (1, length):
            raise ValueError(
                "Candidates must cover the final or every sequence position"
            )
        encode = getattr(self.module, "encode_context", None)
        score = getattr(self.module, "score_positions", None)
        if callable(encode) != callable(score):
            raise TypeError(
                "Explicit scorer requires both encode_context and score_positions"
            )
        if not callable(encode):
            features = self.view.log2feats(sequence)
            if not isinstance(features, torch.Tensor) or features.ndim != 3:
                raise ValueError("Compatibility features must have shape B x L x H")
            # DEBUG: Slicing a tensor drops its request-local history context.
            # Preserve intact context-bearing features so residual scoring uses
            # the actual history and original prefix positions. Legacy scorers
            # without attached context still receive position-aligned features.
            if getattr(features, "_sasrec_history_context", None) is not None:
                scoring_features = features
            else:
                scoring_features = (
                    features[:, -1:, :] if items.shape[1] == 1 else features
                )
            logits = self.view.predict_score(scoring_features, items)
        else:
            context = encode(sequence)
            if not isinstance(context, tuple) or len(context) != 4:
                raise TypeError(
                    "Explicit scorer must expose (features, keys, values, mask)"
                )
            features, keys, values, mask = context
            if features.ndim != 3 or features.shape[:2] != sequence.shape:
                raise ValueError("Encoded features must have shape B x L x H")
            count = items.shape[1]
            rows = torch.arange(batch_size, device=sequence.device).repeat_interleave(
                count
            )
            steps = (
                torch.full_like(rows, length - 1)
                if count == 1
                else torch.arange(length, device=sequence.device).repeat(batch_size)
            )
            flat_items = items.reshape(-1, items.shape[-1])
            pieces = []
            for start in range(0, rows.numel(), 128):
                stop = start + 128
                pieces.append(
                    score(
                        features,
                        keys,
                        values,
                        mask,
                        rows[start:stop],
                        steps[start:stop],
                        flat_items[start:stop],
                    )
                )
            logits = (
                torch.cat(pieces, dim=0).reshape(items.shape)
                if pieces
                else features.new_empty(items.shape)
            )
        if logits.shape != items.shape:
            raise ValueError("Probe scorer must preserve candidate shape B x S x C")
        return features, logits

    @torch.no_grad()
    def _controlled_history_probes(self, batch):
        _, sequences, positives, negatives = batch
        maxlen = sequences.shape[1]
        seeds = (42, 43, 44, 45, 46)
        conditions = (
            "baseline",
            "unchanged",
            "last_only",
            "recent_identity_replace",
            "oldest_identity_replace",
        ) + tuple(f"recent_shuffle_{seed}" for seed in seeds)
        histories = self.user_train.get("History", self.user_train)
        popularity = Counter()
        for history in histories.values():
            popularity.update(map(int, history))
        rows, candidates, records = [], [], []
        for row_index, (seq, pos, neg) in enumerate(
            zip(sequences, positives, negatives)
        ):
            occupied = torch.where((seq != 0) & (pos != 0) & (neg != 0))[0].tolist()
            if not occupied:
                continue
            rng = np.random.default_rng(42 + row_index)
            chosen = sorted(
                rng.choice(occupied, min(2, len(occupied)), replace=False).tolist()
            )
            for t in chosen:
                prefix = seq[: t + 1][seq[: t + 1] != 0].tolist()
                n = len(prefix)
                target, negative = int(pos[t]), int(neg[t])
                # The same cached unseen training negative replaces identities
                # in both controls. It is deliberately also the scored negative;
                # these are intervention controls, not unbiased performance tests.
                k = min(max(1, (n - 1) // 5), n - 1)
                variants = {
                    "baseline": list(prefix),
                    "unchanged": list(prefix),
                    "last_only": prefix[-1:],
                }
                for name, slots in (
                    ("recent_identity_replace", range(n - 1 - k, n - 1)),
                    ("oldest_identity_replace", range(k)),
                ):
                    changed = list(prefix)
                    for slot in slots:
                        changed[slot] = negative
                    variants[name] = changed
                for seed in seeds:
                    # Permute only the preceding four recent slots, preserving
                    # the latest item and every slot outside this window.
                    changed = list(prefix)
                    slots = list(range(max(0, n - 5), n - 1))
                    private_rng = np.random.default_rng([seed, row_index, t])
                    order = private_rng.permutation(len(slots))
                    for destination, source in zip(slots, order):
                        changed[destination] = prefix[slots[int(source)]]
                    variants[f"recent_shuffle_{seed}"] = changed
                records.append(
                    {
                        "length": n,
                        "stratum": (
                            "1-2"
                            if n < 3
                            else "3-5" if n <= 5 else "6-20" if n <= 20 else "21+"
                        ),
                        "ordered_ids": prefix[-5:][::-1],
                        "shuffled_ids": variants["recent_shuffle_42"][-5:][::-1],
                        "popularity_margin": float(
                            np.log1p(popularity[target])
                            - np.log1p(popularity[negative])
                        ),
                    }
                )
                for name in conditions:
                    history = variants[name]
                    rows.append([0] * (maxlen - len(history)) + history)
                    candidates.append([target, negative])
        if not records:
            return {"available": False, "eligible_prefixes": 0}, {
                "available": False,
                "eligible_prefixes": 0,
            }

        outputs, baseline_features = [], []
        for start in range(0, len(rows), 64):
            sequence = torch.tensor(
                rows[start : start + 64], dtype=torch.long, device=self.device
            )
            items = torch.tensor(
                candidates[start : start + 64], dtype=torch.long, device=self.device
            )
            features, logits = self._score_probe(sequence, items.unsqueeze(1))
            final = features[:, -1:, :]
            if logits.shape != items.unsqueeze(1).shape:
                raise ValueError("Controlled probes require B x 1 x 2 scores")
            outputs.append(logits[:, 0].double().cpu())
            for local in range(len(sequence)):
                if (start + local) % len(conditions) == 0:
                    baseline_features.append(final[local, 0].double().cpu())
        logits = torch.cat(outputs).reshape(len(records), len(conditions), 2)
        finite = torch.isfinite(logits).all(dim=(1, 2))
        result = {
            "available": bool(finite.any()),
            "eligible_prefixes": len(records),
            "paired_finite_prefixes": int(finite.sum()),
            "nonfinite_prefixes": int((~finite).sum()),
            "short_prefixes_without_identifiable_prior_order": sum(
                r["length"] < 3 for r in records
            ),
            "original_length_counts": dict(Counter(r["stratum"] for r in records)),
            "shuffle_seed": 42,
            "conditions": {},
        }
        if finite.any():
            margins = logits[finite, :, 0] - logits[finite, :, 1]
            base = margins[:, 0]
            labels = [r["stratum"] for r, keep in zip(records, finite.tolist()) if keep]
            for column, name in enumerate(conditions):
                current = margins[:, column]
                result["conditions"][name] = {
                    "mean_margin": float(current.mean()),
                    "mean_margin_delta": float((current - base).mean()),
                    "mean_absolute_margin_change": float((current - base).abs().mean()),
                    "target_win_rate": float((current > 0).double().mean()),
                    "length_strata": {},
                }
                for label in ("1-2", "3-5", "6-20", "21+"):
                    mask = torch.tensor([x == label for x in labels], dtype=torch.bool)
                    if mask.any():
                        result["conditions"][name]["length_strata"][label] = {
                            "count": int(mask.sum()),
                            "mean_margin_delta": float(
                                (current[mask] - base[mask]).mean()
                            ),
                        }
            popularity_margins = torch.tensor(
                [r["popularity_margin"] for r in records], dtype=torch.float64
            )[finite]
            result["history_independent_popularity"] = {
                "count": len(popularity_margins),
                "mean_log1p_count_margin": float(popularity_margins.mean()),
                "target_win_rate": float((popularity_margins > 0).double().mean()),
                "ties": int((popularity_margins == 0).sum()),
                "margin_delta_under_every_history_intervention": 0.0,
            }
            result["unchanged_max_absolute_logit_difference"] = float(
                (logits[finite, 1] - logits[finite, 0]).abs().max()
            )

        # Observe geometry without changing the architecture or claiming that
        # these hypothetical residuals were trained or host-validated.
        features = torch.stack(baseline_features)
        embedding_device = self.view.item_emb.weight.device
        ordered = torch.zeros(
            (len(records), 5), dtype=torch.long, device=embedding_device
        )
        shuffled = torch.zeros_like(ordered)
        for row, record in enumerate(records):
            ordered[row, : len(record["ordered_ids"])] = torch.tensor(
                record["ordered_ids"], device=embedding_device
            )
            shuffled[row, : len(record["shuffled_ids"])] = torch.tensor(
                record["shuffled_ids"], device=embedding_device
            )
        valid = ordered.ne(0).cpu()
        raw = self.view.item_emb(ordered).detach().double().cpu()
        shuffled_raw = self.view.item_emb(shuffled).detach().double().cpu()
        raw = raw.masked_fill(~valid.unsqueeze(-1), 0)
        shuffled_raw = shuffled_raw.masked_fill(~valid.unsqueeze(-1), 0)
        weights = torch.exp(-torch.arange(5, dtype=torch.float64))[None, :] * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        residuals = {
            "last_only": raw[:, 0],
            "uniform_five": raw.sum(1) / valid.sum(1, keepdim=True).clamp_min(1),
            "exponential_five": (raw * weights.unsqueeze(-1)).sum(1),
            "shuffled_exponential_five": (shuffled_raw * weights.unsqueeze(-1)).sum(1),
        }
        geometry = {
            "available": False,
            "eligible_prefixes": len(records),
            "feature_source": "actual encoded final states; candidate-conditioned score contributions are not query vectors",
            "scope": "untrained additive counterfactual geometry only",
            "arms": {},
        }
        for name, residual in residuals.items():
            keep = (
                finite
                & torch.isfinite(features).all(-1)
                & torch.isfinite(residual).all(-1)
            )
            entry = {
                "finite_prefixes": int(keep.sum()),
                "nonfinite_prefixes": int((~keep).sum()),
            }
            if keep.any():
                geometry["available"] = True
                hidden, pooled = features[keep], residual[keep]
                hidden_norm, pooled_norm = hidden.norm(dim=-1), pooled.norm(dim=-1)
                nonzero = hidden_norm.gt(0)
                aligned = nonzero & pooled_norm.gt(0)
                entry.update(
                    {
                        "mean_feature_norm": float(hidden_norm.mean()),
                        "mean_raw_residual_norm": float(pooled_norm.mean()),
                        "nonzero_feature_prefixes": int(nonzero.sum()),
                        "mean_raw_residual_to_feature_norm": (
                            float((pooled_norm[nonzero] / hidden_norm[nonzero]).mean())
                            if nonzero.any()
                            else None
                        ),
                        "cosine_eligible_prefixes": int(aligned.sum()),
                        "mean_feature_residual_cosine": (
                            float(
                                F.cosine_similarity(
                                    hidden[aligned], pooled[aligned], dim=-1
                                ).mean()
                            )
                            if aligned.any()
                            else None
                        ),
                        "alpha_query_norms": {
                            str(alpha): float(
                                (hidden + alpha * pooled).norm(dim=-1).mean()
                            )
                            for alpha in (0.0, 0.1, 0.3)
                        },
                    }
                )
            geometry["arms"][name] = entry
        return result, geometry

    def _history_dependence(self, batch):
        # The original runner invokes this inside its existing eval/no-grad
        # transaction, so all added forward calls receive the same restoration.
        result = self._original_history_dependence(batch)
        controls, geometry = self._controlled_history_probes(batch)
        result["controlled_history_probes"] = controls
        result["residual_geometry"] = geometry
        return result

    def run_full_diagnosis(self, train_loader):
        result = self._original_run_full_diagnosis(train_loader)
        dependence = result["metrics"]["training_history_dependence"]
        result["metrics"]["probe_protocol"].update(
            {
                "scoring": "actual explicit architecture scorer when available; compatibility scorer otherwise",
                "trained_recency_arm_comparisons": "unavailable; this patch changes diagnostics only",
                "paired_host_validation_uncertainty": "unavailable; host per-user hits were not supplied",
            }
        )
        for key in ("controlled_history_probes", "residual_geometry"):
            result["metrics"][key] = dependence.pop(key, {"available": False})
        result["metric_definitions"].update(
            {
                "controlled_history_probes": (
                    "Auxiliary paired training controls using up to two causal prefixes per cached user, "
                    "including histories of length one or two. Every condition uses the same candidates "
                    "and jointly finite prefixes. Unchanged input calibrates numerical variation. Last-only "
                    "compacts the input and therefore changes packing and positions. Identity replacement "
                    "preserves length, mask, latest item and positions, but substitutes the scored cached "
                    "negative into matched recent or oldest slots; its effects are intentionally adversarial. "
                    "Five seeded shuffles permute only the preceding four recent slots and preserve the "
                    "latest item, full multiset and all other positions. Lengths below three cannot identify "
                    "ordering beyond the latest item. Popularity uses only public training interaction counts, "
                    "ignores history, and has zero intervention delta by construction. These are sensitivity "
                    "controls, not trained residual-arm comparisons, validation HR@10 or significance tests."
                ),
                "residual_geometry": (
                    "Observational geometry on the same fixed training prefixes. Raw shared item embeddings "
                    "pool the latest at most five valid inputs. Ordered weights are exp(-j), normalized over "
                    "valid inputs, with j=0 latest; shuffled pooling uses private seed 42 and preserves latest. "
                    "Reports finite counts, feature and residual norms, norm ratios, cosine alignment and "
                    "hypothetical ||feature+alpha*residual|| for alpha in {0,0.1,0.3}. Geometry uses jointly "
                    "finite scored prefixes. Features are actual encoded states when the explicit scorer "
                    "is exposed, otherwise compatibility-view states. Candidate-conditioned score terms "
                    "are not represented by these hypothetical query vectors. "
                    "These additions are not trained, serialized or used for selection. No architectural "
                    "gain, host validation contrast, confidence interval or held-out result is established."
                ),
            }
        )
        return result

    ### <<< Self_EvolveRec-BLOCK-END
