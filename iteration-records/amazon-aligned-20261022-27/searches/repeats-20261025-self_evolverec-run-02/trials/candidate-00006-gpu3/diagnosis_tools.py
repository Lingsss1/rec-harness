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


### >>> Self_EvolveRec-BLOCK-START: Reproducible identified-prefix diagnostics with cached scoring and isolated RNG state
class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.user_train = view.user_train
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.category = getattr(view, "dataset_name", None)
        self.maxlen = int(getattr(self.model, "maxlen", 128))
        self.itemnum = int(view.item_emb.weight.shape[0] - 1)
        self._manifest = None
        self._popularity = None
        self._score_cache = {}
        if self.maxlen < 1:
            raise ValueError("Diagnostic maximum sequence length must be positive")

    @staticmethod
    def _seed(*parts):
        import hashlib

        payload = json.dumps([42, *parts], sort_keys=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else None

    def _freeze_panel(self):
        if self._manifest is not None:
            return self._manifest
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError(
                "user_train must contain identified ordered training histories"
            )
        normalized = {}
        popularity = Counter()
        for uid, history in histories.items():
            items = [int(x) for x in history]
            if any(x < 1 or x > self.itemnum for x in items):
                raise ValueError(
                    "Diagnostic histories must contain valid nonpadding item IDs"
                )
            stable_uid = str(uid)
            if stable_uid in normalized:
                raise ValueError(
                    "Ambiguous diagnostic user IDs after string conversion"
                )
            normalized[stable_uid] = items
            popularity.update(items)
        self._popularity = popularity
        users = sorted(
            (uid for uid, h in normalized.items() if len(h) >= 2),
            key=lambda uid: (self._seed(self.category, uid, "panel"), uid),
        )[:20]
        examples = []
        for uid in users:
            history = normalized[uid]
            excluded = set(history)
            available = self.itemnum - len(excluded)
            for j in range(max(1, len(history) - 5), len(history)):
                prefix = history[max(0, j - self.maxlen) : j]
                target = history[j]
                rng = random.Random(self._seed(self.category, uid, j, "candidates"))
                count = min(99, available)
                negatives = []
                seen = set()
                if available < self.itemnum // 4:
                    pool = [i for i in range(1, self.itemnum + 1) if i not in excluded]
                    negatives = rng.sample(pool, count)
                else:
                    while len(negatives) < count:
                        item = rng.randint(1, self.itemnum)
                        if item not in excluded and item not in seen:
                            seen.add(item)
                            negatives.append(item)
                examples.append(
                    {
                        "category": self.category,
                        "user_id": uid,
                        "transition_index": j,
                        "original_prefix_length": j,
                        "truncated_events": j - len(prefix),
                        "prefix": prefix,
                        "target": target,
                        "candidates": [target] + negatives,
                    }
                )
        self._manifest = examples
        return examples

    def analyze_embeddings_collapse_cosine(self, sample_size=2048, seed=42):
        with torch.no_grad():
            emb = self.view.item_emb.weight.detach()[1:]
            # CPU sampling avoids changing either training RNG stream.
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(emb), generator=g)[:sample_size].to(emb.device)
            x = emb.index_select(0, idx)
            x = x[x.norm(dim=-1) > 0]
            if len(x) < 2:
                return None
            x = F.normalize(x.float(), dim=-1)
            # Sum identity avoids materializing a quadratic Gram matrix.
            mean_cos = (
                (x.sum(0).square().sum() - x.square().sum()) / (len(x) * (len(x) - 1))
            ).clamp(-1, 1)
            return float((mean_cos.item() + 1.0) / 2.0)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches=10,
        eps=0.5,
        topk_categories=20,
    ):
        from itertools import islice

        was_training = self.model.training
        self.model.eval()
        margins = []
        try:
            for batch in islice(dataloader, num_batches):
                if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                    continue
                seq, pos, neg = [
                    torch.as_tensor(x, dtype=torch.long, device=self.device)
                    for x in batch[1:4]
                ]
                if seq.ndim != 2 or seq.shape != pos.shape or seq.shape != neg.shape:
                    raise ValueError(
                        "Expected aligned B x L sequence/positive/negative tensors"
                    )
                features = self.view.log2feats(seq)
                # Separate calls preserve the adapter's B x L scoring contract.
                positive_scores = self.view.predict_score(features, pos)
                negative_scores = self.view.predict_score(features, neg)
                if (
                    positive_scores.shape != pos.shape
                    or negative_scores.shape != neg.shape
                ):
                    raise ValueError("Loader diagnostic scores must have shape B x L")
                margin = positive_scores - negative_scores
                valid = (seq != 0) & (pos != 0) & (neg != 0)
                if valid.any():
                    values = margin[valid].detach().float().cpu()
                    if not torch.isfinite(values).all():
                        raise ValueError("Nonfinite diagnostic margins")
                    margins.append(values)
        finally:
            self.model.train(was_training)
        m = torch.cat(margins) if margins else torch.empty(0)
        result = {
            "count": m.numel(),
            "mean": float(m.mean()) if m.numel() else None,
            "p10": (
                float(torch.kthvalue(m, max(1, int(np.ceil(0.1 * m.numel())))).values)
                if m.numel()
                else None
            ),
            "neg_beats_pos_rate": float((m < 0).float().mean()) if m.numel() else None,
            "small_margin_rate": float((m < eps).float().mean()) if m.numel() else None,
        }
        return result, {
            "value": None,
            "reason": "Item categories are unavailable",
            "eps": eps,
        }

    @torch.no_grad()
    def _score(self, prefixes, candidates):
        if self.model.training:
            raise RuntimeError("Diagnostic scoring requires model.eval()")
        candidate_key = tuple(int(item) for item in candidates)
        if not candidate_key or len(set(candidate_key)) != len(candidate_key):
            raise ValueError("Diagnostic candidates must be nonempty and unique")
        if any(item < 1 or item > self.itemnum for item in candidate_key):
            raise ValueError("Diagnostic candidate ID outside the catalog")
        keys = []
        missing = {}
        for history in prefixes:
            h = tuple(int(item) for item in history[-self.maxlen :])
            if not h:
                raise ValueError("Cannot score an empty diagnostic history")
            if any(item < 1 or item > self.itemnum for item in h):
                raise ValueError("Diagnostic prefix ID outside the catalog")
            key = (h, candidate_key)
            keys.append(key)
            if key not in self._score_cache:
                missing[key] = h
        pending = list(missing)
        candidate_tensor = torch.tensor(
            candidate_key, dtype=torch.long, device=self.device
        )
        for start in range(0, len(pending), 64):
            batch_keys = pending[start : start + 64]
            seq = torch.zeros(
                (len(batch_keys), self.maxlen),
                dtype=torch.long,
                device=self.device,
            )
            for row, key in enumerate(batch_keys):
                h = missing[key]
                seq[row, -len(h) :] = torch.tensor(
                    h, dtype=torch.long, device=self.device
                )
            # Independent right-aligned prefixes use the final position L-1.
            feature = self.view.log2feats(seq)[:, -1, :]
            items = candidate_tensor.unsqueeze(0).expand(len(batch_keys), -1)
            scores = self.view.predict_score(feature.unsqueeze(1), items)
            if scores.shape != items.shape:
                raise ValueError("Diagnostic scoring must return B x candidates")
            values = scores.detach().float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite diagnostic candidate scores")
            for key, value in zip(batch_keys, values):
                self._score_cache[key] = value.copy()
        return [self._score_cache[key] for key in keys]

    @staticmethod
    def _ranking(scores, candidates):
        ids = np.asarray(candidates)
        order = np.lexsort((ids, -np.asarray(scores)))
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        wins = (len(ids) - rank) / (len(ids) - 1) if len(ids) > 1 else None
        return rank, wins, set(ids[order[: min(10, len(ids))]].tolist())

    def _effect(self, baseline, changed, candidates):
        br, bw, bt = self._ranking(baseline, candidates)
        cr, cw, ct = self._ranking(changed, candidates)
        return {
            "rank_delta": cr - br,
            "win_rate_delta": cw - bw if bw is not None else None,
            "top10_overlap": len(bt & ct) / min(10, len(candidates)),
        }

    def _controls(self, prefix, removed, seed):
        n, k = len(prefix), len(removed)
        mask = tuple(sorted(removed))

        def contiguous(indices):
            return indices[-1] - indices[0] + 1 == len(indices)

        def signature(indices):
            return sorted(
                (min(4, 5 * i // n), int(self._popularity[prefix[i]]).bit_length() - 1)
                for i in indices
            )

        rng = random.Random(seed)
        is_block = contiguous(mask)
        if is_block:
            proposals = [tuple(range(i, i + k)) for i in range(n - k + 1)]
            rng.shuffle(proposals)
        else:
            proposals = [tuple(sorted(rng.sample(range(n), k))) for _ in range(256)]
        desired = signature(mask)
        seen, controls = set(), []
        duplicate_proposals = 0
        for proposal in proposals:
            if proposal in seen:
                duplicate_proposals += 1
                continue
            seen.add(proposal)
            if (
                proposal != mask
                and contiguous(proposal) == is_block
                and signature(proposal) == desired
            ):
                controls.append(list(proposal))
        return controls[:10], {
            "requested": 10,
            "distinct_matches_found": len(controls),
            "distinct_controls_used": min(10, len(controls)),
            "duplicate_proposals": duplicate_proposals,
            "proposal_search_exhaustive": is_block,
            "unfilled_controls": max(0, 10 - len(controls)),
            "contrast_identified": bool(controls),
        }

    def analyze_sequence_interventions(self):
        panel = self._freeze_panel()
        names = [
            "recent",
            "repeated_support",
            "intersection",
            "recent_only",
            "repeat_only",
            "union",
            "older_shuffle",
        ]
        observations = {name: [] for name in names}
        records = []
        repeat_users, repeat_prefixes, repeated_events = set(), 0, 0
        for example in panel:
            # Scope cached scores to one example and the current model weights.
            # Identical interventions and controls reuse exactly the same scores.
            self._score_cache.clear()
            prefix, candidates = example["prefix"], example["candidates"]
            n = len(prefix)
            recent = set(range(n - max(1, n // 5), n))
            counts = Counter(prefix)
            repeated = {i for i, item in enumerate(prefix) if counts[item] >= 2}
            if repeated:
                repeat_users.add(example["user_id"])
                repeat_prefixes += 1
                repeated_events += len(repeated)
            masks = {
                "recent": recent,
                "repeated_support": repeated,
                "intersection": recent & repeated,
                "recent_only": recent - repeated,
                "repeat_only": repeated - recent,
                "union": recent | repeated,
            }
            baseline = self._score([prefix], candidates)[0]
            br, bw, _ = self._ranking(baseline, candidates)
            entry = dict(
                example,
                baseline_rank=br,
                baseline_win_rate=bw,
                candidate_count=len(candidates),
                interventions={},
            )
            for name in names:
                seed = self._seed(
                    self.category, example["user_id"], example["transition_index"], name
                )
                if name == "older_shuffle":
                    if n < 3:
                        entry["interventions"][name] = {
                            "eligible": False,
                            "reason": "Fewer than two older events",
                        }
                        continue
                    permutation = list(range(n - 1))
                    random.Random(seed).shuffle(permutation)
                    changed = [prefix[i] for i in permutation] + prefix[-1:]
                    removed, controls, matching = set(), [], None
                    if changed == prefix:
                        entry["interventions"][name] = {
                            "eligible": False,
                            "reason": "Permutation leaves item sequence unchanged",
                        }
                        continue
                else:
                    removed = masks[name]
                    if not removed or len(removed) == n:
                        entry["interventions"][name] = {
                            "eligible": False,
                            "reason": "Empty removal set or no surviving event",
                        }
                        continue
                    changed = [
                        item for i, item in enumerate(prefix) if i not in removed
                    ]
                    controls, matching = self._controls(prefix, removed, seed)
                histories = [changed] + [
                    [item for i, item in enumerate(prefix) if i not in set(mask)]
                    for mask in controls
                ]
                scores = self._score(histories, candidates)
                effect = self._effect(baseline, scores[0], candidates)
                control_effects = [
                    self._effect(baseline, s, candidates) for s in scores[1:]
                ]
                control_ranks = [e["rank_delta"] for e in control_effects]
                effect.update(
                    {
                        "removed_events": len(removed),
                        "retained_length": len(changed),
                        "removal_fraction": len(removed) / n,
                        "target_minus_control_rank_delta": (
                            effect["rank_delta"] - self._mean(control_ranks)
                            if control_ranks
                            else None
                        ),
                        "control_rank_delta_std": (
                            float(np.std(control_ranks)) if control_ranks else None
                        ),
                    }
                )
                observations[name].append((example["user_id"], effect))
                entry["interventions"][name] = {
                    "eligible": True,
                    "effects": effect,
                    "removed_indices": sorted(removed),
                    "control_masks": controls,
                    "matching": matching,
                    "control_effects": control_effects,
                    "permutation": permutation if name == "older_shuffle" else None,
                    "position_preserving_effect": None,
                    "position_contrast_reason": "Explicit positions and internal masking are not exposed by the compatibility contract",
                }
            records.append(entry)
        summaries = {}
        for name, rows in observations.items():
            by_user = {}
            for uid, effect in rows:
                by_user.setdefault(uid, []).append(effect)
            fields = (
                list(rows[0][1])
                if rows
                else [
                    "rank_delta",
                    "win_rate_delta",
                    "top10_overlap",
                    "removed_events",
                    "retained_length",
                    "removal_fraction",
                    "target_minus_control_rank_delta",
                    "control_rank_delta_std",
                ]
            )
            summaries[name] = {
                "eligible_prefixes": len(rows),
                "eligible_users": len(by_user),
                "total_prefixes": len(panel),
                "total_users": len({e["user_id"] for e in panel}),
                "user_averaged_effects": {},
                "users_defined_per_statistic": {},
            }
            for field in fields:
                user_values = []
                for effects in by_user.values():
                    values = [e[field] for e in effects if e[field] is not None]
                    if values:
                        user_values.append(self._mean(values))
                summaries[name]["user_averaged_effects"][field] = self._mean(
                    user_values
                )
                summaries[name]["users_defined_per_statistic"][field] = len(user_values)
        return {
            "category": self.category,
            "category_identified": self.category is not None,
            "summaries": summaries,
            "repeat_coverage": {
                "users": len(repeat_users),
                "prefixes": repeat_prefixes,
                "repeated_events_across_prefixes": repeated_events,
            },
            "four_category_macro": None,
            "macro_unavailable_reason": "This probe observes one view; all four identified category results are required",
            "manifest_and_prefix_results": records,
            "prospective_validation_comparison": None,
            "validation_comparison_reason": "No trained intervention trials or host validation outcomes are supplied to this probe",
        }

    def run_full_diagnosis(self, train_loader):
        # Loader iteration can consume global and explicit generator state even
        # when the model itself is evaluated without dropout.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        generators = {}
        for owner in (train_loader, getattr(train_loader, "sampler", None)):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator):
                generators[id(generator)] = (generator, generator.get_state())
        # Preserve mixed child-module modes as well as the root training flag.
        module_modes = [(module, module.training) for module in self.model.modules()]
        self._score_cache.clear()
        try:
            self._freeze_panel()
            self.model.eval()
            with torch.no_grad():
                collapse = self.analyze_embeddings_collapse_cosine()
                margin, _ = self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader
                )
                interventions = self.analyze_sequence_interventions()
            return {
                "metrics": {
                    "embedding_collapse_score": collapse,
                    "pos_neg_margin_mean": margin["mean"],
                    "pos_neg_margin_p10": margin["p10"],
                    "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin["small_margin_rate"],
                    "pos_neg_valid_positions": margin["count"],
                    "sequence_interventions": interventions,
                },
                "metric_definitions": {
                    "embedding_collapse_score": "(Mean off-diagonal item cosine + 1)/2, excluding padding and zero vectors, at most 2048 seeded samples. Values near .5 indicate mean cosine near zero; this is anisotropy evidence, not a recommendation-diversity measurement. Null with fewer than two nonzero samples.",
                    "pos_neg_margin_mean": "Mean positive-minus-loader-negative score at aligned original training positions, with sequence, positive and negative all nonzero. Scale dependent; not fixed-candidate ranking performance.",
                    "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of valid loader margins; null if none.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid loader margins strictly below zero; null if none.",
                    "pos_neg_small_margin_rate": "Fraction of valid loader margins below 0.5; scale dependent; null if none.",
                    "pos_neg_valid_positions": "Number of valid loader positions from at most ten batches. No user identity is inferred from loader rows.",
                    "sequence_interventions": {
                        "scope": "Training-prefix diagnostics only, not host validation or causal intent estimates. No review, rating, category metadata or simulator traits are used. No training or inference architecture is changed.",
                        "panel": "At most 20 identified users per supplied view, chosen by SHA256(seed 42, category, string user ID); up to five latest observed transitions per user. h[:j] predicts h[j], truncated to the encoder maximum length. The complete deterministic manifest is returned for retention and comparison across trials; inputs and protocol must remain fixed.",
                        "candidates": "Target first plus up to 99 unique uniformly sampled catalog negatives, excluding every known training-history item. A repeated target remains an explicit positive exception. Candidates depend only on identified example and seed 42, and are reused across interventions and controls. No held-out histories are consulted.",
                        "protocol": "Independent nonempty prefixes are right aligned and scored at L-1 through view.log2feats and view.predict_score. Loader margins instead use original training positions. Deletions repack survivors and change absolute positions. Causal and padding behavior is inherited from the actual encoder; internal holes are never introduced.",
                        "rank_delta": "Perturbed minus baseline one-based target rank; positive means worse. Rank orders by descending score then ascending item ID, including ties.",
                        "win_rate_delta": "Perturbed minus baseline fraction of negative candidates ranked below the target using the same tie rule; null without negatives.",
                        "top10_overlap": "Intersection of baseline and perturbed top min(10,candidate_count) sets divided by min(10,candidate_count).",
                        "interventions": "Recent deletes the latest max(1,floor(.2*n)) events. Repeated support deletes all occurrences of item IDs appearing at least twice in the observed truncated prefix, independently of target. Intersection, exclusive sets and union are measured separately, never added. Older shuffle permutes all but the latest event and preserves length and multiset. Empty/unchanged interventions abstain with reasons.",
                        "controls": "Up to ten distinct deterministic deletion masks matched exactly on removal count, contiguous-versus-noncontiguous structure and multiset of (five recency bins, floor(log2(training item count))) strata. All contiguous placements or 256 random noncontiguous proposals are searched. No matching relaxation. Returned masks, distinct counts, duplicate proposals and unfilled counts expose coverage; no alternatives means unidentified content-versus-position contrast.",
                        "target_minus_control_rank_delta": "Targeted rank delta minus the mean matched-control rank delta for the same example. Null without controls. Control standard deviation uses population normalization and may be zero with one control; consult distinct-controls-used counts.",
                        "aggregation": "Average defined prefix statistics within identified users, then average users equally in the supplied category. Coverage and users defined per statistic accompany results. A four-category macro requires all four identified category statistics; this single-view probe returns null. Repeat-event counts sum across overlapping prefixes and are not distinct interaction counts.",
                        "limitations": "Deletion effects combine content, length, query and absolute-position changes. Explicit position-preserving contrasts are unavailable through this adapter. Shuffle sensitivity alone does not identify older-context utility. Exact repeat absence is unavailable evidence, not absence of semantic interests. Diagnostic effects do not establish improvement in host HR@10; prospective trained comparisons and uncertainty estimates remain unavailable until actually measured. Global RNG states and exposed loader/sampler generators are restored, but private worker or custom iterator state cannot be rewound; use a separate diagnostic loader. The identified-prefix panel and fixed candidates remain independent of loader sampling.",
                    },
                },
            }
        finally:
            self._score_cache.clear()
            for module, training in module_modes:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)
            for generator, state in generators.values():
                generator.set_state(state)


### <<< Self_EvolveRec-BLOCK-END
