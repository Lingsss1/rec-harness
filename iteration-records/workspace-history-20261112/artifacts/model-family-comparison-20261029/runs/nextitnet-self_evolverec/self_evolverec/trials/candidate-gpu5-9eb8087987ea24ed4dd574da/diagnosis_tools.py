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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret controlled training-prefix sensitivity without generalization claims
    - Every statement must be grounded in the provided metric values/definitions.
    - Training-prefix ranks, HR@10, margins and interventions measure in-sample fit or sensitivity, not held-out generalization or user preferences.
    - Read matching failures, sample counts, repeated targets, positional shifts and structural reachability before interpreting recency effects.
    - User-bootstrap intervals describe sampled-user uncertainty within this category; do not invent a four-category macro interval.
    - Null values and empty category breakdowns mean unavailable evidence. Do not infer ratings, review text, item categories or simulator traits.
    - Embedding cosine alone cannot establish collapse, diversity, or recommendation quality.
    - The unchanged pinned host macro validation HR@10 remains authoritative for model selection.
    - Subsequent model changes require a preregistered hypothesis and one jointly trained NextItNet per category, with matching checkpoint and predictor architecture.
    - With absent or inconclusive evidence, state the limitation without asserting a critical failure or healthy generalization.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the ModelView contract and exclude unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this benchmark.
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
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing embedding evidence from a measured value
            if n <= 2:
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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained NextItNet module
        self.network.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on an invalid diagnostic loader contract
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score aligned causal NextItNet features without an incompatible candidate axis
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Expected B x L sequence, positive and negative tensors"
                )
            log_feats = self.model.log2feats(seq_t)
            margin = self.model.predict_score(
                log_feats, pos_t
            ) - self.model.predict_score(log_feats, neg_t)
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Reuse the measured position eligibility mask
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
            ### >>> Self_EvolveRec-BLOCK-START: Report absent margin observations explicitly
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

    ### >>> Self_EvolveRec-BLOCK-START: Correct exact-rank eligibility, active-token probes, repacking shifts and diagnostic accounting
    @staticmethod
    def _ahead(scores, ids, target_score, target):
        return (scores > target_score) | ((scores == target_score) & (ids < target))

    def _validate_rank(self):
        # Deliberately unsorted IDs, tied scores, and multiple chunk boundaries.
        ids = np.array([7, 2, 9, 1, 5], dtype=np.int64)
        scores = np.array([0.0, 2.0, 2.0, -1.0, 2.0])
        exhaustive = ids[np.lexsort((ids, -scores))].tolist()
        for target, score in zip(ids, scores):
            for chunk in (1, 2, 3, 8):
                rank = 1 + sum(
                    int(
                        self._ahead(
                            scores[j : j + chunk], ids[j : j + chunk], score, target
                        ).sum()
                    )
                    for j in range(0, len(ids), chunk)
                )
                if rank != exhaustive.index(target) + 1:
                    raise AssertionError("Chunked rank tie-breaking failed")
        return True

    def _pack(self, history):
        length = int(self.network.maxlen)
        history = history[-length:]
        if not history:
            raise ValueError("Empty diagnostic prefix")
        seq = torch.zeros((1, length), device=self.device, dtype=torch.long)
        seq[0, -len(history) :] = torch.tensor(history, device=self.device)
        return seq

    def _feature(self, history):
        seq = self._pack(history)
        positions = torch.arange(seq.size(1), device=self.device)
        last = positions.masked_fill(seq[0] == 0, -1).max()
        if int(last) < 0:
            raise ValueError("No nonpadding diagnostic position")
        return self.model.log2feats(seq)[0, last]

    def _scores(self, feature, ids):
        items = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        scores = self.model.predict_score(feature.expand(items.numel(), -1), items)
        if scores.shape != items.shape or not torch.isfinite(scores).all():
            raise ValueError("Invalid NextItNet diagnostic scores")
        return scores.detach().double().cpu().numpy()

    def _catalog_result(self, history, eligible, target, sampled, chunk_size):
        eligible = np.asarray(eligible, dtype=np.int64)
        if (
            chunk_size < 1
            or eligible.ndim != 1
            or len(eligible) == 0
            or np.any(eligible < 1)
            or np.any(eligible > int(self.network.itemnum))
            or np.any(eligible[1:] <= eligible[:-1])
            or np.count_nonzero(eligible == target) != 1
        ):
            raise ValueError(
                "Expected positive chunk size and sorted unique eligible catalog IDs containing target once"
            )
        sampled = np.asarray(sampled, dtype=np.int64)
        if (
            sampled.ndim != 1
            or np.any(sampled == target)
            or len(np.unique(sampled)) != len(sampled)
            or not np.isin(sampled, eligible).all()
        ):
            raise ValueError(
                "Sampled negatives must be distinct eligible non-target IDs"
            )
        feature = self._feature(history)
        target_score = float(self._scores(feature, [target])[0])
        rank, top_ids, top_scores = 1, np.empty(0, dtype=np.int64), np.empty(0)
        for start in range(0, len(eligible), chunk_size):
            ids = eligible[start : start + chunk_size]
            scores = self._scores(feature, ids)
            # Use one canonical target score even if batch-shaped reductions differ.
            scores[ids == target] = target_score
            rank += int(
                ((ids != target) & self._ahead(scores, ids, target_score, target)).sum()
            )
            top_ids = np.concatenate((top_ids, ids))
            top_scores = np.concatenate((top_scores, scores))
            order = np.lexsort((top_ids, -top_scores))[:10]
            top_ids, top_scores = top_ids[order], top_scores[order]
        sampled_scores = self._scores(feature, sampled) if len(sampled) else np.empty(0)
        return {
            "rank": rank,
            "hr10": float(rank <= 10),
            "sampled_margin": (
                float(target_score - sampled_scores.mean()) if len(sampled) else None
            ),
            "top10": top_ids.tolist(),
        }

    def _architecture(self, history):
        offsets = {0}
        convs = []
        for block in self.network.blocks:
            branch = set(offsets)
            for conv in (block.conv1, block.conv2):
                taps = {i * conv.dilation[0] for i in range(conv.kernel_size[0])}
                branch = {a + b for a in branch for b in taps}
                convs.append(
                    {
                        "kernel": int(conv.kernel_size[0]),
                        "dilation": int(conv.dilation[0]),
                        "bias": conv.bias is not None,
                        "internal_padding": list(conv.padding),
                    }
                )
            offsets |= branch
            if hasattr(block, "gate"):
                # The gate also receives the raw causal embedding mean.
                offsets |= set(range(int(self.network.recent_window)))
        seq = self._pack(history)
        original = self.model.log2feats(seq)
        changed = seq.clone()
        active_length = min(len(history), seq.size(1))
        # Probe inside the active prefix rather than comparing padding-only outputs.
        pivot = seq.size(1) - active_length + (active_length - 1) // 2
        suffix = changed[:, pivot + 1 :]
        changed[:, pivot + 1 :] = 1 + suffix.remainder(int(self.network.itemnum))
        changed_tokens = int((changed != seq).sum())
        altered = self.model.log2feats(changed)
        causal_error = float(
            (original[:, : pivot + 1] - altered[:, : pivot + 1]).abs().max()
        )
        empty = torch.zeros_like(seq)
        empty_features = self.model.log2feats(empty)
        padding_norm = float(empty_features.norm(dim=-1).max())
        padding_shift = None
        # Moving a full window left would discard its first token and confound
        # position sensitivity with deletion, so report that probe as unavailable.
        if 2 <= active_length < seq.size(1):
            padded = seq.clone()
            padded[:, -1] = 0
            padded[:, :-1] = seq[:, 1:]
            moved = self.model.log2feats(padded)
            padding_shift = float((original[0, -1] - moved[0, -2]).norm())
        report = {
            "convolutions": convs,
            "structural_reachable_offsets": sorted(offsets),
            "structural_receptive_span": max(offsets) + 1,
            "normalizations": [
                type(m).__name__
                for m in self.network.modules()
                if isinstance(m, torch.nn.LayerNorm)
            ],
            "position_embeddings": hasattr(self.network, "pos_emb"),
            "mask_after_each_block": True,
            "final_normalization_can_reintroduce_padding": True,
            "causality_max_abs_error": causal_error if changed_tokens else None,
            "causality_changed_tokens": changed_tokens,
            "causality_pivot_index": pivot,
            "all_padding_feature_max_norm": padding_norm,
            "same_tokens_one_position_left_feature_l2": padding_shift,
            "position_shift_unavailable_reason": (
                None
                if padding_shift is not None
                else "Requires at least two active tokens and one leading padding position"
            ),
            "readout": "maximum index with sequence != 0; empty prefixes excluded",
            "replacement_probe": None,
            "replacement_probe_reason": "Not preregistered: positional embeddings and padding can change features",
            "scope": "Structural dependency offsets of the supplied gated NextItNet, not measured importance",
        }
        return report, offsets

    @staticmethod
    def _aggregate(rows, bootstrap=400, seed=42):
        rng = np.random.default_rng(seed)
        result = {}
        for variant in sorted({r["variant"] for r in rows}):
            selected = [r for r in rows if r["variant"] == variant]
            by_user = {}
            for row in selected:
                by_user.setdefault(row["user"], []).append(row)
            entry = {"users": len(by_user), "prefixes": len(selected)}
            for metric in (
                "delta_rank",
                "delta_hr10",
                "top10_overlap",
                "delta_sampled_margin",
            ):
                values = []
                for records in by_user.values():
                    observed = [r[metric] for r in records if r.get(metric) is not None]
                    if observed:
                        values.append(float(np.mean(observed)))
                if not values:
                    entry[metric] = {"mean": None, "ci95": None}
                    continue
                values = np.asarray(values)
                interval = None
                if len(values) >= 2:
                    draws = values[
                        rng.integers(0, len(values), size=(bootstrap, len(values)))
                    ].mean(axis=1)
                    interval = np.quantile(draws, [0.025, 0.975]).tolist()
                entry[metric] = {"mean": float(values.mean()), "ci95": interval}
            result[variant] = entry
        return result

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        import time

        protocol = {
            "seed": 42,
            "source": "training prefixes and next training items only",
            "max_users": 12,
            "prefixes_per_user": 2,
            "minimum_prefix_length": 5,
            "history_window": int(self.network.maxlen),
            "deletion_count": 1,
            "control_repetitions": 2,
            "bootstrap_repetitions": 400,
            "catalog_chunk_size": 4096,
            "sampled_margin_negatives": 99,
            "frequency_strata": "training occurrence counts: 1, 2-4, 5-15, >=16",
            "lag_bins": "nonterminal distance from terminal: 1-4, 5-15, >=16",
            "recent_older_rule": "most recent match versus an older-half match in the same frequency stratum",
            "random_control_rule": "distinct singleton deletions in the same frequency and lag stratum; run length exactly one",
            "permutations": ["full window", "terminal preserving"],
            "replacement_rule": "disabled; no new mask token or PAD replacement",
            "candidate_rule": "IDs 1..itemnum excluding original full-prefix observations except target",
            "tie_break": "ascending item ID",
            "eps": 0.5,
        }
        print(
            "Diagnostic protocol: " + json.dumps(protocol, sort_keys=True), flush=True
        )
        states = [(m, m.training) for m in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
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
        if cuda_devices:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                self.network.eval()
                return self._run_protocol(train_loader, protocol, started)
        finally:
            for module, training in states:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _run_protocol(self, train_loader, protocol, started):
        import time

        rng = random.Random(protocol["seed"])
        self._validate_rank()
        source = self.user_train.get("History", self.user_train)
        if not isinstance(source, dict):
            raise ValueError("ModelView user_train must contain training histories")
        itemnum = int(self.network.itemnum)
        histories = {}
        coverage = Counter(
            {
                "empty_users": 0,
                "invalid_id_users": 0,
                "out_of_catalog_users": 0,
                "eligible_users": 0,
                "short_users": 0,
                "selected_users": 0,
                "sampled_prefixes": 0,
                "short_window_prefixes": 0,
                "no_eligible_negatives": 0,
                "recent_older_matching_failures": 0,
                "recent_control_matching_failures": 0,
                "older_control_matching_failures": 0,
                "scored_variants": 0,
            }
        )
        for user in sorted(source, key=str):
            history = list(source[user])
            if not history:
                coverage["empty_users"] += 1
                continue
            try:
                history = [int(i) for i in history]
            except (TypeError, ValueError):
                coverage["invalid_id_users"] += 1
                continue
            if any(i < 1 or i > itemnum for i in history):
                coverage["out_of_catalog_users"] += 1
                continue
            histories[str(user)] = history
        frequency = Counter(i for h in histories.values() for i in h)
        transitions = Counter(
            (a, b) for h in histories.values() for a, b in zip(h, h[1:])
        )
        users = sorted(
            u
            for u, h in histories.items()
            if len(h) > protocol["minimum_prefix_length"]
        )
        coverage["eligible_users"] = len(users)
        coverage["short_users"] = len(histories) - len(users)
        if len(users) > protocol["max_users"]:
            users = sorted(rng.sample(users, protocol["max_users"]))
        coverage["selected_users"] = len(users)
        rows, baseline_rows = [], []
        architecture, reachable = None, set()

        def freq_bin(item):
            return int(np.searchsorted([1, 4, 15], frequency[item], side="left"))

        def lag_bin(index, n):
            return int(np.searchsorted([4, 15], n - 1 - index, side="left"))

        for user in users:
            history = histories[user]
            cuts = list(range(protocol["minimum_prefix_length"], len(history)))
            cuts = sorted(
                rng.sample(cuts, min(len(cuts), protocol["prefixes_per_user"]))
            )
            for cut in cuts:
                coverage["sampled_prefixes"] += 1
                full_prefix, target = history[:cut], history[cut]
                prefix = full_prefix[-protocol["history_window"] :]
                if len(prefix) < 5:
                    coverage["short_window_prefixes"] += 1
                    continue
                if architecture is None:
                    architecture, reachable = self._architecture(prefix)
                excluded = set(full_prefix) - {target}
                eligible = np.fromiter(
                    (i for i in range(1, itemnum + 1) if i not in excluded),
                    dtype=np.int64,
                )
                negatives = [int(i) for i in eligible if i != target]
                if not negatives:
                    coverage["no_eligible_negatives"] += 1
                    continue
                sampled = rng.sample(
                    negatives, min(len(negatives), protocol["sampled_margin_negatives"])
                )
                baseline = self._catalog_result(
                    prefix, eligible, target, sampled, protocol["catalog_chunk_size"]
                )
                n = len(prefix)
                strata = {
                    "prefix_length": (
                        "5-15" if n <= 15 else "16-63" if n <= 63 else ">=64"
                    ),
                    "target_popularity": str(freq_bin(target)),
                    "transition_frequency": (
                        "0"
                        if transitions[(prefix[-1], target)] == 0
                        else "1" if transitions[(prefix[-1], target)] == 1 else ">=2"
                    ),
                    "repeated_target": str(target in full_prefix),
                }
                baseline_rows.append(
                    {
                        "user": user,
                        "cut": cut,
                        "rank": baseline["rank"],
                        "hr10": baseline["hr10"],
                        "eligible_items": len(eligible),
                        **strata,
                    }
                )
                variants = []
                recent_indices = list(range(n - 2, (n - 1) // 2 - 1, -1))
                older_indices = list(range((n - 1) // 2))
                pair = next(
                    (
                        (r, o)
                        for r in recent_indices
                        for o in reversed(older_indices)
                        if freq_bin(prefix[r]) == freq_bin(prefix[o])
                    ),
                    None,
                )
                if pair is None:
                    coverage["recent_older_matching_failures"] += 1
                else:
                    controls = {}
                    for name, index in zip(("recent", "older"), pair):
                        pool = [
                            j
                            for j in range(n - 1)
                            if j != index
                            and freq_bin(prefix[j]) == freq_bin(prefix[index])
                            and lag_bin(j, n) == lag_bin(index, n)
                        ]
                        if len(pool) < protocol["control_repetitions"]:
                            coverage[name + "_control_matching_failures"] += 1
                            continue
                        controls[name] = rng.sample(
                            pool, protocol["control_repetitions"]
                        )
                    # Keep paired recency comparisons on exactly the same prefixes.
                    if len(controls) == 2:
                        for name, index in zip(("recent", "older"), pair):
                            variants.append(
                                (name, prefix[:index] + prefix[index + 1 :], index)
                            )
                            for k, j in enumerate(controls[name]):
                                variants.append(
                                    (
                                        name + "_control_" + str(k),
                                        prefix[:j] + prefix[j + 1 :],
                                        j,
                                    )
                                )
                shuffled = prefix.copy()
                rng.shuffle(shuffled)
                variants.append(("full_permutation", shuffled, None))
                shuffled = prefix[:-1]
                rng.shuffle(shuffled)
                variants.append(
                    ("terminal_preserving_permutation", shuffled + prefix[-1:], None)
                )
                local = {}
                for name, changed, deleted in variants:
                    result = self._catalog_result(
                        changed,
                        eligible,
                        target,
                        sampled,
                        protocol["catalog_chunk_size"],
                    )
                    shifts, membership_changes = [], 0
                    if deleted is not None:
                        for old in range(n):
                            if old == deleted:
                                continue
                            new = old - int(old > deleted)
                            # Repacking increases leading padding by one. Absolute
                            # model positions shift +1 before deletion and 0 after it.
                            shifts.append(1 + new - old)
                            membership_changes += int(
                                ((n - 1 - old) in reachable)
                                != ((n - 2 - new) in reachable)
                            )
                    row = {
                        "user": user,
                        "cut": cut,
                        "variant": name,
                        **strata,
                        "rank": result["rank"],
                        "hr10": result["hr10"],
                        "delta_rank": result["rank"] - baseline["rank"],
                        "delta_hr10": result["hr10"] - baseline["hr10"],
                        "delta_sampled_margin": result["sampled_margin"]
                        - baseline["sampled_margin"],
                        "top10_overlap": len(
                            set(result["top10"]) & set(baseline["top10"])
                        )
                        / len(baseline["top10"]),
                        "retained_token_mean_position_shift": (
                            float(np.mean(shifts)) if shifts else None
                        ),
                        "retained_token_reachable_membership_changes": (
                            membership_changes if deleted is not None else None
                        ),
                        "deleted_index": deleted,
                        "unchanged_history": changed == prefix,
                    }
                    rows.append(row)
                    local[name] = row
                    coverage["scored_variants"] += 1
                for name in ("recent", "older"):
                    if name not in local:
                        continue
                    adjusted = dict(local[name], variant=name + "_control_adjusted")
                    for metric in ("delta_rank", "delta_hr10", "delta_sampled_margin"):
                        adjusted[metric] -= float(
                            np.mean(
                                [
                                    local[name + "_control_" + str(k)][metric]
                                    for k in range(protocol["control_repetitions"])
                                ]
                            )
                        )
                    adjusted["top10_overlap"] = None
                    rows.append(adjusted)
                if "recent" in local and "older" in local:
                    # A direct paired recent-minus-older contrast on matched prefixes.
                    contrast = dict(local["recent"], variant="recent_minus_older")
                    for metric in ("delta_rank", "delta_hr10", "delta_sampled_margin"):
                        contrast[metric] -= local["older"][metric]
                    contrast["top10_overlap"] = None
                    rows.append(contrast)

        collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10
        )
        breakdown["availability"] = (
            "Item categories unavailable; category lists intentionally empty"
        )
        strata_results = {}
        for field in (
            "prefix_length",
            "target_popularity",
            "transition_frequency",
            "repeated_target",
        ):
            strata_results[field] = {
                value: self._aggregate(
                    [r for r in rows if r[field] == value],
                    bootstrap=protocol["bootstrap_repetitions"],
                    seed=protocol["seed"],
                )
                for value in sorted({r[field] for r in rows})
            }
        baseline_users = {}
        for row in baseline_rows:
            baseline_users.setdefault(row["user"], []).append(row["hr10"])
        intervention_effects = self._aggregate(
            rows,
            bootstrap=protocol["bootstrap_repetitions"],
            seed=protocol["seed"],
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        metrics = {
            "protocol": protocol,
            "coverage": dict(coverage),
            "architecture": architecture,
            "chunked_rank_synthetic_check": True,
            "embedding_collapse_score": collapse,
            "pos_neg_margin_mean": margin["mean"],
            "pos_neg_margin_p10": margin["p10"],
            "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
            "pos_neg_small_margin_rate": margin["small_margin_rate"],
            "pos_neg_margin_category_breakdown": breakdown,
            "training_prefix_hr10_user_mean": (
                float(np.mean([np.mean(v) for v in baseline_users.values()]))
                if baseline_users
                else None
            ),
            "intervention_effects": intervention_effects,
            "strata": strata_results,
            "baseline_observations": baseline_rows,
            "paired_observations": rows,
            "four_category_macro": None,
            "four_category_macro_reason": "This call contains one ModelView; four category user samples are required for a macro estimate and independently stratified bootstrap",
            "category_breakdown": {},
            "unavailable": [
                "review ratings/text",
                "item ratings/categories/prices/stores",
                "simulator diversity/conformity traits",
                "held-out generalization effects",
                "replacement effects",
            ],
            "diagnostic_cost": {
                "synchronized_elapsed_seconds": elapsed,
                "device": str(self.device),
                "gpu_compute_seconds": None,
                "gpu_reserved_elapsed_seconds_upper_bound": (
                    elapsed if self.device.type == "cuda" else 0.0
                ),
                "accounting": "Single-device synchronized elapsed time includes CPU work; host GPU budget accounting remains authoritative",
            },
        }
        definitions = {
            "protocol": "Preregistered sampling and matching rules printed before execution; no model fitting or checkpoint changes.",
            "coverage": "Explicit counts of sampled prefixes, scored variants and exclusion/matching failures; unmatched comparisons are absent, never zero effects.",
            "architecture": "Structural offsets include every causal convolution and gated causal-summary dependency. Numerical causality and padding checks are local checks, not proofs for all inputs.",
            "chunked_rank_synthetic_check": "Rank counter and ascending-ID tie handling checked against exhaustive sorting across synthetic chunk sizes.",
            "embedding_collapse_score": "Mean sampled off-diagonal item-embedding cosine mapped from [-1,1] to [0,1]. 0.5 means mean cosine zero; this alone does not measure recommendation diversity or generalization.",
            "pos_neg_margin_mean": "Mean aligned positive-minus-sampled-negative score on valid training-loader positions; training fit only.",
            "pos_neg_margin_p10": "Empirical lower tenth percentile of valid training-loader score margins; null when absent.",
            "pos_neg_neg_beats_pos_rate": "Fraction of valid training-loader positions with negative margin.",
            "pos_neg_small_margin_rate": "Fraction of valid training-loader positions with margin below protocol eps.",
            "pos_neg_margin_category_breakdown": "Counts of low/wrong-margin training positions; item category lists are empty because categories are unavailable.",
            "training_prefix_hr10_user_mean": "Exact eligible-catalog HR@10, averaged within user then across sampled users. Targets are training interactions, so this is in-sample diagnostic fit, not host validation HR@10.",
            "intervention_effects": "Paired changed-minus-original ranks, HR@10 and mean sampled-negative margins; positive delta_rank means worse target rank. Top-10 overlap is intersection size divided by original top-10 size. Effects average prefixes within user, then users. CI95 resamples those user means with replacement; null below two users. Control-adjusted deltas subtract corresponding matched-control means; recent_minus_older compares matched deletion effects directly.",
            "strata": "Same user-weighted effects within fixed prefix-length, training target-frequency, training terminal-to-target transition-frequency and repeated-target strata. Sparse intervals and many comparisons require caution.",
            "baseline_observations": "Original-prefix ranks, HR@10, eligible-catalog size and measured strata. Eligibility uses all earlier training observations, including those outside the model window; target retained once even when repeated.",
            "paired_observations": "Auditable paired effects. Deletion position shifts are absolute model-input index changes after left-padded repacking: +1 before the deleted token and 0 after it. Reachable-membership changes compare each retained token's lag from the terminal position against structural offsets; permutation correspondence is unavailable.",
            "four_category_macro": "Unavailable from a single category call; never substitute a single-category estimate or pool category users as a macro estimate.",
            "category_breakdown": "Empty because item-category metadata is unavailable; dataset-level routing is unchanged.",
            "unavailable": "Unobserved data and unmeasured effects must not be inferred as measurements.",
            "diagnostic_cost": "Measured synchronized elapsed time; GPU compute time is not separately measured. Host accounting determines consumed search budget.",
        }
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
