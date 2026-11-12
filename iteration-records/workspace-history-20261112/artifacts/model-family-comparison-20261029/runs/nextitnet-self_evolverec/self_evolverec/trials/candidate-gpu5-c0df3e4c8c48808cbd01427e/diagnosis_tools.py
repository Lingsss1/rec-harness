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
    - Separate original fixed-catalog effects from restricted-catalog effects removing the same inserted IDs from all paired histories. Restricted ranks are not full-catalog ranks.
    - Report inserted-item score/rank gains separately from target sensitivity; neither identifies causal user intent.
    - Frequency and possible-reachability matching do not establish equal embedding displacement, exact path counts or behavioral plausibility. Training transition evidence is only a corruption constraint.
    - Interpret paired recent-minus-older effects only on successful common-support blocks. Do not infer monotonic dose response, calibrated false-alarm rates or bootstrap coverage from these probes.
    - Corruption Monte Carlo error and exploratory user-bootstrap uncertainty measure different sources of variation; additional corruption draws do not increase independent user counts.
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

    ### >>> Self_EvolveRec-BLOCK-START: Validate chunked ranks, freeze prefix sampling and reduce diagnostic overhead
    @staticmethod
    def _ahead(scores, ids, target_score, target):
        return (scores > target_score) | ((scores == target_score) & (ids < target))

    def _validate_rank(self):
        ids = np.array([7, 2, 9, 1, 5], dtype=np.int64)
        scores = np.array([0.0, 2.0, 2.0, -1.0, 2.0])
        expected = ids[np.lexsort((ids, -scores))].tolist()
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
                if rank != expected.index(target) + 1:
                    raise AssertionError("Chunked rank/tie control failed")
        return True

    def _pack(self, history):
        length = int(self.network.maxlen)
        history = history[-length:]
        if not history:
            raise ValueError("Empty diagnostic prefix")
        seq = torch.zeros((1, length), device=self.device, dtype=torch.long)
        seq[0, -len(history) :] = torch.as_tensor(history, device=self.device)
        return seq

    def _feature(self, history):
        seq = self._pack(history)
        positions = torch.arange(seq.size(1), device=self.device)
        last = positions.masked_fill(seq[0] == 0, -1).max()
        if int(last) < 0:
            raise ValueError("All-padding diagnostic input")
        return self.model.log2feats(seq)[0, last]

    def _scores(self, feature, ids):
        items = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        scores = self.model.predict_score(feature.expand(items.numel(), -1), items)
        if scores.shape != items.shape or not torch.isfinite(scores).all():
            raise ValueError("Invalid diagnostic scores")
        return scores.detach().double().cpu().numpy()

    def _catalog_scores(self, history, ids, chunk):
        # Only one feature computation and chunk-bounded GPU scoring per history.
        ids = np.asarray(ids, dtype=np.int64)
        if (
            chunk < 1
            or ids.ndim != 1
            or len(ids) == 0
            or np.any(ids < 1)
            or np.any(ids > int(self.network.itemnum))
            or np.any(ids[1:] <= ids[:-1])
        ):
            raise ValueError(
                "Expected positive chunk size and sorted unique in-catalog IDs"
            )
        feature = self._feature(history)
        scores = np.empty(len(ids), dtype=np.float64)
        for start in range(0, len(ids), chunk):
            scores[start : start + chunk] = self._scores(
                feature, ids[start : start + chunk]
            )
        return scores

    def _result(self, ids, scores, target, sampled, removed=(), include_top10=True):
        ids = np.asarray(ids, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        if (
            ids.ndim != 1
            or scores.shape != ids.shape
            or len(ids) == 0
            or np.any(ids < 1)
            or np.any(ids > int(self.network.itemnum))
            or np.any(ids[1:] <= ids[:-1])
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Invalid sorted catalog IDs or aligned finite scores")
        removed = set(removed)
        keep = ~np.isin(ids, list(removed)) if removed else slice(None)
        catalog, values = ids[keep], scores[keep]
        index = int(np.searchsorted(catalog, target))
        if index == len(catalog) or int(catalog[index]) != target:
            raise ValueError("Target missing from diagnostic catalog")
        score = float(values[index])
        rank = 1
        top_ids, top_scores = np.empty(0, dtype=np.int64), np.empty(0)
        for start in range(0, len(catalog), 4096):
            chunk_ids = catalog[start : start + 4096]
            chunk_scores = values[start : start + 4096]
            rank += int(self._ahead(chunk_scores, chunk_ids, score, target).sum())
            if include_top10:
                # Retain each chunk's top ten, including ascending-ID tie order.
                # Every global top-ten item must be in its own chunk's top ten.
                order = np.lexsort((chunk_ids, -chunk_scores))[:10]
                top_ids = np.concatenate((top_ids, chunk_ids[order]))
                top_scores = np.concatenate((top_scores, chunk_scores[order]))
                order = np.lexsort((top_ids, -top_scores))[:10]
                top_ids, top_scores = top_ids[order], top_scores[order]
        negatives = np.asarray([i for i in sampled if i not in removed], dtype=np.int64)
        if (
            negatives.ndim != 1
            or len(np.unique(negatives)) != len(negatives)
            or np.any(negatives == target)
            or not np.isin(negatives, catalog).all()
        ):
            raise ValueError("Sampled negatives must be unique eligible non-target IDs")
        margin = None
        if len(negatives):
            margin = float(score - values[np.searchsorted(catalog, negatives)].mean())
        return {
            "rank": rank,
            "hr10": float(rank <= 10),
            "score": score,
            "margin": margin,
            "top10": top_ids.tolist() if include_top10 else None,
            "catalog_size": len(catalog),
            "margin_negatives": len(negatives),
        }

    def _architecture(self, history):
        import inspect

        # Verify the supplied forward implementations before using their graph.
        # Otherwise numerical controls remain available, but reachability is unknown.
        offsets, convs, reason = {0}, [], None
        try:
            encode = inspect.getsource(type(self.network).encode_sequence)
            block_code = [
                inspect.getsource(type(b).forward) for b in self.network.blocks
            ]
            known = (
                type(self.network).__name__ == "NextItNetModel"
                and "summary = (embedding_sum / valid_count" in encode
                and "pos = valid.long().cumsum" in encode
                and "return self.norm(x) * mask" in encode
                and all(
                    "self.gate(summary)" in s and "residual + gate * y" in s
                    for s in block_code
                )
                and all(
                    isinstance(b.norm1, torch.nn.LayerNorm)
                    and isinstance(b.norm2, torch.nn.LayerNorm)
                    and tuple(b.norm1.normalized_shape) == (self.network.hidden_units,)
                    and tuple(b.norm2.normalized_shape) == (self.network.hidden_units,)
                    for b in self.network.blocks
                )
            )
            if not known:
                raise ValueError(
                    "Forward implementation differs from supplied gated NextItNet"
                )
            for block in self.network.blocks:
                branch = set(offsets)
                for conv in (block.conv1, block.conv2):
                    if (
                        not isinstance(conv, torch.nn.Conv1d)
                        or conv.stride != (1,)
                        or conv.padding != (0,)
                    ):
                        raise ValueError("Unsupported convolution geometry")
                    if conv.dilation != (block.dilation,) or conv.kernel_size != (
                        block.kernel_size,
                    ):
                        raise ValueError(
                            "Causal padding and convolution attributes differ"
                        )
                    taps = {i * conv.dilation[0] for i in range(conv.kernel_size[0])}
                    branch = {a + b for a in branch for b in taps}
                    convs.append(
                        {"kernel": conv.kernel_size[0], "dilation": conv.dilation[0]}
                    )
                offsets |= branch | set(range(int(self.network.recent_window)))
        except (AttributeError, TypeError, OSError, ValueError) as exc:
            offsets, reason = None, str(exc)
        seq = self._pack(history)
        features = self.model.log2feats(seq)
        pivot = seq.size(1) - len(history) + (len(history) - 1) // 2
        changed = seq.clone()
        suffix = changed[:, pivot + 1 :]
        changed[:, pivot + 1 :] = 1 + suffix.remainder(int(self.network.itemnum))
        count = int((seq != changed).sum())
        altered = self.model.log2feats(changed)
        causal_error = float(
            (features[:, : pivot + 1] - altered[:, : pivot + 1]).abs().max()
        )
        empty_norm = float(
            self.model.log2feats(torch.zeros_like(seq)).norm(dim=-1).max()
        )
        unreachable = next(
            (
                i
                for i in range(len(history) - 1)
                if offsets is not None and len(history) - 1 - i not in offsets
            ),
            None,
        )
        unreachable_error = None
        if unreachable is not None and int(self.network.itemnum) > 1:
            control = list(history)
            control[unreachable] = 1 + control[unreachable] % int(self.network.itemnum)
            unreachable_error = float(
                (self._feature(control) - features[0, -1]).abs().max()
            )
        return {
            "convolutions": convs,
            "structural_possible_offsets": (
                sorted(offsets) if offsets is not None else None
            ),
            "reachability_unavailable_reason": reason,
            "reachability_scope": "Possible temporal dependencies including each gate's raw causal summary; masks and learned weights can remove influence",
            "exact_path_counts": None,
            "path_count_unavailable_reason": "Input-dependent multiplicative gates and masked intermediate positions prevent a simple convolution/residual path-count comparison",
            "causality_changed_tokens": count,
            "causality_max_abs_error": causal_error if count else None,
            "all_padding_feature_max_norm": empty_norm,
            "unreachable_control_index": unreachable,
            "unreachable_control_max_abs_error": unreachable_error,
            "unreachable_control_unavailable_reason": (
                None
                if unreachable_error is not None
                else "No applicable occupied unreachable position or no alternative item"
            ),
            "readout": "maximum nonpadding index; all-padding prefixes skipped",
        }, offsets

    @staticmethod
    def _aggregate(rows, bootstrap=400, seed=42):
        rng = np.random.default_rng(seed)
        result = {}
        for variant in sorted({r["variant"] for r in rows}):
            selected = [r for r in rows if r["variant"] == variant]
            users = sorted({r["user"] for r in selected})
            entry = {
                "users": len(users),
                "prefixes": len({(r["user"], r["cut"]) for r in selected}),
                "observations": len(selected),
            }
            for metric in (
                "delta_rank",
                "delta_hr10",
                "delta_sampled_margin",
                "delta_target_score",
                "top10_overlap",
            ):
                user_values, user_mc_variances = [], []
                for user in users:
                    records = [
                        r
                        for r in selected
                        if r["user"] == user and r.get(metric) is not None
                    ]
                    cuts = sorted({r["cut"] for r in records})
                    groups = [
                        np.asarray(
                            [r[metric] for r in records if r["cut"] == cut], dtype=float
                        )
                        for cut in cuts
                    ]
                    if not groups:
                        continue
                    user_values.append(float(np.mean([g.mean() for g in groups])))
                    # Conditional corruption variability only; missing draws cannot
                    # establish MC precision, and are never assigned zero variance.
                    user_mc_variances.append(
                        sum(float(g.var(ddof=1) / len(g)) for g in groups)
                        / len(groups) ** 2
                        if all(len(g) >= 2 for g in groups)
                        else None
                    )
                values = np.asarray(user_values)
                interval = None
                if len(values) >= 2:
                    draws = values[
                        rng.integers(0, len(values), size=(bootstrap, len(values)))
                    ].mean(axis=1)
                    interval = np.quantile(draws, [0.025, 0.975]).tolist()
                mc = None
                if user_mc_variances and all(v is not None for v in user_mc_variances):
                    mc = float(np.sqrt(sum(user_mc_variances)) / len(user_mc_variances))
                entry[metric] = {
                    "mean": float(values.mean()) if len(values) else None,
                    "ci95": interval,
                    "corruption_mc_se": mc,
                    "observed_users": len(values),
                }
            result[variant] = entry
        return result

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        import time

        protocol = {
            "version": "nextitnet-position-replacement-v3",
            "seed": 42,
            "max_users": 12,
            "prefixes_per_user": 2,
            "minimum_prefix_length": 5,
            "history_window": int(self.network.maxlen),
            "draws": 5,
            "doses": [1, 2],
            "bootstrap_repetitions": 400,
            "catalog_chunk_size": 4096,
            "sampled_margin_negatives": 99,
            "eps": 0.5,
            "source": "Training histories and next training items only",
            "frequency_bins": "0, 1, 2-4, 5-15, >=16; replacements require positive training frequency",
            "lag_bands": {"recent": [1, 4], "older": [5, None]},
            "position_matching": "Equal dose, identical ordered original/replacement frequency bins and possible-reachability status across bands; nonadjacent selected positions",
            "replacement_types": ["frequency_random", "local_plausible"],
            "local_plausibility": "Every available immediate in-window directed neighbor transition must be observed in training; each edge count must match the original edge's bin (1, 2-4, 5-15, >=16)",
            "common_support": "Score all four plans only after successful randomized greedy position and distinct-item assignment; algorithm failure does not prove infeasibility",
            "prefix_sampling": "Freeze all user cuts with a separate seed-42 RNG before sampling negatives, positions, replacements or permutations",
            "assignment_algorithm": "One randomized greedy attempt per draw; no retries or tolerance relaxation",
            "displacement_bin_edges": [0.0, 0.5, 1.0, None],
            "replacement_exclusions": "Padding, target, all original full-prefix IDs, duplicate replacements",
            "candidate_rule": "Frozen sorted IDs 1..itemnum excluding original full-prefix items except target",
            "restricted_rule": "Remove the same union of inserted IDs across both bands/types from clean and altered catalogs",
            "permutations": ["full window", "terminal preserving"],
            "tie_break": "Exact score, then ascending item ID",
        }
        states = [(m, m.training) for m in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            print(
                "Diagnostic protocol: " + json.dumps(protocol, sort_keys=True),
                flush=True,
            )
            # Isolate any legacy loader sampling as well as the dedicated plan RNG.
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
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _run_protocol(self, train_loader, protocol, started):
        import time

        rng = random.Random(protocol["seed"])
        self._validate_rank()
        source = self.user_train.get("History", self.user_train)
        if not isinstance(source, dict):
            raise ValueError("Expected ModelView training histories")
        itemnum = int(self.network.itemnum)
        coverage, histories = Counter(), {}
        for user in sorted(source, key=str):
            try:
                history = [int(i) for i in source[user]]
            except (TypeError, ValueError):
                coverage["invalid_history_users"] += 1
                continue
            if not history or any(i < 1 or i > itemnum for i in history):
                coverage["empty_or_out_of_catalog_users"] += 1
                continue
            key = str(user)
            if key in histories:
                raise ValueError("Training user IDs collide after string normalization")
            histories[key] = history
        frequency = Counter(i for h in histories.values() for i in h)
        transitions = Counter(
            (a, b) for h in histories.values() for a, b in zip(h, h[1:])
        )
        outgoing, incoming = {}, {}
        for a, b in transitions:
            outgoing.setdefault(a, set()).add(b)
            incoming.setdefault(b, set()).add(a)

        def bin_count(count):
            return (
                -1
                if count == 0
                else int(np.searchsorted([1, 4, 15], count, side="left"))
            )

        buckets = {}
        for item in sorted(frequency):
            buckets.setdefault(bin_count(frequency[item]), set()).add(item)
        users = sorted(
            u
            for u, h in histories.items()
            if len(h) > protocol["minimum_prefix_length"]
        )
        coverage["eligible_users"] = len(users)
        coverage["short_users"] = len(histories) - len(users)
        users = sorted(rng.sample(users, min(len(users), protocol["max_users"])))
        coverage["selected_users"] = len(users)
        prefix_rng = random.Random(protocol["seed"])
        cuts_by_user = {}
        for user in users:
            cuts = range(protocol["minimum_prefix_length"], len(histories[user]))
            cuts_by_user[user] = sorted(
                prefix_rng.sample(cuts, min(len(cuts), protocol["prefixes_per_user"]))
            )
        rows, baselines, failures, controls = [], [], [], []
        architecture, reachable = None, None

        def effects(clean, altered):
            return {
                "delta_rank": altered["rank"] - clean["rank"],
                "delta_hr10": altered["hr10"] - clean["hr10"],
                "delta_target_score": altered["score"] - clean["score"],
                "delta_sampled_margin": (
                    altered["margin"] - clean["margin"]
                    if altered["margin"] is not None and clean["margin"] is not None
                    else None
                ),
                "top10_overlap": len(set(clean["top10"]) & set(altered["top10"]))
                / len(clean["top10"]),
            }

        for user in users:
            history = histories[user]
            for cut in cuts_by_user[user]:
                coverage["sampled_prefixes"] += 1
                full_prefix, target = history[:cut], history[cut]
                prefix = full_prefix[-protocol["history_window"] :]
                if not prefix:
                    coverage["all_padding_prefixes"] += 1
                    continue
                if len(prefix) < protocol["minimum_prefix_length"]:
                    coverage["short_window_prefixes"] += 1
                    continue
                if architecture is None:
                    architecture, reachable = self._architecture(prefix)
                excluded = set(full_prefix) - {target}
                ids = np.asarray(
                    [i for i in range(1, itemnum + 1) if i not in excluded],
                    dtype=np.int64,
                )
                negatives = [int(i) for i in ids if i != target]
                if not negatives:
                    coverage["no_eligible_negatives"] += 1
                    continue
                sampled = rng.sample(
                    negatives, min(len(negatives), protocol["sampled_margin_negatives"])
                )
                clean_scores = self._catalog_scores(
                    prefix, ids, protocol["catalog_chunk_size"]
                )
                clean = self._result(ids, clean_scores, target, sampled)
                identity_scores = self._catalog_scores(
                    list(prefix), ids, protocol["catalog_chunk_size"]
                )
                identity = self._result(ids, identity_scores, target, sampled)
                controls.append(
                    {
                        "user": user,
                        "cut": cut,
                        "identity_score_max_abs_error": float(
                            np.max(np.abs(clean_scores - identity_scores))
                        ),
                        "identity_delta_rank": identity["rank"] - clean["rank"],
                    }
                )
                strata = {
                    "prefix_length": (
                        "5-15"
                        if len(prefix) <= 15
                        else "16-63" if len(prefix) <= 63 else ">=64"
                    ),
                    "target_popularity": str(bin_count(frequency[target])),
                    "repeated_target": str(target in full_prefix),
                    "transition_frequency": str(
                        bin_count(transitions[(prefix[-1], target)])
                    ),
                }
                common = {"user": user, "cut": cut, **strata}
                baselines.append({**common, **clean})
                n = len(prefix)

                def reach(index):
                    return None if reachable is None else n - 1 - index in reachable

                forbidden = set(full_prefix) | {target, 0}
                random_pools = {
                    key: sorted(bucket - forbidden) for key, bucket in buckets.items()
                }

                def candidate_pool(index, plausible):
                    key = bin_count(frequency[prefix[index]])
                    if not plausible:
                        return random_pools.get(key, [])
                    # Intersect sparse neighborhoods before copying frequency buckets.
                    following = prefix[index + 1]
                    pool = incoming.get(following, set()) & buckets.get(key, set())
                    pool -= forbidden
                    if plausible:
                        if index:
                            pool &= outgoing.get(prefix[index - 1], set())
                        pool = {
                            i
                            for i in pool
                            if bin_count(transitions[(i, following)])
                            == bin_count(transitions[(prefix[index], following)])
                            and (
                                index == 0
                                or bin_count(transitions[(prefix[index - 1], i)])
                                == bin_count(
                                    transitions[(prefix[index - 1], prefix[index])]
                                )
                            )
                        }
                    return sorted(pool)

                pools = {
                    (i, kind): candidate_pool(i, kind == "local_plausible")
                    for i in range(n - 1)
                    for kind in protocol["replacement_types"]
                }
                pairs = [
                    (r, o)
                    for r in range(n - 1)
                    for o in range(n - 1)
                    if 1 <= n - 1 - r <= 4
                    and n - 1 - o >= 5
                    and bin_count(frequency[prefix[r]])
                    == bin_count(frequency[prefix[o]])
                    and reach(r) == reach(o)
                ]
                for dose in protocol["doses"]:
                    for draw in range(protocol["draws"]):
                        shuffled_pairs = list(pairs)
                        rng.shuffle(shuffled_pairs)
                        selected, occupied = [], set()
                        for pair in shuffled_pairs:
                            if (
                                any(abs(i - j) <= 1 for i in pair for j in occupied)
                                or abs(pair[0] - pair[1]) <= 1
                            ):
                                continue
                            selected.append(pair)
                            occupied.update(pair)
                            if len(selected) == dose:
                                break
                        tag = {**common, "dose": dose, "draw": draw}
                        if len(selected) != dose:
                            coverage["position_matching_failures"] += 1
                            failures.append(
                                {
                                    **tag,
                                    "reason": "Greedy position assignment failed; feasibility not established",
                                }
                            )
                            continue
                        plans, pool_sizes = [], []
                        failed = False
                        for kind in protocol["replacement_types"]:
                            for band_index, band in enumerate(("recent", "older")):
                                indices = [p[band_index] for p in selected]
                                inserted = []
                                for index in indices:
                                    pool = [
                                        i
                                        for i in pools[index, kind]
                                        if i not in inserted
                                    ]
                                    pool_sizes.append(
                                        {
                                            "type": kind,
                                            "band": band,
                                            "index": index,
                                            "available": len(pool),
                                        }
                                    )
                                    if not pool:
                                        failed = True
                                        break
                                    inserted.append(rng.choice(pool))
                                if len(inserted) == dose:
                                    plans.append((kind, band, indices, inserted))
                        if failed or len(plans) != 4:
                            coverage["replacement_pool_matching_failures"] += 1
                            failures.append(
                                {
                                    **tag,
                                    "reason": "Greedy distinct-replacement assignment failed; feasibility not established",
                                    "pool_sizes": pool_sizes,
                                }
                            )
                            continue
                        # The restricted catalog is identical across all four plans.
                        removed = sorted(
                            {i for _, _, _, inserted in plans for i in inserted}
                        )
                        clean_restricted = self._result(
                            ids, clean_scores, target, sampled, removed
                        )
                        local = {}
                        for kind, band, indices, inserted in plans:
                            changed = list(prefix)
                            for index, item in zip(indices, inserted):
                                changed[index] = item
                            if (
                                len(changed) != n
                                or changed[-1] != prefix[-1]
                                or any(i == 0 for i in changed)
                            ):
                                raise AssertionError(
                                    "Replacement violated occupancy/terminal invariants"
                                )
                            altered_scores = self._catalog_scores(
                                changed, ids, protocol["catalog_chunk_size"]
                            )
                            altered = self._result(ids, altered_scores, target, sampled)
                            restricted = self._result(
                                ids, altered_scores, target, sampled, removed
                            )
                            old_ids = torch.as_tensor(
                                [prefix[i] for i in indices], device=self.device
                            )
                            new_ids = torch.as_tensor(inserted, device=self.device)
                            displacement = (
                                (
                                    self.model.item_emb(old_ids)
                                    - self.model.item_emb(new_ids)
                                )
                                .norm(dim=-1)
                                .cpu()
                                .tolist()
                            )
                            inserted_effects = []
                            for item in inserted:
                                before = self._result(
                                    ids, clean_scores, item, [], include_top10=False
                                )
                                after = self._result(
                                    ids, altered_scores, item, [], include_top10=False
                                )
                                inserted_effects.append(
                                    {
                                        "item": item,
                                        "score_gain": after["score"] - before["score"],
                                        "rank_gain": before["rank"] - after["rank"],
                                    }
                                )
                            metadata = {
                                **tag,
                                "replacement_type": kind,
                                "band": band,
                                "indices": indices,
                                "lags": [n - 1 - i for i in indices],
                                "original_ids": [prefix[i] for i in indices],
                                "replacement_ids": inserted,
                                "frequency_bins": [
                                    bin_count(frequency[prefix[i]]) for i in indices
                                ],
                                "reachable": [reach(i) for i in indices],
                                "embedding_l2": displacement,
                                "pool_sizes": pool_sizes,
                                "restricted_removed_ids": removed,
                                "inserted_item_effects": inserted_effects,
                                "matching_status": "all four plans feasible",
                            }
                            for scope, before, after in (
                                ("fixed", clean, altered),
                                ("restricted", clean_restricted, restricted),
                            ):
                                row = {
                                    **metadata,
                                    "catalog_scope": scope,
                                    "variant": f"{kind}/dose{dose}/{band}/{scope}",
                                    "catalog_size": before["catalog_size"],
                                    "sampled_negative_count": before[
                                        "margin_negatives"
                                    ],
                                    **effects(before, after),
                                }
                                rows.append(row)
                                local[kind, band, scope] = row
                            coverage["scored_replacement_plans"] += 1
                        for kind in protocol["replacement_types"]:
                            for scope in ("fixed", "restricted"):
                                recent, older = (
                                    local[kind, "recent", scope],
                                    local[kind, "older", scope],
                                )
                                contrast = dict(
                                    recent,
                                    variant=f"{kind}/dose{dose}/recent_minus_older/{scope}",
                                )
                                for metric in (
                                    "delta_rank",
                                    "delta_hr10",
                                    "delta_sampled_margin",
                                    "delta_target_score",
                                ):
                                    contrast[metric] = (
                                        recent[metric] - older[metric]
                                        if recent[metric] is not None
                                        and older[metric] is not None
                                        else None
                                    )
                                contrast["top10_overlap"] = None
                                rows.append(contrast)
                for name in ("full_permutation", "terminal_preserving_permutation"):
                    changed = list(
                        prefix if name == "full_permutation" else prefix[:-1]
                    )
                    rng.shuffle(changed)
                    if name != "full_permutation":
                        changed += prefix[-1:]
                    scores = self._catalog_scores(
                        changed, ids, protocol["catalog_chunk_size"]
                    )
                    result = self._result(ids, scores, target, sampled)
                    rows.append(
                        {
                            **common,
                            "variant": name,
                            "draw": 0,
                            "catalog_scope": "fixed",
                            "unchanged_history": changed == prefix,
                            **effects(clean, result),
                        }
                    )
                    coverage["scored_permutations"] += 1

        collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10
        )
        breakdown["availability"] = "Item categories unavailable"
        baseline_users = {}
        for row in baselines:
            baseline_users.setdefault(row["user"], []).append(row["hr10"])
        strata_results = {}
        for field in (
            "prefix_length",
            "target_popularity",
            "transition_frequency",
            "repeated_target",
        ):
            strata_results[field] = {
                value: self._aggregate([r for r in rows if r[field] == value])
                for value in sorted({r[field] for r in rows})
            }
        # Embedding displacement is measured, not claimed to be matched.
        # Stratify only after assigning fixed displacement edges.
        replacement_rows = [
            r
            for r in rows
            if "embedding_l2" in r and "recent_minus_older" not in r["variant"]
        ]
        strata_results["reachability_and_embedding_l2"] = {}
        for label in ("reachable", "unreachable", "unknown"):
            for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, float("inf"))):
                subset = [
                    r
                    for r in replacement_rows
                    if (
                        "unknown"
                        if any(v is None for v in r["reachable"])
                        else "reachable" if all(r["reachable"]) else "unreachable"
                    )
                    == label
                    and lo <= float(np.mean(r["embedding_l2"])) < hi
                ]
                if subset:
                    strata_results["reachability_and_embedding_l2"][
                        f"{label}/{lo}-{hi}"
                    ] = self._aggregate(subset)
        coverage["successful_prefixes"] = len(baselines)
        coverage["successful_replacement_users"] = len(
            {r["user"] for r in replacement_rows}
        )
        coverage["successful_replacement_prefixes"] = len(
            {(r["user"], r["cut"]) for r in replacement_rows}
        )
        coverage["paired_rows_including_contrasts"] = len(rows)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        metrics = {
            "protocol": protocol,
            "coverage": dict(coverage),
            "architecture": architecture,
            "chunked_rank_synthetic_check": True,
            "identity_controls": controls,
            "matching_failures": failures,
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
            "intervention_effects": self._aggregate(rows),
            "strata": strata_results,
            "baseline_observations": baselines,
            "paired_observations": rows,
            "four_category_macro": None,
            "four_category_macro_reason": "One category ModelView; equal-category aggregation requires all four measured categories",
            "category_breakdown": {},
            "unavailable": [
                "review ratings/text",
                "item ratings/categories/prices/stores",
                "simulator traits",
                "held-out intervention effects",
                "exact gated path counts",
                "lexical plausibility (not measured)",
                "causal user intent",
            ],
            "diagnostic_cost": {
                "synchronized_elapsed_seconds": elapsed,
                "device": str(self.device),
                "gpu_compute_seconds": None,
                "gpu_reserved_elapsed_seconds_upper_bound": (
                    elapsed if self.device.type == "cuda" else 0.0
                ),
                "accounting": "Elapsed includes CPU work; host GPU accounting is authoritative",
            },
        }
        definitions = {
            "protocol": "Versioned rules printed before effects; only diagnostic corruption pools use training transition counts. No auxiliary recommendation scores or model fitting.",
            "coverage": "Actual users, unique prefixes, plans and failures. Failed matching removes the entire four-plan block; absence is not zero effect. Doses can have different support.",
            "architecture": "Possible offsets include causal-summary gates when supplied implementation is recognized. Unknown architecture disables structural interpretation. Masks may reduce possible reachability. Structural reachability is not learned influence; exact path counts unavailable.",
            "chunked_rank_synthetic_check": "Chunked score/rank tie counter verified against exhaustive synthetic sorting at several chunk sizes.",
            "identity_controls": "Repeated identical-input catalog score discrepancy and target-rank change; expected zero within numerical precision. Causality, all-padding and applicable unreachable checks are in architecture. These local controls do not prove calibration or CI coverage.",
            "matching_failures": "Auditable user/cut/dose/draw failures and pool sizes. A failed greedy assignment is an algorithmic failure, not proof that no valid assignment exists. No tolerance is relaxed; effects condition on successful assignments.",
            "embedding_collapse_score": "Sampled mean pairwise item cosine mapped to [0,1]. Near 0.5 means mean cosine near zero; alignment alone cannot establish collapse or recommendation quality.",
            "pos_neg_margin_mean": "Mean positive-minus-negative logit over valid aligned training-loader positions; in-sample sampled separation only.",
            "pos_neg_margin_p10": "Empirical lower tenth percentile of valid training-loader margins; null if unavailable.",
            "pos_neg_neg_beats_pos_rate": "Fraction of valid training positions where negative score exceeds positive score.",
            "pos_neg_small_margin_rate": "Fraction of valid training positions with margin below protocol eps.",
            "pos_neg_margin_category_breakdown": "Low/wrong-margin counts; category lists empty because item categories are unavailable.",
            "training_prefix_hr10_user_mean": "Training-target exact eligible-catalog HR@10 averaged within user then across users; not host validation or generalization.",
            "intervention_effects": "Changed-minus-clean ranks, HR@10, target score and fixed-negative mean margin; positive delta_rank is worse. Overlap is top-10 intersection/original top-10 size. Draws average within prefix, prefixes within user, then users equally. CI95 is exploratory percentile bootstrap of user means, null below two users. corruption_mc_se separately propagates within-prefix draw variance, conditional on successful matching and treating prefix draws as independent; null when any contributing prefix has fewer than two observations. More draws do not add independent users. recent_minus_older is a paired difference on the same successful four-plan blocks, with the same restricted catalog; never assume monotonic dose response.",
            "strata": "Fixed prefix, training-frequency, repeated-target, transition, reachability and displacement strata. Displacement bins are [0,.5), [.5,1), [1,infinity); displacement is measured, not matched. Compare bands only where both have support; intervals are exploratory without multiplicity or coverage guarantees.",
            "baseline_observations": "Clean exact ranks with fixed original full-prefix eligibility, target retained once even when repeated; catalog size and margin-negative count are denominators.",
            "paired_observations": "In-place replacements preserve occupancy, padding, terminal item, sequence length and timestamp positions. Fixed catalogs include inserted candidates. Restricted catalogs remove the same union of inserted IDs from all paired histories and both bands/types; they are not full-catalog ranks. Inserted score/rank gains refer to the fixed eligible catalog. Frequency bins and possible reachability match across bands; path counts and displacement do not. Permutations separately preserve the original item multiset. Neither score changes nor restricted-rank decomposition identify causal user intent.",
            "four_category_macro": "Unavailable in a single-category call; never substitute a pooled-user or single-category value. The pinned host macro validation HR@10 alone selects model trials.",
            "category_breakdown": "No item-category metadata is available; dataset routes remain unchanged.",
            "unavailable": "Explicit unobserved data or unmeasured quantities; missing evidence must not be invented.",
            "diagnostic_cost": "Synchronized wall time includes CPU work and all probes. GPU compute time is not separately measured; host search-budget accounting remains authoritative.",
        }
        # Review checks: statically traced API calls, eligibility, ties, restricted
        # catalogs, fixed prefix sampling, RNG restoration and replacement pools.
        # Inserted-item rank queries skip top-ten sorting. Training, checkpoints,
        # predictor architecture and host evaluation are unchanged. Not executed.
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
