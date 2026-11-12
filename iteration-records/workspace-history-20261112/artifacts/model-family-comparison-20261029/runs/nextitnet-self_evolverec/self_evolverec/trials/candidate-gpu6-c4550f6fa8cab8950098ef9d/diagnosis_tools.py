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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measured sensitivity with explicit uncertainty and missingness
    - Missing measurements are unknown, never evidence of healthy behavior.
    - Training margins and interventions are not validation HR@10 or evidence of user intent, satisfaction, or simulator traits.
    - Treat old-versus-random frequency-matched substitution as the prespecified contrast; other contrasts are controls or exploratory.
    - State observation counts, intervention failures, uncertainty units, and missing audit checks when interpreting results.
    - Directional embedding alignment does not establish recommendation diversity; score margins depend on score scale and supplied negatives.
    - Bootstrap intervals are conditional on this checkpoint and protocol, exclude training/search uncertainty, and are not simultaneous guarantees.
    - Do not recommend model selection from these diagnostics; the pinned host macro validation HR@10 remains the selection criterion.
    # Review checks: statically traced final-readout scoring, candidate validity,
    # finite-value exclusions, paired contrasts, substitution exclusions, hook cleanup,
    # RNG/mode restoration, and single-checkpoint consistency. No execution or metric
    # improvement is claimed. Training and predictor architecture remain unchanged.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and restrict diagnostics to available public histories
    def __init__(self, view, device):
        self.model = view
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
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
            ### >>> Self_EvolveRec-BLOCK-START: Empty embedding measurements are explicitly missing
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            ### >>> Self_EvolveRec-BLOCK-START: Require at least two sampled embeddings for pairwise alignment
            k = min(int(sample_size), n)
            if k < 2:
                return None
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Require evaluation mode without mutating caller state
        if any(module.training for module in self.model.model.modules()):
            raise RuntimeError(
                "Margin diagnostics require evaluation mode; use run_full_diagnosis "
                "to save and restore the checkpoint's module modes."
            )
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

            seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            log_feats = self.model.log2feats(seq_np)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Score only valid candidates at the original final occupied position
            sequence_t = torch.as_tensor(seq_np, device=log_feats.device)
            if sequence_t.ndim != 2 or sequence_t.size(1) == 0:
                raise ValueError(
                    "Margin diagnostics require nonzero-width B x L sequences"
                )
            if pos_t.shape != sequence_t.shape or neg_t.shape != sequence_t.shape:
                raise ValueError("Sequence, positive and negative shapes must match")
            pos_t = pos_t.to(log_feats.device)
            neg_t = neg_t.to(log_feats.device)
            positions = torch.arange(sequence_t.size(1), device=log_feats.device)
            last = (
                positions.expand_as(sequence_t)
                .masked_fill(sequence_t == 0, -1)
                .max(1)
                .values
            )
            rows = torch.arange(sequence_t.size(0), device=log_feats.device)
            safe = last.clamp_min(0)
            vocab = int(self.model.model.itemnum)
            candidates = torch.stack((pos_t[rows, safe], neg_t[rows, safe]), -1)
            pair_valid = (last >= 0) & ((candidates > 0) & (candidates <= vocab)).all(1)
            pair_valid &= candidates[:, 0] != candidates[:, 1]
            margin = log_feats.new_full(sequence_t.shape, float("nan"))
            if pair_valid.any():
                selected_rows = rows[pair_valid]
                selected_last = last[pair_valid]
                logits = self.model.predict_score(
                    log_feats[selected_rows, selected_last], candidates[pair_valid]
                )
                if tuple(logits.shape) != tuple(candidates[pair_valid].shape):
                    raise ValueError("ModelView.predict_score must return B x 2 scores")
                margin[selected_rows, selected_last] = logits[:, 0] - logits[:, 1]
            valid = torch.zeros_like(sequence_t, dtype=torch.bool)
            valid[rows[pair_valid], last[pair_valid]] = True
            valid &= torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Share final-readout and finite-value validity with counters
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
            ### >>> Self_EvolveRec-BLOCK-START: Report missing margins with an observation count
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "count": 0,
                "missing_reason": "No valid original-readout candidate pairs.",
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            ### >>> Self_EvolveRec-BLOCK-START: Retain the number of measured margins
            margin_metrics = {
                "count": int(N),
                "missing_reason": None,
                "mean": round(float(m.mean().item()), 4),
                ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Matched frozen-checkpoint interventions with finite controls and cached substitution pools
    def run_full_diagnosis(self, train_loader):
        actual = self.model.model
        saved_mode = actual.training
        # Preserve mixed submodule modes as well as the root mode.
        saved_modes = [(module, module.training) for module in actual.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        handles = []
        counts = Counter()
        records = {}
        lag_counts = {band: Counter() for band in ("recent", "old", "random")}
        replacement_counts = Counter()
        rng = np.random.default_rng(42)
        vocab = int(actual.itemnum)
        if vocab < 1:
            raise ValueError("Diagnosis requires a nonempty item vocabulary")
        audit = {"padding_activations": {}, "locality_verified": False}

        def features(sequence):
            result = self.model.log2feats(sequence.detach().cpu().numpy())
            if not isinstance(result, torch.Tensor) or result.ndim != 3:
                raise ValueError("ModelView.log2feats must return B x L x D features")
            if tuple(result.shape[:2]) != tuple(sequence.shape):
                raise ValueError("ModelView changed sequence positions")
            return result

        def scores(feats, last, candidates):
            rows = torch.arange(feats.shape[0], device=feats.device)
            selected = feats[rows, last.to(feats.device)]
            result = self.model.predict_score(selected, candidates.to(feats.device))
            if tuple(result.shape) != tuple(candidates.shape):
                raise ValueError("ModelView.predict_score must return B x 2 scores")
            return result

        def stratum(length):
            if length <= 5:
                return "1-5"
            if length <= 15:
                return "6-15"
            if length <= 31:
                return "16-31"
            return "32+"

        def add_record(name, user, value):
            records.setdefault(name, {}).setdefault(user, []).append(float(value))

        def summarize(by_user):
            # Stable user order makes the local bootstrap reproducible.
            values = np.asarray(
                [np.mean(by_user[user]) for user in sorted(by_user)], dtype=float
            )
            if not len(values):
                return {
                    "mean": None,
                    "users": 0,
                    "observations": 0,
                    "ci95": None,
                    "missing_reason": "No finite outcomes from eligible uniquely identified training users.",
                }
            result = {
                "mean": float(values.mean()),
                "users": int(len(values)),
                "observations": int(sum(len(v) for v in by_user.values())),
                "ci95": None,
                "missing_reason": None,
            }
            if len(values) >= 2:
                boot_rng = np.random.default_rng(42)
                means = [
                    float(values[boot_rng.integers(0, len(values), len(values))].mean())
                    for _ in range(2000)
                ]
                result["ci95"] = np.quantile(means, [0.025, 0.975]).tolist()
            else:
                result["interval_missing_reason"] = (
                    "Fewer than two independent user IDs."
                )
            return result

        try:
            actual.eval()
            histories = self.user_train.get("History", self.user_train)
            histories = {
                str(user): [int(i) for i in items] for user, items in histories.items()
            }
            frequencies = np.zeros(vocab + 1, dtype=np.int64)
            for history in histories.values():
                for item in history:
                    if 0 < item <= vocab:
                        frequencies[item] += 1
            # Quantile thresholds over item frequencies keep every frequency tie together.
            thresholds = np.quantile(frequencies[1:], [0.2, 0.4, 0.6, 0.8])
            bins = np.searchsorted(thresholds, frequencies, side="left")
            pools = {
                b: np.flatnonzero((bins == b) & (np.arange(vocab + 1) > 0))
                for b in range(5)
            }
            counts["frequency_bins_nonempty"] = sum(
                bool(len(v)) for v in pools.values()
            )
            cached = []
            for _, batch in zip(range(10), train_loader):
                counts["loader_batches_seen"] += 1
                if not isinstance(batch, (list, tuple)) or len(batch) < 4:
                    counts["malformed_batches"] += 1
                    continue
                tensors = [
                    torch.as_tensor(x, dtype=torch.long).detach().cpu().clone()
                    for x in batch[1:4]
                ]
                seq, pos, neg = tensors
                if (
                    seq.ndim != 2
                    or seq.shape[1] == 0
                    or pos.shape != seq.shape
                    or neg.shape != seq.shape
                ):
                    raise ValueError(
                        "Diagnostic loader must yield matching nonzero-width B x L tensors"
                    )
                counts["rows_seen"] += seq.shape[0]
                occupied = seq > 0
                last = (
                    torch.arange(seq.shape[1])
                    .expand_as(seq)
                    .masked_fill(~occupied, -1)
                    .max(1)
                    .values
                )
                rows = torch.arange(seq.shape[0])
                safe = last.clamp_min(0)
                sequence_valid = ((seq >= 0) & (seq <= vocab)).all(1)
                # The supplied sampler left-pads: occupied positions must form a suffix.
                left_padded = ~(occupied[:, :-1] & ~occupied[:, 1:]).any(1)
                candidate_valid = (pos[rows, safe] > 0) & (pos[rows, safe] <= vocab)
                candidate_valid &= (neg[rows, safe] > 0) & (neg[rows, safe] <= vocab)
                candidate_valid &= pos[rows, safe] != neg[rows, safe]
                valid = (last >= 0) & sequence_valid & left_padded & candidate_valid
                counts["invalid_sequence_rows"] += int((~sequence_valid).sum())
                counts["non_left_padded_rows"] += int((~left_padded).sum())
                counts["invalid_aligned_candidate_rows"] += int(
                    ((last >= 0) & ~candidate_valid).sum()
                )
                counts["empty_rows"] += int((last < 0).sum())
                counts["excluded_rows"] += int((~valid).sum())
                if valid.any():
                    cached.append((seq[valid], pos[valid], neg[valid], last[valid]))
            counts["cached_batches"] = len(cached)
            aligned_batches = []
            for seq, pos, neg, last in cached:
                aligned_pos, aligned_neg = torch.zeros_like(pos), torch.zeros_like(neg)
                rows = torch.arange(len(seq))
                aligned_pos[rows, last] = pos[rows, last]
                aligned_neg[rows, last] = neg[rows, last]
                aligned_batches.append((None, seq, aligned_pos, aligned_neg))

            with torch.no_grad():
                collapse = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048, seed=42
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        aligned_batches, num_batches=10, eps=0.5, topk_categories=0
                    )
                )
                audit["id0_weight_norm"] = float(
                    self.model.item_emb.weight[0].norm().item()
                )
                audit["embedding_padding_idx"] = self.model.item_emb.padding_idx
                audit["id0_initialization"] = (
                    "Unavailable from a trained checkpoint; measured weight reported."
                )
                audit["convolutions"] = [
                    {
                        "name": name,
                        "kernel": list(module.kernel_size),
                        "dilation": list(module.dilation),
                        "bias_norm": (
                            None
                            if module.bias is None
                            else float(module.bias.norm().item())
                        ),
                    }
                    for name, module in actual.named_modules()
                    if isinstance(module, torch.nn.Conv1d)
                ]
                audit["normalization_axes"] = [
                    {"name": name, "normalized_shape": list(module.normalized_shape)}
                    for name, module in actual.named_modules()
                    if isinstance(module, torch.nn.LayerNorm)
                ]
                audit["residual_and_mask_audit"] = (
                    "Runtime block outputs are checked below. Module metadata alone cannot certify "
                    "functional residual, masking, gate, normalization, or global dependency paths."
                )
                span = 1
                span_known = True
                for block in getattr(actual, "blocks", []):
                    if type(block).__name__ == "NextItNetBlock":
                        span += sum(
                            (conv.kernel_size[0] - 1) * conv.dilation[0]
                            for conv in (block.conv1, block.conv2)
                        )
                    elif type(block).__name__ == "GatedNextItNetBlock":
                        span += max(
                            sum(
                                (layer.kernel_size[0] - 1) * layer.dilation[0]
                                for layer in branch
                                if isinstance(layer, torch.nn.Conv1d)
                            )
                            for branch in block.branches
                        )
                    else:
                        span_known = False
                audit["structural_candidate_span"] = span if span_known else None
                audit["outside_field_control"] = {
                    "value": None,
                    "missing_reason": "Every functional dependency path has not been independently audited; "
                    "a structural candidate span is not a locality certificate.",
                }

                user_maps = {}
                for seq, pos, neg, last in cached:
                    width = seq.shape[1]
                    if width not in user_maps:
                        lookup = {}
                        for user, history in histories.items():
                            if len(history) < 2:
                                continue
                            # The sampler's final aligned positive is history[-1].
                            key = (tuple(history[:-1][-width:]), history[-1])
                            if key in lookup:
                                lookup[key] = None
                            else:
                                lookup[key] = user
                        user_maps[width] = lookup
                    seq = seq.to(self.device)
                    last = last.to(self.device)
                    row_ids = torch.arange(len(seq))
                    candidates = torch.stack(
                        (pos[row_ids, last.cpu()], neg[row_ids, last.cpu()]), -1
                    ).to(self.device)
                    if not audit["padding_activations"]:
                        pad = seq == 0

                        def capture(name):
                            def hook(module, args, output):
                                if (
                                    isinstance(output, torch.Tensor)
                                    and output.ndim == 3
                                    and output.shape[:2] == seq.shape
                                ):
                                    norms = output[pad].float().norm(dim=-1)
                                    audit["padding_activations"][name] = {
                                        "positions": int(norms.numel()),
                                        "mean_norm": (
                                            float(norms.mean())
                                            if norms.numel()
                                            else None
                                        ),
                                        "max_norm": (
                                            float(norms.max())
                                            if norms.numel()
                                            else None
                                        ),
                                    }

                            return hook

                        for index, block in enumerate(getattr(actual, "blocks", [])):
                            handles.append(
                                block.register_forward_hook(capture(f"block_{index}"))
                            )
                    try:
                        intact_features = features(seq)
                    finally:
                        for handle in handles:
                            handle.remove()
                        handles.clear()
                    intact_scores = scores(intact_features, last, candidates)
                    repeat_scores = scores(features(seq.clone()), last, candidates)
                    tolerance = {
                        torch.float64: (1e-9, 1e-7),
                        torch.float32: (1e-6, 1e-5),
                    }.get(intact_scores.dtype, (1e-3, 1e-2))
                    audit["score_tolerance"] = {
                        "atol": tolerance[0],
                        "rtol": tolerance[1],
                        "dtype": str(intact_scores.dtype),
                    }
                    feature_rows = torch.arange(len(seq), device=intact_features.device)
                    original_readout = intact_features[
                        feature_rows, last.to(intact_features.device)
                    ]
                    repeated_features = features(seq.clone())
                    repeated_readout = repeated_features[
                        feature_rows, last.to(repeated_features.device)
                    ]
                    finite = torch.isfinite(intact_scores).all(-1)
                    finite &= torch.isfinite(repeat_scores).all(-1)
                    feature_ok = torch.isfinite(original_readout).all(-1)
                    feature_ok &= torch.isfinite(repeated_readout).all(-1)
                    feature_tolerance = {
                        torch.float64: (1e-9, 1e-7),
                        torch.float32: (1e-6, 1e-5),
                    }.get(original_readout.dtype, (1e-3, 1e-2))
                    feature_ok &= torch.isclose(
                        original_readout,
                        repeated_readout,
                        atol=feature_tolerance[0],
                        rtol=feature_tolerance[1],
                    ).all(-1)
                    no_op_ok = (
                        finite
                        & torch.isclose(
                            intact_scores,
                            repeat_scores,
                            atol=tolerance[0],
                            rtol=tolerance[1],
                        ).all(-1)
                        & feature_ok.to(finite.device)
                    )
                    counts["no_op_failed_rows"] += int((~no_op_ok).sum())
                    counts["no_op_passed_rows"] += int(no_op_ok.sum())
                    audit["feature_tolerance"] = {
                        "atol": feature_tolerance[0],
                        "rtol": feature_tolerance[1],
                        "dtype": str(original_readout.dtype),
                    }
                    baseline = intact_scores[:, 0] - intact_scores[:, 1]
                    eligible = []
                    seq_cpu = seq.cpu().numpy()
                    last_cpu = last.cpu().numpy()
                    candidates_cpu = candidates.cpu().numpy()
                    no_op_cpu = no_op_ok.cpu().numpy()
                    baseline_cpu = baseline.float().cpu().numpy()
                    for row, history in enumerate(seq_cpu):
                        occupied = np.flatnonzero(history > 0)
                        if not no_op_cpu[row]:
                            continue
                        original_last = int(last_cpu[row])
                        recent = occupied[
                            (original_last - occupied >= 1)
                            & (original_last - occupied <= 5)
                        ]
                        old = occupied[original_last - occupied >= 16]
                        m = min(3, len(recent), len(old))
                        if not m:
                            counts["ineligible_lag_bands"] += 1
                            continue
                        user = user_maps[width].get(
                            (
                                tuple(history[occupied].tolist()),
                                int(candidates_cpu[row, 0]),
                            )
                        )
                        if user is None:
                            counts["unidentified_or_ambiguous_user_rows"] += 1
                            continue
                        if not np.isfinite(baseline_cpu[row]):
                            counts["nonfinite_baseline_rows"] += 1
                            continue
                        eligible.append(
                            (
                                row,
                                user,
                                m,
                                recent,
                                old,
                                occupied[occupied < original_last],
                                stratum(len(occupied)),
                            )
                        )
                        counts["eligible_rows"] += 1
                        counts[f"matched_count_{m}"] += 1
                    if not eligible:
                        continue
                    # Compute exclusions once per row and requested frequency bin.
                    # This replaces repeated Python scans of large catalog bins.
                    allowed_pools = {}
                    for row, user, m, recent, old, all_lags, group in eligible:
                        forbidden = np.asarray(
                            sorted(
                                set(histories[user])
                                | set(candidates_cpu[row].tolist())
                                | {0}
                            ),
                            dtype=np.int64,
                        )
                        for bin_id in np.unique(bins[seq_cpu[row, all_lags]]):
                            pool = pools[int(bin_id)]
                            allowed_pools[(row, int(bin_id))] = pool[
                                ~np.isin(pool, forbidden)
                            ]
                    for draw in range(3):
                        selections = {band: [] for band in lag_counts}
                        replacements = {band: [] for band in lag_counts}
                        frequency_ok = np.ones(len(eligible), dtype=bool)
                        for index, (
                            row,
                            user,
                            m,
                            recent,
                            old,
                            all_lags,
                            group,
                        ) in enumerate(eligible):
                            for band, positions in (
                                ("recent", recent),
                                ("old", old),
                                ("random", all_lags),
                            ):
                                chosen = rng.choice(positions, m, replace=False)
                                selections[band].append(chosen)
                                lag_counts[band].update(
                                    (int(last_cpu[row]) - chosen).tolist()
                                )
                                substitutes = []
                                for position in chosen:
                                    original = int(seq_cpu[row, position])
                                    # The original item belongs to the user's observed history
                                    # and was excluded when this pool was constructed.
                                    allowed = allowed_pools[(row, int(bins[original]))]
                                    if len(allowed) == 0:
                                        frequency_ok[index] = False
                                        counts[
                                            "frequency_no_qualifying_item_positions"
                                        ] += 1
                                        substitutes.append(original)
                                    else:
                                        substitutes.append(int(rng.choice(allowed)))
                                replacements[band].append(substitutes)
                        counts["frequency_complete_draw_rows"] += int(
                            frequency_ok.sum()
                        )
                        counts["frequency_abstained_draw_rows"] += int(
                            (~frequency_ok).sum()
                        )
                        for replacement in (
                            "id0",
                            "zero_item_embedding",
                            "frequency_matched",
                        ):
                            selected = [
                                i
                                for i in range(len(eligible))
                                if replacement != "frequency_matched" or frequency_ok[i]
                            ]
                            if not selected:
                                continue
                            original_rows = torch.tensor(
                                [eligible[i][0] for i in selected], device=self.device
                            )
                            deltas = {}
                            for band in lag_counts:
                                edited = seq.index_select(0, original_rows).clone()
                                mask = torch.zeros_like(edited, dtype=torch.bool)
                                for new_row, index in enumerate(selected):
                                    positions = torch.as_tensor(
                                        selections[band][index], device=self.device
                                    )
                                    mask[new_row, positions] = True
                                    if replacement == "id0":
                                        edited[new_row, positions] = 0
                                    elif replacement == "frequency_matched":
                                        edited[new_row, positions] = torch.tensor(
                                            replacements[band][index],
                                            device=self.device,
                                        )
                                hook_calls = [0]
                                if replacement == "zero_item_embedding":

                                    def zero_hook(module, args, output):
                                        hook_calls[0] += 1
                                        if output.shape[
                                            :2
                                        ] != mask.shape or not torch.equal(
                                            args[0], edited
                                        ):
                                            raise ValueError(
                                                "Input embedding hook encountered a different lookup path"
                                            )
                                        return output.masked_fill(mask.unsqueeze(-1), 0)

                                    handles.append(
                                        self.model.item_emb.register_forward_hook(
                                            zero_hook
                                        )
                                    )
                                try:
                                    edited_features = features(edited)
                                finally:
                                    for handle in handles:
                                        handle.remove()
                                    handles.clear()
                                if (
                                    replacement == "zero_item_embedding"
                                    and hook_calls[0] != 1
                                ):
                                    counts["zero_embedding_path_audit_failures"] += len(
                                        selected
                                    )
                                    continue
                                edited_scores = scores(
                                    edited_features,
                                    last[original_rows],
                                    candidates[original_rows],
                                )
                                delta = (
                                    edited_scores[:, 0]
                                    - edited_scores[:, 1]
                                    - baseline[original_rows]
                                )
                                deltas[band] = delta.float().cpu().numpy()
                                for local, index in enumerate(selected):
                                    if not np.isfinite(deltas[band][local]):
                                        counts["nonfinite_intervention_values"] += 1
                                        continue
                                    user = eligible[index][1]
                                    add_record(
                                        f"{replacement}/{band}",
                                        user,
                                        deltas[band][local],
                                    )
                                    replacement_counts[
                                        f"{replacement}/{band}"
                                    ] += eligible[index][2]
                            if "old" in deltas and "random" in deltas:
                                for local, index in enumerate(selected):
                                    contrast = (
                                        deltas["old"][local] - deltas["random"][local]
                                    )
                                    if np.isfinite(contrast):
                                        user, group = (
                                            eligible[index][1],
                                            eligible[index][-1],
                                        )
                                        name = f"{replacement}/old_minus_random"
                                        add_record(name, user, contrast)
                                        add_record(
                                            f"{name}/length_{group}", user, contrast
                                        )

            effect_names = [
                f"{replacement}/{band}"
                for replacement in ("id0", "zero_item_embedding", "frequency_matched")
                for band in ("recent", "old", "random", "old_minus_random")
            ]
            for name in effect_names:
                records.setdefault(name, {})
            for replacement in ("id0", "zero_item_embedding", "frequency_matched"):
                for group in ("1-5", "6-15", "16-31", "32+"):
                    records.setdefault(
                        f"{replacement}/old_minus_random/length_{group}", {}
                    )
            counts["nonfinite_margin_rows"] = (
                sum(len(seq) for seq, _, _, _ in cached) - margin_metrics["count"]
            )
            audit["zero_embedding_path_limitations"] = (
                "A single matching embedding-module invocation is checked per intervention. "
                "This does not certify the absence of direct functional weight lookups or "
                "other input-embedding paths; no complete path audit is claimed."
            )
            metrics = {
                "embedding_collapse_score": collapse,
                "embedding_observation_count": min(2048, vocab),
                "embedding_missing_reason": (
                    None if collapse is not None else "Fewer than two item embeddings."
                ),
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "margin_observations": margin_metrics["count"],
                "margin_missing_reason": margin_metrics["missing_reason"],
                "pos_neg_margin_category_breakdown": breakdown,
                "category_breakdown_missing_reason": "Item categories are unavailable in this task.",
                "counts": dict(counts),
                "realized_lag_counts": {
                    k: dict(sorted(v.items())) for k, v in lag_counts.items()
                },
                "realized_replacement_position_counts": dict(replacement_counts),
                "intervention_margin_deltas": {
                    name: summarize(values) for name, values in records.items()
                },
                "audit": audit,
                "validation_interventions": {
                    "metrics": None,
                    "missing_reason": "ModelView and train_loader do not expose the exact authorized host validation "
                    "users, candidates, and tie policy. Candidates were not reconstructed.",
                },
                "macro_validation_bootstrap": {
                    "metrics": None,
                    "missing_reason": "This probe observes one category checkpoint and no pinned validation candidate sets.",
                },
            }
            return {
                "metrics": metrics,
                "metric_definitions": {
                    "embedding_collapse_score": "(mean off-diagonal cosine + 1)/2, sampled without replacement "
                    "with seed 42. Measures directional alignment, not recommendation diversity.",
                    "pos_neg_margin_mean": "Mean positive-minus-negative raw score at the ORIGINAL final "
                    "occupied position; neither empty rows nor invalid candidate pairs are included.",
                    "pos_neg_margin_p10": "Nearest-rank tenth percentile of these margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of these margins below zero.",
                    "pos_neg_small_margin_rate": "Fraction of these margins below 0.5; score-scale dependent.",
                    "pos_neg_margin_category_breakdown": "Category lists are empty because item category metadata is unavailable.",
                    "protocol": "Up to ten identical cached loader batches, fixed input width, candidates, checkpoint, "
                    "and original readout. Lag zero is never edited. Recent lags 1-5; old lags >=16; "
                    "random controls sample all occupied positive lags. m=min(3,recent count,old count). "
                    "Three draws with local seed 42 share positions across replacement types. "
                    "These defaults are prespecified, not established optima.",
                    "interventions": "ID0 changes the model's padding/position mask. Zero-item-embedding "
                    "uses a temporary lookup-output hook and preserves original IDs, positional embeddings, "
                    "and padding masks. These interventions are not equivalent. Frequency replacements "
                    "use training-interaction frequency quintiles, keep ties together, exclude original IDs, "
                    "observed training items and candidates, and abstain jointly across bands for each "
                    "row/draw if any required substitution is unavailable. No predictor is fitted.",
                    "intervention_margin_deltas": "Edited minus intact positive-minus-negative margin. "
                    "Repeated draws and repeated sampled rows are averaged within uniquely identified "
                    "training user IDs before aggregation. Ambiguous suffix-to-user matches are excluded. "
                    "Old-minus-random is paired within the same row and draw. Positive values mean "
                    "old replacement increases the margin more than random replacement. "
                    "User identification uses the sampler's truncated training prefix plus its "
                    "aligned final positive. Strata use occupied input length, truncated by input width. "
                    "Missing length-stratum results are explicit; old-lag eligibility requires at least "
                    "17 occupied positions for contiguous left-padded histories.",
                    "uncertainty": "2,000 paired user bootstrap replicates with seed 42, percentile 95% "
                    "intervals within this category only. Length strata are 1-5,6-15,16-31,32+. "
                    "Intervals condition on checkpoint, supplied negatives, cached histories and replacement "
                    "draws; they exclude training and adaptive-search uncertainty. Users are the sampling "
                    "units, never replacement draws. Overall training intervals use an unstratified "
                    "user bootstrap; length-specific intervals resample within each reported stratum. "
                    "They are not the proposed stratified four-category validation bootstrap. "
                    "No macro validation interval is inferred.",
                    "primary_contrast": "frequency_matched/old_minus_random is prespecified. Recent, ID0, "
                    "zero-item-embedding and stratum results are controls or exploratory, without multiple-comparison guarantees.",
                    "locality": "Structural candidate span sums longest convolutional increments across residual "
                    "blocks, including both convolutions and the longest gated branch. It is not certified: "
                    "functional paths and dilation holes require a complete dependency audit. Outside-field "
                    "invariance is explicitly unavailable until then. Padding norms are observations at "
                    "block outputs on a bounded batch; no clamping intervention is performed.",
                    "scope": "Training sensitivity only; not validation ranking performance, user intent, "
                    "compatibility, satisfaction, or simulator traits. Selection remains pinned host macro HR@10. "
                    "No training, architecture, checkpoint, predictor, routing, or data changes are made.",
                },
            }
        finally:
            for handle in handles:
                handle.remove()
            actual.train(saved_mode)
            for module, mode in saved_modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    ### <<< Self_EvolveRec-BLOCK-END
