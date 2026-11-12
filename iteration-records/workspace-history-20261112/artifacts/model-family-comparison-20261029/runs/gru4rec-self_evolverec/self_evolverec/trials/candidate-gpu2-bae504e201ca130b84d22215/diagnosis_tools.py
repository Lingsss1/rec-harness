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
    ### >>> Self_EvolveRec-BLOCK-START: Keep the trained module distinct from its diagnostic view
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories and review attributes are unavailable in this task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Read real item embeddings through the ModelView contract
            emb = self.view.item_emb.weight.detach()
            emb = emb[1 : self.model.itemnum + 1]
            if not torch.isfinite(emb).all().item():
                raise FloatingPointError("Nonfinite item embeddings in diagnosis")
            ### <<< Self_EvolveRec-BLOCK-END

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Require at least two sampled embeddings for pairwise geometry
            if min(int(sample_size), n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute exact mean pairwise cosine without a quadratic Gram matrix
            # Sum all pairwise dot products using the squared norm of the sum.
            # Subtract measured diagonal terms rather than assuming unit norms:
            # F.normalize leaves zero embeddings at zero.
            x = x.float()
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    ### >>> Self_EvolveRec-BLOCK-START: Restore model mode for standalone margin probes including failures
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        was_training = self.model.training
        self.model.eval()
        try:
            return self._measure_pos_neg_margins(
                dataloader, num_batches, eps, topk_categories
            )
        finally:
            self.model.train(was_training)

    def _measure_pos_neg_margins(self, dataloader, num_batches, eps, topk_categories):
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

            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned family-specific inputs before diagnostic scoring
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic inputs must be aligned B x L tensors")
            if any(
                ((ids < 0) | (ids > self.model.itemnum)).any().item()
                for ids in (seq_t, pos_t, neg_t)
            ):
                raise ValueError(
                    "Diagnostic item IDs are outside the trained vocabulary"
                )
            log_feats = self.view.log2feats(seq_t)
            if log_feats.ndim != 3 or log_feats.shape[:2] != seq_t.shape:
                raise ValueError("log2feats must return aligned B x L x H features")
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Score diagnostic candidates through the family-specific view
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.view.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padded inputs and absent negative targets
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Reject invalid measured margins instead of reporting misleading rates
            if valid.any():
                measured = margin[valid].detach().float()
                if not torch.isfinite(measured).all().item():
                    raise FloatingPointError(
                        "Nonfinite valid positive-negative margins"
                    )
                margins_all.append(measured.cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Keep diagnostic counts aligned with measured margins
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Count measured errors without unavailable category lookups
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Do not report absent samples as healthy zero-error measurements
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "valid_positions": 0,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            ### >>> Self_EvolveRec-BLOCK-START: Report the measured diagnostic sample count
            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
                "valid_positions": int(N),
            }
            ### <<< Self_EvolveRec-BLOCK-END

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Separate measured margin counts from unavailable category breakdowns
        breakdown = {
            "eps": float(eps),
            "valid_positions": margin_metrics["valid_positions"],
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "category_metadata_available": False,
            "low_margin_top_categories": None,
            "wrong_rank_top_categories": None,
        }
        ### <<< Self_EvolveRec-BLOCK-END

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Measure padding correctness and paired context sensitivity before crop experiments
    @torch.no_grad()
    def analyze_context_sensitivity(self, train_loader, max_rows=64, seed=42):
        rng = np.random.default_rng(seed)
        records = []
        padding_errors = []
        was_training = self.model.training
        self.model.eval()
        try:
            for step, batch in enumerate(train_loader):
                if step >= 2 or len(records) >= max_rows:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                    continue
                seq, pos, neg = [
                    torch.as_tensor(value, device=self.device, dtype=torch.long)
                    for value in batch[1:4]
                ]
                if seq.ndim != 2 or pos.shape != seq.shape or neg.shape != seq.shape:
                    raise ValueError("Diagnostic inputs must be aligned B x L tensors")
                for row in range(seq.size(0)):
                    if len(records) >= max_rows:
                        break
                    x = seq[row : row + 1]
                    positions = torch.nonzero(x[0] != 0, as_tuple=False).flatten()
                    n = int(positions.numel())
                    if n < 3:
                        continue
                    last = int(positions[-1].item())
                    positive = pos[row, last]
                    negative = neg[row, last]
                    if positive.item() == 0 or negative.item() == 0:
                        continue
                    if positive.item() == negative.item():
                        continue
                    # The fixed target is the original latest adjacent target.
                    # Prefix and suffix interventions remove equal event counts.
                    k = int(rng.integers(1, n - 1))
                    prefix = x.clone()
                    suffix = x.clone()
                    prefix[0, positions[:k]] = 0
                    suffix[0, positions[-k:]] = 0
                    variants = torch.cat((x, prefix, suffix), dim=0)
                    valid = variants != 0
                    original_positions = torch.arange(
                        variants.size(1), device=self.device
                    ).expand_as(variants)
                    final_positions = original_positions.masked_fill(~valid, -1).amax(
                        dim=1
                    )
                    outputs = self.view.log2feats(variants)
                    features = outputs[
                        torch.arange(3, device=self.device), final_positions
                    ]
                    candidates = (
                        torch.stack((positive, negative)).view(1, 2).expand(3, 2)
                    )
                    scores = self.view.predict_score(features, candidates).float()
                    if scores.shape != (3, 2):
                        raise ValueError(
                            "Context probe requires aligned B x K candidate scores"
                        )
                    if not torch.isfinite(scores).all().item():
                        raise FloatingPointError("Nonfinite context diagnostic scores")
                    margins = scores[:, 0] - scores[:, 1]
                    # Zero-based positive ranks in the same two-item pool.
                    # Ties follow positive-first ordering, as in validation.
                    ranks = (scores[:, 1] > scores[:, 0]).float()
                    cosines = F.cosine_similarity(
                        features[0:1].expand(2, -1).float(),
                        features[1:].float(),
                        dim=-1,
                    )
                    if not torch.isfinite(features).all().item():
                        raise FloatingPointError(
                            "Nonfinite context diagnostic features"
                        )
                    # A padding-safe GRU should agree with the identical real
                    # history after leading padding is removed. Never pack the
                    # original left-padded layout using count-based lengths.
                    # Compare identical real events under different padding.
                    # Also test full-length rows, which have no padding to trim.
                    compact = x[:, positions]
                    if n < x.size(1):
                        compact_feature = self.view.log2feats(compact)[:, -1]
                        padded_feature = features[0:1]
                    else:
                        padded = F.pad(compact, (1, 0), value=0)
                        compact_feature = features[0:1]
                        padded_feature = self.view.log2feats(padded)[:, -1]
                    error = (
                        (compact_feature.float() - padded_feature.float()).abs().max()
                    )
                    if not torch.isfinite(error).item():
                        raise FloatingPointError(
                            "Nonfinite padding diagnostic features"
                        )
                    padding_errors.append(float(error.item()))
                    records.append(
                        {
                            "length": n,
                            "removed": k,
                            "prefix_margin_delta": float(
                                (margins[1] - margins[0]).item()
                            ),
                            "suffix_margin_delta": float(
                                (margins[2] - margins[0]).item()
                            ),
                            "prefix_rank_delta": float((ranks[1] - ranks[0]).item()),
                            "suffix_rank_delta": float((ranks[2] - ranks[0]).item()),
                            "prefix_feature_cosine": float(cosines[0].item()),
                            "suffix_feature_cosine": float(cosines[1].item()),
                        }
                    )
        finally:
            self.model.train(was_training)

        def summarize(rows):
            keys = (
                "length",
                "removed",
                "prefix_margin_delta",
                "suffix_margin_delta",
                "prefix_rank_delta",
                "suffix_rank_delta",
                "prefix_feature_cosine",
                "suffix_feature_cosine",
            )
            return {
                "sample_count": len(rows),
                **{
                    key
                    + "_mean": (
                        float(np.mean([record[key] for record in rows]))
                        if rows
                        else None
                    )
                    for key in keys
                },
            }

        return {
            "overall": summarize(records),
            "history_length_strata": {
                "3_to_9": summarize([r for r in records if r["length"] < 10]),
                "10_to_49": summarize([r for r in records if 10 <= r["length"] < 50]),
                "50_plus": summarize([r for r in records if r["length"] >= 50]),
            },
            "padding_comparisons": len(padding_errors),
            "padding_feature_max_abs_error": (
                max(padding_errors) if padding_errors else None
            ),
            "padding_check_tolerance": 1e-5,
            "padding_check_passed": (
                max(padding_errors) <= 1e-5 if padding_errors else None
            ),
            "timestamp_gap_strata": None,
        }

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Audit the existing packed GRU against a verified float64 padding-skipping reference
    @torch.no_grad()
    def analyze_gru_padding_audit(self, batches):
        import copy

        gru = self.model.gru
        if gru.bidirectional or not gru.batch_first:
            raise ValueError("Padding audit requires the supplied unidirectional GRU")
        width = max([5] + [int(batch[1].shape[1]) for batch in batches])
        device = self.view.item_emb.weight.device
        sequence = torch.zeros((6, width), dtype=torch.long, device=device)
        sequence[0, -3:] = torch.tensor([1, 2, 3], device=device)
        sequence[1, -1] = 2
        sequence[2] = torch.arange(width, device=device) % self.model.itemnum + 1
        sequence[3, -3:] = torch.tensor([3, 2, 1], device=device)
        sequence[5, -3:] = torch.tensor([1, 2, 3], device=device)
        sequence = torch.where(
            sequence > 0,
            (sequence - 1).remainder(self.model.itemnum) + 1,
            sequence,
        )
        valid = sequence.ne(0)
        candidates = torch.arange(1, min(self.model.itemnum, 100) + 1, device=device)
        was_training = self.model.training
        self.model.eval()
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            # Copy only the recurrent module, never the full catalog embedding.
            reference_gru = copy.deepcopy(gru).cpu().double().eval()
            embedded = self.view.item_emb(sequence).detach().cpu().double()
            cpu_valid = valid.cpu()
            states = [
                embedded.new_zeros(sequence.size(0), gru.hidden_size)
                for _ in range(gru.num_layers)
            ]
            reference = embedded.new_zeros(sequence.size(0), width, gru.hidden_size)
            for position in range(width):
                layer_input = embedded[:, position]
                for layer in range(gru.num_layers):
                    previous = states[layer]
                    input_gates = F.linear(
                        layer_input,
                        getattr(reference_gru, f"weight_ih_l{layer}"),
                        getattr(reference_gru, f"bias_ih_l{layer}", None),
                    )
                    hidden_gates = F.linear(
                        previous,
                        getattr(reference_gru, f"weight_hh_l{layer}"),
                        getattr(reference_gru, f"bias_hh_l{layer}", None),
                    )
                    ir, iz, inn = input_gates.chunk(3, dim=-1)
                    hr, hz, hn = hidden_gates.chunk(3, dim=-1)
                    reset = torch.sigmoid(ir + hr)
                    update = torch.sigmoid(iz + hz)
                    new = torch.tanh(inn + reset * hn)
                    proposed = (1.0 - update) * new + update * previous
                    states[layer] = torch.where(
                        cpu_valid[:, position, None], proposed, previous
                    )
                    layer_input = states[layer]
                reference[:, position] = torch.where(
                    cpu_valid[:, position, None],
                    states[-1],
                    torch.zeros_like(states[-1]),
                )

            def errors(actual, expected, atol, rtol):
                actual = actual.detach().cpu().double()
                expected = expected.detach().cpu().double()
                if actual.shape != expected.shape:
                    raise ValueError(
                        f"GRU audit shape mismatch: {actual.shape} vs {expected.shape}"
                    )
                difference = (actual - expected).abs()
                finite = torch.isfinite(actual) & torch.isfinite(expected)
                failed = (~finite) | (difference > atol + rtol * expected.abs())
                return {
                    "elements": int(actual.numel()),
                    "failing_elements": int(failed.sum()),
                    "max_absolute_error": (
                        float(difference.max()) if difference.numel() else 0.0
                    ),
                    "relative_l2_error": float(
                        torch.linalg.vector_norm(actual - expected)
                        / torch.linalg.vector_norm(expected).clamp_min(1e-30)
                    ),
                }

            native_failures = 0
            native_max_error = 0.0
            for row in range(sequence.size(0)):
                if not cpu_valid[row].any():
                    continue
                native, _ = reference_gru(embedded[row, cpu_valid[row]].unsqueeze(0))
                result = errors(native[0], reference[row, cpu_valid[row]], 1e-7, 1e-7)
                native_failures += result["failing_elements"]
                native_max_error = max(native_max_error, result["max_absolute_error"])
            if native_failures:
                return {
                    "reference_verified": False,
                    "reference_failing_elements": native_failures,
                    "reference_max_absolute_error": native_max_error,
                    "execution_checks": None,
                }

            final_reference = states[-1]
            candidate_embeddings = (
                self.view.item_emb(candidates).detach().cpu().double()
            )
            reference_scores = final_reference @ candidate_embeddings.T
            permutation = torch.tensor([4, 2, 0, 5, 1, 3], device=device)

            def measure_execution():
                features = self.view.log2feats(sequence)
                final = self.model.encode(sequence)
                # Explicit per-row pools avoid the aligned-target ambiguity
                # when the catalog candidate count equals the batch size.
                candidate_pools = candidates.unsqueeze(0).expand(final.size(0), -1)
                scores = self.view.predict_score(final, candidate_pools)
                checks = {
                    "aligned_features": errors(features, reference, 1e-5, 1.3e-6),
                    "final_features": errors(final, final_reference, 1e-5, 1.3e-6),
                    "candidate_scores": errors(scores, reference_scores, 1e-5, 1.3e-6),
                    "permuted_batch": errors(
                        self.view.log2feats(sequence[permutation]),
                        features[permutation],
                        1e-5,
                        1.3e-6,
                    ),
                    "repeated_call": errors(
                        self.view.log2feats(sequence), features, 1e-5, 1.3e-6
                    ),
                }
                padded = F.pad(sequence, (3, 0))
                checks["additional_padding"] = errors(
                    self.view.log2feats(padded)[:, 3:], features, 1e-5, 1.3e-6
                )
                isolated = torch.zeros_like(features)
                for row in range(sequence.size(0)):
                    positions = valid[row].nonzero(as_tuple=True)[0]
                    if positions.numel():
                        output = self.view.log2feats(sequence[row, positions][None])
                        isolated[row, positions] = output[0]
                checks["isolated_histories"] = errors(isolated, features, 1e-5, 1.3e-6)
                # Unrelated intervening calls must not leave recurrent state.
                self.view.log2feats(sequence.flip(0))
                checks["after_unrelated_call"] = errors(
                    self.view.log2feats(sequence), features, 1e-5, 1.3e-6
                )
                actual_scores = scores.detach().cpu().double()
                actual_order = torch.argsort(
                    actual_scores, descending=True, stable=True
                )
                expected_order = torch.argsort(
                    reference_scores, descending=True, stable=True
                )
                k = min(10, candidates.numel())
                overlaps = [
                    len(set(a[:k].tolist()) & set(b[:k].tolist())) / k
                    for a, b in zip(actual_order, expected_order)
                ]
                # The first fixed candidate is an audit target, not a label.
                actual_rank = (actual_order == 0).long().argmax(dim=-1)
                expected_rank = (expected_order == 0).long().argmax(dim=-1)
                return {
                    "checks": checks,
                    "passed": all(v["failing_elements"] == 0 for v in checks.values()),
                    "top10_overlap_mean": float(np.mean(overlaps)),
                    "fixed_candidate_rank_changes": int(
                        (actual_rank != expected_rank).sum()
                    ),
                    "reference_rank10_gap_min": (
                        float(
                            (
                                reference_scores[cpu_valid.any(dim=1)]
                                .sort(descending=True)
                                .values[:, 9]
                                - reference_scores[cpu_valid.any(dim=1)]
                                .sort(descending=True)
                                .values[:, 10]
                            ).min()
                        )
                        if candidates.numel() > 10 and bool(cpu_valid.any(dim=1).any())
                        else None
                    ),
                }

            with torch.autocast(device_type=device.type, enabled=False):
                executions = {"configured_backend": measure_execution()}
                if not executions["configured_backend"]["passed"]:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    executions["tf32_disabled"] = measure_execution()
                    if not executions["tf32_disabled"]["passed"]:
                        with torch.backends.cudnn.flags(enabled=False):
                            executions["cudnn_disabled"] = measure_execution()
                        if not executions["cudnn_disabled"]["passed"]:
                            # Exercise the actual encoder implementation in CPU
                            # float64 without copying the full catalog embedding.
                            # Fixed fixtures access only this vocabulary prefix.
                            prefix_size = (
                                int(max(sequence.max().item(), candidates.max().item()))
                                + 1
                            )
                            cpu_model = copy.copy(self.model)
                            cpu_model._modules = self.model._modules.copy()
                            cpu_model.item_emb = torch.nn.Embedding.from_pretrained(
                                self.view.item_emb.weight[:prefix_size]
                                .detach()
                                .to(device="cpu", dtype=torch.float64)
                                .clone(),
                                freeze=True,
                                padding_idx=0,
                            )
                            cpu_model.gru = reference_gru
                            cpu_model.dropout = copy.deepcopy(self.model.dropout).eval()
                            cpu_sequence = sequence.cpu()
                            cpu_candidates = candidates.cpu()
                            cpu_features = cpu_model.log2feats(cpu_sequence)
                            cpu_final = cpu_model.encode(cpu_sequence)
                            cpu_scores = cpu_model.predict_score(
                                cpu_final,
                                cpu_candidates[None].expand(cpu_sequence.size(0), -1),
                            )
                            cpu_checks = {
                                "aligned_features": errors(
                                    cpu_features, reference, 1e-7, 1e-7
                                ),
                                "final_features": errors(
                                    cpu_final, final_reference, 1e-7, 1e-7
                                ),
                                "candidate_scores": errors(
                                    cpu_scores, reference_scores, 1e-7, 1e-7
                                ),
                                "additional_padding": errors(
                                    cpu_model.log2feats(F.pad(cpu_sequence, (3, 0)))[
                                        :, 3:
                                    ],
                                    cpu_features,
                                    1e-7,
                                    1e-7,
                                ),
                            }
                            executions["cpu_float64"] = {
                                "checks": cpu_checks,
                                "passed": all(
                                    value["failing_elements"] == 0
                                    for value in cpu_checks.values()
                                ),
                            }
            return {
                "reference_verified": True,
                "reference_max_absolute_error": native_max_error,
                "fp32_atol": 1e-5,
                "fp32_rtol": 1.3e-6,
                "fp64_atol": 1e-7,
                "fp64_rtol": 1e-7,
                "fixture_width": width,
                "execution_checks": executions,
                "torch_version": str(torch.__version__),
                "device": str(device),
                "configured_matmul_tf32": old_matmul_tf32,
                "configured_cudnn_tf32": old_cudnn_tf32,
                "configured_cudnn_enabled": torch.backends.cudnn.enabled,
                "architecture_changed": False,
            }
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            self.model.train(was_training)

    ### <<< Self_EvolveRec-BLOCK-END

    def run_full_diagnosis(self, train_loader):
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        ### >>> Self_EvolveRec-BLOCK-START: Snapshot diagnostic batches and restore model mode after all probes
        was_training = self.view.model.training
        self.view.model.eval()
        try:
            # Own each snapshot: loaders may reuse mutable NumPy/tensor buffers.
            # CPU storage bounds GPU residency while probes share identical data.
            batches = []
            iterator = iter(train_loader)
            for _ in range(10):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                    raise ValueError(
                        "Diagnostic loader must yield (None, sequence, positive, negative)"
                    )
                values = tuple(
                    torch.as_tensor(value)
                    .detach()
                    .to(device="cpu", dtype=torch.long)
                    .clone()
                    for value in batch[1:4]
                )
                if values[0].ndim != 2 or any(
                    value.shape != values[0].shape for value in values[1:]
                ):
                    raise ValueError("Diagnostic inputs must be aligned B x L tensors")
                batches.append((None, *values))
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            eps = 0.5
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    batches, num_batches=10, eps=eps, topk_categories=20
                )
            )
            padding_audit = self.analyze_gru_padding_audit(batches)
            context_metrics = self.analyze_context_sensitivity(batches)
            if context_metrics["padding_check_passed"] is False:
                print(
                    "WARNING: GRU padding-invariance check failed; context deltas "
                    "may include padding-induced recurrent-state changes. "
                    "Correct training and prediction recurrence before interpreting "
                    "cropping effects.",
                    flush=True,
                )
        finally:
            self.view.model.train(was_training)
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                ### >>> Self_EvolveRec-BLOCK-START: Expose measured context and padding diagnostics
                "pos_neg_margin_category_breakdown": breakdown,
                "context_sensitivity": context_metrics,
                "gru_padding_audit": padding_audit,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define verified recurrence checks separately from recommendation accuracy
                "gru_padding_audit": (
                    "Fixed training-independent item-ID fixtures cover empty, singleton, "
                    "short, full loader-width, tied-length and mixed-length histories. "
                    "A CPU float64 reference applies PyTorch reset/update/new GRU gate "
                    "equations, including reset-after-recurrent-affine candidate bias, "
                    "and preserves every layer state at padding. Every independent "
                    "history starts from zero. The reference must first match native "
                    "individually unpadded CPU float64 GRU outputs at atol=rtol=1e-7. "
                    "Then eval-mode ModelView outputs, final features, raw candidate "
                    "scores, isolated histories, batch permutations, added leading "
                    "padding, repeated calls and calls after unrelated histories are "
                    "checked with autocast disabled at atol=1e-5, rtol=1.3e-6. "
                    "Failing elements violate abs(actual-reference)<=atol+rtol*abs(reference), "
                    "including nonfinite elements. Relative L2 uses a 1e-30 denominator "
                    "floor. Backend escalation disables TF32, then cuDNN only after "
                    "failure, preserving thresholds and restoring original settings. "
                    "Persistent failure additionally checks the actual encoder on CPU "
                    "float64 at atol=rtol=1e-7, using identical recurrent parameters "
                    "and a copied embedding prefix containing every fixture/candidate ID. "
                    "This temporary diagnostic copy is never fitted or deployed. CPU "
                    "checks cover aligned features, final features, scores and added "
                    "padding; other execution checks remain in the preceding backend stages. "
                    "The minimum rank-10/11 reference gap excludes empty histories, "
                    "whose all-zero scores otherwise force that minimum to zero. "
                    "Candidate IDs 1..min(itemnum,100) are fixed audit candidates, not "
                    "validation labels; rank changes refer to candidate 1. Stable sorting "
                    "breaks ties by candidate order. Top-10 overlap and the minimum "
                    "reference rank-10/11 score gap describe numerical rank sensitivity. "
                    "An unverified reference leaves execution checks unavailable. "
                    "The supplied encoder already compacts and packs real tokens, so "
                    "this audit leaves its architecture, loss, sampler, optimizer, "
                    "checkpoint selection and matching imported predictor unchanged. "
                    "Failures do not automatically establish a semantic defect; no "
                    "validation improvement, gradient equivalence or predictor-export "
                    "equivalence is established by these diagnostic measurements."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Define paired interventions without treating sensitivity as validation improvement
                "context_sensitivity": (
                    "At most 64 eligible rows from the first two cached training batches. "
                    "Each row has at least three real inputs and distinct nonzero latest "
                    "positive/negative targets. Remove k events from either prefix or suffix, "
                    "with k uniform in 1..n-2 using a dedicated seed-42 RNG. Both interventions "
                    "retain the original latest target and identical two-item candidate pool. "
                    "Margin deltas are perturbed minus full-context positive-negative margins; "
                    "rank deltas are changes in zero-based positive rank among those two items "
                    "(positive wins ties), so positive rank deltas mean deterioration. Feature "
                    "cosines measure sensitivity, not usefulness. Length and removed means "
                    "describe sampled input and removal event counts; length strata use original "
                    "input lengths. Suffix removal also increases the event gap to the fixed "
                    "target, so it is not a pure forgetting control. Padding error is the maximum "
                    "absolute final-feature difference for identical chronological real inputs "
                    "under different padding: compare original padded rows against compacted "
                    "real inputs; for full-length rows, compare against one added leading zero. "
                    "Measurements use eval mode. padding_check_passed reports whether the "
                    "maximum error is at most the declared absolute tolerance of 1e-5; this is "
                    "a numerical diagnostic, not a proof of recurrence correctness. Failure "
                    "confounds context deltas with padding-induced state changes. Null "
                    "denotes no eligible measurements. Timestamp gap strata are unavailable "
                    "because loader batches do not include timestamps. These are conditional "
                    "training diagnostics, not HR@10, unseen full-catalog ranks, or evidence "
                    "that cropped training improves validation. No model scores are fused."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Interpret embedding geometry without inventing behavioral diversity
                "embedding_collapse_score": (
                    "Mean off-diagonal cosine similarity of up to 2048 real item embeddings, "
                    "mapped from [-1,1] to [0,1]. About 0.5 corresponds to zero mean cosine; "
                    "values approaching 1 indicate directional alignment. This does not measure "
                    "recommendation diversity or unobserved simulator traits. Null means insufficient pairs."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define diagnostic availability and sampling limitations
                "pos_neg_margin_category_breakdown": (
                    "Contains measured valid-position, low-margin and wrong-rank counts with eps=0.5. "
                    "Valid positions require nonzero input, positive and negative IDs. Category "
                    "breakdowns are null because item categories are unavailable. All margin metrics "
                    "are null when no valid samples exist. These are sampled training margins, "
                    "not validation HR@10 or a full-catalog ranking measurement."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
