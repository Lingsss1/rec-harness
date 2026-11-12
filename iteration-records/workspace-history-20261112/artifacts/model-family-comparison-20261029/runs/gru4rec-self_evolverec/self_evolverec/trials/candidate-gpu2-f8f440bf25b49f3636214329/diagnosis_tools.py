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

            ### >>> Self_EvolveRec-BLOCK-START: Compute pairwise mean cosine without a quadratic Gram matrix
            # Accumulate in FP64 without relying on the caller's TF32 setting.
            # The identity includes the actual diagonal, including zero vectors.
            x = x.double()
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore model mode for standalone margin diagnostics
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.autocast(
                device_type=torch.device(self.device).type, enabled=False
            ):
                return self._measure_pos_neg_margins(
                    dataloader, num_batches, eps, topk_categories
                )
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def _measure_pos_neg_margins(self, dataloader, num_batches, eps, topk_categories):
        ### <<< Self_EvolveRec-BLOCK-END
        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and reject malformed diagnostic batches
        iterator = iter(dataloader)
        for step in range(num_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError("Expected (None, sequence, positive, negative)")
            _, seq, pos, neg = batch
            shapes = [tuple(torch.as_tensor(value).shape) for value in (seq, pos, neg)]
            if len(shapes[0]) != 2 or any(shape != shapes[0] for shape in shapes[1:]):
                raise ValueError("Diagnostic tensors must be aligned B x L")
            for value in (seq, pos, neg):
                ids = torch.as_tensor(value)
                if ((ids < 0) | (ids > self.model.itemnum)).any().item():
                    raise ValueError("Diagnostic IDs exceed the trained vocabulary")
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Use aligned tensor inputs for family-specific GRU diagnostics
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.view.log2feats(seq_t)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
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

    ### >>> Self_EvolveRec-BLOCK-START: Bounded precision gate with independent FP64 GRU reference
    @torch.no_grad()
    def analyze_padding_calibration(self, batches, max_rows=16):
        device = torch.device(self.device)
        was_training = self.model.training
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        atol, rtol = 1e-5, 1.3e-6
        result = {
            "sample_count": 0,
            "atol": atol,
            "rtol": rtol,
            "absolute_only_tolerance": 1e-5,
            "valid_state_max_abs_error": 0.0,
            "final_feature_max_abs_error": 0.0,
            "score_max_abs_error": 0.0,
            "fp64_reference_max_abs_error": 0.0,
            "repeat_max_abs_error": 0.0,
            "fresh_state_max_abs_error": 0.0,
            "max_normalized_tolerance_violation": 0.0,
            "top10_membership_changed_rows": 0,
            "top10_order_changed_rows": 0,
            "top10_within_set_order_changed_rows": 0,
            "minimum_tenth_eleventh_gap": None,
            "first_divergent_valid_token": None,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "dtype": str(self.view.item_emb.weight.dtype),
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }

        def compare(actual, reference, key):
            if actual.shape != reference.shape or not actual.numel():
                raise ValueError(f"Invalid precision comparison: {key}")
            if not (torch.isfinite(actual).all() and torch.isfinite(reference).all()):
                raise FloatingPointError(f"Nonfinite precision comparison: {key}")
            error = (actual.double() - reference.double()).abs()
            limit = atol + rtol * reference.double().abs()
            result[key] = max(result[key], float(error.max().item()))
            result["max_normalized_tolerance_violation"] = max(
                result["max_normalized_tolerance_violation"],
                float((error / limit).max().item()),
            )
            return (error > limit).reshape(error.shape[0], -1).any(dim=1)

        def reference_features(tokens):
            # Current supplied architecture: embedding -> GRU -> linear scores.
            # Promote weights BEFORE arithmetic; no trained module is mutated.
            # Gather first: promoting the entire catalog for every probe row
            # wastes O(catalog_size * hidden_size) temporary FP64 memory.
            current = F.embedding(tokens, self.view.item_emb.weight.detach()).double()
            gru = self.model.gru
            for layer in range(gru.num_layers):
                wi = getattr(gru, f"weight_ih_l{layer}").detach().double()
                wh = getattr(gru, f"weight_hh_l{layer}").detach().double()
                bi = getattr(gru, f"bias_ih_l{layer}", None)
                bh = getattr(gru, f"bias_hh_l{layer}", None)
                bi = None if bi is None else bi.detach().double()
                bh = None if bh is None else bh.detach().double()
                hidden = current.new_zeros(1, gru.hidden_size)
                states = []
                for step in range(tokens.size(1)):
                    ir, iz, inn = F.linear(current[:, step], wi, bi).chunk(3, -1)
                    hr, hz, hn = F.linear(hidden, wh, bh).chunk(3, -1)
                    reset = torch.sigmoid(ir + hr)
                    update = torch.sigmoid(iz + hz)
                    proposed = torch.tanh(inn + reset * hn)
                    hidden = (1.0 - update) * proposed + update * hidden
                    states.append(hidden)
                current = torch.stack(states, dim=1)
            return current[0]

        self.model.eval()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type=device.type, enabled=False):
                for batch in batches[:2]:
                    remaining = max_rows - result["sample_count"]
                    if remaining <= 0:
                        break
                    seq = torch.as_tensor(batch[1], dtype=torch.long, device=device)
                    seq = seq[seq.ne(0).any(dim=1)][:remaining]
                    if not seq.size(0):
                        continue
                    full = self.view.log2feats(seq)
                    repeated = self.view.log2feats(seq)
                    padded = self.view.log2feats(F.pad(seq, (3, 0)))[:, 3:]
                    order = torch.arange(seq.size(0) - 1, -1, -1, device=device)
                    permuted = self.view.log2feats(seq[order])[order]
                    duplicated = self.view.log2feats(torch.cat((seq, seq), dim=0))
                    # A -> different B -> A verifies that calls start fresh.
                    self.view.log2feats(seq.flip(1))
                    fresh = self.view.log2feats(seq)
                    items = torch.arange(
                        1, min(self.model.itemnum, 100) + 1, device=device
                    )
                    pool = items.unsqueeze(0)
                    for row in range(seq.size(0)):
                        positions = seq[row].ne(0).nonzero(as_tuple=True)[0]
                        tokens = seq[row, positions].unsqueeze(0)
                        singleton = self.view.log2feats(tokens)[0]
                        reference = reference_features(tokens)
                        compare(singleton, reference, "fp64_reference_max_abs_error")
                        compare(
                            full[row, positions],
                            reference,
                            "fp64_reference_max_abs_error",
                        )
                        compare(repeated[row], full[row], "repeat_max_abs_error")
                        compare(fresh[row], full[row], "fresh_state_max_abs_error")
                        reference_scores = (
                            reference[-1:]
                            @ F.embedding(items, self.view.item_emb.weight.detach())
                            .double()
                            .t()
                        )
                        singleton_scores = self.view.predict_score(singleton[-1:], pool)
                        compare(
                            singleton_scores, reference_scores, "score_max_abs_error"
                        )
                        k = min(10, items.numel())
                        reference_order = torch.argsort(
                            singleton_scores[0], descending=True, stable=True
                        )
                        membership_changed = ordering_changed = False
                        within_set_order_changed = False
                        if items.numel() >= 11:
                            gap = float(
                                (
                                    singleton_scores[0, reference_order[9]]
                                    - singleton_scores[0, reference_order[10]]
                                ).item()
                            )
                            previous = result["minimum_tenth_eleventh_gap"]
                            result["minimum_tenth_eleventh_gap"] = (
                                gap if previous is None else min(previous, gap)
                            )
                        variants = (
                            full,
                            padded,
                            permuted,
                            duplicated[: seq.size(0)],
                            duplicated[seq.size(0) :],
                        )
                        for variant in variants:
                            actual = variant[row, positions]
                            divergent = compare(
                                actual, singleton, "valid_state_max_abs_error"
                            )
                            if (
                                divergent.any()
                                and result["first_divergent_valid_token"] is None
                            ):
                                result["first_divergent_valid_token"] = {
                                    "probe_row": result["sample_count"],
                                    "chronological_index": int(
                                        divergent.nonzero(as_tuple=True)[0][0].item()
                                    ),
                                    "history_length": int(positions.numel()),
                                }
                            compare(
                                actual[-1:],
                                singleton[-1:],
                                "final_feature_max_abs_error",
                            )
                            scores = self.view.predict_score(actual[-1:], pool)
                            compare(scores, singleton_scores, "score_max_abs_error")
                            actual_order = torch.argsort(
                                scores[0], descending=True, stable=True
                            )
                            same_membership = torch.equal(
                                actual_order[:k].sort().values,
                                reference_order[:k].sort().values,
                            )
                            same_order = torch.equal(
                                actual_order[:k], reference_order[:k]
                            )
                            membership_changed |= not same_membership
                            ordering_changed |= not same_order
                            within_set_order_changed |= (
                                same_membership and not same_order
                            )
                        result["top10_membership_changed_rows"] += int(
                            membership_changed
                        )
                        result["top10_order_changed_rows"] += int(ordering_changed)
                        result["top10_within_set_order_changed_rows"] += int(
                            within_set_order_changed
                        )
                        result["sample_count"] += 1
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
            self.model.train(was_training)
        if result["sample_count"]:
            result["elementwise_check_passed"] = (
                result["max_normalized_tolerance_violation"] <= 1.0
            )
            result["absolute_only_check_passed"] = (
                max(
                    result["valid_state_max_abs_error"],
                    result["fp64_reference_max_abs_error"],
                )
                <= 1e-5
            )
            result["context_attribution_confounded"] = not (
                result["elementwise_check_passed"]
                and result["absolute_only_check_passed"]
            )
        else:
            for key in tuple(result):
                if key.endswith("error") or key == "max_normalized_tolerance_violation":
                    result[key] = None
            result["elementwise_check_passed"] = None
            result["absolute_only_check_passed"] = None
            result["context_attribution_confounded"] = None
        return result

    def run_full_diagnosis(self, train_loader):
        was_training = self.model.training
        self.model.eval()
        try:
            batches = []
            iterator = iter(train_loader)
            for _ in range(10):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                values = tuple(
                    torch.as_tensor(value).detach().to("cpu", dtype=torch.long).clone()
                    for value in batch[1:4]
                )
                if values[0].ndim != 2 or any(
                    v.shape != values[0].shape for v in values[1:]
                ):
                    raise ValueError("Diagnostic tensors must be aligned B x L")
                if any(
                    ((v < 0) | (v > self.model.itemnum)).any().item() for v in values
                ):
                    raise ValueError("Diagnostic IDs exceed the trained vocabulary")
                batches.append((None, *values))
            precision = self.analyze_padding_calibration(batches)
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    batches, num_batches=10, eps=0.5, topk_categories=20
                )
            )
        finally:
            self.model.train(was_training)
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "padding_calibration": precision,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Mean off-diagonal item-embedding cosine mapped to [0,1], using at most "
                    "2048 real items. 0.5 means zero mean cosine; alignment is not behavioral diversity."
                ),
                "pos_neg_margin_mean": "Mean positive-minus-negative score over valid sampled training positions.",
                "pos_neg_margin_p10": "Empirical lower tenth percentile of sampled training margins.",
                "pos_neg_neg_beats_pos_rate": "Fraction of valid sampled margins below zero.",
                "pos_neg_small_margin_rate": "Fraction of valid sampled margins below 0.5.",
                "pos_neg_margin_category_breakdown": (
                    "Measured valid, low-margin and wrong-rank counts. Category breakdowns are "
                    "unavailable. Valid positions require nonzero input, positive and negative IDs. "
                    "Absent margin samples return null metrics. These are not validation results."
                ),
                "padding_calibration": (
                    "At most 16 nonempty training rows from two cached batches, eval/no_grad, "
                    "autocast and both TF32 flags disabled and then restored. Compare chronological "
                    "singleton, packed batch, extra left padding, reversed batch order, duplicate "
                    "rows, repeated execution and fresh-state A-B-A calls. Independent FP64 "
                    "reference promotes embeddings and all trained GRU weights before arithmetic "
                    "and uses PyTorch r,z,n equations with both biases. Elementwise tolerance is "
                    "1e-5+1.3e-6*abs(reference); absolute-only state tolerance remains 1e-5. "
                    "Report state/final/score errors, normalized violations and first divergent "
                    "token with history length. Candidate probes use the first at most 100 real "
                    "IDs; top10 membership and ordered-list discrepancies are separate row counts "
                    "over any tested layout. Ordered-list discrepancies can include membership "
                    "changes; within-set ordering counts only layouts with identical membership "
                    "but different order. A row may have both kinds across different layouts. "
                    "Ordering alone does not change HR@10. The minimum tenth/eleventh score gap "
                    "is descriptive; empirical "
                    "errors are not certified uniform bounds. Failures confound context attribution "
                    "and do not alone establish semantic padding leakage. No samples gives null "
                    "errors/pass flags. These training probes do not measure validation HR@10."
                ),
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
