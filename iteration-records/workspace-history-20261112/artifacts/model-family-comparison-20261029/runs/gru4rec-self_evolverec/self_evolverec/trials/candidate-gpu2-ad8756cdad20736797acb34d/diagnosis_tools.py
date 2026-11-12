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

            ### >>> Self_EvolveRec-BLOCK-START: Measure embedding alignment without a quadratic Gram matrix
            x = x.float()
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore model mode even when a standalone margin probe fails
        was_training = self.model.training
        self.model.eval()
        try:
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

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

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

    ### >>> Self_EvolveRec-BLOCK-START: Calibrate frozen GRU padding errors against singleton and explicit recurrence
    @torch.no_grad()
    def analyze_padding_calibration(self, batches, max_rows=16):
        device = torch.device(self.device)
        atol, rtol = 1e-5, 1.3e-6
        result = {
            "sample_count": 0,
            "atol": atol,
            "rtol": rtol,
            "absolute_only_tolerance": 1e-5,
            "valid_state_max_abs_error": 0.0,
            "final_feature_max_abs_error": 0.0,
            "score_max_abs_error": 0.0,
            "max_normalized_tolerance_violation": 0.0,
            "explicit_reference_max_abs_error": 0.0,
            "repeat_max_abs_error": 0.0,
            "top10_changed_rows": 0,
            "first_divergent_valid_token": None,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "dtype": str(self.view.item_emb.weight.dtype),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }

        def compare(actual, reference, key):
            if actual.shape != reference.shape or actual.numel() == 0:
                raise ValueError(f"Invalid padding comparison shapes for {key}")
            if not (
                torch.isfinite(actual).all().item()
                and torch.isfinite(reference).all().item()
            ):
                raise FloatingPointError(f"Nonfinite padding comparison: {key}")
            error = (actual.double() - reference.double()).abs()
            limit = atol + rtol * reference.double().abs()
            result[key] = max(result[key], float(error.max().item()))
            result["max_normalized_tolerance_violation"] = max(
                result["max_normalized_tolerance_violation"],
                float((error / limit).max().item()),
            )
            return (error > limit).reshape(error.shape[0], -1).any(dim=1)

        def explicit_features(tokens):
            # PyTorch GRU gate order is reset, update, new. In eval mode the
            # recurrent inter-layer dropout is inactive. No padded step runs.
            embedded = self.model.dropout(self.view.item_emb(tokens))
            layer_input = embedded
            gru = self.model.gru
            for layer in range(gru.num_layers):
                w_i = getattr(gru, f"weight_ih_l{layer}")
                w_h = getattr(gru, f"weight_hh_l{layer}")
                b_i = getattr(gru, f"bias_ih_l{layer}", None)
                b_h = getattr(gru, f"bias_hh_l{layer}", None)
                hidden = embedded.new_zeros(1, gru.hidden_size)
                states = []
                for step in range(tokens.size(1)):
                    ir, iz, inn = F.linear(layer_input[:, step], w_i, b_i).chunk(3, -1)
                    hr, hz, hn = F.linear(hidden, w_h, b_h).chunk(3, -1)
                    reset = torch.sigmoid(ir + hr)
                    update = torch.sigmoid(iz + hz)
                    proposed = torch.tanh(inn + reset * hn)
                    hidden = (1.0 - update) * proposed + update * hidden
                    states.append(hidden)
                layer_input = torch.stack(states, dim=1)
            current = embedded
            gate = torch.sigmoid(
                self.model.recent_gate(torch.cat((layer_input, current), dim=-1))
            )
            return (1.0 - gate) * layer_input + gate * self.model.recent_project(
                current
            )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.autocast(device_type=device.type, enabled=False):
                for batch in batches[:2]:
                    remaining = max_rows - result["sample_count"]
                    if remaining <= 0:
                        break
                    seq = torch.as_tensor(batch[1], device=device, dtype=torch.long)
                    seq = seq[seq.ne(0).any(dim=1)][:remaining]
                    if seq.size(0) == 0:
                        continue
                    full = self.view.log2feats(seq)
                    repeated = self.view.log2feats(seq)
                    padded = self.view.log2feats(F.pad(seq, (3, 0)))[:, 3:]
                    order = torch.arange(seq.size(0) - 1, -1, -1, device=device)
                    permuted = self.view.log2feats(seq[order])[order]
                    items = torch.arange(
                        1, min(self.model.itemnum, 100) + 1, device=device
                    )
                    for row in range(seq.size(0)):
                        positions = seq[row].ne(0).nonzero(as_tuple=True)[0]
                        tokens = seq[row, positions].unsqueeze(0)
                        reference = self.view.log2feats(tokens)[0]
                        explicit = explicit_features(tokens)[0]
                        # Include independent recurrence and repeated execution
                        # in the declared tolerance check; reject nonfinite values.
                        compare(
                            reference,
                            explicit,
                            "explicit_reference_max_abs_error",
                        )
                        compare(repeated[row], full[row], "repeat_max_abs_error")
                        # Explicit B x K pools avoid the aligned-target ambiguity
                        # when a catalog contains exactly one real item.
                        candidate_pool = items.unsqueeze(0)
                        reference_scores = self.view.predict_score(
                            reference[-1:], candidate_pool
                        )
                        for variant in (full, padded, permuted):
                            actual = variant[row, positions]
                            divergent = compare(
                                actual, reference, "valid_state_max_abs_error"
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
                                }
                            compare(
                                actual[-1:],
                                reference[-1:],
                                "final_feature_max_abs_error",
                            )
                            scores = self.view.predict_score(
                                actual[-1:], candidate_pool
                            )
                            compare(scores, reference_scores, "score_max_abs_error")
                        # Stable sorting makes candidate-ID order the tie rule.
                        scores = self.view.predict_score(
                            full[row, positions[-1]].unsqueeze(0), candidate_pool
                        )
                        k = min(10, items.numel())
                        changed = not torch.equal(
                            torch.argsort(scores[0], descending=True, stable=True)[:k],
                            torch.argsort(
                                reference_scores[0], descending=True, stable=True
                            )[:k],
                        )
                        result["top10_changed_rows"] += int(changed)
                        result["sample_count"] += 1
        finally:
            self.model.train(was_training)
        if result["sample_count"]:
            result["elementwise_check_passed"] = (
                result["max_normalized_tolerance_violation"] <= 1.0
            )
            result["absolute_only_check_passed"] = (
                result["valid_state_max_abs_error"] <= 1e-5
            )
        else:
            for key in tuple(result):
                if key.endswith("error") or key == "max_normalized_tolerance_violation":
                    result[key] = None
            result["elementwise_check_passed"] = None
            result["absolute_only_check_passed"] = None
        return result

    ### <<< Self_EvolveRec-BLOCK-END
    def run_full_diagnosis(self, train_loader):
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        ### >>> Self_EvolveRec-BLOCK-START: Share immutable probe batches and restore model mode on failure
        was_training = self.view.model.training
        self.view.model.eval()
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
                    value.shape != values[0].shape for value in values[1:]
                ):
                    raise ValueError(
                        "Diagnostic tensors must have aligned B x L shapes"
                    )
                if any(
                    ((value < 0) | (value > self.model.itemnum)).any().item()
                    for value in values
                ):
                    raise ValueError("Diagnostic IDs exceed the trained vocabulary")
                batches.append((None, *values))
            padding_calibration = self.analyze_padding_calibration(batches)
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            eps = 0.5
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    batches, num_batches=10, eps=eps, topk_categories=20
                )
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
                ### >>> Self_EvolveRec-BLOCK-START: Report measured frozen-checkpoint padding calibration
                "pos_neg_margin_category_breakdown": breakdown,
                "padding_calibration": padding_calibration,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Distinguish precision calibration from semantic and validation conclusions
                "padding_calibration": (
                    "Eval/no-grad, autocast-disabled measurements on at most 16 nonempty "
                    "rows from the first two cached training batches. Compare chronological "
                    "unpadded singleton features with original batches, three extra leading "
                    "zeros and reversed batch order. Errors cover all valid states, final "
                    "features and scores for the first at most 100 real item IDs. The "
                    "elementwise criterion is abs(error)<=1e-5+1.3e-6*abs(reference); "
                    "normalized violation is error divided by that bound, including "
                    "explicit-reference and repeated-execution comparisons. Nonfinite "
                    "features or scores raise an error. The separate "
                    "absolute-only state criterion remains 1e-5. Explicit-reference error "
                    "compares singleton features with an eval-mode recurrence using the "
                    "trained GRU weights, PyTorch gate equations and joint recent-item "
                    "readout. Repeat error measures identical repeated batch executions. "
                    "First divergence gives a probe row and zero-based chronological token "
                    "index. Top10 changes compare ordered candidate lists for original batch "
                    "versus singleton, with stable candidate-order tie handling. These are "
                    "fixed training-history candidate probes, not full-catalog rankings or "
                    "validation HR@10. Precision settings and software versions are recorded. "
                    "A failed tolerance alone does not establish padding leakage; unresolved "
                    "errors confound mechanistic context claims. No samples gives null errors "
                    "and pass flags. No model parameters or scores are averaged."
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
