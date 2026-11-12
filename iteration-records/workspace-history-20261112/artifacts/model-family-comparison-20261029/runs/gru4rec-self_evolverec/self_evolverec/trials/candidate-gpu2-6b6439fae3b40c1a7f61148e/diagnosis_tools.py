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
    - Separate measured observations from hypotheses. Training margins and numerical probes do not establish validation improvements, immediate intent, compatibility, or behavioral diversity.
    - Missing measurements are unavailable evidence; never treat null values as passing checks.
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

            ### >>> Self_EvolveRec-BLOCK-START: Measure embedding alignment without a dense Gram matrix
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore mode and precision even when standalone margin diagnosis fails
        was_training = self.model.training
        old_mm = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        self.model.eval()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(
                device_type=torch.device(self.device).type, enabled=False
            ):
                return self._measure_pos_neg_margins(
                    dataloader, num_batches, eps, topk_categories
                )
        finally:
            self.model.train(was_training)
            torch.backends.cuda.matmul.allow_tf32 = old_mm
            torch.backends.cudnn.allow_tf32 = old_cudnn

    @torch.no_grad()
    def _measure_pos_neg_margins(self, dataloader, num_batches, eps, topk_categories):
        if num_batches < 0 or not np.isfinite(eps):
            raise ValueError("diagnostic batch count and margin threshold are invalid")
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and reject malformed diagnostic batches
        from itertools import islice

        for batch in islice(iter(dataloader), num_batches):
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError("expected (None, sequence, positive, negative)")
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned diagnostic IDs before embedding lookup
            values = tuple(torch.as_tensor(value) for value in (seq, pos, neg))
            if values[0].ndim != 2 or any(
                value.shape != values[0].shape for value in values[1:]
            ):
                raise ValueError("diagnostic tensors must be aligned B x L")
            for value in values:
                if value.dtype not in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError("diagnostic item IDs must use an integer dtype")
                if ((value < 0) | (value > self.model.itemnum)).any().item():
                    raise ValueError("diagnostic item IDs exceed the vocabulary")
            seq_t, pos_t, neg_t = tuple(
                value.to(device=self.device, dtype=torch.long) for value in values
            )
            log_feats = self.view.log2feats(seq_t)
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

    ### >>> Self_EvolveRec-BLOCK-START: Bounded precision regression for the supplied GRU and joint recent-mean readout
    @torch.no_grad()
    def analyze_padding_calibration(self, batches, max_rows=8):
        device = torch.device(self.device)
        old_mode = self.model.training
        old_mm = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        result = {
            "sample_count": 0,
            "valid_state_max_abs_error": 0.0,
            "final_feature_max_abs_error": 0.0,
            "score_max_abs_error": 0.0,
            "fp64_reference_max_abs_error": 0.0,
            "repeat_max_abs_error": 0.0,
            "fresh_state_max_abs_error": 0.0,
            "max_normalized_tolerance_violation": 0.0,
            "top10_membership_changed_rows": 0,
            "top10_order_changed_rows": 0,
            "first_divergent_valid_token": None,
            "atol": 1e-5,
            "rtol": 1.3e-6,
            "absolute_only_tolerance": 1e-5,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "dtype": str(self.view.item_emb.weight.dtype),
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }

        def compare(a, b, key):
            if a.shape != b.shape or not a.numel():
                raise ValueError(f"invalid calibration shapes for {key}")
            if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise FloatingPointError(f"nonfinite calibration values for {key}")
            error = (a.double() - b.double()).abs()
            limit = 1e-5 + 1.3e-6 * b.double().abs()
            result[key] = max(result[key], float(error.max().item()))
            result["max_normalized_tolerance_violation"] = max(
                result["max_normalized_tolerance_violation"],
                float((error / limit).max().item()),
            )
            return (error > limit).reshape(error.shape[0], -1).any(-1)

        def fp64_reference(tokens):
            current = F.embedding(tokens, self.view.item_emb.weight.detach()).double()
            embedded = current
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
                    hidden = (1 - update) * torch.tanh(
                        inn + reset * hn
                    ) + update * hidden
                    states.append(hidden)
                current = torch.stack(states, dim=1)
            # Independent causal pooling; promote both readout modules first.
            recent = torch.stack(
                [
                    embedded[:, max(0, t - self.model.recent_window + 1) : t + 1].mean(
                        1
                    )
                    for t in range(tokens.size(1))
                ],
                dim=1,
            )
            gate = torch.sigmoid(
                F.linear(
                    torch.cat((current, recent), -1),
                    self.model.recent_gate.weight.detach().double(),
                    self.model.recent_gate.bias.detach().double(),
                )
            )
            projected = F.linear(
                recent, self.model.recent_project.weight.detach().double()
            )
            return ((1 - gate) * current + gate * projected)[0]

        self.model.eval()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type=device.type, enabled=False):
                for batch in batches[:2]:
                    seq = torch.as_tensor(batch[1], device=device, dtype=torch.long)
                    seq = seq[seq.ne(0).any(1)][: max_rows - result["sample_count"]]
                    if not seq.size(0):
                        continue
                    full = self.view.log2feats(seq)
                    repeat = self.view.log2feats(seq)
                    self.view.log2feats(seq.flip(1))
                    fresh = self.view.log2feats(seq)
                    order = torch.arange(seq.size(0) - 1, -1, -1, device=device)
                    variants = (
                        full,
                        self.view.log2feats(F.pad(seq, (3, 0)))[:, 3:],
                        self.view.log2feats(seq[order])[order],
                        self.view.log2feats(torch.cat((seq, seq)))[seq.size(0) :],
                    )
                    items = torch.arange(
                        1, min(100, self.model.itemnum) + 1, device=device
                    )
                    for row in range(seq.size(0)):
                        positions = seq[row].ne(0).nonzero(as_tuple=True)[0]
                        tokens = seq[row, positions].unsqueeze(0)
                        ref = self.view.log2feats(tokens)[0]
                        independent = fp64_reference(tokens)
                        compare(ref, independent, "fp64_reference_max_abs_error")
                        compare(
                            full[row, positions],
                            independent,
                            "fp64_reference_max_abs_error",
                        )
                        compare(repeat[row], full[row], "repeat_max_abs_error")
                        compare(fresh[row], full[row], "fresh_state_max_abs_error")
                        ref_scores = self.view.predict_score(ref[-1:], items[None])[0]
                        independent_scores = (
                            independent[-1:]
                            @ F.embedding(items, self.view.item_emb.weight.detach())
                            .double()
                            .t()
                        )[0]
                        compare(ref_scores, independent_scores, "score_max_abs_error")
                        ref_top = ref_scores.argsort(descending=True, stable=True)[:10]
                        member_changed = order_changed = False
                        for variant in variants:
                            actual = variant[row, positions]
                            divergent = compare(
                                actual, ref, "valid_state_max_abs_error"
                            )
                            compare(
                                actual[-1:], ref[-1:], "final_feature_max_abs_error"
                            )
                            if (
                                divergent.any()
                                and result["first_divergent_valid_token"] is None
                            ):
                                result["first_divergent_valid_token"] = {
                                    "probe_row": result["sample_count"],
                                    "chronological_index": int(
                                        divergent.nonzero()[0, 0]
                                    ),
                                }
                            scores = self.view.predict_score(actual[-1:], items[None])[
                                0
                            ]
                            compare(scores, ref_scores, "score_max_abs_error")
                            top = scores.argsort(descending=True, stable=True)[:10]
                            member_changed |= not torch.equal(
                                top.sort().values, ref_top.sort().values
                            )
                            order_changed |= not torch.equal(top, ref_top)
                        result["top10_membership_changed_rows"] += int(member_changed)
                        result["top10_order_changed_rows"] += int(order_changed)
                        result["sample_count"] += 1
                    if result["sample_count"] >= max_rows:
                        break
        finally:
            self.model.train(old_mode)
            torch.backends.cuda.matmul.allow_tf32 = old_mm
            torch.backends.cudnn.allow_tf32 = old_cudnn
        if result["sample_count"]:
            result["elementwise_check_passed"] = (
                result["max_normalized_tolerance_violation"] <= 1
            )
            result["absolute_only_check_passed"] = (
                max(
                    result["valid_state_max_abs_error"],
                    result["fp64_reference_max_abs_error"],
                )
                <= 1e-5
            )
        else:
            for key in result:
                if key.endswith("error") or key == "max_normalized_tolerance_violation":
                    result[key] = None
            result["elementwise_check_passed"] = None
            result["absolute_only_check_passed"] = None
        return result

    ### <<< Self_EvolveRec-BLOCK-END

    def run_full_diagnosis(self, train_loader):
        ### >>> Self_EvolveRec-BLOCK-START: Record static review scope without claiming executed checks
        # Static review covered augmentation arguments and call sites, causal
        # input deletion, diagnostic shape/ID validation, mode/precision cleanup,
        # bounded loader consumption, and independent reference scoring.
        # Architecture, objective, seeds, routes and one checkpoint/category
        # remain consistent. Runtime checks and HR improvement are unverified.
        print(">>> Running bounded GRU diagnosis...")
        ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Cache bounded diagnostic batches and restore numerical settings and model mode
        was_training = self.model.training
        old_mm = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        self.model.eval()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            batches = []
            iterator = iter(train_loader)
            for _ in range(10):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    raise ValueError("expected (None, sequence, positive, negative)")
                values = tuple(torch.as_tensor(v).detach() for v in batch[1:])
                if any(
                    v.dtype
                    not in (
                        torch.uint8,
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                    )
                    for v in values
                ):
                    raise ValueError("diagnostic item IDs must use an integer dtype")
                values = tuple(v.to("cpu", dtype=torch.long).clone() for v in values)
                if values[0].ndim != 2 or any(
                    v.shape != values[0].shape for v in values
                ):
                    raise ValueError("diagnostic tensors must be aligned B x L")
                if any(((v < 0) | (v > self.model.itemnum)).any() for v in values):
                    raise ValueError("diagnostic item IDs exceed the vocabulary")
                batches.append((None, *values))
            with torch.autocast(
                device_type=torch.device(self.device).type, enabled=False
            ):
                padding_calibration = self.analyze_padding_calibration(batches)
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=10, eps=eps, topk_categories=20
                    )
                )
        finally:
            self.model.train(was_training)
            torch.backends.cuda.matmul.allow_tf32 = old_mm
            torch.backends.cudnn.allow_tf32 = old_cudnn
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                ### >>> Self_EvolveRec-BLOCK-START: Return measured numerical regression results
                "pos_neg_margin_category_breakdown": breakdown,
                "padding_calibration": padding_calibration,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define bounded calibration measurements without generalization claims
                "padding_calibration": (
                    "At most eight nonempty rows from two cached training batches. Eval/no_grad, "
                    "autocast and both TF32 flags disabled; mode and flags restored afterward. "
                    "Compare chronological singleton features with original batches, extra left "
                    "padding, reversed batch order, duplicate rows, repeated calls and fresh-state "
                    "A-B-A calls. FP64 reference promotes trained GRU weights before arithmetic, "
                    "uses all recurrent layers and both biases, and independently applies the "
                    "joint causal recent-mean readout. Singleton and original-batch states are "
                    "compared with this reference; candidate scores are also compared with "
                    "FP64 dot products. Elementwise tolerance is "
                    "1e-5+1.3e-6*abs(reference); separate absolute-only state tolerance is 1e-5. "
                    "Top10 membership and ordered-list changes are distinct row counts over the "
                    "first at most 100 real item IDs. Ordering changes may include membership "
                    "changes. Missing samples return null errors/pass flags. These bounded "
                    "training probes do not measure validation HR, prove universal invariance, "
                    "or establish semantic padding leakage from numerical discrepancies."
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
