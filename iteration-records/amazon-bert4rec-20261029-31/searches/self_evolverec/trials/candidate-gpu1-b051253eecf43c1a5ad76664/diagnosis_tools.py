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
    def __init__(self, model, device):
        ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and resolve the actual BERT4Rec sampler
        import inspect

        self.model = model
        self.device = device
        self.meta_data = model.meta_data
        self.review_data = model.review_data
        self.user_train = model.user_train
        self.module = model.model
        defining_module = inspect.getmodule(type(self.module))
        self.sampler = getattr(defining_module, "sample_batch", None)
        if self.sampler is None:
            encoder = getattr(type(self.module), "encode_sequence", None)
            self.sampler = getattr(encoder, "__globals__", {}).get("sample_batch")
        if self.sampler is not None and not callable(self.sampler):
            raise TypeError("The defining sample_batch must be callable")
        ### <<< Self_EvolveRec-BLOCK-END

    def _get_categories(self, item_id: int):
        ### >>> Self_EvolveRec-BLOCK-START: Mark unavailable item categories without inventing metadata
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
            ### >>> Self_EvolveRec-BLOCK-START: Empty cosine probes are unavailable
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
        ### >>> Self_EvolveRec-BLOCK-START: Separate target coverage from valid-negative margin coverage
        was_training = self.module.training
        self.module.eval()
        observed = {
            "supervised_positions": 0,
            "mask_input": 0,
            "different_real_input": 0,
            "same_real_input": 0,
            "empty_batches": 0,
            "presented_sequences": 0,
            "valid_target_positions": 0,
            "terminal_position_targets": 0,
        }
        margins_all = []
        matched_branches = {"mask": 0, "random": 0, "unchanged": 0}
        bookkeeping_batches = 0
        unmatched_batches = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume exactly the requested diagnostic batches and reject malformed input
        from itertools import islice

        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                raise ValueError(
                    "Diagnosis requires (None, sequence, positive, negative) batches"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic tensors and consume only exactly matched bookkeeping
            seq_np = (
                seq.detach().cpu().numpy()
                if isinstance(seq, torch.Tensor)
                else np.asarray(seq)
            )
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_np.ndim != 2
                or tuple(seq_np.shape) != tuple(pos_t.shape)
                or pos_t.shape != neg_t.shape
            ):
                raise ValueError(
                    "Diagnosis requires matching B x L sequence/positive/negative tensors"
                )
            if np.any((seq_np < 0) | (seq_np > self.module.mask_token)):
                raise ValueError("Diagnostic sequence contains an invalid token")
            if torch.any((pos_t < 0) | (pos_t > self.module.itemnum)) or torch.any(
                (neg_t < 0) | (neg_t > self.module.itemnum)
            ):
                raise ValueError("Diagnostic targets must be padding or real item IDs")

            book = getattr(self.sampler, "last_bookkeeping", None)
            matched = isinstance(book, dict) and all(
                key in book and np.array_equal(book[key], value)
                for key, value in (
                    ("sequence", seq_np),
                    ("positive", pos_t.detach().cpu().numpy()),
                    ("negative", neg_t.detach().cpu().numpy()),
                )
            )
            if matched:
                branch = np.asarray(book["corruption_branch"])
                if branch.shape != seq_np.shape:
                    raise ValueError("Sampler bookkeeping shape mismatch")
                supervised_np = pos_t.detach().cpu().numpy() != 0
                if np.any(branch[supervised_np] < 1) or np.any(
                    branch[supervised_np] > 3
                ):
                    raise ValueError(
                        "Missing corruption branch at a supervised position"
                    )
                for name, code in (("mask", 1), ("random", 2), ("unchanged", 3)):
                    matched_branches[name] += int(
                        ((branch == code) & supervised_np).sum()
                    )
                bookkeeping_batches += 1
            else:
                unmatched_batches += 1

            log_feats = self.model.log2feats(seq_np)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if tuple(all_logits.shape) != tuple(all_items.shape):
                raise ValueError("ModelView.predict_score must return B x L x 2 logits")
            margin = all_logits[..., 0] - all_logits[..., 1]
            if not torch.isfinite(margin[pos_t != 0]).all():
                raise FloatingPointError("Nonfinite diagnostic margins")
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Measure terminal target coverage separately from positive-negative margins
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            itemnum = self.module.itemnum
            target_valid = (pos_t >= 1) & (pos_t <= itemnum) & (seq_t != 0)
            positions = torch.arange(seq_t.shape[1], device=self.device)
            last = (
                positions.unsqueeze(0)
                .expand_as(seq_t)
                .masked_fill(seq_t == 0, -1)
                .amax(dim=1)
            )
            terminal_targets = target_valid & (
                positions.unsqueeze(0) == last.unsqueeze(1)
            )
            observed["presented_sequences"] += int(seq_t.shape[0])
            observed["valid_target_positions"] += int(target_valid.sum().item())
            observed["terminal_position_targets"] += int(terminal_targets.sum().item())
            valid = target_valid & (neg_t >= 1) & (neg_t <= itemnum)
            observed["supervised_positions"] += int(valid.sum().item())
            observed["mask_input"] += int(
                (valid & (seq_t == self.module.mask_token)).sum().item()
            )
            observed["different_real_input"] += int(
                (valid & (seq_t >= 1) & (seq_t <= itemnum) & (seq_t != pos_t))
                .sum()
                .item()
            )
            observed["same_real_input"] += int((valid & (seq_t == pos_t)).sum().item())
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())
            else:
                observed["empty_batches"] += 1
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same real-item supervision mask in all summaries
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
            ### >>> Self_EvolveRec-BLOCK-START: Empty margin probes are unavailable
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

        ### >>> Self_EvolveRec-BLOCK-START: Report measured target coverage without inferring example branches
        breakdown["category_status"] = "unavailable: item categories are not supplied"
        breakdown["observed_inputs_at_supervised_positions"] = observed
        breakdown["terminal_position_coverage"] = (
            observed["terminal_position_targets"] / observed["presented_sequences"]
            if observed["presented_sequences"]
            else None
        )
        breakdown["corruption_branch_counts"] = (
            matched_branches if bookkeeping_batches else None
        )
        breakdown["bookkeeping_matched_batches"] = bookkeeping_batches
        breakdown["bookkeeping_unmatched_batches"] = unmatched_batches
        breakdown["corruption_branch_status"] = (
            "Branch counts cover only batches whose sequence, positive and negative "
            "arrays exactly match sampler bookkeeping. Unmatched batches are "
            "unavailable; observed same-item inputs cannot identify their branches."
        )
        self.module.train(was_training)
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Separate raw norms from centered representation spectra
    @staticmethod
    def _representation_stats(features):
        if features is None or features.ndim != 2 or features.shape[0] < 2:
            return None
        x = features.detach().float()
        if not torch.isfinite(x).all():
            return None
        centered = x - x.mean(dim=0, keepdim=True)
        sigma = torch.linalg.svdvals(centered)
        total = sigma.sum()
        rankme = None
        if float(total) > 0:
            p = sigma[sigma > 0] / total
            rankme = float(torch.exp(-(p * p.log()).sum()).item())
        return {
            "sample_count": int(x.shape[0]),
            "mean_raw_norm": float(x.norm(dim=-1).mean().item()),
            "centered_variance_per_dimension": float(centered.square().mean().item()),
            "centered_singular_values": sigma.cpu().tolist(),
            "rankme": rankme,
        }

    @torch.no_grad()
    def analyze_representation_geometry(self, sample_size=256, seed=42):
        was_training = self.module.training
        self.module.eval()
        try:
            itemnum = int(self.module.itemnum)
            rng = random.Random(seed)
            item_ids = rng.sample(range(1, itemnum + 1), min(sample_size, itemnum))
            ids = torch.as_tensor(item_ids, device=self.device, dtype=torch.long)
            item_stats = self._representation_stats(self.model.item_emb(ids))
            histories = self.user_train.get("History", self.user_train)
            eligible = sorted(histories, key=str)
            eligible = [u for u in eligible if len(histories[u]) > 0]
            users = rng.sample(eligible, min(sample_size, len(eligible)))
            rows = []
            for user in users:
                history = [int(i) for i in histories[user]]
                row = np.zeros(self.module.maxlen, dtype=np.int64)
                context = (
                    history[-(self.module.maxlen - 1) :]
                    if self.module.maxlen > 1
                    else []
                )
                values = context + [self.module.mask_token]
                row[-len(values) :] = values
                rows.append(row)
            user_stats = None
            if rows:
                representations = []
                for offset in range(0, len(rows), 32):
                    seq_np = np.stack(rows[offset : offset + 32])
                    feats = self.model.log2feats(seq_np)
                    sequence = torch.as_tensor(seq_np, device=feats.device)
                    positions = torch.arange(sequence.shape[1], device=feats.device)
                    last = (
                        positions.unsqueeze(0)
                        .expand_as(sequence)
                        .masked_fill(sequence == 0, -1)
                        .amax(dim=1)
                    )
                    representations.append(
                        feats[torch.arange(feats.shape[0], device=feats.device), last]
                    )
                user_stats = self._representation_stats(torch.cat(representations))
            return {"real_items": item_stats, "appended_mask_users": user_stats}
        finally:
            self.module.train(was_training)

    def run_full_diagnosis(self, train_loader):
        print(">>> Running masked BERT4Rec diagnosis...")
        geometry = self.analyze_representation_geometry()
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        ### >>> Self_EvolveRec-BLOCK-START: Restore module mode even when a diagnostic fails
        was_training = self.module.training
        try:
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=eps, topk_categories=20
                )
            )
        finally:
            self.module.train(was_training)
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Report directional alignment and centered geometry separately
                "embedding_directional_alignment": collapse_score,
                "representation_geometry": geometry,
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Expose measured training branch allocation when available
                "training_branch_allocation": getattr(
                    self.module, "training_branch_allocation", None
                ),
                "pos_neg_margin_mean": margin_metrics["mean"],
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define geometry without equating cosine alignment to collapse
                "embedding_directional_alignment": (
                    "Mean pairwise cosine of fixed-seed sampled real-item embeddings, "
                    "mapped to [0,1]. Measures directional alignment, not conclusive "
                    "collapse or recommendation diversity. Null means unavailable."
                ),
                "representation_geometry": (
                    "Separate fixed-seed real-item and appended-MASK user samples. "
                    "Reports raw mean L2 norm, mean squared centered coordinates, "
                    "and singular values of the centered representation matrix. "
                    "RankMe = exp(-sum(p*log(p))), p=sigma/sum(sigma); zero spectra "
                    "and fewer than two samples yield unavailable rank/statistics. "
                    "User readout uses the maximum nonpadding index."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Interpret branch measurements using their recorded protocol
                "training_branch_allocation": (
                    "Measured snapshot; null when unavailable. Consult its protocol "
                    "for the actual objective coefficients and cutoff policy. The "
                    "025_075 experiment preserves fixed equal branch sequence counts "
                    "and independent user draws, changing only reconstruction/terminal "
                    "target-mean CE coefficients from 0.5/0.5 to 0.25/0.75. "
                    "Reconstruction retains MASK-only corruption and fallback; terminal "
                    "views retain strict-prefix context followed by MASK. Catalog CE "
                    "uses tied real-item weights, excluding PAD/MASK output classes. "
                    "The executable sampler/loss raises on empty branches. Each batch "
                    "records detached summed CE, exact valid-target denominators, "
                    "example counts, final-position targets, per-target coefficients "
                    "and weighted contributions. Contributions are the fixed scalar "
                    "weight times the measured batch target-mean CE, subject to normal "
                    "floating-point rounding. Coefficient masses sum target coefficients; "
                    "they are neither loss-value shares nor gradient shares. Pooled CE "
                    "and mean batch objectives have different aggregation semantics. "
                    "Checkpoint sidecars restore selected-epoch records and their named "
                    "cumulative interval. Diagnostic-loader final-position coverage "
                    "cannot recover training allocation or full-history endpoint rates. "
                    "These measurements are descriptive training probes, not validation "
                    "HR@10, paired-baseline evidence or proof of improvement."
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define target coverage separately from margin and branch measurements
                "pos_neg_margin_category_breakdown": (
                    "Counts low-margin and wrong-rank supervised real-item positions. "
                    "Category breakdowns are unavailable because item categories are "
                    "not supplied. Observed input counts distinguish MASK, different "
                    "real item and same real item, but cannot identify exact stochastic "
                    "corruption branches without matched sampler bookkeeping. Empty "
                    "batch counts measure probe coverage; empty margin metrics are null. "
                    "The legacy supervised_positions and observed-input counters require "
                    "both a valid positive and a valid negative. valid_target_positions "
                    "counts real positive labels at nonpadding inputs independently of "
                    "negative availability. terminal_position_targets counts those labels "
                    "at each sequence's maximum nonpadding index. Terminal-position "
                    "coverage divides that count by presented_sequences; it is null "
                    "when no sequences were measured. These diagnostic-loader counts "
                    "do not establish the training branch allocation: both reconstruction "
                    "and terminal examples can supervise their final position. "
                    "These training probes are not host validation HR@10."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
