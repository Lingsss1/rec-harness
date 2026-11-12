import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bounded family-specific diagnostics
from collections import Counter
from itertools import islice
import time

### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measured training support without unavailable evidence
    - Null means unavailable, never zero; missing measurements are untested.
    - Report support and finite exclusions with findings.
    - Training-pair margins are not pinned validation HR@10 or catalog accuracy.
    - Margins depend on score scale; mapped cosine near 0.5 means mean cosine near zero and does not demonstrate collapse.
    - Item categories, reviews, ratings and simulator traits are unavailable.
    - Repeated loader rows need not be distinct users; do not invent uncertainty estimates.
    - Missing predictor reload parity remains unverified.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView contract and unavailable category annotations
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
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

            ### >>> Self_EvolveRec-BLOCK-START: Explicit undefined or nonfinite embedding support
            n = emb.size(0)
            if min(sample_size, n) < 2 or not torch.isfinite(emb).all():
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Linear-memory mean off-diagonal embedding cosine
            x = F.normalize(emb.index_select(0, idx).float(), p=2, dim=-1)
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate actual NextItNet module with explicit finite support
        self.network.eval()
        margins_all = []
        valid_total = nonfinite_total = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Avoid consuming an extra diagnostic batch
        for batch in islice(iter(dataloader), max(0, num_batches)):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Tensor inputs for family-specific log2feats
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and report nonfinite diagnostic pairs
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            finite = torch.isfinite(all_logits).all(-1) & torch.isfinite(margin)
            valid_total += int(valid.sum().item())
            nonfinite_total += int((valid & ~finite).sum().item())
            valid = valid & finite
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
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
            ### >>> Self_EvolveRec-BLOCK-START: Empty diagnostic measurements remain unavailable
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

        ### >>> Self_EvolveRec-BLOCK-START: Report support and category missingness
        margin_metrics.update(
            {
                "valid_positions": valid_total,
                "comparisons": valid_total - nonfinite_total,
                "excluded_nonfinite": nonfinite_total,
                "distinct_users": None,
            }
        )
        breakdown = {
            "availability": "unavailable: item categories are not provided",
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

    ### >>> Self_EvolveRec-BLOCK-START: Bounded measured diagnosis with RNG and module-mode restoration
    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        started = time.perf_counter()
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            self.network.eval()
            with torch.random.fork_rng(devices=devices):
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(42)
                batches = []
                for batch in islice(iter(train_loader), 4):
                    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                        raise ValueError(
                            "expected (None, sequence, positive, negative)"
                        )
                    tensors = [
                        torch.as_tensor(x, device=self.device, dtype=torch.long)[
                            :32
                        ].clone()
                        for x in batch[1:]
                    ]
                    if any(x.ndim != 2 or x.shape != tensors[0].shape for x in tensors):
                        raise ValueError(
                            "diagnostic tensors must have matching B x L shapes"
                        )
                    if tensors[0].size(0) and tensors[0].size(1):
                        batches.append((None, *tensors))
                collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=4, eps=0.5, topk_categories=20
                    )
                )
                embedding = self.model.item_emb.weight.detach()[
                    1 : self.network.itemnum + 1
                ]
                if devices:
                    torch.cuda.synchronize(devices[0])
                metrics = {
                    "embedding_collapse_score": collapse,
                    "embedding_support": {
                        "catalog_items": embedding.size(0),
                        "sampled_items": (
                            min(2048, embedding.size(0)) if collapse is not None else 0
                        ),
                        "nonfinite_items": int(
                            (~torch.isfinite(embedding).all(-1)).sum().item()
                        ),
                    },
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "all_position_support": {
                        key: margins[key]
                        for key in (
                            "valid_positions",
                            "comparisons",
                            "excluded_nonfinite",
                            "distinct_users",
                        )
                    },
                    "pos_neg_margin_category_breakdown": breakdown,
                    "deployment_reload_parity": {
                        "available": False,
                        "reason": "ModelView provides no generated-predictor reload interface",
                    },
                    "validation_rank_metrics": {
                        "available": False,
                        "reason": "diagnostic loader does not supply pinned validation candidates",
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                }
                definitions = {
                    "embedding_collapse_score": "(mean sampled off-diagonal item cosine + 1)/2; 0.5 means mean cosine zero. Not a collapse probability, accuracy or recommendation-diversity measure. Null if undefined/nonfinite.",
                    "embedding_support": "Catalog and sampled item counts excluding padding; nonfinite catalog embedding count.",
                    "pos_neg_margin_mean": "Scale-dependent mean positive-minus-supplied-negative score over finite positions with nonzero input and both targets nonzero.",
                    "pos_neg_margin_p10": "Nearest-rank tenth percentile of the same supported training margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of supported training margins strictly below zero; ties excluded from negative wins. Not validation HR@10.",
                    "pos_neg_small_margin_rate": "Fraction of supported training margins below the scale-dependent threshold 0.5.",
                    "all_position_support": "Valid positions, finite comparisons and exclusions from at most four batches and 32 rows per batch. Distinct user identities unavailable.",
                    "pos_neg_margin_category_breakdown": "Measured low-margin and negative-win position totals. Item categories unavailable; category lists empty.",
                    "deployment_reload_parity": "Unavailable without an actual generated-predictor reload check; no deployment correctness inferred.",
                    "validation_rank_metrics": "Unavailable; supplied training pairs do not estimate pinned validation HR@10.",
                    "elapsed_seconds": "Wall time for bounded loader collection, embedding and margin diagnostics, including CUDA synchronization when applicable.",
                }
                return {"metrics": metrics, "metric_definitions": definitions}
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    ### <<< Self_EvolveRec-BLOCK-END
