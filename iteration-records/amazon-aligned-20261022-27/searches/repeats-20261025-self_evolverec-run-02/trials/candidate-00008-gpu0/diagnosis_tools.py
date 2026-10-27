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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the diagnostic view contract without unavailable categories
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.margin_valid_positions = 0
        self.sampled_logit_std = None

    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Bounded base-embedding cosine with explicit unavailable values
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        with torch.no_grad():
            emb = self.view.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2:
                return None
            generator = torch.Generator(device=emb.device).manual_seed(seed)
            k = min(sample_size, n)
            indices = torch.randperm(n, generator=generator, device=emb.device)[:k]
            x = F.normalize(emb.index_select(0, indices).float(), p=2, dim=-1)
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite diagnostic embeddings")
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1).item()
            return round(float((mean_cos + 1.0) * 0.5), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Reset bounded margin measurements for each diagnosis
        if num_batches < 1:
            raise ValueError("num_batches must be positive")
        self.margin_valid_positions = 0
        self.sampled_logit_std = None
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Score aligned diagnostic targets through the SASRec view
        from itertools import islice

        for batch in islice(dataloader, num_batches):
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError("Expected (None, sequence, positive, negative)")
            _, seq, pos, neg = batch
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic tensors must have matching B x L shapes")
            features = self.view.log2feats(seq_t)
            positive_scores = self.view.predict_score(features, pos_t)
            negative_scores = self.view.predict_score(features, neg_t)
            if (
                positive_scores.shape != pos_t.shape
                or negative_scores.shape != pos_t.shape
            ):
                raise ValueError("Aligned diagnostic scores must have shape B x L")
            margin = positive_scores.float() - negative_scores.float()
            valid = pos_t.ne(0) & neg_t.ne(0) & seq_t.ne(0)
            if valid.any():
                logits = torch.stack(
                    (positive_scores[valid], negative_scores[valid]), dim=-1
                ).float()
                if (
                    not torch.isfinite(logits).all()
                    or not torch.isfinite(margin[valid]).all()
                ):
                    raise FloatingPointError("Nonfinite diagnostic scores")
                if self.sampled_logit_std is None:
                    self.sampled_logit_std = float(logits.std(unbiased=False).item())
                self.margin_valid_positions += int(valid.sum().item())
                margins_all.append(margin[valid].detach().float().cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Exclude missing diagnostic negatives and padding from counts
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
            ### >>> Self_EvolveRec-BLOCK-START: Report absent margin measurements as unavailable
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

    ### >>> Self_EvolveRec-BLOCK-START: Isolate diagnostic random streams and restore every module mode
    def run_full_diagnosis(self, train_loader):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        modes = [(module, module.training) for module in self.model.modules()]
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    torch.cuda.current_device()
                    if self.device.index is None
                    else self.device.index
                )
            ]
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(1042)
                np.random.seed(1042)
                torch.random.default_generator.manual_seed(1042)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(1042)
                self.model.eval()
                with torch.no_grad():
                    return self._run_full_diagnosis(train_loader)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running bounded base-embedding and sampled-margin diagnosis...")
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                ### >>> Self_EvolveRec-BLOCK-START: Expose margin availability and sampled score scale
                "pos_neg_margin_category_breakdown": breakdown,
                "margin_valid_positions": self.margin_valid_positions,
                "sampled_logit_std": self.sampled_logit_std,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define base-table alignment without unsupported diversity claims
                "embedding_collapse_score": (
                    "Mean pairwise cosine of sampled base item embeddings, mapped to [0,1]. "
                    "0.5 means mean cosine zero. Higher means greater directional alignment; "
                    "this does not establish collapse, recommendation diversity, or compatibility. "
                    "Null means fewer than two catalog items."
                ),
                "margin_valid_positions": "Number of valid sampled training positions; zero means margin summaries are unavailable.",
                "sampled_logit_std": "Population standard deviation of positive and negative logits on the first usable diagnostic batch; score scale, not validation accuracy. Null means unavailable.",
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: State training-negative and metadata limitations
                "pos_neg_margin_category_breakdown": (
                    "Counts valid sampled training positions with margin<eps or margin<0. "
                    "Category lists are empty because item categories are unavailable. "
                    "Sampled training margins do not estimate validation HR@10 or full-catalog accuracy."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
