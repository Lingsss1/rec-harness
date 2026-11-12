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
    ### >>> Self_EvolveRec-BLOCK-START: Diagnose the actual module using observed training frequencies
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.frequency = Counter(
            int(item)
            for history in self.user_train.get("History", {}).values()
            for item in history
            if int(item) > 0
        )
        self.dataset_name = getattr(view, "dataset_name", None)
        self.embedding_sample_count = 0

    def _get_categories(self, item_id: int):
        # Preserve the helper signature; labels now denote training frequency.
        count = self.frequency[int(item_id)]
        if count == 0:
            return ["unseen_in_training"]
        if count <= 5:
            return ["1_to_5"]
        if count <= 20:
            return ["6_to_20"]
        return ["21_or_more"]

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Measure finite embedding alignment without constructing a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            k = min(sample_size, n)
            self.embedding_sample_count = k
            self.embedding_invalid_count = 0
            if k < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).double()
            norms = x.norm(dim=-1)
            valid = torch.isfinite(x).all(dim=-1) & torch.isfinite(norms) & norms.gt(0)
            self.embedding_invalid_count = int((~valid).sum().item())
            # Do not report apparently healthy alignment for invalid embeddings.
            if self.embedding_invalid_count:
                return None
            x = x / norms.unsqueeze(-1)
            # sum_{i != j} x_i dot x_j = ||sum_i x_i||^2 - sum_i ||x_i||^2.
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (offdiag_sum / (k * (k - 1))).clamp(-1, 1)
            return round(float(((mean_cos + 1.0) * 0.5).item()), 4)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Restore module modes after measured training-margin probes
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        modes = [(module, module.training) for module in self.network.modules()]
        self.network.eval()
        try:
            return self._analyze_training_margins(
                dataloader, num_batches, eps, topk_categories
            )
        finally:
            for module, training in modes:
                module.training = training

    @torch.no_grad()
    def _analyze_training_margins(self, dataloader, num_batches, eps, topk_categories):
        batches_seen = 0
        batches_scored = 0
        nonfinite_positions = 0
        frequency_totals = Counter()
        residual_base_sum = 0.0
        residual_branch_sum = 0.0
        residual_positions = 0
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Consume exactly the declared number of diagnostic batches
        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        if not np.isfinite(eps):
            raise ValueError("eps must be finite")
        if topk_categories < 1:
            raise ValueError("topk_categories must be positive")
        # range is exhausted first, so the loader never produces an extra batch.
        for step, batch in zip(range(num_batches), dataloader):
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Use device tensors in the shared augmented feature path
            batches_seen += 1
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue
            _, seq, pos, neg = batch[:4]
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            if seq_t.ndim != 2:
                raise ValueError("Diagnostic sequences must have shape B x L")
            log_feats = self.model.log2feats(seq_t)
            batches_scored += 1
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            ### >>> Self_EvolveRec-BLOCK-START: Validate adapter shapes and avoid overflow in diagnostic score differences
            if pos_t.shape != seq_t.shape or neg_t.shape != seq_t.shape:
                raise ValueError(
                    "Diagnostic sequence, positive and negative tensors must share shape B x L"
                )
            if log_feats.ndim != 3 or log_feats.shape[:2] != seq_t.shape:
                raise ValueError("log2feats must return features with shape B x L x D")
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError(
                    "predict_score must return logits with shape B x L x 2"
                )
            all_logits = all_logits.double()
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Preserve finite diagnostic precision and validate backbone scoring
            target_valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            finite = torch.isfinite(all_logits).all(dim=-1) & torch.isfinite(margin)
            nonfinite_positions += int((target_valid & ~finite).sum().item())
            valid = target_valid & finite
            if valid.any():
                margins_all.append(margin[valid].detach().cpu())
                for item in pos_t[valid].detach().cpu().tolist():
                    frequency_totals.update(self._get_categories(item))

            if callable(getattr(self.network, "_backbone_log2feats", None)):
                base = self.network._backbone_log2feats(seq_t)
                base_logits = self.model.predict_score(base, all_items)
                if base_logits.shape != all_logits.shape:
                    raise ValueError(
                        "Backbone and augmented logits must share shape B x L x 2"
                    )
                base_logits = base_logits.double()
                branch_logits = all_logits - base_logits
                measurable = (
                    valid
                    & torch.isfinite(base_logits).all(dim=-1)
                    & torch.isfinite(branch_logits).all(dim=-1)
                )
                if measurable.any():
                    residual_base_sum += float(
                        base_logits[measurable].abs().sum().item()
                    )
                    residual_branch_sum += float(
                        branch_logits[measurable].abs().sum().item()
                    )
                    residual_positions += int(measurable.sum().item())

            margin_cpu = margin.detach().cpu().numpy()
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
            ### >>> Self_EvolveRec-BLOCK-START: Report missing training margins explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Expose measured denominators and frequency summaries instead of unavailable categories
        def _topk(counter: Counter, k: int):
            return [
                {"training_frequency_bucket": c, "count": int(n)}
                for c, n in counter.most_common(k)
            ]

        observation_count = sum(m.numel() for m in margins_all)
        breakdown = {
            "dataset_name": self.dataset_name,
            "eps": float(eps),
            "batches_seen": batches_seen,
            "batches_scored": batches_scored,
            "finite_valid_positions": observation_count,
            "nonfinite_valid_positions": nonfinite_positions,
            "positive_frequency_denominators": dict(frequency_totals),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_training_frequency": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_training_frequency": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
            "residual_logit_positions": residual_positions,
            "base_mean_absolute_logit": (
                residual_base_sum / (2 * residual_positions)
                if residual_positions
                else None
            ),
            "residual_mean_absolute_logit": (
                residual_branch_sum / (2 * residual_positions)
                if residual_positions
                else None
            ),
            "residual_to_base_absolute_logit_ratio": (
                residual_branch_sum / residual_base_sum
                if residual_positions and residual_base_sum > 0
                else None
            ),
        }
        ### <<< Self_EvolveRec-BLOCK-END

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Measure item norm association using only training interaction counts
    @torch.no_grad()
    def _norm_frequency_summary(self):
        weight = self.model.item_emb.weight.detach()[1:]
        counts = torch.tensor(
            [self.frequency[i] for i in range(1, weight.size(0) + 1)],
            device=weight.device,
            dtype=torch.float64,
        )
        norms = weight.double().norm(dim=-1)
        valid = torch.isfinite(norms)
        x, y = counts[valid].log1p(), norms[valid]
        correlation = None
        if x.numel() >= 2:
            x, y = x - x.mean(), y - y.mean()
            denominator = x.norm() * y.norm()
            if denominator.item() > 0:
                correlation = float(((x * y).sum() / denominator).item())
        gate = None
        gate_definition = "Unavailable: no recognized scalar residual gate."
        a = getattr(self.network, "a", None)
        if isinstance(a, torch.Tensor) and a.numel() == 1:
            # Recognize the supplied bounded-tanh architecture explicitly.
            if hasattr(self.network, "recent_bound"):
                gate = float((self.network.recent_bound * a.detach().tanh()).item())
                gate_definition = "Actual residual coefficient: recent_bound * tanh(a)."
        return {
            "item_norm_log_training_frequency_pearson": correlation,
            "item_norm_frequency_items": int(valid.sum().item()),
            "residual_coefficient": gate,
        }, gate_definition

    def run_full_diagnosis(self, train_loader):
        print(">>> Running observed training-data diagnosis...")
        norm_metrics, gate_definition = self._norm_frequency_summary()
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
                ### >>> Self_EvolveRec-BLOCK-START: Return observation counts including invalid sampled embeddings
                "pos_neg_margin_training_frequency_breakdown": breakdown,
                "embedding_sample_count": self.embedding_sample_count,
                "embedding_invalid_count": self.embedding_invalid_count,
                **norm_metrics,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define alignment availability and invalid-embedding counts
                "embedding_collapse_score": (
                    "Mean off-diagonal cosine similarity among sampled nonpadding item embeddings, "
                    "mapped from [-1,1] to [0,1]. About 0.5 indicates zero mean cosine; "
                    "higher values indicate directional alignment, not demonstrated recommendation "
                    "diversity loss or observed user preferences. Null with fewer than two sampled "
                    "items or any sampled nonfinite or zero-norm embedding."
                ),
                "embedding_invalid_count": (
                    "Number of sampled nonpadding embeddings with nonfinite coordinates, "
                    "nonfinite L2 norm, or zero L2 norm. Zero when fewer than two items were "
                    "sampled and alignment was not attempted."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define available probes and their limits
                "pos_neg_margin_training_frequency_breakdown": (
                    "At most ten training batches with dropout disabled. Only nonpadding sequence, "
                    "positive and negative positions with finite logits enter margins. Reports "
                    "batch counts, excluded nonfinite positions, and valid-position denominators. "
                    "Bucket counts use only training interactions: 0, 1..5, 6..20, >=21; repeated "
                    "occurrences count separately. Positive bucket denominators cover all finite "
                    "valid positions; low-margin and wrong-rank counts are subsets. Dataset name "
                    "is the adapter's route when exposed, otherwise null. Residual logit means "
                    "average absolute positive and negative logits over residual_logit_positions; "
                    "the residual is augmented logits minus backbone logits, with no second addition. "
                    "Ratio is the ratio of these absolute means, null without observations or with "
                    "zero base magnitude. These are training probes, not validation HR estimates. "
                    "Metadata categories, review content and simulator traits are unavailable."
                ),
                "embedding_sample_count": "Number of nonpadding catalog embeddings sampled with local seed 42.",
                "item_norm_log_training_frequency_pearson": (
                    "Pearson correlation between item embedding L2 norm and log1p(training "
                    "interaction count) across finite nonpadding catalog embeddings. Null for "
                    "insufficient items or zero variance; association does not establish causation."
                ),
                "item_norm_frequency_items": "Number of finite item norms entering the frequency association.",
                "residual_coefficient": gate_definition,
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
