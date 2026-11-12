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
    ### >>> Self_EvolveRec-BLOCK-START: Keep the diagnostic view and actual trained module distinct
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.residual_metrics = {}

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this benchmark adapter.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute sampled mean cosine without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if sample_size < 2:
            raise ValueError("Embedding cosine diagnosis requires sample_size >= 2")
        with torch.no_grad():
            emb = self.view.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = F.normalize(emb.index_select(0, idx).float(), p=2, dim=-1)
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite sampled item embeddings")

            # sum_{i != j} x_i dot x_j = ||sum_i x_i||^2 - sum_i ||x_i||^2.
            # Subtract actual squared norms so zero embeddings are handled too.
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
        self.model.eval()

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Validate diagnostic contracts and distinguish recent residuals from legacy scorers
        from itertools import islice

        if num_batches < 1:
            raise ValueError("num_batches must be positive")
        cfg = getattr(self.model, "residual_config", None)
        is_recent = isinstance(cfg, dict) and cfg.get("version") == 2
        if is_recent and (
            not callable(getattr(self.model, "recent_pool", None))
            or not isinstance(getattr(self.model, "alpha", None), torch.Tensor)
            or self.model.alpha.numel() != 1
        ):
            raise TypeError("Version-2 residual requires recent_pool and scalar alpha")
        self.residual_metrics = {
            "sampled_logit_std": None,
            "recent_residual_probe_status": (
                "no_valid_positions"
                if is_recent
                else "baseline" if cfg is None else "unsupported_residual_architecture"
            ),
        }
        measured_first_batch = False
        for step, batch in enumerate(islice(dataloader, num_batches)):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError("Expected (None, sequence, positive, negative) batch")

            _, seq, pos, neg = batch
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequences and targets must have matching B x L shapes"
                )
            log_feats = self.view.log2feats(seq_t)
            pos_logits = self.view.predict_score(log_feats, pos_t)
            neg_logits = self.view.predict_score(log_feats, neg_t)
            if pos_logits.shape != pos_t.shape or neg_logits.shape != neg_t.shape:
                raise ValueError("Aligned diagnostic scores must have shape B x L")
            valid = pos_t.ne(0) & neg_t.ne(0) & seq_t.ne(0)
            # Subtract in float32 to avoid reduced-precision margin overflow.
            margin = pos_logits.float() - neg_logits.float()
            if valid.any() and (
                not torch.isfinite(pos_logits[valid]).all()
                or not torch.isfinite(neg_logits[valid]).all()
                or not torch.isfinite(margin[valid]).all()
            ):
                raise FloatingPointError("Nonfinite diagnostic scores or margins")

            if not measured_first_batch and valid.any():
                measured_first_batch = True
                logits = torch.stack((pos_logits[valid], neg_logits[valid]), dim=-1)
                self.residual_metrics["sampled_logit_std"] = float(
                    logits.float().std(unbiased=False).item()
                )
                if is_recent:
                    pooled = self.model.recent_pool(seq_t)
                    if pooled.shape != log_feats.shape:
                        raise ValueError(
                            "Recent pool must match sequence feature shape"
                        )
                    # Direct encoder evaluation avoids cancellation when alpha grows.
                    base = self.model._encoder_features(seq_t)
                    base_norm = base[valid].float().norm(dim=-1)
                    pool_norm = pooled[valid].float().norm(dim=-1)
                    if (
                        not torch.isfinite(base_norm).all()
                        or not torch.isfinite(pool_norm).all()
                    ):
                        raise FloatingPointError(
                            "Nonfinite recent-residual diagnostic norms"
                        )
                    ratio = pool_norm / base_norm.clamp_min(1e-12)
                    alpha = self.model.alpha.detach().float()
                    if not torch.isfinite(alpha).all():
                        raise FloatingPointError("Nonfinite recent-residual alpha")
                    grad = self.model.alpha.grad
                    self.residual_metrics.update(
                        {
                            "recent_residual_probe_status": "measured",
                            "recent_alpha": float(alpha.item()),
                            "recent_alpha_last_retained_gradient": (
                                None if grad is None else float(grad.detach().item())
                            ),
                            "recent_pool_feature_norm_ratio": float(
                                ratio.mean().item()
                            ),
                            "recent_effective_residual_norm": float(
                                (alpha.abs() * pool_norm).mean().item()
                            ),
                            "recent_effective_residual_feature_ratio": float(
                                (alpha.abs() * ratio).mean().item()
                            ),
                            "recent_probe_positions": int(valid.sum().item()),
                        }
                    )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Apply the same padding mask to all margin summaries
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Count margin failures directly when categories are unavailable
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Represent missing margin measurements as unknown
        if len(margins_all) == 0:
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

        ### >>> Self_EvolveRec-BLOCK-START: Report the sample size behind margin estimates
        self.residual_metrics["margin_valid_positions"] = sum(
            values.numel() for values in margins_all
        )
        ### <<< Self_EvolveRec-BLOCK-END

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

    ### >>> Self_EvolveRec-BLOCK-START: Isolate diagnostic RNG and restore the trained module's mode
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
                return self._run_full_diagnosis(train_loader)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for module, training in modes:
                module.training = training

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running bounded embedding, margin and recent-residual diagnosis...")
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
                ### >>> Self_EvolveRec-BLOCK-START: Report measured residual scale without inventing unavailable attributes
                "pos_neg_margin_category_breakdown": breakdown,
                **self.residual_metrics,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define diagnostic coverage and architecture availability
                "margin_valid_positions": (
                    "Number of valid sampled training positions included in margin summaries; "
                    "zero means margin measurements are unavailable and reported as null."
                ),
                "recent_residual_probe_status": (
                    "measured: version-2 recent residual measured; no_valid_positions: no usable "
                    "positions; baseline: no residual; unsupported_residual_architecture: residual "
                    "scale probe does not apply to this architecture. Margin probes still run."
                ),
                "measurement_availability": (
                    "Null means unavailable, not zero. Embedding cosine requires at least two "
                    "catalog items; sampled-logit spread requires a usable diagnostic batch."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                "embedding_collapse_score": (
                    "Range ~[0,1]. Estimated by mean pairwise cosine similarity of sampled item embeddings "
                    ### >>> Self_EvolveRec-BLOCK-START: Limit cosine interpretation to embedding geometry
                    "(mapped to [0,1]). Higher means stronger mean directional alignment; "
                    "0.5 corresponds to mean cosine zero. This does not measure recommendation "
                    "diversity, semantic compatibility or simulator traits."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define bounded residual diagnostics and unavailable category breakdowns
                "pos_neg_margin_category_breakdown": (
                    "Counts low-margin and wrong-rank valid positions. Category lists remain "
                    "empty because item categories are unavailable. Margins use sampled training "
                    "negatives and do not estimate full-catalog ranking performance."
                ),
                "sampled_logit_std": (
                    "Population standard deviation of positive and negative logits on the first "
                    "usable diagnostic batch; describes score scale, not validation accuracy."
                ),
                "recent_alpha": "Current unconstrained learned scalar; initialized to exactly zero.",
                "recent_alpha_last_retained_gradient": (
                    "Signed alpha gradient retained from the most recent backward pass, or null "
                    "when unavailable; this probe does not perform backward or optimizer steps."
                ),
                "recent_pool_feature_norm_ratio": (
                    "Mean per-position raw recent-pool L2 norm divided by encoder-feature L2 "
                    "norm, on valid positions of the first usable diagnostic batch."
                ),
                "recent_effective_residual_norm": (
                    "Mean L2 norm of alpha times the raw recent pool on the same positions."
                ),
                "recent_effective_residual_feature_ratio": (
                    "Mean per-position effective-residual L2 norm divided by encoder-feature "
                    "L2 norm on the same positions."
                ),
                "recent_probe_positions": "Number of valid positions used for residual diagnostics.",
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
