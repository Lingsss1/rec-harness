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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the diagnostic view and actual trained module
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("view.model must be the actual trained torch.nn.Module")
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Do not infer unavailable item categories
    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute descriptive mean cosine without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2 or sample_size < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    "Sampled item embeddings contain nonfinite values"
                )
            x = F.normalize(x, p=2, dim=-1).double()

            # Sum of all pairwise dots minus the actual diagonal. This also
            # preserves the original zero-vector cosine convention.
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
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
        ### >>> Self_EvolveRec-BLOCK-START: Validate probe settings and restore module modes on failure
        if not isinstance(num_batches, int) or num_batches < 0:
            raise ValueError("num_batches must be a nonnegative integer")
        if not np.isfinite(eps):
            raise ValueError("eps must be finite")
        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            return self._analyze_margins(dataloader, num_batches, eps, topk_categories)
        finally:
            for module, training in prior_modes:
                module.training = training

    @torch.no_grad()
    def _analyze_margins(self, dataloader, num_batches, eps, topk_categories):
        margins_all = []
        observed_positions = 0
        nonfinite_positions = 0
        ### <<< Self_EvolveRec-BLOCK-END
        ### >>> Self_EvolveRec-BLOCK-START: Consume only requested batches and reject malformed diagnostic input
        from itertools import islice

        low_total, wrong_total = 0, 0
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Measure valid causal positions through the compatibility view
            seq_np = (
                seq.detach().cpu().numpy()
                if isinstance(seq, torch.Tensor)
                else np.asarray(seq)
            )
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            seq_t = torch.as_tensor(seq_np, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequences, positives and negatives must have shape B x L"
                )
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0) & (pos_t != neg_t)
            if not valid.any():
                continue

            log_feats = self.model.log2feats(seq_np)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError(
                    "predict_score must return one score per supplied item"
                )
            margin = all_logits[..., 0] - all_logits[..., 1]
            finite = torch.isfinite(all_logits).all(dim=-1) & torch.isfinite(margin)
            observed_positions += int(valid.sum().item())
            nonfinite_positions += int((valid & ~finite).sum().item())
            valid = valid & finite
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize aggregate counts without unavailable category lookups
            low_total += int(np.count_nonzero(valid_cpu & (margin_cpu < float(eps))))
            wrong_total += int(np.count_nonzero(valid_cpu & (margin_cpu < 0.0)))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Empty measurements are unavailable rather than successful zeros
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

        ### >>> Self_EvolveRec-BLOCK-START: Retain only the explicitly unavailable category report below
        # Category counters are intentionally absent: categories are unobserved.
        ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Report coverage and explicitly unavailable category evidence
        margin_metrics["observed_positions"] = observed_positions
        margin_metrics["finite_positions"] = observed_positions - nonfinite_positions
        margin_metrics["nonfinite_positions"] = nonfinite_positions
        breakdown = {
            "available": False,
            "reason": "Item categories are not provided in this benchmark.",
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
        }
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Validate standalone probes and check actual-module causal scores
    @torch.no_grad()
    def analyze_causal_score_contract(self, batches, atol=1e-5, rtol=1e-4):
        if not all(np.isfinite(value) and value >= 0 for value in (atol, rtol)):
            raise ValueError("Diagnostic tolerances must be finite and nonnegative")
        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            return self._analyze_causal_score_contract(batches, atol, rtol)
        finally:
            for module, training in prior_modes:
                module.training = training

    @torch.no_grad()
    def _analyze_causal_score_contract(self, batches, atol, rtol):
        feature_fn = getattr(self.module, "log2feats", None)
        score_fn = getattr(self.module, "predict_score", None)
        available = callable(feature_fn) and callable(score_fn)
        counts = {
            "causal_features": [0, 0, 0, None],
            "causal_scores": [0, 0, 0, None],
            "view_score_parity": [0, 0, 0, None],
        }

        def features(sequence, actual=False):
            value = (
                feature_fn(sequence)
                if actual
                else self.model.log2feats(sequence.detach().cpu().numpy())
            )
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or value.shape[:2] != sequence.shape
                or not value.is_floating_point()
            ):
                raise ValueError("log2feats must return a floating B x L x D tensor")
            return value

        def scores(value, items, actual=False):
            result = (score_fn if actual else self.model.predict_score)(value, items)
            if (
                not isinstance(result, torch.Tensor)
                or result.shape != items.shape
                or not result.is_floating_point()
            ):
                raise ValueError(
                    "predict_score must return one floating tensor score per supplied ID"
                )
            return result

        def compare(name, before, after, valid):
            if before.shape != after.shape:
                raise ValueError("Diagnostic comparison shapes differ")
            before = before.detach().to(device=self.device, dtype=torch.float64)
            after = after.detach().to(device=self.device, dtype=torch.float64)
            finite = torch.isfinite(before).all(-1) & torch.isfinite(after).all(-1)
            entry = counts[name]
            entry[0] += int(valid.sum().item())
            entry[1] += int((valid & ~finite).sum().item())
            usable = valid & finite
            if usable.any():
                error = (before - after).abs()
                failed = (error > atol + rtol * before.abs()).any(-1)
                entry[2] += int((usable & failed).sum().item())
                maximum = float(error[usable].max().item())
                entry[3] = maximum if entry[3] is None else max(entry[3], maximum)

        for _, sequence, positive, negative in batches[:2]:
            seq, pos, neg = [
                torch.as_tensor(x[:8], device=self.device, dtype=torch.long)
                for x in (sequence, positive, negative)
            ]
            if seq.ndim != 2 or pos.shape != seq.shape or neg.shape != seq.shape:
                raise ValueError("Diagnostic tensors must have matching B x L shapes")
            valid_features = seq != 0
            if not valid_features.any():
                continue
            valid_scores = valid_features & (pos != 0) & (neg != 0) & (pos != neg)
            items = torch.stack((pos, neg), dim=-1)
            view_features = features(seq)
            view_scores = scores(view_features, items)
            if available:
                reference_features = features(seq, actual=True)
                reference_scores = scores(reference_features, items, actual=True)
                compare(
                    "view_score_parity", reference_scores, view_scores, valid_scores
                )
            else:
                reference_features = view_features
                reference_scores = view_scores

            cuts = set()
            for row in seq:
                positions = torch.nonzero(row != 0, as_tuple=False).flatten().tolist()
                if len(positions) >= 2:
                    cuts.update(
                        positions[k]
                        for k in {1, len(positions) // 2, len(positions) - 1}
                    )
            for cut in sorted(cuts):
                future = (seq[:, cut:] != 0).any(-1)
                prefix_valid = valid_features[:, :cut] & future[:, None]
                if not prefix_valid.any():
                    continue
                masked = seq.clone()
                masked[:, cut:] = 0
                altered_features = features(masked, actual=available)
                altered_scores = scores(altered_features, items, actual=available)
                compare(
                    "causal_features",
                    reference_features[:, :cut],
                    altered_features[:, :cut],
                    valid_features[:, :cut] & future[:, None],
                )
                compare(
                    "causal_scores",
                    reference_scores[:, :cut],
                    altered_scores[:, :cut],
                    valid_scores[:, :cut] & future[:, None],
                )

        metrics = {"view_score_parity_available": available}
        definitions = {
            "view_score_parity_available": (
                "Whether the actual trained module exposes both log2feats and predict_score. "
                "False indicates unavailable parity, not successful parity."
            )
        }
        causal_source = "Actual-module" if available else "Compatibility-view fallback"
        descriptions = {
            "causal_features": f"{causal_source} feature changes after masking future events",
            "causal_scores": f"{causal_source} supplied-candidate score changes after masking future events",
            "view_score_parity": "Compatibility-view score differences from actual-module scores",
        }
        for name, (checked, nonfinite, violations, maximum) in counts.items():
            finite = checked - nonfinite
            values = {
                "checked_positions": checked,
                "nonfinite_positions": nonfinite,
                "violation_rate": violations / finite if finite else None,
                "max_absolute_error": maximum,
            }
            for suffix, value in values.items():
                key = f"{name}_{suffix}"
                metrics[key] = value
                definitions[key] = (
                    f"{descriptions[name]}; at most eight sequences from each of two batches. "
                    "Causal cutpoints precede the second, middle and last observed events; "
                    "padding width, absolute positions and candidate IDs remain fixed. "
                    "Checked positions may repeat across cutpoints. Nonfinite positions are "
                    "excluded from violation rates. A violation exceeds "
                    f"{atol:g} + {rtol:g} times the reference coordinate magnitude. "
                    "Maximum error uses finite positions; "
                    "rates and maxima are null without finite comparisons. These sampled checks "
                    "do not verify exported-predictor parity or validation improvement."
                )
        return metrics, definitions

    def run_full_diagnosis(self, train_loader):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            from itertools import islice

            # Materialize once so one-shot iterators feed every probe identically.
            batches = []
            for batch in islice(train_loader, 10):
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                tensors = tuple(
                    torch.as_tensor(value).detach().cpu().clone() for value in batch[1:]
                )
                if tensors[0].ndim != 2 or any(
                    value.shape != tensors[0].shape for value in tensors
                ):
                    raise ValueError(
                        "Diagnostic tensors must have matching B x L shapes"
                    )
                batches.append((None,) + tensors)
            self.module.eval()
            with torch.no_grad():
                result = self._run_full_diagnosis(batches)
                metrics, definitions = self.analyze_causal_score_contract(batches)
                result["metrics"].update(metrics)
                result["metric_definitions"].update(definitions)
                return result
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            for module, training in prior_modes:
                module.training = training

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running embedding, margin and causal-score diagnosis...")

        # Review summary: statically checked scorer dispatch, candidate shapes,
        # finite-value accounting, tolerance definitions, one-shot loader reuse,
        # standalone evaluation mode and exception-safe module-mode restoration.
        # Full diagnosis restores global Python, NumPy and Torch RNG states;
        # dedicated loader generators and worker state require host isolation.
        # These probes do not implement the freshness residual or verify exported
        # predictor parity. No execution or validation improvement is claimed.
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
                ### >>> Self_EvolveRec-BLOCK-START: Expose diagnostic coverage without inventing observations
                "pos_neg_margin_category_breakdown": breakdown,
                "pos_neg_observed_positions": margin_metrics["observed_positions"],
                "pos_neg_finite_positions": margin_metrics["finite_positions"],
                "pos_neg_nonfinite_positions": margin_metrics["nonfinite_positions"],
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define descriptive probes and their observation limits
                "embedding_collapse_score": (
                    "Mean off-diagonal cosine of sampled nonpadding item embeddings, mapped to [0,1]. "
                    "0.5 corresponds to mean cosine zero. This descriptive statistic does not establish "
                    "representation collapse, recommendation quality or recommendation diversity. "
                    "Null means fewer than two sampled embeddings."
                ),
                "pos_neg_margin_mean": (
                    "Mean positive-minus-negative logit over finite, nonpadding training positions "
                    "with distinct positive and negative IDs. Scale-dependent and sampler-dependent; "
                    "not validation HR@10. Null means no finite observations."
                ),
                "pos_neg_margin_p10": (
                    "Nearest-rank 10th percentile of the same training logit margins; "
                    "scale-dependent. Null means no finite observations."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of finite valid training pairs with negative logit strictly above "
                    "positive logit; ties are excluded from this numerator. Null means no observations."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of finite valid training pairs with margin below 0.5 logit units. "
                    "This threshold is scale-dependent. Null means no observations."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Category breakdown unavailable because item categories are not provided. "
                    "Includes aggregate finite-pair counts below the margin thresholds only."
                ),
                "pos_neg_observed_positions": (
                    "Number of nonpadding training positions with distinct positive and negative IDs "
                    "examined in at most ten loader batches, including nonfinite scores."
                ),
                "pos_neg_finite_positions": "Observed positions with two finite logits and a finite margin.",
                "pos_neg_nonfinite_positions": (
                    "Observed positions excluded from margin summaries due to a nonfinite logit or margin; "
                    "positive counts indicate numerical failures requiring investigation."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
