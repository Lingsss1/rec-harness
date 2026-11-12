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

    ### >>> Self_EvolveRec-BLOCK-START: Validate candidate-score parity and causal dependence
    @torch.no_grad()
    def analyze_score_contract(self, batches, atol=1e-5, rtol=1e-4):
        if not all(np.isfinite(x) and x >= 0 for x in (atol, rtol)):
            raise ValueError("Score tolerances must be finite and nonnegative")
        feature_fn = getattr(self.module, "log2feats", None)
        score_fn = getattr(self.module, "predict_score", None)
        available = callable(feature_fn) and callable(score_fn)
        checked = nonfinite = mismatches = 0
        causal_checked = causal_nonfinite = causal_mismatches = 0
        max_error = causal_max_error = None
        prior_modes = [(module, module.training) for module in self.module.modules()]

        def checked_scores(features, items, scorer):
            scores = scorer(features, items)
            if not isinstance(scores, torch.Tensor) or scores.shape != items.shape:
                raise ValueError(
                    "Scorer must return one tensor score per supplied item"
                )
            return scores.detach().to(device=self.device, dtype=torch.float64)

        try:
            self.module.eval()
            for _, sequence, positive, negative in batches[:2]:
                seq = torch.as_tensor(
                    sequence[:8], device=self.device, dtype=torch.long
                )
                pos = torch.as_tensor(
                    positive[:8], device=self.device, dtype=torch.long
                )
                neg = torch.as_tensor(
                    negative[:8], device=self.device, dtype=torch.long
                )
                valid = (seq != 0) & (pos != 0) & (neg != 0) & (pos != neg)
                if not valid.any():
                    continue
                items = torch.stack((pos, neg), dim=-1)
                view_features = self.model.log2feats(seq.detach().cpu().numpy())
                view_scores = checked_scores(
                    view_features, items, self.model.predict_score
                )
                if available:
                    reference = checked_scores(feature_fn(seq), items, score_fn)
                    finite = torch.isfinite(reference).all(-1) & torch.isfinite(
                        view_scores
                    ).all(-1)
                    checked += int(valid.sum().item())
                    nonfinite += int((valid & ~finite).sum().item())
                    usable = valid & finite
                    if usable.any():
                        error = (view_scores - reference).abs()
                        failed = (error > atol + rtol * reference.abs()).any(-1)
                        mismatches += int((usable & failed).sum().item())
                        largest = float(error[usable].max().item())
                        max_error = (
                            largest if max_error is None else max(max_error, largest)
                        )
                else:
                    reference = view_scores

                # Score the same supplied candidates after masking only future
                # events. This checks candidate-dependent context as well as h.
                cuts = set()
                for row in seq:
                    positions = (
                        torch.nonzero(row != 0, as_tuple=False).flatten().tolist()
                    )
                    if len(positions) >= 2:
                        cuts.update(
                            positions[i]
                            for i in {1, len(positions) // 2, len(positions) - 1}
                        )
                for cut in sorted(cuts):
                    future = (seq[:, cut:] != 0).any(-1)
                    usable_prefix = valid[:, :cut] & future[:, None]
                    if not usable_prefix.any():
                        continue
                    masked = seq.clone()
                    masked[:, cut:] = 0
                    if available:
                        altered = checked_scores(feature_fn(masked), items, score_fn)
                    else:
                        altered_features = self.model.log2feats(
                            masked.detach().cpu().numpy()
                        )
                        altered = checked_scores(
                            altered_features, items, self.model.predict_score
                        )
                    before, after = reference[:, :cut], altered[:, :cut]
                    finite = torch.isfinite(before).all(-1) & torch.isfinite(after).all(
                        -1
                    )
                    causal_checked += int(usable_prefix.sum().item())
                    causal_nonfinite += int((usable_prefix & ~finite).sum().item())
                    usable = usable_prefix & finite
                    if usable.any():
                        error = (before - after).abs()
                        failed = (error > atol + rtol * before.abs()).any(-1)
                        causal_mismatches += int((usable & failed).sum().item())
                        largest = float(error[usable].max().item())
                        causal_max_error = (
                            largest
                            if causal_max_error is None
                            else max(causal_max_error, largest)
                        )
        finally:
            for module, training in prior_modes:
                module.training = training

        finite_count = checked - nonfinite
        causal_finite = causal_checked - causal_nonfinite
        return {
            "view_score_parity_available": available,
            "view_score_parity_checked_positions": checked,
            "view_score_parity_nonfinite_positions": nonfinite,
            "view_score_parity_mismatch_rate": (
                mismatches / finite_count if finite_count else None
            ),
            "view_score_parity_max_absolute_error": max_error,
            "causal_score_checked_positions": causal_checked,
            "causal_score_nonfinite_positions": causal_nonfinite,
            "causal_score_violation_rate": (
                causal_mismatches / causal_finite if causal_finite else None
            ),
            "causal_score_max_absolute_error": causal_max_error,
        }

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Check tensor or explicit integer-context features without losing precision
    def _feature_parts(self, features, sequence):
        if isinstance(features, dict):
            if "base_features" not in features or "context_ids" not in features:
                raise ValueError(
                    "Feature carriers require base_features and context_ids"
                )
            base = features["base_features"]
            context = features["context_ids"]
            if (
                not isinstance(context, torch.Tensor)
                or context.shape != sequence.shape
                or context.dtype != torch.long
            ):
                raise ValueError("context_ids must be an exact int64 B x L tensor")
            if not torch.equal(context.to(sequence.device), sequence):
                raise ValueError("context_ids must preserve the supplied sequence")
        else:
            base, context = features, None
        if (
            not isinstance(base, torch.Tensor)
            or base.ndim != 3
            or base.shape[:2] != sequence.shape
            or not base.is_floating_point()
        ):
            raise ValueError("base_features must be a floating B x L x D tensor")
        return base, context

    @torch.no_grad()
    def analyze_causal_consistency(self, batches, atol=1e-5, rtol=1e-4):
        if not all(np.isfinite(x) and x >= 0 for x in (atol, rtol)):
            raise ValueError("Causal tolerances must be finite and nonnegative")
        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            return self._analyze_causal_consistency(batches, atol, rtol)
        finally:
            for module, training in prior_modes:
                module.training = training

    @torch.no_grad()
    def _analyze_causal_consistency(self, batches, atol, rtol):
        checked = violations = nonfinite = 0
        max_error = None
        for batch in batches[:2]:
            _, sequence, _, _ = batch
            seq = torch.as_tensor(sequence, device=self.device, dtype=torch.long)
            if seq.ndim != 2:
                raise ValueError("Diagnostic sequences must have shape B x L")
            seq = seq[:8]
            length = seq.shape[1]
            if not seq.shape[0] or length < 2:
                continue
            reference = self.model.log2feats(seq.detach().cpu().numpy())
            reference_base, reference_context = self._feature_parts(reference, seq)
            # Choose boundaries from observed events rather than padded width.
            # Each selected boundary leaves an observed event on both sides.
            cuts = set()
            for row in seq:
                positions = torch.nonzero(row != 0, as_tuple=False).flatten().tolist()
                if len(positions) >= 2:
                    for index in {1, len(positions) // 2, len(positions) - 1}:
                        cuts.add(positions[index])
            for cut in sorted(cuts):
                has_observed_future = (seq[:, cut:] != 0).any(dim=1)
                valid = (seq[:, :cut] != 0) & has_observed_future[:, None]
                if not valid.any():
                    continue
                # Retain absolute positions and padding width. Only future
                # events change, so positional encodings remain comparable.
                masked = seq.clone()
                masked[:, cut:] = 0
                altered = self.model.log2feats(masked.detach().cpu().numpy())
                altered_base, altered_context = self._feature_parts(altered, masked)
                if altered_base.shape != reference_base.shape:
                    raise ValueError("Feature shape changed after future masking")
                if (reference_context is None) != (altered_context is None):
                    raise ValueError("Feature carrier changed after future masking")
                before = reference_base[:, :cut].detach().to(dtype=torch.float64)
                after = (
                    altered_base[:, :cut]
                    .detach()
                    .to(device=before.device, dtype=torch.float64)
                )
                finite = torch.isfinite(before).all(-1) & torch.isfinite(after).all(-1)
                checked += int(valid.sum().item())
                nonfinite += int((valid & ~finite).sum().item())
                usable = valid & finite
                if not usable.any():
                    continue
                error = (before - after).abs()
                failed = (error > atol + rtol * before.abs()).any(-1)
                if reference_context is not None:
                    context_changed = reference_context[:, :cut].to(
                        before.device
                    ) != altered_context[:, :cut].to(before.device)
                    failed = failed | context_changed
                violations += int((usable & failed).sum().item())
                largest = float(error[usable].max().item())
                max_error = largest if max_error is None else max(max_error, largest)
        finite_count = checked - nonfinite
        return {
            "causal_checked_positions": checked,
            "causal_nonfinite_positions": nonfinite,
            "causal_prefix_violation_rate": (
                violations / finite_count if finite_count else None
            ),
            "causal_prefix_max_absolute_error": max_error,
        }

    def run_full_diagnosis(self, train_loader):
        # Loader sampling may consume the same generators used by training.
        # Restore them even when a probe raises; propagate the original error.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            return self._collect_and_run_diagnosis(train_loader)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            for module, training in prior_modes:
                module.training = training

    def _collect_and_run_diagnosis(self, train_loader):
        from itertools import islice

        # Consume a bounded shared sample once, including for one-shot loaders.
        # Retain CPU copies rather than ten batches of device allocations.
        batches = []
        for batch in islice(train_loader, 10):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            tensors = tuple(
                torch.as_tensor(x).detach().cpu().clone() for x in batch[1:]
            )
            if tensors[0].ndim != 2 or any(
                x.shape != tensors[0].shape for x in tensors
            ):
                raise ValueError(
                    "Diagnostic sequences, positives and negatives must have shape B x L"
                )
            batches.append((None,) + tensors)

        prior_modes = [(module, module.training) for module in self.module.modules()]
        try:
            self.module.eval()
            with torch.no_grad():
                return self._run_full_diagnosis(batches)
        finally:
            for module, training in prior_modes:
                module.training = training

    def _run_full_diagnosis(self, batches):
        print(
            ">>> Running embedding, margin, coverage and causal consistency diagnosis..."
        )
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            batches, num_batches=10, eps=eps, topk_categories=20
        )
        causal_metrics = self.analyze_causal_consistency(batches)
        score_contract_metrics = self.analyze_score_contract(batches)
        return {
            ### <<< Self_EvolveRec-BLOCK-END
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Expose feature and candidate-score consistency separately from ranking quality
                **causal_metrics,
                **score_contract_metrics,
                ### <<< Self_EvolveRec-BLOCK-END
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
                ### >>> Self_EvolveRec-BLOCK-START: Define candidate-score parity and causal checks
                "view_score_parity_available": (
                    "Whether the actual trained module exposes log2feats and predict_score. "
                    "False means actual-module parity is unavailable, not successful."
                ),
                "view_score_parity_checked_positions": (
                    "Nonpadding positions with distinct nonpadding supplied positive and negative IDs "
                    "in at most eight sequences from each of the first two diagnostic batches. "
                    "Compares compatibility-view scores against actual-module scores in evaluation mode."
                ),
                "view_score_parity_nonfinite_positions": (
                    "Checked positions containing a nonfinite score in either scoring path; "
                    "excluded from the mismatch-rate denominator."
                ),
                "view_score_parity_mismatch_rate": (
                    "Fraction of finite checked positions where either supplied candidate score "
                    "differs by more than 1e-5 + 1e-4 times the actual-module score magnitude. "
                    "Null means no finite comparisons. Does not verify the generated predictor."
                ),
                "view_score_parity_max_absolute_error": (
                    "Largest absolute candidate-score difference across finite checked positions; "
                    "null means no finite comparisons."
                ),
                "causal_score_checked_positions": (
                    "Supplied-pair prefix comparisons at observed-event cutpoints in the first two "
                    "batches, capped at eight sequences per batch. Future events are replaced with "
                    "padding while positions and candidate IDs remain fixed. Uses actual-module "
                    "scoring when available, otherwise compatibility-view scoring."
                ),
                "causal_score_nonfinite_positions": (
                    "Checked prefix comparisons containing a nonfinite score before or after "
                    "future masking; excluded from the violation-rate denominator."
                ),
                "causal_score_violation_rate": (
                    "Fraction of finite checked prefix comparisons where either candidate score "
                    "changes by more than 1e-5 + 1e-4 times its original magnitude. Includes any "
                    "candidate-dependent residual exposed by the scorer. Zero is a sampled "
                    "consistency result, not proof of causality or validation improvement. "
                    "Null means no finite comparisons."
                ),
                "causal_score_max_absolute_error": (
                    "Largest absolute candidate-score change over finite checked prefix comparisons; "
                    "null means no finite comparisons."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Define populated-prefix checks and distinguish diagnostics from the proposed architecture
                "causal_checked_positions": (
                    "Number of nonpadding prefix-position comparisons in the first two diagnostic "
                    "batches, capped at eight sequences each. Cutpoints are the union of boundaries "
                    "before the second, middle and last observed events of each sampled sequence. "
                    "Only rows with an observed future event contribute at a cutpoint. Future events "
                    "are replaced by padding without shifting absolute positions. A position may "
                    "contribute at multiple cutpoints. Uses evaluation-mode log2feats features."
                ),
                "causal_nonfinite_positions": (
                    "Checked comparisons containing a nonfinite feature in either representation; "
                    "excluded from the violation-rate denominator. Positive counts indicate "
                    "numerical failures."
                ),
                "causal_prefix_violation_rate": (
                    "Fraction of finite checked positions where any feature coordinate changes "
                    "by more than 1e-5 + 1e-4 times its original absolute value after future masking. "
                    "Positive values flag future dependence or numerical sensitivity for investigation. "
                    "Zero is a limited sampled consistency check, not proof of causality or relevance. "
                    "Null means no finite comparisons."
                ),
                "causal_prefix_max_absolute_error": (
                    "Maximum absolute feature-coordinate change over finite checked prefix positions. "
                    "Scale-dependent; null means no finite comparisons. This checks augmented "
                    "features when the trained compatibility view exposes them, without assuming "
                    "an unimplemented residual branch, gate or text-support state exists."
                ),
                # Review summary: statically checked scorer dispatch, tolerance
                # validation, feature shapes, exact integer context, future masking,
                # finite-value denominators and module-mode restoration.
                # Tensor features and explicit base_features/context_ids carriers
                # are supported. Integer context is checked separately and exactly;
                # the maximum feature-error metric describes learned coordinates.
                # No execution, training or validation measurements were performed.
                # These diagnostics do not implement the proposed lexical branch,
                # verify exported predictor parity or establish HR@10 gains.
                # Global RNG restoration does not restore dedicated sampler or
                # DataLoader generators; their isolation requires host integration.
                ### <<< Self_EvolveRec-BLOCK-END
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
