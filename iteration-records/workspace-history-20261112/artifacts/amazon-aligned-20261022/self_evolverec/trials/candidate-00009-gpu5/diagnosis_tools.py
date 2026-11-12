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

    ### >>> Self_EvolveRec-BLOCK-START: Validate compatibility checks and compare packed context IDs exactly
    @torch.no_grad()
    def analyze_view_parity(self, batches, atol=1e-5, rtol=1e-4):
        if not all(np.isfinite(x) and x >= 0 for x in (atol, rtol)):
            raise ValueError("Parity tolerances must be finite and nonnegative")
        feature_fn = getattr(self.module, "log2feats", None)
        score_fn = getattr(self.module, "predict_score", None)
        if not callable(feature_fn) or not callable(score_fn):
            return {
                "view_parity_available": False,
                "view_parity_reason": "Actual module lacks log2feats or predict_score.",
                "view_parity_checked_positions": 0,
                "view_parity_nonfinite_positions": 0,
                "view_feature_mismatch_rate": None,
                "view_score_mismatch_rate": None,
                "view_score_max_absolute_error": None,
            }

        checked = nonfinite = feature_failures = score_failures = 0
        max_score_error = None
        prior_modes = [(module, module.training) for module in self.module.modules()]
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
                actual_features = feature_fn(seq)
                view_features = self.model.log2feats(seq.detach().cpu().numpy())
                if (
                    actual_features.ndim != 3
                    or actual_features.shape[:2] != seq.shape
                    or view_features.shape != actual_features.shape
                ):
                    raise ValueError(
                        "Compatibility view must expose the actual module's B x L x D features"
                    )
                actual_scores = score_fn(actual_features, items)
                view_scores = self.model.predict_score(view_features, items)
                if (
                    actual_scores.shape != items.shape
                    or view_scores.shape != items.shape
                ):
                    raise ValueError(
                        "Both scorers must return one score per supplied item"
                    )

                # Preserve packed integer context coordinates during comparison.
                actual_features = actual_features.detach().to(dtype=torch.float64)
                view_features = view_features.detach().to(
                    device=actual_features.device, dtype=torch.float64
                )
                actual_scores = actual_scores.detach().to(dtype=torch.float64)
                view_scores = view_scores.detach().to(
                    device=actual_scores.device, dtype=torch.float64
                )
                finite = (
                    torch.isfinite(actual_features).all(-1)
                    & torch.isfinite(view_features).all(-1)
                    & torch.isfinite(actual_scores).all(-1)
                    & torch.isfinite(view_scores).all(-1)
                )
                checked += int(valid.sum().item())
                nonfinite += int((valid & ~finite).sum().item())
                usable = valid & finite
                if not usable.any():
                    continue
                feature_error = (actual_features - view_features).abs()
                score_error = (actual_scores - view_scores).abs()
                feature_failed = self._feature_mismatch(
                    actual_features, view_features, atol, rtol
                )
                score_failed = (score_error > atol + rtol * actual_scores.abs()).any(-1)
                feature_failures += int((usable & feature_failed).sum().item())
                score_failures += int((usable & score_failed).sum().item())
                largest = float(score_error[usable].max().item())
                max_score_error = (
                    largest
                    if max_score_error is None
                    else max(max_score_error, largest)
                )
        finally:
            for module, training in prior_modes:
                module.training = training

        finite_count = checked - nonfinite
        return {
            "view_parity_available": True,
            "view_parity_reason": None,
            "view_parity_checked_positions": checked,
            "view_parity_nonfinite_positions": nonfinite,
            "view_feature_mismatch_rate": (
                feature_failures / finite_count if finite_count else None
            ),
            "view_score_mismatch_rate": (
                score_failures / finite_count if finite_count else None
            ),
            "view_score_max_absolute_error": max_score_error,
        }

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Check causal features without rounding or tolerating changes to packed IDs
    def _feature_mismatch(self, reference, altered, atol, rtol):
        if reference.shape != altered.shape:
            raise ValueError("Compared feature tensors must have identical shapes")
        # The supplied transition scorer packs exact integer IDs after H floats.
        # Relative tolerances appropriate for learned features are inappropriate
        # for IDs: at large IDs they could accept changes of multiple items.
        hidden = getattr(self.module, "hidden_units", None)
        packed_context = (
            hasattr(self.module, "transition_keys")
            and isinstance(hidden, int)
            and reference.shape[-1] == hidden + 20
        )
        if packed_context:
            numerical = (
                (reference[..., :hidden] - altered[..., :hidden]).abs()
                > atol + rtol * reference[..., :hidden].abs()
            ).any(-1)
            categorical = (reference[..., hidden:] != altered[..., hidden:]).any(-1)
            return numerical | categorical
        return ((reference - altered).abs() > atol + rtol * reference.abs()).any(-1)

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
            if reference.ndim != 3 or reference.shape[:2] != seq.shape:
                raise ValueError("log2feats must return B x L x D features")
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
                if altered.shape != reference.shape:
                    raise ValueError("log2feats shape changed after future masking")
                # Preserve float64-packed item IDs rather than rounding through
                # float32, which cannot represent every integer above 2**24.
                before = reference[:, :cut].detach().to(dtype=torch.float64)
                after = (
                    altered[:, :cut]
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
                failed = self._feature_mismatch(before, after, atol, rtol)
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
        parity_metrics = self.analyze_view_parity(batches)
        return {
            ### <<< Self_EvolveRec-BLOCK-END
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Expose causal invariance and actual-module compatibility checks
                **causal_metrics,
                **parity_metrics,
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
                ### >>> Self_EvolveRec-BLOCK-START: Define actual-module compatibility measurements without claiming export parity
                "view_parity_available": (
                    "Whether the actual trained module exposes callable log2feats and "
                    "predict_score methods for comparison with the diagnostic compatibility view."
                ),
                "view_parity_reason": (
                    "Reason the actual-module comparison is unavailable; null when supported."
                ),
                "view_parity_checked_positions": (
                    "Nonpadding positions with distinct nonpadding positive and negative IDs "
                    "in the first two diagnostic batches, capped at eight sequences per batch. "
                    "Compares evaluation-mode features and supplied-pair scores from the "
                    "compatibility view against the actual trained module."
                ),
                "view_parity_nonfinite_positions": (
                    "Checked positions containing a nonfinite feature or score in either path; "
                    "excluded from both mismatch-rate denominators."
                ),
                "view_feature_mismatch_rate": (
                    "Fraction of finite checked positions with a compatibility-view feature "
                    "mismatch. Learned coordinates use tolerance 1e-5 + 1e-4 times the actual "
                    "coordinate's absolute value. For the supplied transition architecture, "
                    "the 20 packed context IDs must match exactly. Comparisons preserve "
                    "float64-packed IDs. Null means no finite comparisons."
                ),
                "view_score_mismatch_rate": (
                    "Fraction of finite checked positions where either supplied item score "
                    "differs between the compatibility view and actual module by more than "
                    "1e-5 + 1e-4 times the actual score's absolute value. Null means no finite "
                    "comparisons. Positive values can reveal a view bypassing a trained residual. "
                    "Zero does not verify the independently generated predictor."
                ),
                "view_score_max_absolute_error": (
                    "Maximum absolute supplied-item score difference between the compatibility "
                    "view and actual trained module over finite checked positions; null when "
                    "no finite comparisons exist. This is an implementation check, not HR@10."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Define exact packed-context checks and document implementation limits
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
                    "Fraction of finite checked positions with a feature mismatch after future "
                    "masking. Learned coordinates use tolerance 1e-5 + 1e-4 times the original "
                    "coordinate's absolute value. For the supplied transition architecture, "
                    "the 20 packed context IDs must match exactly, preserving float64 precision. "
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
                # Review summary: inspected signatures, helper call sites, feature shapes,
                # finite-value accounting, packed-ID precision and mode restoration.
                # No execution, training or validation measurements were performed.
                # The supplied model implements a transition-graph residual, not the
                # proposed lexical two-witness residual. These diagnostic changes do not
                # implement that architecture or establish an HR@10 improvement.
                # Export parity remains a separate check. Global RNG restoration does
                # not restore dedicated sampler or DataLoader generator states.
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
