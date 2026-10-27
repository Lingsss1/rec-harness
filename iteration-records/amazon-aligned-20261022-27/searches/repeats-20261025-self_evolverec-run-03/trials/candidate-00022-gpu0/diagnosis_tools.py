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
    ### >>> Self_EvolveRec-BLOCK-START: Retain the compatibility view and evaluate the actual deployed module
    def __init__(self, view, device):
        self.model = view
        self.module = view.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("view.model must be the actual deployed nn.Module")
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self._residual_pairs = []

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Compute directional concentration without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()[1:]
            n = emb.size(0)
            if n < 2:
                return None

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)
            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite sampled item embeddings")
            x = F.normalize(x, p=2, dim=-1, eps=1e-8)

            # Sum of all pairwise dot products equals the squared norm
            # of the vector sum. Subtract self-pairs, including zero vectors.
            total = x.sum(dim=0, dtype=torch.float64)
            diagonal = x.double().square().sum()
            mean_cos = (
                ((total.square().sum() - diagonal) / (k * (k - 1))).clamp(-1, 1).item()
            )
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
        # DEBUG: Anonymous batches cannot route a user-excluded transition table.
        # Measure neural margins here; use a separate identified stream below
        # for the actual transition residual instead of the stale cosine layout.
        from itertools import islice

        if num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        if not np.isfinite(eps):
            raise ValueError("eps must be finite")
        self._residual_pairs = []
        margins_all = []
        if not callable(getattr(self.module, "predict_score_base", None)):
            raise TypeError("The trained module must expose predict_score_base")
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Reject malformed batches without consuming an extra diagnostic batch
        for step, batch in enumerate(islice(dataloader, num_batches)):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    f"Diagnostic batch {step} must contain (None, sequence, positive, negative)"
                )
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned diagnostic tensors before model scoring
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnostic sequence, positive and negative tensors must share shape B x L"
                )
            # DEBUG: The compatibility view may implement only neural scoring;
            # make that scope explicit and score the actual trained architecture.
            log_feats = self.module.log2feats(seq_t)
            if log_feats.ndim != 3 or log_feats.shape[:2] != seq_t.shape:
                raise ValueError(
                    "Model log2feats must return aligned B x L x D features"
                )
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.module.predict_score_base(log_feats, all_items)
            if all_logits.shape != all_items.shape:
                raise ValueError(
                    "Model predict_score_base must return aligned B x L x 2 scores"
                )
            margin = all_logits[..., 0].float() - all_logits[..., 1].float()
            ### <<< Self_EvolveRec-BLOCK-END

            # DEBUG: Keep anonymous neural observations out of fold-dependent
            # comparisons; no identity is inferred from a truncated sequence.
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            if valid.any():
                if (
                    not torch.isfinite(all_logits[valid]).all()
                    or not torch.isfinite(margin[valid]).all()
                ):
                    raise ValueError("Nonfinite neural diagnostic scores or margins")
                margins_all.append(margin[valid].detach().float().cpu())

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize valid-position counts without unavailable category lookups
            # Category observations are unavailable, so their counters stay empty.
            # Count on-device instead of transferring three B x L arrays and
            # iterating over every low-margin or incorrectly ranked position.
            counts = (
                torch.stack(
                    (
                        (valid & (margin < float(eps))).sum(),
                        (valid & (margin < 0.0)).sum(),
                    )
                )
                .cpu()
                .tolist()
            )
            low_total += int(counts[0])
            wrong_total += int(counts[1])
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Missing diagnostic samples are unavailable rather than healthy zeros
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

    # DEBUG: Execute transition-on/off diagnostics on identified causal prefixes.
    # Local RNGs preserve training randomness. Complement-rank sampling avoids
    # catalog scans and excludes every observed training item for this user.
    @torch.no_grad()
    def _collect_transition_pairs(self, max_examples=512):
        from bisect import bisect_right

        self._residual_pairs = []
        transition = getattr(self.module, "transition", None)
        if transition is None:
            return {
                "status": "unavailable",
                "reason": "The supplied trained module has no persisted transition tables",
            }
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        category = transition.provenance.get("dataset_name")
        if category is None or not transition.user_folds:
            return {
                "status": "unavailable",
                "reason": "Persisted category or user-fold provenance is unavailable",
            }

        rng = random.Random(42)
        users = sorted(self.user_train)
        rng.shuffle(users)
        records = []
        missing_identity_users = 0
        no_negative_users = 0
        itemnum = self.module.itemnum
        maxlen = self.module.maxlen
        device = self.module.item_emb.weight.device
        if maxlen < 1:
            raise ValueError("Diagnostic prefixes require a positive maxlen")

        for user in users:
            history = [int(item) for item in self.user_train[user]]
            if any(item < 1 or item > itemnum for item in history):
                raise ValueError("Diagnostic histories contain invalid catalog IDs")
            if len(history) < 2:
                continue
            if str(user) not in transition.user_folds:
                missing_identity_users += 1
                continue
            excluded = sorted(set(history))
            eligible_count = itemnum - len(excluded)
            if eligible_count < 1:
                no_negative_users += 1
                continue
            offsets = [item - index - 1 for index, item in enumerate(excluded)]
            # Three recent next-item examples per user, in chronological order.
            for endpoint in range(max(0, len(history) - 4), len(history) - 1):
                prefix = history[max(0, endpoint + 1 - maxlen) : endpoint + 1]
                sequence = [0] * (maxlen - len(prefix)) + prefix
                rank = rng.randrange(eligible_count)
                negative = rank + 1 + bisect_right(offsets, rank)
                candidates = [history[endpoint + 1], negative]
                seq_t = torch.tensor([sequence], dtype=torch.long, device=device)
                cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
                features = self.module.log2feats(seq_t)[:, -1, :]
                base = self.module.predict_score_base(features, cand_t).float()
                residual = transition(seq_t, cand_t, user_ids=[user]).float()
                deployed = base + transition.alpha * residual
                # Compare with the exact inference entry point on the same inputs.
                exported_formula = self.module.predict(
                    seq_t, cand_t, user_ids=[user]
                ).float()
                if not torch.isfinite(exported_formula).all():
                    raise ValueError("Nonfinite identified transition scores")
                if not torch.allclose(deployed, exported_formula, atol=1e-5, rtol=1e-5):
                    raise ValueError(
                        "Diagnostic and inference transition scores disagree"
                    )
                paired = torch.stack((base, deployed), dim=1)
                if not torch.isfinite(paired).all():
                    raise ValueError("Nonfinite identified paired scores")
                self._residual_pairs.append(paired.detach().cpu())
                records.append(
                    {
                        "category": category,
                        "user_id": str(user),
                        "prefix_endpoint": endpoint,
                        "example_id": f"{category}:{user}:{endpoint}",
                        "fold": transition.route([user], 1)[0],
                        "prefix": prefix,
                        "candidates": candidates,
                        "base_scores": base[0].cpu().tolist(),
                        "deployed_scores": deployed[0].cpu().tolist(),
                    }
                )
                if len(records) >= max_examples:
                    break
            if len(records) >= max_examples:
                break

        summary = self._summarize_residual_pairs()
        summary.update(
            {
                "candidate_protocol": "Training positive and uniform unseen negative; positive-first exact ties",
                "sampling_scope": "Seed-42 shuffled sorted users, up to three recent causal prefixes per user, capped at 512 examples",
                "routing_scope": "Identified training users scored with their persisted excluded-fold table",
                "table_version": transition.provenance.get("version"),
                "transition_alpha": transition.alpha,
                "missing_identity_users_encountered": missing_identity_users,
                "no_negative_users_encountered": no_negative_users,
                "records": records,
            }
        )
        return summary

    ### >>> Self_EvolveRec-BLOCK-START: Audit paired residual statistics while preserving model modes and global random states
    def run_full_diagnosis(self, train_loader):
        # DEBUG: Anonymous neural and identified transition diagnostics have
        # separate scopes. No validation improvement or execution is claimed.
        prior_modes = [(module, module.training) for module in self.module.modules()]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        # This preserves global RNGs only. A supplied loader's private generator
        # and iterator state remain owned by its caller.
        self._residual_pairs = []
        try:
            self.module.eval()
            with torch.no_grad():
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=0.5, topk_categories=20
                    )
                )
            # DEBUG: This path is reachable for every full diagnosis and uses
            # actual persisted transition tables whenever the model has them.
            paired = self._collect_transition_pairs()
        finally:
            self._residual_pairs = []
            for module, training in prior_modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "residual_fixed_candidate_comparison": paired,
                "unavailable_observations": {
                    "item_categories": None,
                    "reviews_and_ratings": None,
                    "item_prices_and_stores": None,
                    "product_specifications": None,
                    "simulator_diversity_and_conformity_traits": None,
                },
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "Mean sampled pairwise embedding cosine mapped to [0,1]; approximately 0.5 "
                    "means mean cosine zero. Measures directional concentration only. "
                    "Zero vectors use F.normalize semantics; null means fewer than two items."
                ),
                # DEBUG: Anonymous loader metrics describe the neural component.
                "pos_neg_margin_mean": "Mean neural-only positive-minus-negative supplied training score; scale dependent; null if unavailable.",
                "pos_neg_margin_p10": "Empirical lower tenth percentile of neural-only supplied training-pair margins; scale dependent.",
                "pos_neg_neg_beats_pos_rate": "Fraction of valid anonymous training pairs where the neural negative strictly outscores the positive.",
                "pos_neg_small_margin_rate": "Fraction of neural-only supplied training-pair margins below 0.5; scale dependent and not probability calibrated.",
                "pos_neg_margin_category_breakdown": (
                    "Counts valid low-margin and wrong-rank positions. Category lists are empty "
                    "because item category observations are unavailable."
                ),
                # DEBUG: Replace the obsolete cosine-feature diagnostic contract.
                "residual_fixed_candidate_comparison": (
                    "Separate deterministic training-only stream carrying category, user ID, "
                    "prefix endpoint, candidates and persisted excluded-fold provenance. Each "
                    "prefix ends before its positive target. Negatives are uniformly drawn "
                    "from the complement of that user's complete training history using a "
                    "local seed-42 generator. Anonymous loader pairs never establish ownership. "
                    "The same candidates receive neural-only and selected transition-on scores. "
                    "Strict negative wins define rank two; exact ties favor the positive. "
                    "Near ties use abs(positive-negative) <= 1e-5 + "
                    "1e-5*max(abs(positive),abs(negative)) without changing ranks. "
                    "Moments and covariance use float64 population estimators. The increment "
                    "includes the selected transition coefficient. Correlation is null for "
                    "fewer than two pairs or zero variance. Records support an audit of "
                    "prefix construction and user-fold routing; missing provenance or valid "
                    "pairs makes comparisons unavailable. Scores are checked against the "
                    "model's inference entry point with atol=rtol=1e-5; this does not execute "
                    "the generated predictor. These are OOF training diagnostics, not "
                    "validation HR@10, probability calibration or evidence of global temporal causality."
                ),
                "unavailable_observations": (
                    "Null denotes unavailable evidence. Histories and embedding similarity do not establish "
                    "ownership, dislike, satisfaction, compatibility, current need or simulator traits."
                ),
            },
        }

    def _summarize_residual_pairs(self):
        if not self._residual_pairs:
            return {
                "status": "unavailable",
                "reason": "No valid paired observations with an explicit residual-off scoring contract",
            }
        # Cast before subtraction: all subsequent statistics use float64.
        scores = torch.cat(self._residual_pairs, dim=0).double()
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Nonfinite paired residual diagnostic scores")
        margins = scores[..., 0] - scores[..., 1]
        base, deployed = margins[:, 0], margins[:, 1]
        increment = deployed - base
        if not bool(torch.isfinite(increment).all()):
            raise ValueError("Nonfinite paired margin increments")
        outranks = margins < 0
        ties = margins == 0
        near_ties = margins.abs() <= (1e-5 + 1e-5 * scores.abs().amax(dim=-1))
        corrected = outranks[:, 0] & ~outranks[:, 1]
        regressed = ~outranks[:, 0] & outranks[:, 1]
        base_centered = base - base.mean()
        increment_centered = increment - increment.mean()
        base_variance = base_centered.square().mean()
        increment_variance = increment_centered.square().mean()
        covariance = (base_centered * increment_centered).mean()
        deployed_variance = (deployed - deployed.mean()).square().mean()
        correlation = None
        if (
            scores.shape[0] >= 2
            and float(base_variance) > 0.0
            and float(increment_variance) > 0.0
        ):
            correlation = float(
                (covariance / (base_variance.sqrt() * increment_variance.sqrt())).clamp(
                    -1.0, 1.0
                )
            )

        def summarize_group(mask):
            count = int(mask.sum())
            return {
                "positions": count,
                "base_margin_mean": float(base[mask].mean()) if count else None,
                "deployed_margin_mean": float(deployed[mask].mean()) if count else None,
                "margin_increment_mean": (
                    float(increment[mask].mean()) if count else None
                ),
                "near_tie_in_either_model": int(near_ties[mask].any(dim=1).sum()),
            }

        expected_variance = base_variance + increment_variance + 2 * covariance
        return {
            "status": "available",
            "candidate_protocol": "Identical supplied positive/negative training pairs; positive-first exact ties",
            "sampling_scope": "At most first ten supplied loader batches; calibration sampling provenance unverified",
            "positions": int(scores.shape[0]),
            "residual_off_mean_positive_rank": float(
                1 + outranks[:, 0].double().mean()
            ),
            "deployed_mean_positive_rank": float(1 + outranks[:, 1].double().mean()),
            "residual_off_negative_outranking_fraction": float(
                outranks[:, 0].double().mean()
            ),
            "deployed_negative_outranking_fraction": float(
                outranks[:, 1].double().mean()
            ),
            "residual_off_tie_fraction": float(ties[:, 0].double().mean()),
            "deployed_tie_fraction": float(ties[:, 1].double().mean()),
            "residual_off_near_tie_fraction": float(near_ties[:, 0].double().mean()),
            "deployed_near_tie_fraction": float(near_ties[:, 1].double().mean()),
            "corrected_pairs": int(corrected.sum()),
            "regressed_pairs": int(regressed.sum()),
            "corrected_pair_margins": summarize_group(corrected),
            "regressed_pair_margins": summarize_group(regressed),
            "mean_absolute_score_change": float(
                (scores[:, 1] - scores[:, 0]).abs().mean()
            ),
            "base_margin_mean": float(base.mean()),
            "deployed_margin_mean": float(deployed.mean()),
            "deployed_increment_margin_mean": float(increment.mean()),
            "base_margin_population_std": float(base_variance.sqrt()),
            "deployed_margin_population_std": float(deployed_variance.sqrt()),
            "deployed_increment_margin_population_std": float(
                increment_variance.sqrt()
            ),
            "base_increment_margin_correlation": correlation,
            "base_increment_margin_population_covariance": float(covariance),
            "variance_identity_absolute_error": float(
                (deployed_variance - expected_variance).abs()
            ),
            "statistic_dtype": "float64_from_float32_scores",
        }

    ### <<< Self_EvolveRec-BLOCK-END
