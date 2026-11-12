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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measurements with their missing-data limitations
    - Missing measurements are unknown, not evidence of healthy behavior.
    - Training margins depend on raw score scale and do not establish validation gains.
    - A mapped cosine near 0.5 means mean cosine near zero, not established collapse.
    - Distinguish feature fusion from an existing candidate-dependent transition residual.
    - Do not invent metadata, preferences, simulator traits, validation outcomes, or test results.
    - Supplied limitations and unavailable measurements:
    {json.dumps({k: v for k, v in raw_diagnosis.items() if k not in ('metrics', 'metric_definitions')}, indent=2)}
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Keep the trained module distinct from its compatibility view
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.module = self.model
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.review_data = getattr(view, "review_data", {}) or {}
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Represent missing geometry measurements explicitly
            if n < 2:
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
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnostic entry point manage module modes
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption and reject malformed diagnostic batches
        import itertools

        for batch in itertools.islice(dataloader, min(num_batches, 4)):
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError("Expected (None, sequence, positive, negative)")
            _, seq, pos, neg = batch
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Measure the first 512 valid positions using the deployed fusion path
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            locations = (seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)).nonzero(
                as_tuple=False
            )[:512]
            if not locations.numel():
                continue
            b, t = locations.unbind(1)
            module = self.module
            if all(
                callable(getattr(module, name, None))
                for name in (
                    "_encode",
                    "recent_windows",
                    "_endpoint_features",
                    "predict_score",
                )
            ):
                full = module._encode(seq_t)[b, t]
                windows, lengths = module.recent_windows(seq_t, locations)
                features = module._endpoint_features(full, windows, lengths)
                items = torch.stack((pos_t[b, t], neg_t[b, t]), dim=-1)
                logits = module.predict_score(features, items)
                measured = logits[:, 0] - logits[:, 1]
            else:
                features = self.view.log2feats(seq_t)
                items = torch.stack((pos_t, neg_t), dim=-1)
                logits = self.view.predict_score(features, items)
                measured = logits[b, t, 0] - logits[b, t, 1]
            if not bool(torch.isfinite(measured).all()):
                raise FloatingPointError("Nonfinite diagnostic scores")
            margin = torch.zeros_like(pos_t, dtype=measured.dtype)
            valid = torch.zeros_like(pos_t, dtype=torch.bool)
            margin[b, t] = measured
            valid[b, t] = True
            margins_all.append(measured.detach().float().cpu())

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
            ### >>> Self_EvolveRec-BLOCK-START: Do not turn absent observations into healthy scores
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

        ### >>> Self_EvolveRec-BLOCK-START: Attach observation counts and category limitations
        breakdown = {
            "valid_positions": sum(int(m.numel()) for m in margins_all),
            "category_missing_reason": "Item category metadata is unavailable.",
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

    ### >>> Self_EvolveRec-BLOCK-START: Preserve descriptive recommendations and verify diagnostic prerequisites
    def _full_catalog_recommendations(self):
        histories = self.user_train.get("History", self.user_train)
        users = sorted(
            (user for user, history in histories.items() if len(history)),
            key=lambda user: str(user),
        )[:5]
        model = self.module
        itemnum = int(self.view.item_emb.weight.shape[0] - 1)
        maxlen = int(getattr(model, "maxlen", 0))
        if maxlen < 1:
            raise ValueError("Full-catalog recommendations require model.maxlen")
        recommendations = []
        for user in users:
            history = [int(item) for item in histories[user]]
            if any(item < 1 or item > itemnum for item in history):
                raise ValueError("Training history contains an invalid item ID")
            seen = set(history)
            suffix = history[-maxlen:]
            sequence = torch.zeros((1, maxlen), dtype=torch.long, device=self.device)
            sequence[0, -len(suffix) :] = torch.tensor(
                suffix, dtype=torch.long, device=self.device
            )
            # This optimization is specific to the supplied dot-product fusion
            # architecture. Other architectures use their complete predict path.
            fusion_api = (
                getattr(model, "fusion_mode", None) is not None
                and callable(getattr(model, "final_features", None))
                and callable(getattr(model, "predict_score", None))
                and not callable(getattr(model, "transition_evidence", None))
                and not callable(getattr(model, "score_components", None))
            )
            features = model.final_features(sequence) if fusion_api else None
            best = []
            for start in range(1, itemnum + 1, 1024):
                ids = [
                    item
                    for item in range(start, min(start + 1024, itemnum + 1))
                    if item not in seen
                ]
                if not ids:
                    continue
                candidates = torch.tensor([ids], dtype=torch.long, device=self.device)
                if fusion_api:
                    scores = model.predict_score(features, candidates)[0]
                elif callable(getattr(model, "predict", None)):
                    scores = model.predict(sequence, candidates)[0]
                else:
                    features = self.view.log2feats(sequence)
                    scores = self.view.predict_score(
                        features,
                        candidates[:, None, :].expand(1, maxlen, -1),
                    )[0, -1]
                if scores.shape != (len(ids),):
                    raise ValueError("Unexpected full-catalog score shape")
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("Nonfinite recommendation scores")
                best.extend(zip(scores.detach().float().cpu().tolist(), ids))
                best = sorted(best, key=lambda pair: (-pair[0], pair[1]))[:10]
            recommendations.append(
                {"user": str(user), "items": [item for _, item in best]}
            )
        return recommendations

    @torch.no_grad()
    def _fusion_diagnostics(self, sequence):
        # DEBUG: Measure the implemented fusion branch through its shared
        # window construction and fusion routines; do not infer a transition
        # residual from differences between normalized representations.
        module = self.module
        required = ("_encode", "recent_windows", "fuse", "_normalize")
        if not all(callable(getattr(module, name, None)) for name in required):
            return None
        locations = sequence.ne(0).nonzero(as_tuple=False)[:512]
        if not locations.numel():
            return {}
        b, t = locations.unbind(1)
        full = module._encode(sequence)[b, t]
        windows, lengths = module.recent_windows(sequence, locations)
        recent = module._encode(windows)[:, -1]
        _, gate = module.fuse(full, recent, lengths)
        full_component = (
            full if module.fusion_mode == "baseline" else module._normalize(full)
        )
        values = {
            "fusion_gate": gate.flatten(),
            "fusion_full_raw_norm": full.norm(dim=-1),
            "fusion_recent_raw_norm": recent.norm(dim=-1),
            "fusion_full_contribution_norm": ((1 - gate) * full_component).norm(dim=-1),
            "fusion_recent_contribution_norm": (gate * module._normalize(recent)).norm(
                dim=-1
            ),
            "fusion_branch_cosine": F.cosine_similarity(full, recent, dim=-1),
        }
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise FloatingPointError("Nonfinite fusion diagnostic measurements")
        return {name: value.detach().float().cpu() for name, value in values.items()}

    def run_full_diagnosis(self, train_loader):
        flags = [(module, module.training) for module in self.module.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        try:
            self.module.eval()
            with torch.no_grad():
                # DEBUG: Materialize at most four batches once. A training
                # loader may be a one-shot iterator; separate passes would
                # otherwise consume different data or leave a probe empty.
                import itertools

                batches = list(itertools.islice(train_loader, 4))
                for batch in batches:
                    if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                        raise ValueError(
                            "Expected (None, sequence, positive, negative)"
                        )
                    shapes = [tuple(torch.as_tensor(x).shape) for x in batch[1:]]
                    if (
                        any(len(shape) != 2 for shape in shapes)
                        or len(set(shapes)) != 1
                    ):
                        raise ValueError(
                            "Diagnostic tensors must have matching B x L shapes"
                        )
                collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=4, eps=0.5, topk_categories=20
                    )
                )
                fusion_samples = {}
                fusion_supported = all(
                    callable(getattr(self.module, name, None))
                    for name in ("_encode", "recent_windows", "fuse", "_normalize")
                )
                for _, sequence, _, _ in batches:
                    measured = self._fusion_diagnostics(
                        torch.as_tensor(sequence, device=self.device, dtype=torch.long)
                    )
                    if measured is not None:
                        for name, values in measured.items():
                            fusion_samples.setdefault(name, []).append(values)
                recommendations = self._full_catalog_recommendations()
                transition_hooks = {
                    name: callable(getattr(self.module, name, None))
                    for name in ("score_components", "transition_evidence")
                }
                transition_state_keys = sorted(
                    name
                    for name, _ in self.module.named_buffers()
                    if name.startswith("transition_")
                )
                if any(transition_hooks.values()) or transition_state_keys:
                    transition_reason = (
                        "Transition-related hooks or buffers exist, but this adapter "
                        "does not verify their score decomposition, checkpoint or "
                        "export parity. Calibration prerequisites remain unverified; "
                        "no alpha comparison was performed."
                    )
                else:
                    transition_reason = (
                        "The actual trained module exposes neither score_components "
                        "nor transition_evidence nor transition-prefixed buffers. "
                        "The supplied fusion implementation scores embedding dot "
                        "products; transition calibration is unavailable. Historical "
                        "diagnostics are not evidence of a deployed transition branch."
                    )
                metrics = {
                    "full_catalog_user_count": len(recommendations),
                    "embedding_collapse_score": collapse,
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "pos_neg_valid_positions": breakdown["valid_positions"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "transition_alpha_sensitivity": None,
                }
                definitions = {
                    "full_catalog_user_count": "Count of actual training users receiving unseen-item full-catalog top-ten lists in this dataset view; five per category are required for twenty users overall. Lists use raw deployed scores and item-ID tie breaking and have no relevance labels.",
                    "embedding_collapse_score": "Unitless mapped mean cosine (cosine+1)/2 of sampled nonpadding embeddings. A value near 0.5 means mean cosine near zero; this alone does not establish collapse or recommendation diversity.",
                    "pos_neg_margin_mean": "Mean deployed positive-minus-negative training score, in raw logit units; depends on score scale.",
                    "pos_neg_margin_p10": "Nearest-rank tenth percentile of the sampled training margins, in raw logit units.",
                    "pos_neg_neg_beats_pos_rate": "Unitless fraction of sampled training margins strictly below zero; ties excluded.",
                    "pos_neg_small_margin_rate": "Unitless fraction of sampled training margins below 0.5 raw logit units; no universal healthy threshold.",
                    "pos_neg_valid_positions": "Observation count for all margin summaries: first 512 valid positions per batch from at most four batches.",
                    "pos_neg_margin_category_breakdown": "Observed pairwise counts with empty category lists because item category metadata is unavailable.",
                    "transition_alpha_sensitivity": "Unitless ranking sensitivity to scaling a verified existing gated transition residual. Null means no verified alpha comparison was performed; it does not mean zero sensitivity.",
                }
                unavailable = {
                    "transition_alpha_sensitivity": transition_reason,
                    "item_categories": "Item category metadata is unavailable.",
                    "review_preferences": "Review ratings and text are unavailable.",
                    "simulator_traits": "Diversity and conformity traits are unobserved.",
                    "paired_validation_diagnostics": "Not supplied by the independent host evaluator.",
                }
                # DEBUG: Keep fusion measurements reachable in the canonical
                # adapter and export their definitions and observation counts.
                fusion_definitions = {
                    "fusion_gate": "Mean recent-branch scalar mixture weight; variation alone does not establish intent.",
                    "fusion_full_raw_norm": "Mean L2 norm of the unnormalized full-prefix branch output.",
                    "fusion_recent_raw_norm": "Mean L2 norm of the unnormalized independent recent-window branch output.",
                    "fusion_full_contribution_norm": "Mean L2 norm of (1-g) times normalized full features; raw full features in baseline mode.",
                    "fusion_recent_contribution_norm": "Mean L2 norm of g times normalized recent features.",
                    "fusion_branch_cosine": "Mean cosine between raw full and recent outputs; cancellation or intent cannot be established from this value alone.",
                }
                for name, definition in fusion_definitions.items():
                    parts = fusion_samples.get(name, [])
                    count = sum(part.numel() for part in parts)
                    definitions[name] = definition
                    definitions[name + "_count"] = (
                        "Number of measured nonpadding endpoints, limited to the "
                        "first 512 per batch in at most four batches."
                    )
                    metrics[name + "_count"] = count
                    metrics[name] = float(torch.cat(parts).mean()) if count else None
                    if not count:
                        unavailable[name] = (
                            "No eligible sampled endpoints."
                            if fusion_supported
                            else "The actual module lacks the shared fusion diagnostic API."
                        )
                if len(recommendations) < 5:
                    unavailable["twenty_user_report"] = (
                        "This category has fewer than five nonempty training histories."
                    )
                if collapse is None:
                    unavailable["embedding_collapse_score"] = "Fewer than two items."
                if not breakdown["valid_positions"]:
                    unavailable["margin_summaries"] = "No eligible sampled positions."
                return {
                    "metrics": metrics,
                    "metric_definitions": definitions,
                    "full_catalog_recommendations": recommendations,
                    "dataset_name": getattr(self.view, "dataset_name", None),
                    "transition_prerequisites": {
                        "module_class": type(self.module).__qualname__,
                        "hooks": transition_hooks,
                        "buffer_names": transition_state_keys,
                        "checkpoint_and_export_verified_by_probe": False,
                    },
                    "unavailable": unavailable,
                    "limitations": [
                        "Training separation does not establish validation improvement or preferences.",
                        "No alpha sweep or substitute intervention is performed by this diagnostic adapter.",
                        "Full-catalog lists are descriptive, with five real training users per category when available.",
                        "Checkpoint and generated-predictor verification belong to the training/export workflow.",
                    ],
                }
        finally:
            for module, training in flags:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    ### <<< Self_EvolveRec-BLOCK-END
