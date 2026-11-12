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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret execution coverage before behavioral sensitivity
    - Ground every finding in measured metrics, definitions, denominators and comparison coverage.
    - Inspect execution_audit checks individually: skipped/error is not executed zero error. Audit success covers only the explicitly executed stages.
    - Separate within-configuration repeatability, batch-shape variation, changed-padding comparisons, same-tensor readout errors and exported-predictor disagreement.
    - Missing internal hooks or float64 references leave those explanations unresolved. Do not diagnose a numerical cause from discrepancies alone.
    - Analytical score and margin dependence does not establish rank changes, top-ten changes, useful memory, user intent or recommendation quality.
    - Do not claim pinned evaluator parity when its ranking/tie helper is unavailable.
    - Null temporal metrics mean unavailable evidence. Finite fixture agreement does not prove general padding invariance or remove deletion confounding.
    - Loader rows have no reliable user identities; descriptive aggregates may overweight frequent users. Null uncertainty is not zero uncertainty.
    - Distinguish composite deletion, final-item sensitivity and context sensitivity with the final item retained. Withhold causal recency conclusions.
    - Do not infer ownership, compatibility, ratings, reviews, simulator traits or unavailable item metadata.
    - Embedding geometry does not directly measure recommendation diversity.
    - Only independent official four-category macro validation HR@10 selects models; these audits establish no validation improvement.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve ModelView and prohibit unavailable category evidence
    def __init__(self, view, device):
        self.model = view
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Category metadata is unavailable in this matched task.
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

            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable embedding geometry without inventing a collapse value
            n = emb.size(0)
            if n < 2 or sample_size < 2:
                return None
            if not bool(torch.isfinite(emb).all()):
                raise RuntimeError(
                    "Nonfinite real-item embeddings encountered in diagnosis"
                )
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
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnostic runner manage actual module evaluation mode
        # run_full_diagnosis sets view.model.eval() and restores its original flag.
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Validate IDs before embedding lookup and reject nonfinite margins
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or seq_t.size(1) == 0
                or seq_t.shape != pos_t.shape
                or seq_t.shape != neg_t.shape
            ):
                raise ValueError(
                    "Diagnostic loader must yield matching nonempty B x L tensors"
                )
            itemnum = int(self.model.model.itemnum)
            history_valid = ((seq_t >= 0) & (seq_t <= itemnum)).all(dim=1)
            valid = (
                history_valid[:, None]
                & (seq_t > 0)
                & (pos_t > 0)
                & (pos_t <= itemnum)
                & (neg_t > 0)
                & (neg_t <= itemnum)
                & (pos_t != neg_t)
            )
            # Invalid histories are excluded entirely. Safe placeholders prevent
            # invalid IDs from reaching embeddings before the validity mask applies.
            safe_seq = seq_t.masked_fill(~history_valid[:, None], 0)
            log_feats = self.model.log2feats(safe_seq.detach().cpu().numpy())
            if log_feats.ndim != 3 or tuple(log_feats.shape[:2]) != tuple(seq_t.shape):
                raise RuntimeError("ModelView.log2feats must return B x L x D features")
            all_items = torch.stack(
                [pos_t.masked_fill(~valid, 0), neg_t.masked_fill(~valid, 0)], dim=-1
            )
            all_logits = self.model.predict_score(log_feats, all_items)
            if tuple(all_logits.shape) != tuple(all_items.shape):
                raise RuntimeError(
                    "ModelView.predict_score must return B x L x 2 scores"
                )
            margin = all_logits[..., 0] - all_logits[..., 1]
            if not bool(torch.isfinite(margin[valid]).all()):
                raise RuntimeError(
                    "Nonfinite scores encountered in valid diagnostic training pairs"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same validity mask for margin breakdown counts
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

        ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing sampled-margin evidence from healthy zero metrics
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

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Count the exact observations used by margin summaries
        breakdown = {
            "valid_positions": sum(int(values.numel()) for values in margins_all),
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

    ### >>> Self_EvolveRec-BLOCK-START: Trace fixed GRU fixtures through readout, scoring and the emitted predictor
    def _execution_audit(self, batches):
        import hashlib
        import importlib.util
        import os
        import pickle
        import platform

        module = self.model.model
        device = next(module.parameters()).device
        checks = []
        audit = {
            "checks": checks,
            "atol": 1e-5,
            "rtol": 1e-4,
            "reference": "First mixed-length batched execution; unchanged ordered candidates",
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": str(device),
                "dtype": str(next(module.parameters()).dtype),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            },
            "checkpoint_sha256": None,
            "architecture": getattr(module, "architecture_config", None),
            "mapping_status": "Public integer IDs retained unchanged; no separate ID mapping supplied",
            "trace": {},
        }

        def unavailable(name, reason, status="skipped"):
            checks.append(
                {
                    "name": name,
                    "status": status,
                    "reason": reason,
                    "elements": 0,
                    "violations": None,
                    "max_abs_error": None,
                    "max_relative_error": None,
                    "error_location": None,
                }
            )

        def compare(name, actual, reference):
            actual = torch.as_tensor(actual).detach().cpu()
            reference = torch.as_tensor(reference).detach().cpu()
            entry = {
                "name": name,
                "status": "executed/fail",
                "reference": audit["reference"],
                "dtype": str(actual.dtype),
                "elements": actual.numel(),
                "violations": None,
                "max_abs_error": None,
                "max_relative_error": None,
                "error_location": None,
            }
            checks.append(entry)
            if actual.shape != reference.shape:
                entry["reason"] = (
                    f"Shape mismatch: {actual.shape} versus {reference.shape}"
                )
                return
            if not actual.numel():
                entry.update(status="skipped", reason="Empty comparison")
                return
            if not bool(
                torch.isfinite(actual).all() and torch.isfinite(reference).all()
            ):
                entry["reason"] = "Nonfinite comparison input"
                return
            a, r = actual.double(), reference.double()
            error = (a - r).abs()
            bad = error > 1e-5 + 1e-4 * r.abs()
            entry.update(
                status="executed/fail" if bool(bad.any()) else "executed/pass",
                violations=int(bad.sum()),
                max_abs_error=float(error.max()),
                max_relative_error=float((error / r.abs().clamp_min(1e-12)).max()),
                # JSON cannot serialize NumPy integer coordinates directly.
                error_location=[
                    int(index)
                    for index in np.unravel_index(int(error.argmax()), error.shape)
                ],
            )

        def features(sequence):
            value = self.model.log2feats(sequence.detach().cpu().numpy())
            if (
                not torch.is_tensor(value)
                or value.shape[:2] != sequence.shape
                or value.ndim != 3
            ):
                raise ValueError("ModelView features must have shape B x L x H")
            return value.detach().clone()

        def last(value, sequence):
            positions = torch.arange(sequence.size(1), device=device).expand_as(
                sequence
            )
            index = positions.masked_fill(sequence.eq(0), -1).amax(1)
            selected = value[
                torch.arange(sequence.size(0), device=device), index.clamp_min(0)
            ]
            return torch.where(
                index[:, None].ge(0), selected, torch.zeros_like(selected)
            )

        try:
            widths = [int(batch[1].shape[1]) for batch in batches if batch[1].ndim == 2]
            width = max(widths) if widths else int(getattr(self.model, "maxlen", 0))
            if width < 1:
                raise ValueError("No observed input window")
            itemnum = int(module.itemnum)
            if itemnum < 2:
                raise ValueError("At least two real items required")
            histories = self.user_train.get("History", self.user_train)
            public = []
            for user in sorted(histories, key=str):
                h = [int(i) for i in histories[user]]
                if h and all(1 <= i <= itemnum for i in h):
                    public.append((str(user), h))
                if len(public) == 32:
                    break
            if not public:
                raise ValueError("No eligible public training histories")
            short = public[0][1][-min(5, width) :]
            longest = max(public, key=lambda record: len(record[1]))[1][-width:]
            panel = [[], short[-1:], short, longest, [short[-1]] * min(4, width)]
            sequence = torch.zeros(len(panel), width, dtype=torch.long, device=device)
            for row, history in enumerate(panel):
                if history:
                    sequence[row, -len(history) :] = torch.tensor(
                        history, device=device
                    )
            candidates = torch.arange(1, min(16, itemnum) + 1, device=device)
            audit["fixtures"] = {
                "selection": "First 32 valid training histories in sorted user-ID order; first and longest within that bounded panel",
                "public_user_ids": [u for u, _ in public],
                "raw_histories": [h for _, h in public],
                "histories": panel,
                "sequence": sequence.cpu().tolist(),
                "candidate_ids": candidates.cpu().tolist(),
                "width": width,
                "rows": len(panel),
                "maximum_window_filled": len(longest) == width,
                "candidate_purpose": "Execution fixtures only; not recommendation-quality evaluation",
            }
            reference = features(sequence)
            state = last(reference, sequence)
            compare("actual_module_features", module.log2feats(sequence), reference)
            compare(
                "production_readout_same_tensor",
                module._readout(reference, sequence),
                state,
            )
            expected = torch.tensor(
                [-1 if not h else width - 1 for h in panel], device=device
            )
            indices = torch.arange(width, device=device).expand_as(sequence)
            compare(
                "final_indices",
                indices.masked_fill(sequence.eq(0), -1).amax(1),
                expected,
            )
            compare("empty_features", reference[0], torch.zeros_like(reference[0]))
            compare(
                "padding_features",
                reference[sequence.eq(0)],
                torch.zeros_like(reference[sequence.eq(0)]),
            )
            compare("repeat_batch", features(sequence), reference)
            order = torch.tensor([3, 0, 4, 2, 1], device=device)
            compare(
                "permuted_batch", features(sequence[order])[order.argsort()], reference
            )
            compare(
                "duplicated_batch",
                features(sequence.repeat(2, 1)),
                reference.repeat(2, 1, 1),
            )
            compare(
                "partitioned_batch",
                torch.cat([features(x) for x in sequence.split(2)]),
                reference,
            )
            singletons = torch.cat([features(x) for x in sequence.split(1)])
            compare("singleton_batch", singletons, reference)
            compare(
                "repeat_singleton", features(sequence[2:3]), features(sequence[2:3])
            )
            for row, history in enumerate(panel):
                if history:
                    compact = torch.tensor([history], device=device)
                    compare(
                        f"changed_padding_row_{row}",
                        features(compact)[0],
                        reference[row, -len(history) :],
                    )
            ids = candidates[None, None, :].expand(sequence.size(0), width, -1)
            position_scores = self.model.predict_score(reference, ids)
            actual_position_scores = module.predict_score(reference, ids)
            compare("view_position_scores", position_scores, actual_position_scores)
            loop = torch.stack(
                [
                    torch.stack(
                        [
                            module.predict_score(reference[b, t], candidates)
                            for t in range(width)
                        ]
                    )
                    for b in range(sequence.size(0))
                ]
            )
            compare("position_score_loop", position_scores, loop)
            scores = module.predict_score(state, candidates)
            compare(
                "actual_prediction_entry",
                module.score_candidates(sequence, candidates),
                scores,
            )
            compare("view_final_scores", position_scores[:, -1], scores)

            # Intervention is after the bounded nonlinear readout, where the
            # supplied model scores by a dot product with learned item embeddings.
            h = state[2]
            matrix = module.item_emb(candidates).detach().clone()
            compare(
                "dot_product_equation", module.predict_score(h, candidates), matrix @ h
            )
            competitor = 1 + int(scores[2, 1:].argmax())
            gradient = matrix[0] - matrix[competitor]
            epsilon = 0.01 * max(float(h.norm()), 1e-8)
            audit["analytical_intervention"] = {
                "location": "Final recommendation feature after readout",
                "target_candidate_index": 0,
                "competitor_index": competitor,
                "competitor_rule": "Highest original score among candidate indices 1 onward; first index breaks ties",
                "epsilon": epsilon,
                "gradient_norm": float(gradient.norm()),
                "zero_pair_gradient": bool(gradient.norm() == 0),
                "interpretation": "Computational score dependence only; no quality claim",
            }
            directions = [("zero", torch.zeros_like(h))]
            if bool(gradient.norm() > 0):
                directions += [
                    ("positive", epsilon * gradient / gradient.norm()),
                    ("negative", -epsilon * gradient / gradient.norm()),
                ]
            else:
                unavailable(
                    "pair_gradient_intervention",
                    "Selected pair has zero feature gradient",
                )
            nonzero = matrix.norm(dim=1).gt(0).nonzero().flatten()
            if nonzero.numel():
                direction = matrix[int(nonzero[0])]
                directions.append(
                    ("score_positive_control", epsilon * direction / direction.norm())
                )
            else:
                unavailable(
                    "score_positive_control", "All selected scoring rows are zero"
                )
            for name, delta in directions:
                change = module.predict_score(
                    h + delta, candidates
                ) - module.predict_score(h, candidates)
                prediction = matrix @ delta
                compare(f"affine_scores_{name}", change, prediction)
                compare(
                    f"affine_margin_{name}",
                    change[0] - change[competitor],
                    gradient @ delta,
                )
                audit["analytical_intervention"][name] = {
                    "raw_score_change": change.cpu().tolist(),
                    "centered_score_change": (change - change.mean()).cpu().tolist(),
                    "margin_change": float(change[0] - change[competitor]),
                }
            audit["ranking"] = {
                "status": "skipped",
                "reason": "Pinned evaluator ranking/tie helper unavailable; raw scores are audited",
                "exact_tied_pairs": int(
                    torch.triu(
                        scores[:, :, None].eq(scores[:, None, :]), diagonal=1
                    ).sum()
                ),
            }
            unavailable(
                "float64_reference",
                "Supplementary backend comparison not executed; deployment dtype retained",
            )
            unavailable(
                "internal_recurrent_hooks",
                "No canonical row-aligned packed-state trace adapter supplied; public features and readout are compared",
            )

            category = getattr(
                self.model, "dataset_name", getattr(module, "dataset_name", None)
            )
            root = os.path.dirname(os.path.abspath(__file__))
            working = os.path.join(root, "working")
            allowed = {
                "Movies_and_TV",
                "Industrial_and_Scientific",
                "Electronics",
                "CDs_and_Vinyl",
            }
            if category not in allowed:
                unavailable(
                    "exported_predictor",
                    "ModelView exposes no valid category routing label",
                )
            else:
                checkpoint_path = os.path.join(working, f"{category}_model.pt")
                predictor_path = os.path.join(working, "predict.py")
                meta_path = os.path.join(working, f"{category}_meta.pkl")
                if not all(
                    os.path.isfile(p)
                    for p in (checkpoint_path, predictor_path, meta_path)
                ):
                    unavailable(
                        "exported_predictor",
                        "Required current-trial checkpoint, metadata or emitted predictor is absent",
                    )
                else:
                    digest = hashlib.sha256()
                    with open(checkpoint_path, "rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    audit["checkpoint_sha256"] = digest.hexdigest()
                    checkpoint = torch.load(
                        checkpoint_path, map_location=device, weights_only=True
                    )
                    saved = checkpoint["state_dict"]
                    current = module.state_dict()
                    if set(saved) != set(current) or any(
                        saved[k].dtype != current[k].dtype
                        or saved[k].shape != current[k].shape
                        or not torch.equal(saved[k], current[k])
                        for k in current
                    ):
                        raise ValueError(
                            "Current module differs from category checkpoint parameters/buffers"
                        )
                    # The exact tensor checks above establish this result;
                    # report their actual coverage rather than a sentinel pair.
                    checks.append(
                        {
                            "name": "checkpoint_parameters_buffers",
                            "status": "executed/pass",
                            "reference": "Current category checkpoint state_dict",
                            "dtype": "Per-tensor dtype checked exactly",
                            "elements": sum(
                                value.numel() for value in current.values()
                            ),
                            "tensors": len(current),
                            "violations": 0,
                            "max_abs_error": 0.0,
                            "max_relative_error": 0.0,
                            "error_location": None,
                            "comparison": "Exact keys, shapes, dtypes and torch.equal for every tensor",
                        }
                    )
                    spec = importlib.util.spec_from_file_location(
                        "current_trial_execution_predictor", predictor_path
                    )
                    exported = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(exported)
                    exported_class = exported._load_model_class()
                    restored = exported_class(**checkpoint["architecture_config"]).to(
                        device=device, dtype=next(module.parameters()).dtype
                    )
                    restored.load_state_dict(saved, strict=True)
                    restored.eval()
                    if any(
                        not torch.equal(restored.state_dict()[k], current[k])
                        for k in current
                    ):
                        raise ValueError(
                            "Export reconstruction changed parameters or buffers"
                        )
                    with open(meta_path, "rb") as handle:
                        metadata = pickle.load(handle)
                    if (
                        metadata["architecture_config"]
                        != checkpoint["architecture_config"]
                    ):
                        raise ValueError("Export metadata architecture mismatch")
                    # Exercise the emitted predict function's preprocessing and
                    # routing using the same checkpoint, without loading three
                    # unrelated category models into a second GPU cache.
                    exported._models = {category: restored}
                    exported._metas = {category: metadata}
                    exported._device = device
                    for row, history in enumerate(panel):
                        value = exported.predict(
                            0, history, candidates.cpu().tolist(), dataset_name=category
                        )
                        compare(
                            f"exported_predictor_row_{row}",
                            torch.tensor(value, dtype=scores.dtype),
                            scores[row],
                        )
                    del restored, exported
            required = [
                c
                for c in checks
                if not c["name"].startswith(
                    ("float64", "internal_recurrent", "pair_gradient", "score_positive")
                )
            ]
            audit["passed"] = bool(required) and all(
                c["status"] == "executed/pass" for c in required
            )
        except Exception as exc:
            unavailable(
                "audit_execution", f"{type(exc).__name__}: {exc}", status="error"
            )
            audit["passed"] = False
        audit["coverage"] = dict(Counter(c["status"] for c in checks))
        audit["relative_error_denominator"] = (
            "max(abs(reference), 1e-12); primary decision remains elementwise atol + rtol*abs(reference)"
        )
        return audit

    def _replay(self, histories, width, candidates):
        """Use ModelView's sequential scoring contract without changing recurrence."""
        if width <= 0 or not histories:
            raise ValueError("Replay requires positive width and at least one history")
        itemnum = int(self.model.model.itemnum)
        if len(candidates) != 2 or not all(1 <= int(x) <= itemnum for x in candidates):
            raise ValueError("Replay requires two real-item candidate IDs")
        if int(candidates[0]) == int(candidates[1]):
            raise ValueError("Replay candidates must be distinct")
        sequence = np.zeros((len(histories), width), dtype=np.int64)
        for row, history in enumerate(histories):
            if not 0 < len(history) <= width:
                raise ValueError(
                    "Replay history must be nonempty and fit the external width"
                )
            if not all(1 <= int(x) <= itemnum for x in history):
                raise ValueError("Replay history contains an invalid real-item ID")
            sequence[row, -len(history) :] = history
        features = self.model.log2feats(sequence)
        if features.ndim != 3 or tuple(features.shape[:2]) != sequence.shape:
            raise RuntimeError("ModelView.log2feats must return B x L x D features")
        # External histories remain LEFT padded. Never gather at count-1.
        positions = torch.arange(width, device=features.device).expand(
            len(histories), -1
        )
        occupied = torch.as_tensor(sequence != 0, device=features.device)
        last = positions.masked_fill(~occupied, -1).max(dim=1).values
        if (last < 0).any():
            raise RuntimeError("Empty diagnostic history")
        states = features[torch.arange(len(histories), device=features.device), last]
        # DEBUG: Score through the same B x L x D / B x L x 2 contract
        # used by the working margin probe. A singleton sequence dimension
        # triggers incompatible ModelView broadcasting; reshaping cannot fix it.
        score_features = features
        if width == 1:
            # DEBUG: Duplicate already-computed features for scoring only.
            # Do not append input items or change the GRU recurrence/state.
            score_features = features.expand(-1, 2, -1)
        items = torch.as_tensor(candidates, device=features.device, dtype=torch.long)
        items = items.view(1, 1, 2).expand(len(histories), score_features.size(1), 2)
        all_scores = self.model.predict_score(score_features, items)
        if tuple(all_scores.shape) != tuple(items.shape):
            raise RuntimeError(
                "ModelView.predict_score must return B x L x 2 scores; "
                f"received {tuple(all_scores.shape)} for "
                f"features {tuple(score_features.shape)} and items {tuple(items.shape)}"
            )
        # DEBUG: Gather the actual final nonpadding position after scoring,
        # preserving left-padding semantics and the actual model scorer.
        scores = all_scores[torch.arange(len(histories), device=features.device), last]
        # Preserve the scoring dtype, including exact ties and small margins.
        return states.detach().clone(), scores.detach().clone()

    def _history_panel(self, batches):
        rng = np.random.default_rng(42)
        counts = Counter()
        observations = {}
        paired = {"recent_nonfinal_minus_earlier": [], "final_minus_earlier": []}
        audit = {
            "model_class": type(self.model.model).__name__,
            "recurrent_module": type(getattr(self.model.model, "gru", None)).__name__,
            "initial_state_contract": (
                "Supplied GRU4Rec calls gru(emb) without hx on every invocation: "
                "implicit zero initial state, with no carried recurrent state. "
                "The diagnostic does not modify recurrence."
            ),
            "supplied_recurrence": (
                "The supplied baseline runs an ordinary GRU over leading zero embeddings; "
                "padding_idx=0 does not skip GRU updates."
            ),
            "repeated_call_max_abs_score_difference": None,
            "batch_permutation_max_abs_score_difference": None,
            "batched_vs_single_max_abs_score_difference": None,
            "unpadded_max_abs_score_difference": None,
            "unpadded_max_abs_state_difference": None,
            "shortened_history_padding_confounded": True,
            "exported_predictor_score_agreement": None,
            "exported_predictor_audit_reason": (
                "No exported predictor callable is supplied to DiagnosisProbe; "
                "training/export agreement cannot be measured here."
            ),
            "sampling": "First up to ten yielded batches; loader order is not controlled by this probe.",
            "perturbation_seed": 42,
            "random_control_draws": 5,
            "audit_observations": 0,
        }
        audit_done = False
        itemnum = int(self.model.model.itemnum)
        for _, sequence, positive, negative in batches:
            seq = torch.as_tensor(sequence).cpu().numpy()
            pos = torch.as_tensor(positive).cpu().numpy()
            neg = torch.as_tensor(negative).cpu().numpy()
            if (
                seq.ndim != 2
                or seq.shape[1] == 0
                or seq.shape != pos.shape
                or seq.shape != neg.shape
            ):
                raise ValueError(
                    "Diagnostic loader must yield matching nonempty B x L tensors"
                )
            width = seq.shape[1]
            for row in range(seq.shape[0]):
                counts["rows_seen"] += 1
                valid_positions = np.flatnonzero((seq[row] > 0) & (pos[row] > 0))
                if not len(valid_positions):
                    counts["skipped_no_valid_position"] += 1
                    continue
                t = int(valid_positions[-1])
                prefix = seq[row, : t + 1]
                history = prefix[prefix != 0].astype(np.int64).tolist()
                if any(x < 1 or x > itemnum for x in history):
                    counts["skipped_invalid_history"] += 1
                    continue
                candidates = [int(pos[row, t]), int(neg[row, t])]
                if not all(1 <= x <= itemnum for x in candidates):
                    counts["skipped_invalid_candidates"] += 1
                    continue
                if candidates[0] == candidates[1]:
                    counts["skipped_identical_candidates"] += 1
                    continue
                n = len(history)
                if n < 4:
                    counts["skipped_short_history"] += 1
                    continue
                counts["eligible_rows"] += 1
                k = min(3, (n - 1) // 3)
                indices = list(range(n))
                variants = []

                def add(name, removed):
                    retained = [i for i in indices if i not in removed]
                    variants.append((name, retained, [history[i] for i in retained]))

                add("original", set())
                add("composite_recent", set(range(n - k, n)))
                add("composite_old", set(range(k)))
                add("recent_nonfinal", set(range(n - 1 - k, n - 1)))
                # These removal regions are disjoint and both preserve the final item.
                for _draw in range(5):
                    removed = rng.choice(n - 1 - k, size=k, replace=False).tolist()
                    add("earlier_matched", set(removed))
                add("final_deleted", {n - 1})
                for _draw in range(5):
                    add("earlier_single", {int(rng.integers(n - 1))})
                add("final_only", set(range(n - 1)))
                seen_lengths = set()
                for requested in (2, 4, 8, 16, n):
                    length = min(requested, n)
                    if length in seen_lengths:
                        continue
                    seen_lengths.add(length)
                    # Name the actual retained length. A clipped full history
                    # must not be pooled with genuinely longer suffixes.
                    name = "suffix_full" if length == n else f"suffix_{length}"
                    add(name, set(range(n - length)))
                states, scores = self._replay(
                    [v[2] for v in variants], width, candidates
                )
                if not bool(
                    torch.isfinite(states).all() and torch.isfinite(scores).all()
                ):
                    counts["skipped_nonfinite_replay"] += 1
                    continue
                if not audit_done:
                    original = [history]
                    state1, score1 = self._replay(original, width, candidates)
                    _, score2 = self._replay(original, width, candidates)
                    unpad_state, unpad_score = self._replay(original, n, candidates)
                    _, reverse_scores = self._replay(
                        [v[2] for v in reversed(variants)], width, candidates
                    )
                    audit.update(
                        {
                            "repeated_call_max_abs_score_difference": float(
                                (score1 - score2).abs().max().item()
                            ),
                            "batch_permutation_max_abs_score_difference": float(
                                (scores - reverse_scores.flip(0)).abs().max().item()
                            ),
                            "batched_vs_single_max_abs_score_difference": float(
                                (scores[:1] - score1).abs().max().item()
                            ),
                            "unpadded_max_abs_score_difference": float(
                                (score1 - unpad_score).abs().max().item()
                            ),
                            "unpadded_max_abs_state_difference": float(
                                (state1 - unpad_state).abs().max().item()
                            ),
                            "audit_observations": 1,
                            "original_audit_history_length": n,
                            "original_external_width": width,
                        }
                    )
                    # Even observed invariance at one prefix does not establish padding-free recurrence.
                    audit_done = True
                wins = (scores[:, 0] > scores[:, 1]).float()
                wins += 0.5 * (scores[:, 0] == scores[:, 1]).float()
                original_win = float(wins[0].item())
                grouped = {}
                original_norm = states[0].norm()
                for index, (name, retained, retained_history) in enumerate(variants):
                    group = grouped.setdefault(
                        name,
                        {
                            "wins": [],
                            "distances": [],
                            "zero": 0,
                            "final_retained": [],
                            "final_id_changed": [],
                        },
                    )
                    group["wins"].append(float(wins[index].item()))
                    group["final_retained"].append(float(n - 1 in retained))
                    group["final_id_changed"].append(
                        float(retained_history[-1] != history[-1])
                    )
                    if (
                        float(original_norm.item()) == 0.0
                        or float(states[index].norm().item()) == 0.0
                    ):
                        group["zero"] += 1
                    else:
                        # Normalize nonzero states explicitly: cosine_similarity's
                        # default epsilon can distort very small, nonzero vectors.
                        base = states[0].double()
                        current = states[index].double()
                        cosine = torch.dot(
                            base / torch.linalg.vector_norm(base),
                            current / torch.linalg.vector_norm(current),
                        ).clamp(-1.0, 1.0)
                        distance = 1.0 - float(cosine.item())
                        group["distances"].append(float(distance))
                per_row = {}
                for name, group in grouped.items():
                    record = observations.setdefault(
                        name,
                        {
                            "wins": [],
                            "deltas": [],
                            "cosine": [],
                            "zero_norm_replays": 0,
                            "final_retained": [],
                            "final_id_changed": [],
                        },
                    )
                    win = float(np.mean(group["wins"]))
                    per_row[name] = win
                    record["wins"].append(win)
                    record["deltas"].append(win - original_win)
                    if group["distances"]:
                        record["cosine"].append(float(np.mean(group["distances"])))
                    record["zero_norm_replays"] += group["zero"]
                    record["final_retained"].append(
                        float(np.mean(group["final_retained"]))
                    )
                    record["final_id_changed"].append(
                        float(np.mean(group["final_id_changed"]))
                    )
                paired["recent_nonfinal_minus_earlier"].append(
                    per_row["recent_nonfinal"] - per_row["earlier_matched"]
                )
                paired["final_minus_earlier"].append(
                    per_row["final_deleted"] - per_row["earlier_single"]
                )
                counts["analyzed_rows"] += 1

        def mean(values):
            return float(np.mean(values)) if values else None

        summaries = {}
        for name, record in observations.items():
            summaries[name] = {
                "observations": len(record["wins"]),
                "mean_pairwise_win": mean(record["wins"]),
                "mean_win_minus_original": mean(record["deltas"]),
                "mean_one_minus_state_cosine": mean(record["cosine"]),
                "cosine_valid_observations": len(record["cosine"]),
                "zero_norm_replays": record["zero_norm_replays"],
                "final_position_retained_fraction": mean(record["final_retained"]),
                "final_item_id_changed_fraction": mean(record["final_id_changed"]),
            }
        audit["recency_conclusions_withheld"] = True
        audit["padding_confound_reason"] = (
            "Supplied ordinary leading-padding GRU updates make deletion change both real context "
            "and the number of initial padding updates. One unpadded replay is an execution check, "
            "not proof of padding invariance. No recurrence correction or retraining is performed."
        )
        return {
            "counts": dict(counts),
            "conditions": summaries,
            "paired_differences": {
                name: {"mean": mean(values), "observations": len(values)}
                for name, values in paired.items()
            },
            "execution_audit": audit,
            "user_bootstrap_macro_95_intervals": None,
            "uncertainty_unavailable_reason": (
                "Loader supplies no reliable user IDs, and this probe has no explicitly user-indexed "
                "observations from all four categories. Rows are not bootstrapped as independent users. "
                "Aggregates are descriptive and may overweight frequent users."
            ),
        }

    def run_full_diagnosis(self, train_loader):
        import os
        import tempfile
        import time

        module = self.model.model
        modes = [(child, child.training) for child in module.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.random.get_rng_state()
        device = next(module.parameters()).device
        # torch.manual_seed and the imported training module can seed every
        # visible CUDA generator. Restore every generator they can affect.
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            module.eval()
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            with torch.inference_mode():
                batches = []
                iterator = iter(train_loader)
                for _ in range(10):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                        raise ValueError(
                            "Expected (None, sequence, positive, negative)"
                        )
                    batches.append(
                        (None,)
                        + tuple(
                            torch.as_tensor(x).detach().cpu().clone()
                            for x in batch[1:4]
                        )
                    )
                metrics = {}
                try:
                    metrics["embedding_collapse_score"] = (
                        self.analyze_embeddings_collapse_cosine(2048)
                    )
                except Exception as exc:
                    metrics["embedding_collapse_score"] = None
                    metrics["embedding_unavailable_reason"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                try:
                    margin, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(batches)
                    )
                    metrics.update(
                        {
                            "pos_neg_margin_mean": margin["mean"],
                            "pos_neg_margin_p10": margin["p10"],
                            "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                            "pos_neg_small_margin_rate": margin["small_margin_rate"],
                            "pos_neg_margin_category_breakdown": breakdown,
                            "margin_valid_positions": breakdown["valid_positions"],
                        }
                    )
                except Exception as exc:
                    for name in (
                        "pos_neg_margin_mean",
                        "pos_neg_margin_p10",
                        "pos_neg_neg_beats_pos_rate",
                        "pos_neg_small_margin_rate",
                        "margin_valid_positions",
                    ):
                        metrics[name] = None
                    metrics["margin_unavailable_reason"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                audit = self._execution_audit(batches)
                metrics["execution_audit"] = audit
                metrics["history_sensitivity"] = None
                if audit["passed"]:
                    try:
                        # Bound work without changing perturbations on retained rows.
                        bounded = [
                            (None,) + tuple(x[:8] for x in b[1:]) for b in batches[:1]
                        ]
                        panel = self._history_panel(bounded)
                        # The legacy panel's hard-coded recurrence claims are
                        # replaced by this audit's measured coverage. Changed
                        # padding agreement is empirical, not a universal proof.
                        panel["execution_audit"] = audit
                        panel["execution_audit"][
                            "shortened_history_padding_confounded"
                        ] = None
                        panel["execution_audit"]["recency_conclusions_withheld"] = True
                        panel["execution_audit"][
                            "padding_confound_reason"
                        ] = "Fixture padding comparisons passed; general padding semantics and temporal causality are not established by finite fixtures"
                        panel["selection"] = (
                            "First eight rows of first cached batch; no user-independent uncertainty claims"
                        )
                        metrics["history_sensitivity"] = panel
                    except Exception as exc:
                        metrics["history_unavailable_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                else:
                    metrics["history_unavailable_reason"] = (
                        "Required feature/readout/scoring/export checks failed, were skipped, or errored; inspect execution_audit coverage"
                    )
                definitions = {
                    "embedding_collapse_score": "Mean off-diagonal real-item embedding cosine mapped by (cosine+1)/2, at most 2048 seed-42 samples. Null means unavailable. 0.5 means mean cosine zero; this is not recommendation diversity.",
                    "pos_neg_margin_mean": "Mean raw positive-minus-negative logit over valid distinct sampled training pairs; score-scale dependent.",
                    "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of the same valid training-pair margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of those valid pairs with negative score strictly greater than positive score.",
                    "pos_neg_small_margin_rate": "Fraction of those valid pairs with margin below 0.5; score-scale dependent.",
                    "margin_valid_positions": "Exact valid training-position denominator; null indicates probe error, zero indicates no valid pairs.",
                    "pos_neg_margin_category_breakdown": "Valid sampled-position counts; category lists are empty because item-category metadata is unavailable.",
                    "execution_audit": "Bounded public-history execution audit. Each comparison records executed/pass, executed/fail, skipped or error; empty comparisons cannot pass. Primary threshold is elementwise abs(actual-reference)<=1e-5+1e-4*abs(reference). Shapes and finiteness are checked first. Relative error denominator is max(abs(reference),1e-12). Singleton, repeated, permuted, duplicated and partitioned executions retain row identities and candidate order. Changed-padding comparisons are separate. Same-tensor final readout uses maximum nonzero index, with zero empty features. Required export coverage checks the current category checkpoint against the actual trained module and exercises the emitted predictor's routing/preprocessing with a strict reconstruction of that same checkpoint; no independently trained model or fusion is introduced. Packed-state hooks and float64 references are explicitly skipped, so audit success is scoped to public features/readout/scoring/export coverage. Analytical interventions act after the nonlinear readout and check the supplied dot-product equation on fixtures; zero pair gradients are reported. Computational dependence, centered score changes and margin changes do not establish ranking utility. Rank/HR parity is unavailable without the pinned tie rule. Fixture coverage cannot establish general correctness or validation improvement.",
                    "history_sensitivity": "Only runs after required audit coverage passes. First eight anonymous rows of the first cached training batch; unchanged prespecified paired deletion/suffix conditions and five local seed-42 random controls. Pairwise wins use 1/0/0.5, not HR@10. Report intervention-minus-original and paired-control differences. Cosine movement alone is representation sensitivity. Finite padding fixtures do not establish causal recency. No user bootstrap or four-category intervals are inferred from loader rows. Null means unavailable, not zero effect.",
                    "diagnostic_inference_seconds": "Synchronized elapsed wall seconds including fixture replay, export reconstruction and artifact writing; charged to search diagnostics.",
                    "simulator_recommendations": "The host's separate full-catalog unseen-item recommendations for five real training users per category remain outside this diagnostic API and are unchanged.",
                }
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                metrics["diagnostic_inference_seconds"] = time.perf_counter() - started
                definitions["diagnostic_inference_seconds"] = (
                    "Synchronized elapsed wall seconds for diagnostic computation, "
                    "including export reconstruction; excludes final JSON artifact "
                    "serialization and cleanup. This is not total billed GPU time."
                )
                for check in audit["checks"]:
                    if check["status"] in {"executed/fail", "error"}:
                        print(
                            f"Diagnosis audit {check['name']}: {check['status']}; "
                            f"{check.get('reason', 'elementwise tolerance exceeded')}",
                            flush=True,
                        )
                result = {"metrics": metrics, "metric_definitions": definitions}
                directory = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
                )
                temporary = None
                try:
                    os.makedirs(directory, exist_ok=True)
                    category = getattr(
                        self.model,
                        "dataset_name",
                        getattr(module, "dataset_name", None),
                    )
                    allowed = {
                        "Movies_and_TV",
                        "Industrial_and_Scientific",
                        "Electronics",
                        "CDs_and_Vinyl",
                    }
                    label = category if category in allowed else "unlabeled"
                    path = os.path.join(directory, f"{label}_execution_audit.json")
                    metrics["artifact_path"] = path
                    with tempfile.NamedTemporaryFile(
                        mode="w", dir=directory, suffix=".json", delete=False
                    ) as handle:
                        temporary = handle.name
                        json.dump(result, handle, allow_nan=False)
                    os.replace(temporary, path)
                except Exception as exc:
                    metrics["artifact_unavailable_reason"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(
                        f"Diagnosis artifact unavailable: {metrics['artifact_unavailable_reason']}",
                        flush=True,
                    )
                finally:
                    if temporary and os.path.isfile(temporary):
                        os.unlink(temporary)
                # Returned and persisted results share the same measured runtime.
                # Static review: checked JSON scalar types, exact checkpoint
                # coverage, replay dtype, RNG restoration and artifact errors.
                # No training or validation improvement has been measured.
                return result
        finally:
            for child, training in modes:
                child.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    ### <<< Self_EvolveRec-BLOCK-END
