import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter


### >>> Self_EvolveRec-BLOCK-START: Evidence-linked interpretation of paired BERT4Rec diagnostics
def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
Analyze these measured BERT4Rec diagnostics and their definitions:
{json.dumps(raw_diagnosis, indent=2, allow_nan=False)}
Return JSON with status, core_findings, key_implications, and evidence.
Use at most three findings and three implications. Cite metric names, values,
counts, availability flags, and uncertainty intervals for each substantive claim.
Distinguish serving defects, measured history sensitivity, missing information,
and hypotheses requiring independent validation. If serving parity or sampler
checks fail or are unavailable, withhold substantive sensitivity conclusions.
Null values mean unavailable, never zero or successful checks. Missing category
coverage prevents a four-category macro conclusion. These training-prefix
candidate results are not host validation HR@10. Stability, sensitivity, and
embedding alignment do not establish recommendation quality or validation gains.
A mapped embedding cosine of 0.5 means approximately zero mean cosine, not proven
healthy representation capacity. Do not infer intent, simulator traits, ratings,
review content, item categories, prices, stores, or ownership. Unknown identity
and abstention are not negative labels. Intervals cover sampled-user uncertainty
only, not training seeds or candidate sampling. Repeated validation selection is
optimistic. No recommendation-score fusion or additional predictors are allowed.
Use INSUFFICIENT_EVIDENCE when the available measurements do not support a status.
"""


### <<< Self_EvolveRec-BLOCK-END


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the host view and explicitly unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.view = view
        self.net = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.category = getattr(view, "dataset_name", None)
        self._contract = {}
        self._embedding_count = 0

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this matched benchmark.
        return []

    # Static review: checked scoring shapes, checkpoint identity, category routing,
    # parity failure paths, coverage gates, and paired bootstrap reuse.
    # No training or runtime tests were executed in this code-only review.
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Seeded real-item embedding alignment with explicit missingness
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.view.item_emb.weight.detach()[1 : self.net.itemnum + 1]
            k = min(2048, sample_size, emb.size(0))
            self._embedding_count = k
            if k < 2:
                return None
            generator = torch.Generator(device="cpu").manual_seed(seed)
            indices = torch.randperm(emb.size(0), generator=generator)[:k].to(
                emb.device
            )
            x = F.normalize(emb.index_select(0, indices).float(), dim=-1)
            if not torch.isfinite(x).all():
                return None
            mean_cos = (x.sum(0).square().sum() - x.square().sum()) / (k * (k - 1))
            return float((mean_cos.clamp(-1, 1).item() + 1.0) / 2.0)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Check supplied all-MASK sampler before in-sample margin measurement
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        from itertools import islice

        margins = []
        checks = {
            "batches": 0,
            "valid_batches": 0,
            "invalid_batches": 0,
            "supervised_positions": 0,
            "errors": [],
        }
        reserved_ok = (
            self.net.mask_token == self.net.itemnum + 1
            and self.view.item_emb.num_embeddings == self.net.itemnum + 2
            and self.view.item_emb.padding_idx == 0
        )
        checks["reserved_tokens_valid"] = bool(reserved_ok)
        checks["policy"] = (
            "Supplied terminal/interior sampler: all selected targets replaced by MASK."
        )
        previous_state = self.net.training
        try:
            self.net.eval()
            for batch in islice(dataloader, min(10, num_batches)):
                checks["batches"] += 1
                try:
                    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                        raise ValueError(
                            "expected (None, sequence, positive, negative)"
                        )
                    seq, pos, neg = [
                        torch.as_tensor(x, device=self.device, dtype=torch.long)
                        for x in batch[1:4]
                    ]
                    if (
                        seq.ndim != 2
                        or seq.shape != pos.shape
                        or seq.shape != neg.shape
                    ):
                        raise ValueError("expected matching B x L tensors")
                    if seq.shape[1] != self.net.maxlen or seq.shape[0] == 0:
                        raise ValueError("unexpected sequence length or empty batch")
                    valid = pos.ne(0)
                    ok = (
                        reserved_ok
                        and bool(((seq >= 0) & (seq <= self.net.mask_token)).all())
                        and bool((valid.sum(1) > 0).all())
                        and bool((seq[valid] == self.net.mask_token).all())
                        and bool(
                            ((pos[valid] >= 1) & (pos[valid] <= self.net.itemnum)).all()
                        )
                        and bool(
                            ((neg[valid] >= 1) & (neg[valid] <= self.net.itemnum)).all()
                        )
                        and bool((pos[valid] != neg[valid]).all())
                        and bool((neg[~valid] == 0).all())
                    )
                    if not ok:
                        raise ValueError(
                            "reserved-token, all-MASK supervision, or real-item contract failed"
                        )
                    features = self.view.log2feats(seq.detach().cpu().numpy())
                    logits = self.view.predict_score(
                        features, torch.stack((pos, neg), -1)
                    )
                    margin = (logits[..., 0] - logits[..., 1])[valid].float()
                    if not torch.isfinite(margin).all():
                        raise ValueError("nonfinite supervised logits")
                    margins.append(margin.detach().cpu())
                    checks["valid_batches"] += 1
                    checks["supervised_positions"] += int(valid.sum().item())
                except (ValueError, RuntimeError, TypeError, IndexError) as exc:
                    checks["invalid_batches"] += 1
                    checks["errors"].append(str(exc))
        finally:
            self.net.train(previous_state)
        checks["passed"] = bool(
            reserved_ok and checks["valid_batches"] and not checks["invalid_batches"]
        )
        checks["provenance_limit"] = (
            "Checks the supplied all-MASK policy; loader tensors alone cannot establish "
            "prefix provenance or verify uniform negative sampling."
        )
        self._contract = checks
        result = dict.fromkeys(
            ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate")
        )
        result["count"] = 0
        low = wrong = 0
        if margins:
            values = torch.cat(margins)
            result.update(
                count=int(values.numel()),
                mean=float(values.mean()),
                p10=float(
                    values.kthvalue(max(1, int(np.ceil(0.1 * values.numel())))).values
                ),
                neg_beats_pos_rate=float((values < 0).float().mean()),
                small_margin_rate=float((values < eps).float().mean()),
            )
            low, wrong = int((values < eps).sum()), int((values < 0).sum())
        breakdown = {
            "available": False,
            "reason": "Item categories are unavailable in this benchmark.",
            "eps": float(eps),
            "low_margin_total_positions": low,
            "wrong_rank_total_positions": wrong,
            "low_margin_top_categories": {"pos_item": [], "neg_item": []},
            "wrong_rank_top_categories": {"pos_item": [], "neg_item": []},
        }
        return result, breakdown

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Shape-safe paired sensitivity with checkpoint and coverage verification
    def _histories(self):
        if "History" in self.user_train:
            return self.user_train["History"]
        return self.user_train

    def _serving_tensors(self, history):
        length = self.net.maxlen
        raw = torch.zeros((1, length), dtype=torch.long, device=self.device)
        retained = list(history[-length:])
        if retained:
            raw[0, -len(retained) :] = torch.tensor(retained, device=self.device)
        serving = torch.zeros_like(raw)
        serving[:, :-1] = raw[:, 1:]
        serving[:, -1] = self.net.mask_token
        positions = torch.arange(1, length + 1, device=self.device)[None, :]
        positions = positions * serving.ne(0)
        return raw, serving, positions

    def _explicit_forward(self, sequence, positions):
        # This is the supplied BERT4Rec encoder, with fixed absolute positions.
        # Interior PAD holes are excluded from attention keys in every layer.
        padding = sequence.eq(0)
        if bool(padding.all(1).any()):
            raise ValueError("diagnostic inputs must contain an attention-visible MASK")
        x = self.net.item_emb(sequence) * (self.net.hidden_units**0.5)
        x = self.net.dropout(x + self.net.pos_emb(positions))
        return self.net.norm(self.net.encoder(x, src_key_padding_mask=padding))

    @staticmethod
    def _ranking(scores, candidates):
        scores = torch.as_tensor(scores).detach().float().cpu().reshape(-1)
        candidates = torch.as_tensor(candidates).detach().long().cpu().reshape(-1)
        if scores.numel() != candidates.numel() or not torch.isfinite(scores).all():
            raise ValueError("candidate scores must have matching length and be finite")
        by_id = torch.argsort(candidates, stable=True)
        order = by_id[torch.argsort(scores[by_id], descending=True, stable=True)]
        return candidates[order].tolist()

    def _feature_scores(self, feature, candidates):
        # Preserve the documented B x L x C scoring contract even for one MASK.
        feature = feature.reshape(1, 1, -1)
        items = candidates.reshape(1, 1, -1)
        scores = self.view.predict_score(feature, items)
        if scores.shape != items.shape or not bool(torch.isfinite(scores).all()):
            raise ValueError("ModelView must return finite B x L x C logits")
        return scores.reshape(-1)

    def _score(self, sequence, positions, candidates):
        features = self._explicit_forward(sequence, positions)
        # Read the actual appended MASK slot, including after deletion holes.
        if int(sequence[0, -1]) != self.net.mask_token:
            raise ValueError("serving readout must contain the reserved MASK")
        return self._feature_scores(features[0, -1], candidates)

    def _examples(self):
        rng = random.Random(42)
        candidate_rng = np.random.default_rng(42)
        histories = self._histories()
        users = sorted(histories, key=lambda user: str(user))
        selected = rng.sample(users, min(64, len(users)))
        counts = {
            "training_users": len(users),
            "selected_users": len(selected),
            "invalid_identity": 0,
            "no_eligible_endpoint": 0,
            "repeated_target_endpoints_excluded": 0,
            "insufficient_candidate_pool": 0,
            "insufficient_visible_history": 0,
        }
        examples = []
        for user in selected:
            try:
                history = [int(item) for item in histories[user]]
            except (TypeError, ValueError):
                counts["invalid_identity"] += 1
                continue
            if any(item < 1 or item > self.net.itemnum for item in history):
                counts["invalid_identity"] += 1
                continue
            endpoints, seen = [], set()
            for endpoint, target in enumerate(history):
                if endpoint >= 4:
                    if target not in seen:
                        endpoints.append(endpoint)
                    else:
                        counts["repeated_target_endpoints_excluded"] += 1
                seen.add(target)
            if not endpoints:
                counts["no_eligible_endpoint"] += 1
                continue
            endpoint = rng.choice(endpoints)
            prefix, target = history[:endpoint], history[endpoint]
            excluded = set(prefix) | {target}
            pool = np.setdiff1d(
                np.arange(1, self.net.itemnum + 1, dtype=np.int64),
                np.fromiter(excluded, dtype=np.int64),
                assume_unique=False,
            )
            if pool.size < 99:
                counts["insufficient_candidate_pool"] += 1
                continue
            candidates = [target] + candidate_rng.choice(
                pool, 99, replace=False
            ).tolist()
            examples.append((str(user), prefix, target, candidates))
        counts["candidate_examples"] = len(examples)
        return examples, counts

    def _exported_predictor(self):
        import importlib.util
        import inspect
        import os

        source_dir = os.path.dirname(os.path.abspath(inspect.getfile(type(self.net))))
        path = getattr(self.view, "predictor_path", None)
        if path is None:
            path = os.path.join(source_dir, "working", "predict.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Current trial's exported predict.py is not available"
            )
        spec = importlib.util.spec_from_file_location(
            "_bert4rec_diagnostic_export", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not all(
            hasattr(module, name)
            for name in ("predict", "_models", "_metas", "_device")
        ):
            raise ValueError(
                "exported predictor does not expose the supplied baseline interface"
            )
        required_routes = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        route = self.category
        if route not in required_routes:
            raise ValueError("ModelView.dataset_name must identify the actual category")
        if not callable(getattr(module, "_ensure_loaded", None)):
            raise ValueError("exported predictor has no checkpoint loading entry point")
        # Exercise the real writer-generated checkpoint loader for this route.
        # This reloads the same fitted checkpoint; it never fits or fuses models.
        module.DATASETS = [route]
        module._ensure_loaded()
        if set(module._models) != {route} or route not in module._metas:
            raise ValueError("exported loader must load exactly the requested category")
        exported = module._models[route].to(self.device)
        module._device = self.device
        exported.eval()
        if int(module._metas[route]["maxlen"]) != self.net.maxlen:
            raise ValueError("exported metadata disagrees with the diagnostic model")
        actual_state = self.net.state_dict()
        loaded_state = exported.state_dict()
        if actual_state.keys() != loaded_state.keys():
            raise ValueError(
                "checkpoint architecture differs from the diagnostic model"
            )
        for name, actual in actual_state.items():
            loaded = loaded_state[name]
            if (
                actual.shape != loaded.shape
                or actual.dtype != loaded.dtype
                or not torch.equal(actual, loaded)
            ):
                raise ValueError(
                    f"checkpoint differs at {name}; diagnose the selected checkpoint"
                )
        return module, route

    def _parity(self, examples):
        result = {
            "passed": False,
            "available": False,
            "atol": 1e-5,
            "rtol": 1e-4,
            "max_absolute_score_error": None,
            "ordered_ranking_agreement": None,
            "top10_agreement": None,
            "cases": 0,
            "coverage": {
                "short": False,
                "full_length": False,
                "truncated": False,
                "batched_vs_individual": False,
            },
            "checkpoint_serialization_verified": False,
            "weight_identity": "Unverified until the exported loader reloads the current category checkpoint and every state tensor matches exactly.",
            "external_label_invariance_tested": False,
        }
        try:
            if not examples:
                raise ValueError("no eligible training-prefix examples")
            module, route = self._exported_predictor()
            result["available"] = True
            result["checkpoint_serialization_verified"] = True
            result["weight_identity"] = (
                "All state tensors from the exported category checkpoint exactly "
                "match the actual diagnostic model, including dtype and shape."
            )
            cases = [(prefix, candidates) for _, prefix, _, candidates in examples]
            # Add a short case, then real full/truncated training-prefix cases.
            cases.append((examples[0][1][:2], examples[0][3]))
            for required in (self.net.maxlen - 1, self.net.maxlen + 1):
                for history in self._histories().values():
                    if len(history) >= required and required > 0:
                        prefix = [int(item) for item in history[:required]]
                        if all(1 <= item <= self.net.itemnum for item in prefix):
                            cases.append((prefix, examples[0][3]))
                            break
            errors, ordered, overlaps, passed = [], [], [], True
            paired_inputs, paired_positions, paired_scores = [], [], []
            for history, ids in cases:
                raw, sequence, positions = self._serving_tensors(history)
                candidates = torch.tensor(ids, dtype=torch.long, device=self.device)
                direct_features = self._explicit_forward(sequence, positions)
                view_features = self.view.log2feats(sequence.detach().cpu().numpy())
                if direct_features.shape != view_features.shape:
                    raise ValueError("ModelView returned an unexpected feature shape")
                visible = sequence.ne(0)
                passed &= bool(
                    torch.allclose(
                        direct_features[visible],
                        view_features[visible],
                        atol=1e-5,
                        rtol=1e-4,
                    )
                )
                # Compare the actual ModelView readout scores as well as features.
                direct = self._feature_scores(direct_features[0, -1], candidates)
                view_scores = self._feature_scores(view_features[0, -1], candidates)
                internal = self.net.score_candidates(raw, candidates).reshape(-1)
                external = torch.tensor(
                    module.predict(0, history, ids, dataset_name=route),
                    dtype=direct.dtype,
                    device=self.device,
                ).reshape(-1)
                reference_rank = self._ranking(direct, candidates)
                for other in (view_scores, internal, external):
                    finite = bool(torch.isfinite(other).all())
                    passed &= finite and bool(
                        torch.allclose(direct, other, atol=1e-5, rtol=1e-4)
                    )
                    if not finite:
                        raise ValueError("nonfinite parity scores")
                    errors.append(float((direct - other).abs().max()))
                    rank = self._ranking(other, candidates)
                    ordered.append(float(rank == reference_rank))
                    overlaps.append(
                        len(set(rank[:10]) & set(reference_rank[:10])) / 10.0
                    )
                result["coverage"]["short"] |= len(history) < self.net.maxlen - 1
                result["coverage"]["full_length"] |= len(history) == self.net.maxlen - 1
                result["coverage"]["truncated"] |= len(history) > self.net.maxlen - 1
                if len(paired_inputs) < 8:
                    paired_inputs.append(sequence)
                    paired_positions.append(positions)
                    paired_scores.append((direct, candidates))
            batch_features = self._explicit_forward(
                torch.cat(paired_inputs), torch.cat(paired_positions)
            )
            for index, (single, candidates) in enumerate(paired_scores):
                batched = self._feature_scores(batch_features[index, -1], candidates)
                if not torch.isfinite(batched).all():
                    raise ValueError("nonfinite batched parity scores")
                passed &= bool(torch.allclose(single, batched, atol=1e-5, rtol=1e-4))
                errors.append(float((single - batched).abs().max()))
                single_rank = self._ranking(single, candidates)
                batch_rank = self._ranking(batched, candidates)
                ordered.append(float(single_rank == batch_rank))
                overlaps.append(
                    len(set(single_rank[:10]) & set(batch_rank[:10])) / 10.0
                )
            result["coverage"]["batched_vs_individual"] = len(paired_inputs) > 1
            coverage_complete = all(result["coverage"].values())
            result.update(
                passed=bool(
                    passed
                    and coverage_complete
                    and result["checkpoint_serialization_verified"]
                    and all(value == 1.0 for value in ordered)
                ),
                coverage_complete=coverage_complete,
                cases=len(cases),
                max_absolute_score_error=max(errors),
                ordered_ranking_agreement=float(np.mean(ordered)),
                top10_agreement=float(np.mean(overlaps)),
            )
            if not coverage_complete:
                result["error"] = (
                    "Required parity coverage is incomplete; sensitivity is withheld."
                )
            elif not result["passed"]:
                result["error"] = (
                    "Feature, score, or ordered-ranking parity failed at fixed tolerances."
                )
        except Exception as exc:
            result["passed"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # Release the checkpoint reload after verification; it is never fused.
            if "module" in locals():
                module._models.clear()
        return result

    def _paired_sensitivity(self, examples, counts):
        rows = []
        for user, prefix, target, ids in examples:
            _, original, positions = self._serving_tensors(prefix)
            visible = torch.nonzero(original[0, :-1].ne(0), as_tuple=False).flatten()
            k = min(2, (visible.numel() - 1) // 2)
            if k < 1:
                counts["insufficient_visible_history"] += 1
                continue
            recent, older, swapped = (
                original.clone(),
                original.clone(),
                original.clone(),
            )
            recent[0, visible[-k:]] = 0
            older[0, visible[:k]] = 0
            left, right = int(visible[-2]), int(visible[-1])
            swapped[0, left], swapped[0, right] = original[0, right], original[0, left]
            candidates = torch.tensor(ids, dtype=torch.long, device=self.device)
            measurements = {}
            for name, sequence in (
                ("original", original),
                ("recent_deleted", recent),
                ("older_deleted", older),
                ("last_two_swapped", swapped),
            ):
                scores = self._score(sequence, positions, candidates)
                ranking = self._ranking(scores, candidates)
                rank = ranking.index(target) + 1
                measurements[name] = {
                    "target_score": float(scores[0]),
                    "target_rank": float(rank),
                    "reciprocal_rank": 1.0 / rank,
                    "diagnostic_hit10": float(rank <= 10),
                    "ranking": ranking,
                }
            baseline = measurements["original"]
            row = {
                "user": user,
                "visible_history": int(visible.numel()),
                "deleted_per_variant": k,
            }
            for name, values in measurements.items():
                if name == "original":
                    row[name] = {
                        key: value for key, value in values.items() if key != "ranking"
                    }
                    continue
                row[name] = {
                    key + "_delta": values[key] - baseline[key]
                    for key in (
                        "target_score",
                        "target_rank",
                        "reciprocal_rank",
                        "diagnostic_hit10",
                    )
                }
                row[name].update(
                    top10_overlap=len(
                        set(values["ranking"][:10]) & set(baseline["ranking"][:10])
                    )
                    / 10.0,
                    ordered_ranking_agreement=float(
                        values["ranking"] == baseline["ranking"]
                    ),
                )
            rows.append(row)
        counts["paired_users"] = len(rows)
        rng = np.random.default_rng(42)
        # Reuse paired-user resamples across variants for comparable intervals.
        indices = rng.integers(0, len(rows), size=(500, len(rows))) if rows else None
        summary = {}
        for variant in ("recent_deleted", "older_deleted", "last_two_swapped"):
            keys = (
                "target_score_delta",
                "target_rank_delta",
                "reciprocal_rank_delta",
                "diagnostic_hit10_delta",
                "top10_overlap",
                "ordered_ranking_agreement",
            )
            if not rows:
                summary[variant] = {
                    key: {"mean": None, "ci95": None, "users": 0} for key in keys
                }
                continue
            values = np.asarray(
                [[row[variant][key] for key in keys] for row in rows], dtype=np.float64
            )
            boot = values[indices].mean(axis=1)
            summary[variant] = {
                key: {
                    "mean": float(values[:, index].mean()),
                    "ci95": np.percentile(boot[:, index], [2.5, 97.5]).tolist(),
                    "users": len(rows),
                }
                for index, key in enumerate(keys)
            }
        return {
            "per_category": [
                {
                    "dataset_name": self.category,
                    "counts": counts,
                    "summary": summary,
                    "paired_user_outcomes": rows,
                }
            ],
            "four_category_macro": None,
            "four_category_macro_available": False,
            "macro_unavailable_reason": "This ModelView contains one category; four category results are required.",
        }

    def run_full_diagnosis(self, train_loader):
        import time

        previous_state = self.net.training
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        metrics = {"status": "diagnostic_failed", "history_sensitivity": None}
        definitions = {
            "embedding_collapse_score": "(Mean ordered off-diagonal cosine + 1)/2 over at most 2048 real item embeddings sampled with seed 42. PAD/MASK excluded. 0.5 means approximately zero mean cosine; capacity or quality is not established. Null if fewer than two or nonfinite observations.",
            "embedding_sample_count": "Number of sampled real item embeddings, not number of pairs.",
            "pos_neg_margin_mean": "In-sample mean positive-minus-negative logit over verified masked supervised positions. Higher means stronger sampled-pair separation.",
            "pos_neg_margin_p10": "In-sample nearest-rank 10th percentile of those margins, in logit units.",
            "pos_neg_neg_beats_pos_rate": "Fraction of verified supervised positions with negative-minus-positive logit > 0; lower is better.",
            "pos_neg_small_margin_rate": "Fraction of verified supervised positions with positive-minus-negative logit < 0.5.",
            "pos_neg_margin_count": "Number of verified supervised positions in at most ten loader batches; zero observations yield null statistics.",
            "pos_neg_margin_category_breakdown": "Unavailable: item categories are not supplied. Category lists remain empty.",
            "sampler_contract": "Checks PAD=0, MASK=itemnum+1, real supervised IDs, and the supplied all-MASK corruption policy. Failure blocks sensitivity. Does not establish sampler provenance from tensors alone.",
            "serving_parity": "Evaluation mode, float32, atol=1e-5/rtol=1e-4. The exported loader reloads exactly the current category checkpoint; every state tensor must exactly match the actual diagnostic model. Compare explicit fixed-position and ModelView features at visible positions, their MASK logits, internal appended-MASK scoring, exported predict, and batched/individual scoring. Scores must be finite and all ordered rankings identical. Rankings use descending score then ascending item ID; top10 agreement is intersection size/10. Short, full-length, truncated, and batched coverage are all required before sensitivity. External-label mutation remains explicitly untested; no hidden labels are accessed.",
            "history_sensitivity": "Training-prefix diagnostic, not host validation. Uniformly select up to 64 users before inspecting outcomes, one endpoint/user uniformly among endpoints with >=4 preceding interactions and a target absent from the complete prefix. Negatives are 99 unique real items absent from that complete prefix and distinct from target; later events never determine eligibility. Seed 42. Complete-prefix exclusions and candidates remain fixed across all variants. Serving truncates to maxlen-1 and appends MASK at index maxlen-1. Delete earliest/latest k=min(2,floor((visible_count-1)/2)) visible items into attention-masked PAD holes; preserve surviving absolute positions and MASK index, without backfill. Swap final two visible items. Repeated historical IDs are permitted. Unknown IDs are excluded and counted, not negative labels.",
            "paired_statistics": "Variant minus original: target_score_delta in logits; target_rank_delta in one-based positions (positive means deterioration); reciprocal_rank_delta uses 1/rank; diagnostic_hit10_delta uses indicator(rank<=10). top10_overlap is intersection/10; ordered_ranking_agreement compares complete ordered candidate rankings. Equal-user category means and 500 CPU paired-user bootstrap replicates, seed 42, percentile 95% intervals. Intervals cover sampled-user uncertainty only. Raw paired outcomes are retained for a four-category aggregation; that macro is unavailable in a single-category ModelView, never computed over fewer categories.",
            "elapsed_diagnostic_seconds": "Wall seconds including warmup, loading, probes, and CPU bootstrap, CUDA synchronized at both boundaries; all diagnostic GPU work consumes search budget.",
            "availability": "Null means unavailable; counts/exclusions and parity coverage must accompany interpretation. No ratings, review text, item categories/prices/stores, or simulator traits are measured.",
        }
        try:
            self.net.eval()
            with torch.no_grad():
                metrics["embedding_collapse_score"] = (
                    self.analyze_embeddings_collapse_cosine()
                )
                metrics["embedding_sample_count"] = self._embedding_count
                margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader
                )
                metrics.update(
                    {
                        "pos_neg_margin_mean": margin["mean"],
                        "pos_neg_margin_p10": margin["p10"],
                        "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                        "pos_neg_small_margin_rate": margin["small_margin_rate"],
                        "pos_neg_margin_count": margin["count"],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "sampler_contract": self._contract,
                    }
                )
                examples, counts = self._examples()
                metrics["prefix_counts"] = counts
                if not self._contract["passed"]:
                    metrics["status"] = "sampler_contract_failed"
                else:
                    # Explicitly require float32 instead of silently widening tolerances.
                    if self.net.item_emb.weight.dtype != torch.float32:
                        raise ValueError(
                            "serving parity requires the predeclared float32 precision"
                        )
                    parity = self._parity(examples)
                    metrics["serving_parity"] = parity
                    if not parity["passed"]:
                        metrics["status"] = (
                            "serving_parity_failed"
                            if parity["available"]
                            else "serving_parity_unavailable"
                        )
                    else:
                        metrics["history_sensitivity"] = self._paired_sensitivity(
                            examples, counts
                        )
                        metrics["status"] = (
                            "measured"
                            if counts.get("paired_users", 0)
                            else "insufficient_history"
                        )
        except Exception as exc:
            metrics["status"] = "diagnostic_failed"
            metrics["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.net.train(previous_state)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["elapsed_diagnostic_seconds"] = time.perf_counter() - start
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
