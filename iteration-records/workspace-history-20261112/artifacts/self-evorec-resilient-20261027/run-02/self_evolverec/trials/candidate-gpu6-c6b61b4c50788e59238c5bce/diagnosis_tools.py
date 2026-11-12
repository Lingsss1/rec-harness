import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Isolated diagnostic randomness and trial-local manifests
import random
import hashlib
import os
import time
from pathlib import Path
from collections import Counter

### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Restrict interpretation to measured diagnostic evidence
    - Separate in-sample training-history sensitivity from validation generalization.
    - Report unsupported probes, incomplete coverage, small samples, and interval limitations.
    - Missing measurements are unknown, never zero or evidence of healthy behavior.
    - Cosine similarity measures directional alignment, not recommendation diversity.
    - Do not infer intent, ownership, dislikes, simulator traits, or unavailable metadata.
    - Do not claim improved HR@10 without independently measured pinned validation results.
    - Manifest creation does not establish preregistration before earlier validation queries.
    - Sensitivity is descriptive; neither higher nor lower sensitivity is inherently better.
    - Identity replacements preserve occupied positions; swaps preserve the item multiset.
    - Frequency/norm matching is partial control, not evidence of realistic transitions.
    - The fitted target is the final TRAIN interaction, never a validation label.
    - Later-target weighting changes target contributions, not conditioning recency directly.
    - Longer prefixes and absolute positions remain possible explanations of differences.
    - Interpret D as recent-minus-older normalized rank harm; D > 0 is a proposed
      deployment threshold, not a validated rule. Require complete category panels.
    - Without matched host outcomes, policy accuracy, gains, and association are unknown.
    ### <<< Self_EvolveRec-BLOCK-END
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Use the trained module and explicitly available dataset routes
    CATEGORIES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )

    def __init__(self, view, device):
        self.model = view  # Compatibility hooks live on the view.
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train
        # Optional explicit collection of category views. Never infer routes from IDs.
        supplied = getattr(view, "diagnostic_views", None)
        if supplied is not None:
            self.views = {c: supplied[c] for c in self.CATEGORIES if c in supplied}
        else:
            category = getattr(view, "dataset_name", None)
            if category is None:
                category = getattr(self.module, "dataset_name", None)
            self.views = {category: view} if category in self.CATEGORIES else {}
        self.root = Path(__file__).resolve().parent / "working" / "diagnostics"
        self.query_cap = 2048
        self.time_cap = 30.0
        self.margin_query_limit = 256
        self.queries = 0
        self.started = None
        for category, routed_view in self.views.items():
            declared = getattr(routed_view, "dataset_name", None)
            if declared is not None and declared != category:
                raise ValueError(
                    f"Diagnostic route {category!r} contains view {declared!r}"
                )
            routed_device = routed_view.item_emb.weight.device
            if routed_device.type != self.device.type or (
                self.device.index is not None
                and routed_device.index != self.device.index
            ):
                raise ValueError(
                    f"Diagnostic route {category!r} is on {routed_device}, expected {self.device}"
                )

    def _get_categories(self, item_id: int):
        # Item category metadata is unavailable in the shared task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish an unavailable alignment estimate from zero
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Exact mean cosine without quadratic Gram allocation
            if k < 2:
                return None
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite sampled item embeddings")
            # Sum all pairwise inner products, then remove the diagonal.
            # Accumulate in float64 to reduce cancellation near mean cosine zero.
            x64 = x.double()
            offdiag_sum = x64.sum(dim=0).square().sum() - x64.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Mode is managed on the actual trained module by the caller
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

            ### >>> Self_EvolveRec-BLOCK-START: Bound sampled-margin rows while reserving the paired diagnostic panel
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic loader must yield matching B x L tensors")
            if self.started is not None:
                remaining = min(
                    self.margin_query_limit - self.queries,
                    self.query_cap - self.queries,
                )
                rows = min(int(seq_t.shape[0]), remaining)
                if rows <= 0 or not self._within_budget(rows):
                    break
                seq_t, pos_t, neg_t = seq_t[:rows], pos_t[:rows], neg_t[:rows]
            log_feats = self.model.log2feats(seq_t)
            self.queries += int(seq_t.shape[0])
            ### <<< Self_EvolveRec-BLOCK-END

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Measure only genuine sampled positive-negative pairs
            valid = (pos_t != 0) & (neg_t != 0)
            if not torch.isfinite(margin[valid]).all():
                raise ValueError("Nonfinite sampled diagnostic margins")
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Keep breakdown eligibility identical to margin eligibility
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
            ### >>> Self_EvolveRec-BLOCK-START: Report empty sampled-margin probes explicitly
            margin_metrics = {
                "count": 0,
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

            ### >>> Self_EvolveRec-BLOCK-START: Record sampled-margin denominator
            margin_metrics = {
                "count": int(N),
                "mean": round(float(m.mean().item()), 4),
                ### <<< Self_EvolveRec-BLOCK-END
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Mark item-category breakdown as unavailable
        breakdown = {
            "status": "unsupported: item category metadata unavailable",
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

    ### >>> Self_EvolveRec-BLOCK-START: Frozen paired history perturbations with bounded descriptive inference
    @staticmethod
    def _seed(*parts):
        payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _within_budget(self, rows=1):
        if self.started is None:
            raise RuntimeError("Diagnostic budget clock has not been started")
        if rows < 0:
            raise ValueError("Diagnostic query count must be nonnegative")
        self._sync()
        return (
            self.queries + rows <= self.query_cap
            and time.monotonic() - self.started < self.time_cap
        )

    @staticmethod
    def _history(view):
        histories = view.user_train
        return histories["History"] if "History" in histories else histories

    @staticmethod
    def _length(view):
        length = getattr(view.model, "maxlen", None)
        if length is None:
            length = getattr(view, "maxlen", None)
        if length is None:
            raise ValueError(
                "Diagnostic view must expose the trained maximum sequence length"
            )
        return int(length)

    def prepare_manifest_before_intervention_validation(self):
        """Freeze inputs before this probe; earlier host validation timing is unknown."""
        self.root.mkdir(parents=True, exist_ok=True)
        examples, eligibility, signatures = {}, {}, {}
        for category, view in self.views.items():
            histories = self._history(view)
            length = self._length(view)
            catalog_size = int(view.item_emb.weight.shape[0]) - 1
            entries = []
            digest = hashlib.sha256()
            ordered = sorted(histories.items(), key=lambda pair: str(pair[0]))
            frequency = np.zeros(catalog_size + 1, dtype=np.float64)
            for _, history in ordered:
                ids = np.asarray(history, dtype=np.int64)
                if ((ids < 1) | (ids > catalog_size)).any():
                    raise ValueError("Training IDs fall outside the item vocabulary")
                np.add.at(frequency, ids, 1)
            norms = view.item_emb.weight.detach().float().norm(dim=1).cpu().numpy()
            if catalog_size < 1 or not np.isfinite(norms).all():
                raise ValueError("Empty catalog or nonfinite item embedding norms")
            norm_sha256 = hashlib.sha256(norms.astype(np.float32).tobytes()).hexdigest()
            covariates = np.column_stack((np.log1p(frequency[1:]), norms[1:]))
            center = covariates.mean(axis=0)
            scale = np.maximum(covariates.std(axis=0), 1e-8)
            standardized = (covariates - center) / scale
            catalog = np.arange(1, catalog_size + 1, dtype=np.int64)
            for user, history in ordered:
                history = [int(i) for i in history]
                digest.update(json.dumps([str(user), history]).encode())
                if len(history) < 6:
                    continue
                target = history[-1]
                prefix = history[:-1][-length:]
                if len(prefix) < 5:
                    continue
                if any(i < 1 or i > catalog_size for i in prefix + [target]):
                    raise ValueError(
                        "Training IDs fall outside the trained item vocabulary"
                    )
                excluded = sorted(set(history))
                count = catalog_size - len(excluded)
                if count < 99:
                    continue
                entries.append(
                    (str(user), len(history) - 1, prefix, target, excluded, count)
                )
            selection_rng = np.random.default_rng(0)
            entries = [entries[int(i)] for i in selection_rng.permutation(len(entries))]
            eligibility[category] = {
                "eligible_users": len(entries),
                "selected_users": min(32, len(entries)),
                "total_training_users": len(histories),
                "catalog_size": catalog_size,
            }
            signatures[category] = {
                "history_sha256": digest.hexdigest(),
                "maxlen": length,
                "catalog_size": catalog_size,
                "matching_norm_sha256": norm_sha256,
                "declared_variant": getattr(view.model, "variant", None),
            }
            examples[category] = []
            for user, cutoff, prefix, target, excluded, count in entries[:32]:
                seed = self._seed(category, user, "candidates")
                rng = np.random.default_rng(seed)
                # Map unique complement ranks to IDs without allocating the catalog.
                ranks = rng.choice(count, size=99, replace=False)
                adjusted_blocked = (
                    np.asarray(excluded, dtype=np.int64)
                    - 1
                    - np.arange(len(excluded), dtype=np.int64)
                )
                negatives = (
                    ranks + np.searchsorted(adjusted_blocked, ranks, side="right") + 1
                ).tolist()
                n = len(prefix)
                positions = {"recent_identity": n - 2, "older_identity": 0}
                variants = {"position_swap": [[n - 2, 0]]}
                matching = {}
                seeds = []
                for name, position in positions.items():
                    original = prefix[position]
                    elsewhere = prefix[:position] + prefix[position + 1 :]
                    elsewhere_set = set(elsewhere)
                    repeated = original in elsewhere_set
                    present = np.isin(catalog, list(elsewhere_set))
                    eligible = (
                        (present == repeated)
                        & (catalog != original)
                        & (catalog != target)
                    )
                    pool = catalog[eligible]
                    differences = standardized[pool - 1] - standardized[original - 1]
                    distances = np.square(differences).sum(axis=1)
                    # Include deterministic item-ID tie breaking at the pool boundary.
                    order = np.lexsort((pool, distances))[:50]
                    closest = pool[order]
                    rep_seed = self._seed(0, category, user, name, "replacement")
                    seeds.append(rep_seed)
                    rep_rng = np.random.default_rng(rep_seed)
                    chosen = rep_rng.choice(
                        closest, size=min(5, len(closest)), replace=False
                    )
                    variants[name] = [[position, int(item)] for item in chosen]
                    details = []
                    for item in chosen:
                        residual = (
                            standardized[int(item) - 1] - standardized[original - 1]
                        )
                        edited = list(prefix)
                        edited[position] = int(item)
                        details.append(
                            {
                                "replacement": int(item),
                                "standardized_log_frequency_residual": float(
                                    residual[0]
                                ),
                                "standardized_embedding_norm_residual": float(
                                    residual[1]
                                ),
                                "squared_matching_distance": float(
                                    np.square(residual).sum()
                                ),
                                "duplicate_count_change": int(
                                    (len(edited) - len(set(edited)))
                                    - (len(prefix) - len(set(prefix)))
                                ),
                            }
                        )
                    matching[name] = {
                        "position": position,
                        "distance_from_end": n - 1 - position,
                        "original_repeated_elsewhere": repeated,
                        "eligible_replacements": int(len(pool)),
                        "nearest_pool_size": int(len(closest)),
                        "requested_replicates": 5,
                        "actual_replicates": int(len(chosen)),
                        "shortage": int(5 - len(chosen)),
                        "draws": details,
                    }
                examples[category].append(
                    {
                        "user": user,
                        "prefix_cutoff": cutoff,
                        "prefix": prefix,
                        "target": target,
                        "candidates": [target] + negatives,
                        "excluded_ids": excluded,
                        "eligible_negatives": count,
                        "candidate_seed": seed,
                        "replicate_seeds": seeds,
                        "prefix_length": n,
                        "length_stratum": (
                            "5-9" if n < 10 else "10-19" if n < 20 else "20+"
                        ),
                        "variants": variants,
                        "matching": matching,
                        "matching_supported": all(variants[name] for name in positions),
                        "baseline_matching_standardization": {
                            "mean": center.tolist(),
                            "std_clamped": scale.tolist(),
                        },
                        "baseline_norm_sha256": norm_sha256,
                    }
                )
        manifest = {
            "version": 3,
            "required_categories": list(self.CATEGORIES),
            "signatures": signatures,
            "examples": examples,
            "eligibility": eligibility,
            "settings": {
                "users_per_category": 32,
                "negatives": 99,
                "replicates": 5,
                "bootstrap_replicates": 10000,
                "diagnostic_seed": 0,
                "bootstrap_seed": 0,
                "replacement_pool_size": 50,
                "replacement_matching": "repetition presence, standardized log1p training frequency and baseline embedding norm",
                "baseline_only_manifest": True,
                "query_cap": self.query_cap,
                "diagnostic_wall_seconds_cap": self.time_cap,
                "atol": 1e-6,
                "rtol": 1e-5,
                "checkpoint_schedule": sorted(
                    set(range(10, int(os.environ.get("SASREC_EPOCHS", "200")) + 1, 10))
                    | {int(os.environ.get("SASREC_EPOCHS", "200"))}
                ),
                "sampled_margin_sequence_cap": 256,
                "paired_panel_encoded_sequences": sum(
                    2
                    + sum(
                        len(replicates) for replicates in example["variants"].values()
                    )
                    for panel in examples.values()
                    for example in panel
                ),
                "diagnostic_checkpoints": "caller-supplied checkpoint only",
                "padding": "zero left padding; occupied positions unchanged by all edits",
                "candidate_exclusions": "padding and all user training interactions, including target and truncated history",
                "tie_rule": "descending score, then ascending item ID",
            },
            "intervention_plan": {
                "status": "host outcomes unavailable; one selected objective is trained per invocation",
                "preregistration_timing": "before this probe; prior validation exposure unknown",
                "training_seed": 42,
                "preprocessing_seed": 0,
                "epoch_cap": int(os.environ.get("SASREC_EPOCHS", "200")),
                "selection": "unchanged pinned macro validation HR@10",
                "shared_search_budget_gpu_seconds": 43200,
                "estimated_training_gpu_seconds": None,
                "remaining_shared_gpu_seconds": None,
                "interventions": [
                    {"name": "baseline", "target_decay": 0.0},
                    {
                        "name": "later-target",
                        "target_decay": float(np.log(2.0) / 10.0),
                        "half_life_positions": 10,
                        "loss": "sum(valid*w*(positive_BCE+negative_BCE))/sum(valid*w)",
                    },
                ],
                "prediction": "larger baseline D predicts larger variant-minus-baseline category HR@10 gain",
                "diagnostic": "D = mean_user(mean_recent_rank_change/99 - mean_older_rank_change/99)",
                "policy": "variant iff baseline D > 0; ties select baseline",
                "comparators": ["always-baseline", "always-variant"],
                "paired_validation_analysis": "only if host exposes paired hits: discordant counts and 10000 seed-0 within-category paired bootstrap resamples",
                "category_analysis": "Spearman with average tie ranks; undefined for constants; enumerate all 24 diagnostic assignments for one-sided association and policy-gain correspondence",
                "limitations": "Four heterogeneous categories; exchangeability questionable; minimum permutation fraction 1/24; one variant cannot establish general diagnostic usefulness",
                "missingness": "No host outcomes or remaining-budget ledger were supplied; policy values, validation intervals, association and permutation fractions remain unavailable",
                "stop_rule": "host shared 43200 GPU-second cap, including all eight fits, diagnostics, debugging and failures; no automatic additional fits",
            },
        }
        # Routes and complete histories identify this frozen selection. Repeated calls
        # must reuse the saved example IDs, candidate order, and perturbation indices.
        key = hashlib.sha256(
            json.dumps(
                {"design_version": 3, "signatures": signatures}, sort_keys=True
            ).encode()
        ).hexdigest()[:20]
        path = self.root / ("manifest_" + key + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved != manifest:
                raise ValueError(
                    "Frozen diagnostic manifest differs from current settings"
                )
            manifest = saved
        else:
            with path.open("x") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        return manifest, path

    @torch.no_grad()
    def _score_many(self, view, prefixes, candidates):
        rows = len(prefixes)
        if rows == 0:
            raise ValueError("Diagnostic scoring requires at least one prefix")
        if not self._within_budget(rows):
            raise TimeoutError("Diagnostic query or elapsed-time cap reached")
        length = self._length(view)
        sequence = torch.zeros((rows, length), dtype=torch.long, device=self.device)
        for row, prefix in enumerate(prefixes):
            if not prefix or len(prefix) > length:
                raise ValueError("Diagnostic prefix length must lie in [1, maxlen]")
            sequence[row, -len(prefix) :] = torch.as_tensor(
                prefix, dtype=torch.long, device=self.device
            )
        features = view.log2feats(sequence)[:, -1, :]
        items = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        if items.ndim != 1:
            raise ValueError("Batched paired diagnostics require shared [C] candidates")
        items = items.unsqueeze(0).expand(rows, -1)
        scores = view.predict_score(features, items)
        self.queries += rows
        if scores.shape != items.shape:
            raise ValueError("Diagnostic scorer must preserve [B,C]")
        if not torch.isfinite(scores).all() or not torch.isfinite(features).all():
            raise ValueError("Nonfinite diagnostic features or candidate scores")
        scores = scores.double().cpu().numpy()
        features = features.double().cpu().numpy()
        return [(scores[row], features[row]) for row in range(rows)]

    def _score(self, view, prefix, candidates):
        return self._score_many(view, [prefix], candidates)[0]

    @staticmethod
    def _ranking(scores, candidates, target):
        ids = np.asarray(candidates)
        order = np.lexsort((ids, -scores))
        target_index = int(np.flatnonzero(ids == target)[0])
        rank = int(np.flatnonzero(order == target_index)[0]) + 1
        return rank, set(ids[order[:10]].tolist())

    def _paired(self, original, perturbed, example):
        rank0, top0 = self._ranking(
            original[0], example["candidates"], example["target"]
        )
        rank1, top1 = self._ranking(
            perturbed[0], example["candidates"], example["target"]
        )
        a, b = original[1], perturbed[1]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return {
            "normalized_rank_effect": float(rank1 - rank0) / 99.0,
            "rank_change": float(rank1 - rank0),
            "hit10_change": float((rank1 <= 10) - (rank0 <= 10)),
            "top10_overlap": len(top0 & top1) / 10.0,
            "feature_cosine": (
                float(np.clip(np.dot(a, b) / denominator, -1, 1))
                if denominator > 0
                else None
            ),
        }

    def _summarize(self, records):
        names = ("recent_identity", "older_identity", "position_swap")
        metrics = (
            "normalized_rank_effect",
            "rank_change",
            "hit10_change",
            "top10_overlap",
            "feature_cosine",
        )
        summaries, draws = {}, {}
        for category in self.CATEGORIES:
            rows = records.get(category, [])
            summaries[category] = {"completed_users": len(rows), "variants": {}}
            if not rows:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([0, self.CATEGORIES.index(category)])
            )
            # Independent category streams from seed 0; identical indices within
            # each category preserve user pairing across metrics and edits.
            indices = rng.integers(len(rows), size=(10000, len(rows)))
            differences = np.asarray([row["D_user"] for row in rows], dtype=float)
            samples = differences[indices].mean(axis=1)
            summaries[category]["D"] = {
                "mean": float(differences.mean()),
                "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            }
            strata = {}
            for stratum in ("5-9", "10-19", "20+"):
                values = [
                    row["D_user"] for row in rows if row["length_stratum"] == stratum
                ]
                strata[stratum] = {
                    "users": len(values),
                    "D_mean": float(np.mean(values)) if values else None,
                }
            summaries[category]["length_strata"] = strata
            for name in names:
                summaries[category]["variants"][name] = {}
                for metric in metrics:
                    means, deviations = [], []
                    for row in rows:
                        values = [
                            r[metric]
                            for r in row["variants"][name]
                            if r[metric] is not None
                        ]
                        means.append(float(np.mean(values)) if values else np.nan)
                        deviations.append(float(np.std(values)) if values else np.nan)
                    values = np.asarray(means)
                    available = np.isfinite(values)
                    entry = {"users_with_defined_metric": int(available.sum())}
                    if available.all():
                        samples = values[indices].mean(axis=1)
                        draws[(category, name, metric)] = samples
                        entry.update(
                            {
                                "mean": float(values.mean()),
                                "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
                                "mean_within_user_replicate_sd": float(
                                    np.mean(deviations)
                                ),
                            }
                        )
                    else:
                        entry.update(
                            {
                                "mean": None,
                                "ci95": None,
                                "mean_within_user_replicate_sd": None,
                            }
                        )
                    summaries[category]["variants"][name][metric] = entry
        macro = {}
        for name in names:
            macro[name] = {}
            for metric in metrics:
                keys = [(category, name, metric) for category in self.CATEGORIES]
                if all(key in draws for key in keys):
                    samples = np.mean([draws[key] for key in keys], axis=0)
                    means = [
                        summaries[c]["variants"][name][metric]["mean"]
                        for c in self.CATEGORIES
                    ]
                    macro[name][metric] = {
                        "mean": float(np.mean(means)),
                        "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
                    }
                else:
                    macro[name][metric] = None
        return summaries, macro

    def run_full_diagnosis(self, train_loader):
        # Snapshot all actual modules, including optional explicit category views.
        modules = list(
            {
                id(v.model): v.model for v in [self.model] + list(self.views.values())
            }.values()
        )
        modes = [
            (submodule, submodule.training)
            for module in modules
            for submodule in module.modules()
        ]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self._sync()
        self.started, self.queries = time.monotonic(), 0
        try:
            for module in modules:
                module.eval()
            manifest, manifest_path = (
                self.prepare_manifest_before_intervention_validation()
            )
            panel_queries = manifest["settings"]["paired_panel_encoded_sequences"]
            self.margin_query_limit = min(256, max(0, self.query_cap - panel_queries))
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=0.5, topk_categories=20
                )
            )
            records, controls, stopped = {}, [], False
            for category in self.CATEGORIES:
                if category not in self.views:
                    continue
                view = self.views[category]
                records[category] = []
                for example in manifest["examples"][category]:
                    prefix, candidates = example["prefix"], example["candidates"]
                    try:
                        required_queries = 2 + sum(
                            len(replicates)
                            for replicates in example["variants"].values()
                        )
                        if not self._within_budget(required_queries):
                            raise TimeoutError(
                                "Insufficient diagnostic budget for a complete paired user"
                            )
                        original = self._score(view, prefix, candidates)
                        unchanged = self._score(view, list(prefix), candidates)
                        difference = float(np.max(np.abs(original[0] - unchanged[0])))
                        passed = bool(
                            np.allclose(original[0], unchanged[0], atol=1e-6, rtol=1e-5)
                        )
                        controls.append(
                            {
                                "category": category,
                                "user": example["user"],
                                "max_abs_score_difference": difference,
                                "passed": passed,
                            }
                        )
                        if not passed:
                            raise ValueError("Unchanged-history score control failed")
                        if not example["matching_supported"]:
                            continue
                        row = {
                            "user": example["user"],
                            "prefix_length": example["prefix_length"],
                            "length_stratum": example["length_stratum"],
                            "matching": example["matching"],
                            "variants": {},
                        }
                        edited_prefixes, edit_names = [], []
                        for name, replicates in example["variants"].items():
                            row["variants"][name] = []
                            for edit in replicates:
                                perturbed_prefix = list(prefix)
                                if name == "position_swap":
                                    a, b = edit
                                    perturbed_prefix[a], perturbed_prefix[b] = (
                                        perturbed_prefix[b],
                                        perturbed_prefix[a],
                                    )
                                else:
                                    position, replacement = edit
                                    perturbed_prefix[position] = replacement
                                if (
                                    len(perturbed_prefix) != len(prefix)
                                    or perturbed_prefix[-1] != prefix[-1]
                                    or any(item == 0 for item in perturbed_prefix)
                                ):
                                    raise ValueError(
                                        "Position/final-item control failed"
                                    )
                                edited_prefixes.append(perturbed_prefix)
                                edit_names.append(name)
                        # Batch only edited histories; repeated unchanged scoring
                        # remains two independent calls to detect nondeterminism.
                        edited_results = self._score_many(
                            view, edited_prefixes, candidates
                        )
                        for name, perturbed in zip(edit_names, edited_results):
                            row["variants"][name].append(
                                self._paired(original, perturbed, example)
                            )
                        row["D_user"] = float(
                            np.mean(
                                [
                                    r["normalized_rank_effect"]
                                    for r in row["variants"]["recent_identity"]
                                ]
                            )
                            - np.mean(
                                [
                                    r["normalized_rank_effect"]
                                    for r in row["variants"]["older_identity"]
                                ]
                            )
                        )
                        # Average replacement draws within user before aggregation.
                        records[category].append(row)
                    except TimeoutError:
                        stopped = True
                        break
                if stopped:
                    break
            per_category, macro = self._summarize(records)
            missing = [c for c in self.CATEGORIES if not records.get(c)]
            incomplete = []
            choices = {}
            for category in self.CATEGORIES:
                panel = manifest["examples"].get(category, [])
                complete = (
                    bool(panel)
                    and all(e["matching_supported"] for e in panel)
                    and len(records.get(category, [])) == len(panel)
                )
                per_category[category]["frozen_panel_complete"] = complete
                if complete:
                    value = per_category[category]["D"]["mean"]
                    choices[category] = "later-target" if value > 0.0 else "baseline"
                else:
                    incomplete.append(category)
                    per_category[category]["D"] = None
            baseline_verified = not incomplete and all(
                getattr(self.views[c].model, "variant", None) == "baseline"
                for c in self.CATEGORIES
            )
            policy = {
                "status": (
                    "baseline diagnostic choices; checkpoint equivalence and prospective timing require external verification"
                    if baseline_verified
                    else "unavailable: incomplete panel or baseline objective not explicitly identified"
                ),
                "baseline_objective_identified": baseline_verified,
                "choices": choices if baseline_verified else None,
                "incomplete_categories": incomplete,
                "threshold": 0.0,
                "ties": "baseline",
                "host_policy_macro_HR10": None,
                "host_always_baseline_macro_HR10": None,
                "host_always_variant_macro_HR10": None,
                "spearman": None,
                "permutation_fractions": None,
                "paired_validation_intervals": None,
                "validation_status": "Matched pinned host outcomes have not been supplied; local checkpoint metrics are not substituted",
                "timing": "Frozen before this probe's results; prior validation exposure unknown. Apply only measurements made on the matched baseline, not the variant",
            }
            if stopped or incomplete:
                macro = None
            self._sync()
            elapsed = time.monotonic() - self.started
            result = {
                "metrics": {
                    "embedding_collapse_score": collapse_score,
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_count": margin_metrics["count"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "history_perturbations": {
                        "per_category": per_category,
                        "four_category_macro": macro,
                        "missing_categories": missing,
                        "routing_status": (
                            "explicit dataset routes available"
                            if self.views
                            else "unsupported: provide dataset_name or an explicit diagnostic_views mapping; routes cannot be inferred from local item IDs"
                        ),
                        "budget_truncated": stopped,
                        "diagnostic_deployment_policy": policy,
                        "eligibility": manifest["eligibility"],
                        "paired_user_records": records,
                        "unchanged_history_controls": controls,
                        "position_preserving_control": {
                            "status": "identity replacement and recent/older swap implemented",
                            "preserved": "shape, padding, occupied positions, final conditioning item, candidates; no token deletion or timestamp edit",
                            "limitations": "Relative recency and absolute position are not independently identified; matching does not guarantee realistic transitions",
                        },
                    },
                    "intervention_experiment": manifest["intervention_plan"],
                    "diagnostic_accounting": {
                        "encoded_sequences": self.queries,
                        "elapsed_synchronized_wall_seconds": elapsed,
                        "manifest": str(
                            manifest_path.relative_to(Path(__file__).resolve().parent)
                        ),
                    },
                },
                "metric_definitions": {
                    "embedding_collapse_score": "(mean off-diagonal cosine + 1)/2 on up to 2048 nonpadding item embeddings, isolated seed 42. Directional alignment only; 0.5 means mean cosine zero. Does not measure recommendation diversity or embedding rank.",
                    "pos_neg_margin_mean": "Mean positive-minus-sampled-negative logit over nonpadding pairs from at most ten loader batches and 256 sequences, further limited to reserve the frozen paired panel. In-sample, dependent on loader sampling, and scoped to the primary view rather than a four-category macro.",
                    "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of sampled margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of sampled margins strictly below zero.",
                    "pos_neg_small_margin_rate": "Fraction of sampled margins below 0.5; raw score scale affects this measurement.",
                    "pos_neg_margin_count": "Number of valid sampled positive-negative positions; missing estimates are null.",
                    "pos_neg_margin_category_breakdown": "Unsupported: item categories are unavailable. Empty category lists are not evidence of no category effects.",
                    "rank_change": "Perturbed target rank minus original rank; positive is worse. Diagnostic ranks use score descending, item ID ascending ties.",
                    "hit10_change": "Perturbed indicator(target rank <= 10) minus original indicator, within the same 100 diagnostic candidates.",
                    "top10_overlap": "Size of original/perturbed top-ten set intersection divided by ten.",
                    "feature_cosine": "Cosine between final-position features; null when either feature has zero norm.",
                    "aggregation": "Replicates averaged within each user, users averaged within each category, then all four required categories equally. Missing categories produce null macro entries; truncation suppresses the macro entirely.",
                    "ci95": "10000 paired user-bootstrap resamples per category from independent category streams rooted at seed 0; percentile interval. At most 32 training users/category. Conditional on fitted model, selected users, candidates and replacements; excludes training-seed variability and adaptive-selection correction. Singleton intervals are degenerate.",
                    "normalized_rank_effect": "(edited target rank - unchanged target rank)/99 within the same target plus 99 unique diagnostic negatives; positive means harm.",
                    "D": "Mean across users of recent-minus-older mean normalized replacement rank effect. Draws are averaged within user first. Unavailable for incomplete frozen panels; not validation HR@10 gain.",
                    "perturbations": "Recent replaces the penultimate occupied item; older replaces the earliest. Final item and occupied positions stay fixed. Five distinct replacements are sampled from the closest 50 eligible items, or all if fewer. Eligibility matches presence elsewhere in prefix, excluding original and target. Squared standardized log1p training frequency and baseline embedding norm determine proximity. Shortages and residuals are recorded. Swap exchanges these two positions while preserving the item multiset.",
                    "position_control": "Identity replacement retains occupied positions and padding; swap preserves the multiset. Neither independently identifies absolute-position effects, and both can induce unrealistic transitions. Repetition presence does not preserve exact multiplicity.",
                    "data_scope": "Final public training interaction is the target and preceding retained history is the prefix. These targets may already have been trained on. No held-out labels, ratings, review text, item categories, prices, stores, or simulator traits are used.",
                    "validation": "Diagnostic candidates and tie rules are independent of the pinned host evaluator. No host users, exclusions, ranking semantics, or splits are changed. Sensitivity and bootstrap intervals do not establish validation improvements.",
                    "experiment_status": "Training exposes baseline and fixed later-target BCE objectives through SASREC_VARIANT; one four-category fit per invocation. Diagnostic policy choices require all four complete panels and explicit model.variant='baseline'. This identifies the declared objective, not checkpoint equivalence or prospective timing. Matching manifests are keyed by history, norms and objective; a variant-generated manifest is a separate descriptive audit and cannot replace a frozen baseline panel. No experiment launcher, shared budget ledger or host-outcome ingestion is implemented here; validation policy values and correspondence remain unavailable until externally measured. Earlier inspected evidence remains retrospective.",
                    "accounting": "Elapsed synchronized wall time includes manifest generation, sampled probes, scoring, and summaries; it is a conservative local diagnostic cap checked before encodes, not GPU event time or a replacement for host shared-budget accounting. An in-flight encode or CPU summary may finish beyond the cap.",
                },
            }
            # A separate artifact preserves replicate-level measurements for audits.
            output = self.root / (
                "result_" + manifest_path.stem + "_" + str(time.time_ns()) + ".json"
            )
            with output.open("x") as handle:
                json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            return result
        finally:
            for submodule, training in modes:
                submodule.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.started = None

    ### <<< Self_EvolveRec-BLOCK-END
