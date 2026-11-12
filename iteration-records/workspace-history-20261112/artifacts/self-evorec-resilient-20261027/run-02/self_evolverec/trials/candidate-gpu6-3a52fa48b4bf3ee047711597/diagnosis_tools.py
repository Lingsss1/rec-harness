import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Isolated diagnostic randomness and trial-local manifests
import random
import hashlib
import os
import time
import copy
import itertools
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
    # Review: checked candidate shapes/IDs, causal pair masks, query reservations,
    # unchanged-score controls, batching, RNG restoration and undefined cosine.
    # No execution or validation improvement is claimed. This diagnostic-only
    # patch does not implement training contribution dropout, energy calibration,
    # matched-arm launching or predictor export; those remain training-side work.
    ### <<< Self_EvolveRec-BLOCK-END
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Explicit diagnostic routing with actionable missing-route status
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
        self.route_status = (
            "Explicit category routes available; category-local results are valid"
            if self.views
            else "Unavailable: attach dataset_name to the view or its trained model, "
            "or supply diagnostic_views keyed by the four required categories"
        )
        for category, routed_view in self.views.items():
            for owner in (routed_view, routed_view.model):
                declared = getattr(owner, "dataset_name", None)
                if declared is not None and declared != category:
                    raise ValueError(
                        f"Diagnostic route {category!r} contains model/view {declared!r}"
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
            ### >>> Self_EvolveRec-BLOCK-START: Reject undefined cosine alignment for zero-norm embeddings
            x = emb.index_select(0, idx).float()
            if not torch.isfinite(x).all() or x.norm(dim=-1).le(1e-12).any():
                return None
            x = F.normalize(x, p=2, dim=-1)
            ### <<< Self_EvolveRec-BLOCK-END

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

        ### >>> Self_EvolveRec-BLOCK-START: Avoid consuming an unavailable probe and reject malformed loader batches
        batch_limit = num_batches
        if self.started is not None and self.margin_query_limit <= self.queries:
            batch_limit = 0
        for step, batch in enumerate(itertools.islice(dataloader, batch_limit)):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

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
            size = self.model.item_emb.num_embeddings
            malformed = seq_t.lt(0) | seq_t.ge(size)
            safe_sequence = seq_t.masked_fill(malformed, 0)
            log_feats = self.model.log2feats(safe_sequence)
            self.queries += int(seq_t.shape[0])
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate causal training pairs before embedding candidate IDs
            valid = (
                seq_t.gt(0)
                & seq_t.lt(size)
                & pos_t.gt(0)
                & pos_t.lt(size)
                & neg_t.gt(0)
                & neg_t.lt(size)
                & pos_t.ne(neg_t)
                & ~malformed.long().cumsum(1).gt(0)
            )
            safe_pos = pos_t.masked_fill(~valid, 0)
            safe_neg = neg_t.masked_fill(~valid, 0)
            positive_scores = self.model.predict_score(log_feats, safe_pos)
            negative_scores = self.model.predict_score(log_feats, safe_neg)
            margin = positive_scores - negative_scores
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

    ### >>> Self_EvolveRec-BLOCK-START: Baseline-frozen fixed-slot probes with descriptive paired inference and state restoration
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
        # Intervention models must reuse an explicitly supplied baseline panel.
        # Never regenerate matching statistics from intervention embeddings.
        supplied_path = getattr(self.model, "diagnostic_manifest_path", None)
        if supplied_path is None:
            supplied_path = getattr(self.module, "diagnostic_manifest_path", None)
        if supplied_path is not None:
            path = Path(supplied_path).resolve()
            manifest = json.loads(path.read_text())
            if manifest.get("version") != 3 or not manifest.get("baseline_verified"):
                raise ValueError(
                    "Expected a version-3 baseline-frozen diagnostic panel"
                )
            for category, view in self.views.items():
                digest = hashlib.sha256()
                histories = self._history(view)
                for user, history in sorted(histories.items(), key=lambda p: str(p[0])):
                    digest.update(
                        json.dumps([str(user), [int(i) for i in history]]).encode()
                    )
                expected = {
                    "history_sha256": digest.hexdigest(),
                    "maxlen": self._length(view),
                    "catalog_size": int(view.item_emb.weight.shape[0]) - 1,
                }
                actual = manifest.get("signatures", {}).get(category, {})
                if any(actual.get(k) != value for k, value in expected.items()):
                    raise ValueError(
                        f"Baseline panel does not match route {category!r}"
                    )
            return manifest, path
        baseline_verified = bool(self.views) and all(
            getattr(view.model, "variant", None) == "baseline"
            for view in self.views.values()
        )
        if not baseline_verified:
            manifest = {
                "version": 3,
                "baseline_verified": False,
                "examples": {},
                "eligibility": {},
                "settings": {"paired_panel_encoded_sequences": 0},
                "intervention_plan": {
                    "status": "Diagnostic panel unavailable",
                    "reason": "Expose model.variant='baseline' for an unchanged baseline, "
                    "or attach diagnostic_manifest_path pointing to its frozen panel",
                    "training_comparison": "Not launched by diagnosis_tools.py",
                    "remaining_shared_gpu_seconds": None,
                },
            }
            path = self.root / "panel_unavailable.json"
            path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
            return manifest, path
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
                raise ValueError("Empty catalog or nonfinite embedding norms")
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
                seeds, matching = [], {}
                for name, position in positions.items():
                    original = prefix[position]
                    elsewhere = set(prefix[:position] + prefix[position + 1 :])
                    repeated = original in elsewhere
                    present = np.isin(catalog, list(elsewhere))
                    pool = catalog[
                        (present == repeated)
                        & (catalog != original)
                        & (catalog != target)
                    ]
                    differences = standardized[pool - 1] - standardized[original - 1]
                    distances = np.square(differences).sum(axis=1)
                    closest = pool[np.lexsort((pool, distances))[:50]]
                    rep_seed = self._seed(0, category, user, name, "replacement")
                    seeds.append(rep_seed)
                    rep_rng = np.random.default_rng(rep_seed)
                    chosen = rep_rng.choice(
                        closest, size=min(5, len(closest)), replace=False
                    )
                    variants[name] = [[position, int(item)] for item in chosen]
                    residuals = standardized[chosen - 1] - standardized[original - 1]
                    matching[name] = {
                        "position": position,
                        "original_repeated_elsewhere": repeated,
                        "eligible_replacements": int(len(pool)),
                        "nearest_pool_size": int(len(closest)),
                        "shortage": int(5 - len(chosen)),
                        "standardized_frequency_norm_residuals": residuals.tolist(),
                        "squared_matching_distances": np.square(residuals)
                        .sum(1)
                        .tolist(),
                        "duplicate_count_changes": [
                            (
                                len(prefix)
                                - len(
                                    set(
                                        prefix[:position]
                                        + [int(item)]
                                        + prefix[position + 1 :]
                                    )
                                )
                            )
                            - (len(prefix) - len(set(prefix)))
                            for item in chosen
                        ],
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
                        "variants": variants,
                        "matching": matching,
                        "matching_supported": all(variants[name] for name in positions),
                        "baseline_matching_standardization": {
                            "mean": center.tolist(),
                            "std_clamped": scale.tolist(),
                        },
                    }
                )
        manifest = {
            "version": 3,
            "baseline_verified": True,
            "baseline_verification_scope": "Declared model.variant only; checkpoint equivalence and prospective timing require external verification",
            "required_categories": list(self.CATEGORIES),
            "signatures": signatures,
            "examples": examples,
            "eligibility": eligibility,
            "settings": {
                "users_per_category": 32,
                "negatives": 99,
                "replicates": 5,
                "bootstrap_replicates": 10000,
                "selection_seed": 0,
                "bootstrap_seed": 0,
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
                "padding": "zero left padding; occupied slots and final conditioning item preserved",
                "candidate_exclusions": "padding and complete user training history, including target and truncated events",
                "tie_rule": "descending score, then ascending item ID",
            },
            "intervention_plan": {
                "status": "Descriptive diagnostic pilot; no training or host selection performed by this module",
                "preregistration_timing": "before this probe; prior validation exposure unknown",
                "training_seed": 42,
                "preprocessing_seed": 0,
                "epoch_cap": int(os.environ.get("SASREC_EPOCHS", "200")),
                "selection": "unchanged pinned macro validation HR@10",
                "shared_search_budget_gpu_seconds": 43200,
                "estimated_training_gpu_seconds": None,
                "remaining_shared_gpu_seconds": None,
                "interventions": [
                    {"name": "baseline", "additional_dropout": 0.0},
                    {"name": "position_p0.1", "additional_dropout": 0.1},
                    {
                        "name": "item_p0.1",
                        "additional_dropout": 0.1,
                        "energy_matched": False,
                    },
                    {
                        "name": "energy_matched_item",
                        "optional": True,
                        "probability_rule": "E/(S_item+E), E=(0.1/0.9)*S_pos",
                        "calibration_status": "Unavailable: requires training-side initialization calibration",
                    },
                ],
                "training_integration": "Not implemented by this diagnostic module; no arm is reported as trained",
                "tie_priority": [
                    "baseline",
                    "position_p0.1",
                    "item_p0.1",
                    "energy_matched_item",
                ],
                "selection_boundary": "External pinned four-category macro validation HR@10 only",
                "analysis": "Across-arm diagnostic effects are descriptive; no diagnostic treatment policy is tested",
                "stop_rule": "External launcher must verify the remaining shared budget and complete-arm ceiling before each arm",
            },
        }
        # Routes and complete histories identify this frozen selection. Repeated calls
        # must reuse the saved example IDs, candidate order, and perturbation indices.
        key = hashlib.sha256(
            json.dumps(
                {
                    "design_version": 3,
                    "signatures": signatures,
                    "examples": examples,
                },
                sort_keys=True,
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
        size = view.item_emb.num_embeddings
        sequence = torch.zeros((rows, length), dtype=torch.long, device=self.device)
        for row, prefix in enumerate(prefixes):
            if not prefix or len(prefix) > length:
                raise ValueError("Diagnostic prefix length must lie in [1, maxlen]")
            ids = torch.as_tensor(prefix, dtype=torch.long, device=self.device)
            if ids.le(0).any() or ids.ge(size).any():
                raise ValueError("Diagnostic prefix contains an invalid item ID")
            sequence[row, -len(prefix) :] = ids
        items = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        if items.ndim != 1 or items.numel() != 100:
            raise ValueError("Paired diagnostics require exactly 100 shared candidates")
        if items.le(0).any() or items.ge(size).any() or items.unique().numel() != 100:
            raise ValueError("Diagnostic candidates must be unique valid item IDs")
        items = items.unsqueeze(0).expand(rows, -1)
        features = view.log2feats(sequence)[:, -1, :]
        scores = view.predict_score(features, items)
        self.queries += rows
        if scores.shape != items.shape:
            raise ValueError("Diagnostic scorer must preserve [B,C], including B=1")
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
            # The same resampled users preserve pairing across all measured effects.
            indices = rng.integers(len(rows), size=(10000, len(rows)))
            differences = np.asarray([row["D_user"] for row in rows], dtype=float)
            summaries[category]["D"] = {
                "mean": float(differences.mean()),
                "ci95": np.percentile(
                    differences[indices].mean(1), [2.5, 97.5]
                ).tolist(),
            }
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
        if all(summaries[c].get("D") is not None for c in self.CATEGORIES):
            macro["D"] = {
                "mean": float(
                    np.mean([summaries[c]["D"]["mean"] for c in self.CATEGORIES])
                ),
                "ci95": np.percentile(
                    np.mean(
                        [
                            draws[(c, "recent_identity", "normalized_rank_effect")]
                            - draws[(c, "older_identity", "normalized_rank_effect")]
                            for c in self.CATEGORIES
                        ],
                        axis=0,
                    ),
                    [2.5, 97.5],
                ).tolist(),
            }
        else:
            macro["D"] = None
        return summaries, macro

    @torch.no_grad()
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
        # Preserve exposed loader/sampler states without claiming that opaque
        # worker processes or a caller-owned iterator can be rewound.
        loader_states, generator_states = [], []
        owners, seen, seen_generators = [train_loader], set(), set()
        while owners:
            owner = owners.pop()
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            owners.extend(
                getattr(owner, name, None) for name in ("sampler", "batch_sampler")
            )
            generator = getattr(owner, "generator", None)
            if (
                isinstance(generator, torch.Generator)
                and id(generator) not in seen_generators
            ):
                seen_generators.add(id(generator))
                generator_states.append((generator, generator.get_state().clone()))
            if callable(getattr(owner, "state_dict", None)) and callable(
                getattr(owner, "load_state_dict", None)
            ):
                loader_states.append((owner, copy.deepcopy(owner.state_dict())))
        self._sync()
        self.started, self.queries = time.monotonic(), 0
        try:
            for module in modules:
                module.eval()
            manifest, manifest_path = (
                self.prepare_manifest_before_intervention_validation()
            )
            # Reserve only panels that this invocation can actually score.
            panel_queries = sum(
                2 + sum(len(edits) for edits in example["variants"].values())
                for category in self.views
                for example in manifest["examples"].get(category, [])
            )
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
                for example in manifest["examples"].get(category, []):
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
                            "prefix_length": len(prefix),
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
                                ):
                                    raise ValueError(
                                        "Fixed-slot/final-item diagnostic control failed"
                                    )
                                edited_prefixes.append(perturbed_prefix)
                                edit_names.append(name)
                        # Keep unchanged controls as separate calls; batch only edits.
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
                        # Only complete paired users enter aggregate statistics.
                        records[category].append(row)
                    except TimeoutError:
                        stopped = True
                        break
                if stopped:
                    break
            per_category, macro = self._summarize(records)
            missing = [c for c in self.CATEGORIES if not records.get(c)]
            incomplete = []
            for category in self.CATEGORIES:
                panel = manifest["examples"].get(category, [])
                complete = (
                    category in self.views
                    and bool(panel)
                    and all(e["matching_supported"] for e in panel)
                    and len(records.get(category, [])) == len(panel)
                )
                per_category[category]["frozen_panel_complete"] = complete
                if not complete:
                    incomplete.append(category)
                    per_category[category]["D"] = None
            # Incomplete or truncated panels never become a four-category macro.
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
                        "routing_status": self.route_status,
                        "incomplete_categories": incomplete,
                        "baseline_panel_verified": manifest.get(
                            "baseline_verified", False
                        ),
                        "budget_truncated": stopped,
                        "eligibility": manifest["eligibility"],
                        "paired_user_records": records,
                        "unchanged_history_controls": controls,
                        "position_preserving_control": {
                            "status": "Fixed-slot identity replacement and recent/older swap",
                            "preserved": "Padding, occupied positions, length, timestamp slots, final conditioning item and candidate IDs",
                            "limitations": "Replacement does not preserve exact repetition multiplicity; swaps preserve the item multiset. Neither independently identifies absolute-position effects or guarantees realistic transitions",
                        },
                    },
                    "intervention_experiment": manifest["intervention_plan"],
                    "diagnostic_accounting": {
                        "encoded_sequences": self.queries,
                        "elapsed_synchronized_wall_seconds": elapsed,
                        "manifest": str(manifest_path.resolve()),
                        "restored_loader_state_objects": len(loader_states),
                        "restored_loader_generators": len(generator_states),
                        "loader_state_limitation": "Only exposed state_dict/load_state_dict and Torch generators are restored. Use a dedicated diagnostic loader for opaque sampler, iterator or worker state",
                    },
                },
                "metric_definitions": {
                    "embedding_collapse_score": "(mean off-diagonal cosine + 1)/2 on up to 2048 nonpadding item embeddings, isolated seed 42. Unavailable for fewer than two samples, sampled zero-norm vectors or nonfinite vectors. Directional alignment only; 0.5 means mean cosine zero. Does not measure recommendation diversity or embedding rank.",
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
                    "ci95": "10000 paired user-bootstrap resamples per category using independent streams rooted at seed 0. Draws are averaged within user first; matching complete categories receive equal weight. Conditional on fitted models and frozen panels; excludes training-seed variability and adaptive-selection correction. Singleton intervals are degenerate.",
                    "normalized_rank_effect": "Perturbed minus unchanged target rank divided by 99; positive values mean harm within the same 100 diagnostic candidates.",
                    "D": "Mean across users of recent-minus-older mean normalized replacement rank effect. Replicates are averaged within user first. Unavailable for incomplete panels; not a treatment-selection rule or validation gain.",
                    "perturbations": "The unchanged baseline freezes up to 32 seed-0 selected training users/category, candidates and replacements. Earliest and penultimate occupied slots receive five distinct replacements from the nearest 50 eligible items, or fewer if unavailable. Eligibility matches presence elsewhere in the prefix; proximity uses standardized log1p training frequency and baseline embedding norm. Original item and target are excluded. A separate swap exchanges the same slots. Intervention models must load the explicit baseline manifest; their embeddings never regenerate matching statistics.",
                    "position_control": "All edits preserve padding, occupied positions, sequence length and final conditioning item. Swaps preserve the item multiset; replacements may change exact repetition multiplicity. Absolute-position and relative-recency effects are not independently identified; realistic transitions are not guaranteed.",
                    "data_scope": "Final public training interaction is the target and preceding retained history is the prefix. These targets may already have been trained on. No held-out labels, ratings, review text, item categories, prices, stores, or simulator traits are used.",
                    "validation": "Diagnostic candidates and tie rules are independent of the pinned host evaluator. No host users, exclusions, ranking semantics, or splits are changed. Sensitivity and bootstrap intervals do not establish validation improvements.",
                    "experiment_status": "This module measures descriptive fitted-target probes only. Positional/item contribution dropout, initialization energy calibration, matched training streams, complete-arm budget checks and pinned host selection require training-side integration and are not reported as executed. Baseline declaration alone does not verify checkpoint equivalence or preregistration. No host scores are reconstructed or inferred.",
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
            try:
                for owner, state in reversed(loader_states):
                    owner.load_state_dict(state)
                for generator, state in generator_states:
                    generator.set_state(state)
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
