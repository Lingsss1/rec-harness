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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret sensitivity as measured training evidence
    - Missing measurements are unknown, not evidence of healthy behavior.
    - History perturbations measure fixed-model training sensitivity, not validation quality or counterfactual preferences.
    - Raw margins depend on score scale. A mapped cosine near 0.5 means mean cosine near zero; it is not a validated collapse measure.
    - Read counts, coverage, missing reasons, and limitations below before interpreting aggregates:
    {json.dumps({k: v for k, v in raw_diagnosis.items() if k not in ('metrics', 'metric_definitions')}, indent=2)}
    - Never infer simulator traits, ownership, compatibility, satisfied needs, review preferences, or unavailable metadata.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Keep the view contract separate from the trained module
    def __init__(self, view, device):
        self.view = view
        self.model = view
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.review_data = getattr(view, "review_data", {}) or {}
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item category metadata is unavailable in the matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing geometry from measured zero
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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained module
        self.module.eval()

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

            ### >>> Self_EvolveRec-BLOCK-START: Use tensor sequences for the trained SASRec architecture
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and invalid pairwise observations
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0) & torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Share the same validity mask in all pairwise counts
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
            ### >>> Self_EvolveRec-BLOCK-START: Report absent pairwise measurements explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Record pairwise denominator and metadata availability
        breakdown = {
            "valid_positions": sum(int(m.numel()) for m in margins_all),
            "category_missing_reason": "Item category metadata is unavailable in this task.",
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

    ### >>> Self_EvolveRec-BLOCK-START: Correct diagnostic candidate pools, preserve deployed positions, and bound scoring cost
    def _history_manifest(self):
        import hashlib
        import os

        histories = self.user_train.get("History", self.user_train)
        histories = {str(u): [int(i) for i in h] for u, h in histories.items()}
        itemnum = int(self.view.item_emb.weight.shape[0] - 1)
        if any(i < 1 or i > itemnum for h in histories.values() for i in h):
            raise ValueError("Diagnostic histories contain invalid original item IDs")
        dataset_name = getattr(self.view, "dataset_name", None)
        digest = hashlib.sha256(
            json.dumps(
                {"histories": histories, "dataset_name": dataset_name},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(directory, "diagnosis_manifest_" + digest[:20] + ".json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if (
                manifest.get("version") != 2
                or manifest.get("sha256") != digest
                or manifest.get("itemnum") != itemnum
            ):
                raise ValueError("Diagnostic manifest mismatch")
            return histories, manifest, path
        rng = np.random.RandomState(42)
        eligible = sorted(u for u, h in histories.items() if len(h) >= 3)
        chosen = rng.permutation(len(eligible))[:200]
        # Diagnostic negatives must have appeared in this category's training data.
        catalog = np.asarray(
            sorted({item for history in histories.values() for item in history}),
            dtype=np.int64,
        )
        users = []
        for index in chosen:
            user = eligible[int(index)]
            history = histories[user]
            prefix, target = history[:-1], history[-1]
            # Exclude the entire original prefix, including events outside the window.
            pool = catalog[~np.isin(catalog, list(set(prefix) | {target}))]
            negatives = rng.choice(pool, size=min(999, len(pool)), replace=False)
            users.append(
                {
                    "user": user,
                    "history": prefix[-50:],
                    "target": target,
                    "target_eligible": target not in prefix,
                    "candidates": [target] + [int(i) for i in negatives],
                    "shuffle_seeds": [
                        int(i) for i in rng.randint(0, 2**31 - 1, size=5)
                    ],
                }
            )
        manifest = {
            "version": 2,
            "sha256": digest,
            "itemnum": itemnum,
            "dataset_name": getattr(self.view, "dataset_name", None),
            "seed": 42,
            "eligible_users": len(eligible),
            "users": users,
            "window": 50,
            "recent_k": 1,
            "bootstrap_replicates": 1000,
            "popularity_tolerance": 0.25,
            "title_cosine_minimum": 0.5,
            "full_catalog_users": sorted(u for u, h in histories.items() if h)[:5],
            "rng_state_after_sampling": {
                "algorithm": rng.get_state()[0],
                "keys": rng.get_state()[1].tolist(),
                "position": int(rng.get_state()[2]),
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
        return histories, manifest, path

    def _scores(self, history, candidates):
        # Retain the deployed tensor length: shortening it moves absolute position IDs.
        cap = int(getattr(self.module, "maxlen", 50))
        if cap < 1 or len(history) > cap:
            raise ValueError("History exceeds diagnostic architecture window")
        if not candidates:
            return np.empty(0, dtype=np.float32)
        sequence = torch.zeros((1, cap), dtype=torch.long, device=self.device)
        if history:
            sequence[0, -len(history) :] = torch.tensor(history, device=self.device)
        items = torch.tensor([candidates], dtype=torch.long, device=self.device)
        transition_api = all(
            callable(getattr(self.module, name, None))
            for name in (
                "log2feats",
                "recent_contexts",
                "score",
            )
        )
        if transition_api:
            # Cache one history only. Full-catalog chunks reuse its encoder output;
            # explicit contexts avoid relying on mutable compatibility bindings.
            key = tuple(history)
            cached = getattr(self, "_score_cache", None)
            if cached is None or cached[0] != key:
                features = self.module.log2feats(sequence)[:, -1, :]
                contexts = self.module.recent_contexts(sequence)[:, -1, :]
                cached = (key, features, contexts)
                self._score_cache = cached
            values = torch.cat(
                [
                    self.module.score(cached[1], cached[2], chunk)
                    for chunk in items.split(1024, dim=1)
                ],
                dim=1,
            )[0]
        elif callable(getattr(self.module, "predict", None)):
            values = self.module.predict(sequence, items)[0]
        else:
            features = self.view.log2feats(sequence)
            values = self.view.predict_score(
                features, items[:, None, :].expand(1, sequence.shape[1], -1)
            )[0, -1]
        values = values.detach().float().cpu().numpy()
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise ValueError("Diagnostic scoring returned invalid values")
        return values

    @staticmethod
    def _rank_summary(baseline, perturbed, candidates, target_eligible):
        ids = np.asarray(candidates)
        original_order = np.lexsort((ids, -baseline))
        changed_order = np.lexsort((ids, -perturbed))
        original_ranks = np.empty(len(ids), dtype=np.int64)
        changed_ranks = np.empty(len(ids), dtype=np.int64)
        original_ranks[original_order] = np.arange(1, len(ids) + 1)
        changed_ranks[changed_order] = np.arange(1, len(ids) + 1)
        k = min(10, len(ids))
        overlap = len(set(original_order[:k]) & set(changed_order[:k])) / k
        result = {
            "candidate_mean_absolute_displacement": float(
                np.abs(changed_ranks - original_ranks).mean()
            ),
            "top10_overlap": overlap,
            "top10_changed": float(overlap < 1.0),
        }
        if target_eligible:
            result["target_displacement"] = float(changed_ranks[0] - original_ranks[0])
            result["target_hit_change"] = float(
                int(changed_ranks[0] <= 10) - int(original_ranks[0] <= 10)
            )
        return result

    def _history_probes(self):
        import os
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer

        histories, manifest, path = self._history_manifest()
        cap = int(getattr(self.module, "maxlen", 50))
        popularity = Counter(i for h in histories.values() for i in h)
        item_ids = sorted(popularity)
        titles = []
        for item in item_ids:
            entry = self.meta_data.get(str(item), self.meta_data.get(item, {}))
            title = entry.get("title", "") if isinstance(entry, dict) else ""
            titles.append(title if isinstance(title, str) else "")
        title_rows = {item: index for index, item in enumerate(item_ids)}
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            smooth_idf=True,
            norm="l2",
        )
        matrix = None
        analyzer = vectorizer.build_analyzer()
        fit_titles = [title for title in titles if analyzer(title)]
        if fit_titles:
            # Missing and untokenizable titles must not become empty IDF documents.
            vectorizer.fit(fit_titles)
            matrix = vectorizer.transform(titles)
        title_artifact = {
            "sklearn_version": sklearn.__version__,
            "available_titles": sum(bool(t.strip()) for t in titles),
            "training_items": len(item_ids),
            "settings": {
                "lowercase": True,
                "ngram_range": [1, 2],
                "min_df": 1,
                "sublinear_tf": True,
                "smooth_idf": True,
                "norm": "l2",
                "token_pattern": vectorizer.token_pattern,
            },
            "fitted_title_count": len(fit_titles),
            "vocabulary": (
                {token: int(index) for token, index in vectorizer.vocabulary_.items()}
                if matrix is not None
                else {}
            ),
            "idf": vectorizer.idf_.tolist() if matrix is not None else [],
        }
        with open(
            path.replace(".json", "_titles.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(title_artifact, handle, allow_nan=False)

        rows = {}
        counts = Counter(selected_users=len(manifest["users"]))
        for record in manifest["users"]:
            history = record["history"]
            if len(history) > cap:
                counts["architecture_window_too_short"] += 1
                continue
            candidates = record["candidates"]
            baseline = self._scores(history, candidates)
            eligible = record["target_eligible"]
            counts["scored_users"] += 1
            counts["target_eligible_users"] += int(eligible)
            user_values = {}

            def measure(name, variant):
                summary = self._rank_summary(
                    baseline, self._scores(variant, candidates), candidates, eligible
                )
                for metric, value in summary.items():
                    user_values.setdefault(name + "/" + metric, []).append(value)
                return summary

            recent = measure("recent_compaction", history[:-1])
            measure("older_prefix_exclusion", history[-1:])
            matches = []
            if matrix is not None:
                recent_row = title_rows[history[-1]]
                if matrix[recent_row].nnz:
                    for position, item in enumerate(history[:-1]):
                        row = title_rows[item]
                        difference = abs(
                            np.log1p(popularity[item])
                            - np.log1p(popularity[history[-1]])
                        )
                        cosine = float(matrix[row].multiply(matrix[recent_row]).sum())
                        if matrix[row].nnz and difference <= 0.25 and cosine >= 0.5:
                            cost = difference / 0.25 + (1.0 - cosine) / 0.5
                            matches.append((cost, position))
            if matches:
                # For k=1, this is the complete maximum-cardinality minimum-cost match.
                _, position = min(matches)
                older = measure(
                    "matched_older_compaction",
                    history[:position] + history[position + 1 :],
                )
                counts["matched_users"] += 1
                for metric in recent:
                    user_values["recent_minus_matched_older/" + metric] = [
                        recent[metric] - older[metric]
                    ]
            else:
                counts["unmatched_users"] += 1
            for name, keep_latest in (
                ("full_shuffle", False),
                ("older_prefix_shuffle", True),
            ):
                outputs = set()
                for seed in record["shuffle_seeds"]:
                    rng = np.random.RandomState(seed)
                    prefix = history[:-1] if keep_latest else history
                    variant = [int(i) for i in rng.permutation(prefix)]
                    if keep_latest:
                        variant.append(history[-1])
                    measure(name, variant)
                    outputs.add(tuple(variant))
                    user_values.setdefault(name + "/unchanged", []).append(
                        float(variant == history)
                    )
                    user_values.setdefault(
                        name + "/changed_position_fraction", []
                    ).append(float(np.mean(np.asarray(variant) != np.asarray(history))))
                user_values[name + "/distinct_outputs"] = [float(len(outputs))]
            for name, values in user_values.items():
                rows.setdefault(name, []).append(float(np.mean(values)))

        aggregate = {}
        for name, values in sorted(rows.items()):
            values = np.asarray(values, dtype=np.float64)
            rng = np.random.RandomState(42)
            draws = values[rng.randint(0, len(values), size=(1000, len(values)))].mean(
                axis=1
            )
            aggregate[name] = {
                "mean": float(values.mean()),
                "users": len(values),
                "user_bootstrap_95_interval": np.quantile(
                    draws, [0.025, 0.975]
                ).tolist(),
            }
        # Separate descriptive lists: training users, full catalog, original history exclusion.
        recommendations = []
        for user in manifest["full_catalog_users"]:
            seen = set(histories[user])
            # Simulator lists use the full deployed context cap, independently of
            # the separate 50-event perturbation window.
            history = histories[user][-cap:]
            best = []
            for start in range(1, manifest["itemnum"] + 1, 1024):
                candidates = [
                    i
                    for i in range(start, min(start + 1024, manifest["itemnum"] + 1))
                    if i not in seen
                ]
                if not candidates:
                    continue
                scores = self._scores(history, candidates)
                best.extend(
                    (float(score), item) for score, item in zip(scores, candidates)
                )
                best = sorted(best, key=lambda pair: (-pair[0], pair[1]))[:10]
            recommendations.append({"user": user, "items": [item for _, item in best]})
        recommendations_path = path.replace(".json", "_recommendations.json")
        with open(recommendations_path, "w", encoding="utf-8") as handle:
            json.dump(recommendations, handle, indent=2, allow_nan=False)
        return {
            "manifest": path,
            "dataset_name": manifest["dataset_name"],
            "counts": dict(counts),
            "aggregates": aggregate,
            "full_catalog_recommendations": recommendations_path,
            "full_catalog_user_count": len(recommendations),
            "position_preserving_content_ablation": None,
            "position_preserving_attention_exclusion": None,
            "constrained_shuffle": None,
            "missing_reasons": {
                "position_preserving_content_ablation": "The current SASRec forward has no embedding-only ablation hook. Padding substitution would change masking and is not used.",
                "position_preserving_attention_exclusion": "The current SASRec forward has no per-layer key/value exclusion hook preserving readout and survivor positions.",
                "constrained_shuffle": "The adapter does not establish a reliable event-aligned timestamp contract. Availability-constrained results and permutation-support checks are unavailable; unconstrained controls are labeled separately.",
            },
        }

    def run_full_diagnosis(self, train_loader):
        flags = [(module, module.training) for module in self.module.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        try:
            self._score_cache = None
            self.module.eval()
            with torch.no_grad():
                # Freeze users and candidate pools before computing any diagnostic outcome.
                self._history_manifest()
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=0.5, topk_categories=20
                )
                embedding = self.view.item_emb.weight.detach()[1:].float()
                norms = embedding.norm(dim=-1)
                metrics = {
                    "embedding_collapse_score": collapse_score,
                    "embedding_item_count": int(embedding.shape[0]),
                    "embedding_norm_mean": (
                        float(norms.mean()) if norms.numel() else None
                    ),
                    "embedding_norm_std": (
                        float(norms.std(unbiased=False)) if norms.numel() else None
                    ),
                    "pos_neg_margin_mean": margin["mean"],
                    "pos_neg_margin_p10": margin["p10"],
                    "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin["small_margin_rate"],
                    "pos_neg_valid_positions": breakdown["valid_positions"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "history_sensitivity": self._history_probes(),
                }
                definitions = {
                    "embedding_collapse_score": "Mapped mean cosine (cosine+1)/2 over seed-42 sampled nonpadding embeddings. Approximately 0.5 means mean cosine zero. This is not a validated collapse or recommendation-diversity measure.",
                    "embedding_item_count": "Number of nonpadding item embeddings.",
                    "embedding_norm_mean": "Mean L2 norm over nonpadding item embeddings.",
                    "embedding_norm_std": "Population standard deviation of nonpadding item L2 norms.",
                    "pos_neg_margin_mean": "Mean positive-minus-negative deployed score on valid training positions; depends on score scale and graph deployment policy.",
                    "pos_neg_margin_p10": "Nearest-rank tenth percentile of measured training score margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of valid positions with negative score strictly greater than positive score; ties are excluded from this rate.",
                    "pos_neg_small_margin_rate": "Fraction of valid training margins below 0.5; scale dependent, without a universal healthy threshold.",
                    "pos_neg_valid_positions": "Denominator for pairwise rates from at most ten loader batches; padding and nonfinite margins excluded.",
                    "pos_neg_margin_category_breakdown": "Pairwise counts and empty category lists; item category metadata is unavailable.",
                    "history_sensitivity": "Frozen training-only pseudo-target probes, separately for the supplied dataset view. Candidates exclude the entire original prefix and are identical across variants. One-based ranks break score ties by item ID. Target displacement is perturbed minus original rank; positive is deterioration. Target-hit change uses rank<=10 and excludes repeated pseudo-targets. Candidate displacement averages absolute rank changes. Top10 overlap divides intersection size by min(10,candidate count). Shuffle draws are averaged within each user before seed-42 1000-replicate user bootstrapping. Recent-minus-matched-older compares identical compaction operators on complete title/popularity matches. Full-catalog lists are descriptive only. Null controls have explicit missing reasons.",
                }
                unavailable = {
                    "review_preferences": "Review ratings and text are unavailable.",
                    "item_categories": "Item category metadata is unavailable.",
                    "simulator_traits": "Diversity and conformity traits are unobserved.",
                }
                if collapse_score is None:
                    unavailable["embedding_collapse_score"] = (
                        "Fewer than two nonpadding items."
                    )
                if margin["mean"] is None:
                    unavailable["pairwise_metrics"] = (
                        "No finite valid positive/negative pairs in the bounded sample."
                    )
                return {
                    "metrics": metrics,
                    "metric_definitions": definitions,
                    "unavailable": unavailable,
                    "limitations": [
                        "These are in-sample fixed-weight training sensitivity measurements, not benchmark validation or counterfactual preference estimates.",
                        "Bootstrap intervals condition on sampled users, fixed diagnostic candidates, and this training seed; they omit candidate randomness and adaptive search uncertainty.",
                        "Title and popularity matching can leave residual confounding; matching coverage must accompany paired comparisons.",
                        "Architecture-specific fixed-position controls are unavailable and are not approximated with padding IDs.",
                        "No guided intervention is selected from missing probes. This patch does not establish that diagnostic guidance improves HR@10.",
                        "Per-view full-catalog lists contain up to five real training users. Four dataset views are required for the prescribed twenty-user report.",
                    ],
                }
        finally:
            self._score_cache = None
            # Assign each flag independently: recursive train() would erase mixed modes.
            for module, training in flags:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    ### <<< Self_EvolveRec-BLOCK-END
