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
    ### >>> Self_EvolveRec-BLOCK-START: Constrain interpretation to observed evidence and available controls
    - Every statement must be grounded in the provided metric values/definitions.
    - Null values and unavailable controls are missing evidence, never zero effects.
    - Training-pair and training-prefix probes do not estimate pinned validation HR@10.
    - Semantic deletion remains unresolved without position-preserving attention controls; do not recommend enabling consistency from compaction alone.
    - Do not infer causality, user dislikes, recommendation diversity, simulator traits, or unavailable review/rating/category/price/store attributes.
    - The historical embedding_collapse_score measures directional alignment, not proof of collapse.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Diagnose the actual module through the supplied compatibility view
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.user_train = getattr(view, "user_train", {}) or {}

    def _get_categories(self, item_id: int):
        # Item-category attributes are unavailable in the matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Use the diagnostic view embedding contract
            emb = self.view.item_emb.weight.detach()
            ### <<< Self_EvolveRec-BLOCK-END
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Report unavailable alignment for insufficient samples
            if n < 2 or sample_size < 2:
                return None
            if not torch.isfinite(emb).all():
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Compute directional alignment without a quadratic Gram matrix
            x = emb.index_select(0, idx).float()
            if (x.norm(dim=-1) <= 1e-12).any():
                return None
            x = F.normalize(x, p=2, dim=-1)
            # Sum of off-diagonal dot products, using O(k*d) work/storage.
            summed = x.double().sum(dim=0)
            offdiag_sum = summed.square().sum() - x.double().square().sum()
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
        self.model.eval()

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score valid causal prefixes without candidate broadcasting errors
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            catalog_size = self.view.item_emb.num_embeddings
            malformed = (seq_t < 0) | (seq_t >= catalog_size)
            safe_seq = seq_t.masked_fill(malformed, 0)
            valid = (
                (seq_t > 0)
                & (seq_t < catalog_size)
                & (pos_t > 0)
                & (pos_t < catalog_size)
                & (neg_t > 0)
                & (neg_t < catalog_size)
                & (pos_t != neg_t)
                & ~(malformed.long().cumsum(dim=1) > 0)
            )
            log_feats = self.view.log2feats(safe_seq)
            margin = torch.zeros_like(pos_t, dtype=log_feats.dtype)
            hybrid = getattr(self.model, "hybrid_score", None)
            if callable(hybrid):
                # Each hidden position receives only its own causal prefix.
                for t in range(seq_t.shape[1]):
                    rows = valid[:, t].nonzero(as_tuple=True)[0]
                    if not rows.numel():
                        continue
                    ids = torch.stack((pos_t[rows, t], neg_t[rows, t]), dim=1)
                    scores = hybrid(log_feats[rows, t], ids, safe_seq[rows, : t + 1])
                    margin[rows, t] = scores[:, 0] - scores[:, 1]
            else:
                safe_pos = pos_t.masked_fill(~valid, 0)
                safe_neg = neg_t.masked_fill(~valid, 0)
                positive = self.view.predict_score(log_feats, safe_pos)
                negative = self.view.predict_score(log_feats, safe_neg)
                margin = positive - negative
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Exclude nonfinite margins from threshold counts
            finite_valid = valid_cpu & np.isfinite(margin_cpu)
            low_mask = finite_valid & (margin_cpu < float(eps))
            wrong_mask = finite_valid & (margin_cpu < 0.0)
            ### <<< Self_EvolveRec-BLOCK-END

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

        ### >>> Self_EvolveRec-BLOCK-START: Preserve counts and distinguish missing or nonfinite margins
        m = torch.cat(margins_all) if margins_all else torch.empty(0)
        margin_metrics = {
            "mean": None,
            "p10": None,
            "neg_beats_pos_rate": None,
            "small_margin_rate": None,
            "valid_pair_count": int(m.numel()),
            "nonfinite_pair_count": int((~torch.isfinite(m)).sum().item()),
        }
        if m.numel() and torch.isfinite(m).all():
            margin_metrics.update(
                {
                    "mean": float(m.mean().item()),
                    "p10": float(torch.quantile(m, 0.1).item()),
                    "neg_beats_pos_rate": float((m < 0).float().mean().item()),
                    "small_margin_rate": float((m < float(eps)).float().mean().item()),
                }
            )
        ### <<< Self_EvolveRec-BLOCK-END

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

        ### >>> Self_EvolveRec-BLOCK-START: Declare unavailable category attributes explicitly
        breakdown["available"] = False
        breakdown["reason"] = "Item-category metadata is unavailable in this task."
        return margin_metrics, breakdown
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Bounded training-only semantic deletion controls with conservative interpretation
    @torch.no_grad()
    def analyze_semantic_deletion(self, max_users=24, maxlen=128):
        from sklearn.feature_extraction.text import TfidfVectorizer

        settings = {
            "seed": 42,
            "max_users": max_users,
            "maxlen": maxlen,
            "random_controls": 5,
            "negatives": 99,
            "text": "training-item titles/descriptions; lowercase word unigrams/bigrams; sublinear TF; L2 normalization",
            "training_target_inclusion": True,
            "position_preserving_mask": {
                "available": False,
                "reason": "The compatibility contract does not expose explicit attention-key/value masking with preserved position IDs.",
            },
            "hypothesis_status": "unresolved",
            "training_intervention": "No consistency penalty is enabled by this diagnostic.",
        }
        result = {
            "settings": settings,
            "eligible_users": 0,
            "attempted_users": 0,
            "evaluated_users": 0,
            "nonfinite_users": 0,
            "outcomes": [],
            "summary": None,
        }
        settings["cohort_rule"] = (
            "First max_users eligible histories in the fixed seed-42 order; "
            "failed or candidate-empty histories are not replaced."
        )
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            result["unavailable_reason"] = "Training histories unavailable."
            return result
        histories = {
            str(u): [int(i) for i in h]
            for u, h in histories.items()
            if isinstance(h, (list, tuple, np.ndarray))
        }
        size = self.view.item_emb.num_embeddings
        train_ids = sorted({i for h in histories.values() for i in h if 0 < i < size})

        def text_for(item):
            info = self.meta_data.get(str(item), self.meta_data.get(item, {}))
            if not isinstance(info, dict):
                return ""
            parts = []
            for key in ("title", "description"):
                value = info.get(key, "")
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    parts.extend(v for v in value if isinstance(v, str))
            return " ".join(parts).strip()

        texts = [text_for(i) for i in train_ids]
        if not any(texts):
            result["unavailable_reason"] = (
                "No usable training-item title/description text."
            )
            return result
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        settings["tfidf_parameters"] = {
            "lowercase": True,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "norm": "l2",
            "token_pattern": vectorizer.token_pattern,
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "max_features": vectorizer.max_features,
            "fit_item_count": len(train_ids),
        }
        try:
            vectors = vectorizer.fit_transform(texts).tocsr()
        except ValueError as exc:
            result["unavailable_reason"] = str(exc)
            return result
        observed = np.asarray(vectors.getnnz(axis=1)).ravel() > 0
        index = {item: row for row, item in enumerate(train_ids)}
        rng = np.random.default_rng(42)
        users = sorted(histories)
        users = [users[i] for i in rng.permutation(len(users))]
        hybrid = getattr(self.model, "hybrid_score", None)
        window = min(maxlen, int(getattr(self.model, "maxlen", maxlen)))
        arm_names = ["clean", "targeted", "reversed"] + [
            f"random_{i}" for i in range(5)
        ]
        for user in users:
            history = histories[user]
            if len(history) < 5 or any(i <= 0 or i >= size for i in history):
                continue
            target = history[-1]
            prefix = history[:-1][-window:]
            text_positions = [
                p
                for p, item in enumerate(prefix)
                if item in index and observed[index[item]]
            ]
            eligible = [p for p in text_positions if p < len(prefix) // 2]
            if len(eligible) < 2:
                continue
            result["eligible_users"] += 1
            if result["attempted_users"] >= max_users:
                continue
            result["attempted_users"] += 1
            # Sparse per-prefix products only; missing text never implies isolation.
            local = vectors[[index[prefix[p]] for p in text_positions]]
            similarity = (local @ local.T).toarray()
            np.fill_diagonal(similarity, -np.inf)
            isolation = 1.0 - similarity.max(axis=1)
            iso = dict(zip(text_positions, isolation.tolist()))
            chosen = min(eligible, key=lambda p: (-iso[p], p))
            reversed_chosen = min(eligible, key=lambda p: (iso[p], p))
            quartile = lambda p: min(3, 4 * p // len(prefix))
            matched = [p for p in eligible if quartile(p) == quartile(chosen)]
            random_deletions = rng.choice(matched, size=5, replace=True).tolist()
            available = np.setdiff1d(np.asarray(train_ids), np.unique(history))
            if not len(available):
                continue
            negatives = rng.choice(
                available, size=min(99, len(available)), replace=False
            )
            candidates = [target] + negatives.tolist()
            deletions = [None, chosen, reversed_chosen] + random_deletions
            edited = []
            for deletion in deletions:
                survivors = [item for p, item in enumerate(prefix) if p != deletion]
                edited.append([0] * (window - len(survivors)) + survivors)
            sequence = torch.tensor(edited, dtype=torch.long, device=self.device)
            ids = torch.tensor(
                [candidates] * len(edited), dtype=torch.long, device=self.device
            )
            features = self.view.log2feats(sequence)[:, -1]
            if callable(hybrid):
                scores = hybrid(features, ids, sequence)
            else:
                scores = torch.stack(
                    [
                        self.view.predict_score(features, ids[:, c])
                        for c in range(ids.shape[1])
                    ],
                    dim=1,
                )
            scores = scores.detach().float().cpu().numpy()
            if not np.isfinite(scores).all():
                result["nonfinite_users"] += 1
                continue
            orders = np.argsort(-scores, axis=1, kind="stable")
            ranks = np.argsort(orders, axis=1)
            k = min(10, len(candidates))
            clean_top = set(orders[0, :k].tolist())
            row = {
                "user": user,
                "history_length": len(prefix),
                "candidate_count": len(candidates),
                "targeted_position": chosen,
                "random_positions": random_deletions,
                "arms": {},
            }
            for a, name in enumerate(arm_names):
                row["arms"][name] = {
                    "target_rank": int(ranks[a, 0]) + 1,
                    "hit_at_10": int(ranks[a, 0] < 10),
                    "top10_overlap": len(clean_top & set(orders[a, :k].tolist())) / k,
                    "clean_top10_mean_absolute_rank_displacement": float(
                        np.abs(ranks[a, orders[0, :k]] - ranks[0, orders[0, :k]]).mean()
                    ),
                }
            result["outcomes"].append(row)
            result["evaluated_users"] += 1
        if result["outcomes"]:
            differences = []
            for row in result["outcomes"]:
                arms = row["arms"]
                differences.append(
                    {
                        "target_rank_improvement": arms["clean"]["target_rank"]
                        - arms["targeted"]["target_rank"],
                        "targeted_minus_random_rank_improvement": float(
                            np.mean(
                                [arms[f"random_{i}"]["target_rank"] for i in range(5)]
                            )
                        )
                        - arms["targeted"]["target_rank"],
                        "targeted_minus_random_hit": arms["targeted"]["hit_at_10"]
                        - float(
                            np.mean(
                                [arms[f"random_{i}"]["hit_at_10"] for i in range(5)]
                            )
                        ),
                    }
                )
            result["summary"] = {
                key: float(np.mean([d[key] for d in differences]))
                for key in differences[0]
            }
            result["summary_by_history_length"] = {}
            for label, lower, upper in (
                ("4_to_15", 4, 16),
                ("16_to_63", 16, 64),
                ("64_plus", 64, float("inf")),
            ):
                selected = [
                    d
                    for row, d in zip(result["outcomes"], differences)
                    if lower <= row["history_length"] < upper
                ]
                result["summary_by_history_length"][label] = {
                    "evaluated_users": len(selected),
                    "paired_mean_differences": {
                        key: (
                            float(np.mean([d[key] for d in selected]))
                            if selected
                            else None
                        )
                        for key in differences[0]
                    },
                }
        return result

    def run_full_diagnosis(self, train_loader):
        previous_mode = self.model.training
        self.model.eval()
        try:
            return self._run_full_diagnosis(train_loader)
        finally:
            self.model.train(previous_mode)

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running bounded training-pair and semantic deletion diagnostics...")
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                ### >>> Self_EvolveRec-BLOCK-START: Report measured semantic controls and margin denominators
                "semantic_deletion": self.analyze_semantic_deletion(),
                "valid_pair_count": margin_metrics["valid_pair_count"],
                "nonfinite_pair_count": margin_metrics["nonfinite_pair_count"],
                "embedding_collapse_score": collapse_score,
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Define observational controls without unsupported collapse or causal claims
                "semantic_deletion": (
                    "Seed-42 bounded cohort from this view's training histories; last training interaction is the diagnostic target. "
                    "TF-IDF is fitted only on training-history item titles/descriptions. Targeted deletion removes the most isolated "
                    "text-observed older-half event, reversed deletion the least isolated; ties use earliest position. "
                    "Five random deletions use eligible events in the targeted event's relative-age quartile, with replacement. "
                    "All compacted views preserve the last context event, right-aligned readout, target and unique training-only "
                    "candidate support. Ranks are one-based; positive rank improvement means a lower target rank. "
                    "Hit values, overlap and rank displacement are measured on these diagnostic candidates, not benchmark candidates. "
                    "Eligible users count all inspected eligible histories; attempted users count the fixed capped cohort, "
                    "including failures and candidate-empty histories; evaluated users count finite scored users in that cohort. "
                    "Failed users are not replaced. Summary differences average paired users; history-length summaries use "
                    "the same differences within each reported length group. Empty groups are unavailable, not zero. "
                    "Results belong to the current dataset view; no four-category macro metric is inferred. "
                    "Position-preserving attention masking is unavailable, so targeting remains unresolved even if compacted removal helps. "
                    "Training weights and residual graphs may have seen the diagnostic targets; this is not held-out generalization or causality."
                ),
                "valid_pair_count": "Number of nonpadding, distinct positive-negative pairs with valid IDs and no malformed ID in their causal prefix, from at most ten loader batches.",
                "nonfinite_pair_count": "Evaluated margins that are nonfinite; all margin summaries are unavailable if this count is nonzero.",
                "embedding_collapse_score": (
                    "Historical metric name: (mean off-diagonal cosine + 1)/2 over at most 2048 seed-42 sampled nonpadding "
                    "item embeddings. Measures directional alignment only, not proof of collapse or recommendation diversity. "
                    "Unavailable for insufficient samples, nonfinite embeddings, or sampled zero-norm embeddings."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Specify percentile and training-score interpretation
                "pos_neg_margin_p10": "Linear-interpolated tenth percentile of valid training-pair margins; unavailable for an empty cohort or any nonfinite evaluated margin. Hybrid models use combined scores on the exact causal prefix; other models use the compatibility scoring hook.",
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Mark category breakdown unavailable under the matched data contract
                "pos_neg_margin_category_breakdown": (
                    "Item categories are unavailable. Category lists are empty because they were not observed. "
                    "Position totals count finite threshold comparisons on valid training pairs; eps=0.5 uses model score units. "
                    "Sampled negatives are unlabeled alternatives, not verified dislikes."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
