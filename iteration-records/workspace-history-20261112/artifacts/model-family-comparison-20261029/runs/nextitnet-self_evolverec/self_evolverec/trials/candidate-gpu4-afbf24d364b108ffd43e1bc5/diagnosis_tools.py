import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Dependencies for a frozen training-only candidate stress audit
from collections import Counter
from itertools import islice
import hashlib
import html
import os
import pickle
import re
import unicodedata

# DEBUG: Replace unavailable SciPy/sklearn dependencies without disabling the audit.
import math


class _SparseTextIndex:
    """Sparse normalized item vectors and an inverted index for cosine queries."""

    def __init__(self, rows, width):
        self.rows = rows
        self.shape = (len(rows), width)
        self.postings = {}
        for item_index, row in enumerate(rows):
            for feature, value in row.items():
                self.postings.setdefault(feature, []).append((item_index, value))

    def similarities(self, item_ids):
        # Repeated prefix occurrences intentionally contribute repeatedly.
        query = Counter()
        for item in item_ids:
            for feature, value in self.rows[int(item) - 1].items():
                query[feature] += value
        # DEBUG: Explicit length also supports NumPy item arrays without an
        # ambiguous-truth-value exception; list-based audit behavior is unchanged.
        if len(item_ids) > 0:
            query = {feature: value / len(item_ids) for feature, value in query.items()}
        norm = math.sqrt(sum(value * value for value in query.values()))
        similarities = np.zeros(self.shape[0], dtype=np.float64)
        if norm == 0.0:
            return similarities, True
        for feature, value in query.items():
            weight = value / norm
            for item_index, item_weight in self.postings.get(feature, ()):
                similarities[item_index] += weight * item_weight
        return similarities, False


def _fit_tfidf(texts, max_features=100000):
    """Word unigram/bigram TF-IDF using only standard-library sparse maps."""
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    documents = []
    document_frequency = Counter()
    corpus_frequency = Counter()
    for text in texts:
        tokens = token_pattern.findall(text)
        terms = tokens + [
            first + " " + second for first, second in zip(tokens, tokens[1:])
        ]
        counts = Counter(terms)
        documents.append(counts)
        document_frequency.update(counts.keys())
        corpus_frequency.update(counts)
    # Select by corpus term frequency with an explicit deterministic tie rule.
    selected = sorted(
        corpus_frequency, key=lambda term: (-corpus_frequency[term], term)
    )[:max_features]
    vocabulary = {term: index for index, term in enumerate(sorted(selected))}
    idf = [0.0] * len(vocabulary)
    for term, index in vocabulary.items():
        idf[index] = (
            math.log((1.0 + len(texts)) / (1.0 + document_frequency[term])) + 1.0
        )
    rows = []
    for counts in documents:
        row = {
            vocabulary[term]: (1.0 + math.log(count)) * idf[vocabulary[term]]
            for term, count in counts.items()
            if term in vocabulary
        }
        norm = math.sqrt(sum(value * value for value in row.values()))
        if norm:
            row = {index: value / norm for index, value in row.items()}
        rows.append(row)
    fitted = {
        "implementation": "stdlib_sparse_word_tfidf_v1",
        "vocabulary": vocabulary,
        "idf": idf,
        "parameters": {
            "analyzer": "word",
            "ngram_range": [1, 2],
            "token_pattern": token_pattern.pattern,
            "stop_words": None,
            "min_df": 1,
            "max_df": 1.0,
            "max_features": max_features,
            "binary": False,
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True,
            "norm": "l2",
            "feature_selection_ties": "term ascending",
        },
    }
    return rows, fitted


def _uniform_sampled_hit10(population, successes, draws=99):
    """Exact Hypergeom(population, successes, draws).cdf(9)."""
    if not (0 <= successes <= population and 0 <= draws <= population):
        raise ValueError("invalid hypergeometric population or sample size")
    lower = max(0, draws - (population - successes))
    upper = min(9, successes, draws)
    if lower > upper:
        return 0.0
    numerator = sum(
        math.comb(successes, count) * math.comb(population - successes, draws - count)
        for count in range(lower, upper + 1)
    )
    # Python integers preserve the combinatorial sum without overflow;
    # division produces the final bounded floating-point probability.
    return numerator / math.comb(population, draws)


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
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    ### >>> Self_EvolveRec-BLOCK-START: Interpret training audits with explicit evidence boundaries
    - Report support, exclusions, selection configuration and missingness with findings.
    - Null means unavailable, never zero. No measurements means untested.
    - Sampled training ranks, pairwise margins and catalog ranks are distinct estimands; none is pinned validation HR@10.
    - Training targets may be memorized. Later training positives annotate candidate ambiguity; unobserved items are not verified negatives.
    - target_lexical_oracle uses the target identity and is not deployable retrieval or evidence of history understanding.
    - Prefix lexical competition can disadvantage interest shifts. Candidate sensitivity does not establish user intent or causal importance.
    - Legacy lexical draws differ only in fallback items. Matched lexical sets are fixed; repeated retrieval is not independent evidence.
    - Interpret matched lexical-minus-random gaps only on their reported common support; compare missingness and metadata coverage before generalizing.
    - Exact frequency matching controls the declared retrospective count distribution, not text, semantics, exposure, relevance or causal confounding.
    - Cross-arm overlap is allowed and can attenuate contrast. Zero-variance balance statistics are explicitly undefined, not infinite or automatically balanced.
    - Uniform hypergeometric probabilities are forward sampling calibration conditional on the fixed scores and pool, not catalog-rank inversion or relevance estimates.
    - Calibration is selected independently of matching success; report chunk-size score checks, ordering changes, near ties and finite exclusions.
    - Raw margins depend on score scale. A mapped cosine near 0.5 means mean cosine near zero, not a collapse probability.
    - Bootstrap intervals condition on the fitted checkpoint, selected users, fixed candidates, catalog and metadata. They omit training-seed uncertainty and shared-item dependence.
    - Never infer benchmark performance, deployment parity, unavailable user traits, ratings, reviews or item categories.
    - Distinguish in-memory checks from generated-predictor reload checks; unavailable deployment parity remains unverified.
    - A four-category macro result requires support from all four categories; never silently omit a missing category.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family-specific ModelView contract and public-data boundary
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Categories are unavailable in this benchmark; never infer them.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and report undefined embedding measurements honestly
            emb = emb[1 : self.network.itemnum + 1]

            n = emb.size(0)
            if min(sample_size, n) < 2 or not torch.isfinite(emb).all():
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Compute mean off-diagonal cosine without a quadratic Gram matrix
            x = F.normalize(emb.index_select(0, idx).float(), p=2, dim=-1)
            offdiag_sum = x.sum(0).square().sum() - x.square().sum()
            ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Keep mode ownership in the diagnosis wrapper and track finite support
        margins_all = []
        valid_total = nonfinite_total = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption without fetching an extra training batch
        for batch in islice(iter(dataloader), max(0, num_batches)):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Use tensor inputs for the corrected NextItNet compatibility path
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude missing targets and nonfinite pairs consistently
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            finite = torch.isfinite(all_logits).all(-1) & torch.isfinite(margin)
            valid_total += int(valid.sum().item())
            nonfinite_total += int((valid & ~finite).sum().item())
            valid &= finite
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

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
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Represent empty margin summaries as unavailable
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                ### <<< Self_EvolveRec-BLOCK-END
            }
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

        ### >>> Self_EvolveRec-BLOCK-START: Expose margin denominators and unavailable category annotations
        margin_metrics.update(
            {
                "valid_positions": valid_total,
                "comparisons": valid_total - nonfinite_total,
                "excluded_nonfinite": nonfinite_total,
                "distinct_users": None,
            }
        )
        breakdown = {
            "availability": "unavailable: item categories are not provided",
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

    ### >>> Self_EvolveRec-BLOCK-START: Frozen frequency-matched lexical audit with independent exact-pool calibration
    @staticmethod
    def _digest(value):
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _clean_text(value):
        if isinstance(value, (list, tuple)):
            value = " ".join(x for x in value if isinstance(x, str))
        if not isinstance(value, str):
            return ""
        value = re.sub(r"<[^>]*>", " ", html.unescape(value))
        return " ".join(unicodedata.normalize("NFKC", value).lower().split())

    @staticmethod
    def _write_json(path, value):
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(
                value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        os.replace(temporary, path)

    def _text_index(self, itemnum, counts):
        metadata = self.meta_data if isinstance(self.meta_data, dict) else {}
        texts, coverage = [], Counter()
        for item in range(1, itemnum + 1):
            info = metadata.get(str(item), metadata.get(item, {}))
            info = info if isinstance(info, dict) else {}
            title = self._clean_text(info.get("title", ""))
            description = self._clean_text(info.get("description", ""))
            texts.append((title + " " + description).strip())
            coverage["items_with_title"] += bool(title)
            coverage["items_with_description"] += bool(description)
        # Fit only training-observed catalog documents. Other rows stay zero.
        fitted_ids = np.flatnonzero(counts > 0)
        fitted_rows, fitted = _fit_tfidf([texts[i] for i in fitted_ids])
        rows = [{} for _ in range(itemnum)]
        for item_index, row in zip(fitted_ids, fitted_rows):
            rows[int(item_index)] = row
        coverage.update(
            {
                "catalog_items": itemnum,
                "fitted_documents": int(fitted_ids.size),
                "nonzero_vector_items": sum(bool(row) for row in rows),
                "vocabulary_size": len(fitted["vocabulary"]),
            }
        )
        coverage = dict(coverage)
        coverage["item_zero_vector_rate"] = (
            sum(not row for row in rows) / itemnum if itemnum else None
        )
        coverage["empty_vocabulary"] = not bool(fitted["vocabulary"])
        fitted["document_item_ids"] = (fitted_ids + 1).tolist()
        fitted["cleaned_character_lengths"] = [len(text) for text in texts]
        return (
            _SparseTextIndex(rows, len(fitted["vocabulary"])),
            fitted,
            coverage,
            self._digest(texts),
        )

    @staticmethod
    def _mean_user_summary(records, conditions):
        # Each record already averages the five frozen draws of one prefix.
        users = {}
        for record in records:
            users.setdefault(record["user_id"], []).append(record["values"])
        names = (
            "rank",
            "hit10",
            "reciprocal_rank",
            "max_competitor_margin",
            "known_later_positive_fraction",
            "fallback_fraction",
        )
        arrays = []
        for user in sorted(users):
            arrays.append(np.asarray(users[user], dtype=np.float64).mean(axis=0))
        values = (
            np.stack(arrays) if arrays else np.empty((0, len(conditions), len(names)))
        )
        means = values.mean(axis=0) if len(values) else None
        summary = {
            "users": len(users),
            "prefixes": len(records),
            "conditions": {
                condition: {
                    name: None if means is None else float(means[c, j])
                    for j, name in enumerate(names)
                }
                for c, condition in enumerate(conditions)
            },
        }
        return summary, values

    @torch.no_grad()
    def _candidate_audit(self):
        import time

        started = time.perf_counter()
        conditions = (
            "uniform",
            "frequency_075",
            "prefix_lexical",
            "target_lexical_oracle",
        )
        matched_conditions = ("prefix_lexical_fixed99", "frequency_matched_random")
        category = getattr(self.model, "dataset_name", None)
        if category is None:
            category = getattr(self.network, "dataset_name", None)
        category = str(category) if category is not None else None
        itemnum, maxlen = int(self.network.itemnum), int(self.network.maxlen)
        raw = self.user_train
        histories = raw.get("History", raw) if isinstance(raw, dict) else None
        if not isinstance(histories, dict):
            return {
                "available": False,
                "reason": "stable mapped training histories unavailable",
            }
        if itemnum < 1 or maxlen < 1:
            raise ValueError("audit requires positive itemnum and maxlen")
        canonical, exclusions = {}, Counter()
        for user, history in histories.items():
            key = str(user)
            if key in canonical:
                raise ValueError("training user IDs collide after string conversion")
            canonical[key] = [int(item) for item in history]
        counts = np.zeros(itemnum, dtype=np.int64)
        for history in canonical.values():
            for item in history:
                if 1 <= item <= itemnum:
                    counts[item - 1] += 1
        eligible = []
        for user in sorted(canonical):
            history = canonical[user]
            if any(item < 1 or item > itemnum for item in history):
                exclusions["users_invalid_mapped_items"] += 1
            elif len(history) < 2:
                exclusions["users_short_history"] += 1
            else:
                eligible.append(user)

        def local_rng(user, position, purpose, draw=0):
            digest = self._digest(
                [42, category, user, int(position), purpose, int(draw)]
            )
            return np.random.default_rng(int(digest, 16))

        rng = np.random.default_rng(42)
        selected = sorted(
            rng.choice(len(eligible), min(200, len(eligible)), replace=False).tolist()
        )
        catalog = np.arange(1, itemnum + 1, dtype=np.int64)
        manifest, last_selected = [], {}
        for user_index in selected:
            user, history = eligible[user_index], canonical[eligible[user_index]]
            positions = sorted(
                local_rng(user, 0, "prefix_selection")
                .choice(
                    np.arange(1, len(history)), min(5, len(history) - 1), replace=False
                )
                .tolist()
            )
            for position in positions:
                prefix = history[max(0, position - maxlen) : position]
                record = {
                    "user_id": user,
                    "target_position": position,
                    "original_prefix_length": position,
                    "prefix": prefix,
                    "target": history[position],
                }
                record["input_sha256"] = self._digest(
                    [0] * (maxlen - len(prefix)) + prefix
                )
                manifest.append(record)
                last_selected[user] = len(manifest) - 1
        # Last selected prefix for the first 40 frozen users, before text/support checks.
        subset = {last_selected[user] for user in sorted(last_selected)[:40]}
        text, fitted, coverage, metadata_hash = self._text_index(itemnum, counts)
        metadata_ok = np.asarray([bool(row) for row in text.rows])
        text_lengths = np.asarray(fitted["cleaned_character_lengths"], dtype=np.float64)
        config = {
            "version": "training_candidate_audit_v3",
            "category": category,
            "category_identity_available": category is not None,
            "unknown_category_seed_policy": "explicit null category in seed payload; no inferred identity",
            "max_users": 200,
            "max_prefixes_per_user": 5,
            "max_catalog_users": 40,
            "draws": 5,
            "negatives": 99,
            "calibration_draws": 100,
            "maxlen": maxlen,
            "chunk_sizes_checked": [4096, 1024],
            "rtol": 1e-4,
            "atol": 1e-5,
            "bootstrap_replicates": 1000,
            "selection_seed": 42,
            "bootstrap_seed": 42,
            "seed_scheme": "SHA256 canonical JSON [42,category,user_id,target_position,purpose,draw]; local NumPy Generator",
            "conditions": list(conditions),
            "matched_conditions": list(matched_conditions),
            "pool": "catalog minus visible truncated prefix IDs and target; padding excluded",
            "matched_pool": "same exclusions, restricted to nonzero training-fitted text vectors",
            "matching": "random complete exact-count bucket matching, without replacement; cross-arm overlap allowed",
            "matching_caliper": 0.1,
            "caliper_used": False,
            "exact_matching_feasibility": "lexical IDs themselves belong to the control pool, so every exact-count bucket has sufficient capacity; no caliper relaxation is needed",
            "counts": "all permitted training occurrences, repeated events included; retrospective, including events later than audited prefixes",
            "tie_order": "score descending then mapped ID ascending; host tie equivalence unverified",
            "lexical_fill": "legacy lexical arms use uniform fill; matched comparison requires 99 strictly positive similarities and never fills",
            "lexical_backend": fitted["implementation"],
            "text": "concatenated title and description; HTML unescape/tag removal, NFKC, lowercase, whitespace collapse; accents retained; no extra truncation",
            "tfidf_fit": "training-observed catalog documents only; actual parameters and IDFs in tfidf.pkl",
            "calibration_selection": "first up to 40 selected users in canonical order; last selected prefix per user; independent of matching and neural scores",
            "hypergeometric_backend": "stdlib exact integer combinations",
            "numpy": np.__version__,
            "torch": torch.__version__,
            "metadata_hash": metadata_hash,
            "history_hash": self._digest(canonical),
            "mapping": "ModelView histories and metadata must already use mapped IDs",
        }
        checkpoint_hash = hashlib.sha256()
        for name, tensor in sorted(self.network.state_dict().items()):
            tensor = tensor.detach().cpu().contiguous()
            checkpoint_hash.update(name.encode())
            checkpoint_hash.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            checkpoint_hash.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        config["model_state_sha256"] = checkpoint_hash.hexdigest()

        def balance(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            va, vb = float(a.var()), float(b.var())
            scale = math.sqrt((va + vb) / 2)
            grid = np.unique(np.concatenate((a, b)))
            return {
                "means": [float(a.mean()), float(b.mean())],
                "standard_deviations": [math.sqrt(va), math.sqrt(vb)],
                "standardized_mean_difference": (
                    float((a.mean() - b.mean()) / scale) if scale else None
                ),
                "variance_ratio": va / vb if vb else None,
                "zero_variance": [va == 0, vb == 0],
                "quantile_probabilities": [0, 0.1, 0.5, 0.9, 1],
                "quantiles": [
                    np.quantile(x, [0, 0.1, 0.5, 0.9, 1]).tolist() for x in (a, b)
                ],
                "ecdf_distance": float(
                    np.max(
                        np.abs(
                            np.searchsorted(np.sort(a), grid, side="right") / len(a)
                            - np.searchsorted(np.sort(b), grid, side="right") / len(b)
                        )
                    )
                ),
            }

        frozen = []
        for record in manifest:
            user, position = record["user_id"], record["target_position"]
            pool = catalog[
                ~np.isin(catalog, list(set(record["prefix"]) | {record["target"]}))
            ]
            entry = {"pool_size": int(pool.size), "legacy": None, "matched": None}
            if pool.size < 99:
                exclusions["prefixes_fewer_than_99_competitors"] += 1
                entry["matched_status"] = "insufficient_full_pool"
                frozen.append(entry)
                continue
            similarities, zero = text.similarities(record["prefix"])
            oracle, oracle_zero = text.similarities([record["target"]])
            metadata_pool = pool[metadata_ok[pool - 1]]
            positive = metadata_pool[similarities[metadata_pool - 1] > 0]
            top = positive[np.lexsort((positive, -similarities[positive - 1]))][:99]
            entry.update(
                {
                    "metadata_pool_size": int(metadata_pool.size),
                    "positive_lexical_count": int(positive.size),
                    "zero_query": bool(zero),
                    "oracle_zero_query": bool(oracle_zero),
                    "target_metadata_available": bool(
                        metadata_ok[record["target"] - 1]
                    ),
                    "prefix_metadata_occurrence_fraction": float(
                        metadata_ok[np.asarray(record["prefix"]) - 1].mean()
                    ),
                }
            )
            legacy = []
            for condition in conditions:
                draws = []
                for draw in range(5):
                    local = local_rng(user, position, condition, draw)
                    fallback = 0
                    if condition == "uniform":
                        ids = local.choice(pool, 99, replace=False)
                    elif condition == "frequency_075":
                        weights = counts[pool - 1].astype(float) ** 0.75
                        supported = np.flatnonzero(weights > 0)
                        clocks = (
                            local.standard_exponential(supported.size)
                            / weights[supported]
                        )
                        order = np.lexsort((pool[supported], clocks))[:99]
                        ids = pool[supported[order]]
                        fallback = 99 - len(ids)
                        if fallback:
                            ids = np.concatenate(
                                (
                                    ids,
                                    local.choice(
                                        pool[weights == 0], fallback, replace=False
                                    ),
                                )
                            )
                    else:
                        sim = (
                            oracle
                            if condition == "target_lexical_oracle"
                            else similarities
                        )
                        candidates = pool[sim[pool - 1] > 0]
                        ids = candidates[
                            np.lexsort((candidates, -sim[candidates - 1]))
                        ][:99]
                        fallback = 99 - len(ids)
                        ids = np.concatenate(
                            (
                                ids,
                                local.choice(
                                    pool[~np.isin(pool, ids)], fallback, replace=False
                                ),
                            )
                        )
                    draws.append(
                        {"negative_ids": ids.tolist(), "fallback_count": fallback}
                    )
                legacy.append(draws)
            entry["legacy"] = legacy
            if zero or len(top) != 99:
                entry["matched_status"] = (
                    "zero_query" if zero else "fewer_than_99_positive_cosine_items"
                )
                exclusions["matched_" + entry["matched_status"]] += 1
            else:
                # Equal-count edges form disjoint complete bipartite graphs.
                # Random item priorities plus random slot order give a complete
                # randomized maximum matching without a catalog-sized edge list.
                controls, balances = [], []
                groups = {
                    int(c): metadata_pool[counts[metadata_pool - 1] == c]
                    for c in np.unique(counts[top - 1])
                }
                for draw in range(5):
                    local = local_rng(user, position, "exact_frequency_matching", draw)
                    control = np.empty(99, dtype=np.int64)
                    for frequency, candidates in groups.items():
                        slots = np.flatnonzero(counts[top - 1] == frequency)
                        if candidates.size < slots.size:
                            raise RuntimeError(
                                "exact matching capacity invariant violated"
                            )
                        control[local.permutation(slots)] = local.permutation(
                            candidates
                        )[: slots.size]
                    if np.unique(control).size != 99 or not np.array_equal(
                        counts[top - 1], counts[control - 1]
                    ):
                        raise RuntimeError("invalid exact frequency matching")
                    intersection = int(np.intersect1d(top, control).size)
                    balances.append(
                        {
                            "raw_frequency": balance(
                                counts[top - 1], counts[control - 1]
                            ),
                            "log1p_frequency": balance(
                                np.log1p(counts[top - 1]), np.log1p(counts[control - 1])
                            ),
                            "text_length": balance(
                                text_lengths[top - 1], text_lengths[control - 1]
                            ),
                            "metadata_coverage": balance(
                                metadata_ok[top - 1], metadata_ok[control - 1]
                            ),
                            "prefix_cosine": balance(
                                similarities[top - 1], similarities[control - 1]
                            ),
                            "intersection_size": intersection,
                            "jaccard": intersection / (198 - intersection),
                        }
                    )
                    controls.append(control.tolist())
                entry.update(
                    {
                        "matched_status": "complete_exact",
                        "matched": [top.tolist(), controls],
                        "balance": balances,
                    }
                )
            frozen.append(entry)
        # Freeze calibration candidates before any neural scoring, independently
        # of lexical support, matching success and finite-score exclusions.
        for index in sorted(subset):
            record = manifest[index]
            pool = catalog[
                ~np.isin(catalog, list(set(record["prefix"]) | {record["target"]}))
            ]
            frozen[index]["calibration_negative_ids"] = (
                [
                    local_rng(
                        record["user_id"],
                        record["target_position"],
                        "calibration_uniform",
                        draw,
                    )
                    .choice(pool, 99, replace=False)
                    .tolist()
                    for draw in range(100)
                ]
                if pool.size >= 99
                else []
            )
        config["manifest_sha256"] = self._digest(manifest)
        config["candidates_sha256"] = self._digest(frozen)
        config["counts_sha256"] = self._digest(counts.tolist())
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "working",
            "diagnosis_audit",
            self._digest(config),
        )
        os.makedirs(directory, exist_ok=True)
        self._write_json(
            os.path.join(directory, "frozen.json"),
            {
                "configuration": config,
                "manifest": manifest,
                "candidates": frozen,
                "catalog_subset_indices": sorted(subset),
                "training_counts": counts.tolist(),
                "metadata_coverage": coverage,
            },
        )
        with open(os.path.join(directory, "tfidf.pkl"), "wb") as stream:
            pickle.dump(fitted, stream, protocol=pickle.HIGHEST_PROTOCOL)
        records, matched_records, calibration = [], [], []

        def score(feature, ids, chunk_size=4096):
            ids = np.asarray(ids, dtype=np.int64).reshape(-1)
            # Score each unique item once per comparison. This saves repeated
            # target/overlap work and gives identical IDs identical cached scores.
            unique_ids, inverse = np.unique(ids, return_inverse=True)
            parts = []
            for start in range(0, unique_ids.size, chunk_size):
                items = torch.as_tensor(
                    unique_ids[start : start + chunk_size],
                    device=self.device,
                    dtype=torch.long,
                )
                result = self.model.predict_score(feature, items)
                if result.numel() != items.numel():
                    raise ValueError(
                        "shared-candidate scoring contract returned an unexpected shape"
                    )
                parts.append(result.detach().float().cpu().numpy().reshape(-1))
            unique_scores = (
                np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
            )
            return unique_scores[inverse]

        def evaluate_draws(feature, record, draws, fallback):
            negatives = np.asarray(draws, dtype=np.int64)
            ids = np.concatenate(
                (
                    np.full(
                        (*negatives.shape[:-1], 1), record["target"], dtype=np.int64
                    ),
                    negatives,
                ),
                axis=-1,
            )
            scores = score(feature, ids).reshape(ids.shape)
            if not np.isfinite(scores).all():
                return None
            margins = scores[..., 0] - scores[..., 1:].max(-1)
            if not np.isfinite(margins).all():
                return None
            target = record["target"]
            ranks = 1 + (
                (scores[..., 1:] > scores[..., :1])
                | ((scores[..., 1:] == scores[..., :1]) & (negatives < target))
            ).sum(-1)
            later = canonical[record["user_id"]][record["target_position"] + 1 :]
            conflict = np.isin(negatives, later).mean(-1)
            values = np.stack(
                (ranks, ranks <= 10, 1 / ranks, margins, conflict, fallback), -1
            )
            return {
                "user_id": record["user_id"],
                "target_position": record["target_position"],
                "values": values.mean(1).tolist(),
                "draw_values": values.tolist(),
            }

        for index, record in enumerate(manifest):
            entry = frozen[index]
            if entry["legacy"] is None and index not in subset:
                continue
            sequence = torch.zeros((1, maxlen), dtype=torch.long, device=self.device)
            sequence[0, -len(record["prefix"]) :] = torch.as_tensor(
                record["prefix"], device=self.device
            )
            features = self.model.log2feats(sequence)
            if features.ndim != 3 or features.shape[:2] != sequence.shape:
                raise ValueError("log2feats must return B x L x D features")
            last = (
                torch.arange(maxlen, device=self.device)
                .masked_fill(sequence[0].eq(0), -1)
                .amax()
            )
            if int(last) < 0:
                exclusions["all_padding_prefixes"] += 1
                continue
            # Preserve B x D for ModelView implementations requiring a batch axis.
            feature = features[:, int(last), :]
            if entry["legacy"] is not None:
                draws = [[d["negative_ids"] for d in arm] for arm in entry["legacy"]]
                fallback = np.asarray(
                    [[d["fallback_count"] / 99 for d in arm] for arm in entry["legacy"]]
                )
                output = evaluate_draws(feature, record, draws, fallback)
                if output is None:
                    exclusions["nonfinite_legacy_common_support"] += 1
                else:
                    output["manifest_index"] = index
                    records.append(output)
            if entry["matched"] is not None:
                lexical, controls = entry["matched"]
                output = evaluate_draws(
                    feature, record, [[lexical] * 5, controls], np.zeros((2, 5))
                )
                if output is None:
                    exclusions["nonfinite_matched_common_support"] += 1
                else:
                    output.update(
                        {"manifest_index": index, "balance": entry["balance"]}
                    )
                    matched_records.append(output)
            # Calibration deliberately does not depend on lexical/matching success
            # or finite scores in any sampled arm.
            if index not in subset:
                continue
            pool = catalog[
                ~np.isin(catalog, list(set(record["prefix"]) | {record["target"]}))
            ]
            if pool.size < 99:
                exclusions["calibration_fewer_than_99_competitors"] += 1
                continue
            universe = np.sort(np.append(pool, record["target"]))
            scores = score(feature, universe)
            alternate = score(feature, universe, chunk_size=1024)
            if not np.isfinite(scores).all() or not np.isfinite(alternate).all():
                exclusions["nonfinite_calibration"] += 1
                continue
            target_index = int(np.searchsorted(universe, record["target"]))
            target_score = scores[target_index]
            better = (scores > target_score) | (
                (scores == target_score) & (universe < record["target"])
            )
            alternate_target = alternate[target_index]
            alternate_better = (alternate > alternate_target) | (
                (alternate == alternate_target) & (universe < record["target"])
            )
            stable = bool(np.allclose(scores, alternate, rtol=1e-4, atol=1e-5))
            ordering_changes = int(np.count_nonzero(better != alternate_better))
            calibration_record = {
                "user_id": record["user_id"],
                "manifest_index": index,
                "pool_size": int(pool.size),
                "chunk_size_scores_close": stable,
                "chunk_size_max_abs_error": float(np.max(np.abs(scores - alternate))),
                "chunk_size_target_order_changes": ordering_changes,
                "near_target_tolerance_items": int(
                    np.count_nonzero(
                        np.isclose(
                            scores[np.searchsorted(universe, pool)],
                            target_score,
                            rtol=1e-4,
                            atol=1e-5,
                        )
                    )
                ),
                "available": stable and ordering_changes == 0,
            }
            if not calibration_record["available"]:
                exclusions["calibration_numerically_unstable"] += 1
                calibration.append(calibration_record)
                continue
            outrankers = int(better.sum())
            probability = _uniform_sampled_hit10(int(pool.size), outrankers, 99)
            sampled_ids = entry["calibration_negative_ids"]
            if len(sampled_ids) != 100:
                raise RuntimeError("missing prespecified uniform calibration draws")
            hits = [
                int(better[np.searchsorted(universe, ids)].sum() <= 9)
                for ids in sampled_ids
            ]
            empirical = float(np.mean(hits))
            calibration_record.update(
                {
                    "outrankers": outrankers,
                    "exact_rank": outrankers + 1,
                    "catalog_hit10": outrankers <= 9,
                    "analytic_uniform_hit10": probability,
                    "empirical_uniform_hit10": empirical,
                    "calibration_residual": empirical - probability,
                    "conditional_mc_standard_error": math.sqrt(
                        probability * (1 - probability) / 100
                    ),
                    "draw_hits": hits,
                    "candidate_ids_sha256": self._digest(sampled_ids),
                }
            )
            calibration.append(calibration_record)

        def summarize(selected_records, names):
            summary, users = self._mean_user_summary(selected_records, names)
            intervals, variability = {}, {}
            if len(users):
                bootstrap = np.random.default_rng(42)
                differences = users[:, 1:] - users[:, :1]
                estimates = np.asarray(
                    [
                        differences[
                            bootstrap.integers(len(users), size=len(users))
                        ].mean(0)
                        for _ in range(1000)
                    ]
                )
                for arm, name in enumerate(names[1:]):
                    intervals[name + "_minus_" + names[0]] = {
                        metric: {
                            "estimate": float(differences[:, arm, j].mean()),
                            "percentile_95_interval": np.quantile(
                                estimates[:, arm, j], [0.025, 0.975]
                            ).tolist(),
                        }
                        for j, metric in enumerate(
                            (
                                "rank",
                                "hit10",
                                "reciprocal_rank",
                                "max_competitor_margin",
                            )
                        )
                    }
                # Average draw variability within prefix, prefixes within user,
                # then users; distinct from the conditional user bootstrap.
                for arm, name in enumerate(names):
                    by_user = {}
                    for row in selected_records:
                        by_user.setdefault(row["user_id"], []).append(
                            np.asarray(row["draw_values"])[arm, :, :4].std(0)
                        )
                    variability[name] = (
                        np.stack([np.mean(v, axis=0) for v in by_user.values()])
                        .mean(0)
                        .tolist()
                    )
            return {
                "summary": summary,
                "paired_user_bootstrap": {
                    "replicates": 1000,
                    "seed": 42,
                    "differences": intervals,
                },
                "mean_within_prefix_draw_sd_rank_hit10_rr_margin": variability,
            }

        matched_summary = summarize(matched_records, matched_conditions)
        balance_summary = {}
        for field in (
            "raw_frequency",
            "log1p_frequency",
            "text_length",
            "metadata_coverage",
            "prefix_cosine",
        ):
            values = [r["balance"][d][field] for r in matched_records for d in range(5)]
            balance_summary[field] = {
                "paired_draws": len(values),
                "maximum_ecdf_distance": max(
                    (x["ecdf_distance"] for x in values), default=None
                ),
                "zero_variance_arm_counts": [
                    sum(x["zero_variance"][a] for x in values) for a in (0, 1)
                ],
            }
        # Exact frequency matching makes both frequency ECDFs identical for
        # every retained draw. Export those curves without plotting dependencies.
        frequency_ecdf = Counter()
        for row in matched_records:
            frequency_ecdf.update(
                counts[
                    np.asarray(frozen[row["manifest_index"]]["matched"][0]) - 1
                ].tolist()
            )
        if frequency_ecdf:
            xs = sorted(frequency_ecdf)
            total, cumulative, points = sum(frequency_ecdf.values()), 0, []
            for j, frequency in enumerate(xs):
                x = 20 + 460 * j / max(1, len(xs) - 1)
                points.append(f"{x:.3f},{280 - 260 * cumulative / total:.3f}")
                cumulative += frequency_ecdf[frequency]
                points.append(f"{x:.3f},{280 - 260 * cumulative / total:.3f}")
            with open(os.path.join(directory, "frequency_ecdf.svg"), "w") as stream:
                stream.write(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="320"><rect width="520" height="320" fill="white"/><text x="10" y="15">Exact-matched frequency ECDFs coincide; x: ordered count groups</text><polyline fill="none" stroke="blue" points="'
                    + " ".join(points)
                    + '"/></svg>'
                )
            self._write_json(
                os.path.join(directory, "frequency_ecdf.json"),
                {
                    "frequencies": xs,
                    "counts": [frequency_ecdf[x] for x in xs],
                    "weighting": "pooled lexical slots over supported prefixes; each exact-matched control has identical frequency ECDF",
                },
            )
        valid_calibration = [r for r in calibration if r["available"]]
        result = {
            "available": bool(records or matched_records or valid_calibration),
            "configuration": config,
            "legacy_comparison": summarize(records, conditions),
            "legacy_on_matched_common_support": summarize(
                [
                    row
                    for row in records
                    if row["manifest_index"]
                    in {matched["manifest_index"] for matched in matched_records}
                ],
                conditions,
            ),
            "matched_comparison": matched_summary,
            "metadata_coverage": coverage,
            "balance_summary": balance_summary,
            "support": {
                "eligible_training_users": len(eligible),
                "selected_users": len(selected),
                "selected_prefixes": len(manifest),
                "legacy_scored_prefixes": len(records),
                "matched_constructed_prefixes": sum(
                    x["matched"] is not None for x in frozen
                ),
                "matched_scored_prefixes": len(matched_records),
                "matched_status_counts": dict(
                    Counter(x["matched_status"] for x in frozen)
                ),
                "catalog_prespecified_examples": len(subset),
                "catalog_attempted_examples": len(calibration),
                "catalog_scored_examples": len(valid_calibration),
                "exclusions": dict(exclusions),
            },
            "calibration": {
                "examples": len(valid_calibration),
                "mean_residual": (
                    float(
                        np.mean([r["calibration_residual"] for r in valid_calibration])
                    )
                    if valid_calibration
                    else None
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "budget_truncation": {
                "occurred": False,
                "unexecuted_prespecified_examples": [],
            },
            "artifact_directory": directory,
            "four_category_macro": {
                "available": False,
                "reason": "single-category ModelView; no authorized all-category aggregation interface",
            },
        }
        self._write_json(
            os.path.join(directory, "results.json"),
            {
                "summary": result,
                "prefix_results": records,
                "matched_prefix_results": matched_records,
                "catalog_results": calibration,
            },
        )
        return result

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            self.network.eval()
            with torch.random.fork_rng(devices=devices):
                # Loader implementations may use global RNGs; restore those too.
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(42)
                batches = []
                for batch in islice(iter(train_loader), 4):
                    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                        raise ValueError(
                            "expected (None, sequence, positive, negative)"
                        )
                    tensors = [
                        torch.as_tensor(x, device=self.device, dtype=torch.long)[
                            :32
                        ].clone()
                        for x in batch[1:]
                    ]
                    if any(x.ndim != 2 or x.shape != tensors[0].shape for x in tensors):
                        raise ValueError(
                            "diagnostic tensors must have matching B x L shapes"
                        )
                    if tensors[0].size(0) and tensors[0].size(1):
                        batches.append((None, *tensors))
                collapse = self.analyze_embeddings_collapse_cosine()
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=4, eps=0.5
                    )
                )
                audit = self._candidate_audit()
                embedding = self.model.item_emb.weight.detach()[
                    1 : self.network.itemnum + 1
                ]
                metrics = {
                    "embedding_collapse_score": collapse,
                    "embedding_support": {
                        "catalog_items": embedding.size(0),
                        "sampled_items": (
                            min(2048, embedding.size(0)) if collapse is not None else 0
                        ),
                        "nonfinite_items": int(
                            (~torch.isfinite(embedding).all(-1)).sum().item()
                        ),
                    },
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "all_position_support": {
                        key: margins[key]
                        for key in (
                            "valid_positions",
                            "comparisons",
                            "excluded_nonfinite",
                            "distinct_users",
                        )
                    },
                    "pos_neg_margin_category_breakdown": breakdown,
                    "training_candidate_stress_audit": audit,
                    "deployment_reload_parity": {
                        "available": False,
                        "reason": "ModelView supplies no generated-predictor reload interface",
                    },
                    "validation_rank_metrics": {
                        "available": False,
                        "reason": "training-only diagnostic; pinned validation candidates unavailable",
                    },
                }
                definitions = {
                    "embedding_collapse_score": "(mean sampled off-diagonal item cosine + 1)/2; 0.5 means mean cosine zero. Not collapse probability or accuracy. Null when undefined/nonfinite.",
                    "embedding_support": "Mapped catalog items excluding padding, sampled items and nonfinite catalog embeddings.",
                    "pos_neg_margin_mean": "Mean positive-minus-negative score over finite nonpadding positions with both targets nonzero; scale dependent.",
                    "pos_neg_margin_p10": "Nearest-rank tenth percentile on the same supported pairs.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of supported margins strictly below zero; not HR@10.",
                    "pos_neg_small_margin_rate": "Fraction of supported margins below 0.5; scale dependent.",
                    "all_position_support": "Position comparisons and finite exclusions from at most four batches and 32 rows/batch; stable user identities unavailable for these loader rows.",
                    "pos_neg_margin_category_breakdown": "Category annotations unavailable; measured low/wrong position totals retained, category lists empty.",
                    "training_candidate_stress_audit": (
                        "Frozen training-target diagnostic, never pinned validation HR@10. "
                        "Configuration, histories, model state, inputs, candidates and counts are hashed; frozen.json precedes all neural scoring. "
                        "At most 200 canonical mapped users and five causal prefixes/user; inputs use baseline truncation and left padding. "
                        "All pools exclude padding, target and visible-prefix IDs. Later training positives remain competitors and are overlap annotations, not verified false negatives. "
                        "Counts include all training occurrences, including repeated events and events later than audited prefixes; this is retrospective, not prospective exposure. "
                        "Legacy arms retain uniform, frequency**0.75 exponential-race sampling and prefix/target lexical retrieval with declared uniform fill. Target retrieval is explicitly an oracle. "
                        "Matched arms share the nonzero-vector metadata pool and require 99 strictly positive prefix-cosine competitors without fill. "
                        "Controls use randomized complete exact-frequency bucket matching. Cross-arm overlap, including self matches, is permitted; within-arm IDs are unique. "
                        "Exact matching always exists because lexical candidates belong to the same control pool; the declared 0.1 log-frequency caliper is never needed or widened. "
                        "Queries normalize the mean of available training-fitted title-plus-description vectors; repeated prefix occurrences retain multiplicity. "
                        "Actual sparse TF-IDF preprocessing, vocabulary, IDFs and parameters are serialized; empty text remains unavailable. "
                        "Each comparison separately enforces finite common support across its arms and draws. legacy_on_matched_common_support additionally restricts legacy summaries to the intersection of finite legacy and matched prefixes; its user/prefix support is reported separately. Draw means precede prefix means within users and equal user means within category. "
                        "Rank orders descending score then ascending ID; hit10=1[rank<=10], reciprocal_rank=1/rank, margin=target minus strongest competitor. Fallback is the filled-slot fraction. "
                        "Draw values and later-positive fractions are saved. Candidate-draw SD is averaged within prefix/user/category and is separate from user uncertainty. "
                        "Paired differences subtract the first named arm; matched output therefore reports random-minus-lexical. Reverse signs for lexical-minus-random. "
                        "User bootstrap resamples complete paired user vectors 1000 times with seed 42; percentile intervals condition on the fitted model, supported users, fixed candidates and matching. "
                        "Per-draw balance records arm means, population SDs, quantiles, pooled-SD standardized mean difference, lexical/control variance ratio and maximum ECDF distance. "
                        "Zero pooled SD or zero control variance yields null with flags. Text length counts cleaned characters; metadata coverage is a nonzero-vector indicator; cosine is the intended contrast. "
                        "Intersection/Jaccard measure cross-arm overlap. Frequency ECDF artifacts pool supported prefix slots descriptively; exact controls have identical frequency distributions. "
                        "Calibration prespecifies the last selected prefix of the first up to 40 selected users before matching or scoring, without replacement of excluded examples. "
                        "Its full catalog pool includes metadata-missing items. M is competitor count and A is total-order outranker count; exact rank=A+1 and exact hit10=1[A<=9]. "
                        "For M>=99 the analytic uniform sampled hit is Hypergeom(M,A,99).cdf(9). Each example uses 100 domain-seeded unique-99 draws frozen in frozen.json before neural scoring and cached exact-pool scores. Repeated candidate IDs within each scoring request reuse one score; calibration separately checks two chunk sizes. "
                        "Residual is empirical minus analytic hit probability; conditional MC SE=sqrt(p*(1-p)/100), including degenerate probabilities. This is forward calibration, not rank inversion. "
                        "Two full scoring chunk sizes are checked; nonfinite, nonclose or changed target-order cases are excluded explicitly. Near-target tolerance counts flag numerical ties without changing the exact tie rule. "
                        "Support contains construction, scoring, missingness and exclusion counts. Empty summaries are null with zero support. Timing includes candidate construction, scoring and aggregation. "
                        "Four-category macro and deployment reload parity remain unavailable through this single-category adapter. No training, inference, architecture or evaluator change is introduced."
                    ),
                    "deployment_reload_parity": "Unavailable without a host reload interface; no deployment correctness is inferred.",
                    "validation_rank_metrics": "Unavailable; no training diagnostic is presented as pinned validation HR@10.",
                }
                return {"metrics": metrics, "metric_definitions": definitions}
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    ### <<< Self_EvolveRec-BLOCK-END
