import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Reproducible trial-local candidate audit dependencies
import os
import re
import hashlib
import itertools
import unicodedata
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
    ### >>> Self_EvolveRec-BLOCK-START: Allow an explicit insufficient-evidence diagnosis
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE | INSUFFICIENT_EVIDENCE>,
    ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Interpret training stress panels with explicit missingness
    Rules:
    - Keep core_findings and key_implications to at most three entries each.
    - Ground every statement in measured values, definitions and denominators.
    - Report identity failures and unavailable measurements first. Null is not zero.
      Use INSUFFICIENT_EVIDENCE when the requested comparison is unmeasured.
    - Separate training-history candidate panels, descriptive candidate-draw
      variation, and independently pinned host validation. Panel top-10 inclusion
      is not validation HR@10. Do not forecast numerical validation improvement.
    - Frequency bins leave residual frequency confounding. Discuss training-target
      memorization, lexical coverage and selection bias when supported by counts.
    - Lexical retrieval uses the known target retrospectively; it is not a
      prospective scoring rule. Competitors are row-unobserved, not true negatives.
    - Do not infer ownership, satisfaction, compatibility, causal preferences,
      user-level confidence intervals, or unavailable attributes and traits.
    - Embedding geometry does not establish recommendation diversity or accuracy.
    - Deletions also change coordinates and attention normalization. Final-event
      removal changes the readout event; keep it separate from prefix deletion.
    - Repeated rows and candidate draws are dependent. Draw ranges and standard
      deviations are descriptive and are not confidence intervals.
    - This audit changes no training, checkpoint, predictor or host evaluator.
      Preserve one independently trained HSTU checkpoint per category.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the actual HSTU module and public adapter references
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Diagnose raw ID geometry through the host view
            emb = self.view.item_emb.weight.detach()
            emb = emb[1 : self.model.itemnum + 1]
            ### <<< Self_EvolveRec-BLOCK-END

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Preserve undefined geometry as missing
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            ### >>> Self_EvolveRec-BLOCK-START: Finite nonzero geometry without a square Gram matrix
            x = x.float()
            valid = torch.isfinite(x).all(-1) & x.norm(dim=-1).gt(1e-12)
            x = F.normalize(x[valid], p=2, dim=-1).double()
            k = len(x)
            if k < 2:
                return None
            offdiag_sum = x.sum(0).square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            ### <<< Self_EvolveRec-BLOCK-END
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

            ### >>> Self_EvolveRec-BLOCK-START: Use the family-specific tensor adapter
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.view.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            ### >>> Self_EvolveRec-BLOCK-START: Measure operational joint-representation scores
            all_logits = self.view.predict_score(log_feats, all_items)
            ### <<< Self_EvolveRec-BLOCK-END
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and nonfinite paired scores
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            valid &= torch.isfinite(all_logits).all(-1) & torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical finite-position denominators
            valid_cpu = valid.cpu().numpy()
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
                ### >>> Self_EvolveRec-BLOCK-START: Empty measurements remain unavailable
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

    ### >>> Self_EvolveRec-BLOCK-START: Bounded deterministic retrieval, indexed overlap summaries and finite diagnostic statistics
    PANELS = ("uniform", "positive_bin", "lexical", "lexical_frequency_control")
    CATEGORIES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )
    CONDITIONS = (
        "prefix_delete_oldest",
        "prefix_delete_recent",
        "prefix_delete_random",
        "delete_final",
        "delete_penultimate",
        "shuffle_prefix",
        "swap_adjacent_prefix",
        "shuffle_before_suffix",
    )

    @staticmethod
    def _missing(reason):
        return {"value": None, "reason": reason}

    @staticmethod
    def _summary(values):
        x = np.asarray(values, dtype=np.float64)
        finite = x[np.isfinite(x)]
        return {
            "count": int(finite.size),
            "nonfinite_count": int(x.size - finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "p10": float(np.quantile(finite, 0.1)) if finite.size else None,
            "p50": float(np.quantile(finite, 0.5)) if finite.size else None,
            "p90": float(np.quantile(finite, 0.9)) if finite.size else None,
            "reason": None if finite.size else "No finite observations",
        }

    @classmethod
    def _margins(cls, values):
        result = cls._summary(values)
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        result.update(
            negative_wins_rate=float((x < 0).mean()) if x.size else None,
            small_margin_rate=float((x < 0.5).mean()) if x.size else None,
            small_margin_threshold=0.5,
        )
        return result

    @staticmethod
    def _last(seq):
        return (
            torch.arange(seq.shape[1], device=seq.device)
            .expand_as(seq)
            .masked_fill(seq.eq(0), -1)
            .amax(-1)
        )

    @classmethod
    def _final(cls, features, seq):
        last = cls._last(seq)
        selected = features[
            torch.arange(len(seq), device=seq.device), last.clamp_min(0)
        ]
        return selected.masked_fill(last.lt(0)[:, None], 0)

    def _score(self, features, candidates):
        scores = self.view.predict_score(features[:, None, :], candidates[:, None, :])
        expected = (len(features), 1, candidates.shape[-1])
        if tuple(scores.shape) != expected:
            raise ValueError(f"ModelView scores {tuple(scores.shape)} != {expected}")
        # Subtract in float64 so finite float32 logits cannot overflow margins.
        return scores[:, 0].double()

    def _public_catalog(self):
        # Require an explicit public mapping; never derive real IDs from weights,
        # observed histories, or numeric-looking metadata keys alone.
        # Supported host contract: public_item_mapping contains dataset_name and
        # id_to_metadata_key, with optional reserved_item_ids. No private files read.
        public = getattr(self.view, "public_item_mapping", None)
        if not isinstance(public, dict):
            return None, "Host public_item_mapping is unavailable"
        category = public.get("dataset_name")
        if category not in self.CATEGORIES:
            return None, "Public mapping lacks a recognized dataset_name"
        mapping = public.get("id_to_metadata_key")
        if not isinstance(mapping, dict) or not mapping:
            return None, "Public id_to_metadata_key is unavailable or empty"
        try:
            reserved = {0} | {int(i) for i in public.get("reserved_item_ids", [])}
            aligned = {}
            for key, metadata_key in mapping.items():
                item = int(key)
                if str(item) != str(key) or item in aligned:
                    raise ValueError("Noncanonical or duplicate integer ID")
                if item in reserved:
                    continue
                if not 1 <= item <= self.model.itemnum:
                    raise ValueError("Mapped item is outside the trained catalog")
                if not isinstance(metadata_key, (str, int)):
                    raise ValueError("Unsupported metadata key")
                aligned[item] = metadata_key
            if not aligned:
                raise ValueError("No real mapped items")
        except (TypeError, ValueError) as error:
            return None, f"Invalid public mapping: {error}"
        return (category, aligned), None

    @staticmethod
    def _bin(frequency):
        return 0 if frequency == 0 else int(frequency).bit_length()

    def _assets(self, category, alignment):
        import platform
        import sklearn
        import scipy
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer

        ids = np.asarray(sorted(alignment), dtype=np.int64)
        counts = Counter()
        histories = (
            self.user_train.get("History")
            if isinstance(self.user_train, dict)
            else None
        )
        if not isinstance(histories, dict):
            return None, "user_train['History'] is unavailable"
        invalid_history_values = 0
        for history in histories.values():
            if not isinstance(history, (list, tuple, np.ndarray)):
                return None, "Unsupported training-history container"
            for value in history:
                try:
                    item = int(value)
                    if isinstance(value, (bool, np.bool_)) or str(item) != str(value):
                        raise ValueError("Expected a canonical integer item ID")
                except (TypeError, ValueError, OverflowError):
                    invalid_history_values += 1
                    continue
                counts[item] += 1
        frequency = {int(i): int(counts[int(i)]) for i in ids}
        bins = {i: self._bin(f) for i, f in frequency.items()}
        texts, unsupported, missing = [], Counter(), Counter()

        def clean_field(value, name):
            if value is None:
                missing[name] += 1
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                unsupported[name] += sum(not isinstance(v, str) for v in value)
                return " ".join(v for v in value if isinstance(v, str))
            unsupported[name] += 1
            return ""

        metadata = self.meta_data if isinstance(self.meta_data, dict) else {}
        for item in ids:
            record = metadata.get(alignment[int(item)])
            if not isinstance(record, dict):
                missing["record"] += 1
                record = {}
            raw = " ".join(
                clean_field(record.get(name), name) for name in ("title", "description")
            )
            texts.append(
                re.sub(r"\s+", " ", unicodedata.normalize("NFKC", raw)).strip()
            )
        parameters = dict(
            input="content",
            encoding="utf-8",
            decode_error="strict",
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            analyzer="word",
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            max_df=1.0,
            min_df=1,
            max_features=100000,
            vocabulary=None,
            binary=False,
            dtype=np.float32,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )
        vectorizer = TfidfVectorizer(**parameters)
        lexical_reason = None
        try:
            matrix = vectorizer.fit_transform(texts).tocsr()
        except ValueError as error:
            if "empty vocabulary" not in str(error).lower():
                raise
            matrix = scipy.sparse.csr_matrix((len(ids), 0), dtype=np.float32)
            vectorizer = None
            lexical_reason = "Empty TF-IDF vocabulary"
        zero = np.asarray(matrix.getnnz(axis=1) == 0).ravel()
        manifest = {
            "category": category,
            "ordered_item_ids": ids.tolist(),
            "id_to_metadata_key": {str(i): alignment[int(i)] for i in ids},
            "cleaning": "title then description; supported string lists in stored order; NFKC; collapse whitespace; strip",
            "text_sha256": [
                hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts
            ],
            "parameters": {
                k: ("float32" if k == "dtype" else v) for k, v in parameters.items()
            },
            # sklearn may expose NumPy integer vocabulary indices.
            "vocabulary": (
                {
                    str(token): int(index)
                    for token, index in vectorizer.vocabulary_.items()
                }
                if vectorizer is not None
                else {}
            ),
            "idf": vectorizer.idf_.tolist() if vectorizer is not None else [],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
            "unsupported_metadata_values": dict(unsupported),
            "missing_metadata": dict(missing),
            "invalid_history_values": invalid_history_values,
            "zero_vector_items": int(zero.sum()),
            "lexical_reason": lexical_reason,
            "frequency_bins": "0: zero; 1: one; b>=2: [2**(b-1), 2**b-1]",
            "interaction_occurrences": sum(frequency.values()),
            "seed_coordinates": [42, "category_index", "batch", "row", "draw", "panel"],
            "panel_indices": dict(zip(self.PANELS, range(4))),
        }
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "working", "diagnosis", category
        )
        os.makedirs(directory, exist_ok=True)
        with open(
            os.path.join(directory, "lexical_manifest.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
        joblib.dump(vectorizer, os.path.join(directory, "tfidf.joblib"))
        np.savez_compressed(
            os.path.join(directory, "catalog.npz"),
            ids=ids,
            frequency=[frequency[int(i)] for i in ids],
            zero=zero,
        )
        scipy.sparse.save_npz(os.path.join(directory, "tfidf.npz"), matrix)
        return (
            dict(
                ids=ids,
                real=set(ids.tolist()),
                index={int(i): j for j, i in enumerate(ids)},
                frequency=frequency,
                bins=bins,
                texts=texts,
                matrix=matrix,
                zero=zero,
                directory=directory,
                manifest=manifest,
                category_index=self.CATEGORIES.index(category),
                neighbors={},
            ),
            None,
        )

    @staticmethod
    def _lexical(asset, target, excluded):
        # Retrieve each target once for all its cached row-specific exclusions.
        # Retain only 99 entries/query; do not accumulate full neighbor lists.
        key = tuple(sorted(excluded))
        cached = asset["neighbors"].get(target, {})
        if key not in cached:
            plans = set(asset.get("query_exclusions", {}).get(target, ())) | {key}
            running = {
                plan: [
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float32),
                    0,
                    set(plan),
                ]
                for plan in plans
            }
            query = asset["matrix"][asset["index"][target]]
            for start in range(0, len(asset["ids"]), 4096):
                sparse = (asset["matrix"][start : start + 4096] @ query.T).tocsr()
                sparse.sum_duplicates()
                scores = sparse.tocoo()
                positive = np.isfinite(scores.data) & (scores.data > 0)
                ids = asset["ids"][start + scores.row[positive]]
                values = scores.data[positive]
                for state in running.values():
                    keep = np.fromiter(
                        (int(i) not in state[3] for i in ids), bool, len(ids)
                    )
                    eligible_ids, eligible_scores = ids[keep], values[keep]
                    state[2] += len(eligible_ids)
                    # Resolve boundary ties before both block and merge truncation.
                    order = np.lexsort((eligible_ids, -eligible_scores))[:99]
                    merged_ids = np.concatenate((state[0], eligible_ids[order]))
                    merged_scores = np.concatenate((state[1], eligible_scores[order]))
                    order = np.lexsort((merged_ids, -merged_scores))[:99]
                    state[0], state[1] = merged_ids[order], merged_scores[order]
            cached = {
                plan: (state[0].tolist() if state[2] >= 99 else None, int(state[2]))
                for plan, state in running.items()
            }
            asset["neighbors"][target] = cached
        return cached[key]

    @staticmethod
    def _rng(category, batch, row, draw, panel):
        return np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([42, category, batch, row, draw, panel])
            )
        )

    @staticmethod
    def _sample(pool, rng, count=99):
        return (
            rng.choice(pool, count, replace=False).tolist()
            if len(pool) >= count
            else None
        )

    @staticmethod
    def _alter(seq, condition, batch, replicate):
        result = np.zeros_like(seq.cpu().numpy())
        eligible = np.zeros(len(seq), dtype=bool)
        for row, values in enumerate(seq.cpu().numpy()):
            items = values[values != 0].copy()
            n = len(items)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        42,
                        batch,
                        row,
                        replicate,
                        DiagnosisProbe.CONDITIONS.index(condition),
                    ]
                )
            )
            if condition.startswith("prefix_delete") and n >= 3:
                k = max(1, (n - 1) // 5)
                remove = (
                    np.arange(k)
                    if condition.endswith("oldest")
                    else (
                        np.arange(n - 1 - k, n - 1)
                        if condition.endswith("recent")
                        else rng.choice(n - 1, k, replace=False)
                    )
                )
                items = np.delete(items, remove)
                eligible[row] = True
            elif condition in ("delete_final", "delete_penultimate") and n >= 2:
                items = np.delete(
                    items, n - 1 if condition == "delete_final" else n - 2
                )
                eligible[row] = True
            elif condition == "shuffle_prefix" and n >= 3:
                items[:-1] = rng.permutation(items[:-1])
                eligible[row] = True
            elif condition == "swap_adjacent_prefix" and n >= 3:
                offset = 0 if n == 3 else int(rng.integers(2))
                for j in range(offset, n - 2, 2):
                    items[j], items[j + 1] = items[j + 1], items[j]
                eligible[row] = True
            elif condition == "shuffle_before_suffix" and n >= 4:
                width = n - min(3, n - 1)
                if width >= 2:
                    items[:width] = rng.permutation(items[:width])
                    eligible[row] = True
            if len(items):
                result[row, -len(items) :] = items
        changed = (result != seq.cpu().numpy()).any(-1)
        return (
            torch.as_tensor(result, device=seq.device),
            torch.as_tensor(eligible, device=seq.device),
            changed,
        )

    def _characteristics(self, queries):
        return {
            "rows": len(queries),
            **{
                name: self._summary([q[name] for q in queries])
                for name in (
                    "history_length",
                    "target_frequency",
                    "target_text_length",
                    "lexical_neighbors",
                )
            },
            "zero_vector_rate": (
                np.mean([q["zero_vector"] for q in queries]).item() if queries else None
            ),
            "fraction_reaching_99": (
                np.mean([q["lexical_neighbors"] >= 99 for q in queries]).item()
                if queries
                else None
            ),
        }

    def _scope(self, queries, panels):
        selected = [
            q
            for q in queries
            if all(
                len(q["statistics"][p]) == (1 if p == "lexical" else 5) for p in panels
            )
        ]
        result = self._characteristics(selected)
        selected_ids = {id(q) for q in selected}
        excluded = [q for q in queries if id(q) not in selected_ids]
        result["excluded_queries"] = self._characteristics(excluded)
        result["excluded_panel_draw_reasons"] = dict(
            Counter(
                f"{record['panel']}: {record['reason']}"
                for q in excluded
                for record in q["missingness"]
                if record["panel"] in panels
            )
        )
        result["panels"] = {}
        for panel in panels:
            result["panels"][panel] = {
                metric: self._summary(
                    [
                        np.mean([s[metric] for s in q["statistics"][panel]])
                        for q in selected
                    ]
                )
                for metric in ("rank", "margin", "ties", "training_panel_top10")
            }
        result["paired_differences"] = {}
        for left, right in itertools.combinations(panels, 2):
            comparisons = {}
            for metric in ("rank", "margin"):
                draws = []
                for draw in range(5):
                    differences = [
                        q["statistics"][left][0 if left == "lexical" else draw][metric]
                        - q["statistics"][right][0 if right == "lexical" else draw][
                            metric
                        ]
                        for q in selected
                    ]
                    draws.append(float(np.mean(differences)) if differences else None)
                finite = [x for x in draws if x is not None]
                comparisons[metric] = {
                    "mean_query_averaged_difference": (
                        float(np.mean(finite)) if finite else None
                    ),
                    "aggregate_difference_by_draw": draws,
                    "descriptive_draw_sd": (
                        float(np.std(finite, ddof=0)) if finite else None
                    ),
                    "descriptive_draw_range": (
                        [min(finite), max(finite)] if finite else None
                    ),
                }
            result["paired_differences"][f"{left}_minus_{right}"] = comparisons
        return result

    def _overlap_summaries(self, overlaps):
        # Index once instead of scanning all overlap records for every query.
        grouped = {}
        for record in overlaps:
            key = (record["left"], record["right"], record["batch"], record["row"])
            grouped.setdefault(key, []).append(record)
        return {
            f"{left}_vs_{right}": {
                metric: self._summary(
                    [
                        float(np.mean([record[metric] for record in records]))
                        for key, records in grouped.items()
                        if key[:2] == (left, right)
                    ]
                )
                for metric in ("intersection", "jaccard")
            }
            for left, right in itertools.combinations(self.PANELS, 2)
        }

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        modes = [(module, module.training) for module in self.model.modules()]
        # Loader construction may use the training RNGs. Restore them even if a
        # loader or optional lexical dependency fails; local audit draws are separate.
        import random

        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        self.model.eval()
        try:
            cache, invalid_batches = [], []
            for batch_id, batch in enumerate(itertools.islice(train_loader, 10)):
                if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                    invalid_batches.append(
                        {"batch": batch_id, "reason": "Expected four-tuple"}
                    )
                    continue
                tensors = tuple(
                    torch.as_tensor(t).detach().cpu().clone() for t in batch[1:]
                )
                if (
                    any(t.ndim != 2 for t in tensors)
                    or any(t.shape != tensors[0].shape for t in tensors)
                    or not tensors[0].shape[1]
                ):
                    invalid_batches.append(
                        {"batch": batch_id, "reason": "Expected matching B x L, L>0"}
                    )
                    continue
                cache.append((batch_id, tensors))
            catalog, mapping_reason = self._public_catalog()
            asset, asset_reason = (None, mapping_reason)
            if catalog is not None:
                asset, asset_reason = self._assets(*catalog)
            if asset is not None:
                # Declare all exclusions before retrieving any target. This permits
                # one sparse catalog traversal per distinct target, with bounded
                # top-99 state per sampled query and exact neighbor counts.
                asset["query_exclusions"] = {}
                for _, (sequence, positive, _) in cache:
                    for history, targets in zip(sequence.numpy(), positive.numpy()):
                        physical = np.flatnonzero(history)
                        if not physical.size:
                            continue
                        target = int(targets[physical[-1]])
                        if target in asset["real"]:
                            excluded = tuple(
                                sorted(set(history[history != 0].tolist()) | {target})
                            )
                            asset["query_exclusions"].setdefault(target, set()).add(
                                excluded
                            )
            margins, identities, queries, exclusions, perturbations = [], [], [], [], []
            batch_counts, overlaps = [], []
            for batch_id, tensors in cache:
                seq, pos, neg = (t.to(self.device, dtype=torch.long) for t in tensors)
                if not len(seq):
                    batch_counts.append({"batch": batch_id, "rows": 0})
                    continue
                # Invalid IDs would otherwise fail inside an embedding lookup.
                # Report the malformed batch; never coerce it into another history.
                if any(
                    bool(((t < 0) | (t > self.model.itemnum)).any())
                    for t in (seq, pos, neg)
                ):
                    invalid_batches.append(
                        {
                            "batch": batch_id,
                            "rows": len(seq),
                            "reason": "Item ID outside HSTU catalog bounds",
                        }
                    )
                    continue
                features = self.view.log2feats(seq)
                final = self._final(features, seq)
                last = self._last(seq)
                rows = torch.arange(len(seq), device=self.device)
                pair = torch.stack(
                    (pos[rows, last.clamp_min(0)], neg[rows, last.clamp_min(0)]), -1
                )
                original = self._score(final, pair)
                repeat_final = self._final(self.view.log2feats(seq.clone()), seq)
                repeated = self._score(repeat_final, pair)
                checked = last.ge(0) & pair.ne(0).all(-1)
                finite = torch.isfinite(original).all(-1) & torch.isfinite(
                    repeated
                ).all(-1)
                close = torch.isclose(original, repeated, atol=1e-5, rtol=1e-4).all(-1)
                # Also check the positive alone for rows with no diagnostic negative.
                targets = pair[:, :1]
                positive_scores = self._score(final, targets)
                repeated_positive = self._score(repeat_final, targets)
                positive_checked = last.ge(0) & targets[:, 0].ne(0)
                positive_ok = (
                    torch.isfinite(positive_scores).all(-1)
                    & torch.isfinite(repeated_positive).all(-1)
                    & torch.isclose(
                        positive_scores, repeated_positive, atol=1e-5, rtol=1e-4
                    ).all(-1)
                )
                failed = int((checked & ~(finite & close)).sum())
                positive_failed = int((positive_checked & ~positive_ok).sum())
                batch_ok = failed == 0 and positive_failed == 0
                identities.append(
                    {
                        "batch": batch_id,
                        "checked_rows": int(checked.sum()),
                        "failed_rows": failed,
                        "nonfinite_rows": int((checked & ~finite).sum()),
                        "positive_checked_rows": int(positive_checked.sum()),
                        "positive_failed_rows": positive_failed,
                    }
                )
                logits = self.view.predict_score(features, torch.stack((pos, neg), -1))
                if tuple(logits.shape) != (*seq.shape, 2):
                    raise ValueError("Expected B x L x 2 paired diagnostic scores")
                valid = seq.ne(0) & pos.ne(0) & neg.ne(0)
                position_margin = logits[..., 0].double() - logits[..., 1].double()
                position_margin = position_margin.masked_fill(
                    ~torch.isfinite(logits).all(-1), float("nan")
                )
                margins.extend(position_margin[valid].cpu().tolist())
                batch_counts.append(
                    {
                        "batch": batch_id,
                        "rows": len(seq),
                        "nonpadding_positions": int(seq.ne(0).sum()),
                        "valid_target_positions": int(valid.sum()),
                        "empty_rows": int(last.lt(0).sum()),
                    }
                )
                for condition in self.CONDITIONS:
                    for replicate in range(5):
                        altered, length_ok, changed = self._alter(
                            seq, condition, batch_id, replicate
                        )
                        eligible = checked & finite & length_ok
                        record = {
                            "batch": batch_id,
                            "condition": condition,
                            "replicate": replicate,
                            "eligible_before_identity": int(eligible.sum()),
                            "identity_excluded_rows": (
                                int(eligible.sum()) if not batch_ok else 0
                            ),
                            "pairs": [],
                        }
                        if batch_ok and bool(eligible.any()):
                            altered_final = self._final(
                                self.view.log2feats(altered), altered
                            )
                            scored = self._score(altered_final, pair)
                            altered_margin = scored[:, 0] - scored[:, 1]
                            original_margin = original[:, 0] - original[:, 1]
                            usable = (
                                eligible
                                & torch.isfinite(scored).all(-1)
                                & torch.isfinite(original_margin)
                                & torch.isfinite(altered_margin)
                                & torch.isfinite(altered_margin - original_margin)
                            )
                            record["nonfinite_rows"] = int((eligible & ~usable).sum())
                            cosine = F.cosine_similarity(
                                final.float(), altered_final.float(), dim=-1
                            )
                            for row in usable.nonzero(as_tuple=True)[0].cpu().tolist():
                                before = float(original[row, 0] - original[row, 1])
                                after = float(scored[row, 0] - scored[row, 1])
                                cos_ok = (
                                    bool(torch.isfinite(cosine[row]))
                                    and float(final[row].norm()) > 1e-12
                                    and float(altered_final[row].norm()) > 1e-12
                                )
                                record["pairs"].append(
                                    dict(
                                        row=row,
                                        original_margin=before,
                                        altered_margin=after,
                                        margin_change=after - before,
                                        pairwise_order_flip=bool(
                                            np.sign(before) != np.sign(after)
                                        ),
                                        input_changed=bool(changed[row]),
                                        feature_cosine=(
                                            float(cosine[row]) if cos_ok else None
                                        ),
                                    )
                                )
                        perturbations.append(record)
                for row in range(len(seq)):
                    target = int(targets[row, 0])
                    reason = (
                        "empty_history"
                        if last[row] < 0
                        else (
                            "padding_target"
                            if target == 0
                            else (
                                "mapping_or_assets_unavailable"
                                if asset is None
                                else (
                                    "reserved_or_unknown_target"
                                    if target not in asset["real"]
                                    else None
                                )
                            )
                        )
                    )
                    if reason:
                        exclusions.append(
                            dict(
                                batch=batch_id,
                                row=row,
                                target=target,
                                reason=reason,
                                history_length=int(seq[row].ne(0).sum()),
                            )
                        )
                        continue
                    history = set(seq[row][seq[row].ne(0)].cpu().tolist())
                    excluded = history | {target}
                    eligible_ids = [
                        int(i) for i in asset["ids"] if int(i) not in excluded
                    ]
                    groups = {}
                    for item in eligible_ids:
                        groups.setdefault(asset["bins"][item], []).append(item)
                    lexical, neighbor_count = self._lexical(asset, target, excluded)
                    positive_pool = groups.get(asset["bins"][target], [])
                    histogram = (
                        Counter(asset["bins"][i] for i in lexical)
                        if lexical is not None
                        else None
                    )
                    index = asset["index"][target]
                    q = dict(
                        batch=batch_id,
                        row=row,
                        target=target,
                        history_length=int(seq[row].ne(0).sum()),
                        target_frequency=asset["frequency"][target],
                        target_text_length=len(asset["texts"][index]),
                        zero_vector=bool(asset["zero"][index]),
                        lexical_neighbors=neighbor_count,
                        lexical_reason=(
                            "Zero TF-IDF query vector"
                            if asset["zero"][index]
                            else (
                                "Fewer than 99 eligible positive-similarity neighbors"
                                if lexical is None
                                else None
                            )
                        ),
                        identity_control_passed=batch_ok,
                        positive_bin_pool_size=len(positive_pool),
                        eligible_pool_size=len(eligible_ids),
                        excluded_ids=sorted(excluded),
                        statistics={p: [] for p in self.PANELS},
                        plans=[],
                        missingness=[],
                    )
                    for draw in range(5):
                        rng = lambda panel: self._rng(
                            asset["category_index"], batch_id, row, draw, panel
                        )
                        matched = None
                        if histogram is not None:
                            matched = []
                            generator = rng(3)
                            for bin_id, count in sorted(histogram.items()):
                                sampled = self._sample(
                                    groups.get(bin_id, []), generator, count
                                )
                                if sampled is None:
                                    matched = None
                                    break
                                matched.extend(sampled)
                        panels = dict(
                            uniform=self._sample(eligible_ids, rng(0)),
                            positive_bin=self._sample(positive_pool, rng(1)),
                            lexical=lexical,
                            lexical_frequency_control=matched,
                        )
                        q["plans"].append({"draw": draw, "panels": panels})
                        for name, competitors in panels.items():
                            if competitors is None:
                                q["missingness"].append(
                                    dict(
                                        draw=draw,
                                        panel=name,
                                        reason=(
                                            q["lexical_reason"]
                                            if name
                                            in ("lexical", "lexical_frequency_control")
                                            and lexical is None
                                            else "Insufficient eligible frequency-bin or catalog pool"
                                        ),
                                    )
                                )
                                continue
                            if not batch_ok:
                                q["missingness"].append(
                                    dict(
                                        draw=draw,
                                        panel=name,
                                        reason="Batch identity failure",
                                    )
                                )
                                continue
                            if name == "lexical" and draw:
                                continue
                            candidates = torch.tensor(
                                [[target] + competitors], device=self.device
                            )
                            scored = self._score(final[row : row + 1], candidates)[0]
                            panel_margin = scored[0] - scored[1:].max()
                            if not bool(
                                torch.isfinite(scored).all()
                                & torch.isfinite(panel_margin)
                            ):
                                q["missingness"].append(
                                    dict(
                                        draw=draw,
                                        panel=name,
                                        reason="Nonfinite panel scores",
                                    )
                                )
                                continue
                            rank = 1 + int((scored[1:] >= scored[0]).sum())
                            q["statistics"][name].append(
                                dict(
                                    draw=draw,
                                    rank=rank,
                                    margin=float(panel_margin),
                                    ties=int((scored[1:] == scored[0]).sum()),
                                    training_panel_top10=int(rank <= 10),
                                    competitor_frequency=self._summary(
                                        [asset["frequency"][i] for i in competitors]
                                    ),
                                    frequency_by_bin={
                                        str(b): self._summary(
                                            [
                                                asset["frequency"][i]
                                                for i in competitors
                                                if asset["bins"][i] == b
                                            ]
                                        )
                                        for b in sorted(
                                            {asset["bins"][i] for i in competitors}
                                        )
                                    },
                                )
                            )
                        for left, right in itertools.combinations(self.PANELS, 2):
                            if panels[left] is not None and panels[right] is not None:
                                a, b = set(panels[left]), set(panels[right])
                                overlaps.append(
                                    dict(
                                        batch=batch_id,
                                        row=row,
                                        draw=draw,
                                        left=left,
                                        right=right,
                                        intersection=len(a & b),
                                        jaccard=len(a & b) / len(a | b),
                                    )
                                )
                    queries.append(q)
            panel_report = (
                self._missing(asset_reason)
                if asset is None
                else {
                    "manifest_path": os.path.join(
                        asset["directory"], "lexical_manifest.json"
                    ),
                    "all_queries": self._characteristics(queries),
                    "all_four_common": self._scope(queries, self.PANELS),
                    "pair_specific": {
                        f"{a}_vs_{b}": self._scope(queries, (a, b))
                        for a, b in itertools.combinations(self.PANELS, 2)
                    },
                    "panel_specific": {
                        p: self._scope(queries, (p,)) for p in self.PANELS
                    },
                    "excluded_by_panel": {
                        p: self._characteristics(
                            [
                                q
                                for q in queries
                                if len(q["statistics"][p])
                                != (1 if p == "lexical" else 5)
                            ]
                        )
                        for p in self.PANELS
                    },
                    "pair_overlaps": self._overlap_summaries(overlaps),
                }
            )
            if asset is not None:
                path = os.path.join(
                    asset["directory"], "candidate_plans_and_measurements.json"
                )
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(
                        dict(
                            queries=queries,
                            exclusions=exclusions,
                            overlaps=overlaps,
                            identity=identities,
                            perturbations=perturbations,
                        ),
                        stream,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                panel_report["records_path"] = path
            intervention_summary = {}
            for condition in self.CONDITIONS:
                records = [r for r in perturbations if r["condition"] == condition]
                pairs = [p for r in records for p in r["pairs"]]
                intervention_summary[condition] = {
                    "finite_pair_denominator": len(pairs),
                    "identity_excluded_rows": sum(
                        r["identity_excluded_rows"] for r in records
                    ),
                    "original_margins": self._margins(
                        [p["original_margin"] for p in pairs]
                    ),
                    "altered_margins": self._margins(
                        [p["altered_margin"] for p in pairs]
                    ),
                    "margin_changes": self._summary(
                        [p["margin_change"] for p in pairs]
                    ),
                    "feature_cosines": self._summary(
                        [
                            p["feature_cosine"]
                            for p in pairs
                            if p["feature_cosine"] is not None
                        ]
                    ),
                    "pairwise_order_flip_rate": (
                        float(np.mean([p["pairwise_order_flip"] for p in pairs]))
                        if pairs
                        else None
                    ),
                    "unchanged_input_rate": (
                        float(np.mean([not p["input_changed"] for p in pairs]))
                        if pairs
                        else None
                    ),
                }
            return {
                "metrics": {
                    "actual_module": type(self.model).__name__,
                    "cached_batches": len(cache),
                    "invalid_batches": invalid_batches,
                    "batch_counts": batch_counts,
                    "embedding_collapse_score": self.analyze_embeddings_collapse_cosine(),
                    "training_pair_separation": self._margins(margins),
                    "identity": {
                        "atol": 1e-5,
                        "rtol": 1e-4,
                        "batches": identities,
                        "checked_rows": sum(i["checked_rows"] for i in identities),
                        "failed_rows": sum(i["failed_rows"] for i in identities),
                        "positive_checked_rows": sum(
                            i["positive_checked_rows"] for i in identities
                        ),
                        "positive_failed_rows": sum(
                            i["positive_failed_rows"] for i in identities
                        ),
                        "passed": (
                            all(
                                i["failed_rows"] == 0 and i["positive_failed_rows"] == 0
                                for i in identities
                            )
                            if any(i["positive_checked_rows"] for i in identities)
                            else None
                        ),
                    },
                    "interventions": intervention_summary,
                    "raw_perturbation_records": perturbations,
                    "candidate_competition": panel_report,
                    "excluded_queries": exclusions,
                    "item_category_breakdown": self._missing(
                        "Item categories are unavailable"
                    ),
                    "user_confidence_intervals": self._missing(
                        "Loader rows have no verified user identities"
                    ),
                    "validation": self._missing(
                        "Training audit cannot measure pinned host validation HR@10"
                    ),
                },
                "metric_definitions": {
                    "scope": "One fitted HSTU/category; no training, checkpoint, predictor or evaluator changes. Batch/row identifiers are not user IDs.",
                    "geometry": "Raw ID embeddings only: (mean finite nonzero off-diagonal cosine+1)/2. Not joint text vectors, accuracy or recommendation diversity.",
                    "training_pair_separation": "Operational positive-minus-negative scores at nonpadding positions with both targets nonzero; nonfinite margins counted separately. Small means margin<0.5 and depends on score scale.",
                    "identity": "Unchanged cloned eval/no_grad inference, atol=1e-5 and rtol=1e-4. Any eligible paired or positive-only failure invalidates all new comparisons from that batch. No checked rows means null.",
                    "readout": "Maximum physical nonzero position, never count-1. Empty rows excluded. Original final positive/negative pair is frozen for perturbations.",
                    "perturbations": "Five seed-42 local plans/condition. Prefix deletion removes max(1,floor((n-1)/5)) events for n>=3, preserving final event. Final/penultimate deletion requires n>=2. Order probes preserve the multiset and final event. Deletion repacks coordinates and changes normalization; final deletion changes the readout event. Repeated deterministic plans are dependent.",
                    "mapping": "Only explicit public_item_mapping with dataset_name and id_to_metadata_key establishes real catalog IDs and metadata alignment. Missing mapping makes candidate audit unavailable; embedding size never establishes real IDs.",
                    "frequency": "Interaction occurrences exclusively from user_train['History']; real zero-frequency items included. Fixed bins: zero; one; [2,3]; [4,7]; successive powers of two. Exact target and competitor frequencies are retained; bins leave residual confounding.",
                    "lexical": "Retrospective target-title/description TF-IDF. Parameters, hashes, vocabulary, IDF and versions are saved. All sampled exclusion plans are declared first. One CSR catalog traversal per distinct target uses blocks of at most 4096 rows, applying each plan's exclusions before immediate top-99 reduction. Block and merge order is descending similarity then ascending ID. Only final top-99 plans and exact eligible positive-neighbor counts are cached; full neighbor lists are not retained.",
                    "panels": "Exactly 99 unique row-unobserved competitors, excluding target and every nonzero row-history ID. Uniform catalog, positive-frequency-bin, fixed lexical top99, and lexical-bin-histogram control. No lexical exclusion from the control. Insufficient pools are unavailable, never shortened or filled with zero-similarity entries.",
                    "draws": "Five PCG64 SeedSequence([42,category_index,batch,row,draw,panel]) draws from sorted pools; category/panel order declared in manifest. Lexical panel scored once/query. Plans persisted for reuse; none are additional models.",
                    "rank": "1+count(competitor_score>=positive_score), pessimistic ties; exact tie count retained. Margin=positive-max(competitors). All 100 scores must be finite. training_panel_top10 means rank<=10 on a training-history stress panel, never validation HR@10.",
                    "aggregation": "Complete finite draws required for each scoped panel. Outcomes averaged within query before row summaries. All-four-common, pair-specific and panel-specific scopes include separate selected/excluded row characteristics. Excluded-panel reasons count panel-draw records, not distinct rows. Paired differences are left-minus-right. Draw SD uses ddof=0; ranges/SD describe aggregate candidate-draw variation, not confidence intervals. Invalid out-of-bounds batches are reported before model lookup and excluded from all measurements.",
                    "coverage": "Sampled-row counts, nonpadding history length, target interaction frequency, cleaned-text character length, TF-IDF zero-vector rate, eligible positive-similarity neighbor count and fraction reaching 99. Repeated rows remain dependent.",
                    "overlap": "Competitor-set intersection and Jaccard measured per available plan, then averaged within query before descriptive row summaries; availability does not imply finite scores or valid identity.",
                    "limits": "Training-target memorization, row sampling, frequency and lexical availability can create shortcuts/selection bias. Competitors are not confirmed negatives. Unknown ownership, satisfaction, compatibility and user intent cannot be inferred. No unavailable ratings, reviews, categories, prices, stores or simulator traits are consumed. Only unchanged host four-category macro validation HR@10 can select later training changes.",
                },
            }
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    # Review checks: traced cached batches -> exclusions -> four candidate panels
    # -> operational ModelView scores -> scoped summaries -> trial-local JSON.
    # Checked tie ordering, frequency controls, finite margins, serialization,
    # bounded retrieval state, identity gating and mode/global-RNG restoration.
    # No execution, training, hyperparameter tuning or validation results claimed.
    ### <<< Self_EvolveRec-BLOCK-END
