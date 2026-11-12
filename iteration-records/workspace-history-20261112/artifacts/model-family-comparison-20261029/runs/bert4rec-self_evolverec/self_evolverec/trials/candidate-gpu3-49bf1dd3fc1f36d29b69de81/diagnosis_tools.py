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
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE | INSUFFICIENT_EVIDENCE>,
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
    ### >>> Self_EvolveRec-BLOCK-START: Evidence-limited candidate audit interpretation
    - Use INSUFFICIENT_EVIDENCE when relevant measurements are unavailable.
    - Report candidate-audit coverage, complete paired-user denominators, costs,
      failures, partial completion and current serving-parity evidence.
    - Distinguish retrospective training-target retrieval from host validation.
      Targets occurred during fitting; withholding them from an input does not
      remove memorization. Frequencies describe complete training popularity.
    - Report user-bootstrap intervals separately from three-draw variability.
      Intervals condition on fitted checkpoint, selected support and fixed draws.
    - Lexical alternatives have unknown relevance. Lower target hits do not
      establish worse recommendations, dislike, incompatibility or semantic causation.
    - Histogram matching changes support and leaves within-band confounding.
      Missing text and complete-case selection limit population coverage.
    - Do not infer convergence, temporal generalization, simulator traits,
      ownership, mood, ratings or unavailable metadata. Embedding alignment
      does not establish collapse or recommendation quality.
    - No validation gain, auxiliary serving scorer, fusion or checkpoint averaging
      is established or authorized by this diagnostic.
    ### <<< Self_EvolveRec-BLOCK-END
    """


### >>> Self_EvolveRec-BLOCK-START: Candidate audit with explicit parity failures, serializable fingerprints and complete-user summaries
class _CandidateAudit:
    ROUTES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )
    REGIMES = (
        "uniform_all_catalog",
        "target_band_all_catalog",
        "uniform",
        "target_band",
        "lexical",
        "lexical_frequency_control",
    )

    def __init__(self, view, device):
        import time

        self.view, self.net = view, view.model
        self.device = torch.device(device)
        self.started = time.perf_counter()
        self.cpu_seconds = 0.0
        self.gpu_seconds = 0.0
        self.scoring_wall = 0.0
        self.records, self.failures = [], []
        self.category = getattr(view, "dataset_name", None)
        if self.category is None:
            self.category = getattr(self.net, "dataset_name", None)

    @staticmethod
    def unavailable(reason, **extra):
        return {"available": False, "reason": reason, **extra}

    def boundary(self):
        import time

        if self.gpu_seconds >= 15.0:
            raise TimeoutError("15-second soft scoring cap reached")
        if time.perf_counter() - self.started >= 60.0:
            raise TimeoutError("60-second soft total preparation/audit cap reached")

    def timed(self, function):
        import time

        self.boundary()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
        wall = time.perf_counter()
        try:
            return function()
        finally:
            if self.device.type == "cuda":
                end.record(torch.cuda.current_stream(self.device))
                torch.cuda.synchronize(self.device)
                self.gpu_seconds += start.elapsed_time(end) / 1000.0
            else:
                self.gpu_seconds += time.perf_counter() - wall
            self.scoring_wall += time.perf_counter() - wall

    def inputs(self, history):
        length = self.net.maxlen
        raw = torch.zeros((1, length), dtype=torch.long, device=self.device)
        retained = history[-length:]
        if retained:
            raw[0, -len(retained) :] = torch.tensor(retained, device=self.device)
        sequence = torch.zeros_like(raw)
        sequence[:, :-1] = raw[:, 1:]
        sequence[:, -1] = self.net.mask_token
        return raw, sequence

    def query(self, sequence):
        features = self.view.log2feats(sequence.detach().cpu().numpy().copy())
        if features.shape != (*sequence.shape, self.net.hidden_units):
            raise ValueError("log2feats must return B x L x D")
        positions = torch.arange(sequence.shape[1], device=self.device)[None, :]
        last = positions.expand_as(sequence).masked_fill(sequence.eq(0), -1).amax(1)
        rows = torch.arange(sequence.shape[0], device=self.device)
        if not bool((last == self.net.maxlen - 1).all()):
            raise ValueError("Appended MASK must occupy the final dense slot")
        if not bool(sequence[rows, last].eq(self.net.mask_token).all()):
            raise ValueError("Readout is not MASK")
        return features[rows, last]

    def score(self, query, ids):
        # Use aligned positions explicitly: B x K x D and B x K x 1.
        # This preserves the host's B x L x C predict_score contract.
        features = query[:, None, :].expand(-1, ids.shape[1], -1)
        scores = self.view.predict_score(features, ids[..., None])
        if scores.shape != (*ids.shape, 1):
            raise ValueError("predict_score must preserve B x K x 1 alignment")
        scores = scores[..., 0].float()
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Nonfinite candidate scores")
        return scores

    @staticmethod
    def order(scores, ids):
        return np.lexsort((np.asarray(ids), -np.asarray(scores, dtype=np.float64)))

    def parity(self, histories):
        import importlib.util
        import os
        import sys

        if self.category not in self.ROUTES:
            return self.unavailable("Explicit category identity is missing or invalid")
        declared = getattr(self.net, "dataset_name", self.category)
        if declared != self.category:
            raise ValueError("Model and ModelView category identities disagree")
        path = getattr(self.view, "predictor_path", None)
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "working", "predict.py"
            )
        if not os.path.isfile(path):
            return self.unavailable("Current trial exported predictor is missing")
        spec = importlib.util.spec_from_file_location("_candidate_audit_export", path)
        if spec is None or spec.loader is None:
            return self.unavailable("Cannot import current trial predictor")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            module.DATASETS = [self.category]
            module._ensure_loaded()
            if set(module._models) != {self.category}:
                raise ValueError("Export did not load exactly the declared route")
            exported = module._models[self.category]
            actual, loaded = self.net.state_dict(), exported.state_dict()
            if actual.keys() != loaded.keys() or any(
                actual[k].shape != loaded[k].shape
                or actual[k].dtype != loaded[k].dtype
                or not torch.equal(actual[k].detach().cpu(), loaded[k].detach().cpu())
                for k in actual
            ):
                raise ValueError("Serialized checkpoint differs from diagnostic model")
            # Mapping evidence must be supplied, rather than inferred from equal
            # item counts or successful state loading.
            mapping = getattr(self.view, "item_id_mapping", None)
            saved_mapping = module._metas[self.category].get("item_id_mapping")
            if mapping is None or saved_mapping is None:
                return self.unavailable(
                    "Authoritative ModelView and serialized item-ID mappings are required",
                    checkpoint_state_identity=True,
                )
            if mapping != saved_mapping:
                raise ValueError("ModelView and exported item-ID mappings disagree")
            ids = list(range(1, min(self.net.itemnum, 100) + 1))
            if len(ids) < 100:
                return self.unavailable(
                    "Parity requires at least 100 real catalog items"
                )
            cases = [next(h for h in histories.values() if h)[:1]]
            for required in (self.net.maxlen - 1, self.net.maxlen + 1):
                source = next(
                    (h for h in histories.values() if len(h) >= required), None
                )
                if source is None:
                    return self.unavailable(
                        "Real full-length/truncated parity coverage is missing"
                    )
                cases.append(source[:required])
            sequences, singles, maximum = [], [], 0.0
            for history in cases:
                raw, sequence = self.inputs(history)
                candidates = torch.tensor([ids], device=self.device)
                query = self.query(sequence)
                head = self.score(query, candidates)[0]
                independent = torch.cat(
                    [self.score(query, candidates[:, i : i + 1])[0] for i in range(100)]
                )
                internal = (
                    self.net.score_candidates(raw, candidates).reshape(-1).float()
                )
                external = torch.as_tensor(
                    module.predict(0, history, ids, dataset_name=self.category),
                    device=self.device,
                    dtype=torch.float32,
                ).reshape(-1)
                reference = head.cpu().numpy()
                for other in (independent, internal, external):
                    if not bool(torch.isfinite(other).all()) or not torch.allclose(
                        head, other, atol=1e-5, rtol=1e-4
                    ):
                        raise ValueError("Internal/head/export numerical parity failed")
                    if not np.array_equal(
                        self.order(reference, ids), self.order(other.cpu().numpy(), ids)
                    ):
                        raise ValueError("Diagnostic ordered-ranking parity failed")
                    maximum = max(maximum, float((head - other).abs().max()))
                sequences.append(sequence)
                singles.append(head)
            batch = self.score(
                self.query(torch.cat(sequences)),
                torch.tensor([ids] * 3, device=self.device),
            )
            for single, batched in zip(singles, batch):
                if not torch.allclose(single, batched, atol=1e-5, rtol=1e-4):
                    raise ValueError("Mixed-length batch parity failed")
                if not np.array_equal(
                    self.order(single.cpu().numpy(), ids),
                    self.order(batched.cpu().numpy(), ids),
                ):
                    raise ValueError("Mixed-length ranking parity failed")
                maximum = max(maximum, float((single - batched).abs().max()))
            return {
                "available": True,
                "passed": True,
                "category": self.category,
                "cases": 3,
                "checkpoint_state_identity": True,
                "mapping_identity": True,
                "maximum_absolute_error": maximum,
                "atol": 1e-5,
                "rtol": 1e-4,
            }
        finally:
            if hasattr(module, "_models"):
                module._models.clear()

    def run(self):
        import hashlib
        import time
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        histories = self.view.user_train
        histories = histories.get("History", histories)
        histories = {str(u): list(h) for u, h in histories.items()}
        result = {
            "available": False,
            "records": self.records,
            "failures": self.failures,
            "protocol": {
                "seed": 42,
                "draws": 3,
                "users_requested": 32,
                "gpu_soft_cap_seconds": 15,
                "total_soft_cap_seconds": 60,
                "frequency_edges": [1, 4, 16, 64, 256],
                "history_edges": [4, 8, 16, 32],
                "regimes": list(self.REGIMES),
            },
            "full_catalog_reference": self.unavailable(
                "Not run under the declared cap"
            ),
            "four_category_macro": self.unavailable(
                "Requires all four category results"
            ),
        }
        try:
            try:
                parity = self.timed(lambda: self.parity(histories))
            except TimeoutError:
                raise
            except Exception as exc:
                parity = self.unavailable(
                    f"{type(exc).__name__}: {exc}",
                    passed=False,
                    status="ERROR",
                )
            result["serving_parity"] = parity
            if not parity.get("passed", False):
                result["reason"] = parity.get("reason", "Serving parity did not pass")
                return result
            catalog = np.arange(1, self.net.itemnum + 1, dtype=np.int64)
            frequency = np.zeros(self.net.itemnum + 1, dtype=np.int64)
            valid_histories = {}
            invalid = 0
            for user, history in histories.items():
                self.boundary()
                try:
                    h = [int(i) for i in history]
                    if any(i < 1 or i > self.net.itemnum for i in h):
                        raise ValueError("Invalid real item ID")
                    valid_histories[user] = h
                    np.add.at(frequency, h, 1)
                except (ValueError, TypeError):
                    invalid += 1
            bands = np.searchsorted([1, 4, 16, 64, 256], frequency, side="right")
            selected = np.random.default_rng(42).permutation(sorted(histories))[:32]
            endpoints = []
            excluded = Counter()
            for index, user in enumerate(selected):
                h = valid_histories.get(str(user))
                if h is None:
                    excluded["invalid_history"] += 1
                    continue
                seen, eligible = set(), []
                for endpoint, target in enumerate(h):
                    if endpoint >= 4 and target not in seen:
                        eligible.append(endpoint)
                    seen.add(target)
                if not eligible:
                    excluded["no_eligible_endpoint"] += 1
                    continue
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [42, self.ROUTES.index(self.category), index]
                    )
                )
                endpoint = int(rng.choice(eligible))
                if min(endpoint, self.net.maxlen - 1) < 4:
                    excluded["insufficient_visible_history"] += 1
                    continue
                endpoints.append(
                    (index, str(user), endpoint, h[:endpoint], h[endpoint])
                )
            result["selection"] = {
                "selected_users": len(selected),
                "endpoints": len(endpoints),
                "invalid_catalog_histories": invalid,
                "exclusions": dict(excluded),
            }
            cpu_start = time.perf_counter()
            texts = []
            metadata = self.view.meta_data or {}

            def text(value):
                if isinstance(value, str):
                    return value
                if isinstance(value, (tuple, list)):
                    return " ".join(v for v in value if isinstance(v, str))
                return ""

            for item in catalog:
                self.boundary()
                info = metadata.get(str(int(item)), metadata.get(int(item), {}))
                info = info if isinstance(info, dict) else {}
                texts.append(
                    text(info.get("title")) + " " + text(info.get("description"))
                )
            vectorizer = TfidfVectorizer(
                lowercase=True,
                analyzer="word",
                token_pattern=r"(?u)\b\w\w+\b",
                ngram_range=(1, 1),
                min_df=1,
                max_df=1.0,
                max_features=50000,
                norm="l2",
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=False,
                dtype=np.float32,
            )
            matrix, text_error = None, None
            try:
                matrix = vectorizer.fit_transform(texts)
            except ValueError as exc:
                text_error = str(exc)
            self.cpu_seconds += time.perf_counter() - cpu_start
            result["text"] = {
                "available": matrix is not None,
                "reason": text_error,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "torch_version": str(torch.__version__),
                "catalog_text_sha256": hashlib.sha256(
                    json.dumps(texts, ensure_ascii=False).encode()
                ).hexdigest(),
                "vocabulary_sha256": hashlib.sha256(
                    json.dumps(
                        (
                            {
                                token: int(index)
                                for token, index in vectorizer.vocabulary_.items()
                            }
                            if matrix is not None
                            else {}
                        ),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            }
            nonzero = np.zeros(self.net.itemnum + 1, dtype=bool)
            if matrix is not None:
                nonzero[1:] = np.diff(matrix.indptr) > 0

            def sample(pool, count, user_index, endpoint, draw, regime, band=0):
                if len(pool) < count:
                    return None
                rng = np.random.Generator(
                    np.random.PCG64(
                        np.random.SeedSequence(
                            [
                                42,
                                self.ROUTES.index(self.category),
                                user_index,
                                endpoint,
                                draw,
                                regime,
                                band,
                            ]
                        )
                    )
                )
                return rng.choice(pool, count, replace=False).tolist()

            for user_index, user, endpoint, prefix, target in endpoints:
                self.boundary()
                _, sequence = self.inputs(prefix)
                query = self.timed(lambda: self.query(sequence))
                eligible = catalog[~np.isin(catalog, sorted(set(prefix) | {target}))]
                eligible_text = eligible[nonzero[eligible]]
                lexical = np.asarray([], dtype=np.int64)
                if matrix is not None and nonzero[target]:
                    cpu_start = time.perf_counter()
                    similarities = cosine_similarity(
                        matrix[target - 1], matrix, dense_output=False
                    ).tocoo()
                    ids = similarities.col + 1
                    keep = (similarities.data > 0) & np.isin(ids, eligible_text)
                    ids, values = ids[keep], similarities.data[keep]
                    lexical = ids[np.lexsort((ids, -values))][:500]
                    self.cpu_seconds += time.perf_counter() - cpu_start
                for draw in range(3):
                    pools = [eligible, eligible[bands[eligible] == bands[target]]]
                    pools += [
                        eligible_text,
                        eligible_text[bands[eligible_text] == bands[target]],
                        lexical,
                    ]
                    chosen = {}
                    for regime, pool in enumerate(pools):
                        text_required = regime >= 2
                        negatives = (
                            None
                            if text_required and not nonzero[target]
                            else sample(pool, 99, user_index, endpoint, draw, regime)
                        )
                        chosen[regime] = negatives
                    if chosen[4] is not None:
                        control, deficiencies = [], []
                        outside = eligible_text[~np.isin(eligible_text, lexical)]
                        for band in range(6):
                            count = int(np.count_nonzero(bands[chosen[4]] == band))
                            pool = outside[bands[outside] == band]
                            sampled = sample(
                                pool, count, user_index, endpoint, draw, 5, band
                            )
                            if sampled is None:
                                deficiencies.append(
                                    {
                                        "band": band,
                                        "required": count,
                                        "available": len(pool),
                                    }
                                )
                            else:
                                control.extend(sampled)
                        chosen[5] = None if deficiencies else control
                    else:
                        deficiencies, chosen[5] = [], None
                    for regime in range(6):
                        self.boundary()
                        negatives = chosen[regime]
                        if negatives is None:
                            self.failures.append(
                                {
                                    "user": user,
                                    "endpoint": endpoint,
                                    "draw": draw,
                                    "regime": self.REGIMES[regime],
                                    "reason": "Insufficient declared support or missing target text",
                                    "support": (
                                        len(pools[regime]) if regime < 5 else None
                                    ),
                                    "deficient_bands": (
                                        deficiencies if regime == 5 else []
                                    ),
                                }
                            )
                            continue
                        ids = [target] + negatives
                        tensor = torch.tensor(
                            [ids], device=self.device, dtype=torch.long
                        )
                        scores = (
                            self.timed(lambda: self.score(query, tensor))[0]
                            .cpu()
                            .numpy()
                        )
                        rank = int(np.flatnonzero(self.order(scores, ids) == 0)[0]) + 1
                        summaries = []
                        for band in range(6):
                            values = frequency[
                                np.asarray(negatives)[bands[negatives] == band]
                            ]
                            summaries.append(
                                {
                                    "band": band,
                                    "count": len(values),
                                    "mean_frequency": (
                                        float(values.mean()) if len(values) else None
                                    ),
                                }
                            )
                        self.records.append(
                            {
                                "user": user,
                                "selected_user_index": user_index,
                                "endpoint": endpoint,
                                "draw": draw,
                                "regime": self.REGIMES[regime],
                                "ids": ids,
                                "scores": scores.tolist(),
                                "rank": rank,
                                "hit10": float(rank <= 10),
                                "percentile": (rank - 1) / 99.0,
                                "margin": float(scores[0] - np.sort(scores[1:])[-10]),
                                "frequency_band": int(bands[target]),
                                "history_bin": int(
                                    np.searchsorted(
                                        [8, 16, 32],
                                        min(len(prefix), self.net.maxlen - 1),
                                        side="right",
                                    )
                                ),
                                "negative_frequency": summaries,
                            }
                        )
            result["available"] = bool(self.records)
            result["partial"] = False
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
            result["partial"] = True
            result["available"] = bool(self.records)
        finally:
            # Include CPU summaries in the reported total audit cost.
            result["summary"] = self.summarize()
            elapsed = time.perf_counter() - self.started
            result["cost"] = {
                "cpu_text_seconds": self.cpu_seconds,
                "scoring_gpu_seconds": (
                    self.gpu_seconds if self.device.type == "cuda" else None
                ),
                "scoring_cpu_seconds": (
                    self.gpu_seconds if self.device.type == "cpu" else None
                ),
                "synchronized_scoring_wall_seconds": self.scoring_wall,
                "total_wall_seconds": elapsed,
                "gpu_cap_overrun_seconds": (
                    max(0.0, self.gpu_seconds - 15.0)
                    if self.device.type == "cuda"
                    else None
                ),
                "total_cap_overrun_seconds": max(0.0, elapsed - 60.0),
            }
        return result

    def summarize(self):
        by_key = {(r["user"], r["regime"], r["draw"]): r for r in self.records}
        users = sorted({r["user"] for r in self.records})
        primary = self.REGIMES[2:]
        common = [
            u
            for u in users
            if all((u, name, d) in by_key for name in primary for d in range(3))
        ]
        coverage = {
            name: {
                "completed_records": sum(r["regime"] == name for r in self.records),
                "complete_users": sum(
                    all((u, name, d) in by_key for d in range(3)) for u in users
                ),
            }
            for name in self.REGIMES
        }
        # Descriptive regime means use users with all three completed draws.
        # Partial records remain available without entering these estimates.
        regime_summaries = {}
        for name in self.REGIMES:
            complete = [
                u for u in users if all((u, name, d) in by_key for d in range(3))
            ]
            regime_summaries[name] = {
                "users": len(complete),
                "draws": 3 * len(complete),
                "means": {
                    metric: (
                        float(
                            np.mean(
                                [
                                    np.mean(
                                        [by_key[u, name, d][metric] for d in range(3)]
                                    )
                                    for u in complete
                                ]
                            )
                        )
                        if complete
                        else None
                    )
                    for metric in ("hit10", "rank", "margin", "percentile")
                },
                "unavailable_draws": sum(
                    row["regime"] == name for row in self.failures
                ),
                "missingness_reasons": dict(
                    Counter(
                        row["reason"] for row in self.failures if row["regime"] == name
                    )
                ),
            }
        contrasts = {}
        for left, right in (
            (primary[0], primary[1]),
            (primary[0], primary[2]),
            (primary[3], primary[2]),
        ):
            complete = [
                u
                for u in users
                if all(
                    (u, name, d) in by_key for name in (left, right) for d in range(3)
                )
            ]

            def aggregate(panel):
                if not panel:
                    return {"users": 0, "statistics": None}
                stats = {}
                for metric in ("hit10", "rank", "margin", "percentile"):
                    differences = np.asarray(
                        [
                            [
                                by_key[u, right, d][metric] - by_key[u, left, d][metric]
                                for d in range(3)
                            ]
                            for u in panel
                        ]
                    )
                    means = differences.mean(1)
                    indices = np.random.default_rng(42).integers(
                        0, len(panel), (500, len(panel))
                    )
                    draw_means = differences.mean(0)
                    stats[metric] = {
                        "mean": float(means.mean()),
                        "user_bootstrap_ci95": np.percentile(
                            means[indices].mean(1), [2.5, 97.5]
                        ).tolist(),
                        "per_draw_means": draw_means.tolist(),
                        "draw_std": float(draw_means.std(ddof=1)),
                        "draw_range": float(np.ptp(draw_means)),
                    }
                flips = Counter(
                    (
                        int(by_key[u, left, d]["hit10"]),
                        int(by_key[u, right, d]["hit10"]),
                    )
                    for u in panel
                    for d in range(3)
                )
                return {
                    "users": len(panel),
                    "draws": 3 * len(panel),
                    "statistics": stats,
                    "hit_flips": {f"{a}->{b}": n for (a, b), n in flips.items()},
                }

            strata = {}
            for field, bins in (
                ("frequency_band", range(6)),
                ("history_bin", range(4)),
            ):
                strata[field] = {
                    str(b): aggregate(
                        [u for u in complete if by_key[u, left, 0][field] == b]
                    )
                    for b in bins
                }
            contrasts[right + "_minus_" + left] = {
                "paired": aggregate(complete),
                "strata": strata,
                "all_four_regime_common_support": aggregate(
                    [u for u in common if u in complete]
                ),
            }
        return {
            "coverage": coverage,
            "regime_summaries": regime_summaries,
            "contrasts": contrasts,
            "all_four_regime_common_users": len(common),
        }


### <<< Self_EvolveRec-BLOCK-END


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the explicit ModelView diagnostic interface
    def __init__(self, view, device):
        self.model = view
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Respect unavailable item-category metadata
    def _get_categories(self, item_id: int):
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

            n = emb.size(0)
            if n <= 2:
                return 0.0

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

            seq_np = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            log_feats = self.model.log2feats(seq_np)

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            valid = pos_t != 0
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            valid_cpu = pos_cpu != 0

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
                "mean": 0.0,
                "p10": 0.0,
                "neg_beats_pos_rate": 0.0,
                "small_margin_rate": 0.0,
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

    ### >>> Self_EvolveRec-BLOCK-START: Independent measured probes and bounded serving-aligned candidate audit
    def run_full_diagnosis(self, train_loader):
        view, net = self.model, self.model.model
        modes = [(module, module.training) for module in net.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        buffers = {name: value.detach().clone() for name, value in net.named_buffers()}
        metrics = {}
        definitions = {
            "embedding_collapse_score": "Mapped ordered off-diagonal cosine over at most 2048 real embeddings, seed 42. PAD/MASK excluded. 0.5 means approximately zero cosine; neither collapse nor quality is established.",
            "training_margins": "Pooled positive-minus-negative logits at verified supervised MASK positions in at most ten cloned already-consumed batches. Mean, nearest-rank p10 and rates below 0 and 0.5; in-sample, scale-dependent measurements.",
            "sampler_contract": "Checks shapes, reserved tokens and all-MASK supervision in cached batches; cannot establish source provenance or proposal probabilities.",
            "candidate_audit": "Retrospective training-prefix candidate-composition sensitivity. Each available regime uses target plus 99 unique prefix/target-excluded real items. Frequencies use complete training histories; bands are zero, [1,4), [4,16), [16,64), [64,256), [256,infinity). Text regimes require nonzero target text and use nonzero-text eligible negatives. Lexical support is the top 500 strictly positive TF-IDF similarities, ordered by descending similarity then ascending ID. Histogram controls exclude that entire pool and match six frequency counts exactly. Diagnostic rank uses descending neural score then ascending item ID. Margin is target minus tenth-highest negative; percentile=(rank-1)/99. Contrast signs are right minus left. Each contrast requires three complete paired draws/user; user means receive equal weight. 500 seed-42 user bootstrap replicates condition on checkpoint, selected support and fixed draws, excluding seed/search uncertainty and candidate-draw variability. Three per-draw means/std/range describe coarse draw variability separately. Shared-item dependence and support selection remain limitations. No causal semantic, temporal-generalization, relevance or host-validation claim follows. TF-IDF/frequencies construct diagnostic candidates only and never contribute serving scores. Missing authoritative mapping or current serialized parity evidence makes comparisons unavailable.",
            "item_category_breakdown": "Unavailable: item categories, ratings, reviews, prices, stores and simulator traits are not supplied.",
        }
        try:
            # The supplied BERT4Rec forward has no mutable caches. Unknown hooks
            # require an explicit isolation adapter rather than silent execution.
            hooked = any(
                m._forward_hooks or m._forward_pre_hooks for m in net.modules()
            )
            net.eval()
            with torch.no_grad():
                try:
                    weight = view.item_emb.weight.detach()[1 : net.itemnum + 1].float()
                    count = min(2048, len(weight))
                    ids = torch.randperm(
                        len(weight), generator=torch.Generator().manual_seed(42)
                    )[:count].to(weight.device)
                    vectors = F.normalize(weight[ids], dim=-1)
                    value = None
                    if count >= 2 and bool(torch.isfinite(vectors).all()):
                        cosine = (
                            vectors.sum(0).square().sum() - vectors.square().sum()
                        ) / (count * (count - 1))
                        value = float((cosine.clamp(-1, 1) + 1) / 2)
                    metrics["embedding_collapse_score"] = value
                    metrics["embedding_sample_count"] = count
                except Exception as exc:
                    metrics["embedding_collapse_score"] = None
                    metrics["embedding_error"] = str(exc)
                source = (
                    train_loader
                    if isinstance(train_loader, (list, tuple))
                    else getattr(train_loader, "diagnostic_replay_batches", None)
                )
                if source is None:
                    source = getattr(view, "diagnostic_replay_batches", None)
                margins, errors, valid_batches = [], [], 0
                if isinstance(source, (list, tuple)) and not hooked:
                    for batch in source[:10]:
                        try:
                            if len(batch) != 4:
                                raise ValueError(
                                    "Expected (None, sequence, positive, negative)"
                                )
                            # Validate original IDs before conversion; casting
                            # floating inputs would conceal a malformed replay.
                            tensors = [torch.as_tensor(x) for x in batch[1:]]
                            if any(x.dtype != torch.long for x in tensors):
                                raise ValueError("Replay IDs must already be int64")
                            seq, pos, neg = [
                                x.to(self.device).detach().clone() for x in tensors
                            ]
                            if (
                                seq.ndim != 2
                                or seq.shape != pos.shape
                                or seq.shape != neg.shape
                                or seq.shape[0] == 0
                                or seq.shape[1] != net.maxlen
                            ):
                                raise ValueError("Expected nonempty B x maxlen tensors")
                            if (
                                net.mask_token != net.itemnum + 1
                                or view.item_emb.num_embeddings != net.itemnum + 2
                                or view.item_emb.padding_idx != 0
                            ):
                                raise ValueError("Reserved-token configuration failed")
                            valid = pos.ne(0)
                            if not bool(
                                valid.any(1).all()
                                and seq[valid].eq(net.mask_token).all()
                                and ((seq >= 0) & (seq <= net.mask_token)).all()
                                and (
                                    (pos[valid] >= 1) & (pos[valid] <= net.itemnum)
                                ).all()
                                and (
                                    (neg[valid] >= 1) & (neg[valid] <= net.itemnum)
                                ).all()
                                and pos[valid].ne(neg[valid]).all()
                                and neg[~valid].eq(0).all()
                            ):
                                raise ValueError("Masked-target contract failed")
                            features = view.log2feats(seq.cpu().numpy().copy())
                            scores = view.predict_score(
                                features, torch.stack((pos, neg), -1)
                            )
                            if scores.shape != (*pos.shape, 2):
                                raise ValueError("Unexpected head alignment")
                            values = (
                                (scores[..., 0] - scores[..., 1])[valid].float().cpu()
                            )
                            if not bool(torch.isfinite(values).all()):
                                raise ValueError("Nonfinite margins")
                            margins.append(values)
                            valid_batches += 1
                        except Exception as exc:
                            errors.append(str(exc))
                metrics["sampler_contract"] = {
                    "available": isinstance(source, (list, tuple)) and not hooked,
                    "valid_batches": valid_batches,
                    "errors": errors,
                    "passed": (False if errors else True if valid_batches else None),
                    "reason": (
                        None
                        if valid_batches
                        else "No usable cloned replay batches; live iterator was not advanced"
                    ),
                }
                metrics["training_margins"] = None
                if margins:
                    values = torch.cat(margins)
                    metrics["training_margins"] = {
                        "count": values.numel(),
                        "mean": float(values.mean()),
                        "p10": float(
                            values.kthvalue(
                                max(1, int(np.ceil(0.1 * values.numel())))
                            ).values
                        ),
                        "negative_beats_positive_rate": float(
                            (values < 0).float().mean()
                        ),
                        "small_margin_rate": float((values < 0.5).float().mean()),
                    }
                metrics["item_category_breakdown"] = {
                    "available": False,
                    "reason": "Item categories are unavailable",
                }
                try:
                    metrics["candidate_audit"] = (
                        _CandidateAudit.unavailable(
                            "Existing forward hooks require an audited isolation adapter"
                        )
                        if hooked
                        else _CandidateAudit(view, self.device).run()
                    )
                except Exception as exc:
                    metrics["candidate_audit"] = _CandidateAudit.unavailable(
                        f"{type(exc).__name__}: {exc}"
                    )
        finally:
            with torch.no_grad():
                for name, buffer in net.named_buffers():
                    buffer.copy_(buffers[name])
            for module, mode in modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
