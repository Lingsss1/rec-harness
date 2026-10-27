import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Deterministic diagnostics and exception-safe evaluation
import random
from collections import Counter
import os
import hashlib
from functools import wraps


def _evaluation_probe(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return fn(self, *args, **kwargs)
        finally:
            self.model.train(previous_mode)

    return wrapped


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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret training evidence with coverage and uncertainty
    - Every statement must be grounded in the provided metric values/definitions.
    - Preserve counts, coverage, unavailable states, and dataset scope when citing evidence.
    - Training-fit ranks do not measure validation generalization or recommendation diversity.
    - Separate observed confusability from hypotheses about its cause; unknown distractor relevance can produce false negatives.
    - A triggered but unexecuted ablation is a hypothesis, not evidence of improved validation HR@10.
    - Missing measurements alone must not imply either STABLE or CRITICAL model health.
    - Do not infer unavailable user traits, review sentiment, ratings, or product categories.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the view contract and enforce permitted metadata
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {}) or {}
        self.review_data = getattr(view, "review_data", {}) or {}
        self.user_train = view.user_train
        self.dataset = str(getattr(view, "dataset_name", "view_local_catalog"))
        self._frozen = None

    def _get_categories(self, item_id: int):
        # Product categories are unavailable in this benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Read embeddings through the compatibility view
            emb = self.view.item_emb.weight.detach()
            ### <<< Self_EvolveRec-BLOCK-END
            emb = emb[1:]

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

    ### >>> Self_EvolveRec-BLOCK-START: Restore model mode for standalone margin calls
    @_evaluation_probe
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        ### <<< Self_EvolveRec-BLOCK-END
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

            ### >>> Self_EvolveRec-BLOCK-START: Pair each loader target with its own causal position
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if seq_t.shape != pos_t.shape or seq_t.shape != neg_t.shape:
                raise ValueError("Diagnostic loader must provide B x L tensors")
            log_feats = self._features(seq_t)
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            margin = torch.zeros_like(pos_t, dtype=log_feats.dtype)
            if valid.any():
                logits = self._scores(log_feats[valid], all_items[valid])
                margin[valid] = logits[:, 0] - logits[:, 1]
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Reuse the complete loader eligibility mask
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
            ### >>> Self_EvolveRec-BLOCK-START: Represent missing margin evidence explicitly
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

        ### >>> Self_EvolveRec-BLOCK-START: Publish margin coverage and metadata availability
        breakdown = {
            "category_status": "unavailable: product categories are not supplied",
            "valid_positions": sum(int(m.numel()) for m in margins_all),
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

    ### >>> Self_EvolveRec-BLOCK-START: Cached frozen challenges, measured temporal controls and explicit scoring checks
    def _seed(self, *parts):
        payload = json.dumps([42, self.dataset, *parts], ensure_ascii=True)
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

    def _features(self, sequence):
        # The original compatibility hook accepts numpy sequences.
        values = self.view.log2feats(sequence.detach().cpu().numpy())
        if values.ndim != 3 or values.shape[:2] != sequence.shape:
            raise ValueError("log2feats must return B x L x D")
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite sequence features")
        return values

    def _scores(self, features, items):
        if features.ndim != 2:
            raise ValueError("Candidate scoring requires B x D features")
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        if items.ndim == 1:
            items = items.unsqueeze(0).expand(features.shape[0], -1)
        if items.ndim != 2 or items.shape[0] != features.shape[0]:
            raise ValueError("Candidates must be C or B x C; features must be B x D")
        itemnum = self.view.item_emb.num_embeddings - 1
        if items.numel() and (items.min() < 1 or items.max() > itemnum):
            raise ValueError("Padding or invalid candidate ID")
        # Explicit singleton sequence dimension preserves the initial B x L x C
        # compatibility contract, including B=1 and C=1.
        scores = self.view.predict_score(features[:, None, :], items[:, None, :])
        if scores.shape != (features.shape[0], 1, items.shape[1]):
            raise ValueError("predict_score must return B x L x C")
        scores = scores[:, 0, :]
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite candidate scores")
        return scores

    @staticmethod
    def _rank(scores, ids):
        scores = np.asarray(scores, dtype=np.float64)
        ids = np.asarray(ids, dtype=np.int64)
        return (
            int(np.sum((scores > scores[0]) | ((scores == scores[0]) & (ids < ids[0]))))
            + 1
        )

    def _freeze(self):
        if self._frozen is not None:
            return self._frozen
        histories = self.user_train.get("History", self.user_train)
        itemnum = self.view.item_emb.num_embeddings - 1
        counts = np.zeros(itemnum + 1, dtype=np.int64)
        for history in histories.values():
            h = np.asarray(history, dtype=np.int64)
            if h.size and (h.min() < 1 or h.max() > itemnum):
                raise ValueError("History item outside this view's catalog")
            np.add.at(counts, h, 1)
        positive = counts[1:][counts[1:] > 0]
        boundaries = (
            np.quantile(positive, np.arange(0.1, 1.0, 0.1)).tolist()
            if positive.size
            else []
        )
        bins = np.zeros(itemnum + 1, dtype=np.int64)
        mask = counts > 0
        bins[mask] = 1 + np.searchsorted(boundaries, counts[mask], side="right")
        length = int(getattr(self.model, "maxlen", 128))
        if length < 1:
            raise ValueError("Invalid model maxlen")
        users = sorted(
            (u for u, h in histories.items() if len(h) >= 2),
            key=lambda u: (self._seed("user", str(u)), str(u)),
        )[:32]
        records = []
        for u in users:
            h = [int(i) for i in histories[u]]
            rng = np.random.default_rng(self._seed("positions", str(u)))
            positions = sorted(
                rng.choice(
                    np.arange(1, len(h)), size=min(2, len(h) - 1), replace=False
                ).tolist()
            )
            for t in positions:
                prefix = h[:t]
                packed = [0] * (length - min(t, length)) + prefix[-length:]
                assert packed[-1] == h[t - 1]
                assert [i for i in packed if i] == prefix[-length:]
                records.append(
                    {
                        "user": str(u),
                        "index": t,
                        "target": h[t],
                        "prefix": prefix,
                        "sequence": packed,
                        "repeated": h[t] in prefix,
                    }
                )

        # Fit only permitted title/description text, in ascending item-ID order.
        texts = []
        for i in range(1, itemnum + 1):
            info = self.meta_data.get(str(i), self.meta_data.get(i, {}))
            if not isinstance(info, dict):
                info = {}
            parts = []
            for field in ("title", "description"):
                value = info.get(field, "")
                if isinstance(value, list):
                    value = " ".join(v for v in value if isinstance(v, str))
                if isinstance(value, str):
                    parts.append(value)
            texts.append(" ".join(" ".join(parts).lower().split())[:2048])
        word_counts = np.array([len(t.split()) for t in texts])
        strata = np.searchsorted([16, 64], word_counts, side="left")
        matrix = None
        vectorizer = None
        text_status = "unavailable: no nonzero permitted text"
        if any(texts):
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                lowercase=False,
                strip_accents="unicode",
                ngram_range=(1, 1),
                token_pattern=r"(?u)\b\w\w+\b",
                min_df=1,
                max_df=1.0,
                max_features=20000,
                sublinear_tf=True,
                norm="l2",
                dtype=np.float32,
            )
            try:
                matrix = vectorizer.fit_transform(texts)
                text_status = "available"
            except ValueError as exc:
                # An empty vocabulary is a coverage condition, not a zero score.
                if "empty vocabulary" not in str(exc):
                    raise
        text_ok = (
            np.asarray(matrix.getnnz(axis=1)).ravel() > 0
            if matrix is not None
            else np.zeros(itemnum, dtype=bool)
        )
        self._frozen = {
            "records": records,
            "counts": counts,
            "bins": bins,
            "boundaries": boundaries,
            "matrix": matrix,
            "vectorizer": vectorizer,
            "strata": strata,
            "text_ok": text_ok,
            "text_status": text_status,
            "itemnum": itemnum,
        }
        # Timestamp controls require aligned, observed timestamps for every
        # training event. Partial coverage is reported rather than imputed.
        times = self.user_train.get("Time", {})
        decayed = None
        temporal = {"status": "unavailable: missing or unaligned training timestamps"}
        if times and all(len(times.get(u, [])) == len(h) for u, h in histories.items()):
            from datetime import datetime, timezone

            parsed = []
            try:
                for u, history in histories.items():
                    for item, value in zip(history, times[u]):
                        text = str(value).strip()
                        try:
                            stamp = float(text)
                        except ValueError:
                            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            stamp = dt.timestamp()
                        else:
                            # Public Amazon timestamps are Unix seconds or
                            # milliseconds; keep the conversion rule explicit.
                            if abs(stamp) >= 1e11:
                                stamp /= 1000.0
                        if not np.isfinite(stamp):
                            raise ValueError("nonfinite timestamp")
                        parsed.append((int(item), stamp))
            except (ValueError, OverflowError) as exc:
                temporal = {
                    "status": "unavailable: invalid observed timestamp",
                    "reason": str(exc),
                }
            else:
                if parsed:
                    reference = max(stamp for _, stamp in parsed)
                    tau = 180.0 * 86400.0
                    decayed = np.zeros(itemnum + 1, dtype=np.float64)
                    for item, stamp in parsed:
                        decayed[item] += np.exp(-(reference - stamp) / tau)
                    temporal = {
                        "status": "measured",
                        "reference_unix_seconds": reference,
                        "decay_seconds": tau,
                        "events": len(parsed),
                        "rule": "exp(-(reference - timestamp)/tau); numeric milliseconds >=1e11; naive ISO treated as UTC",
                    }
        self._frozen.update(
            {
                "decayed_counts": decayed,
                "temporal": temporal,
                "candidate_cache": {},
                "similarity_cache": {},
            }
        )
        return self._frozen

    def _challenges(self, record, replicate, frozen):
        target = record["target"]
        cache_key = (record["user"], record["index"], target, replicate)
        if cache_key in frozen["candidate_cache"]:
            return frozen["candidate_cache"][cache_key]

        def rng_for(challenge):
            return np.random.default_rng(
                self._seed(
                    record["user"], record["index"], target, challenge, replicate
                )
            )

        ids = np.arange(1, frozen["itemnum"] + 1)
        excluded = set(record["prefix"]) | {target}
        eligible = ids[~np.isin(ids, list(excluded))]
        unavailable = {}
        if len(eligible) < 99:
            result = ({}, {"all": "fewer than 99 distinct eligible negatives"})
            frozen["candidate_cache"][cache_key] = result
            return result
        candidates = {
            "conditioned_uniform": rng_for("conditioned_uniform").choice(
                eligible, 99, replace=False
            )
        }
        bins, counts = frozen["bins"], frozen["counts"]
        same = eligible[bins[eligible] == bins[target]]
        if len(same) >= 99:
            pool = same
        else:
            outside = eligible[bins[eligible] != bins[target]]
            distance = np.abs(np.log1p(counts[outside]) - np.log1p(counts[target]))
            outside = outside[np.lexsort((outside, distance))]
            pool = np.concatenate([same, outside[: 99 - len(same)]])
        candidates["popularity_matched"] = rng_for("popularity_matched").choice(
            pool, 99, replace=False
        )
        matrix = frozen["matrix"]
        text_pool = eligible[frozen["text_ok"][eligible - 1]]
        if matrix is None or len(text_pool) < 99:
            unavailable["semantic"] = "fewer than 99 eligible nonzero text vectors"
        else:
            # The same random draw supplies quotas for both semantic queries.
            reference = rng_for("semantic_quota_reference").choice(
                text_pool, 99, replace=False
            )
            cell = bins * 3
            cell = cell.copy()
            cell[1:] += frozen["strata"]
            for name, query in (
                ("latest_text", record["prefix"][-1]),
                ("target_text", target),
            ):
                if not frozen["text_ok"][query - 1]:
                    unavailable[name] = "query has missing or zero-vector text"
                    continue
                # Reuse catalog cosine scores across the three frozen draws.
                if query not in frozen["similarity_cache"]:
                    frozen["similarity_cache"][query] = (
                        (matrix @ matrix[query - 1].T).toarray().ravel()
                    )
                similarities = frozen["similarity_cache"][query]
                chosen = []
                for c in sorted(set(cell[reference].tolist())):
                    quota = int(np.sum(cell[reference] == c))
                    pool = text_pool[cell[text_pool] == c]
                    order = np.lexsort((pool, -similarities[pool - 1]))
                    chosen.extend(pool[order[:quota]].tolist())
                candidates[name] = np.asarray(chosen, dtype=np.int64)
                candidates[name + "_quota_random"] = reference.copy()
        for name, negatives in list(candidates.items()):
            assert len(negatives) == 99 and len(set(negatives.tolist())) == 99
            assert not (set(negatives.tolist()) & excluded)
            candidates[name] = np.concatenate([[target], negatives])
            assert np.count_nonzero(candidates[name] == target) == 1
            candidates[name].setflags(write=False)
        result = (candidates, unavailable)
        frozen["candidate_cache"][cache_key] = result
        return result

    def _checks(self, sequences, features):
        if not len(sequences):
            return {"status": "unavailable: no records"}
        itemnum = self.view.item_emb.num_embeddings - 1
        ids = torch.arange(1, min(itemnum, 3) + 1, device=features.device)
        f = features[: min(2, len(features))]
        scores = self._scores(f, ids)
        # Validate the compatibility hook against the actual SASRec dot-product
        # scorer, rather than only comparing the hook against itself.
        embeddings = self.view.item_emb(ids)
        direct = f @ embeddings.transpose(0, 1)
        torch.testing.assert_close(scores, direct, rtol=1e-4, atol=1e-5)
        per_example_ids = ids.expand(len(f), -1)
        direct_per_example = (f[:, None, :] * self.view.item_emb(per_example_ids)).sum(
            -1
        )
        torch.testing.assert_close(
            self._scores(f, per_example_ids), direct_per_example, rtol=1e-4, atol=1e-5
        )
        scalar = torch.stack(
            [
                torch.stack(
                    [
                        self._scores(row[None], ids[j : j + 1])[0, 0]
                        for j in range(len(ids))
                    ]
                )
                for row in f
            ]
        )
        torch.testing.assert_close(scores, scalar, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            self._scores(f, ids.flip(0)), scores.flip(1), rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            self._scores(f, ids.expand(len(f), -1)), scores, rtol=1e-4, atol=1e-5
        )
        causal_checks = 0
        for sequence in sequences[:8]:
            positions = torch.nonzero(sequence, as_tuple=False).flatten()
            if len(positions) < 2:
                continue
            pivot = int(positions[-2])
            changed = sequence.clone()
            changed[pivot + 1 :] = changed[pivot + 1 :] % itemnum + 1
            before = self._features(sequence[None])[0, pivot]
            after = self._features(changed[None])[0, pivot]
            torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
            causal_checks += 1
        return {
            "status": "passed",
            "scalar_and_batch_shapes": True,
            "candidate_permutation": True,
            "causal_checks": causal_checks,
        }

    def _training_fit(self):
        frozen = self._freeze()
        records = frozen["records"]
        if not records:
            return {"status": "unavailable: no nonempty training prefixes"}
        sequences = torch.tensor(
            [r["sequence"] for r in records], dtype=torch.long, device=self.device
        )
        # Repacked prefixes always use their final position. Repacking changes
        # absolute positions relative to earlier positions of a loader sequence.
        features = torch.cat(
            [self._features(s)[:, -1, :] for s in sequences.split(16)], dim=0
        )
        checks = self._checks(sequences, features)
        shuffled = sequences.clone()
        for row, r in enumerate(records):
            values = np.asarray([i for i in r["sequence"] if i], dtype=np.int64)
            rng = np.random.default_rng(self._seed(r["user"], r["index"], "shuffle"))
            shuffled[row, -len(values) :] = torch.as_tensor(
                rng.permutation(values), device=self.device
            )
        shuffled_features = torch.cat(
            [self._features(s)[:, -1, :] for s in shuffled.split(16)], dim=0
        )
        totals, skips, manifest = {}, Counter(), []
        paired = {
            name: [[] for _ in range(3)] for name in ("latest_text", "target_text")
        }
        expansions = []
        expansion_records = []
        if frozen["decayed_counts"] is None:
            skips["timestamp_popularity: " + frozen["temporal"]["status"]] += 1

        def add(name, recurrence, rank):
            for group in ("all", recurrence):
                key = name + "/" + group
                values = totals.setdefault(key, [])
                values.append(rank)

        # A fixed within-bin cyclic target permutation controls for popularity.
        permuted_targets = [r["target"] for r in records]
        groups = {}
        for i, r in enumerate(records):
            groups.setdefault(int(frozen["bins"][r["target"]]), []).append(i)
        for indices in groups.values():
            targets = [records[i]["target"] for i in indices]
            for i, target in zip(indices, targets[1:] + targets[:1]):
                permuted_targets[i] = target
        effective_permutations = 0
        for index, r in enumerate(records):
            recurrence = "repeated" if r["repeated"] else "novel"
            for replicate in range(3):
                challenges, missing = self._challenges(r, replicate, frozen)
                for name, reason in missing.items():
                    skips[name + ": " + reason] += 1
                entry = {
                    "user": r["user"],
                    "index": r["index"],
                    "target": r["target"],
                    "sequence": r["sequence"],
                    "repeated": r["repeated"],
                    "replicate": replicate,
                    "candidates": {k: v.tolist() for k, v in challenges.items()},
                    "unavailable": missing,
                }
                manifest.append(entry)
                hits = {}
                for name, ids in challenges.items():
                    values = (
                        self._scores(features[index : index + 1], ids)[0].cpu().numpy()
                    )
                    rank = self._rank(values, ids)
                    add(name + "/model", recurrence, rank)
                    hits[name] = float(rank <= 10)
                    shuffled_scores = (
                        self._scores(shuffled_features[index : index + 1], ids)[0]
                        .cpu()
                        .numpy()
                    )
                    add(
                        name + "/shuffled_prefix",
                        recurrence,
                        self._rank(shuffled_scores, ids),
                    )
                    add(
                        name + "/training_popularity",
                        recurrence,
                        self._rank(frozen["counts"][ids], ids),
                    )
                    if frozen["decayed_counts"] is not None:
                        add(
                            name + "/timestamp_popularity",
                            recurrence,
                            self._rank(frozen["decayed_counts"][ids], ids),
                        )
                    query = r["prefix"][-1]
                    if (
                        frozen["matrix"] is not None
                        and frozen["text_ok"][query - 1]
                        and frozen["text_ok"][ids - 1].all()
                    ):
                        text_scores = (
                            (frozen["matrix"][ids - 1] @ frozen["matrix"][query - 1].T)
                            .toarray()
                            .ravel()
                        )
                        add(
                            name + "/latest_text_only",
                            recurrence,
                            self._rank(text_scores, ids),
                        )
                    else:
                        skips[name + ": text-only baseline unavailable"] += 1
                    if name == "popularity_matched":
                        expansions.append(
                            float(
                                np.mean(
                                    frozen["bins"][ids[1:]]
                                    != frozen["bins"][r["target"]]
                                )
                            )
                        )
                if "popularity_matched" in challenges:
                    expansion_records.append(
                        {
                            "user": r["user"],
                            "index": r["index"],
                            "replicate": replicate,
                            "outside_target_bin_fraction": expansions[-1],
                        }
                    )
                for name in paired:
                    if name in hits and name + "_quota_random" in hits:
                        paired[name][replicate].append(
                            hits[name + "_quota_random"] - hits[name]
                        )
            target = permuted_targets[index]
            if target != r["target"]:
                control = dict(r, target=target)
                control_candidates, _ = self._challenges(control, 0, frozen)
                ids = control_candidates.get("conditioned_uniform")
                if ids is not None:
                    values = (
                        self._scores(features[index : index + 1], ids)[0].cpu().numpy()
                    )
                    add(
                        "permuted_target/model",
                        "permuted_label",
                        self._rank(values, ids),
                    )
                    manifest.append(
                        {
                            "control": "within_popularity_bin_target_permutation",
                            "user": r["user"],
                            "index": r["index"],
                            "candidates": ids.tolist(),
                        }
                    )
                    effective_permutations += 1

        summary = {}
        for name, ranks in totals.items():
            ranks = np.asarray(ranks, dtype=np.float64)
            summary[name] = {
                "n": len(ranks),
                "HR@10": float(np.mean(ranks <= 10)),
                "MRR": float(np.mean(1.0 / ranks)),
                "mean_rank": float(ranks.mean()),
            }
        gaps = {}
        for name, draws in paired.items():
            gaps[name] = {
                "paired_records_per_draw": [len(v) for v in draws],
                "random_minus_semantic_hr10": [
                    float(np.mean(v)) if v else None for v in draws
                ],
            }
        target_gap = gaps["target_text"]
        trigger = all(
            n >= 20 and gap is not None and gap >= 0.10
            for n, gap in zip(
                target_gap["paired_records_per_draw"],
                target_gap["random_minus_semantic_hr10"],
            )
        )
        settings = {
            "seed": 42,
            "dataset_scope": self.dataset,
            "records": len(records),
            "replicates": 3,
            "negative_count": 99,
            "max_records": 64,
            "prefix_rule": "left_pad(history[:t][-model.maxlen:])",
            "tie_policy": "descending score, ascending item ID",
            "zero_count_items": int(np.sum(frozen["counts"][1:] == 0)),
            "positive_count_decile_boundaries": frozen["boundaries"],
            "text_status": frozen["text_status"],
            "text_nonzero_items": int(frozen["text_ok"].sum()),
            "catalog_items": frozen["itemnum"],
            "semantic_fallback_rate": 0.0,
            "text_normalization": "title + description; lowercase; collapse whitespace; first 2048 characters",
            "text_length_strata": "normalized whitespace words: <=16, 17..64, >64",
            "semantic_comparison": "identical popularity-bin and text-length quotas; nonzero-text eligibility",
            "timestamp_popularity": frozen["temporal"],
            "unique_users": len({r["user"] for r in records}),
            "novel_records": sum(not r["repeated"] for r in records),
            "repeated_records": sum(r["repeated"] for r in records),
            "candidate_freeze": "cached by dataset-local user, interaction index, target and replicate; challenge-specific SHA256 seeds",
        }
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
        )
        os.makedirs(directory, exist_ok=True)
        safe_scope = hashlib.sha256(self.dataset.encode()).hexdigest()[:16]
        path = os.path.join(directory, safe_scope + "_frozen_challenges.json")
        with open(path, "w") as handle:
            json.dump(
                {
                    "settings": settings,
                    "records": manifest,
                    "popularity_pool_expansions": expansion_records,
                },
                handle,
                indent=2,
            )
        if frozen["vectorizer"] is not None and frozen["matrix"] is not None:
            vectorizer = frozen["vectorizer"]
            index_path = os.path.join(directory, safe_scope + "_text_index.npz")
            if not os.path.exists(index_path):
                from scipy.sparse import save_npz

                save_npz(index_path, frozen["matrix"])
                with open(
                    os.path.join(directory, safe_scope + "_text_settings.json"), "w"
                ) as handle:
                    json.dump(
                        {
                            "vocabulary": {
                                k: int(v) for k, v in vectorizer.vocabulary_.items()
                            },
                            "idf": vectorizer.idf_.tolist(),
                            "parameters": {
                                k: str(v) for k, v in vectorizer.get_params().items()
                            },
                            "normalization": settings["text_normalization"],
                        },
                        handle,
                    )
        return {
            "status": "measured",
            "settings": settings,
            "checks": checks,
            "ranks": summary,
            "unavailable_counts": dict(skips),
            "paired_semantic_gaps": gaps,
            "popularity_pool_expansion_rate": (
                float(np.mean(expansions)) if expansions else None
            ),
            "effective_target_permutation_records": effective_permutations,
            "manifest": path,
            "prespecified_ablation_rule": {
                "condition": "target-text random-minus-semantic HR@10 >= 0.10 in all three draws, each with >=20 paired records",
                "triggered": bool(trigger),
                "contrast": "one popularity-matched hard-negative mixture versus unchanged training loss and sampler",
                "executed": False,
                "selection": "unchanged pinned four-category macro validation HR@10 only",
            },
        }

    @_evaluation_probe
    def run_full_diagnosis(self, train_loader):
        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
                "training_fit_challenges": self._training_fit(),
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    "(mean off-diagonal cosine + 1)/2 over sampled nonpadding item embeddings. "
                    "Measures directional alignment, not recommendation diversity or catalog coverage. "
                    "The legacy value 0 for catalogs with <=2 items is unavailable evidence."
                ),
                "pos_neg_margin_mean": "Mean positive-minus-negative logit at the same loader position. Absolute magnitudes depend on score scale; null means no eligible positions.",
                "pos_neg_margin_p10": "Nearest-rank tenth percentile of aligned loader margins; null means unavailable.",
                "pos_neg_neg_beats_pos_rate": "Fraction of aligned eligible loader positions with strictly negative margin; ties are not counted as errors here.",
                "pos_neg_small_margin_rate": "Fraction of aligned eligible loader margins below eps=0.5; this threshold is score-scale dependent.",
                "pos_neg_margin_category_breakdown": "Position counts remain measured; all category lists are empty because product categories are unavailable.",
                "training_fit_challenges": (
                    "Training-only indexed targets h[t] use h[:t], truncated and left-padded, and features[:, -1, :]. "
                    "Earlier target occurrences are legitimate inputs. Repacking changes absolute positions. "
                    "Each rank uses one target and 99 distinct negatives excluding padding, target and every observed prefix item. "
                    "HR@10 and MRR are training-fit measurements, not validation or test metrics. Counts include repeated draws of the same records, not independent users. "
                    "The training sampler draws uniform negatives with replacement excluding the user's complete training history. "
                    "The conditioned_uniform challenge instead draws without replacement excluding the observed prefix and target, so it is not an exact sampler reproduction. "
                    "Popularity bins reserve bin zero for zero-count items; tied positive-count quantile boundaries are retained and ties enter the upper bin. "
                    "Popularity-matched pools expand by nearest log1p(count) distance, breaking distance ties by item ID. "
                    "Semantic and quota-random candidates share nonzero-text eligibility, popularity-bin quotas and prespecified text-length strata. "
                    "Latest-text queries use the final history item; target-text queries deliberately use the known training target solely to construct distractors. "
                    "Missing semantic queries are unavailable, with no fallback. Unknown relevance and possible false negatives limit preference-error claims. "
                    "Training-count popularity, observed-timestamp decayed popularity and latest-text-only ranks are shortcut baselines; prefix shuffles and within-bin target permutations are descriptive controls. "
                    "Timestamp popularity requires complete aligned observed training timestamps; otherwise its unavailable reason is reported. Its fixed reference is the latest training timestamp and its exponential decay constant is 180 days. "
                    "Target permutations may be ineffective for singleton or identical-target groups; effective counts are reported. "
                    "Positive random-minus-semantic gaps indicate greater confusability on common records; three draws are robustness checks, not confidence intervals. "
                    "The fixed ablation trigger is recorded but does not execute training or establish validation improvement. "
                    "Any ablation must retain seed 42, the host evaluator, all four routes, and account for diagnostics and retraining within the shared GPU budget. "
                    "No review fields, item ratings/categories/prices/stores or latent simulator traits are used or inferred. "
                    "These probes do not replace the native twenty-user full-catalog unseen-item recommendation workflow."
                ),
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
