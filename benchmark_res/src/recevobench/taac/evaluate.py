"""Independent Top-10 grading and feedback-only ranking diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from .common import digest, file_hash, read, rows, write


def keyed(path, key):
    result = {}
    for row in rows(path):
        identity = str(row[key])
        if identity in result:
            raise ValueError("Duplicate identity in " + str(path))
        result[identity] = row
    return result


def score(predictions, answers, candidate_ids, output=None):
    candidates = set(map(str, candidate_ids))
    predicted = keyed(predictions, "user_id")
    truth = keyed(answers, "user_id")
    if not truth or set(predicted) != set(truth):
        raise ValueError("Prediction users must exactly match nonempty answer users")
    hr, ndcg = [], []
    for uid, answer in truth.items():
        items = predicted[uid]["items"]
        if not isinstance(items, list) or len(items) != 10 or len(set(items)) != 10:
            raise ValueError("Exactly ten distinct candidate IDs required per query")
        if not set(items) <= candidates or answer["target"] not in candidates:
            raise ValueError("Unknown candidate or target")
        rank = items.index(answer["target"]) + 1 if answer["target"] in items else None
        hr.append(float(rank is not None))
        ndcg.append(1 / math.log2(rank + 1) if rank else 0.0)
    result = {
        "HitRate@10": sum(hr) / len(hr),
        "NDCG@10": sum(ndcg) / len(ndcg),
        "queries": len(truth),
        "candidate_count": len(candidates),
        "predictions_sha256": file_hash(predictions),
        "answers_sha256": file_hash(answers),
    }
    result["score"] = 0.31 * result["HitRate@10"] + 0.69 * result["NDCG@10"]
    if output:
        write(output, result)
    return result


def diagnose(public, answers, inference, output, limit=128):
    """Exact target rank vs fixed random candidates, not a causal attribution."""
    import numpy as np

    public, inference = Path(public), Path(inference)
    ids = read(public / "candidate_ids.json")
    positions = {oid: i for i, oid in enumerate(ids)}
    matrix = np.load(inference / "candidate_vectors.npy", mmap_mode="r")
    truth = keyed(answers, "user_id")
    queries = keyed(public / "feedback.jsonl", "user_id")
    vectors = keyed(inference / "query_vectors.jsonl", "user_id")
    if set(truth) != set(vectors) or set(truth) != set(queries):
        raise ValueError("Diagnostic input users differ")
    observations = []
    counts = Counter()
    rng = np.random.default_rng(1729)
    selected = sorted(vectors, key=lambda uid: digest(["diagnostic-1729", uid]))[:limit]
    for uid in selected:
        row = vectors[uid]
        q = np.asarray(row["vector"], dtype=np.float32)
        ti = positions[truth[uid]["target"]]
        target = float(matrix[ti] @ q)
        higher = 0
        for start in range(0, len(matrix), 8192):
            scores = matrix[start : start + 8192] @ q
            index = np.arange(start, start + len(scores))
            higher += int(
                (((scores > target) | ((scores == target) & (index < ti))) & (index != ti)).sum()
            )
        sampled = rng.integers(0, len(matrix) - 1, size=128)
        sampled += sampled >= ti
        margin = target - matrix[sampled] @ q
        history = queries[uid]["history"]
        group = "history_lt50" if len(history) < 50 else "history_ge50"
        counts[group] += 1
        observations.append(
            {
                "user_id": uid,
                "rank": higher + 1,
                "random_win_rate": float((margin > 0).mean()),
                "mean_random_margin": float(margin.mean()),
                "group": group,
            }
        )
    result = {
        "schema": "taac-feedback-diagnostics-v1",
        "retrieval": "exact_full_catalog; ANN not enabled",
        "random_win_rate": float(np.mean([r["random_win_rate"] for r in observations])),
        "mean_target_rank": float(np.mean([r["rank"] for r in observations])),
        "group_counts": dict(counts),
        "total_feedback_queries": len(truth),
        "measured_queries": len(selected),
        "query_selection": "lowest SHA256(diagnostic-1729,user_id), max 128; full candidate pool",
        "observations": observations,
        "limitations": [
            "Unobserved ads are not verified negatives",
            "Descriptive measurements do not prove a bottleneck or cause",
        ],
        "answers_sha256": file_hash(answers),
        "candidate_vectors_sha256": file_hash(inference / "candidate_vectors.npy"),
        "query_vectors_sha256": file_hash(inference / "query_vectors.jsonl"),
    }
    write(output, result)
    return result
