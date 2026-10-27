from __future__ import annotations

import os

# Avoid nested BLAS parallelism inside independent regression jobs.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import inspect
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    import scipy.sparse as sp
    from scipy.linalg import eigh
except Exception:
    import traceback

    (Path(__file__).resolve().parent / "training_error.log").write_text(
        traceback.format_exc(), encoding="utf-8"
    )
    raise
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    # BLAS thread counts are already limited before importing NumPy/SciPy.
    def threadpool_limits(limits=1):
        return nullcontext()

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
ROOT = Path(__file__).resolve().parent
WORKING = ROOT / "working"
NEIGHBORS = int(os.environ.get("LOCAL_RIDGE_NEIGHBORS", "128"))
BLOCK_SIZE = 64
WORKERS = min(16, max(2, (os.cpu_count() or 4) // 2))
RIDGES = (10.0, 100.0, 1000.0)

random.seed(42)
np.random.seed(42)


def read_split(path):
    result = defaultdict(list)
    with open(path, "r") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) == 2:
                user, item = map(int, fields)
                result[user].append(item)
    return dict(result)


def load_dataset(name):
    train = read_split(Path(DATA_DIR) / f"{name}_train.txt")
    valid = read_split(Path(DATA_DIR) / f"{name}_valid.txt")
    stats_path = Path(DATA_DIR) / "dataset_stats.json"
    if stats_path.exists():
        with open(stats_path) as stream:
            stats = json.load(stream)
        itemnum = int(stats.get(name, {}).get("itemnum", 0))
    else:
        itemnum = 0
    # Only the permitted train and validation splits determine any fallback bounds.
    for split in (train, valid):
        itemnum = max(itemnum, max((max(v) for v in split.values() if v), default=0))
    return train, valid, itemnum


class SparseLinear:
    """A fitted sparse linear model: binary user history times learned coefficients."""

    def __init__(self, weights):
        self.weights = weights.tocsr().astype(np.float32)
        self.itemnum = self.weights.shape[0] - 1

    def forward(self, history, candidates):
        candidates = np.asarray(candidates, dtype=np.int64)
        if np.any(candidates < 1) or np.any(candidates > self.itemnum):
            raise ValueError("Candidate outside the trained catalog")
        history = np.unique(np.asarray(history, dtype=np.int64))
        history = history[(history >= 1) & (history <= self.itemnum)]
        # This is exactly the regression forward pass used in validation.
        encoded = sp.csr_matrix(
            (
                np.ones(len(history), dtype=np.float32),
                (np.zeros(len(history), dtype=np.int32), history),
            ),
            shape=(1, self.itemnum + 1),
        )
        prediction = encoded @ self.weights
        return prediction[:, candidates].toarray().ravel()

    def predict(self, history, candidates):
        return self.forward(history, candidates)


def training_matrix(train, itemnum):
    rows, cols = [], []
    for row, history in enumerate(train.values()):
        items = sorted(set(history))
        rows.extend([row] * len(items))
        cols.extend(items)
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(train), itemnum + 1),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix


def make_validation(train, valid, itemnum):
    # Match the baseline's user ordering, sampling, and unique-negative protocol,
    # but isolate the evaluation RNG from all model fitting.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for user in users:
        if not valid[user]:
            continue
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Catalog has fewer than 99 eligible validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        examples.append((train[user], [target] + list(negatives)))
    if not examples:
        raise ValueError("No eligible validation examples")
    return examples


def evaluate(model, examples):
    hits = 0
    ndcg = 0.0
    for history, candidates in examples:
        scores = model.predict(history, candidates)
        # Stable ordering follows the baseline's candidate-order tie handling.
        rank = int(np.flatnonzero(np.argsort(-scores, kind="stable") == 0)[0])
        if rank < 10:
            hits += 1
            ndcg += 1.0 / math.log2(rank + 2)
    return hits / len(examples), ndcg / len(examples)


def fit_local_regressions(matrix, name):
    """Fit y_j = X_neighbors beta_j with no self feature.

    Neighborhoods and normal equations use training interactions exclusively.
    Local covariance matrices avoid a dense itemnum x itemnum inverse.
    """
    itemnum = matrix.shape[1] - 1
    csc = matrix.tocsc()
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    source_rows = []
    target_cols = []
    coefficient_chunks = [[] for _ in RIDGES]

    def solve_one(job):
        target, neighbors, rhs = job
        local = csc[:, neighbors]
        covariance = (local.T @ local).toarray()
        # One symmetric decomposition provides all three ridge solutions.
        eigenvalues, vectors = eigh(covariance, check_finite=False)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projection = vectors.T @ rhs
        coefficients = [
            (vectors @ (projection / (eigenvalues + alpha))).astype(np.float32)
            for alpha in RIDGES
        ]
        return target, neighbors, coefficients

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(1, itemnum + 1, BLOCK_SIZE):
            stop = min(itemnum + 1, start + BLOCK_SIZE)
            # Only this small target block's sparse co-occurrences exist in memory.
            gram_block = (csc[:, start:stop].T @ matrix).tocsr()
            jobs = []
            for offset, target in enumerate(range(start, stop)):
                left, right = gram_block.indptr[offset : offset + 2]
                neighbors = gram_block.indices[left:right]
                overlap = gram_block.data[left:right]
                keep = (neighbors != target) & (neighbors != 0) & (overlap > 0)
                neighbors = neighbors[keep]
                overlap = overlap[keep]
                if len(neighbors) == 0:
                    continue
                # Cosine support selection reduces domination by ubiquitous items.
                similarity = overlap / np.sqrt(
                    frequencies[target] * frequencies[neighbors]
                )
                if len(neighbors) > NEIGHBORS:
                    chosen = np.argpartition(similarity, -NEIGHBORS)[-NEIGHBORS:]
                    neighbors, overlap = neighbors[chosen], overlap[chosen]
                order = np.argsort(neighbors)
                jobs.append((target, neighbors[order].copy(), overlap[order].copy()))

            for target, neighbors, coefficients in pool.map(solve_one, jobs):
                source_rows.append(neighbors.astype(np.int32))
                target_cols.append(np.full(len(neighbors), target, dtype=np.int32))
                for chunks, fitted in zip(coefficient_chunks, coefficients):
                    chunks.append(fitted)

    if not source_rows:
        raise ValueError(f"{name}: no item associations in training data")
    rows = np.concatenate(source_rows)
    cols = np.concatenate(target_cols)
    shape = (itemnum + 1, itemnum + 1)
    for alpha, chunks in zip(RIDGES, coefficient_chunks):
        weights = sp.csr_matrix((np.concatenate(chunks), (rows, cols)), shape=shape)
        weights.eliminate_zeros()
        yield alpha, SparseLinear(weights)


def train_one_dataset(name):
    train, valid, itemnum = load_dataset(name)
    matrix = training_matrix(train, itemnum)
    examples = make_validation(train, valid, itemnum)
    best_key = (-1.0, -1.0)
    best_alpha = None
    checkpoint = WORKING / f"{name}_weights.npz"
    for index, (alpha, model) in enumerate(fit_local_regressions(matrix, name), 1):
        hr, ndcg = evaluate(model, examples)
        print(
            f"{name} fit={index}/{len(RIDGES)} ridge={alpha:g} "
            f"validation_users={len(examples)} HR@10={hr:.6f} NDCG@10={ndcg:.6f}",
            flush=True,
        )
        if (hr, ndcg) > best_key:
            best_key = (hr, ndcg)
            best_alpha = alpha
            sp.save_npz(checkpoint, model.weights)
    with open(WORKING / f"{name}_meta.json", "w") as stream:
        json.dump(
            {
                "itemnum": itemnum,
                "ridge": best_alpha,
                "neighbors": NEIGHBORS,
                "validation_hr10": best_key[0],
                "validation_users": len(examples),
                "training_seed": 42,
                "preprocessing": "provided splits, unchanged",
            },
            stream,
        )
    return best_key[0]


def write_predict_script():
    # Embed the same inference class rather than depending on a differently
    # reconstructed architecture or on the training script being importable.
    header = """from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp

"""
    loader = """
_HERE = Path(__file__).resolve().parent
_DATASETS = [
    'Movies_and_TV', 'Industrial_and_Scientific',
    'Electronics', 'CDs_and_Vinyl',
]
_MODELS = {}


def _get_model(dataset_name):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if dataset_name not in _MODELS:
        weights = sp.load_npz(_HERE / f'{dataset_name}_weights.npz')
        _MODELS[dataset_name] = SparseLinear(weights)
    return _MODELS[dataset_name]


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    return _get_model(dataset_name).predict(history, candidates).tolist()
"""
    with open(WORKING / "predict.py", "w") as stream:
        stream.write(header)
        stream.write(inspect.getsource(SparseLinear))
        stream.write(loader)


def main():
    WORKING.mkdir(parents=True, exist_ok=True)
    scores = [train_one_dataset(name) for name in DATASETS]
    write_predict_script()
    score = float(np.mean(scores))
    with open(WORKING / "validation.json", "w") as stream:
        json.dump(
            {
                "metric": "macro HR@10",
                "evaluation_seed": 42,
                "negative_candidates": 99,
                "per_dataset": dict(zip(DATASETS, scores)),
                "score": score,
            },
            stream,
            indent=2,
        )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        # Preserve the underlying exception even when the worker reports only
        # CalledProcessError. Include failures occurring during module imports.
        error_path = Path(__file__).resolve().parent / "training_error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
