### >>> Self_EvolveRec-BLOCK-START: Make diagnostic definitions executable and preserve future-import placement
"""Per-dataset SASRec with training-only sparse collaborative residuals."""

from __future__ import annotations

import json
import numpy as np
import torch

### <<< Self_EvolveRec-BLOCK-END

### >>> Self_EvolveRec-BLOCK-START: Executable prefix-aware diagnostics with safe sequence-ID masking
import json

import numpy as np
import torch


class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.device = torch.device(device)

    def run_full_diagnosis(self, train_loader):
        model = self.view.model
        if not callable(getattr(model, "hybrid_score", None)):
            raise TypeError("The trained model must expose prefix-aware hybrid_score")
        previous_mode = model.training
        model.eval()
        positives, negatives = [], []
        batches = 0
        max_batches, max_pairs = 8, 8192
        pair_count = 0
        try:
            with torch.no_grad():
                for _, sequence, positive, negative in train_loader:
                    batches += 1
                    sequence = sequence.to(self.device, dtype=torch.long)
                    positive = positive.to(self.device, dtype=torch.long)
                    negative = negative.to(self.device, dtype=torch.long)
                    # Mask malformed IDs before embedding lookup. Preserve absolute
                    # positions and exclude any pair whose prefix was malformed.
                    catalog_size = model.item_emb.num_embeddings
                    invalid_sequence = (sequence < 0) | (sequence >= catalog_size)
                    prefix_invalid = invalid_sequence.long().cumsum(dim=1) > 0
                    safe_sequence = sequence.masked_fill(invalid_sequence, 0)
                    hidden = model.log2feats(safe_sequence)
                    for t in range(sequence.shape[1]):
                        valid = (
                            (sequence[:, t] > 0)
                            & (sequence[:, t] < catalog_size)
                            & (positive[:, t] > 0)
                            & (positive[:, t] < catalog_size)
                            & (negative[:, t] > 0)
                            & (negative[:, t] < catalog_size)
                            & ~prefix_invalid[:, t]
                        )
                        rows = valid.nonzero(as_tuple=True)[0][: max_pairs - pair_count]
                        if rows.numel() == 0:
                            continue
                        prefix = safe_sequence[rows, : t + 1]
                        feature = hidden[rows, t]
                        # Score both items together, sharing the graph aggregation.
                        pair_ids = torch.stack(
                            (positive[rows, t], negative[rows, t]), dim=1
                        )
                        pair_scores = model.hybrid_score(feature, pair_ids, prefix)
                        pos, neg = pair_scores[:, 0], pair_scores[:, 1]
                        positives.append(pos.reshape(-1).double().cpu().numpy())
                        negatives.append(neg.reshape(-1).double().cpu().numpy())
                        pair_count += rows.numel()
                        if pair_count >= max_pairs:
                            break
                    if pair_count >= max_pairs or batches >= max_batches:
                        break
        finally:
            model.train(previous_mode)

        definitions = {
            "valid_pair_count": "Number of evaluated nonpadding positions with valid positive/negative IDs and no invalid sequence ID anywhere in their causal prefix.",
            "mean_positive_score": "Arithmetic mean of combined hybrid positive-item scores over evaluated pairs.",
            "mean_negative_score": "Arithmetic mean of combined hybrid negative-item scores over the same pairs.",
            "mean_margin": "Mean hybrid positive score minus hybrid negative score.",
            "margin_p10": "NumPy linear-interpolated tenth percentile of hybrid pair margins.",
            "fraction_margin_lt_0": "Fraction of evaluated margins strictly below zero.",
            "fraction_margin_lt_0_5": "Fraction of evaluated margins strictly below 0.5 in frozen hybrid score units.",
            "pairwise_accuracy": "P(margin > 0) + 0.5 * P(margin == 0) on evaluated training pairs.",
            "nonfinite_pair_count": "Number of evaluated pairs with at least one nonfinite score; summaries are unavailable if nonzero.",
            "evaluated_batches": "Number of loader batches inspected, bounded by eight.",
            "sampling_limit": "At most eight loader batches and 8192 valid pairs, in loader and position order; not a uniform sample of users.",
            "training_target_inclusion": "Training graphs and neural weights can include the diagnosed target interactions.",
            "user_exclusion": "Additional subtraction of all user graph contributions was not performed; this is not a leave-one-user-out estimate.",
            "exclusion_evaluated_user_count": "Number of identifiable users evaluated with rebuilt exclusion graphs.",
            "category_breakdown": "Unavailable: no item-category metadata is supplied by the matched task.",
            "embedding_similarity": "Neural-item-embedding diagnostic, not computed in this bounded probe.",
            "unobserved_fields": "Ratings, review text, item categories/prices/stores, and simulator conformity/diversity traits are not available.",
        }
        metrics = {
            "valid_pair_count": int(pair_count),
            "evaluated_batches": batches,
            "sampling_limit": definitions["sampling_limit"],
            "training_target_inclusion": True,
            "user_exclusion": {
                "available": False,
                "reason": "Anonymous loader identities were not uniquely recovered; exact unpruned-count graph rebuilding was not executed.",
                "frozen_scales_are_leave_one_user_out": False,
            },
            "exclusion_evaluated_user_count": 0,
            "category_breakdown": None,
            "embedding_similarity": None,
            "unobserved_fields": definitions["unobserved_fields"],
        }
        summary_names = (
            "mean_positive_score",
            "mean_negative_score",
            "mean_margin",
            "margin_p10",
            "fraction_margin_lt_0",
            "fraction_margin_lt_0_5",
            "pairwise_accuracy",
        )
        metrics.update(dict.fromkeys(summary_names, None))
        metrics["nonfinite_pair_count"] = 0
        if pair_count:
            pos = np.concatenate(positives)
            neg = np.concatenate(negatives)
            finite = np.isfinite(pos) & np.isfinite(neg)
            metrics["nonfinite_pair_count"] = int((~finite).sum())
            if finite.all():
                margin = pos - neg
                metrics.update(
                    {
                        "mean_positive_score": float(pos.mean()),
                        "mean_negative_score": float(neg.mean()),
                        "mean_margin": float(margin.mean()),
                        "margin_p10": float(np.percentile(margin, 10)),
                        "fraction_margin_lt_0": float((margin < 0).mean()),
                        "fraction_margin_lt_0_5": float((margin < 0.5).mean()),
                        "pairwise_accuracy": float(
                            (margin > 0).mean() + 0.5 * (margin == 0).mean()
                        ),
                    }
                )
        return {"metrics": metrics, "metric_definitions": definitions}


def diagnosis_interpreter_prompt(raw_diagnosis):
    payload = raw_diagnosis
    if isinstance(payload, str):
        payload = json.loads(payload)
    return (
        "Interpret only the following measured metrics using their supplied definitions. "
        "The margins use prefix-aware combined scores at the original hidden position. "
        "Disclose training-target inclusion, missing user-exclusion estimates, bounded "
        "loader sampling, nonfinite scores, and unavailable metadata. Do not infer "
        "validation HR@10, generalization, user dislikes, simulator traits, or statistical "
        "significance from training-pair margins. A missing metric is unavailable, not zero.\n"
        + json.dumps(payload, indent=2, allow_nan=False)
    )


### <<< Self_EvolveRec-BLOCK-END
### >>> Self_EvolveRec-BLOCK-START: Keep architecture documentation separate from executable diagnostics
# SASRec architecture, optimizer, sampling, checkpoint selection and environment
# hyperparameters remain unchanged. The existing stopping condition is
# epoch - best_epoch >= PATIENCE * 5, checked every five epochs.
### <<< Self_EvolveRec-BLOCK-END

import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# ── Hyper-parameters ──────────────────────────────────────────────────
HIDDEN_UNITS = int(os.environ.get("SASREC_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("SASREC_HEADS", "1"))
DROPOUT_RATE = float(os.environ.get("SASREC_DROPOUT", "0.5"))
LR = float(os.environ.get("SASREC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("SASREC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("SASREC_BATCH", "128"))
NUM_BATCHES = int(os.environ.get("SASREC_BATCHES", "200"))
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host-provided data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four benchmark categories
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Data loading ──────────────────────────────────────────────────────


def _read_split(path: str) -> dict[int, list[int]]:
    d: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                d[int(parts[0])].append(int(parts[1]))
    return dict(d)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return stats[dataset_name]["itemnum"]
    # fallback: scan train file
    itemnum = 0
    path = os.path.join(data_dir, f"{dataset_name}_train.txt")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name: str, data_dir: str):
    user_train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    user_valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    itemnum = _load_itemnum(data_dir, dataset_name)
    return user_train, user_valid, itemnum


# ── Batch sampler ─────────────────────────────────────────────────────


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in ts:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


### >>> Self_EvolveRec-BLOCK-START: Training-only sparse residuals with bounded prefix-aggregation caching
import time
from collections import OrderedDict
from scipy import sparse

# No additional candidate generation is used for residual selection. The host's
# pinned candidate cache is not exposed by this template, so use one fixed grid
# member without claiming a validation search or an improvement.
RESIDUAL_WEIGHTS = (0.1, 0.1)
RESIDUAL_GRID = (
    (0.0, 0.0),
    (0.1, 0.0),
    (0.0, 0.1),
    (0.1, 0.1),
    (0.3, 0.0),
    (0.0, 0.3),
    (0.3, 0.3),
)


def _topk_rows(matrix, k=100):
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    values, indices = [], []
    indptr = [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        ids = matrix.indices[start:end]
        scores = matrix.data[start:end]
        if len(scores) > k:
            cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
            better = np.flatnonzero(scores > cutoff)
            tied = np.flatnonzero(scores == cutoff)
            tied = tied[np.argsort(ids[tied], kind="stable")[: k - len(better)]]
            take = np.concatenate((better, tied))
            ids, scores = ids[take], scores[take]
        order = np.argsort(ids, kind="stable")
        indices.extend(ids[order].tolist())
        values.extend(scores[order].tolist())
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
    )


def build_residual_graphs(histories, itemnum):
    size = itemnum + 1
    incidence_ids, incidence_ptr = [], [0]
    tr_source, tr_target = [], []
    for history in histories.values():
        ids = np.asarray(history, dtype=np.int64)
        if np.any((ids < 1) | (ids > itemnum)):
            raise ValueError("Training item outside the declared catalog")
        incidence_ids.extend(np.unique(ids).tolist())
        incidence_ptr.append(len(incidence_ids))
        if len(ids) > 1:
            keep = ids[:-1] != ids[1:]
            tr_source.extend(ids[:-1][keep].tolist())
            tr_target.extend(ids[1:][keep].tolist())

    x = sparse.csr_matrix(
        (
            np.ones(len(incidence_ids), dtype=np.float32),
            np.asarray(incidence_ids, dtype=np.int32),
            np.asarray(incidence_ptr, dtype=np.int64),
        ),
        shape=(len(histories), size),
    )
    support = np.asarray(x.sum(axis=0)).ravel()
    inv_support = np.zeros(size, dtype=np.float32)
    np.divide(1.0, np.sqrt(support), out=inv_support, where=support > 0)
    xt = x.T.tocsr()

    # Bound each product by an upper bound on its output nonzeros. One source
    # row is unavoidable, but no catalog-square dense array is constructed.
    user_degree = np.diff(x.indptr).astype(np.float64)
    row_bound = np.minimum(np.asarray(xt @ user_degree).ravel(), size)
    parts = []
    start = 0
    max_product_nnz = 2_000_000
    largest_product_nnz = 0
    while start < size:
        end = start
        bound = 0.0
        while end < size and end - start < 128:
            if end > start and bound + row_bound[end] > max_product_nnz:
                break
            bound += row_bound[end]
            end += 1
        counts = (xt[start:end] @ x).tocsr()
        largest_product_nnz = max(largest_product_nnz, int(counts.nnz))
        row_ids = np.repeat(np.arange(start, end), np.diff(counts.indptr))
        counts.data *= inv_support[row_ids] * inv_support[counts.indices]
        counts.data[row_ids == counts.indices] = 0.0
        counts.eliminate_zeros()
        parts.append(_topk_rows(counts))
        start = end
    cosine = sparse.vstack(parts, format="csr")

    transitions = sparse.coo_matrix(
        (np.ones(len(tr_source), dtype=np.float64), (tr_source, tr_target)),
        shape=(size, size),
    ).tocsr()
    transitions.sum_duplicates()
    outgoing = np.asarray(transitions.sum(axis=1)).ravel()
    inverse = np.zeros(size, dtype=np.float64)
    np.divide(1.0, outgoing, out=inverse, where=outgoing > 0)
    transitions = sparse.diags(inverse).dot(transitions).tocsr()
    transitions = _topk_rows(transitions)
    return (
        transitions,
        cosine,
        {
            "transition_nnz": int(transitions.nnz),
            "cosine_nnz": int(cosine.nnz),
            "largest_cosine_product_nnz": largest_product_nnz,
            "product_nnz_upper_bound_budget": max_product_nnz,
            "support_source": "binary deduplicated training histories",
        },
    )


class SparseResidual(nn.Module):
    """Module-owned CPU sparse scorer; independent of candidate chunk size."""

    def __init__(
        self,
        transition,
        cosine,
        scales=None,
        weights=RESIDUAL_WEIGHTS,
        window=10,
        decay=3.0,
    ):
        super().__init__()
        self.transition = transition.tocsr(copy=True)
        self.cosine = cosine.tocsr(copy=True)
        if self.transition.shape != self.cosine.shape:
            raise ValueError("Residual graph shapes must agree")
        if self.transition.shape[0] != self.transition.shape[1]:
            raise ValueError("Residual graphs must be square")
        for graph in (self.transition, self.cosine):
            graph.sum_duplicates()
            graph.eliminate_zeros()
            graph.sort_indices()
            if not np.isfinite(graph.data).all():
                raise ValueError("Residual graph contains nonfinite values")
        self.scales = list(scales if scales is not None else (1.0, 1.0, 1.0))
        if (
            len(self.scales) != 3
            or not np.isfinite(self.scales).all()
            or np.any(np.asarray(self.scales) <= 0)
        ):
            raise ValueError("Expected three finite, strictly positive branch scales")
        self.weights = tuple(weights)
        if len(self.weights) != 2 or not np.isfinite(self.weights).all():
            raise ValueError("Expected two finite residual weights")
        self.window = int(window)
        self.decay = float(decay)
        if self.window <= 0 or not np.isfinite(self.decay) or self.decay <= 0:
            raise ValueError("Recency window and decay must be positive")
        # Cache only training-graph row aggregates, never candidate-dependent
        # normalization or validation labels. Each entry has at most 2*W*K
        # nonzeros for the saved top-K graphs. Graphs remain immutable.
        self._prefix_cache = OrderedDict()
        self._prefix_cache_limit = 128

    def _aggregate_recent(self, recent):
        key = tuple(int(item) for item in recent)
        cached = self._prefix_cache.get(key)
        if cached is not None:
            self._prefix_cache.move_to_end(key)
            return cached
        recency = np.exp(-np.arange(len(recent) - 1, -1, -1) / self.decay)
        recency = (recency / recency.sum()).astype(np.float32)
        coefficients = sparse.csr_matrix(recency.reshape(1, -1))
        aggregates = []
        for graph in (self.transition, self.cosine):
            aggregated = (coefficients @ graph[recent]).tocsr()
            aggregated.sum_duplicates()
            aggregated.eliminate_zeros()
            aggregated.sort_indices()
            aggregates.append(aggregated)
        cached = tuple(aggregates)
        self._prefix_cache[key] = cached
        if len(self._prefix_cache) > self._prefix_cache_limit:
            self._prefix_cache.popitem(last=False)
        return cached

    def branch_scores(self, sequence, items):
        sequences = sequence.detach().cpu().numpy()
        candidates = items.detach().cpu().numpy()
        if sequences.ndim != 2:
            raise ValueError("Expected B x L sequences")
        if candidates.ndim == 1:
            candidates = np.broadcast_to(candidates, (len(sequences), len(candidates)))
        if candidates.ndim != 2 or candidates.shape[0] != len(sequences):
            raise ValueError("Expected shared candidates or B x C candidate IDs")
        results = [np.zeros(candidates.shape, dtype=np.float32) for _ in range(2)]
        for b, sequence_row in enumerate(sequences):
            recent = sequence_row[sequence_row != 0][-self.window :]
            if not len(recent):
                continue
            if np.any((recent < 1) | (recent >= self.transition.shape[0])):
                raise ValueError("Sequence item outside the residual catalog")
            valid = (candidates[b] > 0) & (candidates[b] < self.transition.shape[0])
            if not valid.any():
                continue
            aggregates = self._aggregate_recent(recent)
            for output, aggregated in zip(results, aggregates):
                output[b, valid] = aggregated[:, candidates[b, valid]].toarray().ravel()
        return tuple(torch.as_tensor(result, device=items.device) for result in results)

    def forward(self, neural_scores, sequence, items):
        transition, cosine = self.branch_scores(sequence, items)
        return (
            neural_scores / self.scales[0]
            + self.weights[0] * transition / self.scales[1]
            + self.weights[1] * cosine / self.scales[2]
        )


def fit_residual_scales(
    model, residual, histories, itemnum, device, max_prefixes=2048, negatives=32
):
    rng = np.random.default_rng(42)
    eligible = [
        u
        for u, history in histories.items()
        if len(history) >= 2 and len(set(history)) < itemnum
    ]
    totals = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for begin in range(0, max_prefixes if eligible else 0, 64):
            sequences, candidates = [], []
            for _ in range(min(64, max_prefixes - begin)):
                u = eligible[int(rng.integers(len(eligible)))]
                history = histories[u]
                end = int(rng.integers(1, len(history)))
                prefix = history[max(0, end - model.maxlen) : end]
                sequences.append([0] * (model.maxlen - len(prefix)) + prefix)
                excluded = set(history)
                # Rejection sampling is uniform over unseen items; switch to an
                # explicit complement for nearly exhausted catalogs.
                if len(excluded) * 2 >= itemnum:
                    available = np.setdiff1d(
                        np.arange(1, itemnum + 1), np.fromiter(excluded, dtype=np.int64)
                    )
                    chosen = rng.choice(
                        available, size=negatives, replace=True
                    ).tolist()
                else:
                    chosen = []
                    while len(chosen) < negatives:
                        item = int(rng.integers(1, itemnum + 1))
                        if item not in excluded:
                            chosen.append(item)
                candidates.append(chosen)
            seq = torch.tensor(sequences, dtype=torch.long, device=device)
            ids = torch.tensor(candidates, dtype=torch.long, device=device)
            # Fit the neural branch directly, even if a residual was previously
            # attached to the model by another caller.
            features = model.log2feats(seq)[:, -1, :]
            neural = (model.item_emb(ids) * features.unsqueeze(1)).sum(dim=-1)
            graph_scores = residual.branch_scores(seq, ids)
            for j, scores in enumerate((neural,) + graph_scores):
                values = scores.detach().double().cpu().numpy().ravel()
                totals[j] += values.sum()
                squares[j] += np.dot(values, values)
            count += ids.numel()
    model.train(was_training)
    if not count:
        residual.weights = (0.0, 0.0)
        return [1.0, 1.0, 1.0], {
            "sampled_prefixes": 0,
            "reason": "No eligible training prefixes",
        }
    if not np.isfinite(totals).all() or not np.isfinite(squares).all():
        raise FloatingPointError(
            "Nonfinite branch scores during training-only scale fitting"
        )
    variance = np.maximum(squares / count - (totals / count) ** 2, 0.0)
    scales = np.maximum(np.sqrt(variance), 1e-6)
    # A finite random sample containing no graph hits does not establish that
    # the entire graph branch is identically zero.
    disabled = [residual.transition.nnz == 0, residual.cosine.nnz == 0]
    residual.weights = tuple(
        0.0 if disabled[j] else RESIDUAL_WEIGHTS[j] for j in range(2)
    )
    return scales.tolist(), {
        "sampled_prefixes": count // negatives,
        "negative_scores_per_prefix": negatives,
        "seed": 42,
        "sampling": "uniform eligible user, uniform proper prefix, unseen uniform items with replacement",
        "disabled_zero_branches": disabled,
        "zero_branch_definition": "The complete saved sparse graph has no nonzero entries",
        "sampled_zero_score_branches": [bool(squares[j] == 0.0) for j in (1, 2)],
        "centering": False,
    }


def save_residual_artifacts(
    model, histories, itemnum, device, working_dir, dataset_name
):
    started = time.perf_counter()
    transition, cosine, graph_info = build_residual_graphs(histories, itemnum)
    residual = SparseResidual(transition, cosine)
    scales, scale_info = fit_residual_scales(
        model, residual, histories, itemnum, device
    )
    residual.scales = scales
    sparse.save_npz(
        os.path.join(working_dir, f"{dataset_name}_transition.npz"), transition
    )
    sparse.save_npz(os.path.join(working_dir, f"{dataset_name}_cosine.npz"), cosine)
    manifest = {
        "scales": scales,
        "weights": residual.weights,
        "window": 10,
        "decay": 3.0,
        "shared_requested_weights": RESIDUAL_WEIGHTS,
        "prespecified_grid": RESIDUAL_GRID,
        "selection": "Fixed (0.1, 0.1); no residual validation search: host pinned candidate cache unavailable",
        "baseline_checkpoint_rule": (
            "Evaluate every five epochs using unchanged baseline evaluate_val; "
            "save strictly improving HR@10 from initial best=0; stop when "
            "epoch-best_epoch >= PATIENCE*5; save final only if no checkpoint exists"
        ),
        "epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "training_seed": 42,
        "preprocessing": "Host-provided splits, preprocessing seed 0",
        "graph": graph_info,
        "scale_fit": scale_info,
        "residual_elapsed_seconds": time.perf_counter() - started,
        "timing_definition": "Wall time for graph construction and scale fitting; not GPU-active time",
        "training_target_inclusion": True,
        "user_exclusion": "Unavailable: no identity-indexed exclusion pass was executed",
        "validation_ablations_and_bootstrap": "Unavailable without identical host-pinned candidates",
    }
    path = os.path.join(working_dir, f"{dataset_name}_residual.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    model.residual = residual
    model.eval()
    # Check the actual combined inference path, including repeated sources and
    # arbitrary candidate partitioning. Use only public training histories.
    for history in histories.values():
        if not history:
            continue
        prefix = history[-model.maxlen :]
        seq = torch.tensor(
            [[0] * (model.maxlen - len(prefix)) + prefix],
            dtype=torch.long,
            device=device,
        )
        ids = torch.arange(1, min(itemnum, 257) + 1, device=device).unsqueeze(0)
        with torch.no_grad():
            hidden = model.log2feats(seq)[:, -1]
            whole = model.hybrid_score(hidden, ids, seq)
            chunked = torch.cat(
                [
                    model.hybrid_score(hidden, part, seq)
                    for part in ids.split(37, dim=1)
                ],
                dim=1,
            )
            torch.testing.assert_close(whole, chunked, rtol=1e-5, atol=1e-5)
        break
    print(
        f"  [{dataset_name}] residual scales={scales} weights={residual.weights}",
        flush=True,
    )


### <<< Self_EvolveRec-BLOCK-END

# ── Model ─────────────────────────────────────────────────────────────


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum

        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.item_emb.weight.data.normal_(0.0, 1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)

        pos = torch.arange(M, device=seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_emb(pos)
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = seq == 0
        seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)
        attention_mask = ~torch.tril(
            torch.ones((M, M), dtype=torch.bool, device=seq.device)
        )

        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            queries = self.attention_layernorms[i](seq_emb)
            attn_out, _ = self.attention_layers[i](
                queries, seq_emb, seq_emb, attn_mask=attention_mask
            )
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        return self.last_layernorm(seq_emb)

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        log_feats = self.log2feats(seq)
        if pos is None or neg is None:
            return log_feats
        pos_embs = self.item_emb(pos)
        neg_embs = self.item_emb(neg)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    ### >>> Self_EvolveRec-BLOCK-START: Explicit prefix-aware hybrid scoring and artifact loading
    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        # Legacy aligned-item neural compatibility path.
        return (features * self.item_emb(items)).sum(dim=-1)

    def hybrid_score(
        self, features: torch.Tensor, items: torch.Tensor, sequence: torch.Tensor
    ) -> torch.Tensor:
        item_embs = self.item_emb(items)
        neural = item_embs.matmul(features.unsqueeze(-1)).squeeze(-1)
        if getattr(self, "residual", None) is None:
            return neural
        return self.residual(neural, sequence, items)

    def load_residual(self, working_dir: str, dataset_name: str):
        path = os.path.join(working_dir, f"{dataset_name}_residual.json")
        with open(path) as f:
            config = json.load(f)
        self.residual = SparseResidual(
            sparse.load_npz(
                os.path.join(working_dir, f"{dataset_name}_transition.npz")
            ),
            sparse.load_npz(os.path.join(working_dir, f"{dataset_name}_cosine.npz")),
            scales=config["scales"],
            weights=config["weights"],
            window=config["window"],
            decay=config["decay"],
        )

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(seq)
        return self.hybrid_score(log_feats[:, -1, :], candidates, seq)

    ### <<< Self_EvolveRec-BLOCK-END


# ── Validation ────────────────────────────────────────────────────────


def evaluate_val(
    model: SASRec,
    user_train: dict,
    user_valid: dict,
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    num_users: int = 200,
) -> dict[str, float]:
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        random.seed(42)
        val_users = random.sample(val_users, num_users)

    HR10 = HR20 = NDCG10 = NDCG20 = 0.0
    model.eval()
    with torch.no_grad():
        for u in val_users:
            pos_items = user_valid[u]
            if not pos_items:
                continue
            target = pos_items[0]

            hist = user_train[u][-maxlen:]
            seq_arr = np.zeros(maxlen, dtype=np.int32)
            for i, item in enumerate(reversed(hist)):
                seq_arr[maxlen - 1 - i] = item

            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            neg_pool: set[int] = set()
            interacted = set(user_train[u]) | set(pos_items)
            while len(neg_pool) < 99:
                ni = random.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            scores = model.predict(seq_t, cand_t)[0].cpu().numpy().tolist()
            rank = sorted(range(len(candidates)), key=lambda i: -scores[i]).index(0)

            HR10 += 1.0 if rank < 10 else 0.0
            HR20 += 1.0 if rank < 20 else 0.0
            NDCG10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            NDCG20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0

    n = max(len(val_users), 1)
    model.train()
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
            indices = pos_t != 0
            pos_labels = torch.ones(pos_logits[indices].shape, device=device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
            loss = bce_criterion(pos_logits[indices], pos_labels)
            loss = loss + bce_criterion(neg_logits[indices], neg_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        if epoch % 5 == 0:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}  val_score: {hr10:.4f}"
                f"  HR@20={val_m['HR@20']:.4f}  NDCG@10={val_m['NDCG@10']:.4f}",
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE * 5:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
            },
            f,
        )

    ### >>> Self_EvolveRec-BLOCK-START: Fit residuals only after freezing the baseline checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    save_residual_artifacts(
        model, user_train, itemnum, device, working_dir, dataset_name
    )
    print(
        f"  [{dataset_name}] Neural checkpoint val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10
    ### <<< Self_EvolveRec-BLOCK-END


# ── Write unified predict.py ──────────────────────────────────────────


def write_predict_script(working_dir: str) -> None:
    script = '''"""Per-dataset SASRec predict — routes by dataset_name kwarg.

Supports both legacy signature:
    predict(user_id, history, candidates) -> list[float]
and per-dataset signature (used by the RecHarness harness when available):
    predict(user_id, history, candidates, *, dataset_name) -> list[float]
"""
from __future__ import annotations

import importlib.util
import os
import pickle

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four predictor routes
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

_models: dict = {}
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SASRec


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Load identical frozen sparse residuals for host inference
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.load_residual(_HERE, ds)
        m.eval()
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly for missing dataset routes
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END
    elif _models:
        ds    = next(iter(_models))
        model = _models[ds]
        meta  = _metas[ds]
    else:
        return [0.0] * len(candidates)

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    with torch.no_grad():
        cand_t   = torch.tensor([candidates], dtype=torch.long, device=_device)
        scores   = model.predict(seq_t, cand_t)[0].cpu().tolist()

    return scores
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written → {predict_path}", flush=True)


### >>> Self_EvolveRec-BLOCK-START: Materialize the diagnostic adapter alongside the trained architecture
def write_diagnosis_adapter():
    # The supplied implementation placed the diagnostic definitions inside the
    # opening docstring. They are now executable above; expose that same code
    # through the required diagnosis_tools.py module before training starts.
    adapter = '''"""Diagnostic compatibility adapter for the trained SASRec residual model."""
import importlib.util
import os
import sys

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
_name = "_self_evolverec_diagnostic_training_module"
_module = sys.modules.get(_name)
if _module is None:
    _spec = importlib.util.spec_from_file_location(_name, _path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load diagnostic implementation from {_path}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)

DiagnosisProbe = _module.DiagnosisProbe
diagnosis_interpreter_prompt = _module.diagnosis_interpreter_prompt
'''
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "diagnosis_tools.py"
    )
    with open(path, "w") as handle:
        handle.write(adapter)


### <<< Self_EvolveRec-BLOCK-END

# ── Main ──────────────────────────────────────────────────────────────


def main():
    ### >>> Self_EvolveRec-BLOCK-START: Publish the required diagnostic interface before training
    write_diagnosis_adapter()
    ### <<< Self_EvolveRec-BLOCK-END
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)

    all_hr10: list[float] = []
    for ds in DATASETS:
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        hr10 = train_one_dataset(ds, device, DATA_DIR)
        all_hr10.append(hr10)

    write_predict_script(working_dir)

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    ### >>> Self_EvolveRec-BLOCK-START: Distinguish neural checkpoint validation from host hybrid evaluation
    print(
        f"\nAll datasets done. Average neural checkpoint val HR@10 = {avg:.4f}",
        flush=True,
    )
    print(
        "Hybrid HR@10 must be measured by the host pinned evaluator; no local blend search was performed.",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


