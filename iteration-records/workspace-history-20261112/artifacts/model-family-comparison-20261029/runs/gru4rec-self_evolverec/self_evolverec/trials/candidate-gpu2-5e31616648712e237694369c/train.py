"""Per-dataset GRU4Rec baseline for Amazon Reviews.

Trains one independent GRU4Rec checkpoint per dataset while sharing one default
hyper-parameter profile across datasets. The generated predict.py routes by the
optional dataset_name keyword used by the RecHarness Amazon Reviews harness.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

### >>> Self_EvolveRec-BLOCK-START: Enforce the supplied data boundary and training seed
DATA_DIR = os.environ["GAGC_DATA_DIR"]

SEED = 42
### <<< Self_EvolveRec-BLOCK-END
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _read_split(path: str) -> dict[int, list[int]]:
    data: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]["itemnum"])
    itemnum = 0
    for split in ("train", "valid"):
        path = os.path.join(data_dir, f"{dataset_name}_{split}.txt")
        if not os.path.isfile(path):
            continue
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


### >>> Self_EvolveRec-BLOCK-START: Cached training-only negative sampling and absolute target distances
class TrainingSampler:
    def __init__(self, user_train, itemnum, seed=42):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, items in user_train.items() if len(items) >= 2]
        if not self.users:
            raise RuntimeError("no users with at least two training interactions")
        # Separate streams keep user batches and negative IDs identical between
        # uniform and positional variants, regardless of evaluation RNG usage.
        self.user_rng = random.Random(seed)
        self.negative_rng = np.random.default_rng(seed)
        self.membership = {
            u: set(int(i) for i in user_train[u] if 1 <= i <= itemnum)
            for u in self.users
        }
        self.forbidden = {
            u: np.asarray(sorted(items), dtype=np.int64)
            for u, items in self.membership.items()
        }
        self.dense_complements = {}
        self.omitted_positions = 0

    def _negatives(self, user, count):
        forbidden = self.forbidden[user]
        if len(forbidden) >= self.itemnum:
            return None
        if len(forbidden) > self.itemnum // 2:
            # Avoid arbitrarily slow rejection for nearly exhausted catalogs.
            if user not in self.dense_complements:
                self.dense_complements[user] = np.setdiff1d(
                    np.arange(1, self.itemnum + 1, dtype=np.int64),
                    forbidden,
                    assume_unique=True,
                )
            return self.negative_rng.choice(
                self.dense_complements[user], size=count, replace=True
            )
        values = self.negative_rng.integers(
            1, self.itemnum + 1, size=count, dtype=np.int64
        )
        invalid = np.isin(values, forbidden)
        while invalid.any():
            values[invalid] = self.negative_rng.integers(
                1, self.itemnum + 1, size=int(invalid.sum()), dtype=np.int64
            )
            invalid = np.isin(values, forbidden)
        return values

    def sample(self, batch_size, maxlen, negative_count=1):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, negative_count), dtype=np.int64)
        distance = np.zeros_like(seq)
        for row in range(batch_size):
            user = self.user_rng.choice(self.users)
            history = self.user_train[user]
            length = min(maxlen, len(history) - 1)
            start = maxlen - length
            seq[row, start:] = history[-length - 1 : -1]
            pos[row, start:] = history[-length:]
            # Absolute target indices are T-length, ..., T-1. Their distances
            # from the latest training target are length-1, ..., 0.
            distance[row, start:] = np.arange(length - 1, -1, -1)
            values = self._negatives(user, length * negative_count)
            if values is None:
                self.omitted_positions += length
            else:
                neg[row, start:] = values.reshape(length, negative_count)
        return seq, pos, neg, distance


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    # Preserve the public diagnosis adapter's three B x L arrays. The training
    # workflow below retains its sampler and its cached membership across steps.
    sampler = TrainingSampler(user_train, itemnum, seed=random.getrandbits(32))
    seq, pos, neg, _ = sampler.sample(batch_size, maxlen, negative_count=1)
    return seq, pos, neg[..., 0]


### <<< Self_EvolveRec-BLOCK-END


def _history_to_tensor(
    history: list[int], maxlen: int, device: torch.device
) -> torch.Tensor:
    seq_arr = np.zeros(maxlen, dtype=np.int64)
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    return torch.tensor([seq_arr], dtype=torch.long, device=device)


def evaluate_val(
    model: nn.Module,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Reproduce internal validation candidates across checkpoints and ablations
    rng = random.Random(SEED)
    candidate_rng = random.Random(SEED)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    ### >>> Self_EvolveRec-BLOCK-START: Preserve model mode during validation
    was_training = model.training
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            ### >>> Self_EvolveRec-BLOCK-START: Bound internal validation sampling and retain unique negatives
            available = itemnum - sum(1 for i in interacted if 1 <= i <= itemnum)
            if available < 99:
                raise ValueError(
                    f"validation user {user} has only {available} eligible negatives; "
                    "the required protocol needs 99"
                )
            candidates = [target]
            while len(candidates) < 100:
                neg_item = candidate_rng.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = _history_to_tensor(history, maxlen, device)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += 1.0 if rank < 10 else 0.0
            hr20 += 1.0 if rank < 20 else 0.0
            ndcg10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            ndcg20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0
    n = float(len(val_users))
    ### >>> Self_EvolveRec-BLOCK-START: Restore the caller's model mode
    model.train(was_training)
    ### <<< Self_EvolveRec-BLOCK-END
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


### >>> Self_EvolveRec-BLOCK-START: Exact latest-suffix normalization and deterministic score recomputation


def bpr_sequence_loss(
    model: nn.Module,
    seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    distance: torch.Tensor | None = None,
    objective: str = "baseline",
    positional: bool = False,
) -> torch.Tensor:
    if neg.ndim == 2:
        neg = neg.unsqueeze(-1)
    if neg.ndim != 3 or neg.shape[:2] != seq.shape or neg.shape[-1] < 1:
        raise ValueError("negatives must have shape B x L x K with K >= 1")
    if objective not in ("baseline", "bpr", "bpr_max"):
        raise ValueError(f"unknown objective: {objective}")
    if objective != "bpr_max" and SCORE_REG != 0:
        raise ValueError("pairwise BPR controls require zero score regularization")
    valid = (seq != 0) & (pos != 0) & (neg > 0).all(dim=-1)
    features = model.log2feats(seq)
    active_rows = valid.any(dim=1)
    if not active_rows.any():
        return features.sum() * 0.0

    # Retain the supplied control reduction: mean targets within each eligible
    # row, then mean eligible rows. Eligibility is deterministic per user:
    # either their entire latest suffix has negatives, or none of it does.
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    coefficients = valid.float() / counts / active_rows.sum()
    multiplier = torch.ones_like(coefficients)
    if positional:
        if distance is None or distance.shape != seq.shape:
            raise ValueError("recency weighting requires B x L absolute event ages")
        ages = distance.float()
        if not torch.isfinite(ages[valid]).all() or (ages[valid] < 0).any():
            raise ValueError("valid target ages must be finite and nonnegative")
        expected_ages = torch.arange(
            seq.size(1) - 1, -1, -1, device=seq.device
        ).expand_as(seq)
        if not torch.equal(distance[valid], expected_ages[valid]):
            raise ValueError(
                "exact normalizer requires the unchanged latest-suffix sampler"
            )
        if not torch.equal(valid[active_rows], seq.ne(0)[active_rows]):
            raise ValueError(
                "partial target filtering requires a new exposure derivation"
            )
        # Each user always exposes the same m targets, with ages 0..m-1.
        # Z_u = mean_d[0.5 + 0.5*exp(-d/scale)] has this fixed geometric sum.
        # This is an analytic per-user constant, not a batch weight estimate.
        m = counts.float()
        z = 0.5 + 0.5 * (-torch.expm1(-m / HALF_LIFE)) / (
            -math.expm1(-1.0 / HALF_LIFE) * m
        )
        multiplier = (0.5 + 0.5 * torch.exp(-ages / HALF_LIFE)) / z
    weights = (coefficients * multiplier)[valid]
    features = features[valid]
    positive = pos[valid]
    negative = neg[valid]

    from torch.utils.checkpoint import checkpoint

    def chunk_loss(feats, positive_ids, negative_ids):
        positive_score = model.predict_score(feats, positive_ids).float()
        negative_score = model.predict_score(feats, negative_ids).float()
        preference = torch.nn.functional.logsigmoid(
            positive_score.unsqueeze(-1) - negative_score
        )
        if objective != "bpr_max":
            # Matched-pool pairwise control: mean over K negatives.
            return -preference.mean(dim=-1)
        log_q = torch.log_softmax(negative_score / TEMPERATURE, dim=-1)
        ranking = -torch.logsumexp(log_q + preference, dim=-1)
        if SCORE_REG:
            ranking = ranking + SCORE_REG * (log_q.exp() * negative_score.square()).sum(
                dim=-1
            )
        return ranking

    total = features.new_zeros((), dtype=torch.float32)
    for start in range(0, features.size(0), 1024):
        stop = start + 1024
        losses = checkpoint(
            chunk_loss,
            features[start:stop],
            positive[start:stop],
            negative[start:stop],
            use_reentrant=False,
            # This scoring function contains no stochastic operations;
            # dropout has already executed in log2feats outside checkpoint.
            preserve_rng_state=False,
        )
        total = total + (losses * weights[start:stop]).sum()
    return total


### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Begin with uniform BPR-max and keep experimental controls independent
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))
# One configuration per host trial, one independently trained model/category.
LOSS_OBJECTIVE = os.environ.get("GRU4REC_OBJECTIVE", "bpr_max")
WEIGHTING = os.environ.get("GRU4REC_WEIGHTING", "uniform")
NEGATIVE_COUNT = int(os.environ.get("GRU4REC_NEGATIVES", "16"))
# Retain the environment spelling; its value now specifies the e-fold event
# scale in the prespecified bounded weight 0.5 + 0.5*exp(-distance/scale).
HALF_LIFE = float(os.environ.get("GRU4REC_HALF_LIFE", "10"))
TEMPERATURE = float(os.environ.get("GRU4REC_TEMPERATURE", "1"))
SCORE_REG = float(os.environ.get("GRU4REC_SCORE_REG", "0"))
if LOSS_OBJECTIVE not in ("baseline", "bpr", "bpr_max"):
    raise ValueError("GRU4REC_OBJECTIVE must be baseline, bpr or bpr_max")
if WEIGHTING not in ("uniform", "absolute", "window", "positional"):
    raise ValueError("GRU4REC_WEIGHTING must be uniform, absolute or window")
if WEIGHTING == "positional":
    WEIGHTING = "window"
POSITIONAL_WEIGHTING = WEIGHTING != "uniform"
if NEGATIVE_COUNT < 1 or EVAL_EVERY < 1 or PATIENCE < 1:
    raise ValueError("negative count, evaluation cadence and patience must be positive")
if not all(math.isfinite(x) for x in (HALF_LIFE, TEMPERATURE, SCORE_REG)):
    raise ValueError("loss hyperparameters must be finite")
if HALF_LIFE <= 0 or TEMPERATURE <= 0 or SCORE_REG < 0:
    raise ValueError("invalid loss hyperparameters")
if LOSS_OBJECTIVE != "bpr_max" and SCORE_REG != 0:
    raise ValueError("pairwise BPR controls require zero score regularization")
### <<< Self_EvolveRec-BLOCK-END


class GRU4RecModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Padding-invariant GRU features with trimmed embedding allocation
    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        batch_size, width = sequence.shape
        lengths = sequence.ne(0).sum(dim=1)
        original_rows = lengths.gt(0).nonzero(as_tuple=True)[0]
        if original_rows.numel() == 0:
            return self.item_emb(sequence) * 0.0

        tokens = sequence[original_rows]
        valid = tokens.ne(0)
        rows, columns = valid.nonzero(as_tuple=True)
        compact_columns = valid.long().cumsum(dim=1)[rows, columns] - 1
        # Packing needs CPU lengths anyway. Embed only the longest real
        # history in this batch, avoiding unused leading-padding allocation.
        cpu_lengths = lengths[original_rows].detach().cpu()
        compact_width = int(cpu_lengths.max().item())
        compact = tokens.new_zeros(tokens.size(0), compact_width)
        compact[rows, compact_columns] = tokens[rows, columns]
        packed = nn.utils.rnn.pack_padded_sequence(
            self.dropout(self.item_emb(compact)),
            cpu_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_features, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            packed_features, batch_first=True, total_length=width
        )
        # Unpacking restores batch order automatically. Scatter real states
        # to their original indices; padding never executes a recurrent step.
        features = unpacked.new_zeros(batch_size, width, self.hidden_units)
        features[original_rows[rows], columns] = unpacked[rows, compact_columns]
        return features

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        raise ValueError(
            f"incompatible feature and item shapes: {features.shape}, {items.shape}"
        )

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Use the same padding-safe encoder for training and prediction
    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.log2feats(seq)
        if seq.size(1) == 0:
            return out.sum(dim=1)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1)
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp_min(0)]
        return features.masked_fill(last.lt(0).unsqueeze(-1), 0.0)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    ### <<< Self_EvolveRec-BLOCK-END

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.encode(seq)[0]
        cand_emb = self.item_emb(candidates)
        return torch.matmul(cand_emb, user_emb)


### >>> Self_EvolveRec-BLOCK-START: Verify aligned states and scores against individual unpadded GRU executions
@torch.no_grad()
def verify_gru_encoding(model: nn.Module, device: torch.device) -> None:
    was_training = model.training
    model.eval()
    try:
        sequence = torch.tensor(
            [
                [0, 0, 1, 2, 3],
                [0, 0, 0, 0, 2],
                [1, 2, 3, 4, 5],
                [0, 0, 3, 2, 1],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 2, 3],
            ],
            device=device,
            dtype=torch.long,
        )
        sequence = torch.where(
            sequence > 0, (sequence - 1).remainder(model.itemnum) + 1, sequence
        )
        features = model.log2feats(sequence)
        final = model.encode(sequence)
        candidates = torch.arange(1, min(model.itemnum, 7) + 1, device=device)

        def check(actual, expected):
            if not torch.isfinite(actual).all():
                raise FloatingPointError("nonfinite GRU verification output")
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.0)

        check(features[sequence == 0], torch.zeros_like(features[sequence == 0]))
        for row in range(sequence.size(0)):
            valid = sequence[row].ne(0)
            if not valid.any():
                check(final[row], torch.zeros_like(final[row]))
                continue
            tokens = sequence[row, valid].unsqueeze(0)
            reference, _ = model.gru(model.item_emb(tokens))
            check(features[row, valid], reference[0])
            check(final[row], reference[0, -1])
            check(
                model.score_candidates(sequence[row : row + 1], candidates),
                model.item_emb(candidates) @ reference[0, -1],
            )
        permutation = torch.tensor([4, 2, 0, 5, 1, 3], device=device)
        check(model.log2feats(sequence[permutation]), features[permutation])
        padded = torch.nn.functional.pad(sequence, (3, 0))
        check(model.log2feats(padded)[:, 3:], features)
        check(model.encode(padded), final)
        empty = torch.zeros_like(sequence)
        check(model.log2feats(empty), torch.zeros_like(features))
        check(model.encode(empty), torch.zeros_like(final))
    finally:
        model.train(was_training)


### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Seed each category and disclose independent experimental controls
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    sampler = TrainingSampler(user_train, itemnum, seed=SEED)
    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    verify_gru_encoding(model, device)
    print(f"  [{dataset_name}] GRU padding-invariance verification passed", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "seed": SEED,
                "objective": LOSS_OBJECTIVE,
                "weighting": WEIGHTING,
                "recency_e_fold_scale_events": HALF_LIFE,
                "recency_floor": 0.5,
                "score_activation": "identity_raw_dot_product",
                "encoding": "padding_invariant_gru_v1",
                "objective_control": "mean_pairwise_BPR_over_the_same_K_negatives",
                "recency_normalizer": "analytic_fixed_latest_suffix_geometric_mean",
                "temperature": TEMPERATURE,
                "negative_score_regularization": SCORE_REG,
                "negative_count": NEGATIVE_COUNT,
                "negative_sampling": "uniform_with_replacement_excluding_full_training_history",
                "normalization": "within_sampled_sequence_then_mean_nonempty_sequences",
                "sampling": "one_latest_suffix_per_uniformly_sampled_user",
                "age_control_note": "absolute and window ages coincide for latest suffixes",
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT_RATE,
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "batches_per_epoch": NUM_BATCHES,
                "epochs": NUM_EPOCHS,
                "eval_every": EVAL_EVERY,
                "patience": PATIENCE,
            }
        ),
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Invoke matched negative pools and report normalized recency exposure
            seq, pos, neg, distance = sampler.sample(
                BATCH_SIZE, MAXLEN, negative_count=NEGATIVE_COUNT
            )
            if WEIGHTING == "window":
                targets = (seq != 0) & (pos != 0)
                distance = np.cumsum(targets[:, ::-1], axis=1)[:, ::-1] - targets
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            distance_t = torch.tensor(distance, dtype=torch.long, device=device)
            loss = bpr_sequence_loss(
                model,
                seq_t,
                pos_t,
                neg_t,
                distance=distance_t,
                objective=LOSS_OBJECTIVE,
                positional=POSITIONAL_WEIGHTING,
            )
            if epoch == 1 and _ == 0:
                eligible = (seq != 0) & (pos != 0) & (neg > 0).all(axis=-1)
                active = eligible.any(axis=1)
                counts = np.maximum(eligible.sum(axis=1, keepdims=True), 1)
                multiplier = np.ones_like(distance, dtype=float)
                if POSITIONAL_WEIGHTING:
                    z = 0.5 + 0.5 * (-np.expm1(-counts / HALF_LIFE)) / (
                        -math.expm1(-1.0 / HALF_LIFE) * counts
                    )
                    multiplier = (0.5 + 0.5 * np.exp(-distance / HALF_LIFE)) / z
                baseline_weights = eligible.astype(float) / counts
                actual_weights = baseline_weights * multiplier
                pools = neg[eligible]
                unique_counts = np.asarray(
                    [np.unique(pool).size for pool in pools], dtype=float
                )
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "first_batch_baseline_row_mass_range": [
                                float(baseline_weights.sum(axis=1).min()),
                                float(baseline_weights.sum(axis=1).max()),
                            ],
                            "first_batch_weighted_row_mass_range": [
                                float(actual_weights.sum(axis=1).min()),
                                float(actual_weights.sum(axis=1).max()),
                            ],
                            "first_batch_coefficient_ess": float(
                                actual_weights.sum() ** 2
                                / max(float(np.square(actual_weights).sum()), 1e-12)
                            ),
                            "ess_definition": "squared_sum_over_sum_squares_of_position_coefficients",
                            "first_batch_mean_unique_negatives": (
                                float(unique_counts.mean())
                                if unique_counts.size
                                else None
                            ),
                            "first_batch_duplicate_fraction": (
                                float(1.0 - unique_counts.mean() / NEGATIVE_COUNT)
                                if unique_counts.size
                                else None
                            ),
                            "first_batch_normalized_weight_mass_by_target_distance": {
                                label: float(
                                    actual_weights[group].sum()
                                    / max(int(active.sum()), 1)
                                )
                                for label, group in (
                                    ("0_to_9", distance < 10),
                                    ("10_to_49", (distance >= 10) & (distance < 50)),
                                    ("50_plus", distance >= 50),
                                )
                            },
                        }
                    ),
                    flush=True,
                )
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Skip ineligible updates and fail explicitly on nonfinite training values
            optimizer.zero_grad(set_to_none=True)
            if not ((seq != 0) & (pos != 0) & (neg > 0).all(axis=-1)).any():
                # Adam momentum and weight decay must not update an empty batch.
                continue
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"nonfinite loss for {dataset_name} at epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.item())
            ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Report explicitly omitted targets with empty negative catalogs
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            print(
                f"  [{dataset_name}] cumulative_empty_catalog_positions="
                f"{sampler.omitted_positions}",
                flush=True,
            )
        ### <<< Self_EvolveRec-BLOCK-END
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
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
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    ### >>> Self_EvolveRec-BLOCK-START: Verify the selected trained checkpoint before exporting its predictor metadata
    if best_epoch == 0:
        raise RuntimeError(
            f"No validation-selected checkpoint was produced for {dataset_name}"
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    verify_gru_encoding(model, device)
    print(
        f"  [{dataset_name}] selected epoch {best_epoch}: "
        "trained-checkpoint padding and scoring checks passed",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Pin checkpoint encoder semantics for the generated predictor
                "encoding": "padding_invariant_gru_v1",
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""Per-dataset GRU4Rec predict function."""
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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]
_models = {}
_metas = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"gru4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Require one complete checkpoint per routed category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing GRU4Rec artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        ### >>> Self_EvolveRec-BLOCK-START: Load the matching padding-safe GRU architecture from the predictor writer
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("encoding") != "padding_invariant_gru_v1":
            raise ValueError(f"Incompatible GRU encoding metadata for {ds}")
        # ModelClass imports the exact encoder and readout used for training.
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            dropout_rate=meta["dropout_rate"],
        ).to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ### >>> Self_EvolveRec-BLOCK-START: Reject ambiguous category routing instead of scoring with another category
    _ensure_loaded()
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple categories are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"No trained GRU4Rec model for dataset {dataset_name!r}")
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    if not candidates:
        return []
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = int(meta.get("maxlen", 128))
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written -> {predict_path}", flush=True)


def main() -> None:
    ### >>> Self_EvolveRec-BLOCK-START: Require all four benchmark categories exactly once
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError(
            "GAGC_DATASETS must contain all four benchmark categories exactly once"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset GRU4Rec training on {device}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


