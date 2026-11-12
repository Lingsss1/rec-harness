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
    ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required interaction split is missing
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required interaction split is missing: {path}")
    ### <<< Self_EvolveRec-BLOCK-END
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
    model.eval()
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
    model.train()
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


### >>> Self_EvolveRec-BLOCK-START: Score every training objective through the joint GRU readout with complete negative pools


def bpr_sequence_loss(
    model: nn.Module,
    seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    distance: torch.Tensor | None = None,
    objective: str = "baseline",
    positional: bool = False,
) -> torch.Tensor:
    if seq.ndim != 2 or pos.shape != seq.shape:
        raise ValueError("sequence and positives must be aligned B x L")
    if neg.ndim == 2:
        neg = neg.unsqueeze(-1)
    if neg.ndim != 3 or neg.shape[:2] != seq.shape or neg.size(-1) < 1:
        raise ValueError("negatives must have shape B x L x K, K >= 1")
    if objective not in ("baseline", "bpr_max"):
        raise ValueError(f"unknown objective: {objective}")
    valid = seq.ne(0) & pos.ne(0) & neg.gt(0).all(dim=-1)
    if positional:
        if distance is None or distance.shape != seq.shape:
            raise ValueError("aligned training-target distances are required")
        ages = distance[valid].float()
        if not torch.isfinite(ages).all() or ages.lt(0).any():
            raise ValueError("target distances must be finite and nonnegative")

    features = model.log2feats(seq)
    if not valid.any():
        return features.sum() * 0.0
    features = features[valid]
    positive = pos[valid]
    negative = neg[valid]
    weights = (
        0.5 + 0.5 * torch.exp(-ages / 10.0)
        if positional
        else features.new_ones(features.size(0))
    )

    from torch.utils.checkpoint import checkpoint

    def chunk_loss(feats, positive_ids, negative_ids):
        positive_score = model.predict_score(feats, positive_ids)
        negative_score = model.predict_score(feats, negative_ids)
        if positive_score.dtype in (torch.float16, torch.bfloat16):
            positive_score = positive_score.float()
            negative_score = negative_score.float()
        preference = torch.nn.functional.logsigmoid(
            positive_score.unsqueeze(-1) - negative_score
        )
        if objective == "baseline":
            # Mean pairwise BPR across all K columns; K=1 retains the baseline.
            return -torch.logaddexp(preference, preference.new_tensor(1e-8).log()).mean(
                dim=-1
            )
        log_weight = torch.log_softmax(negative_score, dim=-1)
        log_probability = torch.logsumexp(log_weight + preference, dim=-1)
        if BPR_MAX_EPSILON:
            log_probability = torch.logaddexp(
                log_probability,
                log_probability.new_tensor(BPR_MAX_EPSILON).log(),
            )
        ranking = -log_probability
        return ranking + SCORE_REG * (log_weight.exp() * negative_score.square()).sum(
            dim=-1
        )

    # Chunk positions, keeping all candidate columns in each global softmax.
    # Recompute embedding lookup during backward to bound saved activations.
    total = features.new_zeros(())
    for start in range(0, features.size(0), 1024):
        stop = start + 1024
        losses = checkpoint(
            chunk_loss,
            features[start:stop],
            positive[start:stop],
            negative[start:stop],
            use_reentrant=False,
        )
        total = total + (losses * weights[start:stop]).sum()
    return total / weights.sum().clamp_min(1e-8)


### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Declare the unweighted BPR-max configuration for the recent-window experiment
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))
LOSS_OBJECTIVE = os.environ.get("GRU4REC_OBJECTIVE", "bpr_max")
WEIGHTING = os.environ.get("GRU4REC_WEIGHTING", "uniform")
# Hold the pool constant when explicitly requesting a separate BPR control.
NEGATIVE_COUNT = int(os.environ.get("GRU4REC_NEGATIVES", "32"))
# Preserve the supplied BPR-max score penalty and probability convention.
SCORE_REG = 0.001
BPR_MAX_EPSILON = 0.0
if LOSS_OBJECTIVE not in ("baseline", "bpr_max"):
    raise ValueError("GRU4REC_OBJECTIVE must be baseline or bpr_max")
if WEIGHTING not in ("uniform", "positional"):
    raise ValueError("GRU4REC_WEIGHTING must be uniform or positional")
if (
    min(
        MAXLEN,
        HIDDEN_UNITS,
        NUM_LAYERS,
        BATCH_SIZE,
        NEGATIVE_COUNT,
        EVAL_EVERY,
        PATIENCE,
    )
    < 1
):
    raise ValueError("dimensions, counts, cadence and patience must be positive")
if not all(math.isfinite(x) for x in (LR, WEIGHT_DECAY, DROPOUT_RATE)):
    raise ValueError("optimizer and dropout parameters must be finite")
if LR <= 0 or WEIGHT_DECAY < 0 or not 0 <= DROPOUT_RATE < 1:
    raise ValueError("invalid optimizer or dropout configuration")
POSITIONAL_WEIGHTING = WEIGHTING == "positional"
print(
    json.dumps(
        {
            "objective": LOSS_OBJECTIVE,
            "weighting": WEIGHTING,
            "negative_count": NEGATIVE_COUNT,
            "bpr_max_score_regularization": SCORE_REG,
            "bpr_max_probability_epsilon": BPR_MAX_EPSILON,
            "bpr_probability_epsilon": 1e-8,
            "reduction": "global_weighted_valid_position_mean",
            "recent_window": 4,
            "architecture": "packed_gru_recent_mean_v1",
            "checkpoint_rule": "highest_category_validation_hr10_earliest_tie",
        }
    ),
    flush=True,
)
### <<< Self_EvolveRec-BLOCK-END


class GRU4RecModel(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Initialize a jointly trained four-input mean readout without shifting shared initialization
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        recent_window: int = 4,
    ):
        super().__init__()
        if not isinstance(recent_window, int) or recent_window < 1:
            raise ValueError("recent_window must be a positive integer")
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.recent_window = recent_window
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
        # These modules are constructed on CPU before model.to(device).
        with torch.random.fork_rng(devices=[]):
            self.recent_project = nn.Linear(hidden_units, hidden_units, bias=False)
            self.recent_gate = nn.Linear(2 * hidden_units, 1)
            nn.init.xavier_uniform_(self.recent_project.weight)
            nn.init.zeros_(self.recent_gate.weight)
            nn.init.constant_(self.recent_gate.bias, -4.0)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Padding-invariant packed GRU with causal recent-real-input mean and shared scoring
    def recent_mean(self, embedded: torch.Tensor, lengths: torch.Tensor):
        # Inputs are compact chronological rows, with padding on the RIGHT.
        # Direct shifted sums avoid subtracting long, nearly equal prefixes.
        width = embedded.size(1)
        if width == 0:
            return embedded
        positions = torch.arange(width, device=embedded.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        accumulation_dtype = (
            torch.float32
            if embedded.dtype in (torch.float16, torch.bfloat16)
            else embedded.dtype
        )
        x = embedded.to(accumulation_dtype).masked_fill(~valid.unsqueeze(-1), 0.0)
        pooled = x
        for offset in range(1, min(self.recent_window, width)):
            shifted = torch.cat(
                (
                    x.new_zeros(x.size(0), offset, x.size(2)),
                    x[:, : width - offset],
                ),
                dim=1,
            )
            pooled = pooled + shifted
        denominator = (positions + 1).clamp(max=self.recent_window).to(x.dtype)
        pooled = pooled / denominator[None, :, None]
        return pooled.masked_fill(~valid.unsqueeze(-1), 0.0).to(embedded.dtype)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        batch_size, width = sequence.shape
        lengths = sequence.ne(0).sum(dim=1)
        nonempty = lengths.gt(0).nonzero(as_tuple=True)[0]
        if nonempty.numel() == 0:
            return self.item_emb(sequence) * 0.0

        # Compact real inputs before recurrence so GRU biases never process
        # leading padding. This also handles internal zero gaps consistently.
        tokens = sequence[nonempty]
        valid = tokens.ne(0)
        rows, columns = valid.nonzero(as_tuple=True)
        compact_columns = valid.long().cumsum(dim=1)[rows, columns] - 1
        compact = torch.zeros_like(tokens)
        compact[rows, compact_columns] = tokens[rows, columns]
        embedded = self.dropout(self.item_emb(compact))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths[nonempty].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_features, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            packed_features, batch_first=True, total_length=width
        )
        # Pool the exact input/dropout realization supplied to the GRU.
        pooled = self.recent_mean(embedded, lengths[nonempty])
        base = unpacked[rows, compact_columns]
        recent = pooled[rows, compact_columns]
        gate = torch.sigmoid(self.recent_gate(torch.cat((base, recent), dim=-1)))
        mixed = (1.0 - gate) * base + gate * self.recent_project(recent)
        features = mixed.new_zeros(batch_size, width, self.hidden_units)
        features[nonempty[rows], columns] = mixed
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
            return torch.matmul(
                features.unsqueeze(-2), embeddings.transpose(-1, -2)
            ).squeeze(-2)
        raise ValueError(
            f"incompatible feature and item shapes: {features.shape}, {items.shape}"
        )

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Use the same joint features for training, probes and final-nonpadding inference
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


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Reset each independent category and record resolved configuration
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "seed": SEED,
                "objective": LOSS_OBJECTIVE,
                "weighting": WEIGHTING,
                "negative_count": NEGATIVE_COUNT,
                "negative_sampling": "uniform_with_replacement_excluding_full_training_history",
                "objective_comparison_includes_negative_pool_change": True,
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
            ### >>> Self_EvolveRec-BLOCK-START: Release previous gradients before allocating training activations
            optimizer.zero_grad(set_to_none=True)
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Invoke pooled negatives and absolute recency weighting
            seq, pos, neg, distance = sampler.sample(
                BATCH_SIZE, MAXLEN, negative_count=NEGATIVE_COUNT
            )
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
                actual_weights = (
                    0.5 + 0.5 * np.exp(-distance / 10.0)
                    if POSITIONAL_WEIGHTING
                    else np.ones_like(distance, dtype=float)
                )
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "first_batch_weight_by_target_distance": {
                                label: (
                                    float(actual_weights[eligible & group].mean())
                                    if (eligible & group).any()
                                    else None
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
            ### >>> Self_EvolveRec-BLOCK-START: Skip empty updates and validate loss and gradients before optimization
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

    ### >>> Self_EvolveRec-BLOCK-START: Require a validation-selected checkpoint and restore it consistently
    if best_epoch == 0 or not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"No validation-selected checkpoint was produced for {dataset_name}"
        )
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True), strict=True
    )
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                ### >>> Self_EvolveRec-BLOCK-START: Persist the complete joint readout configuration
                "dropout_rate": DROPOUT_RATE,
                "recent_window": model.recent_window,
                "architecture": "packed_gru_recent_mean_v1",
                "pooling": "compact_causal_shifted_mean",
                "empty_history_policy": "zero_features",
                "score_activation": "linear",
                "objective": LOSS_OBJECTIVE,
                "negative_count": NEGATIVE_COUNT,
                "weighting": WEIGHTING,
                "score_regularization": (
                    SCORE_REG if LOSS_OBJECTIVE == "bpr_max" else 0.0
                ),
                "probability_epsilon": (
                    BPR_MAX_EPSILON if LOSS_OBJECTIVE == "bpr_max" else 1e-8
                ),
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


### >>> Self_EvolveRec-BLOCK-START: Export the matching packed GRU and recent-mean architecture with complete category routing
def _ensure_loaded():
    global _device
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError("GAGC_DATASETS must contain all four categories exactly once")
    if _models:
        if set(_models) != set(DATASETS) or set(_metas) != set(DATASETS):
            raise RuntimeError("Incomplete GRU4Rec artifact loading")
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    loaded_models, loaded_metas = {}, {}
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing GRU4Rec artifacts for {ds}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if (meta.get("architecture") != "packed_gru_recent_mean_v1"
                or meta.get("pooling") != "compact_causal_shifted_mean"
                or meta.get("score_activation") != "linear"
                or meta.get("empty_history_policy") != "zero_features"):
            raise ValueError(f"Incompatible GRU4Rec architecture for {ds}")
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            dropout_rate=meta["dropout_rate"],
            recent_window=meta["recent_window"],
        ).to(_device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=_device, weights_only=True), strict=True
        )
        model.eval()
        loaded_models[ds] = model
        loaded_metas[ds] = meta
    _models.update(loaded_models)
    _metas.update(loaded_metas)
### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ### >>> Self_EvolveRec-BLOCK-START: Require explicit routing to the single trained model for each category
    _ensure_loaded()
    if dataset_name is None:
        raise ValueError("dataset_name is required for the four-category benchmark")
    if dataset_name not in _models:
        raise ValueError(f"No trained GRU4Rec model for {dataset_name!r}")
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    if len(candidates) == 0:
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


### >>> Self_EvolveRec-BLOCK-START: Validate all four required category routes before training
def main() -> None:
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError("GAGC_DATASETS must contain all four categories exactly once")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"Per-dataset GRU4Rec training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


