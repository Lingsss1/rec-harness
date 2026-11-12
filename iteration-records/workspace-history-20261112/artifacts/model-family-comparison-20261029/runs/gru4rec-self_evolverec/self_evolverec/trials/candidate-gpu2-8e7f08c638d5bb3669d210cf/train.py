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

### >>> Self_EvolveRec-BLOCK-START: Fixed benchmark data boundary and training seed
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


### >>> Self_EvolveRec-BLOCK-START: Distinct filtered negatives with independent deterministic sampling streams
class TrainingSampler:
    def __init__(self, user_train, itemnum, seed=42):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, items in user_train.items() if len(items) >= 2]
        if not self.users:
            raise RuntimeError("no users with at least two training interactions")
        self.user_rng = random.Random(seed)
        self.negative_rng = np.random.default_rng(seed)
        self.forbidden = {
            u: np.asarray(
                sorted({int(i) for i in user_train[u] if 1 <= int(i) <= itemnum}),
                dtype=np.int64,
            )
            for u in self.users
        }
        self.complements = {}
        self.omitted_positions = 0

    def _negatives(self, user, length, slots):
        forbidden = self.forbidden[user]
        eligible = self.itemnum - len(forbidden)
        count = min(slots, eligible)
        result = np.zeros((length, slots), dtype=np.int64)
        if count == 0:
            self.omitted_positions += length
            return result
        if eligible < 2 * slots or len(forbidden) > self.itemnum // 2:
            if user not in self.complements:
                self.complements[user] = np.setdiff1d(
                    np.arange(1, self.itemnum + 1, dtype=np.int64),
                    forbidden,
                    assume_unique=True,
                )
            pool = self.complements[user]
            for row in range(length):
                result[row, :count] = self.negative_rng.choice(
                    pool, size=count, replace=False
                )
            return result

        # Sorted membership checks avoid repeatedly sorting or constructing
        # lookup tables inside rejection sampling. Conditional draws remain
        # uniform without replacement within each position.
        for column in range(count):
            values = self.negative_rng.integers(1, self.itemnum + 1, size=length)
            while True:
                indices = np.searchsorted(forbidden, values)
                invalid = np.zeros(length, dtype=bool)
                in_bounds = indices < forbidden.size
                invalid[in_bounds] = forbidden[indices[in_bounds]] == values[in_bounds]
                if column:
                    invalid |= (values[:, None] == result[:, :column]).any(axis=1)
                if not invalid.any():
                    break
                values[invalid] = self.negative_rng.integers(
                    1, self.itemnum + 1, size=int(invalid.sum())
                )
            result[:, column] = values
        return result

    def sample(self, batch_size, maxlen, negative_count=16):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, negative_count), dtype=np.int64)
        age = np.zeros_like(seq)
        for row in range(batch_size):
            user = self.user_rng.choice(self.users)
            history = self.user_train[user]
            length = min(maxlen, len(history) - 1)
            start = maxlen - length
            seq[row, start:] = history[-length - 1 : -1]
            pos[row, start:] = history[-length:]
            # Targets have original indices T-length through T-1.
            age[row, start:] = np.arange(length - 1, -1, -1)
            neg[row, start:] = self._negatives(user, length, negative_count)
        return seq, pos, neg, age

    def exposure_summary(self, maxlen):
        # Exact conditional exposure under the unchanged deterministic latest
        # suffix law. User selection is uniform; there is no crop randomness.
        summary = {
            name: {"targets": 0, "uniform_mass": 0.0, "weighted_mass": 0.0}
            for name in ("0_to_9", "10_to_29", "30_plus")
        }
        ess = []
        eligible_users = 0
        for user in self.users:
            if len(self.forbidden[user]) == self.itemnum:
                continue
            eligible_users += 1
            length = min(maxlen, len(self.user_train[user]) - 1)
            ages = np.arange(length - 1, -1, -1)
            weights = 0.5 + 0.5 * np.exp(-ages / 10.0)
            ess.append(float(weights.sum() ** 2 / np.square(weights).sum()))
            for name, mask in (
                ("0_to_9", ages < 10),
                ("10_to_29", (ages >= 10) & (ages < 30)),
                ("30_plus", ages >= 30),
            ):
                summary[name]["targets"] += int(mask.sum())
                summary[name]["uniform_mass"] += float(mask.sum() / length)
                summary[name]["weighted_mass"] += float(
                    weights[mask].sum() / weights.sum()
                )
        for values in summary.values():
            values["uniform_mass"] /= max(eligible_users, 1)
            values["weighted_mass"] /= max(eligible_users, 1)
        return {
            "law": "uniform_user_then_deterministic_latest_suffix",
            "eligible_users": eligible_users,
            "age_bins": summary,
            "mean_window_weight_ess": float(np.mean(ess)) if ess else None,
            "note": "conditional supervision mass; older targets outside suffix have zero exposure",
        }


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    # Retain one dedicated diagnostic stream instead of returning the same
    # first batch on every call. Cache only the currently used dataset.
    sampler = getattr(sample_batch, "_diagnostic_sampler", None)
    if (
        sampler is None
        or sampler.user_train is not user_train
        or sampler.itemnum != itemnum
    ):
        sampler = TrainingSampler(user_train, itemnum, seed=42)
        sample_batch._diagnostic_sampler = sampler
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
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        rng = random.Random(SEED)
        val_users = rng.sample(val_users, num_users)
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    ### >>> Self_EvolveRec-BLOCK-START: Deterministic local validation draws without perturbing training
    was_training = model.training
    candidate_rng = random.Random(SEED)
    model.eval()
    with torch.no_grad():
        ### <<< Self_EvolveRec-BLOCK-END
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            ### >>> Self_EvolveRec-BLOCK-START: Guard unique validation candidate availability
            available = itemnum - sum(1 for item in interacted if 1 <= item <= itemnum)
            if available < 99:
                raise ValueError(
                    f"validation user {user} has fewer than 99 eligible negatives"
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
    ### >>> Self_EvolveRec-BLOCK-START: Restore caller model mode after validation
    model.train(was_training)
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }
    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Uniform-smoothed detached competitive pairwise logistic supervision
@torch.no_grad()
def verify_gru_encoding(model, device):
    """Check aligned recurrence, final readout and scores without RNG draws."""
    was_training = model.training
    model.eval()
    try:
        sequence = torch.tensor(
            [
                [0, 0, 1, 2],
                [1, 2, 3, 4],
                [0, 0, 0, 1],
                [0, 0, 2, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 2],
            ],
            dtype=torch.long,
            device=device,
        )
        sequence = torch.where(
            sequence > 0, (sequence - 1).remainder(model.itemnum) + 1, sequence
        )
        candidates = torch.arange(1, min(model.itemnum, 5) + 1, device=device)
        features = model.log2feats(sequence)
        final = model.encode(sequence)
        torch.testing.assert_close(
            features[sequence == 0],
            torch.zeros_like(features[sequence == 0]),
            atol=0.0,
            rtol=0.0,
        )
        for row in range(sequence.size(0)):
            tokens = sequence[row][sequence[row] != 0].unsqueeze(0)
            if tokens.size(1) == 0:
                torch.testing.assert_close(
                    final[row], torch.zeros_like(final[row]), atol=0.0, rtol=0.0
                )
                continue
            reference, _ = model.gru(model.item_emb(tokens))
            torch.testing.assert_close(
                features[row, sequence[row] != 0],
                reference[0],
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                final[row], reference[0, -1], atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                model.score_candidates(sequence[row : row + 1], candidates),
                model.item_emb(candidates) @ reference[0, -1],
                atol=1e-5,
                rtol=1e-5,
            )
        permutation = torch.tensor([4, 2, 0, 5, 3, 1], device=device)
        torch.testing.assert_close(
            model.log2feats(sequence[permutation]),
            features[permutation],
            atol=1e-5,
            rtol=1e-5,
        )
        padded = torch.nn.functional.pad(sequence, (3, 0))
        torch.testing.assert_close(
            model.log2feats(padded)[:, 3:], features, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(model.encode(padded), final, atol=1e-5, rtol=1e-5)
        empty = torch.zeros_like(sequence)
        torch.testing.assert_close(
            model.log2feats(empty), torch.zeros_like(features), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            model.encode(empty), torch.zeros_like(final), atol=0.0, rtol=0.0
        )
    finally:
        model.train(was_training)


def bpr_sequence_loss(
    model: nn.Module,
    seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    distance: torch.Tensor | None = None,
) -> torch.Tensor:
    if neg.ndim == 2:
        neg = neg.unsqueeze(-1)
    if neg.ndim != 3 or neg.shape[:2] != seq.shape or neg.size(-1) < 1:
        raise ValueError("negative IDs must have shape B x L x K")
    valid = seq.ne(0) & pos.ne(0) & neg.gt(0).any(dim=-1)
    features = model.log2feats(seq)
    if not valid.any():
        return features.sum() * 0.0
    if RECENCY_WEIGHTING:
        if distance is None or distance.shape != seq.shape:
            raise ValueError("recency requires original training-target ages")
        ages = distance[valid].float()
        if not torch.isfinite(ages).all() or (ages < 0).any():
            raise ValueError("target ages must be finite and nonnegative")
    else:
        ages = None

    # Normalize positions within each nonempty window before averaging windows.
    raw_weights = torch.zeros_like(seq, dtype=torch.float32)
    raw_weights[valid] = (
        0.5 + 0.5 * torch.exp(-ages / 10.0) if ages is not None else 1.0
    )
    # Preserve exact window normalization even when its total weight is < 1.
    row_mass = raw_weights.sum(dim=1, keepdim=True)
    active_rows = row_mass.squeeze(-1).gt(0).sum()
    safe_mass = torch.where(row_mass > 0, row_mass, torch.ones_like(row_mass))
    weights = (raw_weights / safe_mass)[valid] / active_rows
    features, positives, negatives = features[valid], pos[valid], neg[valid]

    from torch.utils.checkpoint import checkpoint

    def chunk_loss(feats, positive_ids, negative_ids):
        p = model.predict_score(feats, positive_ids).float()
        n = model.predict_score(feats, negative_ids).float()
        mask = negative_ids.gt(0)
        with torch.no_grad():
            competitive = torch.softmax(
                (n.detach() / COMPETITIVE_TEMPERATURE).masked_fill(~mask, -torch.inf),
                dim=-1,
            )
            uniform = mask.float() / mask.sum(dim=-1, keepdim=True)
            allocation = (
                1.0 - COMPETITIVE_MASS
            ) * uniform + COMPETITIVE_MASS * competitive
        safe_n = n.masked_fill(~mask, 0.0)
        pairwise = torch.nn.functional.softplus(safe_n - p.unsqueeze(-1))
        return (allocation * pairwise.masked_fill(~mask, 0.0)).sum(dim=-1)

    total = features.new_zeros((), dtype=torch.float32)
    for start in range(0, features.size(0), 1024):
        stop = start + 1024
        losses = checkpoint(
            chunk_loss,
            features[start:stop],
            positives[start:stop],
            negatives[start:stop],
            use_reentrant=False,
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
### >>> Self_EvolveRec-BLOCK-START: One controlled configuration per trial within fixed training limits
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))
NEGATIVE_COUNT = 16
COMPETITIVE_MASS = float(os.environ.get("GRU4REC_COMPETITIVE_MASS", "0.25"))
COMPETITIVE_TEMPERATURE = float(
    os.environ.get("GRU4REC_COMPETITIVE_TEMPERATURE", "2.0")
)
RECENCY_MODE = os.environ.get("GRU4REC_WEIGHTING", "uniform")
if RECENCY_MODE not in ("uniform", "absolute"):
    raise ValueError("GRU4REC_WEIGHTING must be uniform or absolute")
RECENCY_WEIGHTING = RECENCY_MODE == "absolute"
if not math.isfinite(COMPETITIVE_MASS) or not 0 <= COMPETITIVE_MASS <= 1:
    raise ValueError("competitive mass must be finite and in [0, 1]")
if not math.isfinite(COMPETITIVE_TEMPERATURE) or COMPETITIVE_TEMPERATURE <= 0:
    raise ValueError("competitive temperature must be finite and positive")
if min(MAXLEN, HIDDEN_UNITS, NUM_LAYERS, BATCH_SIZE, EVAL_EVERY, PATIENCE) < 1:
    raise ValueError("model dimensions and training cadence must be positive")
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

    ### >>> Self_EvolveRec-BLOCK-START: Padding-invariant aligned GRU features shared by training diagnostics and export
    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        batch, width = sequence.shape
        lengths = sequence.ne(0).sum(dim=1)
        original_rows = lengths.gt(0).nonzero(as_tuple=True)[0]
        if original_rows.numel() == 0:
            return self.item_emb(sequence) * 0.0
        tokens = sequence[original_rows]
        mask = tokens.ne(0)
        rows, columns = mask.nonzero(as_tuple=True)
        compact_columns = mask.long().cumsum(dim=1)[rows, columns] - 1
        compact = torch.zeros_like(tokens)
        compact[rows, compact_columns] = tokens[rows, columns]
        packed = nn.utils.rnn.pack_padded_sequence(
            self.dropout(self.item_emb(compact)),
            lengths[original_rows].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_features, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            packed_features, batch_first=True, total_length=width
        )
        # unpacking already restores input order; do not unsort again.
        features = unpacked.new_zeros(batch, width, self.hidden_units)
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
        raise ValueError("incompatible feature and item shapes")

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

    ### >>> Self_EvolveRec-BLOCK-START: Reproducible category initialization and explicit experimental configuration
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
    print(f"  [{dataset_name}] GRU encoding checks passed", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "seed": SEED,
                "objective": "detached_uniform_smoothed_pairwise_logistic",
                "competitive_mass": COMPETITIVE_MASS,
                "temperature": COMPETITIVE_TEMPERATURE,
                "weighting": RECENCY_MODE,
                "negative_count": NEGATIVE_COUNT,
                "negative_sampling": "uniform_distinct_excluding_full_training_history",
                "reduction": "normalized_window_then_mean_nonempty_windows",
                "exposure": sampler.exposure_summary(MAXLEN),
                "epochs": NUM_EPOCHS,
                "batches_per_epoch": NUM_BATCHES,
                "eval_every": EVAL_EVERY,
                "patience": PATIENCE,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "layers": NUM_LAYERS,
                "dropout": DROPOUT_RATE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
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
            ### >>> Self_EvolveRec-BLOCK-START: Invoke distinct-negative competitive objective with original target ages
            seq, pos, neg, age = sampler.sample(
                BATCH_SIZE, MAXLEN, negative_count=NEGATIVE_COUNT
            )
            optimizer.zero_grad(set_to_none=True)
            valid = (seq != 0) & (pos != 0) & (neg > 0).any(axis=-1)
            if not valid.any():
                continue
            if epoch == 1 and _ == 0:
                pools = neg[valid]
                counts = (pools > 0).sum(axis=-1)
                ordered = np.sort(pools, axis=-1)
                if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] > 0)).any():
                    raise AssertionError(
                        "duplicate eligible negatives within a position"
                    )
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "first_batch_valid_positions": int(valid.sum()),
                            "first_batch_mean_distinct_negatives": float(counts.mean()),
                            "first_batch_min_distinct_negatives": int(counts.min()),
                        }
                    ),
                    flush=True,
                )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            age_t = torch.as_tensor(age, dtype=torch.long, device=device)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t, distance=age_t)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"nonfinite loss in {dataset_name}, epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.item())
            ### <<< Self_EvolveRec-BLOCK-END

        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}  val_score: {hr10:.4f}"
                f"  HR@20={val_m['HR@20']:.4f}  NDCG@10={val_m['NDCG@10']:.4f}",
                flush=True,
            )
            ### >>> Self_EvolveRec-BLOCK-START: Retain every evaluated checkpoint metric and skipped-position count
            with open(
                os.path.join(working_dir, f"{dataset_name}_validation.jsonl"), "a"
            ) as record:
                record.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "metrics": val_m,
                            "omitted_positions": sampler.omitted_positions,
                            "competitive_mass": COMPETITIVE_MASS,
                            "temperature": COMPETITIVE_TEMPERATURE,
                            "weighting": RECENCY_MODE,
                        }
                    )
                    + "\n"
                )
            ### <<< Self_EvolveRec-BLOCK-END
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

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Identify the exact exported GRU encoding
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
        ### >>> Self_EvolveRec-BLOCK-START: Load one matching checkpoint per required category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing GRU4Rec artifacts for {ds}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("encoding") != "padding_invariant_gru_v1":
            raise ValueError(f"Incompatible GRU encoding metadata for {ds}")
        # ModelClass imports the same compact/pack/scatter implementation that
        # trained the checkpoint, including zero handling and raw score mapping.
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
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Explicit category routing without cross-category fallback
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required for four-category prediction")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"No trained GRU4Rec model for {dataset_name!r}")
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


### >>> Self_EvolveRec-BLOCK-START: Require all four benchmark category routes
def_main_placeholder = None


def main() -> None:
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError(
            "GAGC_DATASETS must contain all four benchmark categories exactly once"
        )
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


