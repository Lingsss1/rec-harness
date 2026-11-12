# === main_code.py ===
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


### >>> Self_EvolveRec-BLOCK-START: Controlled pairwise and BPR-max objectives on identical joint GRU features


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
        raise ValueError("sequence and positives must be aligned B x L tensors")
    if neg.ndim == 2:
        neg = neg.unsqueeze(-1)
    if neg.ndim != 3 or neg.shape[:2] != seq.shape or neg.size(-1) < 1:
        raise ValueError("negatives must have shape B x L x K with K >= 1")
    if objective not in ("baseline", "bpr_max"):
        raise ValueError(f"unknown objective: {objective}")

    valid = seq.ne(0) & pos.ne(0) & neg.gt(0).all(dim=-1)
    if positional:
        if distance is None or distance.shape != seq.shape:
            raise ValueError("recency weighting requires aligned B x L target ages")
        ages = distance[valid].float()
        if not torch.isfinite(ages).all() or ages.lt(0).any():
            raise ValueError("valid target ages must be finite and nonnegative")

    # This is the same complete encoder used by forward, prediction and probes.
    features = model.log2feats(seq)
    if not valid.any():
        return features.sum() * 0.0
    features = features[valid]
    positive = pos[valid]
    negative = neg[valid]
    if positional:
        weights = 0.5 + 0.5 * torch.exp(-ages / 10.0)
    else:
        weights = features.new_ones(features.size(0))

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
            # Use every sampled column, with count-independent pairwise scaling.
            # Preserve the supplied baseline's probability epsilon convention.
            return -torch.logaddexp(preference, preference.new_tensor(1e-8).log()).mean(
                dim=-1
            )
        log_weight = torch.log_softmax(negative_score, dim=-1)
        ranking = -torch.logsumexp(log_weight + preference, dim=-1)
        if SCORE_REG:
            ranking = ranking + SCORE_REG * (
                log_weight.exp() * negative_score.square()
            ).sum(dim=-1)
        return ranking

    # Chunk positions, retaining ALL negative columns within each softmax.
    # Checkpoint embedding lookup as well as scoring to bound saved activations.
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
    # Preserve the supplied global weighted-position reduction.
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
### >>> Self_EvolveRec-BLOCK-START: Independently configurable objective controls within host limits
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))
# Each invocation fits exactly one joint model per category. Preserve the
# current defaults while allowing separate host trials to isolate each factor.
LOSS_OBJECTIVE = os.environ.get("GRU4REC_OBJECTIVE", "bpr_max")
WEIGHTING = os.environ.get("GRU4REC_WEIGHTING", "positional")
NEGATIVE_COUNT = int(os.environ.get("GRU4REC_NEGATIVES", "32"))
SCORE_REG = float(os.environ.get("GRU4REC_SCORE_REG", "0.001"))
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
    raise ValueError(
        "dimensions, batch size, negatives, cadence and patience must be positive"
    )
if not all(math.isfinite(x) for x in (LR, WEIGHT_DECAY, DROPOUT_RATE, SCORE_REG)):
    raise ValueError("optimizer, dropout and score regularization must be finite")
if LR <= 0 or WEIGHT_DECAY < 0 or SCORE_REG < 0 or not 0 <= DROPOUT_RATE < 1:
    raise ValueError("invalid optimizer, dropout or score regularization settings")
POSITIONAL_WEIGHTING = WEIGHTING == "positional"
print(
    json.dumps(
        {
            "resolved_objective": LOSS_OBJECTIVE,
            "negative_count": NEGATIVE_COUNT,
            "bpr_max_score_regularization": SCORE_REG,
            "pairwise_probability_epsilon": 1e-8,
            "bpr_max_probability_epsilon": 0.0,
            "reduction": "global_weighted_position_mean",
            "weighting": WEIGHTING,
            "maximum_epochs": NUM_EPOCHS,
        }
    ),
    flush=True,
)
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
        ### >>> Self_EvolveRec-BLOCK-START: Jointly trained recent-item readout with isolated initialization
        # Restore the CPU RNG afterward so added parameters do not shift the
        # baseline's subsequent dropout draws. These modules initialize on CPU.
        with torch.random.fork_rng(devices=[]):
            self.recent_project = nn.Linear(hidden_units, hidden_units, bias=False)
            self.recent_gate = nn.Linear(2 * hidden_units, 1)
            nn.init.xavier_uniform_(self.recent_project.weight)
            nn.init.zeros_(self.recent_gate.weight)
            nn.init.constant_(self.recent_gate.bias, -4.0)
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Padding-invariant GRU with a small jointly trained recent-item residual
    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        batch_size, width = sequence.shape
        lengths = sequence.ne(0).sum(dim=1)
        original_rows = lengths.gt(0).nonzero(as_tuple=True)[0]
        if original_rows.numel() == 0:
            return self.item_emb(sequence) * 0.0

        # Pack chronological real tokens, never the original left-padded rows.
        tokens = sequence[original_rows]
        valid = tokens.ne(0)
        rows, columns = valid.nonzero(as_tuple=True)
        compact_columns = valid.long().cumsum(dim=1)[rows, columns] - 1
        compact = torch.zeros_like(tokens)
        compact[rows, compact_columns] = tokens[rows, columns]
        embedded = self.dropout(self.item_emb(compact))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths[original_rows].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_features, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            packed_features, batch_first=True, total_length=width
        )

        # Both paths use the SAME transformed input embeddings and dropout draw.
        # Every scoring and diagnostic entry point receives these joint features.
        base = unpacked[rows, compact_columns]
        current = embedded[rows, compact_columns]
        gate = torch.sigmoid(self.recent_gate(torch.cat((base, current), dim=-1)))
        mixed = (1.0 - gate) * base + gate * self.recent_project(current)
        features = mixed.new_zeros(batch_size, width, self.hidden_units)
        features[original_rows[rows], columns] = mixed
        return features

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        # Aligned B x L targets, or one target for each feature vector.
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        # Shared candidate catalog for every feature vector.
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        # One candidate pool per feature vector, including B x L x K pools.
        if items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        raise ValueError(
            f"incompatible feature and item shapes: {features.shape}, {items.shape}"
        )

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Share the complete trained readout across training and inference
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
                ### >>> Self_EvolveRec-BLOCK-START: Identify the complete joint checkpoint architecture
                "dropout_rate": DROPOUT_RATE,
                "architecture": "packed_gru_recent_gate_v1",
                "empty_history_policy": "zero_features",
                "score_activation": "linear",
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
        ### >>> Self_EvolveRec-BLOCK-START: Load exactly one matching joint GRU checkpoint for each category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing GRU4Rec artifacts for {ds}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("architecture") != "packed_gru_recent_gate_v1":
            raise ValueError(f"Incompatible GRU4Rec architecture for {ds}")
        # Importing the training class includes both the packed encoder and
        # jointly trained projection/gate. Strict loading checks every parameter.
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            dropout_rate=meta["dropout_rate"],
        ).to(_device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=_device, weights_only=True), strict=True
        )
        ### <<< Self_EvolveRec-BLOCK-END
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve explicit category routing without cross-category fallback
    _ensure_loaded()
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required for the four-category benchmark")
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


def main() -> None:
    ### >>> Self_EvolveRec-BLOCK-START: Require every benchmark category exactly once
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


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
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
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Keep the trained module distinct from its diagnostic view
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories and review attributes are unavailable in this task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Read real item embeddings through the ModelView contract
            emb = self.view.item_emb.weight.detach()
            emb = emb[1 : self.model.itemnum + 1]
            if not torch.isfinite(emb).all().item():
                raise FloatingPointError("Nonfinite item embeddings in diagnosis")
            ### <<< Self_EvolveRec-BLOCK-END

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Require at least two sampled embeddings for pairwise geometry
            if min(int(sample_size), n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Measure embedding alignment without a quadratic Gram matrix
            x = x.float()
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Restore model mode even when a standalone margin probe fails
        was_training = self.model.training
        self.model.eval()
        try:
            return self._measure_pos_neg_margins(
                dataloader, num_batches, eps, topk_categories
            )
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def _measure_pos_neg_margins(self, dataloader, num_batches, eps, topk_categories):
        ### <<< Self_EvolveRec-BLOCK-END

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

            ### >>> Self_EvolveRec-BLOCK-START: Use aligned tensor inputs for family-specific GRU diagnostics
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.view.log2feats(seq_t)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Score diagnostic candidates through the family-specific view
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.view.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padded inputs and absent negative targets
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Reject invalid measured margins instead of reporting misleading rates
            if valid.any():
                measured = margin[valid].detach().float()
                if not torch.isfinite(measured).all().item():
                    raise FloatingPointError(
                        "Nonfinite valid positive-negative margins"
                    )
                margins_all.append(measured.cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Keep diagnostic counts aligned with measured margins
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Count measured errors without unavailable category lookups
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Do not report absent samples as healthy zero-error measurements
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "valid_positions": 0,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            ### >>> Self_EvolveRec-BLOCK-START: Report the measured diagnostic sample count
            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
                "valid_positions": int(N),
            }
            ### <<< Self_EvolveRec-BLOCK-END

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Separate measured margin counts from unavailable category breakdowns
        breakdown = {
            "eps": float(eps),
            "valid_positions": margin_metrics["valid_positions"],
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "category_metadata_available": False,
            "low_margin_top_categories": None,
            "wrong_rank_top_categories": None,
        }
        ### <<< Self_EvolveRec-BLOCK-END

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Calibrate frozen GRU padding errors against singleton and explicit recurrence
    @torch.no_grad()
    def analyze_padding_calibration(self, batches, max_rows=16):
        device = torch.device(self.device)
        atol, rtol = 1e-5, 1.3e-6
        result = {
            "sample_count": 0,
            "atol": atol,
            "rtol": rtol,
            "absolute_only_tolerance": 1e-5,
            "valid_state_max_abs_error": 0.0,
            "final_feature_max_abs_error": 0.0,
            "score_max_abs_error": 0.0,
            "max_normalized_tolerance_violation": 0.0,
            "explicit_reference_max_abs_error": 0.0,
            "repeat_max_abs_error": 0.0,
            "top10_changed_rows": 0,
            "first_divergent_valid_token": None,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "dtype": str(self.view.item_emb.weight.dtype),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }

        def compare(actual, reference, key):
            if actual.shape != reference.shape or actual.numel() == 0:
                raise ValueError(f"Invalid padding comparison shapes for {key}")
            if not (
                torch.isfinite(actual).all().item()
                and torch.isfinite(reference).all().item()
            ):
                raise FloatingPointError(f"Nonfinite padding comparison: {key}")
            error = (actual.double() - reference.double()).abs()
            limit = atol + rtol * reference.double().abs()
            result[key] = max(result[key], float(error.max().item()))
            result["max_normalized_tolerance_violation"] = max(
                result["max_normalized_tolerance_violation"],
                float((error / limit).max().item()),
            )
            return (error > limit).reshape(error.shape[0], -1).any(dim=1)

        def explicit_features(tokens):
            # PyTorch GRU gate order is reset, update, new. In eval mode the
            # recurrent inter-layer dropout is inactive. No padded step runs.
            embedded = self.model.dropout(self.view.item_emb(tokens))
            layer_input = embedded
            gru = self.model.gru
            for layer in range(gru.num_layers):
                w_i = getattr(gru, f"weight_ih_l{layer}")
                w_h = getattr(gru, f"weight_hh_l{layer}")
                b_i = getattr(gru, f"bias_ih_l{layer}", None)
                b_h = getattr(gru, f"bias_hh_l{layer}", None)
                hidden = embedded.new_zeros(1, gru.hidden_size)
                states = []
                for step in range(tokens.size(1)):
                    ir, iz, inn = F.linear(layer_input[:, step], w_i, b_i).chunk(3, -1)
                    hr, hz, hn = F.linear(hidden, w_h, b_h).chunk(3, -1)
                    reset = torch.sigmoid(ir + hr)
                    update = torch.sigmoid(iz + hz)
                    proposed = torch.tanh(inn + reset * hn)
                    hidden = (1.0 - update) * proposed + update * hidden
                    states.append(hidden)
                layer_input = torch.stack(states, dim=1)
            current = embedded
            gate = torch.sigmoid(
                self.model.recent_gate(torch.cat((layer_input, current), dim=-1))
            )
            return (1.0 - gate) * layer_input + gate * self.model.recent_project(
                current
            )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.autocast(device_type=device.type, enabled=False):
                for batch in batches[:2]:
                    remaining = max_rows - result["sample_count"]
                    if remaining <= 0:
                        break
                    seq = torch.as_tensor(batch[1], device=device, dtype=torch.long)
                    seq = seq[seq.ne(0).any(dim=1)][:remaining]
                    if seq.size(0) == 0:
                        continue
                    full = self.view.log2feats(seq)
                    repeated = self.view.log2feats(seq)
                    padded = self.view.log2feats(F.pad(seq, (3, 0)))[:, 3:]
                    order = torch.arange(seq.size(0) - 1, -1, -1, device=device)
                    permuted = self.view.log2feats(seq[order])[order]
                    items = torch.arange(
                        1, min(self.model.itemnum, 100) + 1, device=device
                    )
                    for row in range(seq.size(0)):
                        positions = seq[row].ne(0).nonzero(as_tuple=True)[0]
                        tokens = seq[row, positions].unsqueeze(0)
                        reference = self.view.log2feats(tokens)[0]
                        explicit = explicit_features(tokens)[0]
                        # Include independent recurrence and repeated execution
                        # in the declared tolerance check; reject nonfinite values.
                        compare(
                            reference,
                            explicit,
                            "explicit_reference_max_abs_error",
                        )
                        compare(repeated[row], full[row], "repeat_max_abs_error")
                        # Explicit B x K pools avoid the aligned-target ambiguity
                        # when a catalog contains exactly one real item.
                        candidate_pool = items.unsqueeze(0)
                        reference_scores = self.view.predict_score(
                            reference[-1:], candidate_pool
                        )
                        for variant in (full, padded, permuted):
                            actual = variant[row, positions]
                            divergent = compare(
                                actual, reference, "valid_state_max_abs_error"
                            )
                            if (
                                divergent.any()
                                and result["first_divergent_valid_token"] is None
                            ):
                                result["first_divergent_valid_token"] = {
                                    "probe_row": result["sample_count"],
                                    "chronological_index": int(
                                        divergent.nonzero(as_tuple=True)[0][0].item()
                                    ),
                                }
                            compare(
                                actual[-1:],
                                reference[-1:],
                                "final_feature_max_abs_error",
                            )
                            scores = self.view.predict_score(
                                actual[-1:], candidate_pool
                            )
                            compare(scores, reference_scores, "score_max_abs_error")
                        # Stable sorting makes candidate-ID order the tie rule.
                        scores = self.view.predict_score(
                            full[row, positions[-1]].unsqueeze(0), candidate_pool
                        )
                        k = min(10, items.numel())
                        changed = not torch.equal(
                            torch.argsort(scores[0], descending=True, stable=True)[:k],
                            torch.argsort(
                                reference_scores[0], descending=True, stable=True
                            )[:k],
                        )
                        result["top10_changed_rows"] += int(changed)
                        result["sample_count"] += 1
        finally:
            self.model.train(was_training)
        if result["sample_count"]:
            result["elementwise_check_passed"] = (
                result["max_normalized_tolerance_violation"] <= 1.0
            )
            result["absolute_only_check_passed"] = (
                result["valid_state_max_abs_error"] <= 1e-5
            )
        else:
            for key in tuple(result):
                if key.endswith("error") or key == "max_normalized_tolerance_violation":
                    result[key] = None
            result["elementwise_check_passed"] = None
            result["absolute_only_check_passed"] = None
        return result

    ### <<< Self_EvolveRec-BLOCK-END
    def run_full_diagnosis(self, train_loader):
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        ### >>> Self_EvolveRec-BLOCK-START: Share immutable probe batches and restore model mode on failure
        was_training = self.view.model.training
        self.view.model.eval()
        try:
            batches = []
            iterator = iter(train_loader)
            for _ in range(10):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                values = tuple(
                    torch.as_tensor(value).detach().to("cpu", dtype=torch.long).clone()
                    for value in batch[1:4]
                )
                if values[0].ndim != 2 or any(
                    value.shape != values[0].shape for value in values[1:]
                ):
                    raise ValueError(
                        "Diagnostic tensors must have aligned B x L shapes"
                    )
                if any(
                    ((value < 0) | (value > self.model.itemnum)).any().item()
                    for value in values
                ):
                    raise ValueError("Diagnostic IDs exceed the trained vocabulary")
                batches.append((None, *values))
            padding_calibration = self.analyze_padding_calibration(batches)
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            eps = 0.5
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    batches, num_batches=10, eps=eps, topk_categories=20
                )
            )
        finally:
            self.view.model.train(was_training)
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                ### >>> Self_EvolveRec-BLOCK-START: Report measured frozen-checkpoint padding calibration
                "pos_neg_margin_category_breakdown": breakdown,
                "padding_calibration": padding_calibration,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            "metric_definitions": {
                ### >>> Self_EvolveRec-BLOCK-START: Distinguish precision calibration from semantic and validation conclusions
                "padding_calibration": (
                    "Eval/no-grad, autocast-disabled measurements on at most 16 nonempty "
                    "rows from the first two cached training batches. Compare chronological "
                    "unpadded singleton features with original batches, three extra leading "
                    "zeros and reversed batch order. Errors cover all valid states, final "
                    "features and scores for the first at most 100 real item IDs. The "
                    "elementwise criterion is abs(error)<=1e-5+1.3e-6*abs(reference); "
                    "normalized violation is error divided by that bound, including "
                    "explicit-reference and repeated-execution comparisons. Nonfinite "
                    "features or scores raise an error. The separate "
                    "absolute-only state criterion remains 1e-5. Explicit-reference error "
                    "compares singleton features with an eval-mode recurrence using the "
                    "trained GRU weights, PyTorch gate equations and joint recent-item "
                    "readout. Repeat error measures identical repeated batch executions. "
                    "First divergence gives a probe row and zero-based chronological token "
                    "index. Top10 changes compare ordered candidate lists for original batch "
                    "versus singleton, with stable candidate-order tie handling. These are "
                    "fixed training-history candidate probes, not full-catalog rankings or "
                    "validation HR@10. Precision settings and software versions are recorded. "
                    "A failed tolerance alone does not establish padding leakage; unresolved "
                    "errors confound mechanistic context claims. No samples gives null errors "
                    "and pass flags. No model parameters or scores are averaged."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                ### >>> Self_EvolveRec-BLOCK-START: Interpret embedding geometry without inventing behavioral diversity
                "embedding_collapse_score": (
                    "Mean off-diagonal cosine similarity of up to 2048 real item embeddings, "
                    "mapped from [-1,1] to [0,1]. About 0.5 corresponds to zero mean cosine; "
                    "values approaching 1 indicate directional alignment. This does not measure "
                    "recommendation diversity or unobserved simulator traits. Null means insufficient pairs."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                ### >>> Self_EvolveRec-BLOCK-START: Define diagnostic availability and sampling limitations
                "pos_neg_margin_category_breakdown": (
                    "Contains measured valid-position, low-margin and wrong-rank counts with eps=0.5. "
                    "Valid positions require nonzero input, positive and negative IDs. Category "
                    "breakdowns are null because item categories are unavailable. All margin metrics "
                    "are null when no valid samples exist. These are sampled training margins, "
                    "not validation HR@10 or a full-catalog ranking measurement."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
