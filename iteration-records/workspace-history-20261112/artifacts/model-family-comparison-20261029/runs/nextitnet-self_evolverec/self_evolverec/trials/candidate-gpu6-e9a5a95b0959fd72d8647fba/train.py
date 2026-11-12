"""Per-dataset NextItNet baseline for Amazon Reviews.

Trains one independent NextItNet checkpoint per dataset while sharing one default
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

### >>> Self_EvolveRec-BLOCK-START: Fixed benchmark data and training seed
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


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    ### >>> Self_EvolveRec-BLOCK-START: Prevent exhausted-catalog rejection loops in the diagnostic sampler
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    if not users:
        raise RuntimeError(
            "No users have both two training interactions and eligible negatives"
        )
    ### <<< Self_EvolveRec-BLOCK-END
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in interacted:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


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
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            candidates = [target]
            while len(candidates) < 100 and itemnum > 0:
                neg_item = random.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
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


def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    ### >>> Self_EvolveRec-BLOCK-START: Numerically stable baseline objective
    loss = nn.functional.softplus(neg_scores - pos_scores)
    ### <<< Self_EvolveRec-BLOCK-END
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# Hyper-parameters
MAXLEN = int(os.environ.get("NEXTITNET_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("NEXTITNET_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("NEXTITNET_BLOCKS", "8"))
KERNEL_SIZE = int(os.environ.get("NEXTITNET_KERNEL", "3"))
DROPOUT_RATE = float(os.environ.get("NEXTITNET_DROPOUT", "0.2"))
LR = float(os.environ.get("NEXTITNET_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("NEXTITNET_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("NEXTITNET_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Fixed update limit and explicit training controls
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1,
        int(os.environ.get("NEXTITNET_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
# Each invocation trains one configuration, never multiple predictors per category.
# Controls share batches, retained targets and negatives by category/update key.
TRAINING_MODE = os.environ.get("NEXTITNET_TRAINING_MODE", "cropped_ce")
if TRAINING_MODE not in {
    "reference",
    "full_bpr",
    "cropped_bpr",
    "full_ce",
    "cropped_ce",
}:
    raise ValueError(f"Unknown training mode: {TRAINING_MODE}")
NEGATIVE_COUNT = 8
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("NEXTITNET_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("NEXTITNET_PATIENCE", "20"))


class NextItNetBlock(nn.Module):
    def __init__(
        self, hidden_units: int, dilation: int, kernel_size: int, dropout_rate: float
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))
        return conv(x)

    ### >>> Self_EvolveRec-BLOCK-START: Prevent padding leakage between causal convolutions
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x * mask
        y = (self.norm1(residual) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2) * mask
        y = self.dropout(self.activation(y)) * mask
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2) * mask
        return (residual + self.dropout(y)) * mask

    ### <<< Self_EvolveRec-BLOCK-END


class NextItNetModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList(
            [
                NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
                for i in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        ### >>> Self_EvolveRec-BLOCK-START: Mask all temporal block boundaries and final features
        mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        x = self.dropout(x) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x) * mask
        ### <<< Self_EvolveRec-BLOCK-END

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        user_emb = out[0, last[0]]
        cand_emb = self.item_emb(candidates)
        return torch.matmul(cand_emb, user_emb)


### >>> Self_EvolveRec-BLOCK-START: Target-matched suffix supervision with vectorized exact negative sampling
class TargetMatchedSampler:
    def __init__(self, user_train, itemnum, dataset_name):
        self.itemnum = itemnum
        self.category_key = DEFAULT_DATASETS.index(dataset_name)
        self.histories = user_train
        self.users = []
        self.exclusions = {}
        excluded_users = 0
        for user in sorted(user_train):
            history = user_train[user]
            if len(history) < 2:
                continue
            if any(i < 1 or i > itemnum for i in history):
                raise ValueError("Training history contains an invalid item ID")
            excluded = np.asarray(sorted(set(history)), dtype=np.int64) - 1
            eligible = itemnum - len(excluded)
            if eligible == 0:
                excluded_users += 1
                continue
            # Rank-to-ID thresholds for the complement of the training-item set.
            self.exclusions[user] = (excluded - np.arange(len(excluded)), eligible)
            self.users.append(user)
        if not self.users:
            raise RuntimeError(
                "No training users have both targets and eligible negatives"
            )
        print(
            f"  [{dataset_name}] negative_exhausted_users={excluded_users}", flush=True
        )

    @staticmethod
    def distinct_ranks(rng, population, count, size):
        if not (0 < count <= population) or size < 1:
            raise ValueError("Negative sampling requires positive valid dimensions")
        # Each target independently draws ordered uniform ranks without replacement.
        # At each step, map a uniform complement rank past earlier selected ranks.
        # Vectorize across targets; never allocate a catalog-sized permutation.
        ranks = np.empty((size, count), dtype=np.int64)
        for offset in range(count):
            draw = rng.integers(population - offset, size=size)
            if offset:
                thresholds = np.sort(ranks[:, :offset], axis=1)
                thresholds = thresholds - np.arange(offset, dtype=np.int64)
                draw = draw + (thresholds <= draw[:, None]).sum(axis=1)
            ranks[:, offset] = draw
        return ranks

    def batch(self, update, batch_size, maxlen, mode, receptive_field):
        streams = [
            np.random.default_rng(
                np.random.SeedSequence([SEED, self.category_key, update, stream])
            )
            for stream in range(3)
        ]
        batch_rng, suffix_rng, negative_rng = streams
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        retained = np.zeros_like(seq, dtype=bool)
        sampled_users = []
        removed_context = 0
        affected_targets = 0
        for row in range(batch_size):
            user = self.users[int(batch_rng.integers(len(self.users)))]
            sampled_users.append(user)
            history = self.histories[user]
            n = min(maxlen, len(history) - 1)
            start = maxlen - n
            seq[row, start:] = history[-n - 1 : -1]
            pos[row, start:] = history[-n:]
            kept = (
                n if suffix_rng.random() < 0.5 else int(suffix_rng.integers(1, n + 1))
            )
            boundary = maxlen - kept
            retained[row, boundary:] = True
            if mode.startswith("cropped"):
                # Positive targets keep their original positions and identities.
                for target_position in range(boundary, maxlen):
                    removed = max(
                        0, boundary - max(start, target_position - receptive_field + 1)
                    )
                    removed_context += removed
                    affected_targets += int(removed > 0)
                seq[row, :boundary] = 0

        rows, columns = np.nonzero(retained)
        negatives = np.zeros((len(rows), NEGATIVE_COUNT), dtype=np.int64)
        # np.nonzero returns row-major indices, so each row's targets are contiguous.
        # Batch their independent draws instead of executing Python once per target.
        target_counts = np.bincount(rows, minlength=batch_size)
        first = 0
        for row, target_count in enumerate(target_counts):
            if target_count == 0:
                continue
            thresholds, eligible = self.exclusions[sampled_users[row]]
            k = min(NEGATIVE_COUNT, eligible)
            ranks = self.distinct_ranks(negative_rng, eligible, k, int(target_count))
            stop = first + int(target_count)
            negatives[first:stop, :k] = (
                ranks + np.searchsorted(thresholds, ranks, side="right") + 1
            )
            first = stop
        stats = {
            "retained_targets": int(len(rows)),
            "negative_draws": int(np.count_nonzero(negatives)),
            "affected_targets": affected_targets,
            "removed_context_positions": removed_context,
        }
        return seq, pos, rows, columns, negatives, stats


def matched_sequence_loss(model, batch, device, mode):
    seq, pos, rows, columns, negatives, stats = batch
    seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    columns_t = torch.as_tensor(columns, dtype=torch.long, device=device)
    targets = torch.as_tensor(pos[rows, columns], dtype=torch.long, device=device)
    negative_t = torch.as_tensor(negatives, dtype=torch.long, device=device)
    features = model.encode_sequence(seq_t)[rows_t, columns_t]
    positive_scores = (features * model.item_emb(targets)).sum(-1)
    if mode.endswith("bpr"):
        negative_scores = (features * model.item_emb(negative_t[:, 0])).sum(-1)
        loss = nn.functional.softplus(negative_scores - positive_scores).mean()
    else:
        negative_scores = torch.einsum(
            "nd,nkd->nk", features, model.item_emb(negative_t)
        ).masked_fill(negative_t == 0, float("-inf"))
        logits = torch.cat((positive_scores[:, None], negative_scores), dim=1)
        # Uncorrected (K+1)-way classification at temperature one.
        loss = nn.functional.cross_entropy(
            logits, torch.zeros(len(rows), dtype=torch.long, device=device)
        )
    return loss


### <<< Self_EvolveRec-BLOCK-END

### >>> Self_EvolveRec-BLOCK-START: Reset category model randomness independently


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = NextItNetModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)
    ### >>> Self_EvolveRec-BLOCK-START: Wire matched sampler into the unchanged optimizer workflow
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = TargetMatchedSampler(user_train, itemnum, dataset_name)
    # Only causal stride-one convolutions and position-local normalization occur.
    receptive_field = 1 + sum(
        (conv.kernel_size[0] - 1) * conv.dilation[0]
        for block in model.blocks
        for conv in (block.conv1, block.conv2)
    )
    training_counts = defaultdict(int)
    print(
        f"  [{dataset_name}] mode={TRAINING_MODE} receptive_field={receptive_field}"
        f" max_epochs={NUM_EPOCHS} batches_per_epoch={NUM_BATCHES}"
        f" eval_every={EVAL_EVERY} patience={PATIENCE} strict_improvement=True",
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
        ### >>> Self_EvolveRec-BLOCK-START: Execute exactly one selected training configuration
        for batch_index in range(NUM_BATCHES):
            if TRAINING_MODE == "reference":
                seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
                seq_t = torch.tensor(seq, dtype=torch.long, device=device)
                pos_t = torch.tensor(pos, dtype=torch.long, device=device)
                neg_t = torch.tensor(neg, dtype=torch.long, device=device)
                loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
                training_counts["retained_targets"] += int(np.count_nonzero(pos))
            else:
                batch = sampler.batch(
                    (epoch - 1) * NUM_BATCHES + batch_index,
                    BATCH_SIZE,
                    MAXLEN,
                    TRAINING_MODE,
                    receptive_field,
                )
                loss = matched_sequence_loss(model, batch, device, TRAINING_MODE)
                for key, value in batch[-1].items():
                    training_counts[key] += value
            training_counts["updates"] += 1
            ### <<< Self_EvolveRec-BLOCK-END
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        ### >>> Self_EvolveRec-BLOCK-START: Report actual supervision and context removal counts
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        print(f"  [{dataset_name}] training_counts={dict(training_counts)}", flush=True)
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
                "num_blocks": NUM_BLOCKS,
                "kernel_size": KERNEL_SIZE,
                ### >>> Self_EvolveRec-BLOCK-START: Serialize the single-model training specification
                "dropout_rate": DROPOUT_RATE,
                "architecture_version": "nextitnet_masked_causal_v3",
                "training_mode": TRAINING_MODE,
                "negative_count": NEGATIVE_COUNT,
                "negative_policy": "unique_uniform_complete_training_history_exclusion",
                "negative_rng": "seedsequence_42_category_update_stream_vectorized_complement_ranks",
                "suffix_full_probability": 0.5,
                "temperature": 1.0,
                "receptive_field": receptive_field,
                "training_counts": dict(training_counts),
                "actual_epochs": epoch,
                "best_epoch": best_epoch,
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
    script = r'''"""Per-dataset NextItNet predict function."""
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
        f"nextitnet_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NextItNetModel


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
        ### >>> Self_EvolveRec-BLOCK-START: Require every routed category checkpoint
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing NextItNet artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the checkpoint architecture consistently
            dropout_rate=meta["dropout_rate"],
            ### <<< Self_EvolveRec-BLOCK-END
            maxlen=meta["maxlen"],
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid routing instead of scoring with another category
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained NextItNet model for {dataset_name!r}")
        ds = dataset_name
    elif _models:
        ds = next(iter(_models))
    else:
        raise RuntimeError("No trained NextItNet category models were loaded")
    model = _models[ds]
    meta = _metas[ds]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset NextItNet training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


