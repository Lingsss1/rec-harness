# === main_code.py ===
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

### >>> Self_EvolveRec-BLOCK-START: Preserve required data boundary and training seed
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


### >>> Self_EvolveRec-BLOCK-START: Retain optional training user identities and exclude exhausted catalogs
def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users: bool = False,
    eligible_users=None,
):
    users = (
        eligible_users
        if eligible_users is not None
        else [
            u
            for u, items in user_train.items()
            if len(items) >= 2 and len(set(items)) < itemnum
        ]
    )
    if not users:
        raise RuntimeError("no eligible training users with available negatives")
    batch_users = []
    ### <<< Self_EvolveRec-BLOCK-END
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        ### >>> Self_EvolveRec-BLOCK-START: Preserve sampled user identity for training-only exclusions
        history = user_train[user]
        batch_users.append(user)
        ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the diagnostic sampler contract by default
    result = (np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg))
    return (*result, batch_users) if return_users else result
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
    ### >>> Self_EvolveRec-BLOCK-START: Make local checkpoint validation deterministic without consuming training RNG
    rng = random.Random(SEED)
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
            candidates = [target]
            ### >>> Self_EvolveRec-BLOCK-START: Bound local validation candidates by available unseen items
            candidate_count = 1 + min(99, max(0, itemnum - len(interacted - {0})))
            while len(candidates) < candidate_count:
                neg_item = rng.randint(1, itemnum)
                ### <<< Self_EvolveRec-BLOCK-END
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
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
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
### >>> Self_EvolveRec-BLOCK-START: Enforce the fixed training update and epoch limits
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1,
        int(os.environ.get("NEXTITNET_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
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

    ### >>> Self_EvolveRec-BLOCK-START: Prevent biased padding activations from entering causal convolutions
    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(-1).to(x.dtype)
        residual = x * mask
        y = (self.norm1(residual) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2) * mask
        y = self.dropout(self.activation(y)) * mask
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2) * mask
        y = self.dropout(y) * mask
        return (residual + y) * mask

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

    ### >>> Self_EvolveRec-BLOCK-START: Share padding-safe full-prefix features across training and deployment
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        valid = seq.ne(0)
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * valid.long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x) * mask
        for block in self.blocks:
            x = block(x, valid)
        return self.norm(x) * mask

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == 0:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        if items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        if features.ndim == 3 and items.ndim == 2 and features.size(0) == items.size(0):
            return torch.einsum("bld,bcd->blc", features, embeddings)
        raise ValueError(
            f"unsupported scoring shapes: {tuple(features.shape)}, {tuple(items.shape)}"
        )

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
        ### >>> Self_EvolveRec-BLOCK-START: Score every batch row at its final nonpadding position
        rows = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[rows, last]
        user_emb = user_emb * seq.ne(0).any(dim=1, keepdim=True).to(out.dtype)
        scores = self.predict_score(user_emb, candidates)
        # Preserve the predictor's single-history C-shaped return value.
        return scores[0] if seq.size(0) == 1 else scores
        ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Uncorrected candidate CE with CPU-built same-target suffixes and shared parameters
class PrefixCandidateObjective:
    def __init__(self, user_train, itemnum, model):
        self.itemnum = itemnum
        self.negative_rng = np.random.default_rng(SEED)
        self.auxiliary_rng = np.random.default_rng(SEED)
        self.negative_count = 16
        self.auxiliary_weight = 0.25
        self.receptive_field = 1 + sum(
            (conv.kernel_size[0] - 1) * conv.dilation[0]
            for block in model.blocks
            for conv in (block.conv1, block.conv2)
        )
        if self.receptive_field <= 1:
            raise ValueError(
                "recent-prefix regularization requires receptive field > 1"
            )
        self.suffix_length = min(
            model.maxlen,
            8 if self.receptive_field > 8 else self.receptive_field - 1,
        )
        if self.suffix_length < 1:
            raise ValueError("sequence length must be positive")
        self.exclusions = {}
        self.eligible_users = []
        for user, history in user_train.items():
            forbidden = np.asarray(
                sorted({i for i in history if 1 <= i <= itemnum}), dtype=np.int64
            )
            available = itemnum - forbidden.size
            # For eligible zero-based rank r, insert all forbidden IDs having
            # at most r eligible IDs before them. This avoids catalog-sized pools.
            shifted = forbidden - np.arange(1, forbidden.size + 1, dtype=np.int64)
            self.exclusions[user] = (shifted, available)
            if len(history) >= 2 and available > 0:
                self.eligible_users.append(user)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "valid_positions": 0,
            "skipped_no_negatives": 0,
            "usable_positions": 0,
            "longer_than_suffix": 0,
            "auxiliary_positions": 0,
            "optimizer_steps": 0,
        }

    def _candidates(self, sequence, positive, users):
        batch, length = sequence.shape
        candidates = np.zeros((batch, length, 1 + self.negative_count), dtype=np.int64)
        candidate_mask = np.zeros_like(candidates, dtype=np.bool_)
        usable = (sequence != 0) & (positive != 0)
        self.stats["valid_positions"] += int(usable.sum())
        for row, user in enumerate(users):
            slots = np.flatnonzero(usable[row])
            shifted, available = self.exclusions[user]
            if available == 0:
                self.stats["skipped_no_negatives"] += int(slots.size)
                usable[row] = False
                continue
            if slots.size == 0:
                continue
            count = min(self.negative_count, available)
            if available <= self.negative_count:
                # Every eligible item is selected; ordering does not affect CE.
                ranks = np.broadcast_to(np.arange(available), (slots.size, available))
            else:
                ranks = self.negative_rng.integers(available, size=(slots.size, count))
                # Sequential rejection produces an ordered uniform sample
                # without replacement for each prediction position.
                for column in range(1, count):
                    duplicate = (ranks[:, column, None] == ranks[:, :column]).any(
                        axis=1
                    )
                    while duplicate.any():
                        ranks[duplicate, column] = self.negative_rng.integers(
                            available, size=int(duplicate.sum())
                        )
                        duplicate = (ranks[:, column, None] == ranks[:, :column]).any(
                            axis=1
                        )
            negatives = ranks + 1 + np.searchsorted(shifted, ranks, side="right")
            candidates[row, slots, 0] = positive[row, slots]
            candidates[row, slots, 1 : count + 1] = negatives
            candidate_mask[row, slots, : count + 1] = True
        self.stats["usable_positions"] += int(usable.sum())
        prefix_lengths = (sequence != 0).cumsum(axis=1)
        self.stats["longer_than_suffix"] += int(
            (usable & (prefix_lengths > self.suffix_length)).sum()
        )
        return candidates, candidate_mask, np.flatnonzero(usable.reshape(-1))

    def __call__(self, model, sequence, positive, users, device):
        candidates, candidate_mask, positions = self._candidates(
            sequence, positive, users
        )
        if positions.size == 0:
            return None
        seq_t = torch.as_tensor(sequence, dtype=torch.long, device=device)
        position_t = torch.as_tensor(positions, dtype=torch.long, device=device)
        candidate_t = torch.as_tensor(
            candidates.reshape(-1, candidates.shape[-1])[positions],
            dtype=torch.long,
            device=device,
        )
        mask_t = torch.as_tensor(
            candidate_mask.reshape(-1, candidate_mask.shape[-1])[positions],
            dtype=torch.bool,
            device=device,
        )
        features = model.log2feats(seq_t).reshape(-1, model.hidden_units)[position_t]
        logits = model.predict_score(features, candidate_t).masked_fill(
            ~mask_t, -torch.inf
        )
        labels = torch.zeros(positions.size, dtype=torch.long, device=device)
        main_loss = nn.functional.cross_entropy(logits, labels)

        selected = self.auxiliary_rng.choice(
            positions.size, size=min(sequence.shape[0], positions.size), replace=False
        )
        selected_t = torch.as_tensor(selected, dtype=torch.long, device=device)
        # The original batch is already on CPU. Construct suffixes there to
        # avoid one CUDA nonzero synchronization per selected position.
        selected_positions = positions[selected]
        suffix_array = np.zeros((selected.size, sequence.shape[1]), dtype=np.int64)
        for index, flat_position in enumerate(selected_positions):
            row, end = divmod(int(flat_position), sequence.shape[1])
            # Include sequence[end], but never sequence[end + 1].
            prefix = sequence[row, : end + 1]
            recent = prefix[prefix != 0][-self.suffix_length :]
            if recent.size == 0:
                raise RuntimeError(
                    "usable auxiliary positions must have nonempty prefixes"
                )
            suffix_array[index, -recent.size :] = recent
        suffixes = torch.as_tensor(suffix_array, dtype=torch.long, device=device)
        suffix_features = model.log2feats(suffixes)
        indices = torch.arange(suffixes.size(1), device=device).unsqueeze(0)
        last = indices.expand_as(suffixes).masked_fill(suffixes.eq(0), -1).amax(dim=1)
        final_features = suffix_features[
            torch.arange(suffixes.size(0), device=device), last
        ]
        suffix_logits = model.predict_score(
            final_features, candidate_t[selected_t]
        ).masked_fill(~mask_t[selected_t], -torch.inf)
        auxiliary_loss = nn.functional.cross_entropy(suffix_logits, labels[selected_t])
        self.stats["auxiliary_positions"] += int(selected.size)
        return main_loss + self.auxiliary_weight * auxiliary_loss


### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Initialize each independent category training run with seed 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)
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
    ### >>> Self_EvolveRec-BLOCK-START: Initialize the single-model objective and log its concrete configuration
    objective = PrefixCandidateObjective(user_train, itemnum, model)
    if not objective.eligible_users:
        raise RuntimeError("no eligible training users with available negatives")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    import time

    training_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"  [{dataset_name}] objective=uncorrected_candidate_ce"
        f" negatives={objective.negative_count} auxiliary_weight={objective.auxiliary_weight}"
        f" suffix_length={objective.suffix_length} receptive_field={objective.receptive_field}"
        f" parameters={sum(p.numel() for p in model.parameters())}"
        f" batch={BATCH_SIZE} maxlen={MAXLEN} blocks={NUM_BLOCKS} hidden={HIDDEN_UNITS}"
        f" kernel={KERNEL_SIZE} dropout={DROPOUT_RATE} lr={LR} weight_decay={WEIGHT_DECAY}"
        f" precision=float32 eval_every={EVAL_EVERY} patience={PATIENCE}"
        f" checkpoint_rule=strict_local_HR10_improvement seed={SEED}",
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
        ### >>> Self_EvolveRec-BLOCK-START: Optimize complete-prefix CE plus weak same-target suffix CE
        total_loss = 0.0
        objective.reset_stats()
        for _ in range(NUM_BATCHES):
            seq, pos, neg, users = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                return_users=True,
                eligible_users=objective.eligible_users,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, seq, pos, users, device)
            if loss is None:
                continue
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite training loss in {dataset_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
                model.pos_emb.weight[0].zero_()
            objective.stats["optimizer_steps"] += 1
            total_loss += float(loss.item())

        avg_loss = total_loss / max(objective.stats["optimizer_steps"], 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"  [{dataset_name}] epoch={epoch} training_support={objective.stats}"
            f" long_prefix_fraction={objective.stats['longer_than_suffix'] / max(objective.stats['usable_positions'], 1):.6f}"
            f" elapsed_seconds={time.perf_counter() - training_started:.2f}"
            f" peak_allocated_bytes={torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0}",
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
                "num_blocks": NUM_BLOCKS,
                "kernel_size": KERNEL_SIZE,
                ### >>> Self_EvolveRec-BLOCK-START: Serialize the trained backbone and training intervention
                "dropout_rate": DROPOUT_RATE,
                "architecture": "padding_safe_nextitnet_final_state_v1",
                "training_objective": "uncorrected_candidate_ce",
                "negative_count": objective.negative_count,
                "auxiliary_weight": objective.auxiliary_weight,
                "suffix_length": objective.suffix_length,
                "receptive_field": objective.receptive_field,
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
        ### >>> Self_EvolveRec-BLOCK-START: Require exactly one trained checkpoint for every configured category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing NextItNet artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Validate the architecture loaded from the trial training source
        if meta.get("architecture") != "padding_safe_nextitnet_final_state_v1":
            raise ValueError(f"Unsupported NextItNet architecture for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Preserve category routing without silently using another category model
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple category models are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"Unknown or unloaded dataset: {dataset_name!r}")
    model = _models[dataset_name]
    meta = _metas[dataset_name]
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


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Dependencies for a frozen training-only candidate stress audit
from collections import Counter
from itertools import islice
import hashlib
import html
import os
import pickle
import re
import unicodedata

# DEBUG: Replace unavailable SciPy/sklearn dependencies without disabling the audit.
import math


class _SparseTextIndex:
    """Sparse normalized item vectors and an inverted index for cosine queries."""

    def __init__(self, rows, width):
        self.rows = rows
        self.shape = (len(rows), width)
        self.postings = {}
        for item_index, row in enumerate(rows):
            for feature, value in row.items():
                self.postings.setdefault(feature, []).append((item_index, value))

    def similarities(self, item_ids):
        # Repeated prefix occurrences intentionally contribute repeatedly.
        query = Counter()
        for item in item_ids:
            for feature, value in self.rows[int(item) - 1].items():
                query[feature] += value
        if item_ids:
            query = {feature: value / len(item_ids) for feature, value in query.items()}
        norm = math.sqrt(sum(value * value for value in query.values()))
        similarities = np.zeros(self.shape[0], dtype=np.float64)
        if norm == 0.0:
            return similarities, True
        for feature, value in query.items():
            weight = value / norm
            for item_index, item_weight in self.postings.get(feature, ()):
                similarities[item_index] += weight * item_weight
        return similarities, False


def _fit_tfidf(texts, max_features=100000):
    """Word unigram/bigram TF-IDF using only standard-library sparse maps."""
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    documents = []
    document_frequency = Counter()
    corpus_frequency = Counter()
    for text in texts:
        tokens = token_pattern.findall(text)
        terms = tokens + [
            first + " " + second for first, second in zip(tokens, tokens[1:])
        ]
        counts = Counter(terms)
        documents.append(counts)
        document_frequency.update(counts.keys())
        corpus_frequency.update(counts)
    # Select by corpus term frequency with an explicit deterministic tie rule.
    selected = sorted(
        corpus_frequency, key=lambda term: (-corpus_frequency[term], term)
    )[:max_features]
    vocabulary = {term: index for index, term in enumerate(sorted(selected))}
    idf = [0.0] * len(vocabulary)
    for term, index in vocabulary.items():
        idf[index] = (
            math.log((1.0 + len(texts)) / (1.0 + document_frequency[term])) + 1.0
        )
    rows = []
    for counts in documents:
        row = {
            vocabulary[term]: (1.0 + math.log(count)) * idf[vocabulary[term]]
            for term, count in counts.items()
            if term in vocabulary
        }
        norm = math.sqrt(sum(value * value for value in row.values()))
        if norm:
            row = {index: value / norm for index, value in row.items()}
        rows.append(row)
    fitted = {
        "implementation": "stdlib_sparse_word_tfidf_v1",
        "vocabulary": vocabulary,
        "idf": idf,
        "parameters": {
            "analyzer": "word",
            "ngram_range": [1, 2],
            "token_pattern": token_pattern.pattern,
            "stop_words": None,
            "min_df": 1,
            "max_df": 1.0,
            "max_features": max_features,
            "binary": False,
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True,
            "norm": "l2",
            "feature_selection_ties": "term ascending",
        },
    }
    return rows, fitted


def _uniform_sampled_hit10(population, successes, draws=99):
    """Exact Hypergeom(population, successes, draws).cdf(9)."""
    if not (0 <= successes <= population and 0 <= draws <= population):
        raise ValueError("invalid hypergeometric population or sample size")
    lower = max(0, draws - (population - successes))
    upper = min(9, successes, draws)
    if lower > upper:
        return 0.0
    numerator = sum(
        math.comb(successes, count) * math.comb(population - successes, draws - count)
        for count in range(lower, upper + 1)
    )
    # Python integers preserve the combinatorial sum without overflow;
    # division produces the final bounded floating-point probability.
    return numerator / math.comb(population, draws)


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
    - Every statement must be grounded in the provided metric values/definitions.
    ### >>> Self_EvolveRec-BLOCK-START: Interpret training audits with explicit evidence boundaries
    - Report support, exclusions, selection configuration and missingness with findings.
    - Null means unavailable, never zero. No measurements means untested.
    - Sampled training ranks, pairwise margins and catalog ranks are distinct estimands; none is pinned validation HR@10.
    - Training targets may be memorized. Later training positives annotate candidate ambiguity; unobserved items are not verified negatives.
    - target_lexical_oracle uses the target identity and is not deployable retrieval or evidence of history understanding.
    - Prefix lexical competition can disadvantage interest shifts. Candidate sensitivity does not establish user intent or causal importance.
    - Lexical draws differ only in fallback items; deterministic retrieval repetitions are not independent experiments.
    - Raw margins depend on score scale. A mapped cosine near 0.5 means mean cosine near zero, not a collapse probability.
    - Bootstrap intervals condition on the fitted checkpoint, selected users, fixed candidates, catalog and metadata. They omit training-seed uncertainty and shared-item dependence.
    - Never infer benchmark performance, deployment parity, unavailable user traits, ratings, reviews or item categories.
    - Distinguish in-memory checks from generated-predictor reload checks; unavailable deployment parity remains unverified.
    - A four-category macro result requires support from all four categories; never silently omit a missing category.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family-specific ModelView contract and public-data boundary
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Categories are unavailable in this benchmark; never infer them.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and report undefined embedding measurements honestly
            emb = emb[1 : self.network.itemnum + 1]

            n = emb.size(0)
            if min(sample_size, n) < 2 or not torch.isfinite(emb).all():
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Compute mean off-diagonal cosine without a quadratic Gram matrix
            x = F.normalize(emb.index_select(0, idx).float(), p=2, dim=-1)
            offdiag_sum = x.sum(0).square().sum() - x.square().sum()
            ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Keep mode ownership in the diagnosis wrapper and track finite support
        margins_all = []
        valid_total = nonfinite_total = 0
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound loader consumption without fetching an extra training batch
        for batch in islice(iter(dataloader), max(0, num_batches)):
            ### <<< Self_EvolveRec-BLOCK-END
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Use tensor inputs for the corrected NextItNet compatibility path
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude missing targets and nonfinite pairs consistently
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            finite = torch.isfinite(all_logits).all(-1) & torch.isfinite(margin)
            valid_total += int(valid.sum().item())
            nonfinite_total += int((valid & ~finite).sum().item())
            valid &= finite
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
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
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Represent empty margin summaries as unavailable
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                ### <<< Self_EvolveRec-BLOCK-END
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

        ### >>> Self_EvolveRec-BLOCK-START: Expose margin denominators and unavailable category annotations
        margin_metrics.update(
            {
                "valid_positions": valid_total,
                "comparisons": valid_total - nonfinite_total,
                "excluded_nonfinite": nonfinite_total,
                "distinct_users": None,
            }
        )
        breakdown = {
            "availability": "unavailable: item categories are not provided",
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

    ### >>> Self_EvolveRec-BLOCK-START: Frozen training-only candidate audit with efficient weighted sampling and explicit support
    @staticmethod
    def _digest(value):
        payload = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _clean_text(value):
        if isinstance(value, (list, tuple)):
            value = " ".join(x for x in value if isinstance(x, str))
        if not isinstance(value, str):
            return ""
        value = re.sub(r"<[^>]*>", " ", html.unescape(value))
        return " ".join(unicodedata.normalize("NFKC", value).lower().split())

    @staticmethod
    def _write_json(path, value):
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(
                value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        os.replace(temporary, path)

    def _text_index(self, itemnum):
        # DEBUG: Fit both lexical fields without optional scientific libraries.
        metadata = self.meta_data if isinstance(self.meta_data, dict) else {}
        fields, vectorizers = {}, {}
        cleaned = {}
        combined_rows = [{} for _ in range(itemnum)]
        width = 0
        for field in ("title", "description"):
            texts = []
            for item in range(1, itemnum + 1):
                info = metadata.get(str(item), metadata.get(item, {}))
                texts.append(
                    self._clean_text(info.get(field, ""))
                    if isinstance(info, dict)
                    else ""
                )
            cleaned[field] = texts
            rows, fitted = _fit_tfidf(texts)
            vocabulary_size = len(fitted["vocabulary"])
            fields[field] = {
                "nonempty_text_items": sum(bool(text) for text in texts),
                "nonzero_vector_items": sum(bool(row) for row in rows),
                "vocabulary_size": vocabulary_size,
                "empty_vocabulary": vocabulary_size == 0,
            }
            for combined, row in zip(combined_rows, rows):
                combined.update({width + index: value for index, value in row.items()})
            width += vocabulary_size
            vectorizers[field] = fitted
        # Normalize the concatenation of independently normalized fields.
        for row in combined_rows:
            norm = math.sqrt(sum(value * value for value in row.values()))
            if norm:
                for index in row:
                    row[index] /= norm
        fields["item_zero_vector_rate"] = (
            sum(not row for row in combined_rows) / itemnum if itemnum else None
        )
        combined = _SparseTextIndex(combined_rows, width)
        return combined, vectorizers, fields, self._digest(cleaned)

    @staticmethod
    def _mean_user_summary(records, conditions):
        # Each record already averages the five frozen draws of one prefix.
        users = {}
        for record in records:
            users.setdefault(record["user_id"], []).append(record["values"])
        names = (
            "rank",
            "hit10",
            "reciprocal_rank",
            "max_competitor_margin",
            "known_later_positive_fraction",
            "fallback_fraction",
        )
        arrays = []
        for user in sorted(users):
            arrays.append(np.asarray(users[user], dtype=np.float64).mean(axis=0))
        values = (
            np.stack(arrays) if arrays else np.empty((0, len(conditions), len(names)))
        )
        means = values.mean(axis=0) if len(values) else None
        summary = {
            "users": len(users),
            "prefixes": len(records),
            "conditions": {
                condition: {
                    name: None if means is None else float(means[c, j])
                    for j, name in enumerate(names)
                }
                for c, condition in enumerate(conditions)
            },
        }
        return summary, values

    @torch.no_grad()
    def _candidate_audit(self):
        conditions = (
            "uniform",
            "frequency_075",
            "prefix_lexical",
            "target_lexical_oracle",
        )
        category_order = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        category = getattr(self.model, "dataset_name", None)
        if category is None:
            category = getattr(self.network, "dataset_name", None)
        # ModelView does not require a category label. Never guess it from items.
        category_index = (
            category_order.index(category) if category in category_order else 0
        )
        itemnum = int(self.network.itemnum)
        maxlen = int(self.network.maxlen)
        if itemnum < 1 or maxlen < 1:
            raise ValueError("audit requires positive itemnum and maxlen")
        raw = self.user_train
        if not isinstance(raw, dict):
            raise ValueError("user_train must contain mapped training histories")
        histories = raw.get("History", raw)
        times = raw.get("Time", {})
        if not isinstance(histories, dict) or not isinstance(times, dict):
            raise ValueError("training histories and timestamps must be dictionaries")
        canonical = {}
        for user, history in histories.items():
            key = str(user)
            if key in canonical:
                raise ValueError("training user IDs collide after string conversion")
            canonical[key] = [int(i) for i in history]
        counts = np.zeros(itemnum, dtype=np.int64)
        exclusions = Counter()
        eligible = []
        for user in sorted(canonical):
            history = canonical[user]
            if len(history) < 2:
                exclusions["users_short_history"] += 1
                continue
            if any(i < 1 or i > itemnum for i in history):
                exclusions["users_invalid_mapped_items"] += 1
                continue
            eligible.append(user)
        # Count all valid public training events, including single-event users.
        for history in canonical.values():
            for item in history:
                if 1 <= item <= itemnum:
                    counts[item - 1] += 1
        rng = np.random.default_rng(42)
        chosen = sorted(
            rng.choice(len(eligible), min(200, len(eligible)), replace=False).tolist()
        )
        catalog = np.arange(1, itemnum + 1, dtype=np.int64)
        frequency_order = np.lexsort((catalog, counts))
        bands = np.empty(itemnum, dtype=np.int64)
        band_definitions = {}
        for band, indices in enumerate(np.array_split(frequency_order, 3)):
            bands[indices] = band
            band_definitions[("tail", "middle", "head")[band]] = {
                "items": int(indices.size),
                "first_count_id": (
                    None
                    if not indices.size
                    else [int(counts[indices[0]]), int(indices[0] + 1)]
                ),
                "last_count_id": (
                    None
                    if not indices.size
                    else [int(counts[indices[-1]]), int(indices[-1] + 1)]
                ),
            }
        manifest = []
        for sorted_user_index in chosen:
            user = eligible[sorted_user_index]
            history = canonical[user]
            positions = sorted(
                rng.choice(
                    np.arange(1, len(history)), min(5, len(history) - 1), replace=False
                ).tolist()
            )
            stamp = times.get(user, None)
            if stamp is None and user.isdecimal():
                stamp = times.get(int(user), None)
            for position in positions:
                visible = history[max(0, position - maxlen) : position]
                target = history[position]
                forbidden = set(visible) | {target}
                pool = catalog[~np.isin(catalog, list(forbidden))]
                record = {
                    "user_id": user,
                    "sorted_user_index": sorted_user_index,
                    "target_position": position,
                    "prefix": visible,
                    "target": target,
                    "target_timestamp": (
                        str(stamp[position])
                        if isinstance(stamp, (list, tuple)) and position < len(stamp)
                        else None
                    ),
                    "eligible_competitors": int(pool.size),
                    "prefix_stratum": (
                        "1-4"
                        if len(visible) < 5
                        else (
                            "5-9"
                            if len(visible) < 10
                            else "10-19" if len(visible) < 20 else "20+"
                        )
                    ),
                    "frequency_band": ("tail", "middle", "head")[
                        int(bands[target - 1])
                    ],
                }
                record["eligible"] = bool(pool.size >= 99 and visible)
                if not record["eligible"]:
                    exclusions["prefixes_fewer_than_99_competitors_or_empty"] += 1
                manifest.append(record)
        selected_users = sorted({r["user_id"] for r in manifest})
        subset_rng = np.random.default_rng(4242)
        subset_users = set(
            subset_rng.choice(
                selected_users, min(40, len(selected_users)), replace=False
            ).tolist()
        )
        earliest = {}
        for index, record in enumerate(manifest):
            earliest.setdefault(record["user_id"], index)
        subset = {earliest[user] for user in subset_users}
        text, vectorizers, coverage, metadata_hash = self._text_index(itemnum)
        config = {
            "version": "training_candidate_audit_v2",
            "category": category,
            "category_order": category_order,
            "category_index": category_index,
            "category_identity_available": category in category_order,
            "unidentified_category_seed_policy": "index 0; category macro unavailable",
            "conditions": list(conditions),
            "draws": 5,
            "negatives": 99,
            "selection_seed": 42,
            "catalog_subset_seed": 4242,
            "max_users": 200,
            "max_prefixes_per_user": 5,
            "max_catalog_users": 40,
            "maxlen": maxlen,
            "chunk_size": 4096,
            "seed_scheme": "SeedSequence([42, category_index, sorted_user_index, target_position, condition_index, draw_index])",
            "tie_order": "score descending, mapped item ID ascending; host equivalence unverified",
            "weighted_algorithm": "exponential race with rates count**0.75; increasing arrival order is successive weighted sampling without replacement; uniform fill only after positive rates are exhausted",
            "lexical_query": "normalized mean of visible occurrence vectors; target query is oracle-conditioned",
            "lexical_fill": "positive-cosine top 99, ID tie break; uniform remaining eligible IDs",
            "cleaning": "HTML unescape; remove tags; Unicode NFKC; lowercase; collapse whitespace; retain accents",
            "numpy": np.__version__,
            "torch": torch.__version__,
            # DEBUG: Record the actual replacement implementation in audit hashes.
            "lexical_backend": "stdlib_sparse_word_tfidf_v1",
            "lexical_feature_selection_ties": "corpus frequency descending, term ascending",
            "hypergeometric_backend": "stdlib_math_comb_exact_integer_sum_v1",
            "mapping": "ModelView metadata keys and histories must already use mapped item IDs 1..itemnum",
            "mapping_source_artifact": "unavailable through ModelView",
            "checkpoint_file": "unavailable through ModelView; actual module state hashed instead",
            "metadata_hash": metadata_hash,
            "history_hash": self._digest(canonical),
        }
        checkpoint_hash = hashlib.sha256()
        for name, tensor in sorted(self.network.state_dict().items()):
            tensor = tensor.detach().cpu().contiguous()
            checkpoint_hash.update(name.encode())
            checkpoint_hash.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            checkpoint_hash.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        config["model_state_sha256"] = checkpoint_hash.hexdigest()
        frozen = []
        for record in manifest:
            if not record["eligible"]:
                frozen.append(None)
                continue
            prefix, target = record["prefix"], record["target"]
            pool = catalog[~np.isin(catalog, list(set(prefix) | {target}))]
            lexical = []
            for oracle in (False, True):
                # DEBUG: Execute both lexical conditions through the sparse replacement.
                similarities, zero_query = text.similarities(
                    [target] if oracle else prefix
                )
                positive = pool[similarities[pool - 1] > 0]
                order = np.lexsort((positive, -similarities[positive - 1]))
                lexical.append(
                    (
                        positive[order][:99],
                        similarities,
                        zero_query,
                        int(positive.size),
                    )
                )
            draws = []
            for condition_index, condition in enumerate(conditions):
                condition_draws = []
                for draw_index in range(5):
                    local = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                42,
                                category_index,
                                record["sorted_user_index"],
                                record["target_position"],
                                condition_index,
                                draw_index,
                            ]
                        )
                    )
                    fallback = 0
                    if condition_index == 0:
                        ids = local.choice(pool, 99, replace=False)
                    elif condition_index == 1:
                        # Independent exponential clocks implement the same
                        # successive weighted-without-replacement distribution.
                        # This avoids 99 catalog copies/scans for every draw.
                        weights = counts[pool - 1].astype(np.float64) ** 0.75
                        supported = np.flatnonzero(weights > 0)
                        take = min(99, supported.size)
                        if take:
                            clocks = local.standard_exponential(supported.size)
                            clocks /= weights[supported]
                            if supported.size > take:
                                chosen_clocks = np.argpartition(clocks, take - 1)[:take]
                                # Include all boundary ties before applying the
                                # deterministic mapped-ID tie order.
                                boundary = clocks[chosen_clocks].max()
                                chosen_clocks = np.flatnonzero(clocks <= boundary)
                            else:
                                chosen_clocks = np.arange(supported.size)
                            order = np.lexsort(
                                (pool[supported[chosen_clocks]], clocks[chosen_clocks])
                            )[:take]
                            ids = pool[supported[chosen_clocks[order]]]
                        else:
                            ids = np.empty(0, dtype=np.int64)
                        fallback = 99 - take
                        if fallback:
                            zero_weight_pool = pool[weights == 0]
                            ids = np.concatenate(
                                (
                                    ids,
                                    local.choice(
                                        zero_weight_pool, fallback, replace=False
                                    ),
                                )
                            )
                    else:
                        top, similarities, zero_query, overlap = lexical[
                            condition_index - 2
                        ]
                        fallback = 99 - len(top)
                        rest = pool[~np.isin(pool, top)]
                        ids = np.concatenate(
                            (top, local.choice(rest, fallback, replace=False))
                        )
                    entry = {"negative_ids": ids.tolist(), "fallback_count": fallback}
                    if condition_index >= 2:
                        entry.update(
                            {
                                "similarities": similarities[ids - 1].tolist(),
                                "zero_query": bool(zero_query),
                                "positive_overlap_count": overlap,
                            }
                        )
                    if (
                        ids.size != 99
                        or np.unique(ids).size != 99
                        or not np.isin(ids, pool).all()
                    ):
                        raise RuntimeError(
                            f"invalid frozen candidate draw for {condition}"
                        )
                    condition_draws.append(entry)
                draws.append(condition_draws)
            frozen.append(draws)
        config["manifest_sha256"] = self._digest(manifest)
        config["candidates_sha256"] = self._digest(frozen)
        config["counts_sha256"] = self._digest(counts.tolist())
        root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "working", "diagnosis_audit"
        )
        audit_id = self._digest(config)
        directory = os.path.join(root, audit_id)
        os.makedirs(directory, exist_ok=True)
        # Freeze all selections and candidates before the first audit score.
        self._write_json(
            os.path.join(directory, "frozen.json"),
            {
                "configuration": config,
                "manifest": manifest,
                "candidates": frozen,
                "catalog_subset_indices": sorted(subset),
                "training_counts": counts.tolist(),
                "frequency_weights": (counts.astype(float) ** 0.75).tolist(),
                "frequency_bands": band_definitions,
                "metadata_coverage": coverage,
            },
        )
        with open(os.path.join(directory, "tfidf.pkl"), "wb") as stream:
            # Includes exact parameters, fitted vocabularies and IDFs per field.
            pickle.dump(vectorizers, stream, protocol=pickle.HIGHEST_PROTOCOL)
        records, calibration = [], []
        for index, record in enumerate(manifest):
            if not record["eligible"]:
                continue
            sequence = torch.zeros((1, maxlen), device=self.device, dtype=torch.long)
            sequence[0, -len(record["prefix"]) :] = torch.tensor(
                record["prefix"], device=self.device
            )
            slots = torch.where(sequence[0].ne(0))[0]
            if not slots.numel():
                exclusions["all_padding_prefixes"] += 1
                continue
            features = self.model.log2feats(sequence)[0, slots[-1]]
            ids = np.asarray(
                [
                    [record["target"]] + entry["negative_ids"]
                    for draws in frozen[index]
                    for entry in draws
                ],
                dtype=np.int64,
            )
            candidate_tensor = torch.as_tensor(
                ids.ravel(), device=self.device, dtype=torch.long
            )
            scores = (
                self.model.predict_score(features, candidate_tensor)
                .float()
                .cpu()
                .numpy()
                .reshape(4, 5, 100)
            )
            margins = scores[:, :, 0] - scores[:, :, 1:].max(axis=-1)
            if not np.isfinite(scores).all() or not np.isfinite(margins).all():
                exclusions["nonfinite_prefixes_common_to_all_conditions"] += 1
                continue
            ids = ids.reshape(4, 5, 100)
            target = record["target"]
            ranks = 1 + (
                (scores[:, :, 1:] > scores[:, :, :1])
                | ((scores[:, :, 1:] == scores[:, :, :1]) & (ids[:, :, 1:] < target))
            ).sum(-1)
            later = set(canonical[record["user_id"]][record["target_position"] + 1 :])
            conflict = np.isin(ids[:, :, 1:], list(later)).mean(axis=-1)
            fallback = np.asarray(
                [
                    [entry["fallback_count"] / 99 for entry in draws]
                    for draws in frozen[index]
                ]
            )
            values = np.stack(
                (ranks, ranks <= 10, 1.0 / ranks, margins, conflict, fallback), axis=-1
            )
            output = {
                **record,
                "manifest_index": index,
                "values": values.mean(axis=1).tolist(),
                "draw_ranks": ranks.tolist(),
                "draw_margins": margins.tolist(),
                "lexical_queries": {
                    conditions[c]: {
                        "zero_query": frozen[index][c][0]["zero_query"],
                        "positive_overlap_count": frozen[index][c][0][
                            "positive_overlap_count"
                        ],
                    }
                    for c in (2, 3)
                },
            }
            records.append(output)
            if index not in subset:
                continue
            universe = catalog[
                ~np.isin(catalog, list(set(record["prefix"]) - {target}))
            ]
            all_scores = []
            for start in range(0, len(universe), 4096):
                chunk = torch.as_tensor(
                    universe[start : start + 4096], device=self.device, dtype=torch.long
                )
                all_scores.append(
                    self.model.predict_score(features, chunk).float().cpu().numpy()
                )
            all_scores = np.concatenate(all_scores)
            if not np.isfinite(all_scores).all():
                exclusions["nonfinite_catalog_examples"] += 1
                continue
            target_score = all_scores[np.searchsorted(universe, target)]
            better = (all_scores > target_score) | (
                (all_scores == target_score) & (universe < target)
            )
            exact_rank = 1 + int(better.sum())
            # DEBUG: Preserve analytic uniform-sampling calibration without SciPy.
            probability = _uniform_sampled_hit10(len(universe) - 1, exact_rank - 1, 99)
            # Reuse exact-catalog scores to isolate sampling calibration from GEMM shape rounding.
            uniform_positions = np.searchsorted(universe, ids[0, :, 1:])
            uniform_ranks = 1 + better[uniform_positions].sum(axis=-1)
            empirical = float(np.mean(uniform_ranks <= 10))
            calibration.append(
                {
                    "user_id": record["user_id"],
                    "manifest_index": index,
                    "prefix_stratum": record["prefix_stratum"],
                    "frequency_band": record["frequency_band"],
                    "catalog_size": len(universe),
                    "exact_rank": exact_rank,
                    "catalog_hit10": bool(exact_rank <= 10),
                    "analytic_uniform_hit10": probability,
                    "empirical_uniform_hit10": empirical,
                    "calibration_residual": empirical - probability,
                    "sampled_catalog_hit_disagreement_fraction": float(
                        np.mean((uniform_ranks <= 10) != (exact_rank <= 10))
                    ),
                    "primary_vs_catalog_score_rank_disagreements": int(
                        np.sum(uniform_ranks != ranks[0])
                    ),
                }
            )
        summary, users = self._mean_user_summary(records, conditions)

        def lexical_support(selected):
            # One observation per prefix, not five copies of deterministic retrieval.
            return {
                condition: {
                    "prefixes": len(selected),
                    "zero_query_rate": (
                        float(
                            np.mean(
                                [
                                    r["lexical_queries"][condition]["zero_query"]
                                    for r in selected
                                ]
                            )
                        )
                        if selected
                        else None
                    ),
                    "positive_overlap_count_quantiles": (
                        np.quantile(
                            [
                                r["lexical_queries"][condition][
                                    "positive_overlap_count"
                                ]
                                for r in selected
                            ],
                            [0.1, 0.5, 0.9],
                        ).tolist()
                        if selected
                        else None
                    ),
                }
                for condition in conditions[2:]
            }

        summary["lexical_query_support"] = lexical_support(records)
        strata = {}
        for field, labels in (
            ("prefix_stratum", ("1-4", "5-9", "10-19", "20+")),
            ("frequency_band", ("tail", "middle", "head")),
        ):
            strata[field] = {}
            for label in labels:
                selected = [r for r in records if r[field] == label]
                entry, _ = self._mean_user_summary(selected, conditions)
                entry["selected_prefixes_before_exclusions"] = sum(
                    r[field] == label for r in manifest
                )
                entry["excluded_prefixes"] = entry[
                    "selected_prefixes_before_exclusions"
                ] - len(selected)
                entry["rank_quantiles_by_condition"] = {
                    condition: (
                        None
                        if not selected
                        else np.quantile(
                            [rank for r in selected for rank in r["draw_ranks"][c]],
                            [0.1, 0.5, 0.9],
                        ).tolist()
                    )
                    for c, condition in enumerate(conditions)
                }
                exact = [r for r in calibration if r[field] == label]
                entry["catalog_examples"] = len(exact)
                entry["sampled_catalog_hit_disagreement_fraction"] = (
                    float(
                        np.mean(
                            [
                                r["sampled_catalog_hit_disagreement_fraction"]
                                for r in exact
                            ]
                        )
                    )
                    if exact
                    else None
                )
                entry["lexical_query_support"] = lexical_support(selected)
                strata[field][label] = entry
        intervals = {}
        bootstrap_rng = np.random.default_rng(42)
        if len(users):
            estimates = []
            for _ in range(1000):
                indices = bootstrap_rng.integers(len(users), size=len(users))
                means = users[indices].mean(axis=0)
                estimates.append(means[1:] - means[:1])
            estimates = np.asarray(estimates)
            for c, condition in enumerate(conditions[1:]):
                intervals[condition + "_minus_uniform"] = {
                    name: {
                        "estimate": float((users[:, c + 1, j] - users[:, 0, j]).mean()),
                        "percentile_95_interval": np.quantile(
                            estimates[:, c, j], [0.025, 0.975]
                        ).tolist(),
                    }
                    for j, name in enumerate(
                        ("rank", "hit10", "reciprocal_rank", "max_competitor_margin")
                    )
                }
        result = {
            "available": bool(records),
            "unavailable_reason": (
                None if records else "no finite eligible training prefixes"
            ),
            "configuration": config,
            "summary": summary,
            "paired_user_bootstrap": {
                "replicates": 1000,
                "seed": 42,
                "differences": intervals,
            },
            "strata": strata,
            "metadata_coverage": coverage,
            "frequency_band_boundaries": band_definitions,
            "support": {
                "eligible_training_users": len(eligible),
                "selected_users": len(selected_users),
                "selected_prefixes": len(manifest),
                "scored_prefixes": len(records),
                "catalog_prespecified_examples": len(subset),
                "catalog_scored_examples": len(calibration),
                "exclusions": dict(exclusions),
            },
            "calibration": {
                "examples": len(calibration),
                "mean_residual": (
                    float(np.mean([r["calibration_residual"] for r in calibration]))
                    if calibration
                    else None
                ),
            },
            "artifact_directory": directory,
            "four_category_macro": {
                "available": False,
                "reason": "one category ModelView per invocation; requires all four user-level audit outputs",
            },
        }
        self._write_json(
            os.path.join(directory, "results.json"),
            {
                "summary": result,
                "prefix_results": records,
                "catalog_results": calibration,
            },
        )
        return result

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            self.network.eval()
            with torch.random.fork_rng(devices=devices):
                # Loader implementations may use global RNGs; restore those too.
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(42)
                batches = []
                for batch in islice(iter(train_loader), 4):
                    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                        raise ValueError(
                            "expected (None, sequence, positive, negative)"
                        )
                    tensors = [
                        torch.as_tensor(x, device=self.device, dtype=torch.long)[
                            :32
                        ].clone()
                        for x in batch[1:]
                    ]
                    if any(x.ndim != 2 or x.shape != tensors[0].shape for x in tensors):
                        raise ValueError(
                            "diagnostic tensors must have matching B x L shapes"
                        )
                    if tensors[0].size(0) and tensors[0].size(1):
                        batches.append((None, *tensors))
                collapse = self.analyze_embeddings_collapse_cosine()
                margins, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=4, eps=0.5
                    )
                )
                audit = self._candidate_audit()
                embedding = self.model.item_emb.weight.detach()[
                    1 : self.network.itemnum + 1
                ]
                metrics = {
                    "embedding_collapse_score": collapse,
                    "embedding_support": {
                        "catalog_items": embedding.size(0),
                        "sampled_items": (
                            min(2048, embedding.size(0)) if collapse is not None else 0
                        ),
                        "nonfinite_items": int(
                            (~torch.isfinite(embedding).all(-1)).sum().item()
                        ),
                    },
                    "pos_neg_margin_mean": margins["mean"],
                    "pos_neg_margin_p10": margins["p10"],
                    "pos_neg_neg_beats_pos_rate": margins["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margins["small_margin_rate"],
                    "all_position_support": {
                        key: margins[key]
                        for key in (
                            "valid_positions",
                            "comparisons",
                            "excluded_nonfinite",
                            "distinct_users",
                        )
                    },
                    "pos_neg_margin_category_breakdown": breakdown,
                    "training_candidate_stress_audit": audit,
                    "deployment_reload_parity": {
                        "available": False,
                        "reason": "ModelView supplies no generated-predictor reload interface",
                    },
                    "validation_rank_metrics": {
                        "available": False,
                        "reason": "training-only diagnostic; pinned validation candidates unavailable",
                    },
                }
                definitions = {
                    "embedding_collapse_score": "(mean sampled off-diagonal item cosine + 1)/2; 0.5 means mean cosine zero. Not collapse probability or accuracy. Null when undefined/nonfinite.",
                    "embedding_support": "Mapped catalog items excluding padding, sampled items and nonfinite catalog embeddings.",
                    "pos_neg_margin_mean": "Mean positive-minus-negative score over finite nonpadding positions with both targets nonzero; scale dependent.",
                    "pos_neg_margin_p10": "Nearest-rank tenth percentile on the same supported pairs.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of supported margins strictly below zero; not HR@10.",
                    "pos_neg_small_margin_rate": "Fraction of supported margins below 0.5; scale dependent.",
                    "all_position_support": "Position comparisons and finite exclusions from at most four batches and 32 rows/batch; stable user identities unavailable for these loader rows.",
                    "pos_neg_margin_category_breakdown": "Category annotations unavailable; measured low/wrong position totals retained, category lists empty.",
                    "training_candidate_stress_audit": (
                        "Frozen fitted-training-target audit, not a holdout or benchmark evaluation. "
                        "Primary pools exclude model-visible prefix items and the target, never later events; target is reinserted once. "
                        "All four conditions require 99 unique competitors and share finite-prefix exclusions. "
                        "Five draws are averaged within prefix, prefixes within identified user, users within this category. "
                        "Rank uses exact score equality and ascending mapped ID tie breaks; hit10 is rank<=10 and reciprocal rank is 1/rank. "
                        "Margin is target score minus maximum competitor score. Known-positive conflict fractions use only later training events. "
                        "Uniform draws are without replacement. Frequency draws use sequential remaining-count**0.75 probabilities, not marginal inclusion probabilities. "
                        "Prefix lexical queries average visible occurrences; target lexical queries are explicitly oracle-conditioned. "
                        "TF-IDF is fitted only on public mapped titles/descriptions; fitted vectorizers, coverage, candidate similarities and fallback counts are trial-local artifacts. "
                        "Catalog selection prespecifies each selected user's earliest manifest prefix; excluded prefixes are counted and never replaced. "
                        "Lexical query support counts prefixes once; query rates and overlap quantiles are descriptive prefix-weighted summaries. "
                        "For catalog size N and total-order target rank r, expected uniform sampled hit10 is Hypergeom(N-1,r-1,99).cdf(9). "
                        "Empirical calibration reuses exact-catalog scores; primary-versus-catalog numerical rank discrepancies are recorded separately. "
                        "Rank quantiles pool draws descriptively; they are not independent-user uncertainty estimates. "
                        "Paired percentile intervals resample identified users 1000 times, retaining their prefixes/draws; conditional on fitted model and frozen artifacts. "
                        "Empty groups return null and zero support. Four-category macro aggregation is unavailable in a single-category view. "
                        "No training intervention, score fusion, extra fitted predictor or evaluator change is performed."
                    ),
                    "deployment_reload_parity": "Unavailable without a host reload interface; no deployment correctness is inferred.",
                    "validation_rank_metrics": "Unavailable; no training diagnostic is presented as pinned validation HR@10.",
                }
                return {"metrics": metrics, "metric_definitions": definitions}
        finally:
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    ### <<< Self_EvolveRec-BLOCK-END
