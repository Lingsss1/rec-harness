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


### >>> Self_EvolveRec-BLOCK-START: Causal competitive curriculum with batched score transfer and explicit finite checks
class PrefixCandidateObjective:
    def __init__(self, user_train, itemnum, model):
        self.itemnum = itemnum
        self.negative_rng = np.random.default_rng(SEED)
        self.auxiliary_rng = np.random.default_rng(SEED)
        self.negative_count = 99
        self.pool_size = 256
        self.warmup_epochs = 5
        self.ramp_epochs = 15
        self.maximum_competitive_fraction = 0.20
        self.pool_rank_first = 9
        self.pool_rank_last = 64
        self.row_chunk_size = 512
        self.competitive_fraction = 0.0
        # Only training interactions define the mining/uniform catalog.
        self.catalog = np.asarray(
            sorted(
                {
                    int(item)
                    for history in user_train.values()
                    for item in history
                    if 1 <= item <= itemnum
                }
            ),
            dtype=np.int64,
        )
        if self.catalog.size < 2:
            raise RuntimeError("candidate CE requires at least two training items")
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
        # Preserve sample_batch eligibility because its diagnostic negative still
        # excludes full histories. Those negatives are not used by this objective.
        self.eligible_users = [
            user
            for user, history in user_train.items()
            if len(history) >= 2 and len(set(history)) < itemnum
        ]
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "valid_positions": 0,
            "skipped_no_negatives": 0,
            "usable_positions": 0,
            "longer_than_suffix": 0,
            "auxiliary_positions": 0,
            "optimizer_steps": 0,
            "negative_slots": 0,
            "competitive_requested": 0,
            "competitive_selected": 0,
            "competitive_fallback_rows": 0,
            "competitive_pool_rank_sum": 0,
            "small_catalog_rows": 0,
            "nonfinite_pool_scores": 0,
        }

    def sampling_config(self):
        return {
            "negative_count": self.negative_count,
            "pool_size": self.pool_size,
            "warmup_epochs": self.warmup_epochs,
            "ramp_epochs": self.ramp_epochs,
            "maximum_competitive_fraction": self.maximum_competitive_fraction,
            "eligible_pool_rank_first": self.pool_rank_first,
            "eligible_pool_rank_last": self.pool_rank_last,
            "row_chunk_size": self.row_chunk_size,
            "catalog_size": int(self.catalog.size),
            "exclusions": "current target and visible truncated causal prefix",
            "tie_break": "mapped item ID ascending",
            "rng": "numpy.default_rng(42), separate from auxiliary RNG",
            "proposal_correction": "none",
        }

    def _candidates(self, model, features, sequence, positive, epoch):
        # epoch is zero-based: epochs 0..4 are uniform, 5..19 ramp up.
        progress = max(
            0.0, min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)
        )
        self.competitive_fraction = self.maximum_competitive_fraction * progress
        batch, length = sequence.shape
        candidates = np.zeros((batch, length, 1 + self.negative_count), dtype=np.int64)
        candidate_mask = np.zeros_like(candidates, dtype=np.bool_)
        usable = (sequence != 0) & (positive != 0)
        positions = np.flatnonzero(usable.reshape(-1))
        self.stats["valid_positions"] += int(positions.size)
        flat_features = features.reshape(-1, model.hidden_units)
        pool = None
        pool_t = None
        if self.competitive_fraction > 0:
            # Sorted IDs make stable descending-score sorting break ties by ID.
            pool = np.sort(
                self.negative_rng.choice(
                    self.catalog,
                    size=min(self.pool_size, self.catalog.size),
                    replace=False,
                )
            )
            pool_t = torch.as_tensor(pool, dtype=torch.long, device=features.device)

        # Keep scoring calls chunked, then transfer the small N x pool_size
        # matrix once instead of synchronizing CUDA after every row chunk.
        # No B x L x catalog tensor is constructed.
        all_pool_scores = None
        if pool is not None and positions.size:
            with torch.no_grad():
                position_t = torch.as_tensor(
                    positions, dtype=torch.long, device=features.device
                )
                score_chunks = []
                for start in range(0, positions.size, self.row_chunk_size):
                    selected = position_t[start : start + self.row_chunk_size]
                    score_chunks.append(
                        model.predict_score(
                            flat_features.detach()[selected], pool_t
                        ).float()
                    )
                all_pool_scores = torch.cat(score_chunks, dim=0).cpu().numpy()
                del score_chunks

        for start in range(0, positions.size, self.row_chunk_size):
            chunk = positions[start : start + self.row_chunk_size]
            pool_scores = (
                None
                if all_pool_scores is None
                else all_pool_scores[start : start + chunk.size]
            )
            for local, flat_position in enumerate(chunk):
                row, end = divmod(int(flat_position), length)
                target = int(positive[row, end])
                # Never inspect later sequence positions or complete user histories.
                forbidden_ids = np.unique(np.append(sequence[row, : end + 1], target))
                forbidden_ids = forbidden_ids[forbidden_ids != 0]
                forbidden = np.searchsorted(self.catalog, forbidden_ids)
                if np.any(forbidden >= self.catalog.size) or not np.array_equal(
                    self.catalog[forbidden], forbidden_ids
                ):
                    raise ValueError(
                        "training target/prefix item outside training catalog"
                    )
                available = int(self.catalog.size - forbidden.size)
                if available == 0:
                    usable[row, end] = False
                    self.stats["skipped_no_negatives"] += 1
                    continue
                count = min(self.negative_count, available)
                requested = int(math.floor(self.competitive_fraction * count))
                competitive = np.empty(0, dtype=np.int64)
                if requested:
                    scores = pool_scores[local]
                    allowed = ~np.isin(pool, forbidden_ids)
                    finite = np.isfinite(scores)
                    nonfinite_count = int((allowed & ~finite).sum())
                    self.stats["nonfinite_pool_scores"] += nonfinite_count
                    if nonfinite_count:
                        raise FloatingPointError(
                            "nonfinite eligible competitive-pool scores: "
                            f"epoch={epoch}, row={row}, position={end}, "
                            f"count={nonfinite_count}"
                        )
                    eligible_columns = np.flatnonzero(allowed)
                    ranked = eligible_columns[
                        np.argsort(-scores[eligible_columns], kind="stable")
                    ]
                    band = ranked[self.pool_rank_first - 1 : self.pool_rank_last]
                    selected = self.negative_rng.choice(
                        band.size, size=min(requested, band.size), replace=False
                    )
                    competitive = pool[band[selected]]
                    self.stats["competitive_pool_rank_sum"] += int(
                        (selected + self.pool_rank_first).sum()
                    )
                self.stats["competitive_requested"] += requested
                self.stats["competitive_selected"] += int(competitive.size)
                self.stats["competitive_fallback_rows"] += int(
                    competitive.size < requested
                )
                self.stats["small_catalog_rows"] += int(count < self.negative_count)
                self.stats["negative_slots"] += count

                # Sample uniformly without replacement from the complement of
                # prefix, target and competitive IDs, without a catalog-sized mask.
                blocked = np.union1d(
                    forbidden, np.searchsorted(self.catalog, competitive)
                )
                shifted = blocked - np.arange(blocked.size, dtype=np.int64)
                remaining = int(self.catalog.size - blocked.size)
                uniform_count = count - competitive.size
                ranks = self.negative_rng.choice(
                    remaining, size=uniform_count, replace=False
                )
                indices = ranks + np.searchsorted(shifted, ranks, side="right")
                negatives = np.concatenate((competitive, self.catalog[indices]))
                candidates[row, end, 0] = target
                candidates[row, end, 1 : count + 1] = negatives
                candidate_mask[row, end, : count + 1] = True
        self.stats["usable_positions"] += int(usable.sum())
        prefix_lengths = (sequence != 0).cumsum(axis=1)
        self.stats["longer_than_suffix"] += int(
            (usable & (prefix_lengths > self.suffix_length)).sum()
        )
        return candidates, candidate_mask, np.flatnonzero(usable.reshape(-1))

    def __call__(self, model, sequence, positive, users, device, epoch):
        from torch.utils.checkpoint import checkpoint

        seq_t = torch.as_tensor(sequence, dtype=torch.long, device=device)
        # Mine from this exact training-mode forward, with selection detached.
        all_features = model.log2feats(seq_t)
        candidates, candidate_mask, positions = self._candidates(
            model, all_features, sequence, positive, epoch
        )
        if positions.size == 0:
            return None
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
        features = all_features.reshape(-1, model.hidden_units)[position_t]
        labels = torch.zeros(positions.size, dtype=torch.long, device=device)

        def chunk_loss(h, ids, mask):
            logits = model.predict_score(h, ids).masked_fill(~mask, -torch.inf)
            targets = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            return nn.functional.cross_entropy(logits, targets, reduction="sum")

        # Recompute only candidate scoring during backward. This bounds retained
        # N x 100 x D embedding activations while preserving the full-position mean.
        # The unchanged dot-product scoring head contains no stochastic layers,
        # so checkpointing it requires no RNG-state preservation.
        main_loss = features.new_zeros(())
        for start in range(0, positions.size, self.row_chunk_size):
            stop = start + self.row_chunk_size
            main_loss = main_loss + checkpoint(
                chunk_loss,
                features[start:stop],
                candidate_t[start:stop],
                mask_t[start:stop],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        main_loss = main_loss / positions.size

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
        f" sampling={json.dumps(objective.sampling_config(), sort_keys=True)}"
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
        ### >>> Self_EvolveRec-BLOCK-START: Optimize curriculum CE and reject nonfinite gradients before updates
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
            loss = objective(model, seq, pos, users, device, epoch=epoch - 1)
            if loss is None:
                continue
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite training loss in {dataset_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
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
            f" requested_competitive_fraction={objective.competitive_fraction:.6f}"
            f" effective_competitive_fraction={objective.stats['competitive_selected'] / max(objective.stats['negative_slots'], 1):.6f}"
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
                ### >>> Self_EvolveRec-BLOCK-START: Serialize the unchanged backbone and competitive sampling protocol
                "dropout_rate": DROPOUT_RATE,
                "architecture": "padding_safe_nextitnet_final_state_v1",
                "training_objective": "uncorrected_candidate_ce",
                "negative_count": objective.negative_count,
                "negative_sampling": objective.sampling_config(),
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

# DEBUG: Embed the trained architecture so deployment does not depend on importing
# a parent train.py file or executing its environment-dependent initialization.
import math
import os
import pickle

import torch
import torch.nn as nn

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


# DEBUG: The writer replaces this marker with the exact training model classes.
# __NEXTITNET_MODEL_SOURCE__


def _load_model_class():
    return NextItNetModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEBUG: No traceback was supplied; the reported failure cannot be localized.
    # Populate caches atomically so a failed category load cannot leave a partial
    # cache that silently bypasses loading on the next prediction call.
    ModelClass = _load_model_class()
    loaded_models = {}
    loaded_metas = {}
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
        # DEBUG: Publish category models only after every required load succeeds.
        loaded_models[ds] = model
        loaded_metas[ds] = meta
    _metas.update(loaded_metas)
    _models.update(loaded_models)


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
    # DEBUG: No traceback was supplied, so the original failure is unlocalized.
    # Remove the concrete export dependency on a source file named train.py;
    # preserve both classes verbatim, including padding and scoring behavior.
    import inspect

    marker = "# __NEXTITNET_MODEL_SOURCE__"
    if script.count(marker) != 1:
        raise RuntimeError("predictor architecture marker must occur exactly once")
    model_source = "\n\n".join(
        inspect.getsource(cls) for cls in (NextItNetBlock, NextItNetModel)
    )
    script = script.replace(marker, model_source)
    # Check generated Python syntax before publishing the predictor.
    predict_path = os.path.join(working_dir, "predict.py")
    compile(script, predict_path, "exec")
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
    # DEBUG: The supplied error contains no exception or traceback, so its cause
    # cannot be determined from the report. Emit the full traceback to stdout
    # for hosts that capture training output separately from stderr.
    # Preserve failure propagation and execute the complete training workflow.
    import sys
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise


