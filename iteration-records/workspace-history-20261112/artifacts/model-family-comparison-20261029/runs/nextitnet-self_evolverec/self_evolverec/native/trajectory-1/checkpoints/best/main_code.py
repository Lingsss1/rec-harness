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

### >>> Self_EvolveRec-BLOCK-START: Preserve required data source and training seed
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


### >>> Self_EvolveRec-BLOCK-START: Sample unique prefix-filtered negatives without oversampling long histories
def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    *,
    num_negatives: int = 1,
):
    if min(itemnum, batch_size, maxlen, num_negatives) < 1:
        raise ValueError(
            "itemnum, batch_size, maxlen and num_negatives must be positive"
        )
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    catalog = range(1, itemnum + 1)
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros((maxlen, num_negatives), dtype=np.int64)
        start = max(0, len(history) - 1 - maxlen)
        offset = maxlen - (len(history) - 1 - start)
        # Retain observations preceding the truncated input window, but never
        # exclude later interactions or held-out labels.
        prefix = {value for value in history[:start] if 1 <= value <= itemnum}
        for index in range(start, len(history) - 1):
            position = offset + index - start
            item, target = history[index], history[index + 1]
            if not (1 <= item <= itemnum and 1 <= target <= itemnum):
                raise ValueError("Training interaction outside the item vocabulary")
            seq[position] = item
            pos[position] = target
            prefix.add(item)
            available = itemnum - len(prefix) - int(target not in prefix)
            count = min(num_negatives, available)
            if count <= 0:
                continue
            if available <= max(4 * count, itemnum // 4):
                # Dense exclusions: sample the explicit eligible catalog.
                eligible = [
                    value
                    for value in catalog
                    if value not in prefix and value != target
                ]
                chosen = random.sample(eligible, count)
            else:
                # Sparse exclusions: rejection sampling avoids drawing an
                # extra candidate for every historical item. Accepted IDs are
                # an ordered uniform sample without replacement.
                chosen = []
                selected = set()
                attempts = 0
                while len(chosen) < count and attempts < 8 * count:
                    value = random.randint(1, itemnum)
                    attempts += 1
                    if (
                        value not in prefix
                        and value != target
                        and value not in selected
                    ):
                        selected.add(value)
                        chosen.append(value)
                if len(chosen) < count:
                    eligible = [
                        value
                        for value in catalog
                        if value not in prefix
                        and value != target
                        and value not in selected
                    ]
                    chosen.extend(random.sample(eligible, count - len(chosen)))
            neg[position, :count] = chosen
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg[:, 0] if num_negatives == 1 else neg)
    # Default diagnostic calls retain B x L; only training requests B x L x K.
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


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
    ### >>> Self_EvolveRec-BLOCK-START: Fix validation candidates across checkpoint comparisons without perturbing training RNG
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
            ### >>> Self_EvolveRec-BLOCK-START: Use repeatable unique validation negatives with bounded catalog availability
            candidates = [target]
            excluded_count = sum(1 for value in interacted if 1 <= value <= itemnum)
            candidate_count = min(100, 1 + itemnum - excluded_count)
            while len(candidates) < candidate_count:
                neg_item = rng.randint(1, itemnum)
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


### >>> Self_EvolveRec-BLOCK-START: Preserve candidate CE while exposing summed losses for immediate window-microbatch backward
def candidate_sequence_loss(
    model: nn.Module,
    seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    target_positions: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError("Expected mean or sum reduction")
    if seq.ndim != 2 or pos.shape != seq.shape:
        raise ValueError("Expected aligned B x L sequences and targets")
    if neg.ndim == 2:
        neg = neg.unsqueeze(-1)
    if neg.ndim != 3 or neg.shape[:2] != seq.shape or neg.size(-1) < 1:
        raise ValueError("Expected B x L x K negative candidates with K >= 1")
    # Training already computed eligibility once for the logical batch.
    # Only the default compatibility path needs to compute it here.
    if target_positions is None:
        valid = (seq != 0) & (pos != 0) & (neg != 0).any(dim=-1)
        target_positions = valid.nonzero(as_tuple=False)
    if (
        target_positions.ndim != 2
        or target_positions.size(1) != 2
        or target_positions.dtype != torch.long
        or target_positions.device != seq.device
    ):
        raise ValueError("Expected device-aligned N x 2 int64 target positions")
    if target_positions.size(0) == 0:
        raise RuntimeError("Caller must skip batches without eligible targets")
    rows, columns = target_positions.unbind(dim=1)
    features = model.encode_window_terminal(seq, target_positions)
    candidates = torch.cat((pos[rows, columns, None], neg[rows, columns]), dim=1)
    logits = model.predict_score(features[:, None, :], candidates)
    logits = logits.masked_fill(candidates == 0, -torch.inf)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    # Inherited uncorrected 32-negative candidate CE, temperature one.
    return nn.functional.cross_entropy(logits, targets, reduction=reduction)


### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("NEXTITNET_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("NEXTITNET_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("NEXTITNET_BLOCKS", "8"))
KERNEL_SIZE = int(os.environ.get("NEXTITNET_KERNEL", "3"))
DROPOUT_RATE = float(os.environ.get("NEXTITNET_DROPOUT", "0.2"))
LR = float(os.environ.get("NEXTITNET_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("NEXTITNET_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("NEXTITNET_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Enforce the matched training limits
NUM_BATCHES = 200
NUM_EPOCHS = max(
    1,
    min(
        200,
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore the corrected ungated NextItNet residual backbone
        self.activation = nn.GELU()
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Omit convolution taps that reach only causal zero padding
    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        dilation = conv.dilation[0]
        kernel = conv.kernel_size[0]
        active_taps = min(kernel, 1 + (x.size(-1) - 1) // dilation)
        pad = (active_taps - 1) * dilation
        padded = nn.functional.pad(x, (pad, 0))
        # Conv1d stores the current-position tap last. Earlier omitted taps
        # multiply only zeros and have zero gradient in the original operation.
        if active_taps == kernel:
            return conv(padded)
        return nn.functional.conv1d(
            padded,
            conv.weight[..., kernel - active_taps :].contiguous(),
            conv.bias,
            stride=conv.stride,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve ungated causal updates and padding masks
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        y = (self.norm1(x) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        y = self.dropout(y)
        return (residual + y) * mask

    ### <<< Self_EvolveRec-BLOCK-END


class NextItNetModel(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Configure one shared NextItNet with a fixed ten-item raw context
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
        context_window: int = 10,
        window_batch_size: int = 256,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.context_window = int(context_window)
        self.window_batch_size = int(window_batch_size)
        if self.context_window != 10 or self.window_batch_size < 1 or maxlen < 1:
            raise ValueError(
                "Require context_window=10, positive maxlen and window batch size"
            )
        ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Independently encode raw ten-item windows with preserved corrected-baseline absolute positions
    def raw_windows(self, sequence: torch.Tensor, target_positions: torch.Tensor):
        if sequence.ndim != 2 or not 0 < sequence.size(1) <= self.maxlen:
            raise ValueError("Expected B x L histories with 0 < L <= maxlen")
        if target_positions.ndim != 2 or target_positions.size(1) != 2:
            raise ValueError("Expected N x 2 row/position indices")
        rows, terminal = target_positions.unbind(dim=1)
        histories = sequence.index_select(0, rows)
        indices = torch.arange(sequence.size(1), device=sequence.device)
        eligible = (histories != 0) & (indices[None, :] <= terminal[:, None])
        # Select raw input IDs, never states computed from an older context.
        # Sorting the selected indices restores chronological order, with PAD
        # indices at the left. No aligned next-item target enters this window.
        selected = indices.expand_as(histories).masked_fill(~eligible, -1)
        selected = selected.topk(
            min(self.context_window, sequence.size(1)), dim=1, sorted=True
        ).values.flip(dims=(1,))
        if selected.size(1) < self.context_window:
            selected = nn.functional.pad(
                selected, (self.context_window - selected.size(1), 0), value=-1
            )
        occupied = selected >= 0
        windows = histories.gather(1, selected.clamp(min=0))
        windows = windows.masked_fill(~occupied, 0)
        # The supplied corrected ungated baseline uses absolute input indices.
        # Keep those IDs instead of renumbering each independently encoded window.
        position_ids = (selected + 1).masked_fill(~occupied, 0)
        return windows, position_ids

    def _encode_raw_windows(
        self, windows: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        mask = (windows != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.item_emb(windows) * (self.hidden_units**0.5)
        x = self.dropout(x + self.pos_emb(position_ids)) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x) * mask

    def encode_window_terminal(
        self, sequence: torch.Tensor, target_positions: torch.Tensor
    ) -> torch.Tensor:
        windows, position_ids = self.raw_windows(sequence, target_positions)
        features = self._encode_raw_windows(windows, position_ids)
        indices = torch.arange(windows.size(1), device=windows.device)
        last = indices.expand_as(windows).masked_fill(windows == 0, -1).amax(dim=1)
        rows = torch.arange(windows.size(0), device=windows.device)
        values = features[rows, last.clamp(min=0)]
        return values * (last >= 0).unsqueeze(-1).to(values.dtype)

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2 or not 0 < seq.size(1) <= self.maxlen:
            raise ValueError("Expected B x L histories with 0 < L <= maxlen")
        positions = (seq != 0).nonzero(as_tuple=False)
        output = self.item_emb.weight.new_zeros((seq.numel(), self.hidden_units))
        # Compatibility path for ModelView and probes. Training calls the
        # terminal helper directly and releases each microbatch graph promptly.
        for chunk in positions.split(self.window_batch_size):
            if chunk.size(0) == 0:
                continue
            values = self.encode_window_terminal(seq, chunk)
            flat = chunk[:, 0] * seq.size(1) + chunk[:, 1]
            output = output.index_copy(0, flat, values)
        return output.reshape(seq.size(0), seq.size(1), self.hidden_units)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve aligned and multi-negative forward compatibility
    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        pos_scores = self.predict_score(out, pos)
        neg_scores = self.predict_score(out, neg)
        return pos_scores, neg_scores

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Use the identical terminal-window path for prediction and support family diagnostic shapes
    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if features.shape[:-1] == items.shape:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == features.ndim and items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.size(0) == 0 or not 0 < seq.size(1) <= self.maxlen:
            raise ValueError("Expected nonempty B x L histories with L <= maxlen")
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1)
        rows = torch.arange(seq.size(0), device=seq.device)
        selected = torch.stack((rows, last.clamp(min=0)), dim=1)
        user_emb = self.encode_window_terminal(seq, selected)
        user_emb = user_emb * (last >= 0).unsqueeze(-1).to(user_emb.dtype)
        if candidates.ndim == 1:
            scores = user_emb @ self.item_emb(candidates).transpose(0, 1)
        elif candidates.ndim == 2 and candidates.size(0) == seq.size(0):
            scores = self.predict_score(user_emb[:, None, :], candidates)
        else:
            raise ValueError("Expected shared K or per-history B x K candidates")
        return scores[0] if seq.size(0) == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Initialize one ungated bounded-context NextItNet per category with seed 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = NextItNetModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        context_window=10,
        window_batch_size=256,
    ).to(device)
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "architecture": "nextitnet_raw_window10_v1",
                "context_window": model.context_window,
                "window_batch_size": model.window_batch_size,
                "window_padding": "left pad raw IDs to ten",
                "position_policy": "preserve corrected-baseline absolute input indices",
                "hidden_units": HIDDEN_UNITS,
                "dilations": [block.dilation for block in model.blocks],
                "temporal_convolutions_per_block": 2,
                "kernel_size": KERNEL_SIZE,
                "normalization": "tokenwise channel LayerNorm",
                "dropout": DROPOUT_RATE,
                "objective": "candidate_ce_temperature1_no_logq",
                "negative_count": 32,
                "negative_exclusions": "inherited prefix and aligned target",
                "sampler": "unchanged sample_batch using Python seed 42",
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": 5.0,
                "batch_size": BATCH_SIZE,
                "maxlen": MAXLEN,
                "batches_per_epoch": NUM_BATCHES,
                "max_epochs": NUM_EPOCHS,
                "eval_every": EVAL_EVERY,
                "patience": PATIENCE,
                "seed": SEED,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ### >>> Self_EvolveRec-BLOCK-START: Accumulate exact logical position-mean gradients over lazy raw-window microbatches
        total_loss = 0.0
        supervised_positions = 0
        skipped_positions = 0
        skipped_steps = 0
        for _ in range(NUM_BATCHES):
            # Sampling remains unchanged. Model/dropout use the torch RNG;
            # window microbatching never consumes this Python sampling stream.
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, num_negatives=32
            )
            valid = (seq != 0) & (pos != 0)
            eligible = valid & (neg != 0).any(axis=-1)
            skipped_positions += int((valid & ~eligible).sum())
            count = int(eligible.sum())
            optimizer.zero_grad(set_to_none=True)
            if count == 0:
                skipped_steps += 1
                continue
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            positions = torch.as_tensor(
                np.argwhere(eligible), dtype=torch.long, device=device
            )
            batch_loss = torch.zeros((), device=device)
            for chunk in positions.split(model.window_batch_size):
                loss_sum = candidate_sequence_loss(
                    model,
                    seq_t,
                    pos_t,
                    neg_t,
                    target_positions=chunk,
                    reduction="sum",
                )
                # Backward now, retaining no graph from earlier windows.
                (loss_sum / count).backward()
                batch_loss += loss_sum.detach()
                del loss_sum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(batch_loss.item())
            supervised_positions += count

        avg_loss = total_loss / max(supervised_positions, 1)
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "candidate_loss_position_mean": avg_loss,
                    "supervised_positions": supervised_positions,
                    "skipped_no_negative_positions": skipped_positions,
                    "skipped_optimizer_steps": skipped_steps,
                },
                sort_keys=True,
            ),
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

    ### >>> Self_EvolveRec-BLOCK-START: Return the validation-selected network consistently with the saved predictor
    if best_epoch == 0 or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"No validation-selected checkpoint for {dataset_name}")
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
                "num_blocks": NUM_BLOCKS,
                "kernel_size": KERNEL_SIZE,
                ### >>> Self_EvolveRec-BLOCK-START: Persist bounded-window architecture and inherited training configuration
                "architecture": "nextitnet_raw_window10_v1",
                "dropout_rate": DROPOUT_RATE,
                "context_window": model.context_window,
                "window_batch_size": model.window_batch_size,
                "window_padding": "left pad raw IDs to ten",
                "position_policy": "original absolute input index plus one; PAD zero",
                "dilations": [block.dilation for block in model.blocks],
                "temporal_convolutions_per_block": 2,
                "objective": "candidate_ce_temperature1_no_logq",
                "negative_count": 32,
                "negative_exclusions": "inherited prefix and aligned target",
                "sampler": "unchanged sample_batch",
                "seed": SEED,
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": 5.0,
                "batch_size": BATCH_SIZE,
                "batches_per_epoch": NUM_BATCHES,
                "max_epochs": NUM_EPOCHS,
                "eval_every": EVAL_EVERY,
                "patience": PATIENCE,
                "best_epoch": best_epoch,
                "selection": "strict validation HR@10 improvement",
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

### >>> Self_EvolveRec-BLOCK-START: Supply dependencies for the embedded trained architecture
import math
import os
import pickle

import torch
import torch.nn as nn
### <<< Self_EvolveRec-BLOCK-END

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


### >>> Self_EvolveRec-BLOCK-START: Load the architecture embedded by the predictor writer
# __NEXTITNET_ARCHITECTURE__

def _load_model_class():
    return NextItNetModel
### <<< Self_EvolveRec-BLOCK-END


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
        ### >>> Self_EvolveRec-BLOCK-START: Require a checkpoint and architecture metadata for every routed category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing NextItNet artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        ### >>> Self_EvolveRec-BLOCK-START: Reject incompatible predictor architecture metadata explicitly
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("architecture") != "nextitnet_raw_window10_v1":
            raise RuntimeError(f"Incompatible NextItNet architecture for {ds}")
        if meta.get("context_window") != 10:
            raise RuntimeError(f"Expected ten-interaction context for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the identical raw-window architecture in the generated predictor
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            context_window=meta["context_window"],
            window_batch_size=meta["window_batch_size"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Prevent cross-category fallback and fabricated scores
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple categories are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"No trained NextItNet model for category {dataset_name!r}")
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
    ### >>> Self_EvolveRec-BLOCK-START: Embed the exact training classes without importing training during inference
    import inspect
    import textwrap

    architecture = "\n\n".join(
        textwrap.dedent(inspect.getsource(cls))
        for cls in (NextItNetBlock, NextItNetModel)
    )
    placeholder = "# __NEXTITNET_ARCHITECTURE__"
    if script.count(placeholder) != 1:
        raise RuntimeError("Predictor architecture placeholder must occur exactly once")
    script = script.replace(placeholder, architecture)
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    ### <<< Self_EvolveRec-BLOCK-END
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


