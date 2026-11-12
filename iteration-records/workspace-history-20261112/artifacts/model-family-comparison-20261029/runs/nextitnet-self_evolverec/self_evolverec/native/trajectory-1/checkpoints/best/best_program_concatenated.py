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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret controlled training-prefix sensitivity without generalization claims
    - Every statement must be grounded in the provided metric values/definitions.
    - Training-prefix ranks, HR@10, margins and interventions measure in-sample fit or sensitivity, not held-out generalization or user preferences.
    - Read matching failures, sample counts, repeated targets, positional shifts and structural reachability before interpreting recency effects.
    - Separate original fixed-catalog effects from restricted-catalog effects removing the same inserted IDs from all paired histories. Restricted ranks are not full-catalog ranks.
    - Report inserted-item score/rank gains separately from target sensitivity; neither identifies causal user intent.
    - Frequency and possible-reachability matching do not establish equal embedding displacement, exact path counts or behavioral plausibility. Training transition evidence is only a corruption constraint.
    - Interpret paired recent-minus-older effects only on successful common-support blocks. Do not infer monotonic dose response, calibrated false-alarm rates or bootstrap coverage from these probes.
    - Corruption Monte Carlo error and exploratory user-bootstrap uncertainty measure different sources of variation; additional corruption draws do not increase independent user counts.
    - User-bootstrap intervals describe sampled-user uncertainty within this category; do not invent a four-category macro interval.
    - Null values and empty category breakdowns mean unavailable evidence. Do not infer ratings, review text, item categories or simulator traits.
    - Embedding cosine alone cannot establish collapse, diversity, or recommendation quality.
    - Separate certified feasibility, unresolved searches, implementation errors, and scoring censoring. Matching coverage alone does not prove joint feasibility.
    - Identification bounds conditional on frozen pools are not sampling confidence intervals. Common-support legacy effects and exact-only deterministic-recovery effects have different selectors and populations; never call recovery uniform or interpret recovered support as validation improvement.
    - The unchanged pinned host macro validation HR@10 remains authoritative for model selection.
    - Subsequent model changes require a preregistered hypothesis and one jointly trained NextItNet per category, with matching checkpoint and predictor architecture.
    - With absent or inconclusive evidence, state the limitation without asserting a critical failure or healthy generalization.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the ModelView contract and exclude unavailable metadata
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this benchmark.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish missing embedding evidence from a measured value
            if n <= 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained NextItNet module
        self.network.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on an invalid diagnostic loader contract
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score aligned causal NextItNet features without an incompatible candidate axis
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Expected B x L sequence, positive and negative tensors"
                )
            log_feats = self.model.log2feats(seq_t)
            margin = self.model.predict_score(
                log_feats, pos_t
            ) - self.model.predict_score(log_feats, neg_t)
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Reuse the measured position eligibility mask
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
            ### >>> Self_EvolveRec-BLOCK-START: Report absent margin observations explicitly
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
            ### <<< Self_EvolveRec-BLOCK-END
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

        breakdown = {
            "eps": float(eps),
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

    ### >>> Self_EvolveRec-BLOCK-START: Check bounded-context architecture, preserve feasibility evidence and report construction support explicitly
    @staticmethod
    def _ahead(scores, ids, target_score, target):
        return (scores > target_score) | ((scores == target_score) & (ids < target))

    def _validate_rank(self):
        ids = np.array([7, 2, 9, 1, 5], dtype=np.int64)
        scores = np.array([0.0, 2.0, 2.0, -1.0, 2.0])
        expected = ids[np.lexsort((ids, -scores))].tolist()
        for target, score in zip(ids, scores):
            for chunk in (1, 2, 3, 8):
                rank = 1 + sum(
                    int(
                        self._ahead(
                            scores[j : j + chunk], ids[j : j + chunk], score, target
                        ).sum()
                    )
                    for j in range(0, len(ids), chunk)
                )
                if rank != expected.index(target) + 1:
                    raise AssertionError("Chunked rank/tie control failed")
        return True

    def _pack(self, history):
        length = int(self.network.maxlen)
        history = history[-length:]
        if not history:
            raise ValueError("Empty diagnostic prefix")
        seq = torch.zeros((1, length), device=self.device, dtype=torch.long)
        seq[0, -len(history) :] = torch.as_tensor(history, device=self.device)
        return seq

    def _feature(self, history):
        seq = self._pack(history)
        positions = torch.arange(seq.size(1), device=self.device)
        last = positions.masked_fill(seq[0] == 0, -1).max()
        if int(last) < 0:
            raise ValueError("All-padding diagnostic input")
        return self.model.log2feats(seq)[0, last]

    def _scores(self, feature, ids):
        items = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        scores = self.model.predict_score(feature.expand(items.numel(), -1), items)
        if scores.shape != items.shape or not torch.isfinite(scores).all():
            raise ValueError("Invalid diagnostic scores")
        return scores.detach().double().cpu().numpy()

    def _catalog_scores(self, history, ids, chunk):
        # Only one feature computation and chunk-bounded GPU scoring per history.
        ids = np.asarray(ids, dtype=np.int64)
        if (
            chunk < 1
            or ids.ndim != 1
            or len(ids) == 0
            or np.any(ids < 1)
            or np.any(ids > int(self.network.itemnum))
            or np.any(ids[1:] <= ids[:-1])
        ):
            raise ValueError(
                "Expected positive chunk size and sorted unique in-catalog IDs"
            )
        feature = self._feature(history)
        scores = np.empty(len(ids), dtype=np.float64)
        for start in range(0, len(ids), chunk):
            scores[start : start + chunk] = self._scores(
                feature, ids[start : start + chunk]
            )
        return scores

    def _result(self, ids, scores, target, sampled, removed=(), include_top10=True):
        ids = np.asarray(ids, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        if (
            ids.ndim != 1
            or scores.shape != ids.shape
            or len(ids) == 0
            or np.any(ids < 1)
            or np.any(ids > int(self.network.itemnum))
            or np.any(ids[1:] <= ids[:-1])
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Invalid sorted catalog IDs or aligned finite scores")
        removed = set(removed)
        keep = ~np.isin(ids, list(removed)) if removed else slice(None)
        catalog, values = ids[keep], scores[keep]
        index = int(np.searchsorted(catalog, target))
        if index == len(catalog) or int(catalog[index]) != target:
            raise ValueError("Target missing from diagnostic catalog")
        score = float(values[index])
        rank = 1
        top_ids, top_scores = np.empty(0, dtype=np.int64), np.empty(0)
        for start in range(0, len(catalog), 4096):
            chunk_ids = catalog[start : start + 4096]
            chunk_scores = values[start : start + 4096]
            rank += int(self._ahead(chunk_scores, chunk_ids, score, target).sum())
            if include_top10:
                # Retain each chunk's top ten, including ascending-ID tie order.
                # Every global top-ten item must be in its own chunk's top ten.
                order = np.lexsort((chunk_ids, -chunk_scores))[:10]
                top_ids = np.concatenate((top_ids, chunk_ids[order]))
                top_scores = np.concatenate((top_scores, chunk_scores[order]))
                order = np.lexsort((top_ids, -top_scores))[:10]
                top_ids, top_scores = top_ids[order], top_scores[order]
        negatives = np.asarray([i for i in sampled if i not in removed], dtype=np.int64)
        if (
            negatives.ndim != 1
            or len(np.unique(negatives)) != len(negatives)
            or np.any(negatives == target)
            or not np.isin(negatives, catalog).all()
        ):
            raise ValueError("Sampled negatives must be unique eligible non-target IDs")
        margin = None
        if len(negatives):
            margin = float(score - values[np.searchsorted(catalog, negatives)].mean())
        return {
            "rank": rank,
            "hr10": float(rank <= 10),
            "score": score,
            "margin": margin,
            "top10": top_ids.tolist() if include_top10 else None,
            "catalog_size": len(catalog),
            "margin_negatives": len(negatives),
        }

    def _architecture(self, history):
        import inspect

        # Verify the supplied forward implementations before using their graph.
        # Otherwise numerical controls remain available, but reachability is unknown.
        offsets, convs, reason = {0}, [], None
        try:
            encode = inspect.getsource(type(self.network)._encode_raw_windows)
            windows = inspect.getsource(type(self.network).raw_windows)
            terminal = inspect.getsource(type(self.network).encode_window_terminal)
            block_code = [
                inspect.getsource(type(b).forward) for b in self.network.blocks
            ]
            known = (
                type(self.network).__name__ == "NextItNetModel"
                and self.network.context_window == 10
                and "self.pos_emb(position_ids)" in encode
                and "x = block(x, mask)" in encode
                and "return self.norm(x) * mask" in encode
                and "indices[None, :] <= terminal[:, None]" in windows
                and "min(self.context_window, sequence.size(1))" in windows
                and "self._encode_raw_windows(windows, position_ids)" in terminal
                and all(
                    "return (residual + y) * mask" in s
                    and "self.norm1(x) * mask" in s
                    and "self.norm2(y) * mask" in s
                    for s in block_code
                )
                and all(
                    isinstance(b.norm1, torch.nn.LayerNorm)
                    and isinstance(b.norm2, torch.nn.LayerNorm)
                    and tuple(b.norm1.normalized_shape) == (self.network.hidden_units,)
                    and tuple(b.norm2.normalized_shape) == (self.network.hidden_units,)
                    for b in self.network.blocks
                )
            )
            if not known:
                raise ValueError(
                    "Forward implementation differs from supplied raw-window NextItNet"
                )
            for block in self.network.blocks:
                branch = set(offsets)
                for conv in (block.conv1, block.conv2):
                    if (
                        not isinstance(conv, torch.nn.Conv1d)
                        or conv.stride != (1,)
                        or conv.padding != (0,)
                    ):
                        raise ValueError("Unsupported convolution geometry")
                    if conv.dilation != (block.dilation,) or conv.kernel_size != (
                        block.kernel_size,
                    ):
                        raise ValueError(
                            "Causal padding and convolution attributes differ"
                        )
                    taps = {i * conv.dilation[0] for i in range(conv.kernel_size[0])}
                    branch = {a + b for a in branch for b in taps}
                    convs.append(
                        {"kernel": conv.kernel_size[0], "dilation": conv.dilation[0]}
                    )
                offsets |= branch
            # Probe histories are contiguous nonpadding items. Independently
            # encoded raw windows exclude all earlier items before convolution.
            # This positional interpretation does not extend to inputs with gaps.
            offsets &= set(range(int(self.network.context_window)))
        except (AttributeError, TypeError, OSError, ValueError) as exc:
            offsets, reason = None, str(exc)
        seq = self._pack(history)
        features = self.model.log2feats(seq)
        pivot = seq.size(1) - len(history) + (len(history) - 1) // 2
        changed = seq.clone()
        suffix = changed[:, pivot + 1 :]
        changed[:, pivot + 1 :] = 1 + suffix.remainder(int(self.network.itemnum))
        count = int((seq != changed).sum())
        altered = self.model.log2feats(changed)
        causal_error = float(
            (features[:, : pivot + 1] - altered[:, : pivot + 1]).abs().max()
        )
        empty_norm = float(
            self.model.log2feats(torch.zeros_like(seq)).norm(dim=-1).max()
        )
        unreachable = next(
            (
                i
                for i in range(len(history) - 1)
                if offsets is not None and len(history) - 1 - i not in offsets
            ),
            None,
        )
        unreachable_error = None
        if unreachable is not None and int(self.network.itemnum) > 1:
            control = list(history)
            control[unreachable] = 1 + control[unreachable] % int(self.network.itemnum)
            unreachable_error = float(
                (self._feature(control) - features[0, -1]).abs().max()
            )
        return {
            "convolutions": convs,
            "structural_possible_offsets": (
                sorted(offsets) if offsets is not None else None
            ),
            "reachability_unavailable_reason": reason,
            "reachability_scope": "Possible offsets for contiguous diagnostic histories under independently encoded raw ten-item windows; masks and learned weights can remove influence. Source recognition is a guard, not a formal proof.",
            "exact_path_counts": None,
            "path_count_unavailable_reason": "Active nonlinear paths are not measured; structural accessibility is not learned influence",
            "causality_changed_tokens": count,
            "causality_max_abs_error": causal_error if count else None,
            "all_padding_feature_max_norm": empty_norm,
            "unreachable_control_index": unreachable,
            "unreachable_control_max_abs_error": unreachable_error,
            "unreachable_control_unavailable_reason": (
                None
                if unreachable_error is not None
                else "No applicable occupied unreachable position or no alternative item"
            ),
            "readout": "maximum nonpadding index; all-padding prefixes skipped",
        }, offsets

    @staticmethod
    def _recover_assignments(domains, groups, checker, incumbent, deadline, cap=10000):
        import time

        nodes, matching_calls, candidate_checks = 0, 0, 0
        certificate = None
        if set(groups) != set(domains) or cap < 0:
            raise ValueError("Expected one capacity group per slot and nonnegative cap")
        if any(len(set(values)) != len(values) for values in domains.values()):
            raise ValueError("Candidate domains must contain unique IDs")
        if incumbent is not None and not checker(incumbent):
            raise ValueError("Supplied greedy incumbent failed complete validation")
        solution = dict(incumbent) if incumbent is not None else None

        def bounded():
            if time.perf_counter() >= deadline:
                raise TimeoutError("category_safety_cap")

        def matching(remaining, assigned):
            nonlocal matching_calls
            matching_calls += 1
            used = {(groups[s], value) for s, value in assigned.items()}
            edges = {}
            for s in remaining:
                bounded()
                edges[s] = [
                    (groups[s], v) for v in domains[s] if (groups[s], v) not in used
                ]
            owners = {}

            def augment(slot, seen):
                bounded()
                for item in edges[slot]:
                    bounded()
                    if item in seen:
                        continue
                    seen.add(item)
                    if item not in owners or augment(owners[item], seen):
                        owners[item] = slot
                        return True
                return False

            for slot in sorted(remaining):
                augment(slot, set())
            if len(owners) == len(remaining):
                return True, None
            # Alternating reachability yields a checkable Hall deficiency.
            slots = set(remaining) - set(owners.values())
            neighbors = set()
            while True:
                bounded()
                expanded = {item for s in slots for item in edges[s]}
                reached = slots | {owners[i] for i in expanded if i in owners}
                if reached == slots:
                    neighbors = expanded
                    break
                slots = reached
            # Verify the witness directly from the frozen domains, separately
            # from the augmenting-path traversal that constructed it.
            witness_neighbors = {
                (groups[s], v)
                for s in slots
                for v in domains[s]
                if (groups[s], v) not in used
            }
            if neighbors != witness_neighbors or not len(neighbors) < len(slots):
                raise AssertionError("Invalid Hall certificate")
            return False, {
                "slots": sorted(slots),
                "scoped_neighbors": sorted(neighbors),
            }

        def search(assigned):
            nonlocal nodes, candidate_checks
            bounded()
            if nodes >= cap:
                raise TimeoutError("node_cap")
            nodes += 1
            if len(assigned) == len(domains):
                return dict(assigned) if checker(assigned) else None
            remaining = [s for s in domains if s not in assigned]
            possible, _ = matching(remaining, assigned)
            if not possible:
                return None
            used = {(groups[s], v) for s, v in assigned.items()}
            available = {
                s: [v for v in domains[s] if (groups[s], v) not in used]
                for s in remaining
            }
            slot = min(remaining, key=lambda s: (len(available[s]), s))
            for value in available[slot]:
                candidate_checks += 1
                assigned[slot] = value
                found = search(assigned)
                del assigned[slot]
                if found is not None:
                    return found
            return None

        reason = None
        proof = None
        try:
            possible, certificate = matching(list(domains), {})
            if solution is not None:
                if not possible:
                    raise AssertionError("Feasible incumbent conflicts with matching")
            elif possible:
                solution = search({})
            status = (
                "certified_feasible" if solution is not None else "certified_infeasible"
            )
            proof = (
                "independent_complete_check"
                if solution is not None
                else (
                    "scoped_Hall_deficiency"
                    if certificate is not None
                    else "exhaustive_joint_search"
                )
            )
        except TimeoutError as exc:
            reason = str(exc)
            status = "certified_feasible" if solution is not None else "unresolved"
            proof = "independent_complete_check" if solution is not None else None
        return solution, {
            "proof": proof,
            "feasibility": status,
            "optimization": "not_requested",
            "nodes": nodes,
            "matching_calls": matching_calls,
            "candidate_checks": candidate_checks,
            "certificate": certificate,
            "interruption": reason,
            "assignment": solution,
        }

    def _validate_constructor(self):
        import itertools
        import time

        cases = [
            ({0: [1, 2], 1: [1]}, {0: 0, 1: 0}, False),
            ({0: [], 1: [1]}, {0: 0, 1: 0}, False),
            ({0: [1], 1: [1]}, {0: 0, 1: 0}, False),
            ({0: [1], 1: [1]}, {0: 0, 1: 1}, False),
            ({0: [1, 2, 3], 1: [1, 2]}, {0: 0, 1: 0}, True),
            # Hall deficiency without any isolated slot.
            ({0: [1, 2], 1: [1, 2], 2: [1, 2]}, {0: 0, 1: 0, 2: 0}, False),
        ]
        for domains, groups, reject_all in cases:

            def direct(a):
                return (
                    set(a) == set(domains)
                    and not reject_all
                    and all(a[s] in domains[s] for s in domains)
                    and all(
                        groups[s] != groups[t] or a[s] != a[t]
                        for s in domains
                        for t in domains
                        if s < t
                    )
                )

            exhaustive = [
                dict(zip(domains, values))
                for values in itertools.product(*(domains[s] for s in domains))
                if direct(dict(zip(domains, values)))
            ]
            found, result = self._recover_assignments(
                domains, groups, direct, None, time.perf_counter() + 5
            )
            assert (found is not None) == bool(exhaustive)
            assert result["feasibility"] == (
                "certified_feasible" if exhaustive else "certified_infeasible"
            )
            _, stopped = self._recover_assignments(
                domains, groups, direct, None, time.perf_counter() - 1
            )
            assert stopped["feasibility"] == "unresolved"
            if exhaustive:
                kept, stopped = self._recover_assignments(
                    domains, groups, direct, exhaustive[0], time.perf_counter() - 1
                )
                assert kept == exhaustive[0]
                assert stopped["feasibility"] == "certified_feasible"
            _, limited = self._recover_assignments(
                domains, groups, direct, None, time.perf_counter() + 5, cap=0
            )
            assert limited["feasibility"] in {"unresolved", "certified_infeasible"}
            if exhaustive:
                assert limited["feasibility"] == "unresolved"
                kept, limited = self._recover_assignments(
                    domains,
                    groups,
                    direct,
                    exhaustive[0],
                    time.perf_counter() + 5,
                    cap=0,
                )
                assert kept == exhaustive[0]
                assert limited["feasibility"] == "certified_feasible"
        # A covering matching can still violate a cross-plan joint predicate.
        domains, groups = {0: [1, 2], 1: [1, 2]}, {0: 0, 1: 1}

        def joint(a):
            return (
                set(a) == {0, 1}
                and all(a[s] in domains[s] for s in domains)
                and a[0] < a[1]
            )

        exhaustive = [{0: a, 1: b} for a in domains[0] for b in domains[1] if a < b]
        found, result = self._recover_assignments(
            domains, groups, joint, None, time.perf_counter() + 5
        )
        assert found in exhaustive
        assert result["feasibility"] == "certified_feasible"
        return True

    @staticmethod
    def _aggregate(rows, bootstrap=400, seed=42):
        rng = np.random.default_rng(seed)
        result = {}
        for variant in sorted({r["variant"] for r in rows}):
            selected = [r for r in rows if r["variant"] == variant]
            users = sorted({r["user"] for r in selected})
            entry = {
                "users": len(users),
                "prefixes": len({(r["user"], r["cut"]) for r in selected}),
                "observations": len(selected),
            }
            for metric in (
                "delta_rank",
                "delta_hr10",
                "delta_sampled_margin",
                "delta_target_score",
                "top10_overlap",
            ):
                user_values, user_mc_variances = [], []
                for user in users:
                    records = [
                        r
                        for r in selected
                        if r["user"] == user and r.get(metric) is not None
                    ]
                    cuts = sorted({r["cut"] for r in records})
                    groups = [
                        np.asarray(
                            [r[metric] for r in records if r["cut"] == cut], dtype=float
                        )
                        for cut in cuts
                    ]
                    if not groups:
                        continue
                    user_values.append(float(np.mean([g.mean() for g in groups])))
                    # Conditional corruption variability only; missing draws cannot
                    # establish MC precision, and are never assigned zero variance.
                    user_mc_variances.append(
                        sum(float(g.var(ddof=1) / len(g)) for g in groups)
                        / len(groups) ** 2
                        if all(len(g) >= 2 for g in groups)
                        else None
                    )
                values = np.asarray(user_values)
                interval = None
                if len(values) >= 2:
                    draws = values[
                        rng.integers(0, len(values), size=(bootstrap, len(values)))
                    ].mean(axis=1)
                    interval = np.quantile(draws, [0.025, 0.975]).tolist()
                mc = None
                if user_mc_variances and all(v is not None for v in user_mc_variances):
                    mc = float(np.sqrt(sum(user_mc_variances)) / len(user_mc_variances))
                entry[metric] = {
                    "mean": float(values.mean()) if len(values) else None,
                    "ci95": interval,
                    "corruption_mc_se": mc,
                    "observed_users": len(values),
                }
            result[variant] = entry
        return result

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        import time

        protocol = {
            "version": "nextitnet-frozen-pool-feasibility-v5",
            "legacy_selector_law": "Sequential Python Random(42) stream: sample users, sampled-margin negatives, shuffle position pairs, greedy nonadjacent position selection, then choose uniformly from each remaining within-plan domain; earlier failures change later stream consumption. Exact recovery consumes no RNG. This is not uniform complete-assignment sampling.",
            "safety_cap_semantics": "Cooperative wall-time cap; indivisible pool, model and summary operations can overshoot. Unstarted blocks stay unresolved; feasibility and scoring completion are separate.",
            "constructor_node_cap": 10000,
            "category_safety_seconds": 30.0,
            "reuse": "Distinct within each plan; reuse permitted across plans",
            "recovery_selector": "First feasible DFS; minimum remaining domain then stable slot ID; ascending item ID; no distance optimization",
            "selector_distance": "Constant zero on admissible edges; item ID breaks ties; no embedding-dependent selection",
            "feasibility_scope": "Frozen selected positions and finite candidate pools only; missing position assignments remain unresolved",
            "temporal_availability": "Not established from available timestamps",
            "seed": 42,
            "max_users": 12,
            "prefixes_per_user": 2,
            "minimum_prefix_length": 5,
            "history_window": int(self.network.maxlen),
            "draws": 5,
            "doses": [1, 2],
            "bootstrap_repetitions": 400,
            "catalog_chunk_size": 4096,
            "sampled_margin_negatives": 99,
            "eps": 0.5,
            "source": "Training histories and next training items only",
            "frequency_bins": "0, 1, 2-4, 5-15, >=16; replacements require positive training frequency",
            "lag_bands": {"recent": [1, 4], "older": [5, None]},
            "position_matching": "Equal dose, identical ordered original/replacement frequency bins and possible-reachability status across bands; nonadjacent selected positions",
            "replacement_types": ["frequency_random", "local_plausible"],
            "local_plausibility": "Every available immediate in-window directed neighbor transition must be observed in training; each edge count must match the original edge's bin (1, 2-4, 5-15, >=16)",
            "common_support": "Preserve independently checked greedy assignments; report deterministic recovery separately on exact-only support",
            "prefix_sampling": "Freeze all user cuts with a separate seed-42 RNG before sampling negatives, positions, replacements or permutations",
            "assignment_algorithm": "One randomized greedy attempt per draw; no retries or tolerance relaxation",
            "displacement_bin_edges": [0.0, 0.5, 1.0, None],
            "replacement_exclusions": "Padding, target, all original full-prefix IDs, duplicate replacements",
            "candidate_rule": "Frozen sorted IDs 1..itemnum excluding original full-prefix items except target",
            "restricted_rule": "Remove the same union of inserted IDs across both bands/types from clean and altered catalogs",
            "permutations": ["full window", "terminal preserving"],
            "tie_break": "Exact score, then ascending item ID",
        }
        states = [(m, m.training) for m in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            print(
                "Diagnostic protocol: " + json.dumps(protocol, sort_keys=True),
                flush=True,
            )
            # Isolate any legacy loader sampling as well as the dedicated plan RNG.
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            self.network.eval()
            return self._run_protocol(train_loader, protocol, started)
        finally:
            for module, training in states:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _run_protocol(self, train_loader, protocol, started):
        import time

        rng = random.Random(protocol["seed"])
        self._validate_rank()
        constructor_check = self._validate_constructor()
        deadline = started + protocol["category_safety_seconds"]
        source = self.user_train.get("History", self.user_train)
        if not isinstance(source, dict):
            raise ValueError("Expected ModelView training histories")
        itemnum = int(self.network.itemnum)
        coverage, histories = Counter(), {}
        for user in sorted(source, key=str):
            try:
                history = [int(i) for i in source[user]]
            except (TypeError, ValueError):
                coverage["invalid_history_users"] += 1
                continue
            if not history or any(i < 1 or i > itemnum for i in history):
                coverage["empty_or_out_of_catalog_users"] += 1
                continue
            key = str(user)
            if key in histories:
                raise ValueError("Training user IDs collide after string normalization")
            histories[key] = history
        frequency = Counter(i for h in histories.values() for i in h)
        transitions = Counter(
            (a, b) for h in histories.values() for a, b in zip(h, h[1:])
        )
        outgoing, incoming = {}, {}
        for a, b in transitions:
            outgoing.setdefault(a, set()).add(b)
            incoming.setdefault(b, set()).add(a)

        def bin_count(count):
            return (
                -1
                if count == 0
                else int(np.searchsorted([1, 4, 15], count, side="left"))
            )

        buckets = {}
        for item in sorted(frequency):
            buckets.setdefault(bin_count(frequency[item]), set()).add(item)
        users = sorted(
            u
            for u, h in histories.items()
            if len(h) > protocol["minimum_prefix_length"]
        )
        coverage["eligible_users"] = len(users)
        coverage["short_users"] = len(histories) - len(users)
        users = sorted(rng.sample(users, min(len(users), protocol["max_users"])))
        coverage["selected_users"] = len(users)
        prefix_rng = random.Random(protocol["seed"])
        cuts_by_user = {}
        for user in users:
            cuts = range(protocol["minimum_prefix_length"], len(histories[user]))
            cuts_by_user[user] = sorted(
                prefix_rng.sample(cuts, min(len(cuts), protocol["prefixes_per_user"]))
            )
        rows, baselines, failures, controls = [], [], [], []
        assignment_records = {
            (u, cut, dose, draw): {
                "user": u,
                "cut": cut,
                "dose": dose,
                "draw": draw,
                "feasibility": "unresolved",
                "optimization": "not_requested",
                "reason": "not_started",
                "selector": None,
            }
            for u in users
            for cut in cuts_by_user[u]
            for dose in protocol["doses"]
            for draw in range(protocol["draws"])
        }
        architecture, reachable = None, None

        def effects(clean, altered):
            return {
                "delta_rank": altered["rank"] - clean["rank"],
                "delta_hr10": altered["hr10"] - clean["hr10"],
                "delta_target_score": altered["score"] - clean["score"],
                "delta_sampled_margin": (
                    altered["margin"] - clean["margin"]
                    if altered["margin"] is not None and clean["margin"] is not None
                    else None
                ),
                "top10_overlap": len(set(clean["top10"]) & set(altered["top10"]))
                / len(clean["top10"]),
            }

        for user in users:
            history = histories[user]
            for cut in cuts_by_user[user]:
                if time.perf_counter() >= deadline:
                    coverage["category_safety_cap_hits"] += 1
                    continue
                coverage["sampled_prefixes"] += 1
                full_prefix, target = history[:cut], history[cut]
                prefix = full_prefix[-protocol["history_window"] :]
                if not prefix:
                    coverage["all_padding_prefixes"] += 1
                    continue
                if len(prefix) < protocol["minimum_prefix_length"]:
                    coverage["short_window_prefixes"] += 1
                    continue
                if architecture is None:
                    architecture, reachable = self._architecture(prefix)
                excluded = set(full_prefix) - {target}
                ids = np.asarray(
                    [i for i in range(1, itemnum + 1) if i not in excluded],
                    dtype=np.int64,
                )
                negatives = [int(i) for i in ids if i != target]
                if not negatives:
                    coverage["no_eligible_negatives"] += 1
                    continue
                sampled = rng.sample(
                    negatives, min(len(negatives), protocol["sampled_margin_negatives"])
                )
                clean_scores = self._catalog_scores(
                    prefix, ids, protocol["catalog_chunk_size"]
                )
                clean = self._result(ids, clean_scores, target, sampled)
                identity_scores = self._catalog_scores(
                    list(prefix), ids, protocol["catalog_chunk_size"]
                )
                identity = self._result(ids, identity_scores, target, sampled)
                controls.append(
                    {
                        "user": user,
                        "cut": cut,
                        "identity_score_max_abs_error": float(
                            np.max(np.abs(clean_scores - identity_scores))
                        ),
                        "identity_delta_rank": identity["rank"] - clean["rank"],
                    }
                )
                strata = {
                    "prefix_length": (
                        "5-15"
                        if len(prefix) <= 15
                        else "16-63" if len(prefix) <= 63 else ">=64"
                    ),
                    "target_popularity": str(bin_count(frequency[target])),
                    "repeated_target": str(target in full_prefix),
                    "transition_frequency": str(
                        bin_count(transitions[(prefix[-1], target)])
                    ),
                }
                common = {"user": user, "cut": cut, **strata}
                baselines.append({**common, **clean})
                n = len(prefix)

                def reach(index):
                    return None if reachable is None else n - 1 - index in reachable

                forbidden = set(full_prefix) | {target, 0}
                random_pools = {
                    key: sorted(bucket - forbidden) for key, bucket in buckets.items()
                }

                def candidate_pool(index, plausible):
                    key = bin_count(frequency[prefix[index]])
                    if not plausible:
                        return random_pools.get(key, [])
                    # Intersect sparse neighborhoods before copying frequency buckets.
                    following = prefix[index + 1]
                    pool = incoming.get(following, set()) & buckets.get(key, set())
                    pool -= forbidden
                    if plausible:
                        if index:
                            pool &= outgoing.get(prefix[index - 1], set())
                        pool = {
                            i
                            for i in pool
                            if bin_count(transitions[(i, following)])
                            == bin_count(transitions[(prefix[index], following)])
                            and (
                                index == 0
                                or bin_count(transitions[(prefix[index - 1], i)])
                                == bin_count(
                                    transitions[(prefix[index - 1], prefix[index])]
                                )
                            )
                        }
                    return sorted(pool)

                pools = {
                    (i, kind): candidate_pool(i, kind == "local_plausible")
                    for i in range(n - 1)
                    for kind in protocol["replacement_types"]
                }
                pairs = [
                    (r, o)
                    for r in range(n - 1)
                    for o in range(n - 1)
                    if 1 <= n - 1 - r <= 4
                    and n - 1 - o >= 5
                    and bin_count(frequency[prefix[r]])
                    == bin_count(frequency[prefix[o]])
                    and reach(r) == reach(o)
                ]
                for dose in protocol["doses"]:
                    for draw in range(protocol["draws"]):
                        audit = assignment_records[user, cut, dose, draw]
                        if time.perf_counter() >= deadline:
                            audit["reason"] = "category_safety_cap"
                            continue
                        shuffled_pairs = list(pairs)
                        rng.shuffle(shuffled_pairs)
                        selected, occupied = [], set()
                        for pair in shuffled_pairs:
                            if (
                                any(abs(i - j) <= 1 for i in pair for j in occupied)
                                or abs(pair[0] - pair[1]) <= 1
                            ):
                                continue
                            selected.append(pair)
                            occupied.update(pair)
                            if len(selected) == dose:
                                break
                        tag = {**common, "dose": dose, "draw": draw}
                        if len(selected) != dose:
                            audit["reason"] = "frozen_positions_not_constructed"
                            coverage["position_matching_failures"] += 1
                            failures.append(
                                {
                                    **tag,
                                    "reason": "Greedy position assignment failed; feasibility not established",
                                }
                            )
                            continue
                        plans, pool_sizes = [], []
                        failed = False
                        for kind in protocol["replacement_types"]:
                            for band_index, band in enumerate(("recent", "older")):
                                indices = [p[band_index] for p in selected]
                                inserted = []
                                for index in indices:
                                    pool = [
                                        i
                                        for i in pools[index, kind]
                                        if i not in inserted
                                    ]
                                    pool_sizes.append(
                                        {
                                            "type": kind,
                                            "band": band,
                                            "index": index,
                                            "available": len(pool),
                                        }
                                    )
                                    if not pool:
                                        failed = True
                                        break
                                    inserted.append(rng.choice(pool))
                                if len(inserted) == dose:
                                    plans.append((kind, band, indices, inserted))
                        import hashlib

                        specs = [
                            (kind, band, [p[b] for p in selected])
                            for kind in protocol["replacement_types"]
                            for b, band in enumerate(("recent", "older"))
                        ]
                        slots = [
                            (p, j, index)
                            for p, (_, _, indices) in enumerate(specs)
                            for j, index in enumerate(indices)
                        ]
                        domains = {
                            s: tuple(pools[index, specs[p][0]])
                            for s, (p, j, index) in enumerate(slots)
                        }
                        groups = {s: p for s, (p, j, index) in enumerate(slots)}

                        def complete_check(assignment):
                            if set(assignment) != set(domains):
                                return False
                            all_positions = [i for pair in selected for i in pair]
                            if len(set(all_positions)) != 2 * dose or any(
                                abs(a - b) <= 1
                                for a in all_positions
                                for b in all_positions
                                if a != b
                            ):
                                return False
                            for p, (kind, band, indices) in enumerate(specs):
                                changed = list(prefix)
                                values = [
                                    assignment[s] for s in domains if groups[s] == p
                                ]
                                if len(set(values)) != dose:
                                    return False
                                for s, (plan, j, index) in enumerate(slots):
                                    if plan != p:
                                        continue
                                    value = assignment[s]
                                    if value not in domains[s] or value in forbidden:
                                        return False
                                    if bin_count(frequency[value]) != bin_count(
                                        frequency[prefix[index]]
                                    ):
                                        return False
                                    changed[index] = value
                                if changed[-1] != prefix[-1] or len(changed) != n:
                                    return False
                                if kind == "local_plausible":
                                    for index in indices:
                                        for a, b in (
                                            (index - 1, index),
                                            (index, index + 1),
                                        ):
                                            if a >= 0 and (
                                                transitions[changed[a], changed[b]] <= 0
                                                or bin_count(
                                                    transitions[changed[a], changed[b]]
                                                )
                                                != bin_count(
                                                    transitions[prefix[a], prefix[b]]
                                                )
                                            ):
                                                return False
                            return True

                        greedy = None
                        if not failed and len(plans) == 4:
                            greedy = {
                                s: plans[p][3][j]
                                for s, (p, j, index) in enumerate(slots)
                            }
                        legacy_valid = greedy is not None and complete_check(greedy)
                        audit.update(
                            {
                                "selected_pairs": selected,
                                "ordered_domains": domains,
                                "capacity_groups": groups,
                                "pool_sha256": hashlib.sha256(
                                    json.dumps(domains, sort_keys=True).encode()
                                ).hexdigest(),
                                "greedy_assignment": greedy,
                                "greedy_valid": legacy_valid,
                                "slot_specs": slots,
                                "plan_specs": specs,
                                "constraint_scope": protocol["reuse"],
                                "selector_distance": protocol["selector_distance"],
                                "reason": None,
                            }
                        )
                        construction_started = time.perf_counter()
                        if greedy is not None and not legacy_valid:
                            audit.update(
                                feasibility="implementation_error",
                                reason="Greedy constructor returned an invalid complete block",
                                scored=False,
                            )
                            failures.append(dict(audit))
                            continue
                        try:
                            assignment, outcome = self._recover_assignments(
                                domains,
                                groups,
                                complete_check,
                                greedy if legacy_valid else None,
                                deadline,
                                protocol["constructor_node_cap"],
                            )
                            audit.update(outcome)
                            if assignment is not None and not complete_check(
                                assignment
                            ):
                                raise AssertionError(
                                    "Independent complete check failed"
                                )
                        except (AssertionError, ValueError, KeyError) as exc:
                            audit.update(
                                feasibility="implementation_error", reason=str(exc)
                            )
                            assignment = None
                        audit["elapsed_seconds"] = (
                            time.perf_counter() - construction_started
                        )
                        if assignment is None:
                            failures.append(dict(audit))
                            continue
                        selector = (
                            "legacy_greedy"
                            if legacy_valid
                            else "deterministic_recovery"
                        )
                        audit["selector"] = selector
                        audit["scored"] = False
                        plans = [
                            (
                                kind,
                                band,
                                indices,
                                [assignment[s] for s in domains if groups[s] == p],
                            )
                            for p, (kind, band, indices) in enumerate(specs)
                        ]
                        if time.perf_counter() >= deadline:
                            audit["reason"] = "feasible_but_scoring_censored"
                            continue
                        # Cross-plan reuse is allowed; restrict by the shared union.
                        removed = sorted(
                            {i for _, _, _, inserted in plans for i in inserted}
                        )
                        clean_restricted = self._result(
                            ids, clean_scores, target, sampled, removed
                        )
                        local = {}
                        for kind, band, indices, inserted in plans:
                            changed = list(prefix)
                            for index, item in zip(indices, inserted):
                                changed[index] = item
                            if (
                                len(changed) != n
                                or changed[-1] != prefix[-1]
                                or any(i == 0 for i in changed)
                            ):
                                raise AssertionError(
                                    "Replacement violated occupancy/terminal invariants"
                                )
                            altered_scores = self._catalog_scores(
                                changed, ids, protocol["catalog_chunk_size"]
                            )
                            altered = self._result(ids, altered_scores, target, sampled)
                            restricted = self._result(
                                ids, altered_scores, target, sampled, removed
                            )
                            old_ids = torch.as_tensor(
                                [prefix[i] for i in indices], device=self.device
                            )
                            new_ids = torch.as_tensor(inserted, device=self.device)
                            displacement = (
                                (
                                    self.model.item_emb(old_ids)
                                    - self.model.item_emb(new_ids)
                                )
                                .norm(dim=-1)
                                .cpu()
                                .tolist()
                            )
                            inserted_effects = []
                            for item in inserted:
                                before = self._result(
                                    ids, clean_scores, item, [], include_top10=False
                                )
                                after = self._result(
                                    ids, altered_scores, item, [], include_top10=False
                                )
                                inserted_effects.append(
                                    {
                                        "item": item,
                                        "score_gain": after["score"] - before["score"],
                                        "rank_gain": before["rank"] - after["rank"],
                                    }
                                )
                            metadata = {
                                **tag,
                                "replacement_type": kind,
                                "band": band,
                                "indices": indices,
                                "lags": [n - 1 - i for i in indices],
                                "original_ids": [prefix[i] for i in indices],
                                "replacement_ids": inserted,
                                "frequency_bins": [
                                    bin_count(frequency[prefix[i]]) for i in indices
                                ],
                                "reachable": [reach(i) for i in indices],
                                "embedding_l2": displacement,
                                "pool_sizes": pool_sizes,
                                "restricted_removed_ids": removed,
                                "inserted_item_effects": inserted_effects,
                                "matching_status": "independently checked complete block",
                                "assignment_selector": selector,
                                "construction_support": (
                                    "common" if legacy_valid else "exact_only"
                                ),
                            }
                            for scope, before, after in (
                                ("fixed", clean, altered),
                                ("restricted", clean_restricted, restricted),
                            ):
                                row = {
                                    **metadata,
                                    "catalog_scope": scope,
                                    "variant": f"{kind}/dose{dose}/{band}/{scope}",
                                    "catalog_size": before["catalog_size"],
                                    "sampled_negative_count": before[
                                        "margin_negatives"
                                    ],
                                    **effects(before, after),
                                }
                                rows.append(row)
                                local[kind, band, scope] = row
                            coverage["scored_replacement_plans"] += 1
                        for kind in protocol["replacement_types"]:
                            for scope in ("fixed", "restricted"):
                                recent, older = (
                                    local[kind, "recent", scope],
                                    local[kind, "older", scope],
                                )
                                contrast = dict(
                                    recent,
                                    variant=f"{kind}/dose{dose}/recent_minus_older/{scope}",
                                )
                                for metric in (
                                    "delta_rank",
                                    "delta_hr10",
                                    "delta_sampled_margin",
                                    "delta_target_score",
                                ):
                                    contrast[metric] = (
                                        recent[metric] - older[metric]
                                        if recent[metric] is not None
                                        and older[metric] is not None
                                        else None
                                    )
                                contrast["top10_overlap"] = None
                                rows.append(contrast)
                        audit["scored"] = True
                        audit["scored_plans"] = len(plans)
                for name in ("full_permutation", "terminal_preserving_permutation"):
                    changed = list(
                        prefix if name == "full_permutation" else prefix[:-1]
                    )
                    rng.shuffle(changed)
                    if name != "full_permutation":
                        changed += prefix[-1:]
                    if time.perf_counter() >= deadline:
                        coverage["censored_permutations"] += 1
                        continue
                    scores = self._catalog_scores(
                        changed, ids, protocol["catalog_chunk_size"]
                    )
                    result = self._result(ids, scores, target, sampled)
                    rows.append(
                        {
                            **common,
                            "variant": name,
                            "draw": 0,
                            "catalog_scope": "fixed",
                            "unchanged_history": changed == prefix,
                            **effects(clean, result),
                        }
                    )
                    coverage["scored_permutations"] += 1

        if time.perf_counter() < deadline:
            collapse = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        else:
            collapse = None
        if time.perf_counter() < deadline:
            margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                train_loader, num_batches=10
            )
        else:
            margin = dict.fromkeys(
                ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate")
            )
            breakdown = {"unavailable_reason": "category safety cap"}
        breakdown["availability"] = "Item categories unavailable"
        baseline_users = {}
        for row in baselines:
            baseline_users.setdefault(row["user"], []).append(row["hr10"])
        strata_results = {}
        for field in (
            "prefix_length",
            "target_popularity",
            "transition_frequency",
            "repeated_target",
        ):
            strata_results[field] = {
                value: self._aggregate([r for r in rows if r[field] == value])
                for value in sorted({r[field] for r in rows})
            }
        # Embedding displacement is measured, not claimed to be matched.
        # Stratify only after assigning fixed displacement edges.
        replacement_rows = [
            r
            for r in rows
            if "embedding_l2" in r and "recent_minus_older" not in r["variant"]
        ]
        strata_results["reachability_and_embedding_l2"] = {}
        for label in ("reachable", "unreachable", "unknown"):
            for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, float("inf"))):
                subset = [
                    r
                    for r in replacement_rows
                    if (
                        "unknown"
                        if any(v is None for v in r["reachable"])
                        else "reachable" if all(r["reachable"]) else "unreachable"
                    )
                    == label
                    and lo <= float(np.mean(r["embedding_l2"])) < hi
                ]
                if subset:
                    strata_results["reachability_and_embedding_l2"][
                        f"{label}/{lo}-{hi}"
                    ] = self._aggregate(subset)
        coverage["successful_prefixes"] = len(baselines)
        coverage["successful_replacement_users"] = len(
            {r["user"] for r in replacement_rows}
        )
        coverage["successful_replacement_prefixes"] = len(
            {(r["user"], r["cut"]) for r in replacement_rows}
        )
        coverage["paired_rows_including_contrasts"] = len(rows)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        records = list(assignment_records.values())
        for record in records:
            record.setdefault("scored", False)
            if record["reason"] == "not_started" and time.perf_counter() >= deadline:
                record["reason"] = "category_safety_cap_before_start"
        counts = Counter(r["feasibility"] for r in records)
        feasible = counts["certified_feasible"]
        unknown = counts["unresolved"] + counts["implementation_error"]
        denominator = len(records)
        user_lower = sum(
            any(
                r["feasibility"] == "certified_feasible"
                for r in records
                if r["user"] == u
            )
            for u in users
        )
        user_upper = sum(
            any(
                r["feasibility"] != "certified_infeasible"
                for r in records
                if r["user"] == u
            )
            for u in users
        )
        support_effects = {
            support: self._aggregate(
                [r for r in rows if r.get("construction_support") == support]
            )
            for support in ("common", "exact_only")
        }
        metrics = {
            "constructor_validation": constructor_check,
            "assignment_records": records,
            "construction_counts": dict(counts),
            "scoring_counts": {
                "complete_blocks": sum(r["scored"] for r in records),
                "feasible_unscored_blocks": sum(
                    r["feasibility"] == "certified_feasible" and not r["scored"]
                    for r in records
                ),
            },
            "block_feasibility_bounds_error_free": (
                [
                    feasible / (denominator - counts["implementation_error"]),
                    (feasible + counts["unresolved"])
                    / (denominator - counts["implementation_error"]),
                ]
                if denominator > counts["implementation_error"]
                else None
            ),
            "block_feasibility_bounds_including_errors": (
                [feasible / denominator, (feasible + unknown) / denominator]
                if denominator
                else None
            ),
            "user_feasibility_bounds": (
                [user_lower / len(users), user_upper / len(users)] if users else None
            ),
            "support_specific_effects": support_effects,
            "rescued_blocks": sum(
                r.get("selector") == "deterministic_recovery" for r in records
            ),
            "rescued_users": len(
                {
                    r["user"]
                    for r in records
                    if r.get("selector") == "deterministic_recovery"
                }
            ),
            "protocol": protocol,
            "coverage": dict(coverage),
            "architecture": architecture,
            "chunked_rank_synthetic_check": True,
            "identity_controls": controls,
            "matching_failures": failures,
            "embedding_collapse_score": collapse,
            "pos_neg_margin_mean": margin["mean"],
            "pos_neg_margin_p10": margin["p10"],
            "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
            "pos_neg_small_margin_rate": margin["small_margin_rate"],
            "pos_neg_margin_category_breakdown": breakdown,
            "training_prefix_hr10_user_mean": (
                float(np.mean([np.mean(v) for v in baseline_users.values()]))
                if baseline_users
                else None
            ),
            "intervention_effects": self._aggregate(
                [r for r in rows if r.get("construction_support") != "exact_only"]
            ),
            "strata": strata_results,
            "baseline_observations": baselines,
            "paired_observations": rows,
            "four_category_macro": None,
            "four_category_macro_reason": "One category ModelView; equal-category aggregation requires all four measured categories",
            "category_breakdown": {},
            "unavailable": [
                "review ratings/text",
                "item ratings/categories/prices/stores",
                "simulator traits",
                "held-out intervention effects",
                "exact active path counts",
                "lexical plausibility (not measured)",
                "causal user intent",
            ],
            "diagnostic_cost": {
                "synchronized_elapsed_seconds": elapsed,
                "device": str(self.device),
                "gpu_compute_seconds": None,
                "gpu_reserved_elapsed_seconds_upper_bound": (
                    elapsed if self.device.type == "cuda" else 0.0
                ),
                "accounting": "Elapsed includes CPU work; host GPU accounting is authoritative",
            },
        }
        definitions = {
            "constructor_validation": "Executed tiny exhaustive checks for feasible and infeasible scoped capacities, surplus values, empty domains, Hall deficiency, cross-plan reuse, joint rejection, deadlines, node caps and retained incumbents. No optimization is requested; checks do not prove arbitrary-instance correctness.",
            "assignment_records": "One record per planned block, including unstarted blocks. Stores frozen positions, ordered domains, slot/plan mapping, capacity groups, domain hash, checked assignment, proof type, counters, interruption and scoring status. Scoped Hall witnesses certify the capacity relaxation; exhaustive joint search can certify joint infeasibility. Positions not constructed remain unresolved. Constant-zero edge distance requires no fitted scorer.",
            "construction_counts": "Distinct planned blocks by certified_feasible, certified_infeasible, unresolved and implementation_error. A feasible block may have censored scoring; errors are never infeasibility evidence.",
            "scoring_counts": "Counts of fully scored four-plan blocks and certified-feasible blocks without completed scoring. Construction completion and scoring completion have different denominators.",
            "block_feasibility_bounds_error_free": "Identification bounds [F/N,(F+U)/N] excluding implementation errors, conditional on the planned finite-pool audit; unconstructed positions are unknown. Not sampling confidence intervals.",
            "block_feasibility_bounds_including_errors": "Conservative identification bounds over every planned block, treating unresolved searches and implementation errors as unknown; not confidence intervals.",
            "user_feasibility_bounds": "A user is feasible when at least one planned block is feasible. Lower bound requires a certified feasible block; upper bound also permits unresolved or error blocks. Not block fractions or confidence intervals.",
            "support_specific_effects": "Common support retains checked legacy assignments; exact-only support uses deterministic first-feasible recovery. Different populations and selector laws must be interpreted separately. Intervals resample supported user means conditional on observed completion; they do not resample construction statuses or account for selection uncertainty. Deterministic recovery has no random-assignment Monte Carlo interpretation.",
            "rescued_blocks": "Certified-feasible blocks recovered after legacy construction failure, including feasible blocks whose scoring was censored.",
            "rescued_users": "Distinct users with at least one recovered block, irrespective of scoring completion.",
            "protocol": "Versioned rules printed before effects; only diagnostic corruption pools use training transition counts. No auxiliary recommendation scores or model fitting.",
            "coverage": "Actual users, unique prefixes, plans and failures. Failed matching removes the entire four-plan block; absence is not zero effect. Doses can have different support.",
            "architecture": "Recognized raw-window NextItNet offsets intersect convolution reachability with the ten-item raw context for contiguous diagnostic histories. Unknown implementations disable structural interpretation. Structural accessibility is not learned influence; exact active path counts are unavailable.",
            "chunked_rank_synthetic_check": "Chunked score/rank tie counter verified against exhaustive synthetic sorting at several chunk sizes.",
            "identity_controls": "Repeated identical-input catalog score discrepancy and target-rank change; expected zero within numerical precision. Causality, all-padding and applicable unreachable checks are in architecture. These local controls do not prove calibration or CI coverage.",
            "matching_failures": "Auditable user/cut/dose/draw failures and pool sizes. A failed greedy assignment is an algorithmic failure, not proof that no valid assignment exists. No tolerance is relaxed; effects condition on successful assignments.",
            "embedding_collapse_score": "Sampled mean pairwise item cosine mapped to [0,1]. Near 0.5 means mean cosine near zero; alignment alone cannot establish collapse or recommendation quality.",
            "pos_neg_margin_mean": "Mean positive-minus-negative logit over valid aligned training-loader positions; in-sample sampled separation only.",
            "pos_neg_margin_p10": "Empirical lower tenth percentile of valid training-loader margins; null if unavailable.",
            "pos_neg_neg_beats_pos_rate": "Fraction of valid training positions where negative score exceeds positive score.",
            "pos_neg_small_margin_rate": "Fraction of valid training positions with margin below protocol eps.",
            "pos_neg_margin_category_breakdown": "Low/wrong-margin counts; category lists empty because item categories are unavailable.",
            "training_prefix_hr10_user_mean": "Training-target exact eligible-catalog HR@10 averaged within user then across users; not host validation or generalization.",
            "intervention_effects": "Changed-minus-clean ranks, HR@10, target score and fixed-negative mean margin; positive delta_rank is worse. Overlap is top-10 intersection/original top-10 size. Draws average within prefix, prefixes within user, then users equally. CI95 is exploratory percentile bootstrap of user means, null below two users. corruption_mc_se separately propagates within-prefix draw variance, conditional on successful matching and treating prefix draws as independent; null when any contributing prefix has fewer than two observations. More draws do not add independent users. recent_minus_older is a paired difference on the same successful four-plan blocks, with the same restricted catalog; never assume monotonic dose response.",
            "strata": "Fixed prefix, training-frequency, repeated-target, transition, reachability and displacement strata. Displacement bins are [0,.5), [.5,1), [1,infinity); displacement is measured, not matched. Compare bands only where both have support; intervals are exploratory without multiplicity or coverage guarantees.",
            "baseline_observations": "Clean exact ranks with fixed original full-prefix eligibility, target retained once even when repeated; catalog size and margin-negative count are denominators.",
            "paired_observations": "In-place replacements preserve occupancy, padding, terminal item, sequence length and timestamp positions. Fixed catalogs include inserted candidates. Restricted catalogs remove the same union of inserted IDs from all paired histories and both bands/types; they are not full-catalog ranks. Inserted score/rank gains refer to the fixed eligible catalog. Frequency bins and possible reachability match across bands; path counts and displacement do not. Permutations separately preserve the original item multiset. Neither score changes nor restricted-rank decomposition identify causal user intent.",
            "four_category_macro": "Unavailable in a single-category call; never substitute a pooled-user or single-category value. The pinned host macro validation HR@10 alone selects model trials.",
            "category_breakdown": "No item-category metadata is available; dataset routes remain unchanged.",
            "unavailable": "Explicit unobserved data or unmeasured quantities; missing evidence must not be invented.",
            "diagnostic_cost": "Synchronized wall time includes CPU work and all probes. GPU compute time is not separately measured; host search-budget accounting remains authoritative.",
        }
        # Review summary: statically checked ModelView calls, raw-window
        # architecture recognition, scoped capacities, Hall witnesses, incumbent
        # preservation, scoring flags and metric definitions. Tiny constructor
        # checks execute above during diagnosis; this patch was not executed
        # during code generation. Training and predictor computation are unchanged.
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
