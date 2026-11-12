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

### >>> Self_EvolveRec-BLOCK-START: Fixed benchmark data location and training seed
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
    ### >>> Self_EvolveRec-BLOCK-START: Prevent exhausted negative-support rejection loops
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    if not users:
        raise RuntimeError("no eligible users with available training negatives")
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
    ### >>> Self_EvolveRec-BLOCK-START: Deterministic local validation without changing training RNG
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound unique validation candidates by available support
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
    ### >>> Self_EvolveRec-BLOCK-START: Numerically stable full-prefix BPR on supported positions
    pos_scores, neg_scores = model(seq, pos, neg)
    valid = seq.ne(0) & pos.ne(0) & neg.ne(0)
    margins = (pos_scores - neg_scores)[valid]
    return nn.functional.softplus(-margins).sum() / valid.sum().clamp(min=1)
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
### >>> Self_EvolveRec-BLOCK-START: Enforce matched training limits
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

    ### >>> Self_EvolveRec-BLOCK-START: Padding-safe causal residual convolutions
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

    ### >>> Self_EvolveRec-BLOCK-START: Position-preserving compact auxiliary encoding and compatible scoring
    def encode_sequence(
        self, seq: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.size(1) < 1:
            raise ValueError("sequence must have nonempty B x L shape")
        batch, length = seq.shape
        if position_offset < 0 or position_offset + length > self.maxlen:
            raise ValueError("sequence positions exceed the configured maximum length")
        valid = seq.ne(0)
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        pos = (
            torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
            + position_offset
            + 1
        )
        pos = pos * valid.long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x) * mask
        for block in self.blocks:
            x = block(x, valid)
        return self.norm(x) * mask

    def log2feats(
        self, sequence: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        return self.encode_sequence(sequence, position_offset=position_offset)

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
        raise ValueError(f"unsupported scoring shapes: {features.shape}, {items.shape}")

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
        ### >>> Self_EvolveRec-BLOCK-START: Final nonpadding readout with explicit empty-history scores
        rows = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[rows, last]
        user_emb = user_emb * seq.ne(0).any(dim=1, keepdim=True).to(out.dtype)
        scores = self.predict_score(user_emb, candidates)
        return scores[0] if seq.size(0) == 1 else scores
        ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Weak recent-prefix boundary supervision with compact position-preserving encoding
class RecentBoundaryObjective:
    """Training-only auxiliary; deployment retains one full-prefix network."""

    def __init__(self, user_train, model):
        self.position_rng = np.random.default_rng(SEED)
        self.candidate_rng = np.random.default_rng(SEED)
        self.catalog = np.asarray(
            sorted(
                {
                    int(item)
                    for history in user_train.values()
                    for item in history
                    if 1 <= int(item) <= model.itemnum
                }
            ),
            dtype=np.int64,
        )
        self.receptive_field = 1 + sum(
            (conv.kernel_size[0] - 1) * conv.dilation[0]
            for block in model.blocks
            for conv in (block.conv1, block.conv2)
        )
        if self.receptive_field <= 1:
            raise ValueError("recent-prefix supervision requires temporal context")
        self.suffix_length = min(
            model.maxlen,
            8 if self.receptive_field > 8 else self.receptive_field - 1,
        )
        if self.suffix_length < 1:
            raise ValueError("suffix length must be positive")
        self.weight = 0.25
        self.temperature = 1.0
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "rows": 0,
            "short_or_invalid_rows": 0,
            "insufficient_catalog_rows": 0,
            "auxiliary_rows": 0,
            "cropped_input_items": 0,
            "boundary_margin_sum": 0.0,
            "auxiliary_loss_sum": 0.0,
        }

    def __call__(self, model, sequence, positive, device):
        crops, candidates = [], []
        length = sequence.shape[1]
        for row in range(sequence.shape[0]):
            self.stats["rows"] += 1
            prefix_lengths = np.cumsum(sequence[row] != 0)
            eligible = np.flatnonzero(
                (sequence[row] != 0)
                & (positive[row] != 0)
                & (prefix_lengths > self.suffix_length)
            )
            if not eligible.size:
                self.stats["short_or_invalid_rows"] += 1
                continue
            end = int(self.position_rng.choice(eligible))
            prefix = sequence[row, : end + 1]
            visible = prefix[prefix != 0]
            target = int(positive[row, end])
            # Only the visible causal prefix and current target affect exclusions.
            # Future training interactions never filter these candidates.
            forbidden_ids = np.union1d(visible, np.asarray([target]))
            blocked = np.searchsorted(self.catalog, forbidden_ids)
            if np.any(blocked >= self.catalog.size):
                raise ValueError("prefix or target outside training catalog")
            if not np.array_equal(self.catalog[blocked], forbidden_ids):
                raise ValueError("prefix or target outside training catalog")
            available = int(self.catalog.size - blocked.size)
            if available < 99:
                self.stats["insufficient_catalog_rows"] += 1
                continue
            # Uniform complement ranks, sampled without replacement, map to IDs
            # without allocating a complete per-row catalog mask.
            ranks = self.candidate_rng.choice(available, size=99, replace=False)
            shifted = blocked - np.arange(blocked.size, dtype=np.int64)
            indices = ranks + np.searchsorted(shifted, ranks, side="right")
            ids = np.concatenate((np.asarray([target]), self.catalog[indices]))
            recent = visible[-self.suffix_length :]
            crop = np.zeros(length, dtype=np.int64)
            crop[-recent.size :] = recent
            crops.append(crop)
            candidates.append(ids)
            self.stats["cropped_input_items"] += int(visible.size - recent.size)
        if not crops:
            return model.item_emb.weight.new_zeros(())
        # Every supported crop contains exactly suffix_length valid items.
        # Remove only leading padding and preserve its absolute position offset.
        # Masking inside every convolutional sublayer makes omitted padding zero.
        compact = np.stack(crops)[:, -self.suffix_length :].copy()
        crop_t = torch.as_tensor(compact, dtype=torch.long, device=device)
        ids_t = torch.as_tensor(np.stack(candidates), dtype=torch.long, device=device)
        features = model.log2feats(crop_t, position_offset=length - self.suffix_length)
        positions = torch.arange(crop_t.size(1), device=device).expand_as(crop_t)
        last = positions.masked_fill(crop_t.eq(0), -1).amax(dim=1)
        h = features[torch.arange(crop_t.size(0), device=device), last]
        scores = model.predict_score(h, ids_t).float()
        if scores.shape != ids_t.shape or not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("invalid auxiliary candidate scores")
        boundary = scores[:, 1:].topk(10, dim=1).values[:, -1]
        margins = scores[:, 0] - boundary
        # Keep the selected boundary attached to autograd; average once per row.
        loss = nn.functional.softplus(-margins / self.temperature).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite recent-prefix boundary loss")
        self.stats["auxiliary_rows"] += len(crops)
        self.stats["boundary_margin_sum"] += float(margins.detach().sum().item())
        self.stats["auxiliary_loss_sum"] += float(loss.detach().item()) * len(crops)
        return loss


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
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
    ### >>> Self_EvolveRec-BLOCK-START: Initialize independent auxiliary sampling streams
    auxiliary = RecentBoundaryObjective(user_train, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(
        f"  [{dataset_name}] main=BPR auxiliary=recent_prefix_boundary"
        f" weight={auxiliary.weight} temperature={auxiliary.temperature}"
        f" suffix={auxiliary.suffix_length} receptive_field={auxiliary.receptive_field}"
        f" auxiliary_negatives=99 exclusion=visible_truncated_prefix_and_target"
        f" seed={SEED} batch={BATCH_SIZE} maxlen={MAXLEN}",
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
        ### >>> Self_EvolveRec-BLOCK-START: Record auxiliary support per completed epoch
        total_loss = 0.0
        auxiliary.reset_stats()
        for _ in range(NUM_BATCHES):
            ### <<< Self_EvolveRec-BLOCK-END
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Preserve main BPR and add weak cropped boundary loss
            optimizer.zero_grad(set_to_none=True)
            main_loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            auxiliary_loss = auxiliary(model, seq, pos, device)
            loss = main_loss + auxiliary.weight * auxiliary_loss
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
            total_loss += float(loss.item())
            ### <<< Self_EvolveRec-BLOCK-END

        ### >>> Self_EvolveRec-BLOCK-START: Report measured training-only auxiliary statistics
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        print(
            f"  [{dataset_name}] epoch={epoch} auxiliary_support={auxiliary.stats}",
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
                ### >>> Self_EvolveRec-BLOCK-START: Record trained architecture and training-only intervention
                "dropout_rate": DROPOUT_RATE,
                "architecture": "padding_safe_nextitnet_boundary_v1",
                "training_objective": "BPR_plus_recent_prefix_boundary",
                "auxiliary_weight": auxiliary.weight,
                "auxiliary_temperature": auxiliary.temperature,
                "auxiliary_negatives": 99,
                "suffix_length": auxiliary.suffix_length,
                "receptive_field": auxiliary.receptive_field,
                "auxiliary_exclusions": "visible_truncated_causal_prefix_and_target",
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
        ### >>> Self_EvolveRec-BLOCK-START: Require category artifacts and matching exported architecture
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing NextItNet artifacts for {ds}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("architecture") != "padding_safe_nextitnet_boundary_v1":
            raise ValueError(f"Incompatible NextItNet architecture for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained configuration before evaluation mode
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
    ### >>> Self_EvolveRec-BLOCK-START: Explicit single-checkpoint category routing
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required for multiple category models")
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


