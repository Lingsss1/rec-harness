# === main_code.py ===
"""ID-only HSTU-style per-dataset template for Amazon Reviews.

Trains one sequential recommender per dataset and writes checkpoints under
``./working``.  The public benchmark contract remains unchanged:
``predict(user_id, history, candidates, *, dataset_name=None) -> list[float]``.

This is intentionally ID-only: no text/image/category metadata is loaded, so it
is directly comparable to SASRec/GRU4Rec/BERT4Rec per-dataset templates.
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
import torch.nn.functional as F

# ── Hyper-parameters ──────────────────────────────────────────────────
HIDDEN_UNITS = int(os.environ.get("HSTU_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("HSTU_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("HSTU_HEADS", "2"))
DROPOUT_RATE = float(os.environ.get("HSTU_DROPOUT", "0.3"))
LR = float(os.environ.get("HSTU_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("HSTU_WD", "1e-6"))
BATCH_SIZE = int(os.environ.get("HSTU_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Fixed-budget uniform multi-negative control
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get("HSTU_EPOCHS", "200"))))
NUM_NEGATIVES = 32
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("HSTU_MAXLEN", "128"))
PATIENCE = int(os.environ.get("HSTU_PATIENCE", "20"))
EVAL_USERS = int(os.environ.get("HSTU_EVAL_USERS", "200"))

### >>> Self_EvolveRec-BLOCK-START: Required host dataset boundary
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four category models
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Data loading ──────────────────────────────────────────────────────


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
    ### >>> Self_EvolveRec-BLOCK-START: Require public catalog statistics without reading test labels
    raise FileNotFoundError(
        f"Public itemnum for {dataset_name} is missing from {stats_path}"
    )
    ### <<< Self_EvolveRec-BLOCK-END


def load_dataset(dataset_name: str, data_dir: str):
    user_train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    user_valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    itemnum = _load_itemnum(data_dir, dataset_name)
    return user_train, user_valid, itemnum


# ── Batch sampler ─────────────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Expose training user identities without changing diagnostic batches
def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users: bool = False,
):
    ### <<< Self_EvolveRec-BLOCK-END
    users = [user for user, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No users with at least two training interactions")
    ### >>> Self_EvolveRec-BLOCK-START: Track sampled users for full-history exclusion
    batch_seq, batch_pos, batch_neg = [], [], []
    batch_users = []
    ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Bounded diagnostic sampling and separate training negatives
            if nxt != 0 and not return_users and len(interacted) < itemnum:
                for attempt in range(32):
                    neg_item = random.randint(1, itemnum)
                    if neg_item not in interacted:
                        neg[idx] = neg_item
                        break
                else:
                    eligible = [i for i in range(1, itemnum + 1) if i not in interacted]
                    if eligible:
                        neg[idx] = random.choice(eligible)
            ### <<< Self_EvolveRec-BLOCK-END
            nxt = item
            idx -= 1
            if idx < 0:
                break
        ### >>> Self_EvolveRec-BLOCK-START: Preserve default sampler compatibility
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
        batch_users.append(user)
    result = np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)
    if return_users:
        return (*result, batch_users)
    return result
    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Uniform-only 32-negative CE control with exact exclusions
class UniformNegativeSampler:
    """Distinct uniform negatives excluding each user's entire training history."""

    def __init__(self, user_train, itemnum, seed=42):
        self.itemnum = itemnum
        self.rng = np.random.default_rng(seed)
        self.histories = {
            user: np.asarray(
                sorted({i for i in history if 1 <= i <= itemnum}), dtype=np.int64
            )
            for user, history in user_train.items()
        }

    def sample(self, users, positives, count=32):
        result = np.zeros((*positives.shape, count), dtype=np.int64)
        for row, user in enumerate(users):
            positions = np.flatnonzero(positives[row] != 0)
            observed = self.histories[user]
            width = min(count, self.itemnum - len(observed))
            if not len(positions) or width <= 0:
                continue
            draws = self.rng.integers(
                1, self.itemnum + 1, size=(len(positions), width), dtype=np.int64
            )
            for attempt in range(8):
                invalid = np.isin(draws, observed)
                order = np.argsort(draws, axis=1, kind="stable")
                ordered = np.take_along_axis(draws, order, axis=1)
                repeated = np.zeros_like(invalid)
                repeated[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
                duplicate = np.zeros_like(invalid)
                np.put_along_axis(duplicate, order, repeated, axis=1)
                invalid |= duplicate
                if not invalid.any():
                    break
                if attempt < 7:
                    draws[invalid] = self.rng.integers(
                        1, self.itemnum + 1, size=int(invalid.sum()), dtype=np.int64
                    )
            if invalid.any():
                # Exact fallback only for unresolved rows. Accepted entries remain
                # unique, and all remaining entries are sampled without replacement.
                eligible = np.setdiff1d(
                    np.arange(1, self.itemnum + 1, dtype=np.int64),
                    observed,
                    assume_unique=True,
                )
                for local_row in np.flatnonzero(invalid.any(axis=1)):
                    keep = draws[local_row, ~invalid[local_row]]
                    pool = np.setdiff1d(eligible, keep, assume_unique=False)
                    draws[local_row, invalid[local_row]] = self.rng.choice(
                        pool, size=int(invalid[local_row].sum()), replace=False
                    )
            result[row, positions, :width] = draws
        return result


### <<< Self_EvolveRec-BLOCK-END

# ── HSTU-style model ──────────────────────────────────────────────────


class HSTUBlock(nn.Module):
    """A compact HSTU-style block: causal attention + SiLU gates + residual FFN."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(
            hidden_units,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        seq_emb: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        ### >>> Self_EvolveRec-BLOCK-START: Prevent fully masked left-padding attention rows
        normed = self.norm(seq_emb)
        blocked = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        # A padded query can attend to itself; its output is subsequently zeroed.
        # Valid queries still cannot attend to padding or future positions.
        blocked.diagonal(dim1=-2, dim2=-1).fill_(False)
        blocked = blocked.repeat_interleave(self.attn.num_heads, dim=0)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=blocked,
            need_weights=False,
        )
        ### <<< Self_EvolveRec-BLOCK-END
        value, gate = self.gate(attn_out).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb


class HSTURec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
    ):
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError(
                f"hidden_units={hidden_units} must be divisible by num_heads={num_heads}"
            )
        self.hidden_units = hidden_units
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(hidden_units, num_heads, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.last_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        seq_emb = self.item_emb(seq) * math.sqrt(self.hidden_units)
        pos = (
            torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            + 1
        )
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        ### >>> Self_EvolveRec-BLOCK-START: Finite padding states and shared compatibility scorer
        causal_mask = torch.triu(
            torch.ones((maxlen, maxlen), dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        padding_mask = seq == 0
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            seq_emb = block(seq_emb, causal_mask, padding_mask)
            seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.last_norm(seq_emb).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.forward(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    def final_features(self, sequence: torch.Tensor) -> torch.Tensor:
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        final = (
            positions.unsqueeze(0).expand_as(sequence).masked_fill(sequence == 0, -1)
        )
        final = final.amax(dim=1)
        result = features[
            torch.arange(sequence.shape[0], device=sequence.device), final.clamp_min(0)
        ]
        return result.masked_fill((final < 0).unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END


# ── Validation ────────────────────────────────────────────────────────


def evaluate_val(
    model: HSTURec,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Deterministic validation without resetting training RNG
    val_rng = random.Random(42)
    val_users = [user for user in user_valid if user in user_train]
    if len(val_users) > num_users:
        val_users = val_rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            pos_items = user_valid[user]
            if not pos_items:
                continue
            target = pos_items[0]
            hist = user_train[user][-maxlen:]
            seq_arr = np.zeros(maxlen, dtype=np.int64)
            for idx, item in enumerate(reversed(hist)):
                seq_arr[maxlen - 1 - idx] = item
            interacted = set(user_train[user]) | set(pos_items)
            neg_pool = [
                item for item in range(1, itemnum + 1) if item not in interacted
            ]
            ### >>> Self_EvolveRec-BLOCK-START: Match the validation candidate count
            if len(neg_pool) > 99:
                neg_pool = val_rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Shared greatest-nonpadding readout and scorer
            user_emb = model.final_features(seq_t)[0]
            scores = model.predict_score(user_emb, cand_t)
            ### <<< Self_EvolveRec-BLOCK-END
            rank = int(torch.argsort(scores, descending=True).tolist().index(0))
            if rank < 10:
                hr10 += 1
                ndcg10 += 1.0 / math.log2(rank + 2)
            if rank < 20:
                hr20 += 1
                ndcg20 += 1.0 / math.log2(rank + 2)
    denom = len(val_users)
    return {
        "HR@10": hr10 / denom,
        "HR@20": hr20 / denom,
        "NDCG@10": ndcg10 / denom,
        "NDCG@20": ndcg20 / denom,
    }


# ── Training ──────────────────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### >>> Self_EvolveRec-BLOCK-START: Require training data for every routed category
    if itemnum <= 0 or not user_train:
        raise ValueError(f"[{dataset_name}] Missing training data or invalid itemnum")
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Independent seed-42 initialization per category
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    negative_sampler = UniformNegativeSampler(user_train, itemnum, seed=42)
    model = HSTURec(
        itemnum=itemnum,
        ### <<< Self_EvolveRec-BLOCK-END
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always select a checkpoint from this training run
    best_hr10 = -1.0
    ### <<< Self_EvolveRec-BLOCK-END
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Bound CE backward memory while preserving the uniform control
            seq, pos, _, users = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, return_users=True
            )
            negatives = negative_sampler.sample(users, pos, NUM_NEGATIVES)
            valid = (pos != 0) & (negatives != 0).any(axis=-1)
            if not valid.any():
                continue
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
            positives = torch.as_tensor(pos[valid], dtype=torch.long, device=device)
            negatives_t = torch.as_tensor(
                negatives[valid], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            features = model.log2feats(seq_t)[valid_t]
            feature_grads = torch.empty_like(features)
            loss_value = features.new_zeros(())
            # Release each candidate-embedding graph immediately. Accumulate the
            # direct scorer gradients, then propagate feature gradients through
            # the single causal forward pass before taking one optimizer step.
            for start in range(0, len(positives), 1024):
                stop = min(start + 1024, len(positives))
                feats = features[start:stop].detach().requires_grad_(True)
                items = torch.cat(
                    (positives[start:stop, None], negatives_t[start:stop]), dim=1
                )
                logits = model.predict_score(feats, items)
                logits = logits.masked_fill(items == 0, float("-inf"))
                labels = torch.zeros(len(feats), dtype=torch.long, device=device)
                chunk_loss = F.cross_entropy(logits, labels, reduction="sum") / len(
                    positives
                )
                if not torch.isfinite(chunk_loss):
                    raise FloatingPointError(
                        f"Nonfinite CE in {dataset_name}, epoch {epoch}"
                    )
                loss_value += chunk_loss.detach()
                chunk_loss.backward()
                feature_grads[start:stop].copy_(feats.grad)
                del feats, items, logits, labels, chunk_loss
            features.backward(feature_grads)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss_value)
            ### <<< Self_EvolveRec-BLOCK-END
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % 5 == 0:
            val_m = evaluate_val(
                model, user_train, user_valid, itemnum, device, MAXLEN, EVAL_USERS
            )
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
            if epoch - best_epoch >= PATIENCE * 5:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )
    ### >>> Self_EvolveRec-BLOCK-START: Prevent stale checkpoints when no scheduled validation ran
    if best_epoch == 0:
        # Short configured runs may finish before the first five-epoch check.
        # Save this run explicitly, regardless of any existing checkpoint file.
        val_m = evaluate_val(
            model, user_train, user_valid, itemnum, device, MAXLEN, EVAL_USERS
        )
        best_hr10 = val_m["HR@10"]
        best_epoch = epoch
        torch.save(model.state_dict(), ckpt_path)
    ### <<< Self_EvolveRec-BLOCK-END
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Record the actual single-model scoring configuration
                "maxlen": MAXLEN,
                "scorer": "raw_dot_product",
                "readout": "greatest_nonpadding_index",
                "training_loss": "uniform_candidate_ce",
                "num_negatives": NUM_NEGATIVES,
                "training_seed": 42,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10


# ── Main ──────────────────────────────────────────────────────────────


def write_predict_script(working_dir):
    script = '"""Per-dataset HSTU predict — routes by dataset_name kwarg."""\nfrom __future__ import annotations\n\nimport importlib.util\nimport os\nimport pickle\n\nimport torch\n\n_HERE = os.path.dirname(os.path.abspath(__file__))\n\nDEFAULT_DATASETS = [\n    "Movies_and_TV",\n    "Industrial_and_Scientific",\n    "Electronics",\n    "CDs_and_Vinyl",\n]\nDATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]\n\n_models: dict = {}\n_metas: dict = {}\n_device = None\n\n\ndef _load_model_class():\n    spec = importlib.util.spec_from_file_location(\n        f"hstu_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")\n    )\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n    return module.HSTURec\n\n\ndef _ensure_loaded() -> None:\n    global _device\n    if _models:\n        return\n    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    model_cls = _load_model_class()\n    working_dir = _HERE\n    for dataset_name in DATASETS:\n        meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")\n        ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")\n        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n            continue\n        with open(meta_path, "rb") as f:\n            meta = pickle.load(f)\n        _metas[dataset_name] = meta\n        model = model_cls(\n            itemnum=meta["itemnum"],\n            hidden_units=meta["hidden_units"],\n            num_blocks=meta["num_blocks"],\n            num_heads=meta["num_heads"],\n            dropout_rate=0.0,\n            maxlen=meta["maxlen"],\n        ).to(_device)\n        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))\n        model.eval()\n        _models[dataset_name] = model\n\n\ndef predict(user_id: int, history: list[int], candidates: list[int],\n            *, dataset_name: str | None = None) -> list[float]:\n    _ensure_loaded()\n    if dataset_name and dataset_name in _models:\n        model = _models[dataset_name]\n        meta = _metas[dataset_name]\n    elif _models:\n        first_dataset = next(iter(_models))\n        model = _models[first_dataset]\n        meta = _metas[first_dataset]\n    else:\n        return [0.0] * len(candidates)\n\n    maxlen = meta["maxlen"]\n    seq_arr = [0] * maxlen\n    for idx, item in enumerate(reversed(history[-maxlen:])):\n        seq_arr[maxlen - 1 - idx] = item\n    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)\n    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)\n    with torch.no_grad():\n        out = model(seq_t)\n        user_emb = out[0, -1]\n        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()\n    return scores\n'
    ### >>> Self_EvolveRec-BLOCK-START: Keep generated prediction aligned with the trained HSTU
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    script = script.replace("dropout_rate=0.0,", 'dropout_rate=meta["dropout_rate"],')
    script = script.replace(
        "    if dataset_name and dataset_name in _models:",
        "    if dataset_name is not None and dataset_name not in _models:\n"
        '        raise ValueError(f"Missing category model: {dataset_name}")\n'
        "    if dataset_name and dataset_name in _models:",
    )
    old_scoring = (
        "        out = model(seq_t)\n"
        "        user_emb = out[0, -1]\n"
        "        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()"
    )
    new_scoring = (
        "        user_emb = model.final_features(seq_t)[0]\n"
        "        scores = model.predict_score(user_emb, cand_t).cpu().tolist()"
    )
    if old_scoring not in script:
        raise RuntimeError("Generated predictor scoring template changed")
    script = script.replace(old_scoring, new_scoring)
    with open(os.path.join(working_dir, "predict.py"), "w") as stream:
        stream.write(script)
    ### <<< Self_EvolveRec-BLOCK-END


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(
        f"HSTU ID-only: hidden={HIDDEN_UNITS} blocks={NUM_BLOCKS} heads={NUM_HEADS} "
        f"dropout={DROPOUT_RATE} maxlen={MAXLEN}",
        flush=True,
    )
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    all_hr10: list[float] = []
    for dataset_name in DATASETS:
        print(f"\n{'=' * 55}", flush=True)
        print(f"Training {dataset_name}", flush=True)
        print(f"{'=' * 55}", flush=True)
        all_hr10.append(train_one_dataset(dataset_name, device, DATA_DIR))
    write_predict_script(working_dir)
    avg = sum(all_hr10) / max(len(all_hr10), 1)
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
    ### >>> Self_EvolveRec-BLOCK-START: Interpret paired diagnostics with explicit scope and missingness
    - Unavailable/null measurements are missing evidence, never healthy zeros.
    - Training-prefix measurements are in-sample, including their reserved targets; do not call them validation scores.
    - Interpret compacted context deletion as a joint evidence-and-position intervention. Anchor removal changes the final observed item.
    - Sensitivity does not establish causal intent, user traits, or guaranteed generalization.
    - Bootstrap intervals condition on fitted weights and fixed candidates. They exclude training-seed variation, candidate-sampling variation, and repeated-selection optimism.
    - Do not infer unavailable metadata, simulator traits, or a four-category macro result from a single-category report.
    - Raw cosine and representation rank alone do not establish HR@10 failure.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the host view and distinguish its trained module
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are explicitly unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Bounded raw cosine without a quadratic Gram matrix
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            n = min(int(self.network.itemnum), emb.shape[0] - 1)
            rng = np.random.default_rng(seed)
            ids = rng.choice(n, size=min(sample_size, n), replace=False) + 1
            indices = torch.as_tensor(ids, device=emb.device, dtype=torch.long)
            x = emb.index_select(0, indices).to(device="cpu", dtype=torch.float64)
            norms = x.norm(dim=-1)
            usable = torch.isfinite(x).all(dim=-1) & (norms > 1e-12)
            x = x[usable]
            if len(x) < 2:
                return None
            x = F.normalize(x, p=2, dim=-1, eps=1e-12)
            cosine = (x.sum(dim=0).square().sum() - len(x)) / (len(x) * (len(x) - 1))
            return float(cosine.clamp(-1, 1))

    ### <<< Self_EvolveRec-BLOCK-END

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Let the diagnosis wrapper manage and restore module mode
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: HSTU tensor input compatibility
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding negatives and nonfinite margins
            valid = (seq_t != 0) & (pos_t != 0) & (neg_t != 0) & torch.isfinite(margin)
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same valid positions for breakdowns
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize counts for unavailable item categories
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Empty margin samples are unavailable
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                "valid_positions": 0,
                "unavailable_reason": "no_finite_positions_with_real_positive_and_negative",
                ### <<< Self_EvolveRec-BLOCK-END
            }
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Disclose margin sample size
                "valid_positions": int(N),
                "unavailable_reason": None,
                "mean": round(float(m.mean().item()), 4),
                ### <<< Self_EvolveRec-BLOCK-END
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            ### >>> Self_EvolveRec-BLOCK-START: Explicit unavailable metadata breakdown
            "category_breakdown_status": "unavailable",
            "category_breakdown_reason": "item_categories_not_available_in_matched_task",
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

    ### >>> Self_EvolveRec-BLOCK-START: Bound paired HSTU diagnostics and preserve training RNG state
    CONDITIONS = (
        "intact",
        "suffix",
        "remove_recent_context",
        "remove_old_context",
        "remove_anchor",
    )
    CATEGORIES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )

    @staticmethod
    def _timestamp_span(values):
        if not values or any(value is None for value in values):
            return {"value": None, "units": None, "reason": "missing_timestamps"}
        try:
            parsed = np.asarray([float(value) for value in values], dtype=np.float64)
            if not np.isfinite(parsed).all():
                raise ValueError("nonfinite timestamp")
            span = float(parsed.max()) - float(parsed.min())
            if not np.isfinite(span):
                raise ValueError("nonfinite timestamp span")
            return {
                "value": span,
                "units": "source_numeric_units",
                "reason": None,
            }
        except (ValueError, TypeError, OverflowError):
            try:
                from datetime import datetime

                parsed = [
                    datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                    for v in values
                ]
                span = (max(parsed) - min(parsed)).total_seconds()
                return {"value": float(span), "units": "seconds", "reason": None}
            except (ValueError, TypeError, OverflowError):
                return {
                    "value": None,
                    "units": None,
                    "reason": "unparseable_timestamps",
                }

    @classmethod
    def _paired_summary(cls, records):
        if not records:
            return {
                "eligible_users": 0,
                "conditions": {},
                "unavailable_reason": "no_eligible_users",
            }
        rng = np.random.default_rng(42)
        n = len(records)
        # Same resampled user indices for every condition and statistic.
        indices = rng.integers(0, n, size=(2000, n))
        summary = {}
        for name in cls.CONDITIONS:
            rows = [r["conditions"][name] for r in records]
            values = np.asarray(
                [
                    [
                        r["hit"],
                        r["rank"],
                        r["delta_hr"],
                        r["delta_rank"],
                        r["top10_overlap"],
                    ]
                    for r in rows
                ],
                dtype=np.float64,
            )
            boot = values[indices].mean(axis=1)
            means = values.mean(axis=0)
            intervals = np.percentile(boot, [2.5, 97.5], axis=0)
            names = ("hr10", "rank", "delta_hr", "delta_rank", "top10_overlap")
            summary[name] = {
                "eligible_users": n,
                "affected_users": sum(int(r["affected"]) for r in rows),
                "estimates": {
                    key: {
                        "mean": float(means[j]),
                        "ci95": [float(intervals[0, j]), float(intervals[1, j])],
                    }
                    for j, key in enumerate(names)
                },
            }
        return {"eligible_users": n, "conditions": summary, "unavailable_reason": None}

    @classmethod
    def aggregate_categories(cls, reports):
        """Explicit aggregation of four reports from this trial; never read other trials."""
        missing = [c for c in cls.CATEGORIES if c not in reports]
        if missing:
            return {
                "value": None,
                "reason": "missing_category_reports",
                "missing": missing,
            }
        groups = {}
        for category in cls.CATEGORIES:
            probe = reports[category]["metrics"]["paired_recent_context"]
            if probe.get("category") != category:
                return {"value": None, "reason": "category_identity_not_verified"}
            groups[category] = probe["records"]
        if any(not rows for rows in groups.values()):
            return {"value": None, "reason": "category_without_eligible_users"}
        rng = np.random.default_rng(42)
        sampled = {
            c: rng.integers(0, len(rows), size=(2000, len(rows)))
            for c, rows in groups.items()
        }
        output = {}
        for condition in cls.CONDITIONS:
            means, bootstraps = [], []
            for category, rows in groups.items():
                values = np.asarray(
                    [
                        [
                            r["conditions"][condition][key]
                            for key in (
                                "hit",
                                "delta_hr",
                                "delta_rank",
                                "top10_overlap",
                            )
                        ]
                        for r in rows
                    ],
                    dtype=np.float64,
                )
                means.append(values.mean(axis=0))
                bootstraps.append(values[sampled[category]].mean(axis=1))
            mean = np.mean(means, axis=0)
            interval = np.percentile(np.mean(bootstraps, axis=0), [2.5, 97.5], axis=0)
            output[condition] = {
                key: {
                    "mean": float(mean[j]),
                    "ci95": [float(interval[0, j]), float(interval[1, j])],
                }
                for j, key in enumerate(
                    ("hr10", "delta_hr", "delta_rank", "top10_overlap")
                )
            }
        return {
            "value": output,
            "eligible_users": {c: len(v) for c, v in groups.items()},
            "scope": "in_sample_training_prefix",
            "reason": None,
        }

    @torch.no_grad()
    def _paired_recent_context(self):
        import os
        import hashlib

        histories = self.user_train.get("History", {})
        times = self.user_train.get("Time", {})
        category = getattr(self.model, "dataset_name", None)
        if category not in self.CATEGORIES:
            category = getattr(self.network, "dataset_name", None)
        if category not in self.CATEGORIES:
            category = None
        itemnum = int(self.network.itemnum)
        maxlen = int(self.network.maxlen)
        rng = np.random.default_rng(42)
        users = sorted(histories, key=lambda u: str(u))
        order = rng.permutation(len(users))
        reasons = Counter()
        records = []
        considered = 0
        selected_users = []
        unavailable_cases = []
        for index in order:
            # Cap attempted eligible cases, including nonfinite model outputs.
            # Otherwise failed scoring can trigger an unbounded replacement search
            # and make the selected population depend on the fitted model.
            if len(selected_users) >= 200:
                break
            considered += 1
            user = users[int(index)]
            full = [int(i) for i in histories[user]]
            if any(i < 1 or i > itemnum for i in full):
                reasons["invalid_training_item_id"] += 1
                continue
            history = full[:-1][-maxlen:]
            if len(history) < 3:
                reasons["insufficient_history"] += 1
                continue
            target = full[-1]
            excluded = set(full)
            available = itemnum - len(excluded)
            if available < 99:
                reasons["insufficient_catalog"] += 1
                continue
            # Rejection sampling is uniform without replacement and bounded.
            negatives = set()
            for _ in range(32):
                draws = rng.integers(1, itemnum + 1, size=256)
                for draw in draws:
                    item = int(draw)
                    if item not in excluded:
                        negatives.add(item)
                        if len(negatives) == 99:
                            break
                if len(negatives) == 99:
                    break
            if len(negatives) < 99:
                pool = np.asarray(
                    [
                        i
                        for i in range(1, itemnum + 1)
                        if i not in excluded and i not in negatives
                    ],
                    dtype=np.int64,
                )
                negatives.update(
                    map(int, rng.choice(pool, 99 - len(negatives), replace=False))
                )
            candidates = [target] + sorted(negatives)
            selected_users.append(str(user))
            n = len(history)
            k = min(5, (n - 1) // 2)
            positions = list(range(n))
            retained = {
                "intact": positions,
                "suffix": positions[-min(10, n) :],
                "remove_recent_context": positions[: n - 1 - k] + positions[-1:],
                "remove_old_context": positions[k:],
                "remove_anchor": positions[:-1],
            }
            raw_times = times.get(user, times.get(str(user), []))
            aligned_times = (
                list(raw_times[:-1][-maxlen:])
                if len(raw_times) == len(full)
                else [None] * n
            )
            sequence = np.zeros((len(self.CONDITIONS), maxlen), dtype=np.int64)
            for row, name in enumerate(self.CONDITIONS):
                items = [history[j] for j in retained[name]]
                sequence[row, -len(items) :] = items
            seq_t = torch.as_tensor(sequence, device=self.device, dtype=torch.long)
            features = self.model.log2feats(seq_t)
            positions_t = torch.arange(maxlen, device=seq_t.device).expand_as(seq_t)
            final = positions_t.masked_fill(seq_t == 0, -1).amax(dim=1)
            features = features[torch.arange(len(sequence), device=seq_t.device), final]
            candidate_t = torch.as_tensor(
                candidates, device=self.device, dtype=torch.long
            )
            candidate_t = candidate_t.unsqueeze(0).expand(len(sequence), -1)
            scores = (
                self.model.predict_score(features, candidate_t)
                .detach()
                .double()
                .cpu()
                .numpy()
            )
            if scores.shape != (len(self.CONDITIONS), 100):
                raise ValueError(
                    "Diagnostic predict_score must return [conditions, candidates]"
                )
            if not np.isfinite(scores).all():
                reasons["nonfinite_candidate_scores"] += 1
                unavailable_cases.append(
                    {
                        "user_id": str(user),
                        "target": target,
                        "history": history,
                        "timestamps": [
                            None if t is None else str(t) for t in aligned_times
                        ],
                        "candidates": candidates,
                        "reason": "nonfinite_candidate_scores",
                    }
                )
                continue
            candidate_ids = np.asarray(candidates)
            rankings = [np.lexsort((candidate_ids, -row)) for row in scores]
            intact_rank = int(np.flatnonzero(rankings[0] == 0)[0]) + 1
            intact_top = set(candidate_ids[rankings[0][:10]].tolist())
            conditions = {}
            for row, name in enumerate(self.CONDITIONS):
                keep = retained[name]
                rank = int(np.flatnonzero(rankings[row] == 0)[0]) + 1
                conditions[name] = {
                    "rank": rank,
                    "hit": int(rank <= 10),
                    "delta_rank": rank - intact_rank,
                    "delta_hr": int(rank <= 10) - int(intact_rank <= 10),
                    "top10_overlap": len(
                        intact_top & set(candidate_ids[rankings[row][:10]].tolist())
                    )
                    / 10.0,
                    "top10_item_ids": candidate_ids[rankings[row][:10]].tolist(),
                    "retained_indices": keep,
                    "retained_length": len(keep),
                    "timestamp_span": self._timestamp_span(
                        [aligned_times[j] for j in keep]
                    ),
                    "anchor_item": history[keep[-1]],
                    "anchor_preserved": keep[-1] == n - 1,
                    "repeat_target_present": target in [history[j] for j in keep],
                    "affected": keep != positions,
                }
            records.append(
                {
                    "category": category,
                    "user_id": str(user),
                    "target": target,
                    "original_length": n,
                    "full_training_length": len(full),
                    "history": history,
                    "timestamps": [
                        None if t is None else str(t) for t in aligned_times
                    ],
                    "timestamp_span": self._timestamp_span(aligned_times),
                    "repeat_target": target in history,
                    "k": k,
                    "candidates": candidates,
                    "conditions": conditions,
                }
            )
        result = {
            "category": category,
            "category_identity_available": category is not None,
            "scope": "in_sample_training_prefix",
            "seed": 42,
            "considered_users": considered,
            "selected_users": len(selected_users),
            "selected_user_ids": selected_users,
            "scored_users": len(records),
            "eligible_users": len(records),
            "unavailable_scoring_cases": unavailable_cases,
            "not_considered_users": len(users) - considered,
            "unavailable_users": int(sum(reasons.values())),
            "unavailable_reasons": dict(reasons),
            "records": records,
            "summary": self._paired_summary(records),
            "geometry_preserving_masks": {
                "value": None,
                "reason": "supplied_HSTU_has_no_explicit_position_timestamp_and_key_value_mask_interface",
            },
            "authorized_validation": {
                "value": None,
                "reason": "exact_host_cases_and_pinned_candidates_not_exposed",
            },
            "four_category_macro": {
                "value": None,
                "reason": "single_view_report; aggregate_categories_requires_all_four_explicit_category_reports",
            },
            "settings": {
                "maximum_users": 200,
                "maximum_paired_forward_batches": 200,
                "selection_failure_policy": "record_nonfinite_scores_without_replacement",
                "maximum_history": maxlen,
                "negatives": 99,
                "bootstrap_replicates": 2000,
                "tie_rule": "descending_score_then_ascending_item_id",
                "deletion": "compacted_without_backfill; joint_evidence_and_geometry_intervention",
                "temporal_model": "supplied_HSTU_does_not_consume_timestamps",
                "strata": "not_computed",
                "spectra": "disabled_to_bound_diagnostic_cost",
            },
        }
        # Fingerprint identifies this public training input, without consulting other trials.
        fingerprint = hashlib.sha256()
        for user in users:
            fingerprint.update(
                json.dumps([str(user), list(map(int, histories[user]))]).encode()
            )
        fingerprint.update(str(itemnum).encode())
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "working", "diagnostics"
        )
        os.makedirs(directory, exist_ok=True)
        label = category or ("unidentified_" + fingerprint.hexdigest()[:16])
        path = os.path.join(directory, label + "_paired_context.json")
        result["artifact_path"] = path
        with open(path, "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return result

    def run_full_diagnosis(self, train_loader):
        from itertools import islice

        previous_mode = self.network.training
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        device = torch.device(self.device)
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            # Diagnosis must not change subsequent training samples or dropout.
            # fork_rng restores CPU and the single model GPU even on exceptions.
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(42)
                np.random.seed(42)
                torch.random.default_generator.manual_seed(42)
                for index in cuda_devices:
                    torch.cuda.default_generators[index].manual_seed(42)
                self.network.eval()
                with torch.no_grad():
                    cosine = self.analyze_embeddings_collapse_cosine(
                        sample_size=2048, seed=42
                    )
                    margin_metrics, breakdown = (
                        self.analyze_pos_neg_margin_with_category_breakdown(
                            islice(train_loader, 10),
                            num_batches=10,
                            eps=0.5,
                            topk_categories=20,
                        )
                    )
                    paired = self._paired_recent_context()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            self.network.train(previous_mode)
        return {
            "metrics": {
                "embedding_mean_offdiagonal_cosine": cosine,
                "embedding_cosine_unavailable_reason": (
                    "fewer_than_two_finite_nonzero_sampled_rows"
                    if cosine is None
                    else None
                ),
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_valid_positions": margin_metrics["valid_positions"],
                "pos_neg_margin_unavailable_reason": margin_metrics[
                    "unavailable_reason"
                ],
                "pos_neg_margin_category_breakdown": breakdown,
                "paired_recent_context": paired,
            },
            "metric_definitions": {
                "embedding_mean_offdiagonal_cosine": (
                    "Raw mean off-diagonal cosine over up to 2048 uniformly sampled real-item rows, "
                    "seed 42; computed in float64 after excluding nonfinite rows or norms<=1e-12. "
                    "Uses (||sum(unit_rows)||^2-N)/(N*(N-1)); requires N>=2. "
                    "Directional similarity, not a calibrated collapse probability or evidence of HR failure."
                ),
                "pos_neg_margin_mean": "Mean positive-minus-negative score over finite nonpadding sequence/positive/negative positions.",
                "pos_neg_margin_p10": "Nearest-rank tenth percentile of those margins; null when no valid positions exist.",
                "pos_neg_neg_beats_pos_rate": "Fraction of valid margins below zero.",
                "pos_neg_small_margin_rate": "Fraction of valid margins below eps=0.5; score-scale dependent.",
                "pos_neg_margin_category_breakdown": "Item categories unavailable; category lists intentionally empty. Position counts remain measured.",
                "paired_recent_context": {
                    "selection": "Sorted string user IDs, seed-42 permutation; select at most 200 structurally eligible training users per category before inspecting scores. Nonfinite scoring cases are recorded without replacement and excluded from paired estimates; report selected and scored counts separately. Last training item is target; preceding history truncated once to model.maxlen. Requires at least three items.",
                    "scope": "All prefix results are in-sample: fitting may already have used the reserved target. Earlier target occurrences remain and are flagged. No validation reproduction without exact authorized host candidates.",
                    "conditions": "n=bounded length; k=min(5,(n-1)//2). Intact=history; suffix=last min(10,n); recent deletion=history[:n-1-k]+history[-1:]; old deletion=history[k:]; anchor deletion=history[:-1]. No backfill. Recent/old remove equal counts and preserve anchor; suffix is descriptive, and unchanged suffixes are unaffected. Anchor deletion changes readout item.",
                    "geometry": "Left-pad each retained history and read greatest nonzero index. Compaction changes absolute positions and evidence jointly. Faithful geometry-preserving masking unavailable. Existing attention normalization remains fixed. Original aligned timestamps are recorded; this HSTU has no temporal input.",
                    "candidates": "99 unique uniform real-item negatives without replacement, excluding every training item, padding and reserved IDs. Identical 100 candidates across conditions. Rank=1+preceding count under descending score, ascending item ID ties; this diagnostic tie rule is not a claim about host ties.",
                    "effects": "Hit=rank<=10; delta_rank=perturbed-intact rank; delta_hr=perturbed-intact hit. Top10 overlap=intersection with intact top10 divided by 10. All paired estimates use identical eligible users.",
                    "uncertainty": "2000 paired user bootstrap replicates, seed 42, same user indices across conditions; percentile 95% intervals. Conditional on fitted model, selected users and fixed candidates; excludes training-seed/candidate variation and repeated-selection optimism.",
                    "macro": "aggregate_categories accepts four explicitly identified reports from this trial, resamples within each category and equally averages category means. Missing or empty categories yield unavailable, never zero imputation. Single-view reports contain no purported macro estimate.",
                    "timestamps": "Span=max-min of surviving original timestamps. Numeric values retain source units; parseable ISO values use seconds. Missing or unparseable timestamps yield null spans.",
                    "cost": "At most 200 attempted paired forward batches of five sequences, exactly at most ten consumed margin batches, and 2048 sampled embeddings per category. Nonfinite scores do not trigger replacement forwards. No SVD, fitted auxiliary predictor, checkpoint change, or score fusion.",
                    "rng_isolation": "Python, NumPy global, Torch CPU and the model GPU RNG states are restored after diagnosis, including exceptions. Explicit independent loader generators or persistent worker state remain owned by the caller; use a dedicated diagnostic loader for those cases.",
                },
            },
        }

    ### <<< Self_EvolveRec-BLOCK-END
