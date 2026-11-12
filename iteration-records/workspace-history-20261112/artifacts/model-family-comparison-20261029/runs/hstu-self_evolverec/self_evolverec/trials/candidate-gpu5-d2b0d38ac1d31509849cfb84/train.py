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


### >>> Self_EvolveRec-BLOCK-START: Uniform complement-rank sampling without catalog scans
class UniformNegativeSampler:
    """Uniform distinct negatives excluding each user's entire training history.

    Sample ranks in the eligible catalog, then map ranks to item IDs with
    searchsorted. This avoids repeated history-membership tests and allocation
    of full-catalog fallback arrays. Candidate ordering does not affect CE.
    """

    def __init__(self, user_train, itemnum, seed=42):
        if itemnum <= 0:
            raise ValueError("itemnum must be positive")
        self.itemnum = itemnum
        self.rng = np.random.default_rng(seed)
        self.histories = {
            user: np.asarray(
                sorted({i for i in history if 1 <= i <= itemnum}), dtype=np.int64
            )
            for user, history in user_train.items()
        }
        # An excluded ID h shifts every eligible rank >= h - 1 - its index.
        self.thresholds = {
            user: observed - np.arange(1, len(observed) + 1, dtype=np.int64)
            for user, observed in self.histories.items()
        }

    def sample(self, users, positives, count=32):
        positives = np.asarray(positives)
        if positives.ndim != 2 or len(users) != positives.shape[0]:
            raise ValueError("Expected B x L positives and B user identities")
        if count <= 0:
            raise ValueError("Negative count must be positive")
        result = np.zeros((*positives.shape, count), dtype=np.int64)
        for row, user in enumerate(users):
            positions = np.flatnonzero(positives[row] != 0)
            available = self.itemnum - len(self.histories[user])
            width = min(count, available)
            if not len(positions) or width <= 0:
                continue
            if available <= 2 * width:
                # Dense draws use exact sampling without replacement directly.
                ranks = np.stack(
                    [
                        self.rng.choice(available, size=width, replace=False)
                        for _ in positions
                    ]
                )
            else:
                ranks = self.rng.integers(
                    0, available, size=(len(positions), width), dtype=np.int64
                )
                for attempt in range(8):
                    ranks.sort(axis=1)
                    duplicate = np.zeros_like(ranks, dtype=bool)
                    duplicate[:, 1:] = ranks[:, 1:] == ranks[:, :-1]
                    if not duplicate.any():
                        break
                    if attempt < 7:
                        ranks[duplicate] = self.rng.integers(
                            0, available, size=int(duplicate.sum()), dtype=np.int64
                        )
                    else:
                        # Replace unresolved rows with fresh exact uniform subsets.
                        for local_row in np.flatnonzero(duplicate.any(axis=1)):
                            ranks[local_row] = self.rng.choice(
                                available, size=width, replace=False
                            )
            draws = (
                ranks + 1 + np.searchsorted(self.thresholds[user], ranks, side="right")
            )
            result[row, positions, :width] = draws
        return result


### <<< Self_EvolveRec-BLOCK-END

# ── HSTU-style model ──────────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Correct position-only HSTU SiLU attention and gated aggregation
class HSTUBlock(nn.Module):
    """Causal HSTU with finite positional bias and multiplicative U gating.

    Temporal attention is deferred: the supplied sampler and public predictor
    provide item IDs without occurrence-aligned historical timestamps.
    """

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.norm = nn.LayerNorm(hidden_units)
        self.uvqk = nn.Linear(hidden_units, hidden_units * 4)
        # Signed distances are unnecessary because attention is inclusive causal.
        # Distances beyond 128 share the final learned positional bucket.
        self.position_bias = nn.Parameter(torch.zeros(num_heads, 129))
        self.aggregate_norm = nn.LayerNorm(hidden_units)
        self.output = nn.Linear(hidden_units, hidden_units)
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
        batch_size, length, hidden_units = seq_emb.shape
        u, v, q, k = F.silu(self.uvqk(self.norm(seq_emb))).chunk(4, dim=-1)

        def split_heads(tensor):
            return tensor.reshape(
                batch_size, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        positions = torch.arange(length, device=seq_emb.device)
        distances = (positions[:, None] - positions[None, :]).clamp(0, 128)
        bias = self.position_bias[:, distances]
        similarity = torch.matmul(q, k.transpose(-2, -1))
        similarity = similarity / math.sqrt(self.head_dim)
        # HSTU uses SiLU attention rather than softmax. Mask AFTER SiLU:
        # negative infinity before SiLU would create nonfinite values.
        weights = F.silu(similarity + bias.unsqueeze(0)) / max(length, 1)
        allowed = (
            (~causal_mask)[None, None, :, :]
            & (~padding_mask)[:, None, :, None]
            & (~padding_mask)[:, None, None, :]
        )
        weights = weights.masked_fill(~allowed, 0.0)
        aggregated = (
            torch.matmul(weights, v)
            .transpose(1, 2)
            .reshape(batch_size, length, hidden_units)
        )
        update = self.output(u * self.aggregate_norm(aggregated))
        seq_emb = seq_emb + self.dropout(update)
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)


### <<< Self_EvolveRec-BLOCK-END


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
                ### >>> Self_EvolveRec-BLOCK-START: Record position-only HSTU architecture and uniform-negative experiment
                "maxlen": MAXLEN,
                "architecture": "hstu_silu_position_v1",
                "attention": "silu_qk_sqrt_head_dim_div_sequence_length",
                "position_bias": "per_head_causal_distance_clipped_128",
                "temporal_inputs": False,
                "temporal_gate_reason": "Occurrence-aligned historical timestamps are unavailable in the supplied training and prediction interfaces",
                "scorer": "raw_dot_product",
                "readout": "greatest_nonpadding_index",
                "training_loss": "uniform_candidate_ce",
                "loss_scope": "sampled candidate classification; no full-catalog proposal correction",
                "num_negatives": NUM_NEGATIVES,
                "training_seed": 42,
                "configuration_count": 1,
                "maximum_epochs": NUM_EPOCHS,
                "batches_per_epoch": NUM_BATCHES,
                "validation_interval_epochs": 5,
                "early_stopping_patience_epochs": PATIENCE * 5,
                "selected_epoch": best_epoch,
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
    ### >>> Self_EvolveRec-BLOCK-START: Export the identical single-checkpoint position-only HSTU architecture
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    script = script.replace("dropout_rate=0.0,", 'dropout_rate=meta["dropout_rate"],')
    metadata_anchor = "        _metas[dataset_name] = meta"
    if metadata_anchor not in script:
        raise RuntimeError("Generated predictor metadata template changed")
    script = script.replace(
        metadata_anchor,
        '        if meta.get("architecture") != "hstu_silu_position_v1":\n'
        '            raise ValueError(f"Incompatible HSTU checkpoint: {dataset_name}")\n'
        '        if meta.get("temporal_inputs") is not False:\n'
        '            raise ValueError("This predictor requires position-only HSTU")\n'
        + metadata_anchor,
    )
    script = script.replace(
        "        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n"
        "            continue",
        "        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n"
        '            raise FileNotFoundError(f"Missing category checkpoint: {dataset_name}")',
    )
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
    # DEBUG: Export the actual HSTU backbone in predict.py. The dynamic import
    # hid its architecture from predictor inspection and depended on train.py.
    # Copy the trained classes directly so checkpoint keys and scoring agree.
    import inspect
    import textwrap

    script = script.replace("import importlib.util\n", "import math\n")
    script = script.replace(
        "import torch\n",
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    )
    loader_start = script.index("def _load_model_class():")
    loader_end = script.index("def _ensure_loaded()", loader_start)
    backbone_source = "\n\n".join(
        textwrap.dedent(inspect.getsource(cls)) for cls in (HSTUBlock, HSTURec)
    )
    script = (
        script[:loader_start]
        + backbone_source
        + "\n\ndef _load_model_class():\n    return HSTURec\n\n\n"
        + script[loader_end:]
    )
    # DEBUG: Check the standalone export before writing it; failures remain
    # visible instead of silently exporting an incompatible predictor.
    compile(script, os.path.join(working_dir, "predict.py"), "exec")
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


