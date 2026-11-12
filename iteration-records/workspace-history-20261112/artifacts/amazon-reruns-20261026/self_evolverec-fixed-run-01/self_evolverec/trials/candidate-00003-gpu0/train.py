"""Per-dataset SASRec — trains one model per Amazon Reviews dataset.

Architecture (aligned with LLM-SRec SeqRec/sasrec, adapted for the RecHarness raw .txt data format):
  - One SASRec model per dataset: Movies_and_TV, Industrial_and_Scientific,
    Electronics, CDs_and_Vinyl
  - Checkpoints saved to ./working/{dataset}_model.pt
  - Official SASRec positional encoding: positions 0..maxlen-1 with padding masked after dropout
  - Official PointWiseFeedForward: Conv1d → Dropout → ReLU → Conv1d → Dropout + residual
  - Official BCEWithLogits loss on positive and negative sequence logits
  - Early stopping with patience=20 (checked every 5 epochs)
  - Writes predict.py routing to per-dataset models via dataset_name kwarg

Key hyper-parameters (all tunable by RecHarness):
  HIDDEN_UNITS : embedding / attention dimension
  NUM_BLOCKS   : number of self-attention blocks
  NUM_HEADS    : number of attention heads
  DROPOUT_RATE : dropout on attention + FFN
  LR           : Adam learning rate
  WEIGHT_DECAY : Adam weight decay
  BATCH_SIZE   : training batch size
  NUM_BATCHES  : gradient steps per epoch
  NUM_EPOCHS   : max training epochs
  MAXLEN       : max history length
  PATIENCE     : early stopping patience (epochs without val improvement)
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

# ── Hyper-parameters ──────────────────────────────────────────────────
HIDDEN_UNITS = int(os.environ.get("SASREC_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("SASREC_HEADS", "1"))
DROPOUT_RATE = float(os.environ.get("SASREC_DROPOUT", "0.5"))
LR = float(os.environ.get("SASREC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("SASREC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("SASREC_BATCH", "128"))
NUM_BATCHES = int(os.environ.get("SASREC_BATCHES", "200"))
### >>> Self_EvolveRec-BLOCK-START: Preserve the host epoch cap
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Required data boundary and explicit ablation configuration
DATA_DIR = os.environ["GAGC_DATA_DIR"]
VARIANT = os.environ.get("SASREC_VARIANT", "combined")
if VARIANT not in {"baseline", "gate_only", "loss_only", "combined", "fixed_gate"}:
    raise ValueError(f"Unsupported SASREC_VARIANT: {VARIANT}")
POOLING_ENABLED = VARIANT in {"gate_only", "combined", "fixed_gate"}
MULTI_NEGATIVE = VARIANT in {"loss_only", "combined", "fixed_gate"}
FIXED_GATE = VARIANT == "fixed_gate"
NUM_NEGATIVES = 64 if MULTI_NEGATIVE else 1
POOL_WINDOW = 5
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four benchmark datasets
DATASETS = list(DEFAULT_DATASETS)

random.seed(42)
### <<< Self_EvolveRec-BLOCK-END
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Data loading ──────────────────────────────────────────────────────


def _read_split(path: str) -> dict[int, list[int]]:
    d: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                d[int(parts[0])].append(int(parts[1]))
    return dict(d)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return stats[dataset_name]["itemnum"]
    # fallback: scan train file
    itemnum = 0
    path = os.path.join(data_dir, f"{dataset_name}_train.txt")
    if os.path.isfile(path):
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


# ── Batch sampler ─────────────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Exact uniform complement sampling without rejection or catalog-sized allocations
class TrainingSampler:
    def __init__(self, user_train, itemnum, num_negatives, seed=42):
        if itemnum <= 1:
            raise ValueError("Training requires at least two real catalog items")
        if num_negatives < 1:
            raise ValueError("num_negatives must be positive")
        self.user_train = user_train
        self.itemnum = itemnum
        self.num_negatives = num_negatives
        self.users = [u for u, items in user_train.items() if len(items) >= 2]
        if not self.users:
            raise ValueError("No users with valid next-item training positions")
        self.history_sets = {
            u: np.unique(np.asarray(user_train[u], dtype=np.int64)) for u in self.users
        }
        for u, excluded in self.history_sets.items():
            if excluded[0] < 1 or excluded[-1] > itemnum:
                raise ValueError(
                    f"Training history for user {u} contains invalid item IDs"
                )
        # For sorted exclusions e_i, e_i-i-1 counts eligible IDs below e_i.
        # Right-sided search maps a uniform complement rank to its real ID,
        # including runs of adjacent excluded IDs. Storage is O(history size).
        self.exclusion_offsets = {
            u: excluded - np.arange(len(excluded), dtype=np.int64) - 1
            for u, excluded in self.history_sets.items()
        }
        self.user_rng = random.Random(seed)
        self.negative_rng = np.random.default_rng(seed)
        self.omitted_positions = 0
        if not any(len(self.history_sets[u]) < itemnum for u in self.users):
            raise ValueError("No eligible negatives for any training user")

    def draw_negatives(self, u, count):
        available = self.itemnum - len(self.history_sets[u])
        if available <= 0:
            return None
        ranks = self.negative_rng.integers(0, available, size=count)
        skipped = np.searchsorted(self.exclusion_offsets[u], ranks, side="right")
        # This is a bijection onto the complement: draws remain independent
        # and uniform with replacement, with duplicate negatives preserved.
        return ranks + skipped + 1

    def sample(self, batch_size, maxlen):
        if batch_size < 1 or maxlen < 1:
            raise ValueError("batch_size and maxlen must be positive")
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, self.num_negatives), dtype=np.int64)
        for row in range(batch_size):
            u = self.user_rng.choice(self.users)
            history = self.user_train[u]
            length = min(len(history) - 1, maxlen)
            seq[row, -length:] = history[-length - 1 : -1]
            pos[row, -length:] = history[-length:]
            draws = self.draw_negatives(u, length * self.num_negatives)
            if draws is None:
                self.omitted_positions += length
            else:
                neg[row, -length:] = draws.reshape(length, self.num_negatives)
        if self.num_negatives == 1:
            neg = neg[..., 0]
        return seq, pos, neg


def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, sampler=None
):
    # Training supplies its persistent seed-42 sampler. Compatibility calls
    # use advancing global randomness instead of restarting the same batch.
    if sampler is None:
        sampler = TrainingSampler(user_train, itemnum, 1, seed=random.getrandbits(64))
    return sampler.sample(batch_size, maxlen)


### <<< Self_EvolveRec-BLOCK-END


# ── Model ─────────────────────────────────────────────────────────────


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        ### >>> Self_EvolveRec-BLOCK-START: Persisted pooling architecture options
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        pooling_enabled: bool = POOLING_ENABLED,
        pool_window: int = POOL_WINDOW,
        fixed_gate: bool = FIXED_GATE,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum

        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.item_emb.weight.data.normal_(0.0, 1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        ### >>> Self_EvolveRec-BLOCK-START: Initialize the recency gate without perturbing backbone or dropout RNG
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.pooling_enabled = pooling_enabled
        self.pool_window = int(pool_window)
        self.fixed_gate = fixed_gate
        if self.pool_window < 1:
            raise ValueError("pool_window must be positive")
        if pooling_enabled and not fixed_gate:
            with torch.random.fork_rng(devices=[]):
                self.recent_gate = nn.Linear(2 * hidden_units, 1)
            nn.init.zeros_(self.recent_gate.weight)
            nn.init.constant_(self.recent_gate.bias, -2.0)
        self.register_buffer("empty_history_scores", torch.zeros(itemnum + 1))
        ### <<< Self_EvolveRec-BLOCK-END

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)

        pos = torch.arange(M, device=seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_emb(pos)
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = seq == 0
        seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)
        attention_mask = ~torch.tril(
            torch.ones((M, M), dtype=torch.bool, device=seq.device)
        )

        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            queries = self.attention_layernorms[i](seq_emb)
            ### >>> Self_EvolveRec-BLOCK-START: Avoid materializing unused attention weights
            attn_out, _ = self.attention_layers[i](
                queries,
                seq_emb,
                seq_emb,
                attn_mask=attention_mask,
                need_weights=False,
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        ### >>> Self_EvolveRec-BLOCK-START: Causal five-position pooling with baseline embedding scaling and normalization
        features = self.last_layernorm(seq_emb)
        if not self.pooling_enabled:
            return features
        valid = (~timeline_mask).unsqueeze(-1)
        # SASRec scales item inputs by sqrt(hidden_units); pool the same scale
        # before positional encoding, excluding padding from both sums and counts.
        raw = self.item_emb(seq) * (self.hidden_units**0.5)
        raw = raw * valid
        prefix = torch.cat(
            (raw.new_zeros(B, 1, self.hidden_units), raw.cumsum(1)), dim=1
        )
        counts = valid.to(raw.dtype)
        count_prefix = torch.cat((raw.new_zeros(B, 1, 1), counts.cumsum(1)), dim=1)
        ends = torch.arange(1, M + 1, device=seq.device)
        starts = (ends - self.pool_window).clamp_min(0)
        recent = (prefix[:, ends] - prefix[:, starts]) / (
            count_prefix[:, ends] - count_prefix[:, starts]
        ).clamp_min(1)
        if self.fixed_gate:
            gate = features.new_tensor(-2.0).sigmoid()
        else:
            gate = self.recent_gate(torch.cat((features, recent), dim=-1)).sigmoid()
        return ((1.0 - gate) * features + gate * recent) * valid
        ### <<< Self_EvolveRec-BLOCK-END

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        log_feats = self.log2feats(seq)
        if pos is None or neg is None:
            return log_feats
        pos_embs = self.item_emb(pos)
        neg_embs = self.item_emb(neg)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        ### >>> Self_EvolveRec-BLOCK-START: Score both diagnostic single negatives and training multi-negatives
        if neg.ndim == seq.ndim + 1:
            neg_logits = (log_feats.unsqueeze(-2) * neg_embs).sum(dim=-1)
        else:
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
        ### <<< Self_EvolveRec-BLOCK-END
        return pos_logits, neg_logits

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(seq)
        ### >>> Self_EvolveRec-BLOCK-START: Greatest nonpadding position and deterministic training-frequency fallback
        positions = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1)
        final_feat = log_feats[
            torch.arange(seq.shape[0], device=seq.device), last.clamp_min(0)
        ]
        item_embs = self.item_emb(candidates)
        scores = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return torch.where(
            (last < 0).unsqueeze(-1), self.empty_history_scores[candidates], scores
        )
        ### <<< Self_EvolveRec-BLOCK-END


# ── Validation ────────────────────────────────────────────────────────


def evaluate_val(
    model: SASRec,
    user_train: dict,
    user_valid: dict,
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Isolate validation candidate randomness from training
    evaluation_rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        val_users = evaluation_rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END

    HR10 = HR20 = NDCG10 = NDCG20 = 0.0
    model.eval()
    with torch.no_grad():
        for u in val_users:
            pos_items = user_valid[u]
            if not pos_items:
                continue
            target = pos_items[0]

            hist = user_train[u][-maxlen:]
            seq_arr = np.zeros(maxlen, dtype=np.int32)
            for i, item in enumerate(reversed(hist)):
                seq_arr[maxlen - 1 - i] = item

            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            neg_pool: set[int] = set()
            interacted = set(user_train[u]) | set(pos_items)
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic validation sampling and explicit unsupported catalogs
            if itemnum - len(interacted) < 99:
                raise ValueError(
                    "Validation requires 99 unique eligible negative candidates"
                )
            while len(neg_pool) < 99:
                ni = evaluation_rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            scores = model.predict(seq_t, cand_t)[0].cpu().numpy().tolist()
            rank = sorted(range(len(candidates)), key=lambda i: -scores[i]).index(0)

            HR10 += 1.0 if rank < 10 else 0.0
            HR20 += 1.0 if rank < 20 else 0.0
            NDCG10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            NDCG20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0

    n = max(len(val_users), 1)
    model.train()
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    ### >>> Self_EvolveRec-BLOCK-START: Matched initialization, persistent sampler and training-only fallback
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        pooling_enabled=POOLING_ENABLED,
        pool_window=POOL_WINDOW,
        fixed_gate=FIXED_GATE,
    ).to(device)
    sampler = TrainingSampler(user_train, itemnum, NUM_NEGATIVES)
    frequencies = np.zeros(itemnum + 1, dtype=np.int64)
    for history in user_train.values():
        np.add.at(frequencies, np.asarray(history, dtype=np.int64), 1)
    ids = np.arange(1, itemnum + 1)
    order = ids[np.lexsort((ids, -frequencies[1:]))]
    fallback = np.full(itemnum + 1, -float(itemnum + 1), dtype=np.float32)
    fallback[order] = -np.arange(itemnum, dtype=np.float32)
    model.empty_history_scores.copy_(torch.as_tensor(fallback, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()
    # Heuristic gBCE coefficient: full-history exclusion changes the support
    # assumed by the theoretical calibration result.
    beta = 0.25 + 0.75 * NUM_NEGATIVES / (itemnum - 1)
    checkpoint_metrics = {}
    if NUM_EPOCHS < 1 or NUM_BATCHES < 1:
        raise ValueError("Training needs positive epoch and batch counts")
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Invoke identity-aware persistent negative sampling
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, sampler=sampler
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            ### >>> Self_EvolveRec-BLOCK-START: Correctly reduced generalized BCE over eligible positions
            indices = pos_t != 0
            indices = indices & (
                (neg_t[..., 0] != 0) if MULTI_NEGATIVE else (neg_t != 0)
            )
            if not indices.any():
                continue
            if MULTI_NEGATIVE:
                # Discard padded/ineligible positions before expanding K item
                # embeddings, reducing memory on short Amazon histories.
                features = model.log2feats(seq_t)[indices]
                positives = model.item_emb(pos_t[indices])
                pos_logits = (features * positives).sum(-1)
                negative_ids = neg_t[indices]
                # Chunk the position axis and checkpoint its embedding product
                # so backward does not retain all P x K x hidden activations.
                from torch.utils.checkpoint import checkpoint

                def negative_penalty(feats, item_ids):
                    embeddings = model.item_emb(item_ids)
                    logits = (feats.unsqueeze(1) * embeddings).sum(-1)
                    return torch.nn.functional.softplus(logits).sum(-1)

                negative_sums = []
                for start in range(0, features.shape[0], 256):
                    negative_sums.append(
                        checkpoint(
                            negative_penalty,
                            features[start : start + 256],
                            negative_ids[start : start + 256],
                            use_reentrant=False,
                        )
                    )
                loss = (
                    (
                        beta * torch.nn.functional.softplus(-pos_logits)
                        + torch.cat(negative_sums)
                    )
                    / (NUM_NEGATIVES + 1)
                ).mean()
            else:
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                pos_labels = torch.ones_like(pos_logits[indices])
                neg_labels = torch.zeros_like(neg_logits[indices])
                loss = bce_criterion(pos_logits[indices], pos_labels)
                loss = loss + bce_criterion(neg_logits[indices], neg_labels)
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Declared checkpoint schedule including the configured final epoch
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            ### <<< Self_EvolveRec-BLOCK-END
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}  val_score: {hr10:.4f}"
                f"  HR@20={val_m['HR@20']:.4f}  NDCG@10={val_m['NDCG@10']:.4f}",
                flush=True,
            )
            ### >>> Self_EvolveRec-BLOCK-START: Retain common-epoch candidates for macro validation selection
            checkpoint_metrics[epoch] = val_m
            torch.save(model.state_dict(), f"{ckpt_path}.epoch{epoch}")
            if best_epoch == 0 or hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
            print(
                f"  [{dataset_name}] omitted_negative_positions={sampler.omitted_positions}",
                flush=True,
            )
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    ### >>> Self_EvolveRec-BLOCK-START: Defer final artifact selection until every category is evaluated
    if not checkpoint_metrics:
        raise RuntimeError("No completed validation checkpoints")
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
                ### >>> Self_EvolveRec-BLOCK-START: Save the precise inference architecture and checkpoint measurements
                "maxlen": MAXLEN,
                "pooling_enabled": POOLING_ENABLED,
                "pool_window": POOL_WINDOW,
                "fixed_gate": FIXED_GATE,
                "embedding_scale": HIDDEN_UNITS**0.5,
                "score_scale": 1.0,
                "variant": VARIANT,
                "num_negatives": NUM_NEGATIVES,
            },
            f,
        )

    metrics_path = os.path.join(working_dir, f"{dataset_name}_validation.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "variant": VARIANT,
                "validation_source": "training-side evaluator; host evaluation is independent",
                "checkpoints": checkpoint_metrics,
                "omitted_negative_positions": sampler.omitted_positions,
            },
            f,
            indent=2,
        )
    return checkpoint_metrics
    ### <<< Self_EvolveRec-BLOCK-END


# ── Write unified predict.py ──────────────────────────────────────────


def write_predict_script(working_dir: str) -> None:
    script = '''"""Per-dataset SASRec predict — routes by dataset_name kwarg.

Supports both legacy signature:
    predict(user_id, history, candidates) -> list[float]
and per-dataset signature (used by the RecHarness harness when available):
    predict(user_id, history, candidates, *, dataset_name) -> list[float]
"""
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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four predictor routes
DATASETS = list(DEFAULT_DATASETS)

_models: dict = {}
### <<< Self_EvolveRec-BLOCK-END
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SASRec


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required dataset artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing required model artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained pooling and gate configuration
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            pooling_enabled=meta["pooling_enabled"],
            pool_window=meta["pool_window"],
            fixed_gate=meta["fixed_gate"],
        ).to(_device)
            ### <<< Self_EvolveRec-BLOCK-END
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit dataset routes
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"No trained model for dataset {dataset_name!r}")
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END
    elif _models:
        ds    = next(iter(_models))
        model = _models[ds]
        meta  = _metas[ds]
    else:
        return [0.0] * len(candidates)

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    with torch.no_grad():
        cand_t   = torch.tensor([candidates], dtype=torch.long, device=_device)
        scores   = model.predict(seq_t, cand_t)[0].cpu().tolist()

    return scores
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written → {predict_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)

    ### >>> Self_EvolveRec-BLOCK-START: Select a shared checkpoint epoch by equal-weight macro validation HR@10
    import shutil
    import time

    started = time.monotonic()
    results = {}
    for ds in DATASETS:
        print(f"Training {ds}; variant={VARIANT}", flush=True)
        results[ds] = train_one_dataset(ds, device, DATA_DIR)
        with open(os.path.join(working_dir, "search_progress.json"), "w") as f:
            json.dump(
                {
                    "variant": VARIANT,
                    "completed_categories": list(results),
                    "elapsed_seconds": time.monotonic() - started,
                    "complete": False,
                },
                f,
                indent=2,
            )

    common_epochs = sorted(set.intersection(*(set(results[ds]) for ds in DATASETS)))
    if not common_epochs:
        raise RuntimeError("No common completed checkpoint across all four categories")
    macro = {
        epoch: sum(results[ds][epoch]["HR@10"] for ds in DATASETS) / len(DATASETS)
        for epoch in common_epochs
    }
    # Ascending epochs and strict comparison preserve the earliest exact tie.
    selected_epoch = common_epochs[0]
    for epoch in common_epochs[1:]:
        if macro[epoch] > macro[selected_epoch]:
            selected_epoch = epoch
    for ds in DATASETS:
        final_path = os.path.join(working_dir, f"{ds}_model.pt")
        shutil.copyfile(f"{final_path}.epoch{selected_epoch}", final_path)
    write_predict_script(working_dir)
    with open(os.path.join(working_dir, "selection.json"), "w") as f:
        json.dump(
            {
                "variant": VARIANT,
                "selected_epoch": selected_epoch,
                "macro_validation_hr10": macro[selected_epoch],
                "checkpoint_macro_validation_hr10": macro,
                "category_metrics": {
                    ds: results[ds][selected_epoch] for ds in DATASETS
                },
                "selection_source": "training-side validation; independent host evaluator unchanged",
                "elapsed_seconds": time.monotonic() - started,
                "complete": True,
            },
            f,
            indent=2,
        )
    for ds in DATASETS:
        for epoch in results[ds]:
            os.remove(os.path.join(working_dir, f"{ds}_model.pt.epoch{epoch}"))
    print(
        f"Selected common epoch {selected_epoch}; macro validation HR@10={macro[selected_epoch]:.4f}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


