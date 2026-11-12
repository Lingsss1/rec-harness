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
### >>> Self_EvolveRec-BLOCK-START: Match the host epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the supplied training data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four benchmark categories
DATASETS = list(DEFAULT_DATASETS)
RESIDUAL_MODE = os.environ.get("SASREC_RESIDUAL_MODE", "transition")
if RESIDUAL_MODE not in {"transition", "popularity", "baseline"}:
    raise ValueError("SASREC_RESIDUAL_MODE must be transition, popularity, or baseline")
### <<< Self_EvolveRec-BLOCK-END

random.seed(42)
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


### >>> Self_EvolveRec-BLOCK-START: Carry sampled user folds without changing the default sampler contract
def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, user_folds=None
):
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    if not users:
        raise ValueError("No training users have a valid unseen negative item")
    batch_seq, batch_pos, batch_neg = [], [], []
    batch_folds = []
    ### <<< Self_EvolveRec-BLOCK-END
    for _ in range(batch_size):
        ### >>> Self_EvolveRec-BLOCK-START: Recover folds from the actual sampled user
        u = random.choice(users)
        if user_folds is not None:
            batch_folds.append(user_folds[u] + 1)
        history = user_train[u]
        ### <<< Self_EvolveRec-BLOCK-END
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in ts:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    ### >>> Self_EvolveRec-BLOCK-START: Preserve three-array callers and optionally return aligned folds
    result = (np.array(batch_seq), np.array(batch_pos), np.array(batch_neg))
    if user_folds is not None:
        return result + (np.asarray(batch_folds, dtype=np.int64),)
    return result
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
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
    ):
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

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid allocating unused attention weights
            attn_out, _ = self.attention_layers[i](
                queries, seq_emb, seq_emb, attn_mask=attention_mask, need_weights=False
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        return self.last_layernorm(seq_emb)

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
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(candidates)
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)


### >>> Self_EvolveRec-BLOCK-START: Cache exact cross-fitted edge evidence and mask padded positions


def build_transition_tables(user_train, itemnum):
    base = itemnum + 1
    users = np.asarray(sorted(user_train), dtype=np.int64)
    np.random.RandomState(0).shuffle(users)
    user_folds = {int(u): i % 5 for i, u in enumerate(users)}
    full_counts = defaultdict(int)
    excluded_counts = [defaultdict(int) for _ in range(5)]
    full_dest = np.zeros(base, dtype=np.float64)
    excluded_dest = np.zeros((5, base), dtype=np.float64)
    for u, history in user_train.items():
        fold = user_folds[u]
        for source, target in zip(history[:-1], history[1:]):
            if not (1 <= source <= itemnum and 1 <= target <= itemnum):
                raise ValueError("Training item ID is outside the supplied catalog")
            key = source * base + target
            full_counts[key] += 1
            excluded_counts[fold][key] += 1
            full_dest[target] += 1
            excluded_dest[fold, target] += 1

    edge_keys = np.asarray(sorted(full_counts), dtype=np.int64)
    edge_counts = np.asarray([full_counts[int(k)] for k in edge_keys], dtype=np.float32)
    sources = edge_keys // base
    all_keys, all_counts, all_rows, all_priors, all_dest = [], [], [], [], []
    for table_id in range(6):
        if table_id == 0:
            counts = edge_counts.copy()
            destinations = full_dest.copy()
        else:
            excluded = excluded_counts[table_id - 1]
            counts = edge_counts - np.asarray(
                [excluded.get(int(k), 0) for k in edge_keys], dtype=np.float32
            )
            destinations = full_dest - excluded_dest[table_id - 1]
        keep = counts > 0
        rows = np.bincount(sources[keep], weights=counts[keep], minlength=base)
        prior = (destinations + 1.0) / (destinations.sum() + itemnum)
        prior[0] = 1.0
        all_keys.append(edge_keys[keep] + table_id * base * base)
        all_counts.append(counts[keep])
        all_rows.append(rows.astype(np.float32))
        all_priors.append(prior.astype(np.float32))
        all_dest.append(destinations.astype(np.float32))

    tables = {
        "keys": np.concatenate(all_keys),
        "counts": np.concatenate(all_counts),
        "rows": np.stack(all_rows),
        "prior": np.stack(all_priors),
        "destinations": np.stack(all_dest),
    }
    # Table zero is the only table needed by validation, diagnostics and inference.
    full = {
        "keys": all_keys[0],
        "counts": all_counts[0],
        "rows": tables["rows"][:1].copy(),
        "prior": tables["prior"][:1].copy(),
        "destinations": tables["destinations"][:1].copy(),
    }
    return user_folds, tables, full


class TransitionSASRec(SASRec):
    def __init__(self, *args, residual_mode="transition", **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_mode = residual_mode
        self.transition_gate_logit = nn.Parameter(torch.tensor(-5.0))
        # Tables are serialized separately; checkpoints contain neural parameters and gate.
        for name in ("keys", "counts", "rows", "prior", "destinations"):
            dtype = torch.long if name == "keys" else torch.float32
            self.register_buffer(
                "transition_" + name, torch.empty(0, dtype=dtype), persistent=False
            )

        self.register_buffer(
            "transition_evidence",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )

    def set_transition_tables(self, tables):
        device = self.item_emb.weight.device
        for name in ("keys", "counts", "rows", "prior", "destinations"):
            dtype = torch.long if name == "keys" else torch.float32
            setattr(
                self,
                "transition_" + name,
                torch.as_tensor(tables[name], dtype=dtype, device=device),
            )
        # Counts, priors and smoothing are fixed, so edge evidence needs no
        # per-batch recomputation. Unstored edges have nonpositive lift and
        # therefore exactly zero evidence after the prescribed clamp.
        with torch.no_grad():
            base = self.itemnum + 1
            keys = self.transition_keys
            table = keys // (base * base)
            source = (keys // base) % base
            target = keys % base
            n = self.transition_rows[table, source]
            q = self.transition_prior[table, target]
            lift = torch.log((self.transition_counts / q + 10.0) / (n + 10.0)).clamp(
                0.0, 5.0
            )
            self.transition_evidence = n / (n + 10.0) * lift

    def residual(self, sequence, items, table_ids=None, final=False):
        if self.residual_mode == "baseline" or self.transition_rows.numel() == 0:
            return torch.zeros_like(items, dtype=self.item_emb.weight.dtype)
        if table_ids is None:
            table_ids = torch.zeros(
                sequence.shape[0], dtype=torch.long, device=sequence.device
            )
        table_ids = table_ids[:, None].expand_as(items)
        valid_items = (items > 0) & (items <= self.itemnum)
        if final:
            valid_items = valid_items & sequence.ne(0).any(dim=1, keepdim=True)
        else:
            valid_items = valid_items & sequence.ne(0)
        safe_items = items.clamp(0, self.itemnum)
        if self.residual_mode == "popularity":
            values = self.transition_destinations[table_ids, safe_items]
            return torch.log1p(values).clamp(max=5.0) * valid_items

        if self.transition_keys.numel() == 0:
            return torch.zeros_like(items, dtype=self.item_emb.weight.dtype)
        base = self.itemnum + 1
        length = sequence.shape[1]
        positions = torch.arange(length, device=sequence.device)[None, :]
        # Follow preceding nonpadding positions, including the current causal item.
        previous = torch.where(sequence != 0, positions, -1).cummax(dim=1).values
        cursor = previous[:, -1:] if final else previous
        numerator = torch.zeros_like(items, dtype=torch.float32)
        denominator = torch.zeros_like(items, dtype=torch.float32)
        for weight in (1.0, 0.5, 0.25):
            source = sequence.gather(1, cursor.clamp(min=0))
            source = torch.where(cursor >= 0, source, 0)
            if final:
                source = source.expand_as(items)
            supported_prefix = source != 0
            query = (table_ids * base + source) * base + safe_items
            index = torch.searchsorted(self.transition_keys, query.contiguous())
            index = index.clamp(max=self.transition_keys.numel() - 1)
            evidence = torch.where(
                self.transition_keys[index] == query,
                self.transition_evidence[index],
                0.0,
            )
            numerator = numerator + weight * evidence * supported_prefix
            denominator = denominator + weight * supported_prefix
            before = cursor - 1
            cursor = previous.gather(1, before.clamp(min=0))
            cursor = torch.where(before >= 0, cursor, -1)
        return numerator / denominator.clamp(min=1e-12) * valid_items

    def forward(self, seq, pos=None, neg=None, table_ids=None):
        if pos is None or neg is None:
            return super().forward(seq, pos, neg)
        positive, negative = super().forward(seq, pos, neg)
        gate = self.transition_gate_logit.sigmoid()
        return (
            positive + gate * self.residual(seq, pos, table_ids),
            negative + gate * self.residual(seq, neg, table_ids),
        )

    def predict(self, seq, candidates):
        neural = super().predict(seq, candidates)
        return neural + self.transition_gate_logit.sigmoid() * self.residual(
            seq, candidates, final=True
        )


class TransitionDiagnosisView:
    """Compatibility view carrying causal sequences beside neural features."""

    def __init__(self, model, user_train, meta_data=None, review_data=None):
        self.model = model
        self.item_emb = model.item_emb
        self.user_train = user_train
        self.meta_data = {} if meta_data is None else meta_data
        self.review_data = {} if review_data is None else review_data

    def log2feats(self, sequence):
        return self.model.log2feats(sequence), sequence

    def predict_score(self, features, items):
        neural, sequence = features
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            score = torch.matmul(neural, embeddings.transpose(0, 1))
            residual = (
                torch.stack(
                    [
                        self.model.residual(
                            sequence, torch.full_like(sequence, int(item))
                        )
                        for item in items
                    ],
                    dim=-1,
                )
                if items.numel()
                else score.new_empty(score.shape)
            )
        elif items.shape == sequence.shape:
            score = (neural * embeddings).sum(-1)
            residual = self.model.residual(sequence, items)
        else:
            score = embeddings.matmul(neural[:, -1, :].unsqueeze(-1)).squeeze(-1)
            residual = self.model.residual(sequence, items, final=True)
        return score + self.model.transition_gate_logit.sigmoid() * residual


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
    ### >>> Self_EvolveRec-BLOCK-START: Keep validation candidates deterministic without perturbing training randomness
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
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
            while len(neg_pool) < 99:
                ### >>> Self_EvolveRec-BLOCK-START: Use the isolated validation random stream
                ni = rng.randint(1, itemnum)
                ### <<< Self_EvolveRec-BLOCK-END
                if ni not in interacted:
                    neg_pool.add(ni)
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

    ### >>> Self_EvolveRec-BLOCK-START: Initialize matched neural models and training-only cross-fitted tables
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_folds, transition_tables, full_transition_table = build_transition_tables(
        user_train, itemnum
    )
    model = TransitionSASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        residual_mode=RESIDUAL_MODE,
    ).to(device)
    model.set_transition_tables(transition_tables)
    ### <<< Self_EvolveRec-BLOCK-END

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint even when every hit is zero
    best_hr10 = -float("inf")
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Apply identical excluded-user features to both sampled labels
            seq, pos, neg, folds = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, user_folds=user_folds
            )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            fold_t = torch.tensor(folds, dtype=torch.long, device=device)

            pos_logits, neg_logits = model(seq_t, pos_t, neg_t, table_ids=fold_t)
            ### <<< Self_EvolveRec-BLOCK-END
            indices = pos_t != 0
            pos_labels = torch.ones(pos_logits[indices].shape, device=device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
            loss = bce_criterion(pos_logits[indices], pos_labels)
            ### >>> Self_EvolveRec-BLOCK-START: Penalize the bounded residual gate
            loss = loss + bce_criterion(neg_logits[indices], neg_labels)
            if RESIDUAL_MODE != "baseline":
                loss = loss + 1e-4 * model.transition_gate_logit.sigmoid().square()
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        if epoch % 5 == 0:
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
            if epoch - best_epoch >= PATIENCE * 5:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        else:
            if epoch % 10 == 0:
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
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Serialize the full training-only scorer and fixed constants
                "maxlen": MAXLEN,
                "residual_mode": RESIDUAL_MODE,
                "transition_table": full_transition_table,
                "transition_constants": {
                    "smoothing": 10.0,
                    "lift_cap": 5.0,
                    "recent_weights": [1.0, 0.5, 0.25],
                    "fold_seed": 0,
                    "training_seed": 42,
                    "num_folds": 5,
                    "gate_max": 1.0,
                    "item_mapping": "Supplied integer IDs; zero is padding",
                    "negative_sampling": "Uniform unseen items excluding the entire training history",
                    "checkpoint_policy": "Existing validation schedule and patience",
                },
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10


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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four inference routes
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

_models: dict = {}
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ### >>> Self_EvolveRec-BLOCK-START: Load the trained combined architecture
    return mod.TransitionSASRec
    ### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            ### >>> Self_EvolveRec-BLOCK-START: Restore the same residual mode and training-only full table
            maxlen=meta["maxlen"],
            residual_mode=meta["residual_mode"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Restore sparse transition buffers separately from neural parameters
        m.set_transition_tables(meta["transition_table"])
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        ### <<< Self_EvolveRec-BLOCK-END
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit routes instead of scoring with another category
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"No trained model for dataset {dataset_name!r}")
    ### <<< Self_EvolveRec-BLOCK-END
    # Select model: use dataset_name kwarg if provided, else fall back to first available
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
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


### >>> Self_EvolveRec-BLOCK-START: Write a combined-model diagnostic adapter with explicit in-sample definitions


def write_diagnosis_script(trial_dir):
    source = """import torch


class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.device = device

    def run_full_diagnosis(self, train_loader):
        model = self.view.model
        was_training = model.training
        model.eval()
        count = 0
        neural_margin = fused_margin = residual_margin = 0.0
        try:
            with torch.no_grad():
                for _, sequence, positive, negative in train_loader:
                    sequence = sequence.to(self.device, dtype=torch.long)
                    positive = positive.to(self.device, dtype=torch.long)
                    negative = negative.to(self.device, dtype=torch.long)
                    valid = (sequence != 0) & (positive != 0) & (negative != 0)
                    features = model.log2feats(sequence)
                    neural_pos = (features * model.item_emb(positive)).sum(-1)
                    neural_neg = (features * model.item_emb(negative)).sum(-1)
                    # Omitted table IDs explicitly select the full training table zero.
                    rp = model.residual(sequence, positive)
                    rn = model.residual(sequence, negative)
                    boost = model.transition_gate_logit.sigmoid() * (rp - rn)
                    margin = neural_pos - neural_neg
                    count += int(valid.sum().item())
                    neural_margin += margin[valid].sum().item()
                    fused_margin += (margin + boost)[valid].sum().item()
                    residual_margin += boost[valid].sum().item()
        finally:
            model.train(was_training)
        definitions = {
            "valid_positions": "Number of nonpadding diagnostic training positions scored.",
            "neural_margin": "Mean positive-minus-negative neural logit on diagnostic training positions; in-sample.",
            "fused_margin": "Mean positive-minus-negative combined logit using the full training-only transition table; in-sample, not an unbiased generalization estimate.",
            "residual_margin": "Mean gated residual positive-minus-negative contribution on the same diagnostic positions; in-sample.",
            "gate": "Sigmoid of the learned scalar residual gate, bounded between zero and one.",
        }
        metrics = {
            "valid_positions": count,
            "neural_margin": neural_margin / count if count else None,
            "fused_margin": fused_margin / count if count else None,
            "residual_margin": residual_margin / count if count else None,
            "gate": float(model.transition_gate_logit.sigmoid().detach().cpu()),
        }
        return {"metrics": metrics, "metric_definitions": definitions}
"""
    with open(os.path.join(trial_dir, "diagnosis_tools.py"), "w") as handle:
        handle.write(source)


### <<< Self_EvolveRec-BLOCK-END

# ── Main ──────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)

    all_hr10: list[float] = []
    for ds in DATASETS:
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        hr10 = train_one_dataset(ds, device, DATA_DIR)
        all_hr10.append(hr10)

    ### >>> Self_EvolveRec-BLOCK-START: Emit matching inference and diagnostic implementations
    write_predict_script(working_dir)
    write_diagnosis_script(os.path.dirname(os.path.abspath(__file__)))
    ### <<< Self_EvolveRec-BLOCK-END

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


