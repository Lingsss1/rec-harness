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
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

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

### >>> Self_EvolveRec-BLOCK-START: Retain explicit training identities without changing sampling
TRANSITION_ANCHORS = int(os.environ.get("SASREC_TRANSITION_ANCHORS", "5"))
if TRANSITION_ANCHORS not in (0, 1, 5):
    raise ValueError("SASREC_TRANSITION_ANCHORS must be 0, 1, or 5")


def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users: bool = False,
):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No users have at least two training interactions")
    batch_users = []
    batch_seq, batch_pos, batch_neg = [], [], []
    ### <<< Self_EvolveRec-BLOCK-END
    for _ in range(batch_size):
        u = random.choice(users)
        ### >>> Self_EvolveRec-BLOCK-START: Carry the originating user through the private sampler
        history = user_train[u]
        batch_users.append(u)
        if len(set(history)) >= itemnum:
            raise ValueError("Training user has no eligible negative item")
        seq = np.zeros(maxlen, dtype=np.int32)
        ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the public sampler output by default
    batch = (np.array(batch_seq), np.array(batch_pos), np.array(batch_neg))
    if return_users:
        return (np.asarray(batch_users, dtype=np.int64),) + batch
    return batch
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


### >>> Self_EvolveRec-BLOCK-START: Sparse distinct-user statistics for bounded transition lift
TRANSITION_BUFFER_NAMES = (
    "edge_keys",
    "edge_counts",
    "source_mass",
    "target_mass",
    "total_mass",
    "training_users",
    "user_edge_keys",
    "user_source_keys",
    "user_source_values",
    "user_target_keys",
    "user_target_values",
    "user_mass",
)


def build_transition_statistics(user_train, itemnum):
    stride = itemnum + 1
    if itemnum < 1 or stride * stride > torch.iinfo(torch.int64).max:
        raise ValueError("Unsupported catalog size")
    users = sorted(user_train)
    edges_by_user = []
    counts = defaultdict(int)
    for uid in users:
        history = user_train[uid]
        if any(not 1 <= int(item) <= itemnum for item in history if item != 0):
            raise ValueError("Training item outside the declared catalog")
        edges = {
            int(i) * stride + int(j)
            for i, j in zip(history, history[1:])
            if i != 0 and j != 0
        }
        edges_by_user.append(edges)
        for key in edges:
            counts[key] += 1
    keys = sorted(counts)
    edge_index = {key: index for index, key in enumerate(keys)}
    edge_stride = max(len(keys), 1)
    if len(users) * max(edge_stride, stride) > torch.iinfo(torch.int64).max:
        raise ValueError("User statistics exceed int64 key capacity")
    source = torch.zeros(stride, dtype=torch.float32)
    target = torch.zeros(stride, dtype=torch.float32)
    for key in keys:
        source[key // stride] += counts[key]
        target[key % stride] += counts[key]
    membership, user_source, user_target, masses = [], {}, {}, []
    for row, edges in enumerate(edges_by_user):
        masses.append(len(edges))
        outdegree, indegree = defaultdict(int), defaultdict(int)
        for key in edges:
            membership.append(row * edge_stride + edge_index[key])
            outdegree[key // stride] += 1
            indegree[key % stride] += 1
        for item, degree in outdegree.items():
            user_source[row * stride + item] = degree
        for item, degree in indegree.items():
            user_target[row * stride + item] = degree
    source_keys, target_keys = sorted(user_source), sorted(user_target)
    return {
        "edge_keys": torch.tensor(keys, dtype=torch.long),
        "edge_counts": torch.tensor([counts[k] for k in keys], dtype=torch.float32),
        "source_mass": source,
        "target_mass": target,
        "total_mass": torch.tensor(sum(counts.values()), dtype=torch.float32),
        "training_users": torch.tensor(users, dtype=torch.long),
        "user_edge_keys": torch.tensor(sorted(membership), dtype=torch.long),
        "user_source_keys": torch.tensor(source_keys, dtype=torch.long),
        "user_source_values": torch.tensor(
            [user_source[k] for k in source_keys], dtype=torch.float32
        ),
        "user_target_keys": torch.tensor(target_keys, dtype=torch.long),
        "user_target_values": torch.tensor(
            [user_target[k] for k in target_keys], dtype=torch.float32
        ),
        "user_mass": torch.tensor(masses, dtype=torch.float32),
    }


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        transition_anchors: int = 0,
        transition_statistics=None,
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

        ### >>> Self_EvolveRec-BLOCK-START: Serialize exact statistics and cache immutable inference lift
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if transition_anchors not in (0, 1, 5):
            raise ValueError("Unsupported anchor count")
        self.transition_anchors = transition_anchors
        if transition_statistics is None:
            transition_statistics = build_transition_statistics({}, itemnum)
        for name in TRANSITION_BUFFER_NAMES:
            self.register_buffer(
                name, transition_statistics[name].detach().cpu().clone()
            )
        self.theta = nn.Parameter(
            torch.tensor(math.log(0.005 / 0.995)), requires_grad=transition_anchors != 0
        )
        # Rebuilt lazily after checkpoint loading, on the scorer's device.
        # This derived cache is deliberately absent from checkpoint state.
        self.register_buffer("_inference_lift", None, persistent=False)
        self.register_load_state_dict_post_hook(self._invalidate_lift_cache)

    @staticmethod
    def _invalidate_lift_cache(module, incompatible_keys):
        module._inference_lift = None

    def _backbone_feats(self, seq: torch.Tensor) -> torch.Tensor:
        ### <<< Self_EvolveRec-BLOCK-END
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

        return self.last_layernorm(seq_emb)

    ### >>> Self_EvolveRec-BLOCK-START: Cache full-training lift while preserving explicit user exclusion and causal scoring
    @staticmethod
    def _lookup(keys, query, values=None):
        if keys.numel() == 0:
            return torch.zeros_like(query, dtype=torch.float32)
        location = torch.searchsorted(keys, query.contiguous())
        safe = location.clamp_max(keys.numel() - 1)
        found = (location < keys.numel()) & (keys[safe] == query)
        if values is None:
            return found.float()
        return torch.where(found, values[safe], 0.0)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        hidden = self._backbone_feats(seq)
        if self.transition_anchors == 0:
            return hidden
        # Compact each prefix by interaction count, preserving repeated items.
        ranks = (seq != 0).long().cumsum(dim=1)
        compact = torch.zeros(
            (seq.shape[0], seq.shape[1] + 1), dtype=torch.long, device=seq.device
        )
        compact.scatter_add_(1, ranks, seq)
        anchors = []
        for age in range(self.transition_anchors):
            index = (ranks - age).clamp_min(0)
            anchors.append(compact.gather(1, index))
        anchors = torch.stack(anchors, dim=-1)
        packed = anchors.double()
        if not torch.equal(packed.long(), anchors):
            raise RuntimeError("Anchor IDs failed exact float64 packing")
        return torch.cat((hidden.double(), packed), dim=-1)

    def _user_rows(self, user_ids):
        if self.training_users.numel() == 0:
            raise ValueError("User-excluded scoring requires training identities")
        rows = torch.searchsorted(self.training_users, user_ids.contiguous())
        safe = rows.clamp_max(self.training_users.numel() - 1)
        if not bool(
            (
                (rows < self.training_users.numel())
                & (self.training_users[safe] == user_ids)
            ).all()
        ):
            raise ValueError("Unknown training user identity")
        return rows

    def _edge_lift(self, sources, items, user_rows=None):
        sources, items = torch.broadcast_tensors(sources, items)
        stride = self.itemnum + 1
        query = sources * stride + items
        if user_rows is None:
            # Counts are immutable between checkpoint loads. Build once rather
            # than recomputing probabilities for every catalog candidate chunk.
            if self._inference_lift is None:
                with torch.no_grad():
                    source_ids = self.edge_keys // stride
                    target_ids = self.edge_keys % stride
                    c_full = self.edge_counts
                    C_full = self.source_mass[source_ids]
                    p0_full = (self.target_mass[target_ids] + 1.0) / (
                        self.total_mass + self.itemnum
                    )
                    conditional_full = (c_full + 20.0 * p0_full) / (C_full + 20.0)
                    self._inference_lift = (
                        C_full
                        / (C_full + 10.0)
                        * c_full
                        / (c_full + 3.0)
                        * torch.log(conditional_full / p0_full).clamp(0.0, 2.0)
                    )
            value = self._lookup(self.edge_keys, query, self._inference_lift)
            return torch.where((sources != 0) & (items != 0), value, 0.0)
        c = self._lookup(self.edge_keys, query, self.edge_counts)
        C = self.source_mass[sources]
        T = self.target_mass[items]
        M = self.total_mass
        if user_rows is not None:
            user_rows = user_rows.expand_as(items)
            if self.edge_keys.numel():
                edge_index = torch.searchsorted(self.edge_keys, query.contiguous())
                safe = edge_index.clamp_max(self.edge_keys.numel() - 1)
                observed = (edge_index < self.edge_keys.numel()) & (
                    self.edge_keys[safe] == query
                )
                membership_key = user_rows * self.edge_keys.numel() + safe
                c = c - observed.float() * self._lookup(
                    self.user_edge_keys, membership_key
                )
            C = C - self._lookup(
                self.user_source_keys,
                user_rows * stride + sources,
                self.user_source_values,
            )
            T = T - self._lookup(
                self.user_target_keys,
                user_rows * stride + items,
                self.user_target_values,
            )
            M = M - self.user_mass[user_rows]
        p0 = (T + 1.0) / (M + self.itemnum)
        conditional = (c + 20.0 * p0) / (C + 20.0)
        lift = torch.log(conditional / p0).clamp(0.0, 2.0)
        value = C / (C + 10.0) * c / (c + 3.0) * lift
        return torch.where((c > 0) & (sources != 0) & (items != 0), value, 0.0)

    def _score(self, features, items, user_ids=None, fixed_amplitude=None):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        if features.ndim == 2 and items.ndim == 1:
            items = items.unsqueeze(0).expand(features.shape[0], -1)
        hidden = features[..., : self.hidden_units].to(self.item_emb.weight.dtype)
        extra_axes = items.ndim - (hidden.ndim - 1)
        if extra_axes < 0:
            raise ValueError("Candidates must preserve batch and position axes")
        # Reject accidental cross-user/position broadcasting. Shared final-state
        # candidate vectors were explicitly expanded above.
        feature_shape = tuple(hidden.shape[:-1])
        if tuple(items.shape[: len(feature_shape)]) != feature_shape:
            raise ValueError("Candidate batch and position axes must match features")
        aligned = hidden
        for _ in range(extra_axes):
            aligned = aligned.unsqueeze(-2)
        base = (aligned * self.item_emb(items)).sum(dim=-1)
        if self.transition_anchors == 0:
            return base
        user_rows = None
        if user_ids is not None:
            user_ids = torch.as_tensor(
                user_ids, dtype=torch.long, device=features.device
            )
            if user_ids.shape != (features.shape[0],):
                raise ValueError(
                    "Exactly one explicit user ID is required per sequence"
                )
            user_rows = self._user_rows(user_ids)
            user_rows = user_rows.reshape((-1,) + (1,) * (items.ndim - 1))
        anchors = features[..., self.hidden_units :].long()
        numerator = torch.zeros_like(base)
        denominator = torch.zeros_like(hidden[..., 0])
        for age in range(self.transition_anchors):
            source = anchors[..., age]
            weight = (2.0 ** (-age / 2.0)) * (source != 0).to(base.dtype)
            denominator = denominator + weight
            for _ in range(extra_axes):
                source = source.unsqueeze(-1)
                weight = weight.unsqueeze(-1)
            numerator = numerator + weight * self._edge_lift(source, items, user_rows)
        for _ in range(extra_axes):
            denominator = denominator.unsqueeze(-1)
        residual = numerator / denominator.clamp_min(1e-8)
        amplitude = 2.0 * torch.sigmoid(self.theta)
        if fixed_amplitude is not None:
            if not 0.0 <= float(fixed_amplitude) <= 2.0:
                raise ValueError("Diagnostic amplitude must lie in [0, 2]")
            amplitude = float(fixed_amplitude)
        return base + amplitude * residual

    def predict_score(self, features, items):
        # Public diagnostics and inference deliberately use full training counts.
        return self._score(features, items)

    def training_score(self, features, items, user_ids):
        return self._score(features, items, user_ids=user_ids)

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
        user_ids=None,
    ):
        features = self.log2feats(seq)
        if pos is None or neg is None:
            return features
        if self.training and self.transition_anchors and user_ids is None:
            raise ValueError("Training residuals require explicit user identities")
        if pos.shape != seq.shape or neg.shape != seq.shape:
            raise ValueError("Training targets must match the B x L sequence shape")
        # Resolve user rows and construct anchor weights once for both labels.
        logits = self._score(
            features, torch.stack((pos, neg), dim=-1), user_ids=user_ids
        )
        return logits[..., 0], logits[..., 1]

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        features = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1).clamp_min(0)
        final = features[torch.arange(seq.shape[0], device=seq.device), last]
        return self.predict_score(final, candidates)

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
    ### >>> Self_EvolveRec-BLOCK-START: Keep validation deterministic without perturbing training sampling
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
                ### >>> Self_EvolveRec-BLOCK-START: Use the isolated validation candidate RNG
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

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Invoke the count residual and protect its bounded amplitude from decay
        maxlen=MAXLEN,
        transition_anchors=TRANSITION_ANCHORS,
        transition_statistics=build_transition_statistics(user_train, itemnum),
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    p for name, p in model.named_parameters() if name != "theta"
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {"params": [model.theta], "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.98),
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint even at zero hits
    best_hr10 = -1.0
    ### <<< Self_EvolveRec-BLOCK-END
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Pass exact sampled identities to leave-user-out scoring
            user_ids, seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, return_users=True
            )
            user_ids_t = torch.tensor(user_ids, dtype=torch.long, device=device)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            ### >>> Self_EvolveRec-BLOCK-START: Optimize BCE using user-excluded statistics at valid causal positions
            pos_logits, neg_logits = model(seq_t, pos_t, neg_t, user_ids=user_ids_t)
            indices = (seq_t != 0) & (pos_t != 0)
            ### <<< Self_EvolveRec-BLOCK-END
            pos_labels = torch.ones(pos_logits[indices].shape, device=device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
            loss = bce_criterion(pos_logits[indices], pos_labels)
            loss = loss + bce_criterion(neg_logits[indices], neg_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        if epoch % 5 == 0:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### >>> Self_EvolveRec-BLOCK-START: Report paired full-count and excluded-count training diagnostics
            hr10 = val_m["HR@10"]
            prior_mode = model.training
            try:
                model.eval()
                with torch.no_grad():
                    # Reuse actual sampled identities, positions and negatives;
                    # public diagnostic batches cannot establish user exclusion.
                    diagnostic_size = min(16, seq_t.shape[0])
                    diagnostic_seq = seq_t[:diagnostic_size]
                    diagnostic_pos = pos_t[:diagnostic_size]
                    diagnostic_neg = neg_t[:diagnostic_size]
                    diagnostic_items = torch.stack(
                        (diagnostic_pos, diagnostic_neg), dim=-1
                    )
                    diagnostic_valid = (
                        (diagnostic_seq != 0)
                        & (diagnostic_pos != 0)
                        & (diagnostic_neg != 0)
                    )
                    diagnostic_features = model.log2feats(diagnostic_seq)
                    full_scores = model.predict_score(
                        diagnostic_features, diagnostic_items
                    )
                    excluded_scores = model.training_score(
                        diagnostic_features,
                        diagnostic_items,
                        user_ids_t[:diagnostic_size],
                    )
                    full_margin = full_scores[..., 0] - full_scores[..., 1]
                    excluded_margin = excluded_scores[..., 0] - excluded_scores[..., 1]
                    diagnostic_count = int(diagnostic_valid.sum().item())
                    diagnostic_record = {
                        "dataset": dataset_name,
                        "epoch": epoch,
                        "transition_amplitude": float(
                            (2.0 * torch.sigmoid(model.theta)).item()
                        ),
                        "last_update_theta_abs_gradient": (
                            float(model.theta.grad.detach().abs().item())
                            if model.theta.grad is not None
                            else None
                        ),
                        "paired_training_positions": diagnostic_count,
                        "full_training_count_margin_mean": (
                            float(full_margin[diagnostic_valid].mean().item())
                            if diagnostic_count
                            else None
                        ),
                        "user_excluded_count_margin_mean": (
                            float(excluded_margin[diagnostic_valid].mean().item())
                            if diagnostic_count
                            else None
                        ),
                        "full_training_count_negative_wins": (
                            float(
                                (full_margin[diagnostic_valid] < 0)
                                .float()
                                .mean()
                                .item()
                            )
                            if diagnostic_count
                            else None
                        ),
                        "user_excluded_count_negative_wins": (
                            float(
                                (excluded_margin[diagnostic_valid] < 0)
                                .float()
                                .mean()
                                .item()
                            )
                            if diagnostic_count
                            else None
                        ),
                        "interpretation": (
                            "Descriptive training fit on identical sampled positions; "
                            "full counts include the queried user. Neither margin is "
                            "a validation metric or an additional selection criterion."
                        ),
                    }
                    if not bool(torch.isfinite(full_scores).all()) or not bool(
                        torch.isfinite(excluded_scores).all()
                    ):
                        raise FloatingPointError(
                            "Nonfinite transition diagnostic scores"
                        )
                with open(
                    os.path.join(working_dir, f"{dataset_name}_diagnostics.jsonl"),
                    "a",
                ) as diagnostic_file:
                    diagnostic_file.write(json.dumps(diagnostic_record) + "\n")
            finally:
                model.train(prior_mode)
            # Static review covered causal anchors, candidate broadcasting,
            # explicit exclusion, cache reloads and shared predictor scoring.
            # Runtime parity and benchmark gains remain unverified here.
            ### <<< Self_EvolveRec-BLOCK-END
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
                ### >>> Self_EvolveRec-BLOCK-START: Record architecture and statistic semantics for exact reloads
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
                "transition_anchors": TRANSITION_ANCHORS,
                "transition_prior_strength": 20.0,
                "transition_source_shrinkage": 10.0,
                "transition_edge_shrinkage": 3.0,
                "transition_lift_clip": 2.0,
                "transition_amplitude_bound": 2.0,
                "transition_training_statistics": "entire originating user excluded",
                "transition_inference_statistics": "all training users included",
                "public_diagnostic_limitation": (
                    "Full-training statistics include queried users; training margins "
                    "measure descriptive fit rather than independent transition support."
                ),
                "feature_layout": "float64 backbone state followed by causal anchor IDs",
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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

_models: dict = {}
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
            raise FileNotFoundError(f"Missing trained artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        ### >>> Self_EvolveRec-BLOCK-START: Reload the exact sparse residual architecture in the exported predictor
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        buffer_names = (
            "edge_keys", "edge_counts", "source_mass", "target_mass", "total_mass",
            "training_users", "user_edge_keys", "user_source_keys", "user_source_values",
            "user_target_keys", "user_target_values", "user_mass",
        )
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            transition_anchors=meta["transition_anchors"],
            transition_statistics={name: state[name] for name in buffer_names},
        )
        m.load_state_dict(state, strict=True)
        m = m.to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Preserve explicit category routing and reject unknown routes
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown or unloaded dataset: {dataset_name!r}")
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

    ### >>> Self_EvolveRec-BLOCK-START: Reuse causal features during full-catalog chunked inference
    if not candidates:
        return []
    with torch.inference_mode():
        features = model.log2feats(seq_t)
        positions = torch.arange(seq_t.shape[1], device=_device).expand_as(seq_t)
        last = positions.masked_fill(seq_t == 0, -1).amax(dim=1).clamp_min(0)
        final = features[torch.arange(seq_t.shape[0], device=_device), last]
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor([candidates[start:start + 4096]],
                                  dtype=torch.long, device=_device)
            scores.extend(model.predict_score(final, cand_t)[0].cpu().tolist())
    return scores
    ### <<< Self_EvolveRec-BLOCK-END
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

    all_hr10: list[float] = []
    for ds in DATASETS:
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        hr10 = train_one_dataset(ds, device, DATA_DIR)
        all_hr10.append(hr10)

    write_predict_script(working_dir)

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


