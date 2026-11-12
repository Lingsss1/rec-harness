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
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched training epoch limit
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host-provided data directory
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


### >>> Self_EvolveRec-BLOCK-START: Retain explicit stable user provenance in internal training batches
def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, user_rows=None
):
    ### <<< Self_EvolveRec-BLOCK-END
    users = [u for u, items in user_train.items() if len(items) >= 2]
    ### >>> Self_EvolveRec-BLOCK-START: Collect provenance without changing sampling RNG
    batch_seq, batch_pos, batch_neg = [], [], []
    batch_users = []
    ### <<< Self_EvolveRec-BLOCK-END
    for _ in range(batch_size):
        ### >>> Self_EvolveRec-BLOCK-START: Associate every sampled sequence with its training user
        u = random.choice(users)
        if user_rows is not None:
            batch_users.append(user_rows[u])
        history = user_train[u]
        ### <<< Self_EvolveRec-BLOCK-END
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ### >>> Self_EvolveRec-BLOCK-START: Reject histories with no available negative item
        ts = set(history)
        if len(ts - {0}) >= itemnum:
            raise ValueError(f"Training user {u!r} has no unseen negative item")
        ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the public three-array sampler while enabling provenance
    result = (np.array(batch_seq), np.array(batch_pos), np.array(batch_neg))
    if user_rows is not None:
        return result + (np.asarray(batch_users, dtype=np.int64),)
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


### >>> Self_EvolveRec-BLOCK-START: Build distinct-user source and edge evidence with complete user exclusion
RESIDUAL_VARIANT = os.environ.get("SASREC_RESIDUAL_VARIANT", "feature")
if RESIDUAL_VARIANT not in {"baseline", "fixed", "global", "feature"}:
    raise ValueError(
        "SASREC_RESIDUAL_VARIANT must be baseline, fixed, global, or feature"
    )


def build_transition_counts(user_train, itemnum):
    counts = defaultdict(int)
    source_counts = np.zeros(itemnum + 1, dtype=np.float32)
    user_rows = {u: row for row, u in enumerate(sorted(user_train))}
    memberships = []
    for u in sorted(user_train):
        history = user_train[u]
        pairs = set()
        for i, j in zip(history, history[1:]):
            i, j = int(i), int(j)
            if i == 0 or j == 0:
                continue
            if not (1 <= i <= itemnum and 1 <= j <= itemnum):
                raise ValueError("Training item outside the declared catalog")
            pairs.add(i * (itemnum + 1) + j)
        sources = {key // (itemnum + 1) for key in pairs}
        for key in pairs:
            counts[key] += 1
        for source in sources:
            source_counts[source] += 1
        memberships.append((user_rows[u], pairs, sources))
    keys = sorted(counts)
    edge_rows = {key: row for row, key in enumerate(keys)}
    edge_stride = max(len(keys), 1)
    largest_key = max(len(user_rows), 1) * max(edge_stride, itemnum + 1)
    if largest_key >= 2**63:
        raise ValueError("User membership encoding exceeds int64")
    user_edges, user_sources = [], []
    for row, pairs, sources in memberships:
        user_edges.extend(row * edge_stride + edge_rows[key] for key in pairs)
        user_sources.extend(row * (itemnum + 1) + source for source in sources)
    return {
        "edge_keys": torch.tensor(keys, dtype=torch.long),
        "edge_counts": torch.tensor([counts[key] for key in keys], dtype=torch.float32),
        "source_support": torch.from_numpy(source_counts),
        "user_edge_keys": torch.tensor(sorted(user_edges), dtype=torch.long),
        "user_source_keys": torch.tensor(sorted(user_sources), dtype=torch.long),
    }, user_rows


### <<< Self_EvolveRec-BLOCK-END


class SASRec(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Configure serialized distinct-user evidence and bounded rank-16 scoring
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        residual_variant: str = "baseline",
        edge_keys=None,
        edge_counts=None,
        source_support=None,
        user_edge_keys=None,
        user_source_keys=None,
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

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        ### >>> Self_EvolveRec-BLOCK-START: Validate and serialize distinct-user evidence without perturbing backbone RNG
        if residual_variant not in {"baseline", "fixed", "global", "feature"}:
            raise ValueError("Unknown residual variant")
        self.residual_variant = residual_variant
        self.transition_rank = 16
        self.transition_window = 3
        if not (1 <= itemnum <= math.isqrt(2**63 - 1) - 1):
            raise ValueError("Catalog size exceeds the supported pair-key range")

        def tensor(value, dtype):
            if value is None:
                return torch.empty(0, dtype=dtype)
            return torch.as_tensor(value, dtype=dtype).detach().cpu().clone()

        keys = tensor(edge_keys, torch.long)
        counts = tensor(edge_counts, torch.float32)
        if keys.ndim != 1 or counts.shape != keys.shape:
            raise ValueError(
                "Transition tables must be aligned one-dimensional tensors"
            )
        if keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("Transition keys must be sorted and unique")
        if keys.numel():
            sources, targets = keys // (itemnum + 1), keys % (itemnum + 1)
            valid = (
                (sources >= 1)
                & (sources <= itemnum)
                & (targets >= 1)
                & (targets <= itemnum)
                & torch.isfinite(counts)
                & (counts > 0)
                & (counts == counts.round())
            )
            if not bool(valid.all()):
                raise ValueError("Invalid directed transition evidence")
        if source_support is None:
            if keys.numel():
                raise ValueError(
                    "Distinct-user source counts must be supplied explicitly"
                )
            support = torch.zeros(itemnum + 1)
        else:
            support = tensor(source_support, torch.float32)
        if (
            support.shape != (itemnum + 1,)
            or not bool(torch.isfinite(support).all())
            or bool((support < 0).any())
            or bool((support != support.round()).any())
            or support[0].item() != 0
        ):
            raise ValueError("Invalid distinct-user source counts")
        if keys.numel() and bool((counts > support[keys // (itemnum + 1)]).any()):
            raise ValueError("Edge support cannot exceed source support")
        self.register_buffer("edge_keys", keys)
        self.register_buffer("edge_counts", counts)
        self.register_buffer("source_support", support)
        for name, value in (
            ("user_edge_keys", user_edge_keys),
            ("user_source_keys", user_source_keys),
        ):
            membership = tensor(value, torch.long)
            if membership.ndim != 1 or bool((membership < 0).any()):
                raise ValueError("Invalid user membership table")
            if membership.numel() > 1 and not bool(
                (membership[1:] > membership[:-1]).all()
            ):
                raise ValueError("User memberships must be sorted and unique")
            self.register_buffer(name, membership)
        if residual_variant != "baseline" and keys.numel():
            if not self.user_edge_keys.numel() or not self.user_source_keys.numel():
                raise ValueError(
                    "Nonempty transition evidence requires complete user membership tables"
                )
        if residual_variant != "baseline":
            with torch.random.fork_rng(devices=[]):
                self.source_embedding = nn.Embedding(itemnum + 1, 16, padding_idx=0)
                self.target_embedding = nn.Embedding(itemnum + 1, 16, padding_idx=0)
                nn.init.normal_(self.source_embedding.weight, 0.0, 0.02)
                nn.init.normal_(self.target_embedding.weight, 0.0, 0.02)
                with torch.no_grad():
                    self.source_embedding.weight[0].zero_()
                    self.target_embedding.weight[0].zero_()
                self.scale_parameter = nn.Parameter(torch.tensor(-4.0))
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Keep the original SASRec backbone unchanged
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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid computing unused attention weights
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

    ### >>> Self_EvolveRec-BLOCK-START: Share causal scoring and batch valid training targets with explicit user exclusion
    def log2feats(self, seq: torch.Tensor, user_rows=None) -> torch.Tensor:
        h = self._backbone_feats(seq)
        if self.residual_variant == "baseline":
            return h
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        latest = positions.masked_fill(seq == 0, -1).cummax(dim=1).values
        cursor = latest
        sources = []
        for _ in range(self.transition_window):
            sources.append(
                seq.gather(1, cursor.clamp_min(0)).masked_fill(cursor < 0, 0)
            )
            previous = latest.gather(1, (cursor - 1).clamp_min(0))
            cursor = previous.masked_fill(cursor <= 0, -1)
        source_ids = torch.stack(sources, dim=-1)
        if user_rows is None:
            rows = torch.full_like(seq, -1)
        else:
            user_rows = torch.as_tensor(user_rows, dtype=torch.long, device=seq.device)
            if user_rows.shape != (seq.shape[0],) or bool((user_rows < 0).any()):
                raise ValueError(
                    "One explicit nonnegative user row is required per sequence"
                )
            if bool((user_rows >= 2**53).any()):
                raise ValueError(
                    "User row cannot be represented exactly in feature payload"
                )
            rows = user_rows[:, None].expand_as(seq)
        # Tensor compatibility payload: backbone[D], recent source IDs[3], user row.
        # Row -1 explicitly denotes full-count scoring, including public diagnostics.
        return torch.cat(
            (
                h.to(torch.float64),
                source_ids.to(torch.float64),
                rows.unsqueeze(-1).to(torch.float64),
            ),
            dim=-1,
        )

    @staticmethod
    def _contains(keys, query):
        if keys.numel() == 0:
            return torch.zeros_like(query, dtype=torch.bool)
        location = torch.searchsorted(keys, query.contiguous())
        safe = location.clamp(max=keys.numel() - 1)
        return (location < keys.numel()) & (keys[safe] == query)

    def _source_support(self, source_ids, rows):
        support = self.source_support[source_ids]
        member = self._contains(
            self.user_source_keys,
            rows.clamp_min(0) * (self.itemnum + 1) + source_ids,
        )
        excluded = member & (rows >= 0) & (source_ids != 0)
        return (support - excluded.to(support.dtype)).clamp_min(0)

    def _edge_support(self, source_ids, items, rows=None):
        source_ids, items = torch.broadcast_tensors(source_ids, items)
        if self.edge_keys.numel() == 0:
            return torch.zeros_like(items, dtype=self.edge_counts.dtype)
        query = source_ids * (self.itemnum + 1) + items
        location = torch.searchsorted(self.edge_keys, query.contiguous())
        safe = location.clamp(max=self.edge_keys.numel() - 1)
        found = (
            (location < self.edge_keys.numel())
            & (self.edge_keys[safe] == query)
            & (source_ids != 0)
            & (items != 0)
        )
        count = torch.where(found, self.edge_counts[safe], 0.0)
        if rows is not None:
            membership_key = rows.clamp_min(0) * max(self.edge_keys.numel(), 1) + safe
            excluded = (
                self._contains(self.user_edge_keys, membership_key)
                & (rows >= 0)
                & found
            )
            count = (count - excluded.to(count.dtype)).clamp_min(0)
        return count

    def score_components(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        if features.ndim == 2 and items.ndim == 1:
            items = items.unsqueeze(0).expand(features.shape[0], -1)
        h = features[..., : self.hidden_units].to(self.item_emb.weight.dtype)
        extra_axes = items.ndim - (h.ndim - 1)
        if extra_axes < 0:
            raise ValueError("Candidates must preserve feature batch and position axes")
        aligned_h = h
        for _ in range(extra_axes):
            aligned_h = aligned_h.unsqueeze(-2)
        base = (aligned_h * self.item_emb(items)).sum(dim=-1)
        if self.residual_variant == "baseline":
            zero = torch.zeros_like(base)
            return {"backbone": base, "residual": zero, "support_coverage": zero}
        d = self.hidden_units
        source_ids = features[..., d : d + self.transition_window].to(torch.long)
        rows = features[..., d + self.transition_window].to(torch.long).unsqueeze(-1)
        valid = source_ids != 0
        recency = torch.pow(
            0.5,
            torch.arange(self.transition_window, device=features.device, dtype=h.dtype),
        )
        weights = valid.to(h.dtype) * recency
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        support = self._source_support(source_ids, rows)
        source_gate = support / (support + 5.0)
        for _ in range(extra_axes):
            source_ids = source_ids.unsqueeze(-2)
            rows = rows.unsqueeze(-2)
            weights = weights.unsqueeze(-2)
            source_gate = source_gate.unsqueeze(-2)
        count = self._edge_support(source_ids, items.unsqueeze(-1), rows)
        active = count >= 2
        edge_gate = active.to(h.dtype) * count / (count + 5.0)
        if self.residual_variant == "fixed":
            # Count-only control: same evidence, window, and bounded amplitude.
            affinity = torch.ones_like(count)
        else:
            source = self.source_embedding(source_ids)
            target = self.target_embedding(items).unsqueeze(-2)
            affinity = torch.tanh((source * target).sum(dim=-1) / 4.0)
        amplitude = 2.0 * torch.sigmoid(self.scale_parameter)
        residual = amplitude * (weights * source_gate * edge_gate * affinity).sum(
            dim=-1
        )
        return {
            "backbone": base,
            "residual": residual,
            "support_coverage": (weights * active.to(h.dtype)).sum(dim=-1),
        }

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        components = self.score_components(features, items)
        return components["backbone"] + components["residual"]

    def forward(self, seq: torch.Tensor, pos=None, neg=None, user_rows=None):
        if (pos is None) != (neg is None):
            raise ValueError("Positive and negative targets must be supplied together")
        if (
            self.training
            and pos is not None
            and self.residual_variant != "baseline"
            and user_rows is None
        ):
            raise ValueError("Residual training requires explicit user provenance")
        features = self.log2feats(seq, user_rows=user_rows)
        if pos is None:
            return features
        if pos.shape != seq.shape or neg.shape != seq.shape:
            raise ValueError("Training targets must match the B x L sequence shape")
        # Encode every causal prefix once, then score only supervised positions.
        # The explicit candidate axis shares source and membership work across
        # positive and negative targets without changing their BCE objective.
        valid = (seq != 0) & (pos != 0)
        targets = torch.stack((pos[valid], neg[valid]), dim=-1)
        scores = self.predict_score(features[valid], targets)
        pos_logits = scores.new_zeros(seq.shape).masked_scatter(valid, scores[:, 0])
        neg_logits = scores.new_zeros(seq.shape).masked_scatter(valid, scores[:, 1])
        return pos_logits, neg_logits

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        features = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1).clamp_min(0)
        final_features = features[torch.arange(seq.shape[0], device=seq.device), last]
        return self.predict_score(final_features, candidates)

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
    ### >>> Self_EvolveRec-BLOCK-START: Isolate deterministic validation sampling from training RNG
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
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when validation cannot supply 99 unique negatives
            interacted = set(user_train[u]) | set(pos_items)
            available = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available < 99:
                raise ValueError(
                    f"Validation user {u!r} has only {available} unseen candidates; "
                    "99 unique negatives are required"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            while len(neg_pool) < 99:
                ### >>> Self_EvolveRec-BLOCK-START: Keep validation candidates reproducible
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

    ### >>> Self_EvolveRec-BLOCK-START: Wire complete training evidence and stable user provenance into each dataset model
    evidence, user_rows = build_transition_counts(user_train, itemnum)
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        residual_variant=RESIDUAL_VARIANT,
        **evidence,
    ).to(device)
    print(
        f"  [{dataset_name}] residual={RESIDUAL_VARIANT} "
        f"training_edges={evidence['edge_keys'].numel()} "
        "training_support=leave-user-out evaluation_support=full-count",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
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
            ### >>> Self_EvolveRec-BLOCK-START: Apply user exclusion to both positive and negative training scores
            seq, pos, neg, rows = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, user_rows=user_rows
            )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            rows_t = torch.tensor(rows, dtype=torch.long, device=device)
            pos_logits, neg_logits = model(seq_t, pos_t, neg_t, user_rows=rows_t)
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Use valid causal source-target positions
            indices = (seq_t != 0) & (pos_t != 0)
            if not bool(indices.any()):
                raise ValueError(
                    "Training batch contains no supervised causal positions"
                )
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
                ### >>> Self_EvolveRec-BLOCK-START: Record the exact three-source evidence-gated scorer and provenance mapping
                "maxlen": MAXLEN,
                "residual_variant": RESIDUAL_VARIANT,
                "transition_rank": 16,
                "transition_window": 3,
                "transition_lambda": 5.0,
                "transition_kappa": 5.0,
                "transition_min_support": 2,
                "transition_max_scale": 2.0,
                "training_support": "distinct-user counts excluding the entire current user",
                "inference_support": "full distinct-user training counts",
                "user_rows": user_rows,
                "auxiliary_weight": 0.0,
                "item_indexing": "input integer IDs preserved; zero is padding",
                "feature_layout": "float64: state[D], recent_source_ids[3], user_row; -1 means full-count",
                "residual_implementation_version": 2,
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
        ### >>> Self_EvolveRec-BLOCK-START: Reject incomplete dataset exports explicitly
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing metadata or checkpoint for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        ### >>> Self_EvolveRec-BLOCK-START: Reload rank-16 scoring with complete distinct-user evidence and memberships
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if meta.get("residual_implementation_version") != 2:
            raise ValueError("Checkpoint requires the matching transition scorer version")
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            residual_variant=meta["residual_variant"],
            edge_keys=state["edge_keys"],
            edge_counts=state["edge_counts"],
            source_support=state["source_support"],
            user_edge_keys=state["user_edge_keys"],
            user_source_keys=state["user_source_keys"],
        )
        m.load_state_dict(state, strict=True)
        m = m.to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Preserve explicit dataset routing and reject missing models
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        raise RuntimeError("No trained dataset models were loaded")
    if len(candidates) == 0:
        return []
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Encode history once and chunk full-catalog residual scoring
    with torch.inference_mode():
        features = model.log2feats(seq_t)
        positions = torch.arange(seq_t.shape[1], device=_device).expand_as(seq_t)
        last = positions.masked_fill(seq_t == 0, -1).amax(dim=1).clamp_min(0)
        final_features = features[torch.arange(seq_t.shape[0], device=_device), last]
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor(
                [candidates[start:start + 4096]],
                dtype=torch.long, device=_device,
            )
            chunk = model.predict_score(final_features, cand_t)
            scores.extend(chunk[0].cpu().tolist())
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


