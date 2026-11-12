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
### >>> Self_EvolveRec-BLOCK-START: Match the host training default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the authorized input directory
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

### >>> Self_EvolveRec-BLOCK-START: Dedicated sampling streams and distinct causal-prefix negatives
SAMPLED_NEGATIVES = 32
_SAMPLER_CACHE = None


def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, num_negatives: int = 1
):
    global _SAMPLER_CACHE
    if min(itemnum, batch_size, maxlen, num_negatives) < 1:
        raise ValueError("Sampling requires positive dimensions")
    if (
        _SAMPLER_CACHE is None
        or _SAMPLER_CACHE["source"] is not user_train
        or _SAMPLER_CACHE["itemnum"] != itemnum
    ):
        _SAMPLER_CACHE = {
            "source": user_train,
            "itemnum": itemnum,
            "users": [u for u, items in user_train.items() if len(items) >= 2],
            "histories": {},
            "user_rng": random.Random(42),
            "negative_rng": np.random.RandomState(42),
        }
    cache = _SAMPLER_CACHE
    if not cache["users"]:
        raise ValueError("Sampling requires users with at least two training events")
    rng = cache["negative_rng"]
    batch_seq = np.zeros((batch_size, maxlen), dtype=np.int32)
    batch_pos = np.zeros_like(batch_seq)
    batch_neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int32)
    for b in range(batch_size):
        user = cache["user_rng"].choice(cache["users"])
        if user not in cache["histories"]:
            history = np.asarray(user_train[user], dtype=np.int64)
            if np.any((history < 1) | (history > itemnum)):
                raise ValueError("Training history contains an out-of-catalog item ID")
            ids, first = np.unique(history, return_index=True)
            cache["histories"][user] = (history, ids, first, np.sort(first))
        history, ids, first, sorted_first = cache["histories"][user]
        start = max(0, len(history) - maxlen - 1)
        ends = np.arange(start, len(history) - 1)
        length = len(ends)
        targets = history[ends + 1]
        batch_seq[b, -length:] = history[ends]
        batch_pos[b, -length:] = targets
        # Full observed prefixes include events before the model's input window.
        target_first = first[np.searchsorted(ids, targets)]
        prefix_counts = np.searchsorted(sorted_first, ends, side="right")
        available = itemnum - prefix_counts - (target_first > ends)
        draws = np.zeros((length, num_negatives), dtype=np.int64)
        active = np.broadcast_to((available >= num_negatives)[:, None], draws.shape)
        invalid = active.copy()
        for _ in range(8):
            if not invalid.any():
                break
            draws[invalid] = rng.randint(1, itemnum + 1, size=int(invalid.sum()))
            locations = np.searchsorted(ids, draws)
            safe = np.minimum(locations, len(ids) - 1)
            seen = (ids[safe] == draws) & (first[safe] <= ends[:, None])
            invalid = active & (seen | (draws == targets[:, None]))
            order = np.argsort(draws, axis=1, kind="stable")
            ordered = np.take_along_axis(draws, order, axis=1)
            duplicate = np.zeros_like(invalid)
            duplicate[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
            duplicate_slots = np.zeros_like(invalid)
            np.put_along_axis(duplicate_slots, order, duplicate, axis=1)
            invalid |= active & duplicate_slots
        fallback = np.flatnonzero((available < num_negatives) | invalid.any(axis=1))
        if len(fallback):
            catalog = np.arange(1, itemnum + 1, dtype=np.int64)
            locations = np.searchsorted(ids, catalog)
            safe = np.minimum(locations, len(ids) - 1)
            present = ids[safe] == catalog
            for row in fallback:
                seen = present & (first[safe] <= ends[row])
                eligible = catalog[(~seen) & (catalog != targets[row])]
                count = min(num_negatives, len(eligible))
                draws[row] = 0
                if count:
                    draws[row, :count] = rng.choice(eligible, count, replace=False)
        batch_neg[b, -length:, :] = draws
    if num_negatives == 1:
        return batch_seq, batch_pos, batch_neg[..., 0]
    return batch_seq, batch_pos, batch_neg


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

        ### >>> Self_EvolveRec-BLOCK-START: Reduce initial sampled-softmax saturation
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        ### <<< Self_EvolveRec-BLOCK-END
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

        ### >>> Self_EvolveRec-BLOCK-START: Sparse transition scoring with vectorized deduplication and bounded lookup batches
        self.transition_amplitude = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))
        self.register_buffer(
            "transition_keys", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer("transition_evidence", torch.empty(0), persistent=False)
        self.register_buffer(
            "transition_outgoing", torch.zeros(itemnum + 1), persistent=False
        )
        self.transition_gate = False

    def fit_transition_graph(self, user_train, condition="C0"):
        if condition not in {"A", "C0", "C2"}:
            raise ValueError("SASREC_TRANSITION_CONDITION must be A, C0 or C2")
        stride = self.itemnum + 1
        if self.itemnum * stride + self.itemnum > np.iinfo(np.int64).max:
            raise ValueError("Transition keys exceed int64")
        counts, support = defaultdict(int), defaultdict(int)
        outgoing = np.zeros(stride, dtype=np.int64)
        destination = np.zeros(stride, dtype=np.int64)
        for history in user_train.values():
            contributed = set()
            for a, b in zip(history[:-1], history[1:]):
                a, b = int(a), int(b)
                if not (1 <= a <= self.itemnum and 1 <= b <= self.itemnum):
                    raise ValueError("Invalid training transition")
                key = a * stride + b
                counts[key] += 1
                outgoing[a] += 1
                destination[b] += 1
                contributed.add(key)
            for key in contributed:
                support[key] += 1
        keys = np.asarray(sorted(counts), dtype=np.int64)
        c = np.asarray([counts[int(k)] for k in keys], dtype=np.int64)
        u = np.asarray([support[int(k)] for k in keys], dtype=np.int64)
        prior = (destination.astype(np.float64) + 1.0) / (
            float(destination.sum()) + self.itemnum
        )
        prior[0] = 0.0
        graph = {
            "condition": condition,
            "itemnum": self.itemnum,
            "tau": 10.0,
            "user_shrinkage": 2.0,
            "lift_clip": 2.0,
            "keys": torch.from_numpy(keys),
            "counts": torch.from_numpy(c),
            "user_support": torch.from_numpy(u),
            "outgoing": torch.from_numpy(outgoing),
            "destination": torch.from_numpy(destination),
            "prior": torch.from_numpy(prior),
        }
        self.set_transition_graph(graph)
        return graph

    def set_transition_graph(self, graph):
        if int(graph["itemnum"]) != self.itemnum:
            raise ValueError("Transition graph catalog mismatch")
        condition = graph["condition"]
        if condition not in {"A", "C0", "C2"}:
            raise ValueError("Unknown transition condition")
        device = self.item_emb.weight.device
        keys = graph["keys"].to(device=device, dtype=torch.long)
        if keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("Transition keys must be strictly increasing")
        tau = float(graph["tau"])
        outgoing = graph["outgoing"].to(device=device, dtype=torch.float64)
        source = keys // (self.itemnum + 1)
        destination = keys % (self.itemnum + 1)
        prior = graph["prior"].to(device=device, dtype=torch.float64)[destination]
        count = graph["counts"].to(device=device, dtype=torch.float64)
        users = graph["user_support"].to(device=device, dtype=torch.float64)
        lift = torch.log((count / prior + tau) / (outgoing[source] + tau))
        evidence = lift.clamp(0.0, float(graph["lift_clip"]))
        evidence = evidence * users / (users + float(graph["user_shrinkage"]))
        self.transition_keys = keys if condition != "A" else keys[:0]
        self.transition_evidence = (
            evidence.float() if condition != "A" else evidence[:0].float()
        )
        self.transition_outgoing = (outgoing / (outgoing + tau)).float()
        self.transition_gate = condition == "C2"

    def _transition_residual(self, context, items):
        # Latest event first; array indices retain original event ages.
        if context.ndim != 2 or context.shape[1] != 20:
            raise ValueError("Transition context must have shape positions x 20")
        if items.ndim != 2 or items.shape[0] != context.shape[0]:
            raise ValueError("Transition candidates must have shape positions x K")
        if self.transition_keys.numel() == 0 or items.numel() == 0:
            return self.transition_evidence.new_zeros(items.shape)
        sources = context.long()
        ages = torch.arange(20, device=items.device)
        # For each event, discard it if the same ID occurred more recently.
        newer = ages[None, :] < ages[:, None]
        equal_ids = sources[:, :, None] == sources[:, None, :]
        valid = (sources != 0) & ~(equal_ids & newer).any(dim=-1)
        weights = torch.exp(-ages.float() / 5.0)[None, :]
        weights = weights * self.transition_outgoing[sources] * valid
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        query = sources[:, :, None] * (self.itemnum + 1) + items[:, None, :]
        flat_query = query.reshape(-1)
        locations = torch.searchsorted(self.transition_keys, flat_query)
        safe = locations.clamp_max(self.transition_keys.numel() - 1)
        exists = (locations < self.transition_keys.numel()) & (
            self.transition_keys[safe] == flat_query
        )
        evidence = torch.where(exists, self.transition_evidence[safe], 0.0).reshape_as(
            query
        )
        evidence = evidence * valid[:, :, None] * (items[:, None, :] != 0)
        supported = evidence > 0
        z = (weights[:, :, None] * evidence).sum(dim=1) / denominator
        confidence = (weights[:, :, None] * supported).sum(dim=1) / denominator
        residual = confidence * torch.tanh(z)
        if self.transition_gate:
            residual = residual * (supported.sum(dim=1) >= 2)
        return residual

    def predict_score(self, features, items):
        """Score aligned IDs (...,) or aligned candidate lists (..., K).

        Features have shape (..., hidden_units + 20), with float64-packed IDs.
        Position and candidate chunking bound temporary sparse-query memory.
        """
        features = torch.as_tensor(features, device=self.item_emb.weight.device)
        items = torch.as_tensor(
            items, device=self.item_emb.weight.device, dtype=torch.long
        )
        if features.ndim < 2 or features.shape[-1] != self.hidden_units + 20:
            raise ValueError("Expected collaborative features plus 20 context IDs")
        leading = features.shape[:-1]
        aligned = items.ndim == features.ndim - 1
        if aligned:
            items = items.unsqueeze(-1)
        if items.ndim != features.ndim or items.shape[:-1] != leading:
            raise ValueError("Candidate leading dimensions must match features")
        shape = items.shape
        if items.numel() == 0:
            out = self.item_emb.weight.new_empty(shape)
            return out.squeeze(-1) if aligned else out
        flat_features = features.reshape(-1, features.shape[-1])
        flat_items = items.reshape(-1, items.shape[-1])
        outputs = []
        amplitude = 2.0 * torch.sigmoid(self.transition_amplitude)
        # At most 1024 x 20 x 64 sparse queries per lookup.
        # Larger position batches reduce Python and GPU launch overhead.
        for start in range(0, flat_features.shape[0], 1024):
            f = flat_features[start : start + 1024]
            ids = flat_items[start : start + 1024]
            h = f[:, : self.hidden_units].to(self.item_emb.weight.dtype)
            candidate_scores = []
            for block in ids.split(64, dim=1):
                score = torch.bmm(self.item_emb(block), h.unsqueeze(-1)).squeeze(-1)
                if self.transition_keys.numel():
                    residual = self._transition_residual(
                        f[:, self.hidden_units :], block
                    )
                    score = score + amplitude * residual
                candidate_scores.append(score)
            outputs.append(torch.cat(candidate_scores, dim=1))
        out = torch.cat(outputs, dim=0).reshape(shape)
        return out.squeeze(-1) if aligned else out
        ### <<< Self_EvolveRec-BLOCK-END

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        ### >>> Self_EvolveRec-BLOCK-START: Accept diagnostic NumPy sequences on the trained model device
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        if seq.ndim != 2 or not 1 <= seq.shape[1] <= self.maxlen:
            raise ValueError("Sequences must have shape B x L with 1 <= L <= maxlen")
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)
        ### <<< Self_EvolveRec-BLOCK-END

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

        ### >>> Self_EvolveRec-BLOCK-START: Pack exact causal context for fused training and diagnostics
        h = self.last_layernorm(seq_emb)
        context = seq.new_zeros((B, M, 20))
        for age in range(min(M, 20)):
            context[:, age:, age] = seq[:, : M - age]
        return torch.cat((h.to(torch.float64), context.to(torch.float64)), dim=-1)
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
        ### >>> Self_EvolveRec-BLOCK-START: Use identical fused scores for forward and diagnostic calls
        if neg.ndim not in (seq.ndim, seq.ndim + 1):
            raise ValueError("Negative IDs must have shape B x L or B x L x K")
        return self.predict_score(log_feats, pos), self.predict_score(log_feats, neg)
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Bound sparse lookup memory and share fused catalog scoring
    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 2 or candidates.shape[0] != seq.shape[0]:
            raise ValueError("Candidates must have shape B x C matching sequences")
        features = self.log2feats(seq)[:, -1, :]
        if candidates.shape[1] == 0:
            return self.item_emb.weight.new_empty((seq.shape[0], 0))
        return torch.cat(
            [
                self.predict_score(features, block)
                for block in candidates.split(1024, dim=1)
            ],
            dim=1,
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
    ### >>> Self_EvolveRec-BLOCK-START: Keep validation randomness independent of training
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve the caller's model mode during validation
    HR10 = HR20 = NDCG10 = NDCG20 = 0.0
    was_training = model.training
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END
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
                ### >>> Self_EvolveRec-BLOCK-START: Use the isolated validation generator
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

    ### >>> Self_EvolveRec-BLOCK-START: Restore the caller's model mode after validation
    n = max(len(val_users), 1)
    model.train(was_training)
    ### <<< Self_EvolveRec-BLOCK-END
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }


# ── Per-dataset training ──────────────────────────────────────────────

### >>> Self_EvolveRec-BLOCK-START: Reproducible category initialization and bounded loss-only trial
import time


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    category_started = time.monotonic()
    category_budget_seconds = 1750.0
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)

    ### >>> Self_EvolveRec-BLOCK-START: Fit training-only graph and explicitly reset dedicated sampler
    global _SAMPLER_CACHE
    _SAMPLER_CACHE = None
    transition_condition = os.environ.get("SASREC_TRANSITION_CONDITION", "C0")
    graph = model.fit_transition_graph(user_train, transition_condition)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Loss-only sampled-softmax control
    ranking_criterion = nn.CrossEntropyLoss()
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ### >>> Self_EvolveRec-BLOCK-START: Save raw sparse counts and graph configuration beneath the trial
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
    torch.save(graph, os.path.join(working_dir, f"{dataset_name}_transition.pt"))
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint even at zero hits
    best_hr10 = -1.0
    best_epoch = 0
    completed_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Correct boolean target indexing in fused sampled softmax
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                num_negatives=SAMPLED_NEGATIVES,
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            indices = (seq_t != 0) & (pos_t != 0) & (neg_t != 0).any(dim=-1)
            if not indices.any():
                continue
            features = model(seq_t)[indices]
            positive_ids = pos_t[indices].unsqueeze(-1)
            candidate_ids = torch.cat((positive_ids, neg_t[indices]), dim=-1)
            logits = model.predict_score(features, candidate_ids)
            logits = logits.masked_fill(candidate_ids == 0, -torch.inf)
            loss = ranking_criterion(
                logits, torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            )
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Release gradient buffers before backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        ### >>> Self_EvolveRec-BLOCK-START: Record complete-epoch exposure
        completed_epoch = epoch
        avg_loss = total_loss / NUM_BATCHES
        ### <<< Self_EvolveRec-BLOCK-END

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
            ### >>> Self_EvolveRec-BLOCK-START: Interpret patience in documented epoch units
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

        ### >>> Self_EvolveRec-BLOCK-START: End the category allocation at an epoch boundary
        if time.monotonic() - category_started >= category_budget_seconds:
            print(
                f"  [{dataset_name}] Category time allocation reached after complete epoch {epoch}",
                flush=True,
            )
            break
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Record actual transition condition and selected checkpoint amplitude
    if best_epoch == 0:
        raise RuntimeError(
            f"{dataset_name}: no scheduled validation checkpoint completed; "
            "this four-category condition is incomplete"
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    manifest = {
        "condition": transition_condition,
        "dataset": dataset_name,
        "training_seed": 42,
        "negative_count": SAMPLED_NEGATIVES,
        "negative_distribution": "uniform_distinct_excluding_causal_prefix_and_target",
        "negative_rng_seed": 42,
        "duplicate_negatives": "excluded",
        "no_eligible_negatives": "skip_position",
        "logQ_correction": False,
        "item_embedding_initialization_std": 0.1,
        "lexical_residual": False,
        "transition_history_events": 20,
        "transition_recency_scale": 5.0,
        "transition_tau": 10.0,
        "transition_user_shrinkage": 2.0,
        "transition_lift_clip": 2.0,
        "transition_two_source_gate": transition_condition == "C2",
        "transition_raw_edges": int(graph["keys"].numel()),
        "selected_amplitude": float(
            (2 * model.transition_amplitude.sigmoid()).detach().cpu()
        ),
        "graph_scope": "adjacent training events only; no held-out interactions",
        "graph_diagnostic_limitation": "training margins include self-contributed edges; user-exclusion sensitivity not measured here",
        "selection_tie_rule": "retain_earliest_checkpoint_on_equal_hr10",
        "completed_epochs": completed_epoch,
        "best_epoch": best_epoch,
        "best_validation_hr10": best_hr10,
        "elapsed_seconds": time.monotonic() - category_started,
        "category_budget_seconds": category_budget_seconds,
        "budget_boundary": "complete_epoch; final epoch may overrun allocation",
        "optimizer": "Adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.98],
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "max_epochs": NUM_EPOCHS,
        "validation_every_epochs": 5,
        "early_stopping_stale_epochs": PATIENCE,
        "inference": "shared packed-context fused scorer; category-specific persisted sparse graph",
    }
    with open(os.path.join(working_dir, f"{dataset_name}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
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
                "maxlen": MAXLEN,
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
        ### >>> Self_EvolveRec-BLOCK-START: Fail clearly for missing category artifacts
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing trained artifacts for {ds}: {meta_path}, {ckpt_path}"
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
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Restore category graph before loading identical trained scorer
        graph_path = os.path.join(_HERE, f"{ds}_transition.pt")
        if not os.path.isfile(graph_path):
            raise FileNotFoundError(f"Missing transition artifact for {ds}: {graph_path}")
        graph = torch.load(graph_path, map_location="cpu", weights_only=True)
        m.set_transition_graph(graph)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Preserve explicit dataset routing without silent substitution
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        # Preserve the documented legacy signature when no category is supplied.
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        raise RuntimeError("No trained category models were loaded")
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Use the updated shared scorer and verify predictor output integrity
    with torch.inference_mode():
        cand_t = torch.tensor([candidates], dtype=torch.long, device=_device)
        scores_t = model.predict(seq_t, cand_t)
        if scores_t.shape != cand_t.shape:
            raise RuntimeError("Predictor must return one score per candidate")
        if not torch.isfinite(scores_t).all():
            raise FloatingPointError("Predictor returned nonfinite candidate scores")
        scores = scores_t[0].cpu().tolist()

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


