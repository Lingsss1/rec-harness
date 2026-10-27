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
### >>> Self_EvolveRec-BLOCK-START: Matched multi-negative objective controls and bounded graph initialization
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
LOSS_TYPE = os.environ.get("SASREC_LOSS", "ce").lower()
# Zero runs the objective-only control; positive values initialize a trainable
# coefficient constrained to (0, 2). Keep raw dot-product backbone logits.
GRAPH_WEIGHT = float(os.environ.get("SASREC_GRAPH_WEIGHT", "0.01"))
NUM_NEGATIVES = int(os.environ.get("SASREC_NEGATIVES", "64"))
if LOSS_TYPE not in {"ce", "bce"}:
    raise ValueError("SASREC_LOSS must be ce or bce")
if NUM_EPOCHS < 1 or NUM_NEGATIVES < 1:
    raise ValueError("Epoch and negative counts must be positive")
if not math.isfinite(GRAPH_WEIGHT) or not 0.0 <= GRAPH_WEIGHT < 2.0:
    raise ValueError("SASREC_GRAPH_WEIGHT must be finite and in [0, 2)")
CHECKPOINT_EPOCHS = sorted(
    {e for e in (50, 100, 150, 200) if e <= NUM_EPOCHS} | {NUM_EPOCHS}
)
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host-provided input directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain every category and isolate deterministic training seeds
DATASETS = list(DEFAULT_DATASETS)


def seed_training():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


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


### >>> Self_EvolveRec-BLOCK-START: Unique multi-negative sampling with complete training-positive exclusion
class TrainingSampler:
    def __init__(self, user_train, itemnum, maxlen):
        self.histories = user_train
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.rng = np.random.default_rng(42)
        self.seen = {u: np.unique(items) for u, items in user_train.items()}
        self.users = [
            u
            for u in sorted(user_train)
            if len(user_train[u]) >= 2 and len(self.seen[u]) < itemnum
        ]
        if not self.users:
            raise ValueError("No users with a causal target and eligible negatives")
        self.complements = {}

    def negatives(self, u, positions, requested):
        seen = self.seen[u]
        available = self.itemnum - len(seen)
        k = min(requested, available)
        result = self.rng.integers(1, self.itemnum + 1, size=(positions, k))
        # Each row is a separate position: retain the first occurrence of a draw.
        for attempt in range(16):
            bad = np.isin(result, seen)
            order = np.argsort(result, axis=1, kind="stable")
            ordered = np.take_along_axis(result, order, axis=1)
            duplicates = ordered[:, 1:] == ordered[:, :-1]
            rr, cc = np.nonzero(duplicates)
            bad[rr, order[rr, cc + 1]] = True
            if not bad.any():
                return result
            if available < 4 * k or len(seen) > self.itemnum // 2:
                break
            result[bad] = self.rng.integers(1, self.itemnum + 1, size=int(bad.sum()))
        # Also bounds runtime for nearly saturated catalogs. Sampling each row
        # without replacement is uniform over the eligible item IDs.
        if u not in self.complements:
            self.complements[u] = np.setdiff1d(
                np.arange(1, self.itemnum + 1), seen, assume_unique=True
            )
        pool = self.complements[u]
        return np.stack(
            [self.rng.choice(pool, k, replace=False) for _ in range(positions)]
        )

    def sample(self, batch_size, requested):
        users = self.rng.choice(self.users, size=batch_size, replace=True)
        seq = np.zeros((batch_size, self.maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, self.maxlen, requested), dtype=np.int64)
        for b, u in enumerate(users):
            history = self.histories[int(u)]
            length = min(len(history) - 1, self.maxlen)
            seq[b, -length:] = history[-length - 1 : -1]
            pos[b, -length:] = history[-length:]
            draws = self.negatives(int(u), length, requested)
            neg[b, -length:, : draws.shape[1]] = draws
        return seq, pos, neg, users


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    # Preserve the diagnostic loader's B x L, single-negative contract.
    sampler = TrainingSampler(user_train, itemnum, maxlen)
    seq, pos, neg, _ = sampler.sample(batch_size, 1)
    return seq, pos, neg[:, :, 0]


### <<< Self_EvolveRec-BLOCK-END


# ── Model ─────────────────────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Identical user-excluded support-shrunk transition residuals at training and inference
class TransitionGraph:
    """Five sparse graphs; scope f excludes every training user in fold f."""

    @staticmethod
    def stable_fold(dataset_name, user_id):
        import hashlib

        identity = str(int(user_id))
        payload = json.dumps([dataset_name, identity, 0], separators=(",", ":"))
        return (
            int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
            % 5
        )

    @staticmethod
    def build(user_train, itemnum, dataset_name):
        from collections import Counter

        if itemnum < 1:
            raise ValueError("Graph requires a nonempty item vocabulary")
        folds = {
            int(u): TransitionGraph.stable_fold(dataset_name, u)
            for u in sorted(user_train)
        }
        contributions = [Counter() for _ in range(5)]
        stride = itemnum + 1
        for u in sorted(user_train):
            history = user_train[u]
            # Count ordered events, preserving repeats and self-transitions.
            contributions[folds[int(u)]].update(
                int(a) * stride + int(b)
                for a, b in zip(history[:-1], history[1:])
                if 0 < a <= itemnum and 0 < b <= itemnum
            )
        full = Counter()
        for contribution in contributions:
            full.update(contribution)
        keys = np.asarray(sorted(full), dtype=np.int64)
        counts = np.asarray([full[int(k)] for k in keys], dtype=np.float64)
        source, destination = keys // stride, keys % stride
        scopes = []
        for excluded in range(5):
            values = counts - np.asarray(
                [contributions[excluded].get(int(k), 0) for k in keys],
                dtype=np.float64,
            )
            observed = values > 0
            edge_keys = keys[observed]
            c = values[observed]
            src, dst = source[observed], destination[observed]
            outgoing = np.bincount(src, weights=c, minlength=stride)
            incoming = np.bincount(dst, weights=c, minlength=stride)
            total = float(c.sum())
            probability = (incoming + 1.0) / (total + itemnum)
            ci, pj = outgoing[src], probability[dst]
            weight = np.clip(np.log((c + 10.0 * pj) / ((ci + 10.0) * pj)), 0.0, 5.0)
            weight *= c / (c + 5.0)
            weight *= ci / (ci + 20.0)
            keep = weight > 0
            scopes.append(
                {
                    "keys": edge_keys[keep],
                    "weights": weight[keep].astype(np.float32),
                }
            )
        state = {
            "itemnum": itemnum,
            "dataset_name": dataset_name,
            "user_folds": folds,
            "scopes": scopes,
            "configuration": {
                "folds": 5,
                "seed": 0,
                "digest": "sha256",
                "eta": 1.0,
                "alpha": 10.0,
                "beta": 5.0,
                "gamma": 20.0,
                "clip": 5.0,
                "window": 5,
                "half_life": 2.0,
                "events": "adjacent ordered events, retaining repeats",
                "pruning": None,
            },
        }
        return state, folds

    def __init__(self, state, device):
        self.itemnum = state["itemnum"]
        self.dataset_name = state["dataset_name"]
        self.user_folds = state["user_folds"]
        if len(state["scopes"]) != 5:
            raise ValueError("Expected five exclusion graphs")
        self.scopes = [
            {key: torch.as_tensor(value, device=device) for key, value in scope.items()}
            for scope in state["scopes"]
        ]

    def folds_for(self, user_ids, device):
        assigned = []
        for user in user_ids:
            user = int(user)
            fold = self.user_folds.get(user)
            if fold is None:
                fold = self.stable_fold(self.dataset_name, user)
            assigned.append(fold)
        return torch.tensor(assigned, dtype=torch.long, device=device)

    @staticmethod
    def recent_items(seq):
        # Last five nonpadding items for every causal prefix. This also handles
        # holes introduced by diagnostic history perturbations.
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        latest = torch.where(seq.ne(0), positions, -1).cummax(dim=1).values
        index = latest
        recent = []
        for _ in range(5):
            recent.append(torch.where(index >= 0, seq.gather(1, index.clamp_min(0)), 0))
            previous = latest.gather(1, (index - 1).clamp_min(0))
            index = torch.where(index > 0, previous, -1)
        return torch.stack(recent, dim=-1)

    @torch.no_grad()
    def score(self, seq, candidates, folds=None, position_mask=None):
        if seq.ndim != 2 or candidates.ndim not in (2, 3):
            raise ValueError(
                "Expected B x L sequences and B x C or B x L x C candidates"
            )
        if candidates.shape[0] != seq.shape[0]:
            raise ValueError("Sequence and candidate batch sizes differ")
        if candidates.ndim == 3 and candidates.shape[1] != seq.shape[1]:
            raise ValueError("Position-level candidates must match sequence length")
        if folds is None:
            raise ValueError(
                "Complete graph scoring requires explicit user/fold context"
            )
        if folds.shape != (seq.shape[0],):
            raise ValueError("Expected one fold per sequence")
        if bool(((folds < 0) | (folds >= 5)).any()):
            raise ValueError("User folds must lie in 0..4")
        windows = self.recent_items(seq)
        if candidates.ndim == 2:
            history = windows[:, -1]
            scope_ids = folds
        else:
            history = windows.reshape(-1, 5)
            scope_ids = folds[:, None].expand_as(seq).reshape(-1)
        shape = candidates.shape
        if position_mask is not None and position_mask.shape != shape[:-1]:
            raise ValueError("Position mask must match candidate leading dimensions")
        active = history.ne(0).any(-1)
        if position_mask is not None:
            active &= position_mask.reshape(-1)
        candidates = candidates.reshape(history.shape[0], shape[-1])
        output = torch.zeros(candidates.shape, device=seq.device, dtype=torch.float32)
        if shape[-1] == 0:
            return output.reshape(shape)
        decay = 2.0 ** (-torch.arange(5, device=seq.device, dtype=torch.float32) / 2.0)
        chunk_size = max(1, min(512, 262144 // (5 * shape[-1])))
        for scope_id, graph in enumerate(self.scopes):
            keys = graph["keys"]
            if not keys.numel():
                continue
            selected = torch.where((scope_ids == scope_id) & active)[0]
            for start in range(0, selected.numel(), chunk_size):
                rows = selected[start : start + chunk_size]
                h, cand = history[rows], candidates[rows]
                weights = h.ne(0).float() * decay
                weights /= weights.sum(-1, keepdim=True).clamp_min(1e-12)
                query = (
                    h.unsqueeze(-1) * (self.itemnum + 1) + cand.unsqueeze(1)
                ).contiguous()
                index = torch.searchsorted(keys, query)
                bounded = index.clamp_max(keys.numel() - 1)
                matched = (index < keys.numel()) & (keys[bounded] == query)
                matched &= h.unsqueeze(-1).ne(0) & cand.unsqueeze(1).ne(0)
                edge = torch.where(matched, graph["weights"][bounded], 0.0)
                output[rows] = (weights.unsqueeze(-1) * edge).sum(1)
        return output.reshape(shape)


### <<< Self_EvolveRec-BLOCK-END


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
        ### >>> Self_EvolveRec-BLOCK-START: Train a bounded dataset-specific graph coefficient
        self.transition_graph = None
        self.graph_enabled = False
        initial_weight = GRAPH_WEIGHT if GRAPH_WEIGHT > 0 else 0.01
        self.graph_logit = nn.Parameter(
            torch.tensor(
                math.log(initial_weight / (2.0 - initial_weight)), dtype=torch.float32
            )
        )
        ### <<< Self_EvolveRec-BLOCK-END

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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unused attention-weight materialization
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

    ### >>> Self_EvolveRec-BLOCK-START: Explicit user context and checkpointed raw-dot scoring with a bounded residual
    @property
    def graph_weight(self):
        return 2.0 * torch.sigmoid(self.graph_logit) if self.graph_enabled else 0.0

    def attach_graph(self, state, weight, device):
        # Do not reset graph_logit here: checkpoint loading restores its learned value.
        self.graph_enabled = bool(weight)
        if self.graph_enabled and state is None:
            raise ValueError("An enabled residual requires persisted exclusion graphs")
        self.transition_graph = TransitionGraph(state, device) if weight else None
        self.graph_logit.requires_grad_(self.graph_enabled)

    def _candidate_dots(self, features, candidates):
        # Batched matrix multiplication avoids materializing an additional
        # positions x candidates x hidden elementwise-product tensor.
        embeddings = self.item_emb(candidates)
        return torch.bmm(embeddings, features.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
        folds=None,
    ):
        features = self.log2feats(seq)
        if pos is None or neg is None:
            return features
        single_negative = neg.ndim == 2
        if single_negative:
            neg = neg.unsqueeze(-1)
        candidates = torch.cat((pos.unsqueeze(-1), neg), dim=-1)
        valid = pos.ne(0)
        active_features = features[valid]
        active_candidates = candidates[valid]
        pieces = []
        for start in range(0, active_features.shape[0], 512):
            f = active_features[start : start + 512]
            c = active_candidates[start : start + 512]
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                value = checkpoint(
                    self._candidate_dots,
                    f,
                    c,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                value = self._candidate_dots(f, c)
            pieces.append(value)
        logits = features.new_zeros(candidates.shape)
        if pieces:
            logits[valid] = torch.cat(pieces, dim=0)
        if self.transition_graph is not None:
            logits = logits + self.graph_weight * self.transition_graph.score(
                seq, candidates, folds, position_mask=valid
            )
        positive, negative = logits[..., 0], logits[..., 1:]
        return positive, negative.squeeze(-1) if single_negative else negative

    def predict(
        self, seq: torch.Tensor, candidates: torch.Tensor, folds=None, user_ids=None
    ) -> torch.Tensor:
        if self.transition_graph is not None:
            if folds is None:
                if user_ids is None:
                    raise ValueError(
                        "Complete scoring requires explicit user IDs or folds"
                    )
                folds = self.transition_graph.folds_for(user_ids, seq.device)
        final_feat = self.log2feats(seq)[:, -1, :]
        scores = []
        for start in range(0, candidates.shape[-1], 4096):
            items = candidates[..., start : start + 4096]
            value = self._candidate_dots(final_feat, items)
            if self.transition_graph is not None:
                value = value + self.graph_weight * self.transition_graph.score(
                    seq, items, folds=folds
                )
            scores.append(value)
        if not scores:
            return final_feat.new_empty((seq.shape[0], 0))
        return torch.cat(scores, dim=-1)

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
    ### >>> Self_EvolveRec-BLOCK-START: Fixed validation candidates without mutating training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic unique validation negatives with saturation guard
            if itemnum - len(interacted) < 99:
                raise ValueError("Validation requires 99 eligible unique negatives")
            while len(neg_pool) < 99:
                ni = rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Validate using the user's persisted exclusion graph
            scores = (
                model.predict(seq_t, cand_t, user_ids=[u])[0].cpu().numpy().tolist()
            )
            ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Shared objective sampling and training-only stable user exclusion
    seed_training()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    sampler = TrainingSampler(user_train, itemnum, MAXLEN)
    graph_state, user_folds = (
        TransitionGraph.build(user_train, itemnum, dataset_name)
        if GRAPH_WEIGHT
        else (None, {})
    )
    requested_negatives = NUM_NEGATIVES
    skipped_positions = sum(
        min(max(len(history) - 1, 0), MAXLEN)
        for u, history in user_train.items()
        if len(sampler.seen[u]) >= itemnum
    )
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "skipped_no_negative_causal_positions_per_history_pass": skipped_positions,
                "negative_count": requested_negatives,
                "graph_positive_edges_by_fold": (
                    [len(s["keys"]) for s in graph_state["scopes"]]
                    if graph_state is not None
                    else []
                ),
                "graph_sparse_bytes": (
                    sum(
                        s["keys"].nbytes + s["weights"].nbytes
                        for s in graph_state["scopes"]
                    )
                    if graph_state is not None
                    else 0
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)

    ### >>> Self_EvolveRec-BLOCK-START: Activate cross-fitted residuals in both objective treatments
    if graph_state is not None:
        model.attach_graph(graph_state, GRAPH_WEIGHT, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Record common-rung checkpoints for macro selection
    rung_metrics = {}
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Matched candidate CE and balanced multi-negative BCE
            seq, pos, neg, users = sampler.sample(BATCH_SIZE, requested_negatives)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            fold_t = (
                torch.tensor(
                    [user_folds[int(u)] for u in users], dtype=torch.long, device=device
                )
                if graph_state is not None
                else None
            )
            valid = pos_t.ne(0) & neg_t.ne(0).any(-1)
            if not bool(valid.any()):
                continue
            pos_logits, neg_logits = model(seq_t, pos_t, neg_t, folds=fold_t)
            if LOSS_TYPE == "ce":
                negative_scores = neg_logits.masked_fill(neg_t == 0, -torch.inf)
                logits = torch.cat((pos_logits.unsqueeze(-1), negative_scores), dim=-1)[
                    valid
                ]
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    torch.zeros(logits.shape[0], dtype=torch.long, device=device),
                )
            else:
                # Each position has weight 1 for its positive term and weight 1
                # for the mean of its eligible negative terms, independent of K.
                negative_mask = neg_t[valid].ne(0)
                negative_loss = torch.nn.functional.softplus(neg_logits[valid])
                negative_mean = negative_loss.masked_fill(~negative_mask, 0.0).sum(
                    -1
                ) / negative_mask.sum(-1).clamp_min(1)
                loss = (
                    torch.nn.functional.softplus(-pos_logits[valid]) + negative_mean
                ).mean()
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate common rungs with identical user-excluded scoring
        if epoch in CHECKPOINT_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            rung_metrics[epoch] = val_m
            torch.save(model.state_dict(), ckpt_path + f".epoch{epoch}")
            learned_weight = (
                float(model.graph_weight) if graph_state is not None else 0.0
            )
            print(
                f"  [{dataset_name}] epoch={epoch} loss={avg_loss:.4f} "
                f"graph_weight={learned_weight:.6f} "
                f"validation={json.dumps(val_m, sort_keys=True)}",
                flush=True,
            )
        elif epoch % 10 == 0:
            print(f"  [{dataset_name}] epoch={epoch} loss={avg_loss:.4f}", flush=True)
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist every exclusion graph and its exact scoring configuration
                "graph_weight": GRAPH_WEIGHT,
                "graph_coefficient": "2*sigmoid(graph_logit); learned value in checkpoint",
                "transition_graph": graph_state,
                "score_normalization": "none",
                "temperature": 1.0,
                "loss_type": LOSS_TYPE,
                "negative_count": requested_negatives,
                "negative_sampling": "uniform unique, excluding complete training history",
                "bce_weighting": "positive term plus per-position mean negative term",
                "training_seed": 42,
                "graph_partition_seed": 0,
                "graph_scope": "same stable user-excluded fold at training and inference",
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    ### >>> Self_EvolveRec-BLOCK-START: Verify serialized complete scores before returning common rungs
    with open(meta_path, "rb") as f:
        saved_meta = pickle.load(f)
    restored = SASRec(
        itemnum=saved_meta["itemnum"],
        hidden_units=saved_meta["hidden_units"],
        num_blocks=saved_meta["num_blocks"],
        num_heads=saved_meta["num_heads"],
        dropout_rate=0.0,
        maxlen=saved_meta["maxlen"],
    ).to(device)
    restored.load_state_dict(
        torch.load(
            ckpt_path + f".epoch{NUM_EPOCHS}", map_location=device, weights_only=True
        )
    )
    restored.attach_graph(
        saved_meta["transition_graph"], saved_meta["graph_weight"], device
    )
    check_users = sampler.users[:4]
    check_seq = np.zeros((len(check_users), MAXLEN), dtype=np.int64)
    for row, user in enumerate(check_users):
        history = user_train[user][-MAXLEN:]
        check_seq[row, -len(history) :] = history
    check_seq = torch.as_tensor(check_seq, dtype=torch.long, device=device)
    check_candidates = (
        torch.arange(1, min(itemnum, 128) + 1, device=device)
        .unsqueeze(0)
        .expand(len(check_users), -1)
    )
    previous_mode = model.training
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model.predict(check_seq, check_candidates, user_ids=check_users)
        actual = restored.predict(check_seq, check_candidates, user_ids=check_users)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    model.train(previous_mode)
    del restored
    return rung_metrics
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
    return mod.SASRec


### >>> Self_EvolveRec-BLOCK-START: Load requested categories lazily and reject missing artifacts explicitly
def _ensure_loaded(dataset_name=None):
    global _device
    if dataset_name is not None and dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset_name!r}")
    # Retain the legacy default route without allocating all four models and
    # their five exclusion graphs on the GPU for a single-category request.
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds in _models:
        return
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
    ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
    if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing model or metadata for dataset {ds!r}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    SASRec = _load_model_class()
    m = SASRec(
        itemnum=meta["itemnum"],
        hidden_units=meta["hidden_units"],
        num_blocks=meta["num_blocks"],
        num_heads=meta["num_heads"],
        dropout_rate=0.0,
        maxlen=meta["maxlen"],
    ).to(_device)
    m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
    m.attach_graph(meta.get("transition_graph"), meta.get("graph_weight", 0.0), _device)
    m.eval()
    # Publish only after construction succeeds, keeping cache state consistent
    # if checkpoint restoration or graph reconstruction raises an exception.
    _metas[ds] = meta
    _models[ds] = m
### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ### >>> Self_EvolveRec-BLOCK-START: Invoke category-specific lazy restoration
    _ensure_loaded(dataset_name)
    ### <<< Self_EvolveRec-BLOCK-END

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject unavailable explicit routes rather than silently using another catalog
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Keep the legacy default independent of previous inference requests
    else:
        ds = DATASETS[0]
        model = _models[ds]
        meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    with torch.no_grad():
        cand_t   = torch.tensor([candidates], dtype=torch.long, device=_device)
        ### >>> Self_EvolveRec-BLOCK-START: Route inference by explicit user identity without sequence matching
        scores   = model.predict(seq_t, cand_t, user_ids=[user_id])[0].cpu().tolist()
        ### <<< Self_EvolveRec-BLOCK-END

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

    ### >>> Self_EvolveRec-BLOCK-START: Common macro selection with explicit scoring treatment and predictor roundtrip
    treatment = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": 42,
        "graph_partition_seed": 0,
        "workers": 0,
        "deterministic_algorithms": True,
        "determinism_warn_only": True,
        "loss": LOSS_TYPE,
        "graph_initial_weight": GRAPH_WEIGHT,
        "graph_coefficient": "2*sigmoid(a), independently trained per category",
        "graph_scope": "same stable user-excluded fold during training and inference",
        "negative_count": NUM_NEGATIVES,
        "score_normalization": "none",
        "temperature": 1.0,
        "bce_weighting": "positive term plus per-position mean negative term",
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
    }
    print(json.dumps(treatment, sort_keys=True), flush=True)
    results = {}
    for ds in DATASETS:
        print(f"Training {ds}", flush=True)
        results[ds] = train_one_dataset(ds, device, DATA_DIR)

    macro = {
        epoch: sum(results[ds][epoch]["HR@10"] for ds in DATASETS) / len(DATASETS)
        for epoch in CHECKPOINT_EPOCHS
    }
    selected_epoch = max(CHECKPOINT_EPOCHS, key=lambda epoch: (macro[epoch], -epoch))
    import shutil

    for ds in DATASETS:
        destination = os.path.join(working_dir, f"{ds}_model.pt")
        shutil.copyfile(destination + f".epoch{selected_epoch}", destination)
    with open(os.path.join(working_dir, "validation_log.json"), "w") as f:
        json.dump(
            {
                "categories": results,
                "macro_hr10": macro,
                "selected_epoch": selected_epoch,
                "selection_metric": "macro validation HR@10",
                "treatment": treatment,
            },
            f,
            indent=2,
        )
    write_predict_script(working_dir)

    # Exercise the actual generated predictor on selected checkpoints. Only
    # public training histories are read; no evaluation labels are involved.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "selected_predictor_roundtrip", os.path.join(working_dir, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    # Exercise each lazy-loading route without preloading unrelated categories.
    for ds in DATASETS:
        with open(os.path.join(working_dir, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        reference = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(device)
        reference.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{ds}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        reference.attach_graph(meta["transition_graph"], meta["graph_weight"], device)
        reference.eval()
        histories = _read_split(os.path.join(DATA_DIR, f"{ds}_train.txt"))
        user = next(u for u in sorted(histories) if histories[u])
        history = histories[user][-meta["maxlen"] :]
        candidates = list(range(1, min(meta["itemnum"], 128) + 1))
        seq = torch.tensor(
            [[0] * (meta["maxlen"] - len(history)) + history],
            dtype=torch.long,
            device=device,
        )
        cand = torch.tensor([candidates], dtype=torch.long, device=device)
        with torch.no_grad():
            expected = reference.predict(seq, cand, user_ids=[user])[0].cpu()
            actual = torch.tensor(
                predictor.predict(user, history, candidates, dataset_name=ds),
                dtype=expected.dtype,
            )
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        del reference
        # Bound roundtrip memory to the reference and predictor for one category.
        predictor._models.clear()
        predictor._metas.clear()
    print(
        f"Selected common epoch={selected_epoch}; "
        f"macro validation HR@10={macro[selected_epoch]:.4f}; predictor roundtrip passed",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


