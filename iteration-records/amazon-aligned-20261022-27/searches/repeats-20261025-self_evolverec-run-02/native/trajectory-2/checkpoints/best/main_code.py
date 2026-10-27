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
### >>> Self_EvolveRec-BLOCK-START: Common epoch ceiling and configurable loss treatment
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
LOSS_TYPE = os.environ.get("SASREC_LOSS", "ce").lower()
GRAPH_WEIGHT = float(os.environ.get("SASREC_GRAPH_WEIGHT", "0.3"))
NUM_NEGATIVES = int(os.environ.get("SASREC_NEGATIVES", "32"))
if LOSS_TYPE not in {"ce", "bce"}:
    raise ValueError("SASREC_LOSS must be ce or bce")
if NUM_EPOCHS < 1 or NUM_NEGATIVES < 1:
    raise ValueError("Epoch and negative counts must be positive")
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


### >>> Self_EvolveRec-BLOCK-START: Cross-fitted transition lift with bounded vectorized lookups and explicit shape checks
class TransitionGraph:
    """Scope 0 is full training; scope f+1 excludes user fold f.

    Arrays contain only training-derived counts. Graph tensors are deliberately
    separate from the encoder state dict; inference serializes scope 0 only.
    """

    @staticmethod
    def build(user_train, itemnum):
        from collections import Counter

        users = np.asarray(sorted(user_train), dtype=np.int64)
        shuffled = np.random.default_rng(0).permutation(users)
        folds = {int(u): i % 5 for i, u in enumerate(shuffled)}
        contributions = [Counter() for _ in range(5)]
        stride = itemnum + 1
        for u in users:
            history = user_train[int(u)]
            pairs = {
                int(a) * stride + int(b)
                for a, b in zip(history[:-1], history[1:])
                if a > 0 and b > 0
            }
            contributions[folds[int(u)]].update(pairs)
        full = Counter()
        for contribution in contributions:
            full.update(contribution)
        keys = np.asarray(sorted(full), dtype=np.int64)
        counts = np.asarray([full[int(key)] for key in keys], dtype=np.float64)
        source, destination = keys // stride, keys % stride
        scopes = []
        for excluded in (None, 0, 1, 2, 3, 4):
            values = counts.copy()
            if excluded is not None:
                values -= np.asarray(
                    [contributions[excluded].get(int(key), 0) for key in keys],
                    dtype=np.float64,
                )
            keep = values > 0
            outgoing = np.bincount(source[keep], weights=values[keep], minlength=stride)
            incoming = np.bincount(
                destination[keep], weights=values[keep], minlength=stride
            )
            probability = (incoming + 1.0) / (float(values.sum()) + itemnum)
            probability[0] = 1.0 / max(itemnum, 1)
            scopes.append(
                {
                    "keys": keys[keep],
                    "counts": values[keep].astype(np.float32),
                    "outgoing": outgoing.astype(np.float32),
                    "probability": probability.astype(np.float32),
                }
            )
        return {"itemnum": itemnum, "scopes": scopes}, folds

    def __init__(self, state, device):
        self.itemnum = state["itemnum"]
        self.scopes = [
            {key: torch.as_tensor(value, device=device) for key, value in scope.items()}
            for scope in state["scopes"]
        ]

    @torch.no_grad()
    def score(self, seq, candidates, folds=None, position_mask=None):
        # B x C scores final positions; B x L x C scores causal prefixes.
        if seq.ndim != 2 or candidates.ndim not in (2, 3):
            raise ValueError(
                "Expected B x L sequences and B x C or B x L x C candidates"
            )
        if candidates.shape[0] != seq.shape[0]:
            raise ValueError("Sequence and candidate batch sizes differ")
        if candidates.ndim == 3 and candidates.shape[1] != seq.shape[1]:
            raise ValueError("Position-level candidates must match sequence length")
        if folds is not None:
            if folds.shape != (seq.shape[0],) or len(self.scopes) != 6:
                raise ValueError(
                    "Cross-fitted scoring requires B folds and six graph scopes"
                )
            if bool(((folds < 0) | (folds > 4)).any()):
                raise ValueError("User folds must lie in 0..4")

        windows = torch.nn.functional.pad(seq, (9, 0)).unfold(1, 10, 1).flip(-1)
        if candidates.ndim == 2:
            history = windows[:, -1, :]
            scope_ids = (
                torch.zeros(seq.shape[0], device=seq.device, dtype=torch.long)
                if folds is None
                else folds + 1
            )
        else:
            history = windows.reshape(-1, 10)
            scope_ids = (
                torch.zeros(history.shape[0], device=seq.device, dtype=torch.long)
                if folds is None
                else (folds[:, None] + 1).expand_as(seq).reshape(-1)
            )
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
        decay = torch.exp(
            -torch.arange(10, device=seq.device, dtype=torch.float32) / 3.0
        )
        # Bound temporary B x history-window x candidates lookup tensors.
        chunk_size = max(1, min(512, 262144 // (10 * shape[-1])))
        for scope_id, graph in enumerate(self.scopes):
            selected = torch.where((scope_ids == scope_id) & active)[0]
            keys = graph["keys"]
            for start in range(0, selected.numel(), chunk_size):
                rows = selected[start : start + chunk_size]
                h, cand = history[rows], candidates[rows]
                weights = h.ne(0).float() * decay
                weights /= weights.sum(-1, keepdim=True).clamp_min(1e-12)
                probability = graph["probability"][cand]
                # Vectorize all ten history positions in one sparse lookup.
                lift = torch.log(10.0 / (graph["outgoing"][h] + 10.0)).unsqueeze(-1)
                if keys.numel():
                    query = (
                        h.unsqueeze(-1) * (self.itemnum + 1) + cand.unsqueeze(1)
                    ).contiguous()
                    index = torch.searchsorted(keys, query)
                    bounded = index.clamp_max(keys.numel() - 1)
                    matched = (index < keys.numel()) & (keys[bounded] == query)
                    count = torch.where(matched, graph["counts"][bounded], 0.0)
                    lift = lift + torch.log1p(count / (10.0 * probability.unsqueeze(1)))
                # Preserve clipped missing-edge penalties, including in inference.
                output[rows] = (weights.unsqueeze(-1) * lift.clamp(-3.0, 3.0)).sum(1)
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
        ### >>> Self_EvolveRec-BLOCK-START: Attach explicitly serialized graph scoring
        self.transition_graph = None
        self.graph_weight = 0.0
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

    ### >>> Self_EvolveRec-BLOCK-START: Score valid targets only and checkpoint candidate embeddings to bound training memory
    def attach_graph(self, state, weight, device):
        self.graph_weight = float(weight)
        self.transition_graph = TransitionGraph(state, device) if weight else None

    def _candidate_dots(self, features, candidates):
        return (features.unsqueeze(-2) * self.item_emb(candidates)).sum(-1)

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
        # Checkpointing recomputes candidate embeddings during backward rather
        # than retaining a potentially large P x candidates x hidden tensor.
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

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        final_feat = self.log2feats(seq)[:, -1, :]
        scores = []
        for start in range(0, candidates.shape[-1], 4096):
            items = candidates[..., start : start + 4096]
            value = self._candidate_dots(final_feat, items)
            if self.transition_graph is not None:
                value = value + self.graph_weight * self.transition_graph.score(
                    seq, items
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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible per-category initialization and training-only graph construction
    seed_training()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    sampler = TrainingSampler(user_train, itemnum, MAXLEN)
    graph_state, user_folds = (
        TransitionGraph.build(user_train, itemnum) if GRAPH_WEIGHT else (None, {})
    )
    requested_negatives = NUM_NEGATIVES if LOSS_TYPE == "ce" else 1
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
            ### >>> Self_EvolveRec-BLOCK-START: Candidate-set CE with masked unused slots and user-fold routing
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
            pos_logits, neg_logits = model(seq_t, pos_t, neg_t, folds=fold_t)
            valid = pos_t != 0
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
                positive_scores = pos_logits[valid]
                negative_scores = neg_logits[..., 0][valid]
                loss = bce_criterion(positive_scores, torch.ones_like(positive_scores))
                loss = loss + bce_criterion(
                    negative_scores, torch.zeros_like(negative_scores)
                )
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate complete common rungs using full-training inference graph
        if epoch in CHECKPOINT_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            rung_metrics[epoch] = val_m
            torch.save(model.state_dict(), ckpt_path + f".epoch{epoch}")
            print(
                f"  [{dataset_name}] epoch={epoch} loss={avg_loss:.4f} "
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
                ### >>> Self_EvolveRec-BLOCK-START: Serialize full inference graph and explicit training treatment
                "graph_weight": GRAPH_WEIGHT,
                "transition_graph": (
                    {"itemnum": itemnum, "scopes": [graph_state["scopes"][0]]}
                    if graph_state is not None
                    else None
                ),
                "loss_type": LOSS_TYPE,
                "negative_count": requested_negatives,
                "training_seed": 42,
                "graph_partition_seed": 0,
                "graph_scope": "full training users; training logits excluded the user's fold",
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    ### >>> Self_EvolveRec-BLOCK-START: Return all rungs instead of independently tuned category winners
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
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Reconstruct identical full-training transition scoring
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        if meta.get("transition_graph") is not None:
            m.attach_graph(meta["transition_graph"], meta["graph_weight"], _device)
        m.eval()
        ### <<< Self_EvolveRec-BLOCK-END
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject unavailable explicit routes rather than silently using another catalog
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
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

    ### >>> Self_EvolveRec-BLOCK-START: Select one common epoch by four-category macro validation HR@10
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": 42,
                "graph_partition_seed": 0,
                "workers": 0,
                "deterministic_algorithms": True,
                "determinism_warn_only": True,
                "loss": LOSS_TYPE,
                "graph_weight": GRAPH_WEIGHT,
                "negative_count": NUM_NEGATIVES if LOSS_TYPE == "ce" else 1,
                "checkpoint_epochs": CHECKPOINT_EPOCHS,
                "graph_limitation": "Full inference graph is denser than excluded-fold training graphs",
            },
            sort_keys=True,
        ),
        flush=True,
    )
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
                "loss": LOSS_TYPE,
                "graph_weight": GRAPH_WEIGHT,
                "negative_count": NUM_NEGATIVES if LOSS_TYPE == "ce" else 1,
            },
            f,
            indent=2,
        )
    write_predict_script(working_dir)
    print(
        f"Selected common epoch={selected_epoch}; "
        f"macro validation HR@10={macro[selected_epoch]:.4f}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


