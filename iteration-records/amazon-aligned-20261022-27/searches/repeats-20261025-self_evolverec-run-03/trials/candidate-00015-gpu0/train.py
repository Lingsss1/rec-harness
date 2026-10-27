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
### >>> Self_EvolveRec-BLOCK-START: Match the host training schedule
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


### >>> Self_EvolveRec-BLOCK-START: Preserve BCE sampling and sample uniform CE negatives without catalog scans
def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    users=None,
    negative_count=None,
    excluded_items=None,
):
    if users is None:
        users = [
            u
            for u, items in user_train.items()
            if len(items) >= 2 and len(set(items)) < itemnum
        ]
    if not users:
        raise ValueError(
            "No training users have a next-item pair and an unseen negative"
        )
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        shape = (maxlen,) if negative_count is None else (maxlen, negative_count)
        neg = np.zeros(shape, dtype=np.int32)
        ts = set(history)
        if negative_count is not None:
            if negative_count < 1:
                raise ValueError("negative_count must be positive")
            excluded = (
                excluded_items[u]
                if excluded_items is not None
                else np.asarray(sorted(ts), dtype=np.int64)
            )
            eligible_count = itemnum - len(excluded)
            count = min(negative_count, eligible_count)
            # Map uniform complement ranks to catalog IDs without scanning N items.
            # Repeated offsets represent consecutive excluded catalog IDs.
            offsets = excluded - np.arange(1, len(excluded) + 1)
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                if negative_count is None:
                    neg_item = random.randint(1, itemnum)
                    while neg_item in ts:
                        neg_item = random.randint(1, itemnum)
                    neg[idx] = neg_item
                elif count:
                    ranks = np.asarray(
                        random.sample(range(eligible_count), count), dtype=np.int64
                    )
                    neg[idx, :count] = (
                        ranks + 1 + np.searchsorted(offsets, ranks, side="right")
                    )
                else:
                    pos[idx] = 0
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Persistent user-excluded supported positive transition lift
FUSION_ALPHAS = (0.0, 0.25, 1.0)
OBJECTIVES = ("baseline_BCE", "uniform_32_negative_CE")
TRANSITION_VERSION = "distinct-user-positive-lift-v1"


def reset_training_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def build_transition_graph(user_train, itemnum, dataset_name):
    users = sorted(user_train)
    order = np.random.RandomState(0).permutation(len(users))
    user_folds = {
        str(users[int(index)]): int(rank % 3) for rank, index in enumerate(order)
    }
    tables = []
    stride = itemnum + 1
    for excluded_fold in range(3):
        frequency = np.zeros(stride, dtype=np.int64)
        edge_counts = defaultdict(int)
        for user in users:
            if user_folds[str(user)] == excluded_fold:
                continue
            items = np.asarray(user_train[user], dtype=np.int64)
            if np.any((items < 1) | (items > itemnum)):
                raise ValueError("Training histories contain invalid catalog IDs")
            np.add.at(frequency, items, 1)
            # Each contributing user can support an ordered edge only once.
            pairs = {
                (int(a), int(b))
                for a, b in zip(items[:-1], items[1:])
                if a != b and a != 0 and b != 0
            }
            for a, b in pairs:
                edge_counts[a * stride + b] += 1
        keys = np.asarray(sorted(edge_counts), dtype=np.int64)
        counts = np.asarray([edge_counts[int(k)] for k in keys], dtype=np.float32)
        tables.append(
            {
                "keys": torch.from_numpy(keys),
                "counts": torch.from_numpy(counts),
                "frequency": torch.from_numpy(frequency),
            }
        )
    return {
        "tables": tables,
        "user_folds": user_folds,
        "fixed_fold": 0,
        "window": 3,
        "rho": 0.5,
        "shrink": 10.0,
        "min_users": 2,
        "cap": 3.0,
        "version": TRANSITION_VERSION,
        "dataset_name": dataset_name,
        "fold_seed": 0,
        "catalog_mapping": torch.arange(stride, dtype=torch.long),
        "identity_namespace": "benchmark_preprocessed_user_id",
        "routing": "known_public_user_fold_else_fixed_fold_0",
    }


class SupportedTransitionTable(nn.Module):
    def __init__(self, itemnum, state, shrink, min_users, cap):
        super().__init__()
        keys = state["keys"].long()
        counts = state["counts"].double()
        frequency = state["frequency"].double()
        q = (frequency + 1.0) / (frequency[1:].sum() + itemnum)
        q[0] = 0.0
        outgoing = torch.zeros(itemnum + 1, dtype=torch.float64)
        if keys.numel():
            outgoing.scatter_add_(0, keys // (itemnum + 1), counts)
        supported = counts >= min_users
        keys, counts = keys[supported], counts[supported]
        source, target = keys // (itemnum + 1), keys % (itemnum + 1)
        # n_i includes ALL distinct-user edge counts, including singleton edges.
        lift = torch.log1p(counts / (shrink * q[target])) + torch.log(
            shrink / (outgoing[source] + shrink)
        )
        self.register_buffer("keys", keys)
        self.register_buffer("lift", lift.clamp(0.0, cap).float())


class TransitionResidual(nn.Module):
    def __init__(
        self,
        itemnum,
        tables,
        user_folds,
        fixed_fold=0,
        window=3,
        rho=0.5,
        shrink=10.0,
        min_users=2,
        cap=3.0,
        alpha=0.0,
        **provenance,
    ):
        super().__init__()
        self.itemnum = int(itemnum)
        self.window = int(window)
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.user_folds = dict(user_folds)
        self.fixed_fold = int(fixed_fold)
        self.tables = nn.ModuleList(
            [
                SupportedTransitionTable(itemnum, state, shrink, min_users, cap)
                for state in tables
            ]
        )
        self.provenance = provenance

    def route(self, user_ids, batch_size):
        if user_ids is None:
            return [self.fixed_fold] * batch_size
        if torch.is_tensor(user_ids):
            user_ids = user_ids.detach().cpu().tolist()
        if len(user_ids) != batch_size:
            raise ValueError("One public user identity is required per sequence")
        return [self.user_folds.get(str(u), self.fixed_fold) for u in user_ids]

    def forward(self, sequence, candidates, user_ids=None, fold_ids=None):
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0).expand(sequence.shape[0], -1)
        if candidates.shape[0] != sequence.shape[0]:
            raise ValueError("Sequence and candidate batch sizes differ")
        result = torch.zeros_like(candidates, dtype=torch.float32)
        width = min(self.window, sequence.shape[1])
        if width == 0:
            return result
        if fold_ids is None:
            fold_ids = self.route(user_ids, sequence.shape[0])
        folds = torch.as_tensor(fold_ids, device=sequence.device, dtype=torch.long)
        if folds.shape != (sequence.shape[0],) or bool(
            ((folds < 0) | (folds >= len(self.tables))).any()
        ):
            raise ValueError("Invalid excluded-fold routing")
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(
            sequence
        )
        positions = positions.masked_fill(sequence == 0, -1)
        recent = positions.topk(width, dim=1, sorted=True).values
        valid = recent >= 0
        anchors = sequence.gather(1, recent.clamp_min(0))
        weights = self.rho ** torch.arange(
            width, device=sequence.device, dtype=torch.float32
        )
        weights = weights.unsqueeze(0) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        for fold, table in enumerate(self.tables):
            rows = torch.nonzero(folds == fold, as_tuple=True)[0]
            if rows.numel() == 0 or table.keys.numel() == 0:
                continue
            selected_candidates = candidates[rows]
            local = torch.zeros_like(selected_candidates, dtype=torch.float32)
            for k in range(width):
                query = (
                    anchors[rows, k : k + 1] * (self.itemnum + 1) + selected_candidates
                )
                index = torch.searchsorted(table.keys, query.contiguous())
                safe_index = index.clamp_max(table.keys.numel() - 1)
                matched = (index < table.keys.numel()) & (
                    table.keys[safe_index] == query
                )
                local += weights[rows, k : k + 1] * matched * table.lift[safe_index]
            result[rows] = local
        return result.masked_fill(candidates == 0, 0.0)


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

    # DEBUG: Expose an explicit neural-only scorer for anonymous diagnostic batches.
    # log2feats retains the initial B x L x D SASRec compatibility contract.
    # Transition diagnostics carry prefix IDs and user provenance separately.
    def predict_score_base(self, features, items):
        # DEBUG: Preserve aligned training scores and support a shared candidate
        # vector without relying on ambiguous embedding broadcasting.
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(-1)
        if items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        raise ValueError(
            "Items must be shared candidates, aligned item IDs, or aligned candidates"
        )

    def predict_score(self, features, items):
        # DEBUG: Restore the public SASRec scoring method required by
        # compatibility adapters. Features contain neural representations only;
        # fold-dependent fusion requires predict(sequence, candidates, user_ids).
        return self.predict_score_base(features, items)

    ### >>> Self_EvolveRec-BLOCK-START: Route identical transition tables during validation and deployment
    def set_transition(self, graph_state, alpha=0.0):
        self.transition = TransitionResidual(
            self.itemnum, alpha=alpha, **graph_state
        ).to(self.item_emb.weight.device)

    def predict(
        self, seq: torch.Tensor, candidates: torch.Tensor, user_ids=None
    ) -> torch.Tensor:
        # DEBUG: Normalize shared and per-sequence candidates before scoring so
        # the neural and transition components use the same B x C layout.
        candidates = torch.as_tensor(candidates, dtype=torch.long, device=seq.device)
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0).expand(seq.shape[0], -1)
        if candidates.ndim != 2 or candidates.shape[0] != seq.shape[0]:
            raise ValueError("Candidates must have shape C or B x C")
        final_feat = self.log2feats(seq)[:, -1, :]
        scores = self.predict_score_base(final_feat, candidates)
        transition = getattr(self, "transition", None)
        if transition is not None and transition.alpha != 0.0:
            scores = scores + transition.alpha * transition(
                seq, candidates, user_ids=user_ids
            )
        return scores

    ### <<< Self_EvolveRec-BLOCK-END


# ── Validation ────────────────────────────────────────────────────────


def evaluate_val(
    model: SASRec,
    user_train: dict,
    user_valid: dict,
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    ### >>> Self_EvolveRec-BLOCK-START: Optional paired local validation records
    num_users: int = 200,
    collect_records: bool = False,
) -> dict:
    records = []
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible validation without perturbing training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Fixed unique validation negatives for every fusion coefficient
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError("Validation requires 99 unique unseen negative items")
            while len(neg_pool) < 99:
                ni = rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Record exact neural and residual components without subtraction error
            if collect_records:
                transition = getattr(model, "transition", None)
                if transition is None:
                    raise RuntimeError(
                        "Paired fusion records require persisted transition tables"
                    )
                final_feat = model.log2feats(seq_t)[:, -1, :]
                # DEBUG: Use the same neural scorer as inference and diagnostics.
                base_t = model.predict_score_base(final_feat, cand_t)
                residual_t = transition(seq_t, cand_t, user_ids=[u])
                scores_t = base_t + transition.alpha * residual_t
                scores = scores_t[0].cpu().tolist()
            else:
                scores = model.predict(seq_t, cand_t, user_ids=[u])[0].cpu().tolist()
            if not np.isfinite(scores).all():
                raise FloatingPointError(f"Nonfinite validation scores for user {u}")
            rank = sorted(range(len(candidates)), key=lambda i: -scores[i]).index(0)
            if collect_records:
                records.append(
                    {
                        "user_id": str(u),
                        "candidates": candidates,
                        "base_scores": base_t[0].cpu().tolist(),
                        "residual_scores": residual_t[0].cpu().tolist(),
                        "fold": transition.route([u], 1)[0],
                        "hit10": int(rank < 10),
                    }
                )
            ### <<< Self_EvolveRec-BLOCK-END

            HR10 += 1.0 if rank < 10 else 0.0
            HR20 += 1.0 if rank < 20 else 0.0
            NDCG10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            NDCG20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0

    n = max(len(val_users), 1)
    model.train()
    ### >>> Self_EvolveRec-BLOCK-START: Return paired records only when explicitly requested
    metrics = {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }
    if collect_records:
        metrics["records"] = records
    return metrics
    ### <<< Self_EvolveRec-BLOCK-END


# ── Per-dataset training ──────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Prespecified separately seeded objective controls
def train_one_dataset(
    dataset_name: str, device: torch.device, data_dir: str, objective="baseline_BCE"
):
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    reset_training_seed()
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

    ### >>> Self_EvolveRec-BLOCK-START: Cache training-only exclusion lists for exact uniform CE sampling
    excluded_items = {}
    for u, items in user_train.items():
        if any(i < 1 or i > itemnum for i in items):
            raise ValueError(
                f"{dataset_name}: invalid training catalog ID for user {u}"
            )
        excluded_items[u] = np.asarray(sorted(set(items)), dtype=np.int64)
    sampling_users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(excluded_items[u]) < itemnum
    ]
    if not sampling_users:
        raise ValueError(f"{dataset_name}: no eligible next-item training users")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ### >>> Self_EvolveRec-BLOCK-START: Keep both objective checkpoints and selection opportunities
    artifact_prefix = f"{dataset_name}_{objective}"
    ckpt_path = os.path.join(working_dir, f"{artifact_prefix}_model.pt")
    checkpoint_checks = []
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Reject empty training schedules and initialize a fresh trial checkpoint
    if NUM_EPOCHS < 1 or NUM_BATCHES < 1:
        raise ValueError("SASREC_EPOCHS and SASREC_BATCHES must both be positive")
    best_hr10 = -1.0
    best_epoch = 0
    torch.save(model.state_dict(), ckpt_path)
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Invoke objective-specific sampling with cached history exclusions
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                users=sampling_users,
                negative_count=32 if objective == "uniform_32_negative_CE" else None,
                excluded_items=excluded_items,
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            ### >>> Self_EvolveRec-BLOCK-START: Preserve BCE and bound activation memory for exact sampled CE
            indices = pos_t != 0
            if objective == "baseline_BCE":
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                pos_labels = torch.ones(pos_logits[indices].shape, device=device)
                neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
                loss = bce_criterion(pos_logits[indices], pos_labels)
                loss = loss + bce_criterion(neg_logits[indices], neg_labels)
            else:
                indices = indices & (neg_t != 0).any(-1)
                if not bool(indices.any()):
                    continue
                features = model.log2feats(seq_t)[indices]
                targets = pos_t[indices]
                negatives = neg_t[indices]
                from torch.utils.checkpoint import checkpoint

                def ce_sum(chunk_features, chunk_targets, chunk_negatives):
                    positive_logits = (
                        chunk_features * model.item_emb(chunk_targets)
                    ).sum(-1, keepdim=True)
                    negative_logits = torch.bmm(
                        model.item_emb(chunk_negatives),
                        chunk_features.unsqueeze(-1),
                    ).squeeze(-1)
                    negative_logits = negative_logits.masked_fill(
                        chunk_negatives == 0, float("-inf")
                    )
                    logits = torch.cat((positive_logits, negative_logits), dim=1)
                    labels = torch.zeros(
                        logits.shape[0], device=logits.device, dtype=torch.long
                    )
                    return nn.functional.cross_entropy(logits, labels, reduction="sum")

                # Recompute candidate embeddings during backward instead of retaining
                # B*L*32*H embeddings. Sum chunks before normalizing so every
                # valid position has exactly the same weight as ordinary CE.
                chunk_losses = []
                for start in range(0, features.shape[0], 512):
                    stop = start + 512
                    chunk_losses.append(
                        checkpoint(
                            ce_sum,
                            features[start:stop],
                            targets[start:stop],
                            negatives[start:stop],
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    )
                loss = torch.stack(chunk_losses).sum() / features.shape[0]
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the final epoch for short training schedules
        should_validate = epoch % 5 == 0 or epoch == NUM_EPOCHS
        ### <<< Self_EvolveRec-BLOCK-END
        if should_validate:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### >>> Self_EvolveRec-BLOCK-START: Record the unchanged baseline checkpoint rule
            hr10 = val_m["HR@10"]
            checkpoint_checks.append({"epoch": epoch, "HR@10": hr10})
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

    ### >>> Self_EvolveRec-BLOCK-START: Compare frozen checkpoints using identical persisted excluded-fold tables
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    graph_path = os.path.join(working_dir, f"{dataset_name}_transition.pt")
    if objective == OBJECTIVES[0]:
        graph_state = build_transition_graph(user_train, itemnum, dataset_name)
        torch.save(graph_state, graph_path)
    else:
        if not os.path.isfile(graph_path):
            raise FileNotFoundError(graph_path)
        graph_state = torch.load(graph_path, map_location="cpu", weights_only=True)
        if (
            graph_state["version"] != TRANSITION_VERSION
            or graph_state["dataset_name"] != dataset_name
            or not torch.equal(
                graph_state["catalog_mapping"], torch.arange(itemnum + 1)
            )
        ):
            raise ValueError("Transition artifact provenance does not match this run")
    model.set_transition(graph_state)
    fusion_metrics = {}
    for alpha in FUSION_ALPHAS:
        model.transition.alpha = alpha
        metrics = evaluate_val(
            model,
            user_train,
            user_valid,
            itemnum,
            device,
            MAXLEN,
            collect_records=True,
        )
        fusion_metrics[alpha] = metrics
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "objective": objective,
                    "experiment": TRANSITION_VERSION,
                    "alpha": alpha,
                    "routing": graph_state["routing"],
                    "checkpoint_epoch": best_epoch,
                    "metrics": {k: v for k, v in metrics.items() if k != "records"},
                }
            ),
            flush=True,
        )
    model.transition.alpha = 0.0
    with open(
        os.path.join(working_dir, f"{artifact_prefix}_fusion_validation.json"), "w"
    ) as f:
        json.dump(
            {
                "objective": objective,
                "checkpoint": os.path.basename(ckpt_path),
                "checkpoint_epoch": best_epoch,
                "checkpoint_checks": checkpoint_checks,
                "table_version": TRANSITION_VERSION,
                "fusion_metrics": fusion_metrics,
                "source": "local_validation_only_host_evaluation_is_independent",
            },
            f,
            indent=2,
        )
    meta_path = os.path.join(working_dir, f"{artifact_prefix}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Defer coefficient selection until all four categories are evaluated
    return fusion_metrics
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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

_models: dict = {}
_metas: dict = {}
_device = None


### >>> Self_EvolveRec-BLOCK-START: Preserve evaluator randomness when importing the seeded training module
def _load_model_class():
    import random
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        spec = importlib.util.spec_from_file_location(
            f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the trained SASRec architecture from train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SASRec
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
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
        ### >>> Self_EvolveRec-BLOCK-START: Require complete artifacts for every dataset route
        for required_path in (meta_path, ckpt_path):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(required_path)
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore the selected transition residual alongside the neural checkpoint
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        graph_path = os.path.join(_HERE, f"{ds}_transition.pt")
        alpha = float(meta.get("transition_alpha", 0.0))
        if os.path.isfile(graph_path):
            graph_state = torch.load(graph_path, map_location="cpu", weights_only=True)
            m.set_transition(graph_state, alpha=alpha)
        elif alpha != 0.0:
            raise FileNotFoundError(graph_path)
        m.eval()
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes while retaining the legacy default
    if not _models:
        raise RuntimeError("No trained dataset models were loaded")
    ds = dataset_name if dataset_name is not None else next(iter(_models))
    if ds not in _models:
        raise ValueError(f"Unknown dataset route: {ds!r}")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Exact itemwise fusion with persisted public-user fold routing
    if not candidates:
        return []
    with torch.no_grad():
        # Encode once; every chunk uses the same prefix and itemwise residual.
        final_feat = model.log2feats(seq_t)[:, -1, :]
        transition = getattr(model, "transition", None)
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor([candidates[start:start + 4096]], dtype=torch.long, device=_device)
            # DEBUG: Keep the generated predictor on the shared scoring contract.
            chunk_scores = model.predict_score_base(final_feat, cand_t)
            if transition is not None and transition.alpha != 0.0:
                chunk_scores = chunk_scores + transition.alpha * transition(seq_t, cand_t, user_ids=[user_id])
            scores.extend(chunk_scores[0].cpu().tolist())
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

    ### >>> Self_EvolveRec-BLOCK-START: Freeze six configurations and select using four-category macro HR10
    import shutil

    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError("This matched task requires all four Amazon datasets")
    # Ordering implements the prespecified tie break: smaller weight, then BCE.
    configurations = [
        (objective, alpha) for alpha in FUSION_ALPHAS for objective in OBJECTIVES
    ]
    plan = {
        "experiment": TRANSITION_VERSION,
        "objectives": list(OBJECTIVES),
        "weights": list(FUSION_ALPHAS),
        "configurations": configurations,
        "datasets": DATASETS,
        "epochs": NUM_EPOCHS,
        "training_seed": 42,
        "fold_seed": 0,
        "folds": 3,
        "smoothing_alpha": 10.0,
        "min_users": 2,
        "lift_cap": 3.0,
        "recent_K": 3,
        "recency_rho": 0.5,
        "ce_negatives": 32,
        "routing": "known_public_user_fold_else_fixed_fold_0",
        "tie_break": "lower_weight_then_baseline_BCE",
        "checkpoint_rule": "strict_HR10_improvement_every_5_epochs_or_final; patience_times_5",
        "complete": False,
        "completed_runs": [],
    }
    plan_path = os.path.join(working_dir, "experiment_plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    results = {objective: {} for objective in OBJECTIVES}
    for objective in OBJECTIVES:
        for ds in DATASETS:
            print(f"Training {ds}: {objective}", flush=True)
            results[objective][ds] = train_one_dataset(
                ds, device, DATA_DIR, objective=objective
            )
            plan["completed_runs"].append([objective, ds])
            with open(plan_path, "w") as f:
                json.dump(plan, f, indent=2)

    def configuration_key(objective, alpha):
        return f"{objective}:lambda={alpha:g}"

    macro = {
        configuration_key(objective, alpha): sum(
            results[objective][ds][alpha]["HR@10"] for ds in DATASETS
        )
        / len(DATASETS)
        for objective, alpha in configurations
    }
    selected_objective, selected_alpha = configurations[0]
    for objective, alpha in configurations[1:]:
        key = configuration_key(objective, alpha)
        current = configuration_key(selected_objective, selected_alpha)
        if macro[key] > macro[current] + 1e-12:
            selected_objective, selected_alpha = objective, alpha

    # Report every configuration against the same frozen BCE/no-residual control.
    paired = {}
    for objective, alpha in configurations:
        per_category = {}
        bootstrap = np.zeros(10000, dtype=np.float64)
        rng = np.random.default_rng(42)
        available = True
        for ds in DATASETS:
            baseline = results["baseline_BCE"][ds][0.0]["records"]
            variant = results[objective][ds][alpha]["records"]
            aligned = len(baseline) == len(variant) and len(baseline) > 0
            if aligned:
                aligned = all(
                    b["user_id"] == v["user_id"] and b["candidates"] == v["candidates"]
                    for b, v in zip(baseline, variant)
                )
            if not aligned:
                available = False
                per_category[ds] = {
                    "available": False,
                    "reason": "No aligned paired candidates",
                }
                continue
            before = np.asarray([r["hit10"] for r in baseline], dtype=np.int8)
            after = np.asarray([r["hit10"] for r in variant], dtype=np.int8)
            difference = after - before
            corrections = int(((before == 0) & (after == 1)).sum())
            regressions = int(((before == 1) & (after == 0)).sum())
            per_category[ds] = {
                "available": True,
                "users": len(before),
                "corrections": corrections,
                "regressions": regressions,
                "discordant_hits": corrections + regressions,
                "hit_change": int(difference.sum()),
                "HR10_change": float(difference.mean()),
            }
            indices = rng.integers(0, len(difference), size=(10000, len(difference)))
            bootstrap += difference[indices].mean(1) / len(DATASETS)
        key = configuration_key(objective, alpha)
        baseline_key = configuration_key("baseline_BCE", 0.0)
        paired[key] = {
            "per_category": per_category,
            "macro_HR10_change": macro[key] - macro[baseline_key],
            "paired_bootstrap_95_percentile_interval": (
                np.percentile(bootstrap, [2.5, 97.5]).tolist() if available else None
            ),
            "interval_scope": "exploratory_local_validation_conditional_on_frozen_models_and_candidates",
            "host_interval": "unavailable_without_host_per_user_outputs",
        }

    for ds in DATASETS:
        prefix = f"{ds}_{selected_objective}"
        with open(os.path.join(working_dir, f"{prefix}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        meta.update(
            {
                "transition_alpha": selected_alpha,
                "objective": selected_objective,
                "transition_version": TRANSITION_VERSION,
                "routing": plan["routing"],
            }
        )
        shutil.copyfile(
            os.path.join(working_dir, f"{prefix}_model.pt"),
            os.path.join(working_dir, f"{ds}_model.pt"),
        )
        with open(os.path.join(working_dir, f"{ds}_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
    with open(os.path.join(working_dir, "fusion_selection.json"), "w") as f:
        json.dump(
            {
                "experiment": TRANSITION_VERSION,
                "source": "local_validation_only_host_evaluation_is_independent",
                "macro_validation_hr10": macro,
                "selected_objective": selected_objective,
                "selected_alpha": selected_alpha,
                "paired_comparisons": paired,
                "per_dataset": {
                    objective: {
                        ds: {
                            alpha: {k: v for k, v in metrics.items() if k != "records"}
                            for alpha, metrics in results[objective][ds].items()
                        }
                        for ds in DATASETS
                    }
                    for objective in OBJECTIVES
                },
            },
            f,
            indent=2,
        )
    write_predict_script(working_dir)
    plan["complete"] = True
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    selected_key = configuration_key(selected_objective, selected_alpha)
    print(
        f"All datasets done. Selected {selected_key}; "
        f"local macro validation HR@10={macro[selected_key]:.4f}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Record static review scope
# DEBUG: No traceback was supplied; these fixes address the visible mismatch
# between the transition model and diagnostics left over from a cosine model.
# Anonymous neural diagnostics and identified transition comparisons are separate.
# The encoder, inference formula, fixed grid and checkpoint rule are unchanged.
# The predictor writer imports this same SASRec class and transition scorer.
# No execution, generated-predictor parity test or measured gain is claimed.
if __name__ == "__main__":
    main()
### <<< Self_EvolveRec-BLOCK-END


