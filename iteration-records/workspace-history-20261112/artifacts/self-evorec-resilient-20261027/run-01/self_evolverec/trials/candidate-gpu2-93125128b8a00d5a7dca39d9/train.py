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
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched training horizon
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Explicit variant controls and required public data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
VARIANTS = {
    "baseline": (False, False),
    "objective-only": (False, True),
    "fusion-only": (True, False),
    "combined": (True, True),
}
VARIANT = os.environ.get("SASREC_VARIANT", "combined")
if VARIANT not in VARIANTS:
    raise ValueError(f"Unknown SASREC_VARIANT: {VARIANT}")
USE_FUSION, USE_CANDIDATE_CE = VARIANTS[VARIANT]
NEGATIVE_COUNT = 32
RECENT_WINDOW = 5
RECENT_DECAY = math.exp(-1.0)
if NUM_EPOCHS < 1:
    raise ValueError("SASREC_EPOCHS must be positive")
CHECKPOINT_EPOCHS = sorted(
    {e for e in (50, 100, 150, 200, NUM_EPOCHS) if e <= NUM_EPOCHS}
)
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Require complete four-category configurations
DATASETS = DEFAULT_DATASETS.copy()
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

### >>> Self_EvolveRec-BLOCK-START: Bounded training-negative sampling and memory-efficient candidate CE


def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    users=None,
    sample_negatives: bool = True,
    candidate_count: int = 0,
):
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("Dataset has no training histories with two items")
    batch_seq, batch_pos, batch_neg = [], [], []
    batch_candidates, batch_pool_sizes = [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history) if sample_negatives else set()
        can_sample = sample_negatives and len(ts) < itemnum
        negative_pool = None
        if can_sample and len(ts) > itemnum // 2:
            negative_pool = [i for i in range(1, itemnum + 1) if i not in ts]
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0 and can_sample:
                if negative_pool is not None:
                    neg_item = random.choice(negative_pool)
                else:
                    neg_item = random.randint(1, itemnum)
                    attempts = 0
                    while neg_item in ts and attempts < 16:
                        neg_item = random.randint(1, itemnum)
                        attempts += 1
                    if neg_item in ts:
                        negative_pool = [
                            i for i in range(1, itemnum + 1) if i not in ts
                        ]
                        neg_item = random.choice(negative_pool)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx == -1:
                break
        if candidate_count:
            candidates, pool_size = sample_training_candidates(
                history, pos, itemnum, candidate_count
            )
            batch_candidates.append(candidates)
            batch_pool_sizes.append(pool_size)
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    ordinary_batch = np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)
    if candidate_count:
        return (
            *ordinary_batch,
            {
                "negatives": np.stack(batch_candidates),
                "pool_sizes": np.asarray(batch_pool_sizes, dtype=np.int64),
            },
        )
    return ordinary_batch


def sample_training_candidates(history, positive, itemnum, count=32):
    """Unique uniform negatives excluding the complete training history.

    This runs inside the user-aware sampler. Held-out interactions are never
    consulted. Zero slots are padding and must be masked from candidate CE.
    """
    seen = np.unique(np.asarray(history, dtype=np.int64))
    seen = seen[(seen >= 1) & (seen <= itemnum)]
    pool_size = itemnum - len(seen)
    result = np.zeros((len(positive), count), dtype=np.int64)
    positions = np.flatnonzero(positive)
    k = min(count, pool_size)
    if not len(positions) or k == 0:
        return result, pool_size

    # Each next-item target belongs to the complete training history.
    if not np.isin(positive[positions], seen).all():
        raise ValueError("Training target missing from complete training history")

    allowed = None
    if pool_size <= 4 * k or len(seen) > itemnum // 2:
        allowed = np.setdiff1d(
            np.arange(1, itemnum + 1, dtype=np.int64),
            seen,
            assume_unique=True,
        )
        draws = np.stack(
            [np.random.choice(allowed, size=k, replace=False) for _ in positions]
        )
    else:
        draws = np.random.randint(1, itemnum + 1, size=(len(positions), k))

        def rejected(values):
            order = np.argsort(values, axis=1, kind="stable")
            ordered = np.take_along_axis(values, order, axis=1)
            duplicate = np.zeros_like(values, dtype=bool)
            duplicate[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
            bad = np.zeros_like(duplicate)
            np.put_along_axis(bad, order, duplicate, axis=1)
            return bad | np.isin(values, seen)

        bad = rejected(draws)
        for _ in range(8):
            n_bad = int(bad.sum())
            if n_bad == 0:
                break
            draws[bad] = np.random.randint(1, itemnum + 1, size=n_bad)
            bad = rejected(draws)
        if bad.any():
            allowed = np.setdiff1d(
                np.arange(1, itemnum + 1, dtype=np.int64),
                seen,
                assume_unique=True,
            )
            for row in np.flatnonzero(bad.any(axis=1)):
                draws[row] = np.random.choice(allowed, size=k, replace=False)

    result[positions, :k] = draws
    return result, pool_size


def candidate_cross_entropy(model, features, positive, negatives, chunk_size=1024):
    """Exact position-mean candidate CE with bounded embedding activation memory."""
    from torch.utils.checkpoint import checkpoint

    if features.shape[0] == 0:
        raise ValueError("Candidate CE requires at least one eligible position")

    def chunk_loss(feature_chunk, positive_chunk, negative_chunk, item_weight):
        positive_embeddings = nn.functional.embedding(positive_chunk, item_weight)
        negative_embeddings = nn.functional.embedding(negative_chunk, item_weight)
        positive_score = (feature_chunk * positive_embeddings).sum(-1, keepdim=True)
        negative_score = torch.einsum("nd,nkd->nk", feature_chunk, negative_embeddings)
        negative_score = negative_score.masked_fill(negative_chunk.eq(0), -torch.inf)
        logits = torch.cat((positive_score, negative_score), dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return nn.functional.cross_entropy(logits, labels, reduction="sum")

    if chunk_size < 1:
        raise ValueError("Candidate CE chunk_size must be positive")
    # A single chunk already satisfies the activation-memory bound. Avoid
    # recomputing its embedding lookups and logits during backward.
    if features.shape[0] <= chunk_size:
        return (
            chunk_loss(features, positive, negatives, model.item_emb.weight)
            / features.shape[0]
        )

    total = features.new_zeros(())
    for start in range(0, features.shape[0], chunk_size):
        stop = start + chunk_size
        total = total + checkpoint(
            chunk_loss,
            features[start:stop],
            positive[start:stop],
            negatives[start:stop],
            model.item_emb.weight,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return total / features.shape[0]


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
    ### >>> Self_EvolveRec-BLOCK-START: Serialize explicit recent-fusion architecture settings
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_fusion: bool = False,
        recent_window: int = 5,
        recent_decay: float = math.exp(-1.0),
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

        ### >>> Self_EvolveRec-BLOCK-START: Initially inactive gated recent-history branch
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.recent_fusion = recent_fusion
        self.recent_window = recent_window
        self.recent_decay = recent_decay
        if recent_window < 1 or not 0.0 < recent_decay <= 1.0:
            raise ValueError("Invalid recent-history window or decay")
        if recent_fusion:
            self.recent_projection = nn.Linear(hidden_units, hidden_units, bias=False)
            self.recent_gate = nn.Linear(3 * hidden_units, 1)
            self.recent_alpha = nn.Parameter(torch.zeros(()))
            nn.init.normal_(self.recent_projection.weight, std=hidden_units**-0.5)
            nn.init.normal_(self.recent_gate.weight, std=0.01)
            nn.init.zeros_(self.recent_gate.bias)

    def recent_mean(self, seq: torch.Tensor) -> torch.Tensor:
        # DEBUG: count observed items rather than physical slots; padding gaps
        # must not shorten the five-observation window or change its decay.
        batch_size, length = seq.shape
        embeddings = self.item_emb(seq)
        if length == 0:
            return embeddings
        valid = seq.ne(0)
        ranks = valid.long().cumsum(dim=1)
        positions = torch.arange(length, device=seq.device).expand(batch_size, -1)
        # Slot zero denotes an empty prefix. Real observations have unique ranks.
        rank_to_position = torch.zeros(
            (batch_size, length + 1), dtype=torch.long, device=seq.device
        )
        rank_to_position.scatter_add_(1, ranks, (positions + 1) * valid.long())
        padded_embeddings = torch.cat(
            (embeddings.new_zeros((batch_size, 1, self.hidden_units)), embeddings),
            dim=1,
        )
        numerator = torch.zeros_like(embeddings)
        denominator = embeddings.new_zeros((batch_size, length, 1))
        for lag in range(min(self.recent_window, length)):
            requested_rank = ranks - lag
            available = requested_rank.gt(0).unsqueeze(-1)
            source_positions = rank_to_position.gather(1, requested_rank.clamp_min(0))
            recent_embedding = padded_embeddings.gather(
                1, source_positions.unsqueeze(-1).expand(-1, -1, self.hidden_units)
            )
            weight = self.recent_decay**lag
            numerator = numerator + weight * recent_embedding * available
            denominator = denominator + weight * available
        return numerator / denominator.clamp_min(1e-12)
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

        ### >>> Self_EvolveRec-BLOCK-START: Shared fused features for training, prediction and diagnostic views
        features = self.last_layernorm(seq_emb)
        if self.recent_fusion:
            recent = self.recent_mean(seq)
            gate = torch.sigmoid(
                self.recent_gate(
                    torch.cat((features, recent, features - recent), dim=-1)
                )
            )
            features = features + torch.tanh(
                self.recent_alpha
            ) * gate * self.recent_projection(recent)
        return features * (~timeline_mask).unsqueeze(-1)
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
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(candidates)
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)


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
    ### >>> Self_EvolveRec-BLOCK-START: Deterministic validation without resetting training RNG
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
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
            ### >>> Self_EvolveRec-BLOCK-START: Keep validation candidates unique and locally seeded
            if itemnum - len(interacted) < 99:
                raise ValueError("Validation requires 99 unique unseen candidates")
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
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    ### >>> Self_EvolveRec-BLOCK-START: Seed each category once and activate the selected architecture
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    training_users = [u for u, items in user_train.items() if len(items) >= 2]
    if not training_users:
        raise ValueError(f"{dataset_name}: no eligible training users")
    if NUM_BATCHES < 1 or BATCH_SIZE < 1 or MAXLEN < 1:
        raise ValueError("Batch count, batch size and maximum length must be positive")
    probe_users = sorted(training_users)[:32]
    probe_seq = torch.zeros((len(probe_users), MAXLEN), dtype=torch.long, device=device)
    for row, user in enumerate(probe_users):
        history = user_train[user][:-1][-MAXLEN:]
        probe_seq[row, -len(history) :] = torch.tensor(
            history, dtype=torch.long, device=device
        )
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        recent_fusion=USE_FUSION,
        recent_window=RECENT_WINDOW,
        recent_decay=RECENT_DECAY,
    ).to(device)
    ### <<< Self_EvolveRec-BLOCK-END

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Retain common epoch checkpoints for joint macro selection
    checkpoint_metrics = {}
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        skipped_positions = 0
        updates = 0
        ### <<< Self_EvolveRec-BLOCK-END

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: User-aware unique candidate CE with explicit nonfinite-loss failure
            sampled = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                users=training_users,
                sample_negatives=True,
                candidate_count=NEGATIVE_COUNT if USE_CANDIDATE_CE else 0,
            )
            seq, pos, neg = sampled[:3]
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            if USE_CANDIDATE_CE:
                payload = sampled[3]
                negatives = torch.as_tensor(
                    payload["negatives"], dtype=torch.long, device=device
                )
                valid_positions = seq_t.ne(0) & pos_t.ne(0)
                indices = valid_positions & negatives.ne(0).any(dim=-1)
                skipped_positions += int((valid_positions & ~indices).sum().item())
                if not bool(indices.any().item()):
                    continue
                features = model(seq_t)[indices]
                loss = candidate_cross_entropy(
                    model, features, pos_t[indices], negatives[indices]
                )
            else:
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                indices = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
                if not bool(indices.any().item()):
                    continue
                loss = bce_criterion(
                    pos_logits[indices], torch.ones_like(pos_logits[indices])
                )
                loss = loss + bce_criterion(
                    neg_logits[indices], torch.zeros_like(neg_logits[indices])
                )
            if not bool(torch.isfinite(loss.detach()).item()):
                raise FloatingPointError(
                    f"Nonfinite loss: dataset={dataset_name}, "
                    f"variant={VARIANT}, epoch={epoch}, update={updates}"
                )
            updates += 1
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        ### >>> Self_EvolveRec-BLOCK-START: Sparse shared checkpoint opportunities without category-wise early stopping
        avg_loss = total_loss / max(updates, 1)
        if epoch in CHECKPOINT_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            checkpoint_metrics[epoch] = val_m
            epoch_path = os.path.join(working_dir, f"{dataset_name}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_path)
            branch_metrics = {}
            if model.recent_fusion:
                model.eval()
                with torch.no_grad():
                    valid_probe = probe_seq.ne(0)
                    recent = model.recent_mean(probe_seq)
                    projected = model.recent_projection(recent)
                    # Use the same public-history probe at every checkpoint.
                    model.recent_fusion = False
                    try:
                        baseline_features = model.log2feats(probe_seq)
                    finally:
                        model.recent_fusion = True
                    actual_gate = torch.sigmoid(
                        model.recent_gate(
                            torch.cat(
                                (baseline_features, recent, baseline_features - recent),
                                -1,
                            )
                        )
                    )
                    correction = (
                        torch.tanh(model.recent_alpha) * actual_gate * projected
                    )
                    branch_metrics = {
                        "alpha": float(model.recent_alpha.item()),
                        "baseline_norm": float(
                            baseline_features[valid_probe].norm(dim=-1).mean().item()
                        ),
                        "recent_norm": float(
                            recent[valid_probe].norm(dim=-1).mean().item()
                        ),
                        "correction_norm": float(
                            correction[valid_probe].norm(dim=-1).mean().item()
                        ),
                        "gate_mean": float(actual_gate[valid_probe].mean().item()),
                        "effective_gate_mean": float(
                            (torch.tanh(model.recent_alpha) * actual_gate)[valid_probe]
                            .abs()
                            .mean()
                            .item()
                        ),
                        "residual_to_backbone_norm_ratio": float(
                            (
                                correction[valid_probe].norm(dim=-1)
                                / baseline_features[valid_probe]
                                .norm(dim=-1)
                                .clamp_min(1e-12)
                            )
                            .mean()
                            .item()
                        ),
                        "observed_positions": int(valid_probe.sum().item()),
                    }
                model.train()
            print(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "variant": VARIANT,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "validation": val_m,
                        "skipped_empty_candidate_positions": skipped_positions,
                        "branch": branch_metrics,
                    }
                ),
                flush=True,
            )
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist architecture and all inspected validation checkpoints
                "maxlen": MAXLEN,
                "recent_fusion": USE_FUSION,
                "recent_window": RECENT_WINDOW,
                "recent_decay": RECENT_DECAY,
                "variant": VARIANT,
            },
            f,
        )

    with open(os.path.join(working_dir, f"{dataset_name}_validation.json"), "w") as f:
        json.dump(checkpoint_metrics, f, indent=2)
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
DATASETS = DEFAULT_DATASETS.copy()
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
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required category artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model or metadata for required dataset {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained fusion branch in generated prediction
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            recent_fusion=meta.get("recent_fusion", False),
            recent_window=meta.get("recent_window", 5),
            recent_decay=meta["recent_decay"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing requested routes instead of using another category
    if dataset_name is not None:
        if dataset_name not in _models:
            raise KeyError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
    ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Joint macro checkpoint selection and generated-predictor parity
    import importlib.util
    import shutil
    import time

    configuration = {
        "variant": VARIANT,
        "training_seed": 42,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT_RATE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "adam_betas": [0.9, 0.98],
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "epochs": NUM_EPOCHS,
        "maxlen": MAXLEN,
        "candidate_count": NEGATIVE_COUNT,
        "recent_window": RECENT_WINDOW,
        "recent_decay": RECENT_DECAY,
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
        "datasets": DATASETS,
    }
    print(json.dumps(configuration), flush=True)
    with open(os.path.join(working_dir, "configuration.json"), "w") as f:
        json.dump(configuration, f, indent=2)

    started = time.perf_counter()
    all_metrics = {}
    for ds in DATASETS:
        print(f"Training {ds}; variant={VARIANT}", flush=True)
        all_metrics[ds] = train_one_dataset(ds, device, DATA_DIR)

    common_epochs = set.intersection(*(set(all_metrics[ds]) for ds in DATASETS))
    if not common_epochs:
        raise RuntimeError("No complete four-category checkpoint is available")
    macro_by_epoch = {
        epoch: sum(all_metrics[ds][epoch]["HR@10"] for ds in DATASETS) / len(DATASETS)
        for epoch in sorted(common_epochs)
    }
    # The local evaluator is not established as identical to the pinned host.
    # Export the predeclared final epoch; only the host may select by pinned HR@10.
    selected_epoch = NUM_EPOCHS
    if selected_epoch not in common_epochs:
        raise RuntimeError(
            "The predeclared final four-category checkpoint is incomplete"
        )
    for ds in DATASETS:
        shutil.copyfile(
            os.path.join(working_dir, f"{ds}_epoch_{selected_epoch}.pt"),
            os.path.join(working_dir, f"{ds}_model.pt"),
        )
    selection = {
        "variant": VARIANT,
        "selected_epoch": selected_epoch,
        "selection_policy": "fixed_final_epoch; pinned metric evaluated by host",
        "local_proxy_macro_HR@10_by_epoch": macro_by_epoch,
        "local_proxy_per_category": {
            ds: all_metrics[ds][selected_epoch] for ds in DATASETS
        },
    }
    with open(os.path.join(working_dir, "selection.json"), "w") as f:
        json.dump(selection, f, indent=2)

    write_predict_script(working_dir)
    spec = importlib.util.spec_from_file_location(
        "generated_predictor_parity", os.path.join(working_dir, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor._ensure_loaded()
    parity = {}
    for ds in DATASETS:
        with open(os.path.join(working_dir, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        reference = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            recent_fusion=meta["recent_fusion"],
            recent_window=meta["recent_window"],
            recent_decay=meta["recent_decay"],
        ).to(device)
        reference.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{ds}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        reference.eval()
        histories = _read_split(os.path.join(DATA_DIR, f"{ds}_train.txt"))
        maximum_error = 0.0
        candidates = list(range(1, min(meta["itemnum"], 100) + 1))
        for user in sorted(histories)[:5]:
            history = histories[user][-meta["maxlen"] :]
            padded = [0] * (meta["maxlen"] - len(history)) + history
            with torch.no_grad():
                expected = reference.predict(
                    torch.tensor([padded], dtype=torch.long, device=device),
                    torch.tensor([candidates], dtype=torch.long, device=device),
                )[0].cpu()
            actual = torch.tensor(
                predictor.predict(user, histories[user], candidates, dataset_name=ds)
            )
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            maximum_error = max(
                maximum_error, float((actual - expected).abs().max().item())
            )
        parity[ds] = maximum_error
        del reference
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with open(os.path.join(working_dir, "verification.json"), "w") as f:
        json.dump(
            {
                "predictor_max_absolute_error": parity,
                "training_and_verification_wall_seconds": time.perf_counter() - started,
            },
            f,
            indent=2,
        )
    print(json.dumps(selection), flush=True)
    ### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


