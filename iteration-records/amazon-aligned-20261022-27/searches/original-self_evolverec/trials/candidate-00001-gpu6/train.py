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
### >>> Self_EvolveRec-BLOCK-START: Match the host training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Required data boundary and independently selectable objective and history ablations
DATA_DIR = os.environ["GAGC_DATA_DIR"]

# Each invocation trains one declared variant across all four datasets.
# Start with CE alone to isolate the objective before testing the history branch.
VARIANT = os.environ.get("SASREC_VARIANT", "ce")
VARIANT_CONFIGS = {
    "baseline": ("none", "bce"),
    "ce": ("none", "ce"),
    "recency_ce": ("recency", "ce"),
    "recurrence_ce": ("recurrence", "ce"),
    "recurrence_bce": ("recurrence", "bce"),
    "uniform_ce": ("uniform", "ce"),
}
if VARIANT not in VARIANT_CONFIGS:
    raise ValueError(f"Unknown SASREC_VARIANT: {VARIANT}")
HISTORY_MODE, OBJECTIVE = VARIANT_CONFIGS[VARIANT]
NUM_NEGATIVES = 32 if OBJECTIVE == "ce" else 1
CE_CHUNK_SIZE = 512
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four dataset routes
DATASETS = list(DEFAULT_DATASETS)
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


### >>> Self_EvolveRec-BLOCK-START: Seeded uniform negatives from complete training-history complements
class TrainingSampler:
    def __init__(self, user_train, itemnum, seed=42):
        self.user_train = user_train
        self.itemnum = itemnum
        self.rng = np.random.default_rng(seed)
        self.users = []
        self.seen = {}
        self.complements = {}
        self.skipped_users = 0
        self.skipped_supervised_positions = 0
        for u, history in user_train.items():
            if len(history) < 2:
                continue
            seen = np.unique(np.asarray(history, dtype=np.int64))
            seen = seen[(seen >= 1) & (seen <= itemnum)]
            if len(seen) == itemnum:
                self.skipped_users += 1
                self.skipped_supervised_positions += len(history) - 1
                continue
            self.users.append(u)
            self.seen[u] = seen
            # Complement ranks are mapped during sampling without allocating
            # a full catalog-sized complement for each dense user.
        if not self.users:
            raise ValueError(
                "No users with both supervised positions and unseen training items"
            )

    def sample(self, batch_size, maxlen, num_negatives=1):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
        user_indices = self.rng.integers(len(self.users), size=batch_size)
        for row, user_index in enumerate(user_indices):
            u = self.users[int(user_index)]
            history = self.user_train[u]
            length = min(len(history) - 1, maxlen)
            seq[row, -length:] = history[-length - 1 : -1]
            pos[row, -length:] = history[-length:]
            shape = (length, num_negatives)
            # Map uniform complement ranks to catalog IDs exactly. This avoids
            # rejection loops and repeated membership scans, retaining duplicates.
            seen = self.seen[u]
            ranks = self.rng.integers(1, self.itemnum - len(seen) + 1, size=shape)
            blocked_ranks = seen - np.arange(len(seen), dtype=np.int64)
            draws = ranks + np.searchsorted(blocked_ranks, ranks, side="right")
            # Duplicate independent draws deliberately remain in the candidate set.
            neg[row, -length:] = draws
        return seq, pos, neg


_diagnostic_sampler = None
_diagnostic_sampler_source = None


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    # Preserve the original B x L public sampling contract for diagnosis adapters.
    global _diagnostic_sampler, _diagnostic_sampler_source
    source = (id(user_train), itemnum)
    if source != _diagnostic_sampler_source:
        _diagnostic_sampler = TrainingSampler(user_train, itemnum, seed=42)
        _diagnostic_sampler_source = source
    seq, pos, neg = _diagnostic_sampler.sample(batch_size, maxlen, 1)
    return seq, pos, neg[:, :, 0]


def candidate_set_loss(model, features, positives, negatives):
    from torch.utils.checkpoint import checkpoint

    valid = positives.ne(0)
    features = features[valid]
    positives = positives[valid]
    negatives = negatives[valid]
    count = positives.numel()
    if count == 0:
        return None

    def chunk_loss(feats, candidate_ids):
        embeddings = model.item_emb(candidate_ids)
        logits = torch.bmm(embeddings, feats.unsqueeze(-1)).squeeze(-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return nn.functional.cross_entropy(logits, labels, reduction="sum")

    loss = features.new_zeros(())
    for start in range(0, count, CE_CHUNK_SIZE):
        stop = start + CE_CHUNK_SIZE
        candidates = torch.cat(
            (positives[start:stop, None], negatives[start:stop]), dim=1
        )
        # Recompute embeddings/logits on backward instead of retaining every
        # chunk's candidate activations. Temperature is fixed at one.
        loss = loss + checkpoint(
            chunk_loss, features[start:stop], candidates, use_reentrant=False
        )
    return loss / count


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
    ### >>> Self_EvolveRec-BLOCK-START: Explicit serialized history architecture
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        history_mode: str = "none",
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

        ### >>> Self_EvolveRec-BLOCK-START: Causal recurrence support and norm-bounded gated residual
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if history_mode not in {"none", "uniform", "recency", "recurrence"}:
            raise ValueError(f"Unknown history_mode: {history_mode}")
        self.history_mode = history_mode
        if history_mode != "none":
            self.history_projection = nn.Linear(hidden_units, hidden_units, bias=False)
            self.history_gate = nn.Parameter(torch.zeros(()))
            if history_mode == "recurrence":
                self.support_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
            if history_mode in {"recency", "recurrence"}:
                self.recency_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))

    def _history_residual(self, seq, backbone):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device)
        valid = seq.ne(0)
        raw_embeddings = self.item_emb(seq)
        # All pair matrices have axes (position j, position k).
        same_item = seq.unsqueeze(2).eq(seq.unsqueeze(1))
        earlier = positions.unsqueeze(0) < positions.unsqueeze(1)
        earlier_same = same_item & earlier.unsqueeze(0) & valid.unsqueeze(1)
        first_occurrence = valid & ~earlier_same.any(dim=-1)

        if self.history_mode == "recurrence":
            normalized = nn.functional.normalize(raw_embeddings, dim=-1, eps=1e-8)
            similarity = normalized @ normalized.transpose(1, 2)
            eligible = (
                earlier.unsqueeze(0)
                & first_occurrence.unsqueeze(1)
                & ~same_item
                & valid.unsqueeze(2)
            )
            top_values = (
                similarity.masked_fill(~eligible, -torch.inf)
                .topk(min(3, length), dim=-1)
                .values
            )
            finite = torch.isfinite(top_values)
            support = torch.where(finite, top_values, 0.0).sum(dim=-1)
            support = support / finite.sum(dim=-1).clamp_min(1)
            support_logits = nn.functional.softplus(self.support_raw) * support
        else:
            support_logits = raw_embeddings.new_zeros((batch, length))

        # A contribution at j remains eligible until its next identical item.
        later = positions.unsqueeze(0) > positions.unsqueeze(1)
        next_occurrence = torch.where(
            same_item & later.unsqueeze(0),
            positions.view(1, 1, length),
            length,
        ).amin(dim=-1)
        t = positions.view(1, length, 1)
        j = positions.view(1, 1, length)
        pool_mask = valid.unsqueeze(1) & (j <= t) & (t < next_occurrence.unsqueeze(1))
        logits = support_logits.unsqueeze(1).expand(-1, length, -1)
        if self.history_mode in {"recency", "recurrence"}:
            logits = logits - nn.functional.softplus(self.recency_raw) * (t - j)
        weights = torch.softmax(logits.masked_fill(~pool_mask, -1e9), dim=-1)
        weights = weights * pool_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = weights @ raw_embeddings
        residual = self.history_projection(pooled)
        limit = 0.25 * backbone.norm(dim=-1, keepdim=True)
        scale = (limit / residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(
            max=1.0
        )
        return backbone + torch.tanh(self.history_gate) * residual * scale
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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unused attention weights and enable optimized attention
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

        ### >>> Self_EvolveRec-BLOCK-START: Share complete features across training prediction and diagnosis
        features = self.last_layernorm(seq_emb)
        if self.history_mode != "none":
            features = self._history_residual(seq, features)
        return features
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
            while len(neg_pool) < 99:
                ### >>> Self_EvolveRec-BLOCK-START: Use the fixed local validation generator
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


### >>> Self_EvolveRec-BLOCK-START: Reset each dataset to training seed 42 for reproducible ablations
def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
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
        ### >>> Self_EvolveRec-BLOCK-START: Invoke selected architecture and private training sampler
        maxlen=MAXLEN,
        history_mode=HISTORY_MODE,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()
    sampler = TrainingSampler(user_train, itemnum, seed=42)
    print(
        f"  [{dataset_name}] variant={VARIANT} negatives={NUM_NEGATIVES}"
        f" skipped_users={sampler.skipped_users}"
        f" skipped_supervised_positions={sampler.skipped_supervised_positions}",
        flush=True,
    )
    import time

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always save the first evaluated checkpoint including zero-hit runs
    best_hr10 = -1.0
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        ### >>> Self_EvolveRec-BLOCK-START: Uniform candidate-set CE with checkpointed scoring and baseline loss ablation
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sampler.sample(BATCH_SIZE, MAXLEN, NUM_NEGATIVES)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)

            if OBJECTIVE == "ce":
                features = model(seq_t)
                loss = candidate_set_loss(model, features, pos_t, neg_t)
                if loss is None:
                    continue
            else:
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t[:, :, 0])
                indices = pos_t != 0
                if not indices.any():
                    continue
                loss = bce_criterion(
                    pos_logits[indices], torch.ones_like(pos_logits[indices])
                ) + bce_criterion(
                    neg_logits[indices], torch.zeros_like(neg_logits[indices])
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        ### <<< Self_EvolveRec-BLOCK-END

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

    ### >>> Self_EvolveRec-BLOCK-START: Prevent stale checkpoint reuse and restore the selected model
    if best_epoch == 0:
        torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Persist declared configuration and measured run cost without invented metrics
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - training_started
    with open(os.path.join(working_dir, f"{dataset_name}_run.json"), "w") as f:
        json.dump(
            {
                "variant": VARIANT,
                "history_mode": HISTORY_MODE,
                "objective": OBJECTIVE,
                "negative_count": NUM_NEGATIVES,
                "negative_sampling": "uniform_with_replacement_training_complement",
                "temperature": 1.0,
                "training_seed": 42,
                "preprocessing": "host_provided_splits_seed_0_unchanged",
                "epoch_limit": NUM_EPOCHS,
                "completed_epochs": NUM_EPOCHS if NUM_EPOCHS == 0 else epoch,
                "validation_every_epochs": 5,
                "selection_metric": "HR@10",
                "selection_rule": "strict_improvement_earliest_tie",
                "patience_validation_checks": PATIENCE,
                "best_epoch": best_epoch,
                "best_internal_validation_hr10": best_hr10 if best_epoch else None,
                "single_device_elapsed_seconds": elapsed_seconds,
                "skipped_users": sampler.skipped_users,
                "skipped_supervised_positions": sampler.skipped_supervised_positions,
            },
            f,
            indent=2,
        )
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                ### >>> Self_EvolveRec-BLOCK-START: Save the exact selected history architecture
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
                "history_mode": HISTORY_MODE,
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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four predictor routes
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
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on missing dataset artifacts
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model artifacts for {ds} in {_HERE}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained architecture in the predictor writer
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            history_mode=meta.get("history_mode", "none"),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes while retaining the legacy signature
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")
        ds = dataset_name
    else:
        ds = DEFAULT_DATASETS[0]
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


