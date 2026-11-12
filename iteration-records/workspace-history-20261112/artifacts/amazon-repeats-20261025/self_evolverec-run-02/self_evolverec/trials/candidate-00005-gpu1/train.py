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

### >>> Self_EvolveRec-BLOCK-START: Track the declared per-category training budget
import random
import time
from collections import defaultdict

### <<< Self_EvolveRec-BLOCK-END

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
### >>> Self_EvolveRec-BLOCK-START: Honor the matched training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Preserve the host data boundary and declare one proposal arm
DATA_DIR = os.environ["GAGC_DATA_DIR"]
POOL_WINDOW = 10
TRAIN_NEGATIVES = 64
CATEGORY_SECONDS = 1600.0
VALIDATION_EPOCHS = {25, 50, 100, 150, 200}
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Always retain the four matched categories
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


### >>> Self_EvolveRec-BLOCK-START: Vectorized uniform unique prefix-excluding candidates with independent sampling randomness
class CandidateSampler:
    def __init__(self, user_train, itemnum, seed=42):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, items in user_train.items() if len(items) >= 2]
        if not self.users:
            raise ValueError("Training requires a user with at least two interactions")
        self.user_rng = random.Random(seed)
        self.negative_rng = np.random.default_rng(seed)

    def draw(self, excluded, count):
        eligible_count = self.itemnum - len(excluded)
        count = min(count, eligible_count)
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        # Enumerate only for dense exclusions; both paths sample the same
        # uniform subset without replacement.
        if eligible_count < self.itemnum // 4 or eligible_count < 2 * count:
            eligible = np.fromiter(
                (i for i in range(1, self.itemnum + 1) if i not in excluded),
                dtype=np.int64,
            )
            return self.negative_rng.choice(eligible, size=count, replace=False)
        # Preserve proposal order: sorting unique item IDs before truncation
        # would favor smaller IDs and violate uniform sampling.
        selected = np.empty(0, dtype=np.int64)
        excluded_ids = np.fromiter(excluded, dtype=np.int64)
        while selected.size < count:
            proposals = self.negative_rng.integers(
                1,
                self.itemnum + 1,
                size=max(32, 2 * (count - selected.size)),
            )
            _, first = np.unique(proposals, return_index=True)
            proposals = proposals[np.sort(first)]
            proposals = proposals[~np.isin(proposals, excluded_ids)]
            if selected.size:
                proposals = proposals[~np.isin(proposals, selected)]
            selected = np.concatenate([selected, proposals[: count - selected.size]])
        return selected

    def batch(self, batch_size, maxlen, candidate_count=64):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, candidate_count), dtype=np.int64)
        for b in range(batch_size):
            history = self.user_train[self.user_rng.choice(self.users)]
            start = max(0, len(history) - 1 - maxlen)
            offset = maxlen - (len(history) - 1 - start)
            # Include the observed prefix preceding the truncated input.
            prefix = {int(i) for i in history[:start] if 1 <= int(i) <= self.itemnum}
            for p in range(start, len(history) - 1):
                t = offset + p - start
                current, target = int(history[p]), int(history[p + 1])
                seq[b, t], pos[b, t] = current, target
                prefix.add(current)
                excluded = prefix | {target}
                sampled = self.draw(excluded, candidate_count)
                neg[b, t, : len(sampled)] = sampled
        return seq, pos, neg


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    # Keep the diagnostic B x L negative interface separate from training's
    # B x L x 64 candidates. Its native all-history exclusion stays fixed.
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No eligible training users")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        excluded = set(history)
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx], pos[idx] = item, nxt
            if nxt != 0 and len(excluded) < itemnum:
                candidate = random.randint(1, itemnum)
                while candidate in excluded:
                    candidate = random.randint(1, itemnum)
                neg[idx] = candidate
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


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
        ### >>> Self_EvolveRec-BLOCK-START: Serialize the recent-history window explicitly
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        pool_window: int = 10,
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

        ### >>> Self_EvolveRec-BLOCK-START: Identity-initialized causal recency and similarity-support residual
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.pool_window = int(pool_window)
        if self.pool_window < 1:
            raise ValueError("pool_window must be positive")
        self.alpha = nn.Parameter(torch.zeros(()))
        self.raw_decay = nn.Parameter(torch.zeros(()))
        self.raw_strength = nn.Parameter(torch.zeros(()))
        # Preserve the baseline RNG stream for subsequent dropout/training.
        with torch.random.fork_rng(devices=[]):
            self.pool_gate = nn.Linear(2 * hidden_units + 1, 1)
        nn.init.zeros_(self.pool_gate.weight)
        nn.init.zeros_(self.pool_gate.bias)

    def _recent_pool(self, seq, values):
        F = torch.nn.functional
        k = self.pool_window
        valid = seq.ne(0)
        values = values * valid.unsqueeze(-1)
        windows = F.pad(values, (0, 0, k - 1, 0)).unfold(1, k, 1)
        windows = windows.permute(0, 1, 3, 2)
        masks = F.pad(valid, (k - 1, 0)).unfold(1, k, 1)
        predecessors = torch.ones(k, k, device=seq.device, dtype=torch.bool).tril(-1)
        age = torch.arange(k - 1, -1, -1, device=seq.device, dtype=values.dtype)
        decay = F.softplus(self.raw_decay)
        strength = F.softplus(self.raw_strength)
        pools = []
        # Recompute support inside each position's own causal window.
        # Chunking bounds the B x L x K x K intermediate.
        for start in range(0, seq.shape[1], 16):
            window = windows[:, start : start + 16]
            mask = masks[:, start : start + 16]
            cosine_vectors = F.normalize(window.detach(), dim=-1, eps=1e-8)
            similarities = (
                cosine_vectors @ cosine_vectors.transpose(-1, -2)
            ).clamp_min(0)
            pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2) & predecessors
            support = (similarities * pairs).sum(-1) / pairs.sum(-1).clamp_min(1)
            logits = -decay * age + strength * support
            # Finite sentinel plus explicit renormalization handles empty windows.
            weights = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1)
            weights = weights * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            pools.append((weights.unsqueeze(-1) * window).sum(-2))
        return torch.cat(pools, dim=1)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        ### <<< Self_EvolveRec-BLOCK-END
        B, M = seq.shape
        ### >>> Self_EvolveRec-BLOCK-START: Pool the same scaled item vectors before positional encoding and dropout
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)
        pool_values = seq_emb

        pos = torch.arange(M, device=seq.device).unsqueeze(0).expand(B, -1)
        ### <<< Self_EvolveRec-BLOCK-END
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

        ### >>> Self_EvolveRec-BLOCK-START: Preserve deployed scoring and bound candidate-loss activation memory
        features = self.last_layernorm(seq_emb)
        pool = self._recent_pool(seq, pool_values)
        prefix_length = seq.ne(0).cumsum(dim=1).to(features.dtype)
        gate_inputs = torch.cat(
            [features, pool, prefix_length.log1p().unsqueeze(-1)], dim=-1
        )
        gate = torch.sigmoid(self.pool_gate(gate_inputs))
        return (features + self.alpha * gate * pool) * seq.ne(0).unsqueeze(-1)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def candidate_loss(self, features, candidates):
        from torch.utils.checkpoint import checkpoint

        if features.shape[0] == 0:
            raise ValueError("candidate_loss requires retained prediction rows")

        def chunk_loss(chunk_features, chunk_candidates):
            logits = self.predict_score(chunk_features, chunk_candidates)
            logits = logits.masked_fill(chunk_candidates.eq(0), float("-inf"))
            labels = torch.zeros(
                logits.shape[0], dtype=torch.long, device=logits.device
            )
            return torch.nn.functional.cross_entropy(logits, labels, reduction="sum")

        # Recompute only candidate scoring during backward. The encoder and
        # its dropout execute once. Sum before dividing to weight every row
        # equally, including the final shorter chunk.
        total = features.new_zeros(())
        for start in range(0, features.shape[0], 512):
            total = total + checkpoint(
                chunk_loss,
                features[start : start + 512],
                candidates[start : start + 512],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return total / features.shape[0]
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
    ### >>> Self_EvolveRec-BLOCK-START: Repeatable local validation without mutating training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Keep validation negatives unique and guard insufficient catalogs
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
    ### >>> Self_EvolveRec-BLOCK-START: Seed each independently trained category and start its compute accounting
    started = time.monotonic()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    sampler = CandidateSampler(user_train, itemnum, seed=42)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Train the pooling branch with baseline Adam settings and no branch decay
        maxlen=MAXLEN,
        pool_window=POOL_WINDOW,
    ).to(device)

    branch_names = {
        "alpha",
        "raw_decay",
        "raw_strength",
        "pool_gate.weight",
        "pool_gate.bias",
    }
    base_parameters = [
        p for name, p in model.named_parameters() if name not in branch_names
    ]
    branch_parameters = [
        p for name, p in model.named_parameters() if name in branch_names
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": base_parameters, "weight_decay": WEIGHT_DECAY},
            {"params": branch_parameters, "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.98),
    )
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Memory-bounded candidate-set cross entropy at predeclared validation opportunities
    manifest = {
        "arm": "candidate_ce_recency_cosine",
        "training_seed": 42,
        "preprocessing": "host-provided seed-0 splits, unchanged",
        "datasets": DATASETS,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "optimizer": "Adam",
        "lr": LR,
        "betas": [0.9, 0.98],
        "base_weight_decay": WEIGHT_DECAY,
        "branch_weight_decay": 0.0,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "maxlen": MAXLEN,
        "epochs": NUM_EPOCHS,
        "pool_window": POOL_WINDOW,
        "negatives": TRAIN_NEGATIVES,
        "temperature": 1.0,
        "sampling": "uniform unique; exclude target and entire observed causal prefix",
        "category_seconds": CATEGORY_SECONDS,
        "validation_epochs": sorted(VALIDATION_EPOCHS),
        "final_checkpoint_validation": True,
        "checkpoint_rule": "highest local validation HR@10, earliest on ties",
        "host_validation": "independent pinned evaluator remains authoritative",
        "unexecuted_controls": [
            "unchanged baseline",
            "loss only",
            "recency only",
            "popularity control",
        ],
    }
    with open(os.path.join(working_dir, f"{dataset_name}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    best_hr10 = -1.0
    best_epoch = 0
    final_epoch = 0
    budget_stopped = False
    log_path = os.path.join(working_dir, f"{dataset_name}_training.jsonl")
    # Clear only this trial's log; always overwrite its checkpoint on first validation.
    with open(log_path, "w"):
        pass

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        completed_steps = 0

        for _ in range(NUM_BATCHES):
            if time.monotonic() - started >= CATEGORY_SECONDS:
                budget_stopped = True
                break
            seq, pos, neg = sampler.batch(BATCH_SIZE, MAXLEN, TRAIN_NEGATIVES)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            rows = pos_t.ne(0) & neg_t.ne(0).any(dim=-1)
            if not rows.any():
                continue
            features = model.log2feats(seq_t)[rows]
            negatives = neg_t[rows]
            candidates = torch.cat([pos_t[rows].unsqueeze(-1), negatives], dim=-1)
            # Positive remains column zero; padding negatives are excluded.
            # Chunking preserves the original equal-row candidate-set CE.
            loss = model.candidate_loss(features, candidates)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            completed_steps += 1

        final_epoch = epoch
        avg_loss = total_loss / max(completed_steps, 1)
        final_opportunity = budget_stopped or epoch == NUM_EPOCHS
        if epoch in VALIDATION_EPOCHS or final_opportunity:
            ### <<< Self_EvolveRec-BLOCK-END
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}  val_score: {hr10:.4f}"
                f"  HR@20={val_m['HR@20']:.4f}  NDCG@10={val_m['NDCG@10']:.4f}",
                flush=True,
            )
            ### >>> Self_EvolveRec-BLOCK-START: Log actual opportunities and select checkpoints without undeclared patience checks
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "completed_epoch": completed_steps == NUM_BATCHES,
                            "completed_steps": completed_steps,
                            "elapsed_seconds": time.monotonic() - started,
                            "loss": avg_loss,
                            "local_validation": val_m,
                            "alpha": float(model.alpha.detach().cpu()),
                            "budget_stopped": budget_stopped,
                        }
                    )
                    + "\n"
                )
        if budget_stopped:
            break

    if best_epoch == 0:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Check round-trip scoring on public training prefixes using the exact
    # architecture constructor that the predictor writer will reconstruct.
    histories = [h for h in user_train.values() if h][:4]
    parity_seq = torch.zeros((len(histories), MAXLEN), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        tail = history[-MAXLEN:]
        parity_seq[row, -len(tail) :] = torch.tensor(
            tail, dtype=torch.long, device=device
        )
    parity_candidates = torch.arange(1, min(itemnum, 32) + 1, device=device)
    parity_candidates = parity_candidates.unsqueeze(0).expand(len(histories), -1)
    with torch.random.fork_rng(devices=[]):
        restored = SASRec(
            itemnum,
            HIDDEN_UNITS,
            NUM_BLOCKS,
            NUM_HEADS,
            dropout_rate=0.0,
            maxlen=MAXLEN,
            pool_window=POOL_WINDOW,
        ).to(device)
    restored.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    restored.eval()
    with torch.no_grad():
        expected = model.predict(parity_seq, parity_candidates)
        actual = restored.predict(parity_seq, parity_candidates)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    del restored
    with open(os.path.join(working_dir, f"{dataset_name}_selection.json"), "w") as f:
        json.dump(
            {
                "selected_epoch": best_epoch,
                "last_attempted_epoch": final_epoch,
                "local_validation_hr10": best_hr10,
                "elapsed_seconds": time.monotonic() - started,
                "roundtrip_score_parity": True,
                "paired_bootstrap": "unavailable: pinned host hit vectors are not exposed here",
            },
            f,
            indent=2,
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist the pooling architecture for inference
                "maxlen": MAXLEN,
                "pool_window": POOL_WINDOW,
                "architecture": "sasrec_recency_cosine_v1",
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
### >>> Self_EvolveRec-BLOCK-START: Preserve four-category predictor routing
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
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained pooling architecture
            maxlen=meta["maxlen"],
            pool_window=meta["pool_window"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

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


