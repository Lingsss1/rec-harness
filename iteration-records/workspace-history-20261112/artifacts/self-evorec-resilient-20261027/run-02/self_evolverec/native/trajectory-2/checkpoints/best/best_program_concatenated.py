# === main_code.py ===
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
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched epoch cap
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Retain the stronger observed baseline while exposing corrected factorial variants
DATA_DIR = os.environ["GAGC_DATA_DIR"]
# Reported baseline HR@10 was 0.505 versus 0.470 for the previous combined
# implementation. Keep baseline as the default until a host-evaluated variant
# improves it. Each invocation trains exactly one complete four-category variant;
# cross-variant selection remains at the common host boundary.
VARIANT = os.environ.get("SASREC_VARIANT", "baseline")
PRIMARY_VARIANTS = ("baseline", "objective-only", "residual-only", "combined")
if VARIANT not in PRIMARY_VARIANTS:
    raise ValueError(f"SASREC_VARIANT must be one of {PRIMARY_VARIANTS}")
USE_CANDIDATE_CE = VARIANT in ("objective-only", "combined")
USE_RECENT_RESIDUAL = VARIANT in ("residual-only", "combined")
RECENT_WINDOW = 5
NUM_NEGATIVES = 64
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four benchmark categories
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


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    ### >>> Self_EvolveRec-BLOCK-START: Reject empty negative pools before baseline rejection sampling
    if itemnum < 1:
        raise ValueError("itemnum must be positive")
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len({i for i in items if 1 <= i <= itemnum}) < itemnum
    ]
    if not users:
        raise ValueError(
            "No training users have both next-item targets and eligible negatives"
        )
    batch_seq, batch_pos, batch_neg = [], [], []
    ### <<< Self_EvolveRec-BLOCK-END
    for _ in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
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
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


### >>> Self_EvolveRec-BLOCK-START: Share exact uniform unseen negatives across sequence positions and score by batched matrix multiplication
class CandidateBatchSampler:
    def __init__(self, user_train, maxlen, itemnum):
        self.histories = user_train
        self.maxlen = maxlen
        self.rng = np.random.default_rng(42)
        self.users = []
        self.exclusions = {}
        empty_pools = 0
        short_users = 0
        for u, history in user_train.items():
            if len(history) < 2:
                short_users += 1
                continue
            if any(i <= 0 or i > itemnum for i in history):
                raise ValueError("Training item IDs must lie in [1, itemnum]")
            # Zero-based blocked catalog indices. Only complete TRAIN histories
            # determine exclusions; no validation or withheld histories are used.
            blocked = np.array(sorted(set(history)), dtype=np.int64) - 1
            count = itemnum - len(blocked)
            if count == 0:
                empty_pools += 1
                continue
            self.exclusions[u] = (blocked - np.arange(len(blocked)), count)
            self.users.append(u)
        if not self.users:
            raise ValueError("No training positions with eligible catalog negatives")
        sizes = np.array([self.exclusions[u][1] for u in self.users])
        self.statistics = {
            "catalog_size": int(itemnum),
            "eligible_users": len(self.users),
            "eligible_pool_min": int(sizes.min()),
            "eligible_pool_mean": float(sizes.mean()),
            "eligible_pool_max": int(sizes.max()),
            "skipped_empty_pool_users": empty_pools,
            "skipped_short_users": short_users,
            "negative_sampling": "uniform_unique_catalog_minus_training_history",
            "shared_across_sequence_positions": True,
            "maximum_negatives": NUM_NEGATIVES,
            "negative_seed": 42,
        }

    def sample(self, batch_size):
        seq = np.zeros((batch_size, self.maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        # Internal candidate IDs are B x K, rather than B x L x K.
        # This sampler is separate from the diagnostic B x L loader contract.
        neg = np.zeros((batch_size, NUM_NEGATIVES), dtype=np.int64)
        for b in range(batch_size):
            u = random.choice(self.users)
            history = self.histories[u]
            inputs = history[:-1][-self.maxlen :]
            targets = history[1:][-self.maxlen :]
            start = self.maxlen - len(inputs)
            seq[b, start:] = inputs
            pos[b, start:] = targets
            adjusted_blocked, count = self.exclusions[u]
            k = min(NUM_NEGATIVES, count)
            # Uniform sampling without replacement in the complement's rank
            # space, followed by an exact monotone map back to catalog IDs.
            ranks = self.rng.choice(count, size=k, replace=False)
            neg[b, :k] = (
                1 + ranks + np.searchsorted(adjusted_blocked, ranks, side="right")
            )
        return seq, pos, neg


def candidate_cross_entropy(model, seq, pos, neg):
    valid = pos.ne(0)
    features = model.log2feats(seq)
    # Embed each user's shared candidates once. This avoids the previous
    # O(B*L*K*H) candidate embedding activation and checkpoint recomputation.
    negative_embeddings = model.item_emb(neg)
    negative_logits = torch.bmm(features, negative_embeddings.transpose(1, 2))
    negative_logits = negative_logits.masked_fill(neg.eq(0).unsqueeze(1), float("-inf"))
    positive_logits = model.predict_score(features, pos)
    logits = torch.cat((positive_logits.unsqueeze(-1), negative_logits), dim=-1)
    logits = logits[valid]
    if logits.shape[0] == 0:
        raise ValueError("Candidate loss received no valid training positions")
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    # The target plus sampled candidates form an uncorrected candidate CE;
    # this is not an estimator of full-catalog softmax.
    return nn.functional.cross_entropy(logits, labels)


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
        ### >>> Self_EvolveRec-BLOCK-START: Serialize the recent-interest architecture
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_residual: bool = False,
        recent_window: int = 5,
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

        ### >>> Self_EvolveRec-BLOCK-START: Preserve projected residual initialization and score candidate sets without expanded products
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.recent_residual = recent_residual
        self.recent_window = recent_window
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        if recent_residual:
            # Isolate residual initialization from subsequent CPU random draws.
            with torch.random.fork_rng(devices=[]):
                self.recent_gate = nn.Linear(2 * hidden_units, 1)
                self.recent_projection = nn.Linear(
                    hidden_units, hidden_units, bias=False
                )
                nn.init.zeros_(self.recent_gate.weight)
                nn.init.constant_(self.recent_gate.bias, -2.0)
                nn.init.xavier_uniform_(self.recent_projection.weight)

    def recent_features(self, seq):
        valid = seq.ne(0)
        embeddings = self.item_emb(seq) * valid.unsqueeze(-1)
        sums = embeddings.cumsum(dim=1)
        counts = valid.to(embeddings.dtype).cumsum(dim=1)
        window = self.recent_window
        if seq.shape[1] > window:
            sums = torch.cat(
                (sums[:, :window], sums[:, window:] - sums[:, :-window]), dim=1
            )
            counts = torch.cat(
                (counts[:, :window], counts[:, window:] - counts[:, :-window]), dim=1
            )
        return sums / counts.clamp_min(1).unsqueeze(-1)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if features.ndim == 2 and items.ndim == 2:
            # B x H features against B x K candidates, including full catalogs.
            return torch.bmm(embeddings, features.unsqueeze(-1)).squeeze(-1)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)
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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unused attention weights and enable efficient attention
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

        ### >>> Self_EvolveRec-BLOCK-START: Add projected recent interests after unchanged baseline normalization
        features = self.last_layernorm(seq_emb)
        if self.recent_residual:
            recent = self.recent_features(seq)
            gate = torch.sigmoid(
                self.recent_gate(torch.cat((features, recent), dim=-1))
            )
            residual = gate * self.recent_projection(recent)
            features = features + residual
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
        ### >>> Self_EvolveRec-BLOCK-START: Share the candidate scorer between training diagnostics and inference
        return self.predict_score(final_feat, candidates)
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
    ### >>> Self_EvolveRec-BLOCK-START: Stable local validation without changing training RNG
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)

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
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic unique validation negatives and finite sampling
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError("Validation requires 99 unique eligible negatives")
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve caller mode after local validation
    model.train(was_training)
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }
    ### <<< Self_EvolveRec-BLOCK-END


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    ### >>> Self_EvolveRec-BLOCK-START: Seed each category at 42 and pass the public catalog size to candidate sampling
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    sampler = (
        CandidateBatchSampler(user_train, MAXLEN, itemnum) if USE_CANDIDATE_CE else None
    )
    print(
        f"  [{dataset_name}] users={len(user_train)} items={itemnum} variant={VARIANT}",
        flush=True,
    )
    if sampler is not None:
        print(json.dumps(sampler.statistics, sort_keys=True), flush=True)
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the declared residual ablation
        maxlen=MAXLEN,
        recent_residual=USE_RECENT_RESIDUAL,
        recent_window=RECENT_WINDOW,
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the earliest checkpoint even when validation hits are zero
    best_hr10 = -1.0
    ### <<< Self_EvolveRec-BLOCK-END
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Route objective variants through filtered multi-negative sampling
            if sampler is None:
                seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            else:
                seq, pos, neg = sampler.sample(BATCH_SIZE)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            ### >>> Self_EvolveRec-BLOCK-START: Uncorrected candidate cross-entropy with unchanged BCE controls
            if USE_CANDIDATE_CE:
                loss = candidate_cross_entropy(model, seq_t, pos_t, neg_t)
            else:
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                indices = pos_t != 0
                pos_labels = torch.ones_like(pos_logits[indices])
                neg_labels = torch.zeros_like(neg_logits[indices])
                loss = bce_criterion(pos_logits[indices], pos_labels)
                loss = loss + bce_criterion(neg_logits[indices], neg_labels)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Release gradient buffers before backward without zero-fill overhead
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Predeclared ten-epoch local validation cadence
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Complete the configured cap without validation-driven early stopping
            # Strict improvement above retains the earlier checkpoint on ties.
            # Local metrics are diagnostics; host validation remains authoritative.
            ### <<< Self_EvolveRec-BLOCK-END
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist projected architecture and actual candidate sharing semantics
                "maxlen": MAXLEN,
                "recent_residual": USE_RECENT_RESIDUAL,
                "recent_window": RECENT_WINDOW,
                "variant": VARIANT,
                "architecture_version": 2,
                "pooling": "causal_occurrence_mean",
                "recent_projection": (
                    "bias_free_xavier" if USE_RECENT_RESIDUAL else None
                ),
                "gate_initial_bias": -2.0 if USE_RECENT_RESIDUAL else None,
                "objective": (
                    "uncorrected_candidate_cross_entropy"
                    if USE_CANDIDATE_CE
                    else "baseline_bce"
                ),
                "num_negatives": NUM_NEGATIVES if USE_CANDIDATE_CE else 1,
                "negative_sharing": "per_sequence" if USE_CANDIDATE_CE else "none",
                "training_seed": 42,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "batches_per_epoch": NUM_BATCHES,
                "epoch_cap": NUM_EPOCHS,
                "best_epoch": best_epoch,
                "validation_source": "local_checkpoint_selection",
                "checkpoint_rule": "strict_HR10_improvement_earliest_tie",
                "validation_interval": 10,
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
### >>> Self_EvolveRec-BLOCK-START: Export all four trained dataset routes
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
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly for incomplete model exports
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint or architecture metadata for {ds}")
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
            ### >>> Self_EvolveRec-BLOCK-START: Restore the shared SASRec class including its trained recent projection
            maxlen=meta["maxlen"],
            recent_residual=meta["recent_residual"],
            recent_window=meta["recent_window"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject unknown routes instead of scoring with an unrelated vocabulary
    ds = DEFAULT_DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError(f"No trained model for dataset_name={ds!r}")
    model = _models[ds]
    meta = _metas[ds]
    if not candidates:
        return []
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


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
import torch
import numpy as np
import torch.nn.functional as F
import json

### >>> Self_EvolveRec-BLOCK-START: Isolated diagnostic randomness and trial-local manifests
import random
import hashlib
import os
import time
from pathlib import Path
from collections import Counter

### <<< Self_EvolveRec-BLOCK-END


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.
    
    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}
    
    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}
    
    Your task: produce a concise diagnosis summary that captures ONLY the core findings and implications.
    Do NOT propose web searches or action plans. Do NOT list "what to look up".

    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": [
        "<1-2 sentences: the most important interpretation of the metrics>",
        "<1-2 sentences: the second most important interpretation (only if truly necessary)>"
    ],
    "key_implications": [
        "<1 sentence: what this implies about model behavior>",
        "<1 sentence: what this implies about training dynamics or representation>",
        "<optional 1 sentence: risk/trade-off if relevant>"
    ],
    "evidence": {{
        "headline_metrics": {{
        "<metric_name>": <value>,
        "<metric_name>": <value>
        }},
        "brief_metric_read": {{
        "<metric_name>": "<very short interpretation tied to its definition>",
        "<metric_name>": "<very short interpretation tied to its definition>"
        }}
    }}
    }}

    Rules:
    ### >>> Self_EvolveRec-BLOCK-START: Restrict interpretation to measured diagnostic evidence
    - Separate in-sample training-history sensitivity from validation generalization.
    - Report unsupported probes, incomplete coverage, small samples, and interval limitations.
    - Missing measurements are unknown, never zero or evidence of healthy behavior.
    - Cosine similarity measures directional alignment, not recommendation diversity.
    - Do not infer intent, ownership, dislikes, simulator traits, or unavailable metadata.
    - Do not claim improved HR@10 without independently measured pinned validation results.
    - Manifest creation does not establish preregistration before earlier validation queries.
    - Sensitivity is descriptive; neither higher nor lower sensitivity is inherently better.
    - Identity replacements preserve occupied positions; swaps preserve the item multiset.
    - Frequency/norm matching is partial control, not evidence of realistic transitions.
    - The fitted target is the final TRAIN interaction, never a validation label.
    - Later-target weighting changes target contributions, not conditioning recency directly.
    - Longer prefixes and absolute positions remain possible explanations of differences.
    - Interpret D as recent-minus-older normalized rank harm; D > 0 is a proposed
      deployment threshold, not a validated rule. Require complete category panels.
    - Without matched host outcomes, policy accuracy, gains, and association are unknown.
    ### <<< Self_EvolveRec-BLOCK-END
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Use the trained module and explicitly available dataset routes
    CATEGORIES = (
        "Movies_and_TV",
        "Industrial_and_Scientific",
        "Electronics",
        "CDs_and_Vinyl",
    )

    def __init__(self, view, device):
        self.model = view  # Compatibility hooks live on the view.
        self.module = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train
        # Optional explicit collection of category views. Never infer routes from IDs.
        supplied = getattr(view, "diagnostic_views", None)
        if supplied is not None:
            self.views = {c: supplied[c] for c in self.CATEGORIES if c in supplied}
        else:
            category = getattr(view, "dataset_name", None)
            if category is None:
                category = getattr(self.module, "dataset_name", None)
            self.views = {category: view} if category in self.CATEGORIES else {}
        self.root = Path(__file__).resolve().parent / "working" / "diagnostics"
        self.query_cap = 2048
        self.time_cap = 30.0
        self.margin_query_limit = 256
        self.queries = 0
        self.started = None
        for category, routed_view in self.views.items():
            declared = getattr(routed_view, "dataset_name", None)
            if declared is not None and declared != category:
                raise ValueError(
                    f"Diagnostic route {category!r} contains view {declared!r}"
                )
            routed_device = routed_view.item_emb.weight.device
            if routed_device.type != self.device.type or (
                self.device.index is not None
                and routed_device.index != self.device.index
            ):
                raise ValueError(
                    f"Diagnostic route {category!r} is on {routed_device}, expected {self.device}"
                )

    def _get_categories(self, item_id: int):
        # Item category metadata is unavailable in the shared task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Distinguish an unavailable alignment estimate from zero
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Exact mean cosine without quadratic Gram allocation
            if k < 2:
                return None
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite sampled item embeddings")
            # Sum all pairwise inner products, then remove the diagonal.
            # Accumulate in float64 to reduce cancellation near mean cosine zero.
            x64 = x.double()
            offdiag_sum = x64.sum(dim=0).square().sum() - x64.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Mode is managed on the actual trained module by the caller
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Bound sampled-margin rows while reserving the paired diagnostic panel
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic loader must yield matching B x L tensors")
            if self.started is not None:
                remaining = min(
                    self.margin_query_limit - self.queries,
                    self.query_cap - self.queries,
                )
                rows = min(int(seq_t.shape[0]), remaining)
                if rows <= 0 or not self._within_budget(rows):
                    break
                seq_t, pos_t, neg_t = seq_t[:rows], pos_t[:rows], neg_t[:rows]
            log_feats = self.model.log2feats(seq_t)
            self.queries += int(seq_t.shape[0])
            ### <<< Self_EvolveRec-BLOCK-END

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Measure only genuine sampled positive-negative pairs
            valid = (pos_t != 0) & (neg_t != 0)
            if not torch.isfinite(margin[valid]).all():
                raise ValueError("Nonfinite sampled diagnostic margins")
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Keep breakdown eligibility identical to margin eligibility
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Report empty sampled-margin probes explicitly
            margin_metrics = {
                "count": 0,
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            ### >>> Self_EvolveRec-BLOCK-START: Record sampled-margin denominator
            margin_metrics = {
                "count": int(N),
                "mean": round(float(m.mean().item()), 4),
                ### <<< Self_EvolveRec-BLOCK-END
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Mark item-category breakdown as unavailable
        breakdown = {
            "status": "unsupported: item category metadata unavailable",
            "eps": float(eps),
            ### <<< Self_EvolveRec-BLOCK-END
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Frozen paired history perturbations with bounded descriptive inference
    @staticmethod
    def _seed(*parts):
        payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _within_budget(self, rows=1):
        if self.started is None:
            raise RuntimeError("Diagnostic budget clock has not been started")
        if rows < 0:
            raise ValueError("Diagnostic query count must be nonnegative")
        self._sync()
        return (
            self.queries + rows <= self.query_cap
            and time.monotonic() - self.started < self.time_cap
        )

    @staticmethod
    def _history(view):
        histories = view.user_train
        return histories["History"] if "History" in histories else histories

    @staticmethod
    def _length(view):
        length = getattr(view.model, "maxlen", None)
        if length is None:
            length = getattr(view, "maxlen", None)
        if length is None:
            raise ValueError(
                "Diagnostic view must expose the trained maximum sequence length"
            )
        return int(length)

    def prepare_manifest_before_intervention_validation(self):
        """Freeze inputs before this probe; earlier host validation timing is unknown."""
        self.root.mkdir(parents=True, exist_ok=True)
        examples, eligibility, signatures = {}, {}, {}
        for category, view in self.views.items():
            histories = self._history(view)
            length = self._length(view)
            catalog_size = int(view.item_emb.weight.shape[0]) - 1
            entries = []
            digest = hashlib.sha256()
            ordered = sorted(histories.items(), key=lambda pair: str(pair[0]))
            frequency = np.zeros(catalog_size + 1, dtype=np.float64)
            for _, history in ordered:
                ids = np.asarray(history, dtype=np.int64)
                if ((ids < 1) | (ids > catalog_size)).any():
                    raise ValueError("Training IDs fall outside the item vocabulary")
                np.add.at(frequency, ids, 1)
            norms = view.item_emb.weight.detach().float().norm(dim=1).cpu().numpy()
            if catalog_size < 1 or not np.isfinite(norms).all():
                raise ValueError("Empty catalog or nonfinite item embedding norms")
            norm_sha256 = hashlib.sha256(norms.astype(np.float32).tobytes()).hexdigest()
            covariates = np.column_stack((np.log1p(frequency[1:]), norms[1:]))
            center = covariates.mean(axis=0)
            scale = np.maximum(covariates.std(axis=0), 1e-8)
            standardized = (covariates - center) / scale
            catalog = np.arange(1, catalog_size + 1, dtype=np.int64)
            for user, history in ordered:
                history = [int(i) for i in history]
                digest.update(json.dumps([str(user), history]).encode())
                if len(history) < 6:
                    continue
                target = history[-1]
                prefix = history[:-1][-length:]
                if len(prefix) < 5:
                    continue
                if any(i < 1 or i > catalog_size for i in prefix + [target]):
                    raise ValueError(
                        "Training IDs fall outside the trained item vocabulary"
                    )
                excluded = sorted(set(history))
                count = catalog_size - len(excluded)
                if count < 99:
                    continue
                entries.append(
                    (str(user), len(history) - 1, prefix, target, excluded, count)
                )
            selection_rng = np.random.default_rng(0)
            entries = [entries[int(i)] for i in selection_rng.permutation(len(entries))]
            eligibility[category] = {
                "eligible_users": len(entries),
                "selected_users": min(32, len(entries)),
                "total_training_users": len(histories),
                "catalog_size": catalog_size,
            }
            signatures[category] = {
                "history_sha256": digest.hexdigest(),
                "maxlen": length,
                "catalog_size": catalog_size,
                "matching_norm_sha256": norm_sha256,
                "declared_variant": getattr(view.model, "variant", None),
            }
            examples[category] = []
            for user, cutoff, prefix, target, excluded, count in entries[:32]:
                seed = self._seed(category, user, "candidates")
                rng = np.random.default_rng(seed)
                # Map unique complement ranks to IDs without allocating the catalog.
                ranks = rng.choice(count, size=99, replace=False)
                adjusted_blocked = (
                    np.asarray(excluded, dtype=np.int64)
                    - 1
                    - np.arange(len(excluded), dtype=np.int64)
                )
                negatives = (
                    ranks + np.searchsorted(adjusted_blocked, ranks, side="right") + 1
                ).tolist()
                n = len(prefix)
                positions = {"recent_identity": n - 2, "older_identity": 0}
                variants = {"position_swap": [[n - 2, 0]]}
                matching = {}
                seeds = []
                for name, position in positions.items():
                    original = prefix[position]
                    elsewhere = prefix[:position] + prefix[position + 1 :]
                    elsewhere_set = set(elsewhere)
                    repeated = original in elsewhere_set
                    present = np.isin(catalog, list(elsewhere_set))
                    eligible = (
                        (present == repeated)
                        & (catalog != original)
                        & (catalog != target)
                    )
                    pool = catalog[eligible]
                    differences = standardized[pool - 1] - standardized[original - 1]
                    distances = np.square(differences).sum(axis=1)
                    # Include deterministic item-ID tie breaking at the pool boundary.
                    order = np.lexsort((pool, distances))[:50]
                    closest = pool[order]
                    rep_seed = self._seed(0, category, user, name, "replacement")
                    seeds.append(rep_seed)
                    rep_rng = np.random.default_rng(rep_seed)
                    chosen = rep_rng.choice(
                        closest, size=min(5, len(closest)), replace=False
                    )
                    variants[name] = [[position, int(item)] for item in chosen]
                    details = []
                    for item in chosen:
                        residual = (
                            standardized[int(item) - 1] - standardized[original - 1]
                        )
                        edited = list(prefix)
                        edited[position] = int(item)
                        details.append(
                            {
                                "replacement": int(item),
                                "standardized_log_frequency_residual": float(
                                    residual[0]
                                ),
                                "standardized_embedding_norm_residual": float(
                                    residual[1]
                                ),
                                "squared_matching_distance": float(
                                    np.square(residual).sum()
                                ),
                                "duplicate_count_change": int(
                                    (len(edited) - len(set(edited)))
                                    - (len(prefix) - len(set(prefix)))
                                ),
                            }
                        )
                    matching[name] = {
                        "position": position,
                        "distance_from_end": n - 1 - position,
                        "original_repeated_elsewhere": repeated,
                        "eligible_replacements": int(len(pool)),
                        "nearest_pool_size": int(len(closest)),
                        "requested_replicates": 5,
                        "actual_replicates": int(len(chosen)),
                        "shortage": int(5 - len(chosen)),
                        "draws": details,
                    }
                examples[category].append(
                    {
                        "user": user,
                        "prefix_cutoff": cutoff,
                        "prefix": prefix,
                        "target": target,
                        "candidates": [target] + negatives,
                        "excluded_ids": excluded,
                        "eligible_negatives": count,
                        "candidate_seed": seed,
                        "replicate_seeds": seeds,
                        "prefix_length": n,
                        "length_stratum": (
                            "5-9" if n < 10 else "10-19" if n < 20 else "20+"
                        ),
                        "variants": variants,
                        "matching": matching,
                        "matching_supported": all(variants[name] for name in positions),
                        "baseline_matching_standardization": {
                            "mean": center.tolist(),
                            "std_clamped": scale.tolist(),
                        },
                        "baseline_norm_sha256": norm_sha256,
                    }
                )
        manifest = {
            "version": 3,
            "required_categories": list(self.CATEGORIES),
            "signatures": signatures,
            "examples": examples,
            "eligibility": eligibility,
            "settings": {
                "users_per_category": 32,
                "negatives": 99,
                "replicates": 5,
                "bootstrap_replicates": 10000,
                "diagnostic_seed": 0,
                "bootstrap_seed": 0,
                "replacement_pool_size": 50,
                "replacement_matching": "repetition presence, standardized log1p training frequency and baseline embedding norm",
                "baseline_only_manifest": True,
                "query_cap": self.query_cap,
                "diagnostic_wall_seconds_cap": self.time_cap,
                "atol": 1e-6,
                "rtol": 1e-5,
                "checkpoint_schedule": sorted(
                    set(range(10, int(os.environ.get("SASREC_EPOCHS", "200")) + 1, 10))
                    | {int(os.environ.get("SASREC_EPOCHS", "200"))}
                ),
                "sampled_margin_sequence_cap": 256,
                "paired_panel_encoded_sequences": sum(
                    2
                    + sum(
                        len(replicates) for replicates in example["variants"].values()
                    )
                    for panel in examples.values()
                    for example in panel
                ),
                "diagnostic_checkpoints": "caller-supplied checkpoint only",
                "padding": "zero left padding; occupied positions unchanged by all edits",
                "candidate_exclusions": "padding and all user training interactions, including target and truncated history",
                "tie_rule": "descending score, then ascending item ID",
            },
            "intervention_plan": {
                "status": "host outcomes unavailable; one selected objective is trained per invocation",
                "preregistration_timing": "before this probe; prior validation exposure unknown",
                "training_seed": 42,
                "preprocessing_seed": 0,
                "epoch_cap": int(os.environ.get("SASREC_EPOCHS", "200")),
                "selection": "unchanged pinned macro validation HR@10",
                "shared_search_budget_gpu_seconds": 43200,
                "estimated_training_gpu_seconds": None,
                "remaining_shared_gpu_seconds": None,
                "interventions": [
                    {"name": "baseline", "target_decay": 0.0},
                    {
                        "name": "later-target",
                        "target_decay": float(np.log(2.0) / 10.0),
                        "half_life_positions": 10,
                        "loss": "sum(valid*w*(positive_BCE+negative_BCE))/sum(valid*w)",
                    },
                ],
                "prediction": "larger baseline D predicts larger variant-minus-baseline category HR@10 gain",
                "diagnostic": "D = mean_user(mean_recent_rank_change/99 - mean_older_rank_change/99)",
                "policy": "variant iff baseline D > 0; ties select baseline",
                "comparators": ["always-baseline", "always-variant"],
                "paired_validation_analysis": "only if host exposes paired hits: discordant counts and 10000 seed-0 within-category paired bootstrap resamples",
                "category_analysis": "Spearman with average tie ranks; undefined for constants; enumerate all 24 diagnostic assignments for one-sided association and policy-gain correspondence",
                "limitations": "Four heterogeneous categories; exchangeability questionable; minimum permutation fraction 1/24; one variant cannot establish general diagnostic usefulness",
                "missingness": "No host outcomes or remaining-budget ledger were supplied; policy values, validation intervals, association and permutation fractions remain unavailable",
                "stop_rule": "host shared 43200 GPU-second cap, including all eight fits, diagnostics, debugging and failures; no automatic additional fits",
            },
        }
        # Routes and complete histories identify this frozen selection. Repeated calls
        # must reuse the saved example IDs, candidate order, and perturbation indices.
        key = hashlib.sha256(
            json.dumps(
                {"design_version": 3, "signatures": signatures}, sort_keys=True
            ).encode()
        ).hexdigest()[:20]
        path = self.root / ("manifest_" + key + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved != manifest:
                raise ValueError(
                    "Frozen diagnostic manifest differs from current settings"
                )
            manifest = saved
        else:
            with path.open("x") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        return manifest, path

    @torch.no_grad()
    def _score_many(self, view, prefixes, candidates):
        rows = len(prefixes)
        if rows == 0:
            raise ValueError("Diagnostic scoring requires at least one prefix")
        if not self._within_budget(rows):
            raise TimeoutError("Diagnostic query or elapsed-time cap reached")
        length = self._length(view)
        sequence = torch.zeros((rows, length), dtype=torch.long, device=self.device)
        for row, prefix in enumerate(prefixes):
            if not prefix or len(prefix) > length:
                raise ValueError("Diagnostic prefix length must lie in [1, maxlen]")
            sequence[row, -len(prefix) :] = torch.as_tensor(
                prefix, dtype=torch.long, device=self.device
            )
        features = view.log2feats(sequence)[:, -1, :]
        items = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        if items.ndim != 1:
            raise ValueError("Batched paired diagnostics require shared [C] candidates")
        items = items.unsqueeze(0).expand(rows, -1)
        scores = view.predict_score(features, items)
        self.queries += rows
        if scores.shape != items.shape:
            raise ValueError("Diagnostic scorer must preserve [B,C]")
        if not torch.isfinite(scores).all() or not torch.isfinite(features).all():
            raise ValueError("Nonfinite diagnostic features or candidate scores")
        scores = scores.double().cpu().numpy()
        features = features.double().cpu().numpy()
        return [(scores[row], features[row]) for row in range(rows)]

    def _score(self, view, prefix, candidates):
        return self._score_many(view, [prefix], candidates)[0]

    @staticmethod
    def _ranking(scores, candidates, target):
        ids = np.asarray(candidates)
        order = np.lexsort((ids, -scores))
        target_index = int(np.flatnonzero(ids == target)[0])
        rank = int(np.flatnonzero(order == target_index)[0]) + 1
        return rank, set(ids[order[:10]].tolist())

    def _paired(self, original, perturbed, example):
        rank0, top0 = self._ranking(
            original[0], example["candidates"], example["target"]
        )
        rank1, top1 = self._ranking(
            perturbed[0], example["candidates"], example["target"]
        )
        a, b = original[1], perturbed[1]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return {
            "normalized_rank_effect": float(rank1 - rank0) / 99.0,
            "rank_change": float(rank1 - rank0),
            "hit10_change": float((rank1 <= 10) - (rank0 <= 10)),
            "top10_overlap": len(top0 & top1) / 10.0,
            "feature_cosine": (
                float(np.clip(np.dot(a, b) / denominator, -1, 1))
                if denominator > 0
                else None
            ),
        }

    def _summarize(self, records):
        names = ("recent_identity", "older_identity", "position_swap")
        metrics = (
            "normalized_rank_effect",
            "rank_change",
            "hit10_change",
            "top10_overlap",
            "feature_cosine",
        )
        summaries, draws = {}, {}
        for category in self.CATEGORIES:
            rows = records.get(category, [])
            summaries[category] = {"completed_users": len(rows), "variants": {}}
            if not rows:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([0, self.CATEGORIES.index(category)])
            )
            # Independent category streams from seed 0; identical indices within
            # each category preserve user pairing across metrics and edits.
            indices = rng.integers(len(rows), size=(10000, len(rows)))
            differences = np.asarray([row["D_user"] for row in rows], dtype=float)
            samples = differences[indices].mean(axis=1)
            summaries[category]["D"] = {
                "mean": float(differences.mean()),
                "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
            }
            strata = {}
            for stratum in ("5-9", "10-19", "20+"):
                values = [
                    row["D_user"] for row in rows if row["length_stratum"] == stratum
                ]
                strata[stratum] = {
                    "users": len(values),
                    "D_mean": float(np.mean(values)) if values else None,
                }
            summaries[category]["length_strata"] = strata
            for name in names:
                summaries[category]["variants"][name] = {}
                for metric in metrics:
                    means, deviations = [], []
                    for row in rows:
                        values = [
                            r[metric]
                            for r in row["variants"][name]
                            if r[metric] is not None
                        ]
                        means.append(float(np.mean(values)) if values else np.nan)
                        deviations.append(float(np.std(values)) if values else np.nan)
                    values = np.asarray(means)
                    available = np.isfinite(values)
                    entry = {"users_with_defined_metric": int(available.sum())}
                    if available.all():
                        samples = values[indices].mean(axis=1)
                        draws[(category, name, metric)] = samples
                        entry.update(
                            {
                                "mean": float(values.mean()),
                                "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
                                "mean_within_user_replicate_sd": float(
                                    np.mean(deviations)
                                ),
                            }
                        )
                    else:
                        entry.update(
                            {
                                "mean": None,
                                "ci95": None,
                                "mean_within_user_replicate_sd": None,
                            }
                        )
                    summaries[category]["variants"][name][metric] = entry
        macro = {}
        for name in names:
            macro[name] = {}
            for metric in metrics:
                keys = [(category, name, metric) for category in self.CATEGORIES]
                if all(key in draws for key in keys):
                    samples = np.mean([draws[key] for key in keys], axis=0)
                    means = [
                        summaries[c]["variants"][name][metric]["mean"]
                        for c in self.CATEGORIES
                    ]
                    macro[name][metric] = {
                        "mean": float(np.mean(means)),
                        "ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
                    }
                else:
                    macro[name][metric] = None
        return summaries, macro

    def run_full_diagnosis(self, train_loader):
        # Snapshot all actual modules, including optional explicit category views.
        modules = list(
            {
                id(v.model): v.model for v in [self.model] + list(self.views.values())
            }.values()
        )
        modes = [
            (submodule, submodule.training)
            for module in modules
            for submodule in module.modules()
        ]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self._sync()
        self.started, self.queries = time.monotonic(), 0
        try:
            for module in modules:
                module.eval()
            manifest, manifest_path = (
                self.prepare_manifest_before_intervention_validation()
            )
            panel_queries = manifest["settings"]["paired_panel_encoded_sequences"]
            self.margin_query_limit = min(256, max(0, self.query_cap - panel_queries))
            collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
            margin_metrics, breakdown = (
                self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader, num_batches=10, eps=0.5, topk_categories=20
                )
            )
            records, controls, stopped = {}, [], False
            for category in self.CATEGORIES:
                if category not in self.views:
                    continue
                view = self.views[category]
                records[category] = []
                for example in manifest["examples"][category]:
                    prefix, candidates = example["prefix"], example["candidates"]
                    try:
                        required_queries = 2 + sum(
                            len(replicates)
                            for replicates in example["variants"].values()
                        )
                        if not self._within_budget(required_queries):
                            raise TimeoutError(
                                "Insufficient diagnostic budget for a complete paired user"
                            )
                        original = self._score(view, prefix, candidates)
                        unchanged = self._score(view, list(prefix), candidates)
                        difference = float(np.max(np.abs(original[0] - unchanged[0])))
                        passed = bool(
                            np.allclose(original[0], unchanged[0], atol=1e-6, rtol=1e-5)
                        )
                        controls.append(
                            {
                                "category": category,
                                "user": example["user"],
                                "max_abs_score_difference": difference,
                                "passed": passed,
                            }
                        )
                        if not passed:
                            raise ValueError("Unchanged-history score control failed")
                        if not example["matching_supported"]:
                            continue
                        row = {
                            "user": example["user"],
                            "prefix_length": example["prefix_length"],
                            "length_stratum": example["length_stratum"],
                            "matching": example["matching"],
                            "variants": {},
                        }
                        edited_prefixes, edit_names = [], []
                        for name, replicates in example["variants"].items():
                            row["variants"][name] = []
                            for edit in replicates:
                                perturbed_prefix = list(prefix)
                                if name == "position_swap":
                                    a, b = edit
                                    perturbed_prefix[a], perturbed_prefix[b] = (
                                        perturbed_prefix[b],
                                        perturbed_prefix[a],
                                    )
                                else:
                                    position, replacement = edit
                                    perturbed_prefix[position] = replacement
                                if (
                                    len(perturbed_prefix) != len(prefix)
                                    or perturbed_prefix[-1] != prefix[-1]
                                    or any(item == 0 for item in perturbed_prefix)
                                ):
                                    raise ValueError(
                                        "Position/final-item control failed"
                                    )
                                edited_prefixes.append(perturbed_prefix)
                                edit_names.append(name)
                        # Batch only edited histories; repeated unchanged scoring
                        # remains two independent calls to detect nondeterminism.
                        edited_results = self._score_many(
                            view, edited_prefixes, candidates
                        )
                        for name, perturbed in zip(edit_names, edited_results):
                            row["variants"][name].append(
                                self._paired(original, perturbed, example)
                            )
                        row["D_user"] = float(
                            np.mean(
                                [
                                    r["normalized_rank_effect"]
                                    for r in row["variants"]["recent_identity"]
                                ]
                            )
                            - np.mean(
                                [
                                    r["normalized_rank_effect"]
                                    for r in row["variants"]["older_identity"]
                                ]
                            )
                        )
                        # Average replacement draws within user before aggregation.
                        records[category].append(row)
                    except TimeoutError:
                        stopped = True
                        break
                if stopped:
                    break
            per_category, macro = self._summarize(records)
            missing = [c for c in self.CATEGORIES if not records.get(c)]
            incomplete = []
            choices = {}
            for category in self.CATEGORIES:
                panel = manifest["examples"].get(category, [])
                complete = (
                    bool(panel)
                    and all(e["matching_supported"] for e in panel)
                    and len(records.get(category, [])) == len(panel)
                )
                per_category[category]["frozen_panel_complete"] = complete
                if complete:
                    value = per_category[category]["D"]["mean"]
                    choices[category] = "later-target" if value > 0.0 else "baseline"
                else:
                    incomplete.append(category)
                    per_category[category]["D"] = None
            baseline_verified = not incomplete and all(
                getattr(self.views[c].model, "variant", None) == "baseline"
                for c in self.CATEGORIES
            )
            policy = {
                "status": (
                    "baseline diagnostic choices; checkpoint equivalence and prospective timing require external verification"
                    if baseline_verified
                    else "unavailable: incomplete panel or baseline objective not explicitly identified"
                ),
                "baseline_objective_identified": baseline_verified,
                "choices": choices if baseline_verified else None,
                "incomplete_categories": incomplete,
                "threshold": 0.0,
                "ties": "baseline",
                "host_policy_macro_HR10": None,
                "host_always_baseline_macro_HR10": None,
                "host_always_variant_macro_HR10": None,
                "spearman": None,
                "permutation_fractions": None,
                "paired_validation_intervals": None,
                "validation_status": "Matched pinned host outcomes have not been supplied; local checkpoint metrics are not substituted",
                "timing": "Frozen before this probe's results; prior validation exposure unknown. Apply only measurements made on the matched baseline, not the variant",
            }
            if stopped or incomplete:
                macro = None
            self._sync()
            elapsed = time.monotonic() - self.started
            result = {
                "metrics": {
                    "embedding_collapse_score": collapse_score,
                    "pos_neg_margin_mean": margin_metrics["mean"],
                    "pos_neg_margin_p10": margin_metrics["p10"],
                    "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                    "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                    "pos_neg_margin_count": margin_metrics["count"],
                    "pos_neg_margin_category_breakdown": breakdown,
                    "history_perturbations": {
                        "per_category": per_category,
                        "four_category_macro": macro,
                        "missing_categories": missing,
                        "routing_status": (
                            "explicit dataset routes available"
                            if self.views
                            else "unsupported: provide dataset_name or an explicit diagnostic_views mapping; routes cannot be inferred from local item IDs"
                        ),
                        "budget_truncated": stopped,
                        "diagnostic_deployment_policy": policy,
                        "eligibility": manifest["eligibility"],
                        "paired_user_records": records,
                        "unchanged_history_controls": controls,
                        "position_preserving_control": {
                            "status": "identity replacement and recent/older swap implemented",
                            "preserved": "shape, padding, occupied positions, final conditioning item, candidates; no token deletion or timestamp edit",
                            "limitations": "Relative recency and absolute position are not independently identified; matching does not guarantee realistic transitions",
                        },
                    },
                    "intervention_experiment": manifest["intervention_plan"],
                    "diagnostic_accounting": {
                        "encoded_sequences": self.queries,
                        "elapsed_synchronized_wall_seconds": elapsed,
                        "manifest": str(
                            manifest_path.relative_to(Path(__file__).resolve().parent)
                        ),
                    },
                },
                "metric_definitions": {
                    "embedding_collapse_score": "(mean off-diagonal cosine + 1)/2 on up to 2048 nonpadding item embeddings, isolated seed 42. Directional alignment only; 0.5 means mean cosine zero. Does not measure recommendation diversity or embedding rank.",
                    "pos_neg_margin_mean": "Mean positive-minus-sampled-negative logit over nonpadding pairs from at most ten loader batches and 256 sequences, further limited to reserve the frozen paired panel. In-sample, dependent on loader sampling, and scoped to the primary view rather than a four-category macro.",
                    "pos_neg_margin_p10": "Nearest-rank lower tenth percentile of sampled margins.",
                    "pos_neg_neg_beats_pos_rate": "Fraction of sampled margins strictly below zero.",
                    "pos_neg_small_margin_rate": "Fraction of sampled margins below 0.5; raw score scale affects this measurement.",
                    "pos_neg_margin_count": "Number of valid sampled positive-negative positions; missing estimates are null.",
                    "pos_neg_margin_category_breakdown": "Unsupported: item categories are unavailable. Empty category lists are not evidence of no category effects.",
                    "rank_change": "Perturbed target rank minus original rank; positive is worse. Diagnostic ranks use score descending, item ID ascending ties.",
                    "hit10_change": "Perturbed indicator(target rank <= 10) minus original indicator, within the same 100 diagnostic candidates.",
                    "top10_overlap": "Size of original/perturbed top-ten set intersection divided by ten.",
                    "feature_cosine": "Cosine between final-position features; null when either feature has zero norm.",
                    "aggregation": "Replicates averaged within each user, users averaged within each category, then all four required categories equally. Missing categories produce null macro entries; truncation suppresses the macro entirely.",
                    "ci95": "10000 paired user-bootstrap resamples per category from independent category streams rooted at seed 0; percentile interval. At most 32 training users/category. Conditional on fitted model, selected users, candidates and replacements; excludes training-seed variability and adaptive-selection correction. Singleton intervals are degenerate.",
                    "normalized_rank_effect": "(edited target rank - unchanged target rank)/99 within the same target plus 99 unique diagnostic negatives; positive means harm.",
                    "D": "Mean across users of recent-minus-older mean normalized replacement rank effect. Draws are averaged within user first. Unavailable for incomplete frozen panels; not validation HR@10 gain.",
                    "perturbations": "Recent replaces the penultimate occupied item; older replaces the earliest. Final item and occupied positions stay fixed. Five distinct replacements are sampled from the closest 50 eligible items, or all if fewer. Eligibility matches presence elsewhere in prefix, excluding original and target. Squared standardized log1p training frequency and baseline embedding norm determine proximity. Shortages and residuals are recorded. Swap exchanges these two positions while preserving the item multiset.",
                    "position_control": "Identity replacement retains occupied positions and padding; swap preserves the multiset. Neither independently identifies absolute-position effects, and both can induce unrealistic transitions. Repetition presence does not preserve exact multiplicity.",
                    "data_scope": "Final public training interaction is the target and preceding retained history is the prefix. These targets may already have been trained on. No held-out labels, ratings, review text, item categories, prices, stores, or simulator traits are used.",
                    "validation": "Diagnostic candidates and tie rules are independent of the pinned host evaluator. No host users, exclusions, ranking semantics, or splits are changed. Sensitivity and bootstrap intervals do not establish validation improvements.",
                    "experiment_status": "Training exposes baseline and fixed later-target BCE objectives through SASREC_VARIANT; one four-category fit per invocation. Diagnostic policy choices require all four complete panels and explicit model.variant='baseline'. This identifies the declared objective, not checkpoint equivalence or prospective timing. Matching manifests are keyed by history, norms and objective; a variant-generated manifest is a separate descriptive audit and cannot replace a frozen baseline panel. No experiment launcher, shared budget ledger or host-outcome ingestion is implemented here; validation policy values and correspondence remain unavailable until externally measured. Earlier inspected evidence remains retrospective.",
                    "accounting": "Elapsed synchronized wall time includes manifest generation, sampled probes, scoring, and summaries; it is a conservative local diagnostic cap checked before encodes, not GPU event time or a replacement for host shared-budget accounting. An in-flight encode or CPU summary may finish beyond the cap.",
                },
            }
            # A separate artifact preserves replicate-level measurements for audits.
            output = self.root / (
                "result_" + manifest_path.stem + "_" + str(time.time_ns()) + ".json"
            )
            with output.open("x") as handle:
                json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            return result
        finally:
            for submodule, training in modes:
                submodule.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.started = None

    ### <<< Self_EvolveRec-BLOCK-END
