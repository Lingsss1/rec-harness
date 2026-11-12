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


