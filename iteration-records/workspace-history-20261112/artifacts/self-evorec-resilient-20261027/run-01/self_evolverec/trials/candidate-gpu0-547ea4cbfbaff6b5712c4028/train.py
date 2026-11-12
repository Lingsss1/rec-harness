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

### >>> Self_EvolveRec-BLOCK-START: Reproducible residual and ranking experiments
import random
import copy
import hashlib
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
### >>> Self_EvolveRec-BLOCK-START: Declare matched ablations and bounded training resources
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

DATA_DIR = os.environ["GAGC_DATA_DIR"]

# Run one predeclared variant per host trial. No comparison is claimed until
# the host has evaluated the corresponding independent trials.
VARIANT = os.environ.get("SASREC_VARIANT", "C")
VARIANTS = {
    "baseline": (False, False),
    "A": (True, False),
    "B": (False, True),
    "C": (True, True),
}
if VARIANT not in VARIANTS:
    raise ValueError(f"Unknown SASREC_VARIANT: {VARIANT}")
USE_RECENT, USE_CANDIDATE_CE = VARIANTS[VARIANT]
RECENT_WINDOW = 5
RECENT_DECAY = 0.7
NUM_NEGATIVES = 32
POSITION_CHUNK = 1024
# Four datasets share a conservative wall-time allocation. The host remains
# authoritative for aggregate GPU accounting across all trials and failures.
VARIANT_SECONDS = min(10000.0, float(os.environ.get("SASREC_VARIANT_SECONDS", "10000")))
if VARIANT_SECONDS <= 200:
    raise ValueError("SASREC_VARIANT_SECONDS must exceed the 200-second reserve")
DATASET_SECONDS = (VARIANT_SECONDS - 200.0) / 4.0
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four training routes
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

### >>> Self_EvolveRec-BLOCK-START: Exact uniform complement sampling with unchanged candidate-loss reduction


def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users=False,
    rng=None,
    users=None,
):
    rng = random if rng is None else rng
    if users is None:
        users = [
            u
            for u, items in user_train.items()
            if len(items) >= 2 and len(set(items) - {0}) < itemnum
        ]
    if not users:
        raise ValueError(
            "No training users with next-item examples and eligible negatives"
        )
    batch_seq, batch_pos, batch_neg, batch_users = [], [], [], []
    for _ in range(batch_size):
        u = rng.choice(users)
        history = user_train[u]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history)
        fallback = None
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = 0
                for _attempt in range(32):
                    proposal = rng.randint(1, itemnum)
                    if proposal not in ts and proposal != nxt:
                        neg_item = proposal
                        break
                if neg_item == 0:
                    if fallback is None:
                        fallback = [i for i in range(1, itemnum + 1) if i not in ts]
                    if not fallback:
                        raise ValueError(
                            "No eligible negative for sampled training user"
                        )
                    neg_item = rng.choice(fallback)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
        batch_users.append(u)
    result = (np.array(batch_seq), np.array(batch_pos), np.array(batch_neg))
    return (*result, batch_users) if return_users else result


def uniform_negatives(excluded, itemnum, shape, rng):
    excluded = np.asarray(excluded, dtype=np.int64)
    excluded = np.unique(excluded[(excluded >= 1) & (excluded <= itemnum)])
    available = itemnum - len(excluded)
    if available <= 0:
        return None
    # Sample one-based ranks in the eligible set, then map ranks to item IDs.
    # For excluded IDs x[j], x[j] - j is their insertion threshold among
    # eligible ranks. side='right' also handles consecutive excluded IDs.
    # This is an exact bijection: independent uniform ranks yield independent
    # uniform negatives with replacement, without rejection or a catalog scan.
    ranks = rng.integers(1, available + 1, size=shape, dtype=np.int64)
    thresholds = excluded - np.arange(len(excluded), dtype=np.int64)
    return ranks + np.searchsorted(thresholds, ranks, side="right")


def candidate_ce_backward(model, seq_t, pos_t, batch_users, exclusions, rng):
    # Each row uses its complete training history, never validation interactions.
    eligible_np = (seq_t.ne(0) & pos_t.ne(0)).detach().cpu().numpy().copy()
    pos_np = pos_t.detach().cpu().numpy()
    negative_rows = []
    for row, user in enumerate(batch_users):
        count = int(eligible_np[row].sum())
        if count == 0:
            continue
        excluded = np.union1d(exclusions[user], pos_np[row, eligible_np[row]])
        negatives = uniform_negatives(
            excluded, model.itemnum, (count, NUM_NEGATIVES), rng
        )
        if negatives is None:
            eligible_np[row] = False
        else:
            negative_rows.append(negatives)
    if not negative_rows:
        return None
    negatives = np.concatenate(negative_rows, axis=0)
    eligible = torch.as_tensor(eligible_np, dtype=torch.bool, device=seq_t.device)
    features = model.log2feats(seq_t)[eligible]
    positives = pos_t[eligible]
    count = features.shape[0]
    if count != len(negatives):
        raise RuntimeError("Candidate rows are not aligned with sequence positions")

    # Accumulate exact feature gradients through a detached leaf, then traverse
    # the shared encoder once. Output-embedding gradients accumulate normally.
    # Chunk losses use the entire batch's eligible-position count.
    relay = features.detach().requires_grad_(True)
    loss_total = torch.zeros((), device=seq_t.device)
    for start in range(0, count, POSITION_CHUNK):
        end = min(count, start + POSITION_CHUNK)
        neg_t = torch.as_tensor(negatives[start:end], device=seq_t.device)
        candidates = torch.cat((positives[start:end, None], neg_t), dim=1)
        logits = torch.einsum(
            "nh,nkh->nk", relay[start:end], model.item_emb(candidates)
        )
        # Repeated negatives remain repeated denominator terms; the positive
        # appears once. Temperature stays at one and no logQ term is applied.
        loss = (
            nn.functional.cross_entropy(
                logits,
                torch.zeros(end - start, dtype=torch.long, device=seq_t.device),
                reduction="sum",
            )
            / count
        )
        loss.backward()
        loss_total += loss.detach()
    if relay.grad is None:
        raise RuntimeError("Candidate loss did not produce feature gradients")
    features.backward(relay.grad)
    return float(loss_total.item())


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
    ### >>> Self_EvolveRec-BLOCK-START: Explicit checkpoint-compatible residual configuration
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_residual: bool = False,
        recent_window: int = 5,
        recent_decay: float = 0.7,
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

        ### >>> Self_EvolveRec-BLOCK-START: Zero-gated residual with isolated initialization randomness
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.recent_residual = recent_residual
        self.recent_window = int(recent_window)
        self.recent_decay = float(recent_decay)
        if self.recent_window < 1 or not 0.0 < self.recent_decay <= 1.0:
            raise ValueError("Invalid recent-history window or decay")
        if recent_residual:
            # Modules are initialized on CPU before the caller moves the model.
            # Restore the CPU RNG so adding this branch does not shift the
            # baseline encoder's subsequent dropout stream across ablations.
            with torch.random.fork_rng(devices=[]):
                self.recent_norm = nn.LayerNorm(hidden_units)
                self.recent_projection = nn.Linear(hidden_units, hidden_units)
                nn.init.zeros_(self.recent_projection.bias)
                self.alpha = nn.Parameter(torch.zeros(()))
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
            attn_out, _ = self.attention_layers[i](
                queries, seq_emb, seq_emb, attn_mask=attention_mask
            )
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        ### >>> Self_EvolveRec-BLOCK-START: Causal branch-normalized exponentially weighted history
        h = self.last_layernorm(seq_emb)
        if not self.recent_residual:
            return h
        valid = seq.ne(0)
        embeddings = self.item_emb(seq)
        numerator = torch.zeros_like(embeddings)
        denominator = embeddings.new_zeros(seq.shape)
        for lag in range(min(self.recent_window, M)):
            weight = self.recent_decay**lag
            source_valid = valid[:, : M - lag].to(embeddings.dtype)
            numerator[:, lag:] += (
                weight * source_valid.unsqueeze(-1) * embeddings[:, : M - lag]
            )
            denominator[:, lag:] += weight * source_valid
        mean = numerator / denominator.clamp_min(1e-8).unsqueeze(-1)
        mean = mean.masked_fill(denominator.eq(0).unsqueeze(-1), 0.0)
        branch = self.recent_norm(mean).masked_fill(~valid.unsqueeze(-1), 0.0)
        branch = self.recent_projection(branch).masked_fill(~valid.unsqueeze(-1), 0.0)
        return (h + self.alpha * branch).masked_fill(~valid.unsqueeze(-1), 0.0)
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
    ### >>> Self_EvolveRec-BLOCK-START: Isolate local validation randomness from training
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    # This local monitor does not replace the independently pinned host evaluator.
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound validation rejection without changing candidate count
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError("Validation user has fewer than 99 eligible negatives")
            for _attempt in range(10000):
                if len(neg_pool) == 99:
                    break
                ni = rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            if len(neg_pool) < 99:
                remaining = [
                    i
                    for i in range(1, itemnum + 1)
                    if i not in interacted and i not in neg_pool
                ]
                neg_pool.update(rng.sample(remaining, 99 - len(neg_pool)))
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

### >>> Self_EvolveRec-BLOCK-START: Matched residual and candidate-loss training with auditable checkpoints


def _check_inference_contract(model, device):
    was_training = model.training
    model.eval()
    try:
        length = min(model.maxlen, 8)
        seq = torch.zeros((2, length), dtype=torch.long, device=device)
        if length > 1:
            seq[0, 1:] = (
                torch.arange(1, length, device=device).remainder(model.itemnum) + 1
            )
        candidates = torch.tensor(
            [[1, model.itemnum], [1, model.itemnum]], device=device
        )
        with torch.no_grad():
            features = model.log2feats(seq)
            expected = model.predict(seq, candidates)
            if model.recent_residual:
                if model.alpha.item() != 0.0:
                    raise AssertionError(
                        "Residual gate must initialize at exactly zero"
                    )
                if not torch.equal(
                    features[seq == 0], torch.zeros_like(features[seq == 0])
                ):
                    raise AssertionError("Padding features are nonzero")
                model.recent_residual = False
                baseline = model.predict(seq, candidates)
                model.recent_residual = True
                torch.testing.assert_close(expected, baseline, rtol=0, atol=0)
            # Activate the branch temporarily to exercise its causal indexing.
            if model.recent_residual:
                model.alpha.fill_(0.3)
            prefix_length = max(1, length // 2)
            changed = seq.clone()
            changed[:, prefix_length:] = model.itemnum
            left = model.log2feats(seq)[:, :prefix_length]
            right = model.log2feats(changed)[:, :prefix_length]
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
            if model.recent_residual:
                model.alpha.zero_()
            clone = copy.deepcopy(model)
            clone.load_state_dict(model.state_dict(), strict=True)
            clone.eval()
            torch.testing.assert_close(
                expected, clone.predict(seq, candidates), rtol=0, atol=0
            )
            del clone
    finally:
        if model.recent_residual:
            with torch.no_grad():
                model.alpha.zero_()
        model.train(was_training)


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    started = time.monotonic()
    deadline = started + DATASET_SECONDS
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum < 1:
        raise ValueError(f"Empty item vocabulary for {dataset_name}")
    print(
        f"  [{dataset_name}] variant={VARIANT} users={len(user_train)} items={itemnum}",
        flush=True,
    )

    # Independent streams keep evaluation and candidate draws out of data order.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    train_rng = random.Random(42)
    negative_rng = np.random.default_rng(42)
    exclusions = {
        u: np.asarray(sorted(set(items) - {0}), dtype=np.int64)
        for u, items in user_train.items()
    }
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(exclusions[u]) < itemnum
    ]
    if not users:
        raise ValueError(f"No eligible training users for {dataset_name}")

    meta = {
        "itemnum": itemnum,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "recent_residual": USE_RECENT,
        "recent_window": RECENT_WINDOW,
        "recent_decay": RECENT_DECAY,
    }
    model = SASRec(**meta).to(device)
    _check_inference_contract(model, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
    final_path = os.path.join(working_dir, f"{dataset_name}_final.pt")
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    def rng_states():
        return {
            "training_sampler": train_rng.getstate(),
            "candidate_sampler": negative_rng.bit_generator.state,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }

    fingerprints = {}
    for split in ("train", "valid"):
        path = os.path.join(data_dir, f"{dataset_name}_{split}.txt")
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        fingerprints[split] = digest.hexdigest()
    settings = {
        "variant": VARIANT,
        "architecture": meta,
        "training_seed": 42,
        "input_preprocessing_seed": 0,
        "optimizer": "Adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.98],
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "requested_epochs": NUM_EPOCHS,
        "gradient_clip": 5.0,
        "num_negatives": NUM_NEGATIVES,
        "position_chunk": POSITION_CHUNK,
        "candidate_sampling": "uniform with replacement; training-history exclusion",
        "loss": (
            "sampled-candidate cross-entropy" if USE_CANDIDATE_CE else "original BCE"
        ),
        "temperature": 1.0,
        "data_sha256": fingerprints,
        "dataset_wall_seconds": DATASET_SECONDS,
        "validation_schedule": "every 5 completed epochs; initial fallback if capped earlier",
        "deployment_checkpoint": "exploratory local-best validation HR@10",
        "controlled_checkpoint": "final epoch, or explicitly resource-limited final state",
        "host_metrics": "independently measured by host; local monitor is not authoritative",
    }
    torch.save(
        {"model": model.state_dict(), "rng": rng_states(), "settings": settings},
        os.path.join(working_dir, f"{dataset_name}_initial.pt"),
    )

    best_hr10 = -math.inf
    best_epoch = 0
    completed_epochs = 0
    total_steps = 0
    history = []
    resource_limited = False

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for _ in range(NUM_BATCHES):
            if time.monotonic() >= deadline:
                resource_limited = True
                break
            # Generate the original single negative in every variant so that
            # switching the objective does not change subsequent sampled users.
            seq, pos, neg, batch_users = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                return_users=True,
                rng=train_rng,
                users=users,
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if USE_CANDIDATE_CE:
                loss_value = candidate_ce_backward(
                    model, seq_t, pos_t, batch_users, exclusions, negative_rng
                )
                if loss_value is None:
                    continue
            else:
                neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                indices = pos_t != 0
                if not indices.any():
                    continue
                loss = bce_criterion(
                    pos_logits[indices], torch.ones_like(pos_logits[indices])
                )
                loss = loss + bce_criterion(
                    neg_logits[indices], torch.zeros_like(neg_logits[indices])
                )
                loss.backward()
                loss_value = float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss_value
            steps += 1
            total_steps += 1

        if resource_limited:
            break
        completed_epochs = epoch
        if epoch % 5 == 0:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": total_steps,
                    "loss": total_loss / max(steps, 1),
                    **val_m,
                }
            )
            print(
                f"  [{dataset_name}] variant={VARIANT} epoch={epoch}/{NUM_EPOCHS}"
                f" loss={total_loss / max(steps, 1):.4f} local_val_HR@10={hr10:.4f}",
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
        # No metric-dependent stopping: the final checkpoint is predeclared.

    torch.save(model.state_dict(), final_path)
    if best_epoch == 0:
        # Record this extra opportunity explicitly for exceptionally short caps.
        val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
        best_hr10 = val_m["HR@10"]
        history.append(
            {
                "epoch": completed_epochs,
                "optimizer_steps": total_steps,
                "fallback_evaluation": True,
                **val_m,
            }
        )
        torch.save(model.state_dict(), ckpt_path)

    settings.update(
        {
            "completed_epochs": completed_epochs,
            "optimizer_steps": total_steps,
            "resource_limited": resource_limited,
            "best_epoch": best_epoch,
            "local_best_HR@10": best_hr10,
            "local_validation_opportunities": history,
            "elapsed_wall_seconds": time.monotonic() - started,
            "per_user_host_hits_available": False,
            "uncertainty_analysis": "omitted: host per-user hits are not exposed here",
        }
    )
    with open(os.path.join(working_dir, f"{dataset_name}_experiment.json"), "w") as f:
        json.dump(settings, f, indent=2)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "rng": rng_states(),
            "completed_epochs": completed_epochs,
            "optimizer_steps": total_steps,
        },
        os.path.join(working_dir, f"{dataset_name}_training_state.pt"),
    )

    # Verify the actual serialized deployment weights using the inference
    # reconstruction arguments, without consuming the training RNG stream.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    sample_user = users[0]
    sequence = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
    sample_history = user_train[sample_user][-MAXLEN:]
    sequence[0, -len(sample_history) :] = torch.tensor(sample_history, device=device)
    candidates = torch.tensor([[1, itemnum]], device=device)
    with torch.no_grad():
        expected = model.predict(sequence, candidates)
        clone = copy.deepcopy(model)
        clone.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        torch.testing.assert_close(
            expected, clone.predict(sequence, candidates), rtol=0, atol=0
        )
        del clone
    print(
        f"  [{dataset_name}] local best HR@10={best_hr10:.4f}"
        f" resource_limited={resource_limited} ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10


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
        ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained recent-residual architecture
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            recent_residual=meta.get("recent_residual", False),
            recent_window=meta.get("recent_window", 5),
            recent_decay=meta.get("recent_decay", 0.7),
        ).to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject incorrect explicit dataset routing
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


