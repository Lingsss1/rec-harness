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
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host-provided training data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain every matched dataset
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


### >>> Self_EvolveRec-BLOCK-START: Reuse eligible users without changing sampling probabilities or RNG order
def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, users=None
):
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("Training requires at least one user with two interactions")
    ### <<< Self_EvolveRec-BLOCK-END
    batch_seq, batch_pos, batch_neg = [], [], []
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
    ### >>> Self_EvolveRec-BLOCK-START: Configure the bounded recent-history residual
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_k: int = 5,
        recent_decay: float = 0.7,
        recent_bound: float = 0.5,
        recent_eps: float = 1e-8,
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

        ### >>> Self_EvolveRec-BLOCK-START: Initialize the branch after the unchanged backbone without consuming RNG
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if recent_k < 1 or not 0.0 < recent_decay <= 1.0:
            raise ValueError(
                "recent_k must be positive and recent_decay must be in (0, 1]"
            )
        if recent_bound < 0.0 or recent_eps <= 0.0:
            raise ValueError("recent_bound must be nonnegative and recent_eps positive")
        self.recent_k = int(recent_k)
        self.recent_decay = float(recent_decay)
        self.recent_bound = float(recent_bound)
        self.recent_eps = float(recent_eps)
        # LayerNorm's ones/zeros initialization and torch.zeros consume no RNG.
        self.recent_norm = nn.LayerNorm(hidden_units, eps=self.recent_eps)
        self.a = nn.Parameter(torch.zeros(()))

    def _backbone_log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        ### <<< Self_EvolveRec-BLOCK-END
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

        return self.last_layernorm(seq_emb)

    ### >>> Self_EvolveRec-BLOCK-START: Fuse a causal normalized recent pool without outer normalization
    def _recent_features(self, seq: torch.Tensor) -> torch.Tensor:
        embeddings = self.item_emb(seq)
        pooled = torch.zeros_like(embeddings)
        denominator = embeddings.new_zeros((*seq.shape, 1))
        length = seq.size(1)
        for lag in range(min(self.recent_k, length)):
            valid = seq[:, : length - lag].ne(0).unsqueeze(-1)
            weight = valid.to(embeddings.dtype) * (self.recent_decay**lag)
            pooled[:, lag:, :] += embeddings[:, : length - lag, :] * weight
            denominator[:, lag:, :] += weight
        pooled = pooled / denominator.clamp_min(1e-8)
        return self.recent_norm(pooled) * seq.ne(0).unsqueeze(-1)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        backbone = self._backbone_log2feats(seq)
        recent = self._recent_features(seq)
        coefficient = self.recent_bound * torch.tanh(self.a)
        return (backbone + coefficient * recent) * seq.ne(0).unsqueeze(-1)

    @torch.no_grad()
    def residual_diagnostics(self, seq: torch.Tensor) -> dict:
        was_training = self.training
        self.eval()
        try:
            valid = seq.ne(0)
            backbone = self._backbone_log2feats(seq)[valid]
            recent = self._recent_features(seq)[valid]
            coefficient = self.recent_bound * torch.tanh(self.a)
            backbone_norm = (
                backbone.norm(dim=-1).mean()
                if valid.any()
                else seq.new_zeros((), dtype=torch.float)
            )
            recent_norm = recent.norm(dim=-1).mean() if valid.any() else backbone_norm
            ratio = recent_norm / backbone_norm.clamp_min(1e-8)
            return {
                "metrics": {
                    "coefficient": float(coefficient.item()),
                    "a_gradient_after_clip": (
                        None if self.a.grad is None else float(self.a.grad.item())
                    ),
                    "recent_to_backbone_norm": float(ratio.item()),
                    "residual_to_backbone_norm": float(
                        (coefficient.abs() * ratio).item()
                    ),
                    "valid_positions": int(valid.sum().item()),
                },
                "metric_definitions": {
                    "coefficient": "Signed bounded residual coefficient: recent_bound * tanh(a).",
                    "a_gradient_after_clip": "Last minibatch gradient of a after global gradient clipping, before its optimizer update; null if unavailable.",
                    "recent_to_backbone_norm": "Mean normalized recent-branch L2 norm divided by mean backbone L2 norm over nonpadding positions, with dropout disabled.",
                    "residual_to_backbone_norm": "Absolute residual coefficient times recent_to_backbone_norm.",
                    "valid_positions": "Number of nonpadding positions in the diagnostic minibatch.",
                },
            }
        finally:
            self.train(was_training)

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


### >>> Self_EvolveRec-BLOCK-START: Verify baseline initialization and exact architecture reconstruction
@torch.no_grad()
def verify_residual_initialization(model: SASRec, device: torch.device) -> dict:
    was_training = model.training
    model.eval()
    try:
        seq = (
            torch.arange(2 * model.maxlen, device=device).reshape(2, model.maxlen)
            % model.itemnum
            + 1
        ).long()
        seq[:, : model.maxlen // 2] = 0
        candidates = torch.arange(1, min(model.itemnum, 16) + 1, device=device)
        candidates = candidates.unsqueeze(0).expand(2, -1)
        baseline = model._backbone_log2feats(seq)
        fused = model.log2feats(seq)
        valid = seq.ne(0)
        torch.testing.assert_close(fused[valid], baseline[valid], atol=1e-6, rtol=1e-5)
        expected = (
            model.item_emb(candidates)
            .matmul(baseline[:, -1, :].unsqueeze(-1))
            .squeeze(-1)
        )
        actual = model.predict(seq, candidates)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert model.a.item() == 0.0
        if model.maxlen > 1:
            changed = seq.clone()
            changed[:, -1] = changed[:, -1] % model.itemnum + 1
            torch.testing.assert_close(
                model.log2feats(changed)[:, :-1], fused[:, :-1], atol=1e-6, rtol=1e-5
            )
        return {
            "initial_coefficient": 0.0,
            "initial_max_score_difference": float(
                (actual - expected).abs().max().item()
            ),
            "atol": 1e-6,
            "rtol": 1e-5,
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def verify_checkpoint_reload(
    model: SASRec,
    checkpoint: str,
    config: dict,
    seq: torch.Tensor,
    candidates: torch.Tensor,
    device: torch.device,
) -> float:
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    expected = model.predict(seq, candidates)
    # Construction initializes CPU tensors only; restore that RNG stream afterward.
    with torch.random.fork_rng(devices=[]):
        restored = SASRec(**config).to(device)
    restored.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    restored.eval()
    actual = restored.predict(seq, candidates)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    return float((actual - expected).abs().max().item())


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
    ### >>> Self_EvolveRec-BLOCK-START: Make local validation repeatable without resetting the training sampler
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
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when unique validation negatives are unavailable
            interacted = set(user_train[u]) | set(pos_items)
            available_negatives = itemnum - sum(
                1 <= item <= itemnum for item in interacted
            )
            if available_negatives < 99:
                raise ValueError(
                    f"Validation user {u} has only {available_negatives} unseen items; "
                    "99 unique negatives are required"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            while len(neg_pool) < 99:
                ### >>> Self_EvolveRec-BLOCK-START: Isolate validation candidate randomness
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


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    ### >>> Self_EvolveRec-BLOCK-START: Train the zero-initialized residual with the unchanged backbone configuration
    model_config = {
        "itemnum": itemnum,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "recent_k": 5,
        "recent_decay": 0.7,
        "recent_bound": 0.5,
        "recent_eps": 1e-8,
    }
    model = SASRec(**model_config).to(device)
    initialization_check = verify_residual_initialization(model, device)
    print(
        f"  [{dataset_name}] initialization_check={json.dumps(initialization_check)}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Cache eligible training users and reject impossible negative sampling
    sampling_users = [u for u, items in user_train.items() if len(items) >= 2]
    if not sampling_users:
        raise ValueError(f"[{dataset_name}] No users eligible for sequence training")
    if MAXLEN < 1 or NUM_EPOCHS < 1 or NUM_BATCHES < 1 or BATCH_SIZE < 1:
        raise ValueError(
            "MAXLEN, NUM_EPOCHS, NUM_BATCHES and BATCH_SIZE must be positive"
        )
    for u in sampling_users:
        if len(set(user_train[u]) & set(range(1, itemnum + 1))) >= itemnum:
            raise ValueError(
                f"[{dataset_name}] User {u} has no unseen training negative"
            )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint and log measured residual diagnostics
    best_hr10 = -float("inf")
    best_epoch = 0
    diagnostics_path = os.path.join(working_dir, f"{dataset_name}_residual.jsonl")
    with open(diagnostics_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "event": "initialization",
                    "config": model_config,
                    "checks": initialization_check,
                    "selection": "HR@10; evaluate every five epochs; retain earlier checkpoint on exact ties",
                }
            )
            + "\n"
        )
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Invoke cached user sampling on every training step
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, users=sampling_users
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
            indices = pos_t != 0
            pos_labels = torch.ones(pos_logits[indices].shape, device=device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
            loss = bce_criterion(pos_logits[indices], pos_labels)
            loss = loss + bce_criterion(neg_logits[indices], neg_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the final epoch even for short training runs
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            ### <<< Self_EvolveRec-BLOCK-END
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### >>> Self_EvolveRec-BLOCK-START: Record the signed coefficient, measured gradient, and branch scale
            hr10 = val_m["HR@10"]
            diagnostics = model.residual_diagnostics(seq_t)
            with open(diagnostics_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "validation",
                            "epoch": epoch,
                            "local_validation": val_m,
                            **diagnostics,
                        }
                    )
                    + "\n"
                )
            print(
                f"  [{dataset_name}] residual={json.dumps(diagnostics['metrics'])}",
                flush=True,
            )
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

    ### >>> Self_EvolveRec-BLOCK-START: Require a checkpoint selected during this run
    if best_epoch == 0 or not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"[{dataset_name}] No validation-selected checkpoint was saved"
        )
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Persist the complete architecture and check selected checkpoint reconstruction
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(model_config, f)
    check_seq = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
    check_history = next(items for items in user_train.values() if len(items) >= 2)[
        -MAXLEN:
    ]
    check_seq[0, -len(check_history) :] = torch.tensor(
        check_history, dtype=torch.long, device=device
    )
    check_candidates = torch.arange(1, min(itemnum, 32) + 1, device=device).unsqueeze(0)
    reload_difference = verify_checkpoint_reload(
        model, ckpt_path, model_config, check_seq, check_candidates, device
    )
    with open(diagnostics_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "event": "checkpoint_reload",
                    "best_epoch": best_epoch,
                    "max_score_difference": reload_difference,
                    "atol": 1e-6,
                    "rtol": 1e-5,
                }
            )
            + "\n"
        )
    print(
        f"  [{dataset_name}] checkpoint_reload_max_score_difference={reload_difference:.8g}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END

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
### >>> Self_EvolveRec-BLOCK-START: Export every matched dataset route
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
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained recent-history residual during inference
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            recent_k=meta["recent_k"],
            recent_decay=meta["recent_decay"],
            recent_bound=meta["recent_bound"],
            recent_eps=meta["recent_eps"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit dataset routes instead of scoring with another category
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset_name={dataset_name!r}")
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
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


