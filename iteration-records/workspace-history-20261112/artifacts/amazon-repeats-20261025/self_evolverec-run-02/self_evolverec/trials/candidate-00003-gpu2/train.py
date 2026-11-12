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

### >>> Self_EvolveRec-BLOCK-START: Configure baseline-preserving recent-interest controls
DATA_DIR = os.environ["GAGC_DATA_DIR"]

# One configuration per host trial; all configurations train all four datasets.
# Checkpoints remain inspected every five epochs with the existing stopping policy.
RESIDUAL_CONFIGS = {
    "baseline": (0, 0.7, False),
    "zero_fixed": (5, 0.7, False),
    "last_item": (1, 0.7, True),
    "recent_five": (5, 0.7, True),
}
RESIDUAL_MODE = os.environ.get("SASREC_RESIDUAL_MODE", "recent_five")
if RESIDUAL_MODE not in RESIDUAL_CONFIGS:
    raise ValueError(f"Unknown SASREC_RESIDUAL_MODE: {RESIDUAL_MODE}")
RECENT_WINDOW, RECENT_DECAY, LEARN_RECENT_ALPHA = RESIDUAL_CONFIGS[RESIDUAL_MODE]
RECENT_NORM_EPS = 1e-5
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four training datasets
DATASETS = DEFAULT_DATASETS[:]
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


### >>> Self_EvolveRec-BLOCK-START: Reuse eligible training users without changing sampling probabilities
def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, users=None
):
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No training users have at least two interactions")
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
    ### >>> Self_EvolveRec-BLOCK-START: Explicit residual architecture configuration
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_window: int = 0,
        recent_decay: float = 0.7,
        recent_norm_eps: float = 1e-5,
        learn_recent_alpha: bool = True,
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

        ### >>> Self_EvolveRec-BLOCK-START: Initialize the residual after the unchanged baseline without consuming RNG
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if recent_window < 0 or not 0.0 < recent_decay <= 1.0:
            raise ValueError("Invalid recent-interest pooling configuration")
        self.recent_window = int(recent_window)
        self.recent_decay = float(recent_decay)
        if self.recent_window:
            # LayerNorm initializes to ones/zeros and consumes no random numbers.
            self.recent_norm = nn.LayerNorm(hidden_units, eps=recent_norm_eps)
            self.recent_alpha = nn.Parameter(
                torch.zeros(()), requires_grad=learn_recent_alpha
            )
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Expose unchanged baseline features for equivalence checks
    def _base_log2feats(self, seq: torch.Tensor) -> torch.Tensor:
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

    ### >>> Self_EvolveRec-BLOCK-START: Causal normalized recent-interest residual
    def _recent_branch(self, seq: torch.Tensor) -> torch.Tensor:
        valid = seq.ne(0)
        # Unscaled shared scoring embeddings; padding contributes zero even if its
        # embedding row was overwritten by the baseline normal initialization.
        emb = self.item_emb(seq) * valid.unsqueeze(-1)
        pooled = torch.zeros_like(emb)
        denominator = emb.new_zeros((*seq.shape, 1))
        length = seq.size(1)
        for lag in range(min(self.recent_window, length)):
            weight = self.recent_decay**lag
            source = emb[:, : length - lag, :]
            source_valid = valid[:, : length - lag].unsqueeze(-1)
            if lag:
                source = torch.nn.functional.pad(source, (0, 0, lag, 0))
                source_valid = torch.nn.functional.pad(source_valid, (0, 0, lag, 0))
            pooled = pooled + weight * source
            denominator = denominator + weight * source_valid.to(emb.dtype)
        pooled = pooled / denominator.clamp_min(torch.finfo(emb.dtype).tiny)
        branch = self.recent_norm(pooled)
        return branch * (valid.unsqueeze(-1) & denominator.gt(0))

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        base = self._base_log2feats(seq)
        if not self.recent_window:
            return base
        branch = self._recent_branch(seq)
        return (base + self.recent_alpha * branch) * seq.ne(0).unsqueeze(-1)

    @torch.no_grad()
    def verify_initial_equivalence(self):
        if not self.recent_window:
            return
        was_training = self.training
        self.eval()
        try:
            seq = torch.zeros(
                (2, self.maxlen), dtype=torch.long, device=self.item_emb.weight.device
            )
            count = min(5, self.maxlen)
            seq[0, -count:] = (
                torch.arange(count, device=seq.device) % self.itemnum
            ) + 1
            seq[1, -1] = 1
            base = self._base_log2feats(seq)
            fused = self.log2feats(seq)
            valid = seq.ne(0)
            if not torch.equal(base[valid], fused[valid]):
                raise RuntimeError(
                    "Zero-initialized residual changed baseline features"
                )
            candidates = seq.new_tensor([[1], [1]])
            expected = (
                self.item_emb(candidates)
                .matmul(base[:, -1, :].unsqueeze(-1))
                .squeeze(-1)
            )
            if not torch.equal(expected, self.predict(seq, candidates)):
                raise RuntimeError(
                    "Zero-initialized residual changed prediction logits"
                )
            if self.recent_alpha.item() != 0.0:
                raise RuntimeError("Residual scalar must initialize to zero")
        finally:
            self.train(was_training)

    @torch.no_grad()
    def residual_diagnostics(self, seq, pos, neg):
        # Called only at an existing validation checkpoint on the last train batch.
        was_training = self.training
        self.eval()
        try:
            base = self._base_log2feats(seq)
            valid = seq.ne(0)
            branch = (
                self._recent_branch(seq)
                if self.recent_window
                else torch.zeros_like(base)
            )
            alpha = self.recent_alpha if self.recent_window else base.new_zeros(())
            residual = alpha * branch
            features = (base + residual) * valid.unsqueeze(-1)
            active = pos.ne(0) & valid
            metrics = {"alpha": float(alpha)}
            if active.any():
                margins = (features * (self.item_emb(pos) - self.item_emb(neg))).sum(
                    -1
                )[active]
                metrics.update(
                    {
                        "baseline_norm": float(base[active].norm(dim=-1).mean()),
                        "branch_norm": float(branch[active].norm(dim=-1).mean()),
                        "residual_norm": float(residual[active].norm(dim=-1).mean()),
                        "margin_p10": float(torch.quantile(margins.float(), 0.1)),
                        "negative_over_positive_rate": float(
                            (margins < 0).float().mean()
                        ),
                    }
                )
            return metrics
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
        ### >>> Self_EvolveRec-BLOCK-START: Score from the latest valid observed position
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1)
        final_feat = log_feats[
            torch.arange(seq.size(0), device=seq.device), last.clamp_min(0)
        ]
        final_feat = final_feat * last.ge(0).unsqueeze(-1)
        ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Keep checkpoint validation deterministic without resetting training sampling
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    if not val_users:
        raise ValueError("No validation users overlap the training split")
    for u in val_users:
        if not user_valid[u]:
            raise ValueError(f"Validation user {u} has no target")
        interacted = set(user_train[u]) | set(user_valid[u])
        excluded = sum(1 <= item <= itemnum for item in interacted)
        if itemnum - excluded < 99:
            raise ValueError(f"Validation user {u} has fewer than 99 unseen candidates")

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
            while len(neg_pool) < 99:
                ### >>> Self_EvolveRec-BLOCK-START: Draw validation candidates from the isolated evaluation RNG
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

    ### >>> Self_EvolveRec-BLOCK-START: Restore the caller's model mode after checkpoint validation
    n = len(val_users)
    model.train(was_training)
    ### <<< Self_EvolveRec-BLOCK-END
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### >>> Self_EvolveRec-BLOCK-START: Validate sampling support and initialize each dataset with seed 42
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    eligible_users = [u for u, items in user_train.items() if len(items) >= 2]
    if not eligible_users or itemnum < 1:
        raise ValueError(f"[{dataset_name}] No usable training interactions")
    if any(len(set(user_train[u])) >= itemnum for u in eligible_users):
        raise ValueError(
            f"[{dataset_name}] A training user has no unseen negative items"
        )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the selected residual and preserve baseline optimizer settings
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        recent_window=RECENT_WINDOW,
        recent_decay=RECENT_DECAY,
        recent_norm_eps=RECENT_NORM_EPS,
        learn_recent_alpha=LEARN_RECENT_ALPHA,
    ).to(device)
    model.verify_initial_equivalence()
    print(
        f"  [{dataset_name}] residual={RESIDUAL_MODE}; seed=42; "
        f"validation every 5 epochs; patience={PATIENCE * 5} epochs; "
        f"maximum={NUM_EPOCHS}; exact checkpoint ties retain earlier checkpoint",
        flush=True,
    )

    baseline_parameters = []
    branch_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name.startswith("recent_"):
                branch_parameters.append(parameter)
            else:
                baseline_parameters.append(parameter)
    parameter_groups = [{"params": baseline_parameters, "weight_decay": WEIGHT_DECAY}]
    if branch_parameters:
        parameter_groups.append({"params": branch_parameters, "weight_decay": 0.0})
    optimizer = torch.optim.Adam(parameter_groups, lr=LR, betas=(0.9, 0.98))
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first scheduled checkpoint even when validation HR is zero
    best_hr10 = -float("inf")
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Invoke cached-user sampling in every training batch
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, users=eligible_users
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

        if epoch % 5 == 0:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### >>> Self_EvolveRec-BLOCK-START: Report measured residual diagnostics at existing checkpoints
            hr10 = val_m["HR@10"]
            if NUM_BATCHES > 0:
                diagnostics = model.residual_diagnostics(seq_t, pos_t, neg_t)
                print(
                    f"  [{dataset_name}] last_train_batch_diagnostics="
                    + json.dumps(diagnostics, sort_keys=True),
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist the complete residual architecture alongside its state dictionary
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
                "architecture": "SASRecRecentResidualV1",
                "residual_mode": RESIDUAL_MODE,
                "recent_window": RECENT_WINDOW,
                "recent_decay": RECENT_DECAY,
                "recent_norm_eps": RECENT_NORM_EPS,
                "learn_recent_alpha": LEARN_RECENT_ALPHA,
                "alpha_initial": 0.0,
                "alpha_state_key": "recent_alpha" if RECENT_WINDOW else None,
                "best_epoch": best_epoch,
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
### >>> Self_EvolveRec-BLOCK-START: Retain all four inference routes
DATASETS = DEFAULT_DATASETS[:]
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
        ### >>> Self_EvolveRec-BLOCK-START: Require artifacts for every dataset in generated predict.py
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model or metadata for {ds} in {_HERE}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the exact trained residual in generated predict.py
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            recent_window=meta["recent_window"],
            recent_decay=meta["recent_decay"],
            recent_norm_eps=meta["recent_norm_eps"],
            learn_recent_alpha=meta["learn_recent_alpha"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid dataset routes instead of silently scoring with another model
    ds = DEFAULT_DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DEFAULT_DATASETS:
        raise ValueError(f"Unknown dataset_name: {ds}")
    if ds not in _models:
        raise RuntimeError(f"Model for {ds} was not loaded")
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


### >>> Self_EvolveRec-BLOCK-START: Reject runs that cannot produce a scheduled validation checkpoint
def main():
    if NUM_EPOCHS < 5 or NUM_BATCHES < 1 or BATCH_SIZE < 1:
        raise ValueError(
            "Require SASREC_EPOCHS >= 5, SASREC_BATCHES >= 1 and SASREC_BATCH >= 1"
        )
    if MAXLEN < 1 or PATIENCE < 1:
        raise ValueError("Require positive SASREC_MAXLEN and SASREC_PATIENCE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### <<< Self_EvolveRec-BLOCK-END
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


