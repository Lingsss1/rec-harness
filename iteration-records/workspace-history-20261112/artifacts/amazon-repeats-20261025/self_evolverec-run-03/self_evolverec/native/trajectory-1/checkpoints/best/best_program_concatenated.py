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
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Required data boundary and explicit fusion controls
DATA_DIR = os.environ["GAGC_DATA_DIR"]
FUSION_MODE = os.environ.get("SASREC_FUSION_MODE", "adaptive")
if FUSION_MODE not in {"baseline", "fixed", "adaptive"}:
    raise ValueError("SASREC_FUSION_MODE must be baseline, fixed, or adaptive")
TRAIN_SECONDS = 3000.0
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


### >>> Self_EvolveRec-BLOCK-START: Restore the sequence sampler
# DEBUG: Replace the misindented placeholder with the missing function and locals.
def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, history in user_train.items() if len(history) >= 2]
    if not users:
        raise ValueError("Training requires a user with at least two interactions")
    if itemnum < 1 or batch_size < 1 or maxlen < 1:
        raise ValueError("Item count, batch size and sequence length must be positive")
    batch_seq, batch_pos, batch_neg = [], [], []
    # DEBUG: Keep the existing sampling order and positive/negative construction.
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
        # DEBUG: Fail explicitly when rejection sampling has no eligible negative.
        if sum(1 <= item <= itemnum for item in ts) >= itemnum:
            raise ValueError(f"User {u} has no unseen training-negative items")
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
    ### >>> Self_EvolveRec-BLOCK-START: Explicit reloadable temporal fusion configuration
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        fusion_mode: str = "baseline",
        fusion_config=None,
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

        ### >>> Self_EvolveRec-BLOCK-START: Preceding-evidence temporal fusion with matched initialization
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if fusion_mode not in {"baseline", "fixed", "adaptive"}:
            raise ValueError("Unknown fusion mode")
        self.fusion_mode = fusion_mode
        self.fusion_config = {
            "window": 5,
            "embedding_eps_squared": 1e-4,
            "agreement_eps_squared": 1e-4,
            "pool_offset": 0.1,
            "support_offset": 2.0,
            "minimum_preceding": 2,
            "amplitude_bound": 2.0,
        }
        if fusion_config is not None:
            if fusion_config != self.fusion_config:
                raise ValueError("Unsupported fusion constants")
        if fusion_mode != "baseline":
            self.theta = nn.Parameter(torch.tensor(math.log(0.005 / 0.995)))
        if fusion_mode == "adaptive":
            # Gate construction must not advance the backbone/dropout RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.fusion_gate = nn.Linear(hidden_units + 4, 2)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)

    def _backbone_feats(self, seq: torch.Tensor) -> torch.Tensor:
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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unused attention weight materialization
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

    ### >>> Self_EvolveRec-BLOCK-START: Causal nonpadding pools and complete tensor-valued scorer
    def fusion_components(self, seq, backbone):
        valid = seq != 0
        cfg = self.fusion_config
        e = self.item_emb(seq).float()
        x = e / torch.sqrt(
            e.square().sum(dim=-1, keepdim=True) + cfg["embedding_eps_squared"]
        )
        x = x * valid.unsqueeze(-1)
        ranks = valid.long().cumsum(dim=1)
        # Interaction-rank compaction handles internal padding without counting it.
        compact = x.new_zeros((seq.shape[0], seq.shape[1] + 1, x.shape[-1]))
        index = ranks.unsqueeze(-1).expand_as(x)
        compact = compact.scatter_add(1, index, x)
        cumulative = compact.cumsum(dim=1)

        def at(rank):
            return cumulative.gather(1, rank.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        def mean(end, window=None):
            start = (
                torch.zeros_like(end) if window is None else (end - window).clamp_min(0)
            )
            count = end - start
            value = (at(end) - at(start)) / count.clamp_min(1).unsqueeze(-1)
            return value, count

        recent, _ = mean(ranks, cfg["window"])
        prefix, _ = mean(ranks)
        previous_end = (ranks - valid.long()).clamp_min(0)
        previous_recent, nr = mean(previous_end, cfg["window"])
        previous_prefix, nprefix = mean(previous_end)

        def support(m, count):
            squared = m.square().sum(dim=-1, keepdim=True)
            concentration = m.norm(dim=-1, keepdim=True)
            agreement = (
                (x * m).sum(dim=-1, keepdim=True)
                / torch.sqrt(squared + cfg["agreement_eps_squared"])
            ).clamp(0.0, 1.0)
            n = count.float().unsqueeze(-1)
            return (
                (n >= cfg["minimum_preceding"]).float()
                * n
                / (n + cfg["support_offset"])
                * concentration
                * agreement
                * valid.unsqueeze(-1)
            )

        rr = support(previous_recent, nr)
        rp = support(previous_prefix, nprefix)
        if self.fusion_mode == "adaptive":
            gate_input = torch.cat(
                (
                    backbone.float(),
                    rr,
                    rp,
                    torch.log1p(nr.float()).unsqueeze(-1),
                    torch.log1p(nprefix.float()).unsqueeze(-1),
                ),
                dim=-1,
            )
            weights = torch.softmax(self.fusion_gate(gate_input), dim=-1)
        else:
            weights = x.new_full((*seq.shape, 2), 0.5)

        def bounded(m):
            return m / (m.norm(dim=-1, keepdim=True) + cfg["pool_offset"])

        branch = weights[..., :1] * rr * bounded(recent) + weights[
            ..., 1:
        ] * rp * bounded(prefix)
        amplitude = cfg["amplitude_bound"] * torch.sigmoid(self.theta)
        return {
            "weights": weights,
            "recent_support": rr.squeeze(-1),
            "prefix_support": rp.squeeze(-1),
            "preceding_recent_count": nr,
            "preceding_prefix_count": nprefix,
            "branch": branch,
            "scaled_branch": amplitude * branch,
            "amplitude": amplitude,
        }

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        backbone = self._backbone_feats(seq)
        if self.fusion_mode == "baseline":
            return backbone
        parts = self.fusion_components(seq, backbone)
        return (backbone + parts["scaled_branch"].to(backbone.dtype)) * (
            seq != 0
        ).unsqueeze(-1)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        if features.ndim == 2 and items.ndim == 1:
            items = items.unsqueeze(0).expand(features.shape[0], -1)
        prefix = tuple(features.shape[:-1])
        if tuple(items.shape[: len(prefix)]) != prefix:
            raise ValueError("Candidate batch/position axes must match features")
        extra_axes = items.ndim - len(prefix)
        if extra_axes < 0:
            raise ValueError("Missing candidate batch/position axes")
        aligned = features
        for _ in range(extra_axes):
            aligned = aligned.unsqueeze(-2)
        return (aligned * self.item_emb(items)).sum(dim=-1)

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        if (pos is None) != (neg is None):
            raise ValueError("Positive and negative targets must be supplied together")
        features = self.log2feats(seq)
        if pos is None:
            return features
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def final_features(self, seq):
        features = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1).clamp_min(0)
        return features[torch.arange(seq.shape[0], device=seq.device), last]

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.predict_score(self.final_features(seq), candidates)

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
    ### >>> Self_EvolveRec-BLOCK-START: Validation must not reset the training sampler stream
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
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic unique validation negatives with feasibility checking
            interacted = set(user_train[u]) | set(pos_items)
            if itemnum - sum(1 <= item <= itemnum for item in interacted) < 99:
                raise ValueError("Validation requires 99 distinct eligible negatives")
            while len(neg_pool) < 99:
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

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Instantiate fusion and exclude amplitude from weight decay
        maxlen=MAXLEN,
        fusion_mode=FUSION_MODE,
    ).to(device)

    groups = [
        {
            "params": [p for name, p in model.named_parameters() if name != "theta"],
            "weight_decay": WEIGHT_DECAY,
        }
    ]
    if hasattr(model, "theta"):
        groups.append({"params": [model.theta], "weight_decay": 0.0})
    optimizer = torch.optim.Adam(groups, lr=LR, betas=(0.9, 0.98))
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: One terminal checkpoint per category within the training allowance
    import time

    if NUM_EPOCHS < 1 or NUM_BATCHES < 1:
        raise ValueError("Epochs and batches must be positive")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Preserve the exact initialization and record it for compatible controls.
    torch.save(
        model.state_dict(), os.path.join(working_dir, f"{dataset_name}_initial.pt")
    )
    best_hr10 = -1.0
    best_epoch = 0
    updates = 0
    longest_update = 0.0
    stopped_for_time = False
    synchronize()
    training_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        ### <<< Self_EvolveRec-BLOCK-END
        model.train()
        total_loss = 0.0

        ### >>> Self_EvolveRec-BLOCK-START: Stop between complete updates with a conservative timing allowance
        epoch_updates = 0
        for _ in range(NUM_BATCHES):
            synchronize()
            elapsed = time.perf_counter() - training_start
            if updates and elapsed + max(2.0 * longest_update, 1.0) >= TRAIN_SECONDS:
                stopped_for_time = True
                break
            update_start = time.perf_counter()
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
            ### >>> Self_EvolveRec-BLOCK-START: Score only supervised nonpadding positions
            indices = (seq_t != 0) & (pos_t != 0)
            if not bool(indices.any()):
                raise ValueError("Batch has no supervised positions")
            ### <<< Self_EvolveRec-BLOCK-END
            pos_labels = torch.ones(pos_logits[indices].shape, device=device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
            loss = bce_criterion(pos_logits[indices], pos_labels)
            loss = loss + bce_criterion(neg_logits[indices], neg_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ### >>> Self_EvolveRec-BLOCK-START: Measure occupied training time and validate only the terminal state
            total_loss += loss.item()
            synchronize()
            longest_update = max(longest_update, time.perf_counter() - update_start)
            updates += 1
            epoch_updates += 1

        avg_loss = total_loss / max(epoch_updates, 1)

        if epoch == NUM_EPOCHS or stopped_for_time:
            synchronize()
            charged_training_seconds = time.perf_counter() - training_start
            print(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "fusion_mode": model.fusion_mode,
                        "optimizer_updates": updates,
                        "charged_training_seconds": charged_training_seconds,
                        "training_cap_seconds": TRAIN_SECONDS,
                        "cap_overshoot_seconds": max(
                            0.0, charged_training_seconds - TRAIN_SECONDS
                        ),
                        "selection_checkpoint": "terminal",
                    }
                ),
                flush=True,
            )
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
            ### >>> Self_EvolveRec-BLOCK-START: Terminal selection and descriptive fusion measurements
            if model.fusion_mode != "baseline":
                model.eval()
                with torch.no_grad():
                    h = model._backbone_feats(seq_t)
                    parts = model.fusion_components(seq_t, h)
                    augmented = (h + parts["scaled_branch"].to(h.dtype)) * (
                        seq_t != 0
                    ).unsqueeze(-1)
                    targets = torch.stack((pos_t, neg_t), dim=-1)
                    base_scores = model.predict_score(h, targets)
                    full_scores = model.predict_score(augmented, targets)
                    base_margin = base_scores[..., 0] - base_scores[..., 1]
                    full_margin = full_scores[..., 0] - full_scores[..., 1]
                    record = {
                        "dataset": dataset_name,
                        "valid_positions": int(indices.sum().item()),
                        "amplitude": float(parts["amplitude"].item()),
                        "last_theta_abs_gradient": (
                            float(model.theta.grad.detach().abs().item())
                            if model.theta.grad is not None
                            else None
                        ),
                        "mean_recent_weight": float(
                            parts["weights"][..., 0][indices].mean().item()
                        ),
                        "mean_recent_support": float(
                            parts["recent_support"][indices].mean().item()
                        ),
                        "mean_prefix_support": float(
                            parts["prefix_support"][indices].mean().item()
                        ),
                        "zero_support_rate": float(
                            (
                                (parts["recent_support"] == 0)
                                & (parts["prefix_support"] == 0)
                            )[indices]
                            .float()
                            .mean()
                            .item()
                        ),
                        "mean_scaled_branch_norm": float(
                            parts["scaled_branch"].norm(dim=-1)[indices].mean().item()
                        ),
                        "backbone_margin_mean": float(
                            base_margin[indices].mean().item()
                        ),
                        "augmented_margin_mean": float(
                            full_margin[indices].mean().item()
                        ),
                        "mean_absolute_score_change": float(
                            (full_scores - base_scores)[indices].abs().mean().item()
                        ),
                        "interpretation": "Descriptive final sampled training batch; not a validation metric",
                    }
                with open(
                    os.path.join(
                        working_dir, f"{dataset_name}_fusion_diagnostics.json"
                    ),
                    "w",
                ) as f:
                    json.dump(record, f, indent=2)
            break
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
                ### >>> Self_EvolveRec-BLOCK-START: Serialize the exact fusion architecture
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
                "fusion_mode": model.fusion_mode,
                "fusion_config": model.fusion_config,
                "selection_checkpoint": "terminal",
                "training_seed": 42,
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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

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
        ### >>> Self_EvolveRec-BLOCK-START: Require every dataset artifact
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing trained artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reload the trained fusion architecture in the predictor writer
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            fusion_mode=meta["fusion_mode"],
            fusion_config=meta["fusion_config"],
        ).to(_device)
            ### <<< Self_EvolveRec-BLOCK-END
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    # Select model: use dataset_name kwarg if provided, else fall back to first available
    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit dataset routing
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown or unloaded dataset {dataset_name!r}")
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

    ### >>> Self_EvolveRec-BLOCK-START: Encode once for candidate and full-catalog chunk scoring
    if len(candidates) == 0:
        return []
    with torch.inference_mode():
        features = model.final_features(seq_t)
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor(
                [candidates[start:start + 4096]], dtype=torch.long, device=_device
            )
            scores.extend(model.predict_score(features, cand_t)[0].cpu().tolist())
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
import random
from collections import Counter


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
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the diagnostic view contract and unavailable metadata boundary
    def __init__(self, view, device):
        self.model = view
        self.trained_model = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[1:]

            n = emb.size(0)
            ### >>> Self_EvolveRec-BLOCK-START: Undefined embedding alignment is null
            if n < 2 or sample_size < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Finite backbone cosine alignment without a quadratic Gram matrix
            x = emb.index_select(0, idx).float()
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError("Nonfinite diagnostic item embeddings")
            x = x[x.norm(dim=-1) > 1e-8]
            k = x.shape[0]
            if k < 2:
                return None
            x = F.normalize(x, p=2, dim=-1, eps=1e-8).double()
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
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
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual trained module through the compatibility view
        self.trained_model.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            ### >>> Self_EvolveRec-BLOCK-START: Clean tensor-valued diagnostic scoring with explicit batch validation
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError("Expected (None, sequence, positive, negative)")
            _, seq, pos, neg = batch
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic tensors must share shape B x L")
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            all_items = torch.stack([pos_t, neg_t], dim=-1)
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Measure finite margins only at valid supervised positions
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if not bool(torch.isfinite(margin[valid]).all()):
                raise FloatingPointError("Nonfinite diagnostic margins")
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())
            ### <<< Self_EvolveRec-BLOCK-END

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same validity mask for margin breakdowns
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
            margin_metrics = {
                ### >>> Self_EvolveRec-BLOCK-START: Empty diagnostic samples have undefined statistics
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
                ### <<< Self_EvolveRec-BLOCK-END
            }
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            "eps": float(eps),
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

    ### >>> Self_EvolveRec-BLOCK-START: Clean diagnosis restores every module mode and training RNG stream
    def run_full_diagnosis(self, train_loader):
        saved_modes = [
            (module, module.training) for module in self.trained_model.modules()
        ]
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        try:
            self.trained_model.eval()
            with torch.no_grad():
                result = self._run_full_diagnosis(train_loader)
            gradients = [
                parameter.grad.detach()
                for parameter in self.trained_model.parameters()
                if parameter.grad is not None
            ]
            result["metrics"]["existing_gradients_finite"] = (
                bool(
                    all(torch.isfinite(gradient).all().item() for gradient in gradients)
                )
                if gradients
                else None
            )
            result["metric_definitions"]["existing_gradients_finite"] = (
                "Whether retained gradients are finite; null when none are retained. "
                "Diagnosis performs no backward pass."
            )
            return result
        finally:
            for module, previous_mode in saved_modes:
                module.training = previous_mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _run_full_diagnosis(self, train_loader):
        print(">>> Running clean descriptive training diagnosis...")
        ### <<< Self_EvolveRec-BLOCK-END

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Limit alignment interpretation to the measured backbone table
                    "Sampled mean off-diagonal cosine of nonzero backbone item embeddings, "
                    "mapped to [0,1]. Higher means directional alignment, not definitive "
                    "collapse or recommendation diversity. Null with fewer than two usable embeddings."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                "pos_neg_margin_mean": "Mean of (pos_logit - neg_logit) over valid positions; higher means clearer separation.",
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                "pos_neg_margin_category_breakdown": (
                    ### >>> Self_EvolveRec-BLOCK-START: Describe only available training-position evidence
                    "Counts valid clean training-loader positions below eps or zero. "
                    "Category lists remain empty because item categories are unavailable. "
                    "These are descriptive training diagnostics, not validation accuracy."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
            },
        }
