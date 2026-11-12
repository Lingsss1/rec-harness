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
### >>> Self_EvolveRec-BLOCK-START: Respect the matched training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Configure bounded recent-anchor retrieval
DATA_DIR = os.environ["GAGC_DATA_DIR"]
RETRIEVAL_MODE = os.environ.get("SASREC_RETRIEVAL_MODE", "retrieval")
if RETRIEVAL_MODE not in {"baseline", "retrieval"}:
    raise ValueError("SASREC_RETRIEVAL_MODE must be baseline or retrieval")
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

### >>> Self_EvolveRec-BLOCK-START: Retain the complete four-category training bundle
if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
    raise ValueError("All four benchmark datasets must be retained exactly once")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Reject impossible sampling without changing the negative distribution
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users or itemnum < 1 or batch_size < 1 or maxlen < 1:
        raise ValueError("Training requires eligible users and positive dimensions")
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
        ### >>> Self_EvolveRec-BLOCK-START: Prevent an infinite history-excluding negative-sampling loop
        ts = set(history)
        if sum(1 <= item <= itemnum for item in ts) >= itemnum:
            raise ValueError(f"Training user {u!r} has no unseen negative item")
        for item in reversed(history[:-1]):
            ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Persist the retrieval architecture explicitly
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        retrieval_mode: str = "baseline",
        retrieval_config=None,
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

        ### >>> Self_EvolveRec-BLOCK-START: Deterministic bounded retrieval amplitude and serialized constants
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if retrieval_mode not in {"baseline", "retrieval"}:
            raise ValueError("Unknown retrieval mode")
        self.retrieval_mode = retrieval_mode
        self.retrieval_config = {
            "embedding_eps_squared": 1e-4,
            "decay_distance": 10.0,
            "kernel_exponent": 2,
            "denominator_constant": 1.0,
            "amplitude_bound": 2.0,
            "initial_amplitude": 0.01,
            "strictly_preceding": True,
            "version": 1,
        }
        if retrieval_config is not None and retrieval_config != self.retrieval_config:
            raise ValueError("Unsupported retrieval configuration")
        if retrieval_mode == "retrieval":
            self.theta = nn.Parameter(torch.tensor(math.log(0.005 / 0.995)))

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

        ### >>> Self_EvolveRec-BLOCK-START: Keep padding features zero after learned normalization
        return self.last_layernorm(seq_emb).masked_fill(
            timeline_mask.unsqueeze(-1), 0.0
        )
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Causal kernel retrieval with selective inference queries and explicit scoring layouts
    def retrieval_components(self, seq: torch.Tensor, query_positions=None):
        if self.retrieval_mode != "retrieval":
            raise ValueError("Retrieval components require retrieval mode")
        cfg = self.retrieval_config
        # Disable autocast locally so the kernel and its denominators stay float32.
        with torch.autocast(device_type=seq.device.type, enabled=False):
            valid = seq.ne(0)
            e = self.item_emb(seq).float()
            x = e / torch.sqrt(
                e.square().sum(dim=-1, keepdim=True) + cfg["embedding_eps_squared"]
            )
            x = x * valid.unsqueeze(-1)
            rank = valid.long().cumsum(dim=1)
            length = seq.shape[1]
            key_positions = torch.arange(length, device=seq.device)
            if query_positions is None:
                query_positions = key_positions.unsqueeze(0).expand(seq.shape[0], -1)
            else:
                query_positions = torch.as_tensor(
                    query_positions, dtype=torch.long, device=seq.device
                )
                if (
                    query_positions.ndim != 2
                    or query_positions.shape[0] != seq.shape[0]
                ):
                    raise ValueError("Query positions must have shape B x Q")
                if bool(((query_positions < 0) | (query_positions >= length)).any()):
                    raise ValueError("Retrieval query position outside the sequence")
            query_x = x.gather(
                1, query_positions.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            )
            query_valid = valid.gather(1, query_positions)
            query_rank = rank.gather(1, query_positions)
            age = query_rank.unsqueeze(-1) - rank.unsqueeze(-2)
            preceding = key_positions.view(1, 1, -1) < query_positions.unsqueeze(-1)
            allowed = preceding & query_valid.unsqueeze(-1) & valid.unsqueeze(-2)
            similarity = torch.matmul(query_x, x.transpose(-1, -2))
            weights = torch.exp(
                -age.clamp_min(0).float() / cfg["decay_distance"]
            ) * similarity.relu().pow(cfg["kernel_exponent"])
            weights = weights.masked_fill(~allowed, 0.0)
            mass = weights.sum(dim=-1, keepdim=True)
            residual = torch.matmul(weights, x) / (cfg["denominator_constant"] + mass)
            amplitude = cfg["amplitude_bound"] * torch.sigmoid(self.theta)
            return {
                "mass": mass.squeeze(-1),
                "residual": residual,
                "scaled_residual": amplitude * residual,
                "amplitude": amplitude,
            }

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        h = self._backbone_feats(seq)
        if self.retrieval_mode == "baseline":
            return h
        parts = self.retrieval_components(seq)
        return (h + parts["scaled_residual"].to(h.dtype)) * seq.ne(0).unsqueeze(-1)

    def predict_score(self, features, items, candidate_layout=None):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        prefix = tuple(features.shape[:-1])
        if candidate_layout is None:
            if items.ndim == 1 and tuple(items.shape) == prefix:
                raise ValueError(
                    "Ambiguous one-dimensional items: specify aligned or shared layout"
                )
            if tuple(items.shape) == prefix:
                candidate_layout = "aligned"
            elif items.ndim == 1:
                candidate_layout = "shared"
            elif tuple(items.shape[:-1]) == prefix:
                candidate_layout = "candidates"
            else:
                raise ValueError("Candidate layout must be supplied explicitly")
        if candidate_layout == "aligned":
            if tuple(items.shape) != prefix:
                raise ValueError("Aligned item axes must match feature axes")
            return (features * self.item_emb(items)).sum(dim=-1)
        if candidate_layout == "shared":
            if items.ndim != 1:
                raise ValueError("Shared candidates must be a one-dimensional catalog")
            return features @ self.item_emb(items).transpose(0, 1)
        if candidate_layout == "candidates":
            if tuple(items.shape[:-1]) != prefix:
                raise ValueError("Candidate axes must preserve batch and position axes")
            return (features.unsqueeze(-2) * self.item_emb(items)).sum(dim=-1)
        raise ValueError(f"Unknown candidate layout {candidate_layout!r}")

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
        return (
            self.predict_score(features, pos, "aligned"),
            self.predict_score(features, neg, "aligned"),
        )

    def final_features(self, seq: torch.Tensor):
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp_min(0)
        rows = torch.arange(seq.shape[0], device=seq.device)
        # Training keeps every causal query; inference retrieves only for the
        # selected final position, reducing retrieval storage from B*L*L to B*L.
        backbone = self._backbone_feats(seq)[rows, last]
        if self.retrieval_mode == "baseline":
            return backbone
        parts = self.retrieval_components(seq, query_positions=last.unsqueeze(-1))
        correction = parts["scaled_residual"][:, 0].to(backbone.dtype)
        return (backbone + correction) * seq.ne(0).any(dim=1, keepdim=True)

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        layout = "shared" if candidates.ndim == 1 else "candidates"
        return self.predict_score(self.final_features(seq), candidates, layout)

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
    ### >>> Self_EvolveRec-BLOCK-START: Keep validation sampling independent of the training RNG
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
            ### >>> Self_EvolveRec-BLOCK-START: Validate distinct-negative availability before sampling
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

### >>> Self_EvolveRec-BLOCK-START: Train the retrieval candidate with unchanged BCE and terminal checkpoint selection


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    import time

    if NUM_EPOCHS < 1 or NUM_BATCHES < 1:
        raise ValueError("Epoch and batch limits must be positive")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    occupied_start = time.perf_counter()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        retrieval_mode=RETRIEVAL_MODE,
    ).to(device)
    # The added scalar consumes no random draws. Persist the complete backbone
    # initialization so independently scheduled controls can explicitly restore it.
    torch.save(
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name != "theta"
        },
        os.path.join(working_dir, f"{dataset_name}_backbone_initial.pt"),
    )
    groups = [
        {
            "params": [p for name, p in model.named_parameters() if name != "theta"],
            "weight_decay": WEIGHT_DECAY,
        }
    ]
    if hasattr(model, "theta"):
        groups.append({"params": [model.theta], "weight_decay": 0.0})
    optimizer = torch.optim.Adam(groups, lr=LR, betas=(0.9, 0.98))
    bce_criterion = nn.BCEWithLogitsLoss()
    updates = valid_positions = examples = 0
    longest_update = 0.0
    stopped_for_time = False
    synchronize()
    training_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        epoch_updates = 0
        for _ in range(NUM_BATCHES):
            synchronize()
            elapsed = time.perf_counter() - training_start
            if updates and elapsed + max(2.0 * longest_update, 2.0) >= TRAIN_SECONDS:
                stopped_for_time = True
                break
            update_start = time.perf_counter()
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
            indices = seq_t.ne(0) & pos_t.ne(0)
            count = int(indices.sum().item())
            if count == 0:
                raise ValueError("Batch has no supervised positions")
            loss = bce_criterion(
                pos_logits[indices], torch.ones_like(pos_logits[indices])
            )
            loss = loss + bce_criterion(
                neg_logits[indices], torch.zeros_like(neg_logits[indices])
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.item())
            updates += 1
            epoch_updates += 1
            examples += seq_t.shape[0]
            valid_positions += count
            synchronize()
            longest_update = max(longest_update, time.perf_counter() - update_start)
        if epoch % 10 == 0 or stopped_for_time or epoch == NUM_EPOCHS:
            print(
                f"  [{dataset_name}] epoch={epoch} updates={updates} "
                f"loss={total_loss / max(epoch_updates, 1):.4f}",
                flush=True,
            )
        if stopped_for_time:
            break

    synchronize()
    training_seconds = time.perf_counter() - training_start
    # Always overwrite stale artifacts with this run's terminal checkpoint.
    torch.save(model.state_dict(), ckpt_path)
    meta = {
        "itemnum": itemnum,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "retrieval_mode": model.retrieval_mode,
        "retrieval_config": model.retrieval_config,
        "selection_checkpoint": "terminal",
        "training_seed": 42,
        "negative_sampler": "uniform with replacement; reject complete training-history items",
        "loss": "mean positive BCE plus mean negative BCE over supervised positions",
    }
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
            "updates": updates,
            "valid_positions": valid_positions,
            "examples": examples,
        },
        os.path.join(working_dir, f"{dataset_name}_training_state.pt"),
    )

    model.eval()
    diagnostic = {}
    if model.retrieval_mode == "retrieval":
        with torch.no_grad():
            h = model._backbone_feats(seq_t)
            parts = model.retrieval_components(seq_t)
            full = (h + parts["scaled_residual"].to(h.dtype)) * seq_t.ne(0).unsqueeze(
                -1
            )
            targets = torch.stack((pos_t, neg_t), dim=-1)
            base_scores = model.predict_score(h, targets, "candidates")
            scores = model.predict_score(full, targets, "candidates")
            margin = scores[..., 0] - scores[..., 1]
            base_margin = base_scores[..., 0] - base_scores[..., 1]
            diagnostic = {
                "valid_positions": int(indices.sum().item()),
                "theta": float(model.theta.item()),
                "amplitude": float(parts["amplitude"].item()),
                "last_theta_gradient": (
                    float(model.theta.grad.item())
                    if model.theta.grad is not None
                    else None
                ),
                "mean_retrieval_mass": float(parts["mass"][indices].mean().item()),
                "zero_mass_fraction": float(
                    parts["mass"][indices].eq(0).float().mean().item()
                ),
                "mean_scaled_residual_norm": float(
                    parts["scaled_residual"].norm(dim=-1)[indices].mean().item()
                ),
                "mean_residual_backbone_ratio": float(
                    (
                        parts["scaled_residual"].norm(dim=-1)
                        / h.norm(dim=-1).clamp_min(1e-8)
                    )[indices]
                    .mean()
                    .item()
                ),
                "backbone_margin_mean": float(base_margin[indices].mean().item()),
                "augmented_margin_mean": float(margin[indices].mean().item()),
                "augmented_margin_p10": float(
                    torch.quantile(margin[indices].float(), 0.1).item()
                ),
                "negative_beats_positive_fraction": float(
                    margin[indices].lt(0).float().mean().item()
                ),
                "margin_below_half_fraction": float(
                    margin[indices].lt(0.5).float().mean().item()
                ),
                "mean_absolute_score_change": float(
                    (scores - base_scores)[indices].abs().mean().item()
                ),
                "definition": "Descriptive terminal sampled training batch, averaged over nonpadding supervised positions; mass is not calibrated confidence.",
            }
    # A single terminal validation report; the host independently evaluates the
    # complete four-category bundle using its unchanged pinned evaluator.
    val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
    synchronize()
    record = {
        "dataset": dataset_name,
        "retrieval_mode": model.retrieval_mode,
        "selection_checkpoint": "terminal",
        "optimizer_updates": updates,
        "examples": examples,
        "valid_positions": valid_positions,
        "effective_epochs": updates / NUM_BATCHES,
        "charged_training_seconds": training_seconds,
        "charged_category_occupied_seconds": time.perf_counter() - occupied_start,
        "training_cap_seconds": TRAIN_SECONDS,
        "cap_overshoot_seconds": max(training_seconds - TRAIN_SECONDS, 0.0),
        "local_terminal_validation": val_m,
        "training_diagnostics": diagnostic,
    }
    with open(os.path.join(working_dir, f"{dataset_name}_run.json"), "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record), flush=True)
    return val_m["HR@10"]


### <<< Self_EvolveRec-BLOCK-END


# ── Write unified predict.py ──────────────────────────────────────────


def write_predict_script(working_dir: str) -> None:
    # DEBUG: No traceback was supplied. Remove an inference integration hazard:
    # importing train.py requires training-only environment variables and resets
    # global RNGs. Export the exact trained architecture beside the predictor.
    import inspect

    os.makedirs(working_dir, exist_ok=True)
    architecture_path = os.path.join(working_dir, "_sasrec_architecture.py")
    architecture_source = (
        "from __future__ import annotations\n"
        "import math\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        + inspect.getsource(PointWiseFeedForward)
        + "\n\n"
        + inspect.getsource(SASRec)
        + "\n"
    )
    with open(architecture_path, "w", encoding="utf-8") as f:
        f.write(architecture_source)

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
    # DEBUG: Load the exported architecture without executing training setup.
    architecture_path = os.path.join(_HERE, "_sasrec_architecture.py")
    spec = importlib.util.spec_from_file_location(
        f"perdataset_architecture_{os.getpid()}", architecture_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load exported architecture: {architecture_path}")
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
        ### >>> Self_EvolveRec-BLOCK-START: Require complete dataset artifacts during inference
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
            ### >>> Self_EvolveRec-BLOCK-START: Restore the identical trained retrieval architecture
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            retrieval_mode=meta["retrieval_mode"],
            retrieval_config=meta["retrieval_config"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Preserve explicit category routing and reject missing models
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown or unloaded dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        raise RuntimeError("No trained dataset models loaded")
    if len(candidates) == 0:
        return []
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Encode once and chunk full-catalog candidate scoring
    with torch.inference_mode():
        features = model.final_features(seq_t)
        scores = []
        for start in range(0, len(candidates), 4096):
            cand_t = torch.tensor(
                [candidates[start:start + 4096]], dtype=torch.long, device=_device
            )
            scores.extend(
                model.predict_score(features, cand_t, "candidates")[0].cpu().tolist()
            )
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


