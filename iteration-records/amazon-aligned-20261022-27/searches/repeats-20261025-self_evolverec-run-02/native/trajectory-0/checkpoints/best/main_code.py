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
### >>> Self_EvolveRec-BLOCK-START: Respect the shared training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Fixed directional residual configurations and required data root
DATA_DIR = os.environ["GAGC_DATA_DIR"]

# One configuration per host trial; selection remains at the host boundary.
RESIDUAL_CONFIGS = {
    "baseline": None,
    "recent5_context": {
        "version": 1,
        "dim": 32,
        "K": 5,
        "decay": 0.7,
        "alpha": 0.1,
        "gate": "context",
        "norm_eps": 1e-5,
    },
    "recent5_one": {
        "version": 1,
        "dim": 32,
        "K": 5,
        "decay": 0.7,
        "alpha": 0.1,
        "gate": "one",
        "norm_eps": 1e-5,
    },
    "recent1_context": {
        "version": 1,
        "dim": 32,
        "K": 1,
        "decay": 0.7,
        "alpha": 0.1,
        "gate": "context",
        "norm_eps": 1e-5,
    },
}
RESIDUAL_NAME = os.environ.get("SASREC_RESIDUAL", "recent5_context")
if RESIDUAL_NAME not in RESIDUAL_CONFIGS:
    raise ValueError(f"Unknown SASREC_RESIDUAL: {RESIDUAL_NAME}")
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four benchmark datasets
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
    users = [u for u, items in user_train.items() if len(items) >= 2]
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
    ### >>> Self_EvolveRec-BLOCK-START: Serializable optional directional residual
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        residual_config: dict | None = None,
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

        ### >>> Self_EvolveRec-BLOCK-START: Nonzero asymmetric projections without consuming baseline RNG
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.residual_config = (
            None if residual_config is None else dict(residual_config)
        )
        if self.residual_config is not None:
            cfg = self.residual_config
            if cfg["version"] != 1 or cfg["gate"] not in ("context", "one"):
                raise ValueError("Unsupported directional residual configuration")
            if cfg["K"] < 1 or cfg["dim"] < 1 or not 0.0 < cfg["decay"] <= 1.0:
                raise ValueError("Invalid directional pooling configuration")
            # Construction occurs on CPU, before model.to(device).
            with torch.random.fork_rng(devices=[]):
                self.source_projection = nn.Linear(hidden_units, cfg["dim"], bias=False)
                self.target_projection = nn.Linear(hidden_units, cfg["dim"], bias=False)
                nn.init.xavier_uniform_(self.source_projection.weight)
                nn.init.xavier_uniform_(self.target_projection.weight)
                self.source_norm = nn.LayerNorm(
                    cfg["dim"], eps=cfg["norm_eps"], elementwise_affine=False
                )
                self.target_norm = nn.LayerNorm(
                    cfg["dim"], eps=cfg["norm_eps"], elementwise_affine=False
                )
                if cfg["gate"] == "context":
                    self.context_gate = nn.Linear(hidden_units, 1)
                    nn.init.zeros_(self.context_gate.weight)
                    nn.init.zeros_(self.context_gate.bias)

    def _encoder_features(self, seq: torch.Tensor) -> torch.Tensor:
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

    ### >>> Self_EvolveRec-BLOCK-START: Causal directional features and shape checked combined scoring
    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        h = self._encoder_features(seq)
        if self.residual_config is None:
            return h
        cfg = self.residual_config
        valid = seq.ne(0)
        a = self.source_norm(self.source_projection(self.item_emb(seq)))
        a = a * valid.unsqueeze(-1)
        pooled = torch.zeros_like(a)
        denom = torch.zeros_like(seq, dtype=a.dtype)
        length = seq.shape[1]
        for lag in range(min(cfg["K"], length)):
            weight = cfg["decay"] ** lag
            # Slice assignment preserves autograd and never reads a future position.
            pooled[:, lag:, :] = pooled[:, lag:, :] + weight * a[:, : length - lag, :]
            denom[:, lag:] = denom[:, lag:] + weight * valid[:, : length - lag].to(
                a.dtype
            )
        pooled = pooled / denom.clamp_min(1e-12).unsqueeze(-1)
        pooled = pooled * valid.unsqueeze(-1)
        gate = torch.sigmoid(self.context_gate(h)) if cfg["gate"] == "context" else 1.0
        branch = cfg["alpha"] * gate * pooled / math.sqrt(cfg["dim"])
        # In particular, an entirely empty history has exactly zero features.
        h = h * valid.unsqueeze(-1)
        return torch.cat((h, branch), dim=-1)

    def candidate_features(self, items: torch.Tensor) -> torch.Tensor:
        e = self.item_emb(items)
        if self.residual_config is None:
            return e
        valid = items.ne(0).unsqueeze(-1)
        b = self.target_norm(self.target_projection(e)) * valid
        return torch.cat((e * valid, b), dim=-1)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        candidates = self.candidate_features(items)
        if features.ndim == 3:
            # Aligned B x L positive or negative training targets.
            if items.ndim != 2 or features.shape[:2] != items.shape:
                raise ValueError("Aligned targets must match feature batch and length")
            return (features * candidates).sum(dim=-1)
        if features.ndim == 2:
            if items.ndim == 1:
                # A shared candidate list, including full-catalog diagnostics.
                return features @ candidates.transpose(0, 1)
            if items.ndim == 2 and items.shape[0] == features.shape[0]:
                return torch.bmm(candidates, features.unsqueeze(-1)).squeeze(-1)
        raise ValueError("Expected B x D features with C or B x C candidates")

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        features = self.log2feats(seq)
        if pos is None or neg is None:
            return features
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.predict_score(self.log2feats(seq)[:, -1, :], candidates)

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
    ### >>> Self_EvolveRec-BLOCK-START: Keep routine validation independent of training RNG
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
                ### >>> Self_EvolveRec-BLOCK-START: Local validation candidate RNG
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

    ### >>> Self_EvolveRec-BLOCK-START: Reproducible per-category initialization and unregularized residual
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        residual_config=RESIDUAL_CONFIGS[RESIDUAL_NAME],
    ).to(device)

    if model.residual_config is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
        )
    else:
        branch_prefixes = ("source_projection.", "target_projection.", "context_gate.")
        base_parameters, branch_parameters = [], []
        for name, parameter in model.named_parameters():
            destination = (
                branch_parameters
                if name.startswith(branch_prefixes)
                else base_parameters
            )
            destination.append(parameter)
        optimizer = torch.optim.Adam(
            [
                {"params": base_parameters, "weight_decay": WEIGHT_DECAY},
                {"params": branch_parameters, "weight_decay": 0.0},
            ],
            lr=LR,
            betas=(0.9, 0.98),
        )
    print(f"  [{dataset_name}] configuration={RESIDUAL_NAME}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
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
            ### >>> Self_EvolveRec-BLOCK-START: Clip residual configurations only
            if model.residual_config is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            ### <<< Self_EvolveRec-BLOCK-END
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate and save only the final common epoch
        if epoch == NUM_EPOCHS:
            # Save unconditionally so an older trial checkpoint cannot be reused.
            torch.save(model.state_dict(), ckpt_path)
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            best_hr10 = val_m["HR@10"]
            best_epoch = epoch
            print(
                f"  [{dataset_name}] final epoch={epoch}"
                f"  loss={avg_loss:.4f}  routine_val_HR10={best_hr10:.6f}"
                "  configuration selection uses the independent host evaluator",
                flush=True,
            )
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}",
                flush=True,
            )
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Check checkpoint scoring and causal residual invariants on public histories
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    saved_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(saved_state)
    model.eval()
    history = next((items for items in user_train.values() if items), [])[-MAXLEN:]
    sequence = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
    if history:
        sequence[0, -len(history) :] = torch.tensor(history, device=device)
    candidates = torch.arange(1, min(itemnum, 16) + 1, device=device).unsqueeze(0)
    # Preserve training random streams while constructing a second model.
    with torch.random.fork_rng(devices=[]):
        reloaded = SASRec(
            itemnum,
            HIDDEN_UNITS,
            NUM_BLOCKS,
            NUM_HEADS,
            DROPOUT_RATE,
            MAXLEN,
            residual_config=model.residual_config,
        ).to(device)
    reloaded.load_state_dict(saved_state)
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model.predict(sequence, candidates), reloaded.predict(sequence, candidates)
        )
        if model.residual_config is not None:
            features = model.log2feats(sequence)
            assert torch.isfinite(features).all()
            torch.testing.assert_close(
                features[sequence.eq(0)], torch.zeros_like(features[sequence.eq(0)])
            )
            empty = torch.zeros_like(sequence)
            torch.testing.assert_close(
                model.predict(empty, candidates),
                torch.zeros_like(model.predict(empty, candidates)),
            )
            # Padding must remain inert even if its stored embedding is nonzero.
            padding = model.item_emb.weight[0].clone()
            try:
                model.item_emb.weight[0].fill_(13.0)
                torch.testing.assert_close(features, model.log2feats(sequence))
            finally:
                model.item_emb.weight[0].copy_(padding)
            if MAXLEN > 1 and itemnum > 0:
                split = max(1, MAXLEN // 2)
                changed = sequence.clone()
                changed[:, split:] = 1
                torch.testing.assert_close(
                    features[:, :split], model.log2feats(changed)[:, :split]
                )
    del reloaded, saved_state

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Persist the complete inference architecture
                "maxlen": MAXLEN,
                "architecture": "sasrec_directional_v1",
                "residual_name": RESIDUAL_NAME,
                "residual_config": model.residual_config,
                "dataset_name": dataset_name,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    print(
        ### >>> Self_EvolveRec-BLOCK-START: Label final checkpoint validation accurately
        f"  [{dataset_name}] Final routine val HR@10={best_hr10:.4f}"
        f"  epoch={best_epoch}  ckpt={ckpt_path}",
        ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Report missing category artifacts explicitly
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing checkpoint or metadata for {ds}: {ckpt_path}, {meta_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the trained directional scorer from checkpoint metadata
        if meta.get("architecture", "sasrec_directional_v1") != "sasrec_directional_v1":
            raise ValueError("Unsupported checkpoint architecture")
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            residual_config=meta.get("residual_config"),
        ).to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes while retaining legacy inference
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")
        ds = dataset_name
    else:
        ds = DATASETS[0]
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


