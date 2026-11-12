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
### >>> Self_EvolveRec-BLOCK-START: Match the host training default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Explicit dataset location and preregistered residual controls
DATA_DIR = os.environ["GAGC_DATA_DIR"]
# One variant per host trial; no automatic sweep or validation-driven expansion.
# baseline: h; recent: r; fixed: h + 0.1*b;
# scalar: h + tanh(alpha)*b; conditional: h + tanh(g(x))*b.
RECENT_MODE = os.environ.get("SASREC_RECENT_MODE", "scalar")
RECENT_WINDOW = 5
RECENT_FIXED_SCALE = 0.1
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


### >>> Self_EvolveRec-BLOCK-START: Reuse eligible training users without changing sampling
def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, eligible_users=None
):
    users = eligible_users
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("Training requires a user with at least two interactions")
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
    ### >>> Self_EvolveRec-BLOCK-START: Serializable recent-context architecture
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_mode: str = "baseline",
        recent_window: int = 5,
        recent_fixed_scale: float = 0.1,
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

        ### >>> Self_EvolveRec-BLOCK-START: Zero-initialized residual gate and efficient recent-only control
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if recent_mode not in {"baseline", "recent", "fixed", "scalar", "conditional"}:
            raise ValueError(f"Unknown recent_mode: {recent_mode}")
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        self.recent_mode = recent_mode
        self.recent_window = int(recent_window)
        self.recent_fixed_scale = float(recent_fixed_scale)
        # All original modules have already been initialized. Preserve the RNG
        # stream so auxiliary initialization does not change training dropout.
        with torch.random.fork_rng(devices=[]):
            if recent_mode in {"fixed", "scalar", "conditional"}:
                self.branch_norm = nn.LayerNorm(hidden_units, eps=1e-8)
                self.branch_projection = nn.Linear(2 * hidden_units + 1, hidden_units)
                nn.init.normal_(
                    self.branch_projection.weight,
                    mean=0.0,
                    std=(2 * hidden_units + 1) ** -0.5,
                )
                nn.init.zeros_(self.branch_projection.bias)
                if recent_mode == "scalar":
                    self.alpha = nn.Parameter(torch.zeros(()))
                elif recent_mode == "conditional":
                    self.context_gate = nn.Linear(2 * hidden_units + 1, 1)
                    nn.init.zeros_(self.context_gate.weight)
                    nn.init.zeros_(self.context_gate.bias)

    def _recent_mean(self, seq: torch.Tensor):
        valid = (seq != 0).unsqueeze(-1)
        embeddings = self.item_emb(seq) * valid
        sums = torch.cat(
            (
                embeddings.new_zeros(seq.size(0), 1, self.hidden_units),
                embeddings.cumsum(dim=1),
            ),
            dim=1,
        )
        counts = torch.cat(
            (
                embeddings.new_zeros(seq.size(0), 1, 1),
                valid.to(embeddings.dtype).cumsum(dim=1),
            ),
            dim=1,
        )
        end = torch.arange(1, seq.size(1) + 1, device=seq.device)
        start = (end - self.recent_window).clamp_min(0)
        count = counts[:, end] - counts[:, start]
        recent = (sums[:, end] - sums[:, start]) / count.clamp_min(1.0)
        return recent, count

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        if self.recent_mode == "recent":
            return self._recent_mean(seq)[0]
        h = self._base_log2feats(seq)
        if self.recent_mode == "baseline":
            return h
        recent, count = self._recent_mean(seq)
        context = torch.cat(
            (self.branch_norm(h), recent, count / float(self.recent_window)), dim=-1
        )
        branch = self.branch_projection(context)
        if self.recent_mode == "scalar":
            gate = torch.tanh(self.alpha)
        elif self.recent_mode == "conditional":
            gate = torch.tanh(self.context_gate(context))
        else:
            gate = self.recent_fixed_scale
        # Deliberately no normalization after residual addition.
        return h + gate * branch * (seq != 0).unsqueeze(-1)

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

        return self.last_layernorm(seq_emb)

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
    ### >>> Self_EvolveRec-BLOCK-START: Fixed validation candidates without perturbing training RNG
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
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic unique validation negatives
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError("Validation requires 99 unique unseen negatives")
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
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### >>> Self_EvolveRec-BLOCK-START: Validate and cache eligible users once per dataset
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    eligible_users = [u for u, items in user_train.items() if len(items) >= 2]
    if not eligible_users:
        raise ValueError(f"No eligible training users for {dataset_name}")
    if any(len(set(user_train[u])) >= itemnum for u in eligible_users):
        raise ValueError(f"A training user has no unseen negative in {dataset_name}")
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the selected branch and verify initial equivalence
        maxlen=MAXLEN,
        recent_mode=RECENT_MODE,
        recent_window=RECENT_WINDOW,
        recent_fixed_scale=RECENT_FIXED_SCALE,
    ).to(device)

    architecture = {
        "itemnum": itemnum,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "recent_mode": RECENT_MODE,
        "recent_window": RECENT_WINDOW,
        "recent_fixed_scale": RECENT_FIXED_SCALE,
    }
    if RECENT_MODE in {"baseline", "scalar", "conditional"}:
        # Construct the unchanged architecture and copy every original parameter.
        # Preserve random state so this assertion does not affect training.
        with torch.random.fork_rng(devices=[]):
            reference = SASRec(
                itemnum, HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS, DROPOUT_RATE, MAXLEN
            ).to(device)
        reference.load_state_dict(
            {k: model.state_dict()[k] for k in reference.state_dict()}
        )
        reference.eval()
        model.eval()
        histories = [items for items in user_train.values() if items][:2]
        if not histories:
            raise ValueError(f"No training histories for {dataset_name}")
        probe = torch.zeros(len(histories), MAXLEN, dtype=torch.long, device=device)
        for row, items in enumerate(histories):
            tail = items[-MAXLEN:]
            probe[row, -len(tail) :] = torch.tensor(tail, device=device)
        candidates = torch.arange(1, min(itemnum, 100) + 1, device=device)
        candidates = candidates.unsqueeze(0).expand(len(histories), -1)
        with torch.no_grad():
            torch.testing.assert_close(
                model.log2feats(probe), reference.log2feats(probe), rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                model.predict(probe, candidates),
                reference.predict(probe, candidates),
                rtol=0.0,
                atol=0.0,
            )
        del reference
        model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated checkpoint even when HR is zero
    best_hr10 = -1.0
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Invoke cached-user sampling in every training batch
            seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                eligible_users=eligible_users,
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
                ### >>> Self_EvolveRec-BLOCK-START: Store architecture and all trained residual parameters
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "architecture": architecture,
                        "epoch": epoch,
                        "validation_hr10": hr10,
                    },
                    ckpt_path,
                )
                ### <<< Self_EvolveRec-BLOCK-END
            if epoch - best_epoch >= PATIENCE * 5:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    ### >>> Self_EvolveRec-BLOCK-START: Evaluate short runs and avoid reusing stale checkpoints
    if best_epoch == 0:
        best_hr10 = evaluate_val(
            model, user_train, user_valid, itemnum, device, MAXLEN
        )["HR@10"]
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": architecture,
                "epoch": NUM_EPOCHS,
                "validation_hr10": best_hr10,
            },
            ckpt_path,
        )
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        ### >>> Self_EvolveRec-BLOCK-START: Persist predictor architecture configuration
        pickle.dump(architecture, f)
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
            ### >>> Self_EvolveRec-BLOCK-START: Reproduce the trained residual scoring path
            maxlen=meta["maxlen"],
            recent_mode=meta.get("recent_mode", "baseline"),
            recent_window=meta.get("recent_window", 5),
            recent_fixed_scale=meta.get("recent_fixed_scale", 0.1),
        ).to(_device)
        checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=True)
        if "model_state_dict" in checkpoint:
            if checkpoint["architecture"] != meta:
                raise ValueError(f"Checkpoint architecture mismatch for {ds}")
            checkpoint = checkpoint["model_state_dict"]
        m.load_state_dict(checkpoint)
        ### <<< Self_EvolveRec-BLOCK-END
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject missing dataset routes instead of scoring with another catalog
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model available for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        raise RuntimeError("No trained dataset checkpoints were loaded")
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


