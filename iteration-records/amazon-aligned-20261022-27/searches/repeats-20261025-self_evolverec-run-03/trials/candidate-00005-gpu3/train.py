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

### >>> Self_EvolveRec-BLOCK-START: Explicit fusion and loss ablations
DATA_DIR = os.environ["GAGC_DATA_DIR"]
VARIANT = os.environ.get("SASREC_VARIANT", "combined")
if VARIANT not in {"baseline", "loss_only", "residual_only", "combined"}:
    raise ValueError(f"Unknown SASREC_VARIANT: {VARIANT}")
USE_SOFTMAX = VARIANT in {"loss_only", "combined"}
USE_RESIDUAL = VARIANT in {"residual_only", "combined"}
NEGATIVE_COUNT = 16 if USE_SOFTMAX else 1
RECENT_WINDOW = 5
TEMPERATURE = 1.0
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four dataset routes
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


### >>> Self_EvolveRec-BLOCK-START: Exact uniform history-excluding sampling with replacement
class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, history in user_train.items() if len(history) >= 2]
        if not self.users:
            raise ValueError("Training requires a user with at least two interactions")
        self.exclusions = {}
        self.eligible_counts = {}
        for u in self.users:
            forbidden = np.unique(np.asarray(user_train[u], dtype=np.int64))
            if np.any((forbidden < 1) | (forbidden > itemnum)):
                raise ValueError(
                    f"Training history contains an invalid item ID: user={u}"
                )
            self.eligible_counts[u] = itemnum - len(forbidden)
            # Maps uniform complement ranks to catalog IDs without rejection loops.
            self.exclusions[u] = forbidden - np.arange(len(forbidden))
        if not any(self.eligible_counts.values()):
            raise ValueError("No training position has an eligible negative item")

    def sample(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, NEGATIVE_COUNT), dtype=np.int64)
        for row in range(batch_size):
            u = random.choice(self.users)
            history = self.user_train[u]
            length = min(len(history) - 1, maxlen)
            seq[row, -length:] = history[-length - 1 : -1]
            eligible = self.eligible_counts[u]
            if eligible == 0:
                continue
            pos[row, -length:] = history[-length:]
            ranks = np.random.randint(1, eligible + 1, size=(length, NEGATIVE_COUNT))
            neg[row, -length:] = ranks + np.searchsorted(
                self.exclusions[u], ranks, side="right"
            )
        return seq, pos, neg if USE_SOFTMAX else neg[..., 0]


def sample_batch(user_train, itemnum, batch_size, maxlen, sampler=None):
    if sampler is None:
        sampler = TrainingSampler(user_train, itemnum)
    return sampler.sample(batch_size, maxlen)


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
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        ### >>> Self_EvolveRec-BLOCK-START: Explicit checkpointed residual configuration
        maxlen: int = 128,
        use_residual: bool = False,
        recent_window: int = 5,
    ):
        super().__init__()
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum
        self.use_residual = use_residual
        self.recent_window = recent_window
        ### <<< Self_EvolveRec-BLOCK-END

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

        ### >>> Self_EvolveRec-BLOCK-START: Initially disabled gated recent-window residual
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if self.use_residual:
            # Preserve the baseline RNG stream when adding the optional branch.
            with torch.random.fork_rng(devices=[]):
                self.recent_projection = nn.Linear(hidden_units, hidden_units)
                self.recent_norm = nn.LayerNorm(hidden_units, eps=1e-8)
                self.recent_gate = nn.Linear(2 * hidden_units + 1, 1)
                nn.init.zeros_(self.recent_gate.weight)
                nn.init.zeros_(self.recent_gate.bias)
            self.residual_alpha = nn.Parameter(torch.zeros(()))
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

        ### >>> Self_EvolveRec-BLOCK-START: Causal masked rolling mean shared by training and inference
        features = self.last_layernorm(seq_emb)
        if not self.use_residual:
            return features
        mask = (~timeline_mask).to(features.dtype).unsqueeze(-1)
        embeddings = self.item_emb(seq) * mask
        sums = torch.cat(
            (torch.zeros_like(embeddings[:, :1]), embeddings.cumsum(1)), dim=1
        )
        counts = torch.cat((torch.zeros_like(mask[:, :1]), mask.cumsum(1)), dim=1)
        end = torch.arange(1, M + 1, device=seq.device)
        start = (end - self.recent_window).clamp_min(0)
        recent = (sums[:, end] - sums[:, start]) / (
            counts[:, end] - counts[:, start]
        ).clamp_min(1.0)
        prefix_length = mask.cumsum(1).log1p()
        gate = torch.sigmoid(
            self.recent_gate(torch.cat((features, recent, prefix_length), dim=-1))
        )
        residual = self.recent_norm(self.recent_projection(recent))
        return features + self.residual_alpha * gate * residual * mask
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
        ### >>> Self_EvolveRec-BLOCK-START: Support both diagnostic and multi-negative tensor shapes
        if neg.ndim == seq.ndim + 1:
            neg_logits = (log_feats.unsqueeze(-2) * neg_embs).sum(dim=-1)
        else:
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
        ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Stable validation candidates without perturbing training RNG
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    was_training = model.training
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound validation sampling and retain unique negatives
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError(
                    f"User {u} has fewer than 99 eligible validation negatives"
                )
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

    ### >>> Self_EvolveRec-BLOCK-START: Fail clearly on missing validation and restore model mode
    if not val_users:
        raise ValueError("No eligible validation users")
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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible per-category training and cached sampler
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    sampler = TrainingSampler(user_train, itemnum)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the selected fusion ablation
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        use_residual=USE_RESIDUAL,
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

    ### >>> Self_EvolveRec-BLOCK-START: Always save the first evaluated checkpoint and record selection
    best_hr10 = float("-inf")
    best_epoch = 0
    trial_log = []
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Uniform 16-negative candidate softmax on valid positions
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, sampler=sampler
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            indices = pos_t != 0
            if not indices.any():
                continue
            # Embed candidates only at valid positions to avoid padding overhead.
            features = model.log2feats(seq_t)[indices]
            pos_logits = (features * model.item_emb(pos_t[indices])).sum(-1)
            if USE_SOFTMAX:
                neg_logits = torch.einsum(
                    "nh,nkh->nk", features, model.item_emb(neg_t[indices])
                )
                logits = torch.cat((pos_logits.unsqueeze(-1), neg_logits), dim=-1)
                loss = nn.functional.cross_entropy(
                    logits / TEMPERATURE,
                    torch.zeros(logits.shape[0], dtype=torch.long, device=device),
                )
            else:
                neg_logits = (features * model.item_emb(neg_t[indices])).sum(-1)
                loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
                loss = loss + bce_criterion(neg_logits, torch.zeros_like(neg_logits))
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite loss for {dataset_name}, epoch {epoch}"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Validate the final epoch and persist observed validation history
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            trial_log.append({"epoch": epoch, "loss": avg_loss, **val_m})
            with open(
                os.path.join(working_dir, f"{dataset_name}_training.json"), "w"
            ) as f:
                json.dump(trial_log, f, indent=2)
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
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Save architecture, sampler, optimizer and checkpoint policy
                "maxlen": MAXLEN,
                "use_residual": USE_RESIDUAL,
                "recent_window": RECENT_WINDOW,
                "variant": VARIANT,
                "negative_count": NEGATIVE_COUNT,
                "temperature": TEMPERATURE,
                "sampler": "uniform_with_replacement_excluding_full_training_history",
                "training_seed": 42,
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "betas": (0.9, 0.98),
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "num_epochs": NUM_EPOCHS,
                "patience_checks": PATIENCE,
                "checkpoint_rule": "maximum validation HR@10; earliest epoch on ties",
                "best_epoch": best_epoch,
                "best_hr10": best_hr10,
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
        ### >>> Self_EvolveRec-BLOCK-START: Report missing artifacts explicitly
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model artifacts for {ds}: {meta_path}, {ckpt_path}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct every trained residual parameter
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            use_residual=meta["use_residual"],
            recent_window=meta["recent_window"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"Unknown dataset route: {dataset_name}")
    ### <<< Self_EvolveRec-BLOCK-END
    # Select model: use dataset_name kwarg if provided, else fall back to first available
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
    elif _models:
        ds    = next(iter(_models))
        model = _models[ds]
        meta  = _metas[ds]
    ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when inference artifacts are unavailable
    else:
        raise RuntimeError("No dataset models loaded")
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

    ### >>> Self_EvolveRec-BLOCK-START: Verify saved predictor and training-view score agreement
    write_predict_script(working_dir)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "saved_predictor_check", os.path.join(working_dir, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor._ensure_loaded()
    for ds in DATASETS:
        meta = predictor._metas[ds]
        reference = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            use_residual=meta["use_residual"],
            recent_window=meta["recent_window"],
        ).to(device)
        reference.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{ds}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        reference.eval()
        histories = _read_split(os.path.join(DATA_DIR, f"{ds}_train.txt"))
        for u in sorted(histories)[:5]:
            history = histories[u][-meta["maxlen"] :]
            candidates = list(range(1, min(meta["itemnum"], 100) + 1))
            seq = torch.tensor(
                [[0] * (meta["maxlen"] - len(history)) + history],
                dtype=torch.long,
                device=device,
            )
            items = torch.tensor([candidates], dtype=torch.long, device=device)
            with torch.no_grad():
                features = reference.log2feats(seq)[:, -1]
                expected = torch.einsum(
                    "bh,bch->bc", features, reference.item_emb(items)
                )[0].cpu()
                actual = torch.tensor(
                    predictor.predict(u, history, candidates, dataset_name=ds),
                    dtype=expected.dtype,
                )
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        del reference
    print("Saved predictor scores agree with training-view fused features", flush=True)

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


