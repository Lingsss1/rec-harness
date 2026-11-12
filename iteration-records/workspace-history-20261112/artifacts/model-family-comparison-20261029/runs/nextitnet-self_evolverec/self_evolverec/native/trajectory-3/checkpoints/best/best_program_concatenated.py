# === main_code.py ===
"""Per-dataset NextItNet baseline for Amazon Reviews.

Trains one independent NextItNet checkpoint per dataset while sharing one default
hyper-parameter profile across datasets. The generated predict.py routes by the
optional dataset_name keyword used by the RecHarness Amazon Reviews harness.
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

### >>> Self_EvolveRec-BLOCK-START: Preserve required data source and training seed
DATA_DIR = os.environ["GAGC_DATA_DIR"]

SEED = 42
### <<< Self_EvolveRec-BLOCK-END
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _read_split(path: str) -> dict[int, list[int]]:
    data: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]["itemnum"])
    itemnum = 0
    for split in ("train", "valid"):
        path = os.path.join(data_dir, f"{dataset_name}_{split}.txt")
        if not os.path.isfile(path):
            continue
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


### >>> Self_EvolveRec-BLOCK-START: Cache eligible training users while retaining diagnostic sampler compatibility
# DEBUG: Restore the missing sampler definition and its local variables; the
# indented placeholder left the sampler body outside any function.
def sample_batch(user_train, itemnum, batch_size, maxlen, users=None):
    if itemnum <= 0 or batch_size <= 0 or maxlen <= 0:
        raise ValueError("itemnum, batch_size and maxlen must be positive")
    if users is None:
        users = [
            user
            for user, history in user_train.items()
            if len(history) >= 2
            and len({item for item in history if 1 <= item <= itemnum}) < itemnum
        ]
    if not users:
        raise ValueError("No eligible training users with available negative items")
    batch_seq, batch_pos, batch_neg = [], [], []
    ### <<< Self_EvolveRec-BLOCK-END
    for _ in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in interacted:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


def _history_to_tensor(
    history: list[int], maxlen: int, device: torch.device
) -> torch.Tensor:
    seq_arr = np.zeros(maxlen, dtype=np.int64)
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    return torch.tensor([seq_arr], dtype=torch.long, device=device)


def evaluate_val(
    model: nn.Module,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Reuse fixed validation candidates without consuming training randomness
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic unique validation negatives with explicit feasibility checks
            candidates = [target]
            available = itemnum - sum(1 for item in interacted if 1 <= item <= itemnum)
            if available < 99:
                raise RuntimeError(
                    f"Validation user {user} has only {available} unseen negatives; 99 required"
                )
            while len(candidates) < 100:
                neg_item = rng.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = _history_to_tensor(history, maxlen, device)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += 1.0 if rank < 10 else 0.0
            hr20 += 1.0 if rank < 20 else 0.0
            ndcg10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            ndcg20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0
    n = float(len(val_users))
    model.train()
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    pos_scores, neg_scores = model(seq, pos, neg)
    ### >>> Self_EvolveRec-BLOCK-START: Stable BPR over valid causal training targets
    valid = (seq != 0) & (pos != 0) & (neg != 0)
    if not valid.any():
        raise RuntimeError("Training batch has no valid next-item targets")
    return nn.functional.softplus(neg_scores[valid] - pos_scores[valid]).mean()
    ### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("NEXTITNET_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("NEXTITNET_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("NEXTITNET_BLOCKS", "8"))
KERNEL_SIZE = int(os.environ.get("NEXTITNET_KERNEL", "3"))
DROPOUT_RATE = float(os.environ.get("NEXTITNET_DROPOUT", "0.2"))
LR = float(os.environ.get("NEXTITNET_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("NEXTITNET_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("NEXTITNET_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Honor the host epoch default within the fixed training ceiling
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1,
        int(os.environ.get("NEXTITNET_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("NEXTITNET_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("NEXTITNET_PATIENCE", "20"))


class NextItNetBlock(nn.Module):
    def __init__(
        self, hidden_units: int, dilation: int, kernel_size: int, dropout_rate: float
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))
        return conv(x)

    ### >>> Self_EvolveRec-BLOCK-START: Prevent padded activations from entering causal convolutions
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x * mask
        y = (self.norm1(residual) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2) * mask
        y = self.dropout(self.activation(y)) * mask
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2) * mask
        y = self.dropout(y) * mask
        return (residual + y) * mask

    ### <<< Self_EvolveRec-BLOCK-END


class NextItNetModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList(
            [
                NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
                for i in range(num_blocks)
            ]
        )
        ### >>> Self_EvolveRec-BLOCK-START: Zero-initialized same-position residual depth correction
        if num_blocks < 2:
            raise ValueError("Residual depth correction requires at least two blocks")
        self.depth_norm_eps = 1e-5
        self.depth_norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    hidden_units,
                    eps=self.depth_norm_eps,
                    elementwise_affine=False,
                )
                for _ in range(num_blocks)
            ]
        )
        # These additions consume no initialization randomness. The original
        # backbone initialization and dropout stream remain undisturbed.
        self.depth_gate_weight = nn.Parameter(torch.zeros(num_blocks, hidden_units))
        self.depth_gate_bias = nn.Parameter(torch.zeros(num_blocks))
        self.depth_scale = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        ### <<< Self_EvolveRec-BLOCK-END
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Causal depth differences with an unchanged direct final-state path
    def depth_components(self, states, sequence):
        if len(states) != len(self.blocks):
            raise ValueError("Backbone and correction depths disagree")
        expected = (*sequence.shape, self.hidden_units)
        if any(tuple(state.shape) != expected for state in states):
            raise ValueError("Depth states must have shape B x L x D")
        mask = sequence.ne(0).unsqueeze(-1).to(states[-1].dtype)
        values = torch.stack(
            [norm(state) * mask for norm, state in zip(self.depth_norms, states)],
            dim=2,
        )
        logits = torch.einsum("blkd,kd->blk", values, self.depth_gate_weight)
        weights = (logits + self.depth_gate_bias).softmax(dim=-1)
        delta = values - values[:, :, -1:, :]
        return states[-1] * mask, delta, weights, mask

    def compose_depth(self, final, delta, weights, mask):
        correction = (weights.unsqueeze(-1) * delta).sum(dim=2)
        # Alpha is unconstrained. At initialization, finite states produce
        # exactly the original final-block features before the final norm.
        return (final + self.depth_scale * correction) * mask

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        valid = seq != 0
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        pos = valid.long().cumsum(dim=1).clamp_max(self.maxlen)
        pos = pos * valid.long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x * mask) * mask
        states = []
        for block in self.blocks:
            x = block(x, mask) * mask
            states.append(x)
        features = self.compose_depth(*self.depth_components(states, seq))
        return self.norm(features) * mask

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    ### <<< Self_EvolveRec-BLOCK-END

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        ### >>> Self_EvolveRec-BLOCK-START: Share trained item scoring with diagnostic compatibility methods
        pos_scores = self.predict_score(out, pos)
        neg_scores = self.predict_score(out, neg)
        ### <<< Self_EvolveRec-BLOCK-END
        return pos_scores, neg_scores

    ### >>> Self_EvolveRec-BLOCK-START: Fix shared-candidate scoring dimensions at the final nonpadding position
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.size(1) == 0:
            raise ValueError("seq must have shape B x L with L > 0")
        if candidates.ndim not in (1, 2):
            raise ValueError("candidates must have shape N or B x N")
        if candidates.ndim == 2 and candidates.size(0) != seq.size(0):
            raise ValueError("Candidate and history batch sizes must match")
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1)
        rows = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[rows, last.clamp_min(0)]
        user_emb = user_emb * (last >= 0).unsqueeze(-1)
        # B x D features produce B x N scores for either shared N candidates
        # or per-history B x N candidates. Adding a singleton feature axis
        # makes the shared-candidate matmul incorrectly return B x 1 x N.
        scores = self.predict_score(user_emb, candidates)
        expected = (seq.size(0), candidates.size(-1))
        if tuple(scores.shape) != expected:
            raise RuntimeError(
                f"Candidate scores must have shape {expected}, got {tuple(scores.shape)}"
            )
        # Empty histories retain zero scores; empty candidate lists remain valid.
        # The generated predictor imports this exact class and readout method.
        return scores[0] if seq.size(0) == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Reset seed 42 independently for each category
def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = NextItNetModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Cache eligible sampler users without changing sampling probabilities
    eligible_users = [
        user
        for user, history in user_train.items()
        if len(history) >= 2
        and len({item for item in history if 1 <= item <= itemnum}) < itemnum
    ]
    if not eligible_users:
        raise RuntimeError(f"No eligible training users for {dataset_name}")
    best_hr10 = -1.0
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Use cached users in every training batch
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, users=eligible_users
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
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
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
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
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "kernel_size": KERNEL_SIZE,
                ### >>> Self_EvolveRec-BLOCK-START: Persist the exact residual depth correction configuration
                "dropout_rate": DROPOUT_RATE,
                "architecture": "nextitnet_zero_residual_depth_v3",
                "prefix_conditioning": False,
                "position_encoding": "observed_prefix_count",
                "depth_value_norm_affine": False,
                "depth_norm_eps": model.depth_norm_eps,
                "depth_gate_mode": "same_position_adaptive",
                "depth_alpha_initial": 0.0,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""Per-dataset NextItNet predict function."""
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
_models = {}
_metas = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"nextitnet_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NextItNetModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Require the trained checkpoint for every routed category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing NextItNet metadata or checkpoint for {ds}: "
                f"{meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        ### >>> Self_EvolveRec-BLOCK-START: Validate the deployed single-model residual depth architecture
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if (
            meta.get("architecture") != "nextitnet_zero_residual_depth_v3"
            or meta.get("prefix_conditioning") is not False
            or meta.get("position_encoding") != "observed_prefix_count"
            or meta.get("depth_value_norm_affine") is not False
            or meta.get("depth_norm_eps") != 1e-5
            or meta.get("depth_gate_mode") != "same_position_adaptive"
            or meta.get("num_blocks", 0) < 2
        ):
            raise RuntimeError(f"Incompatible NextItNet architecture for {ds}")
        # The imported training class reconstructs the same correction;
        # strict state loading restores its gates, alpha and backbone.
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject ambiguous routing instead of scoring with another category
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple categories are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"No trained NextItNet model for {dataset_name!r}")
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = int(meta.get("maxlen", 128))
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written -> {predict_path}", flush=True)


def main() -> None:
    ### >>> Self_EvolveRec-BLOCK-START: Require four-category routing and record the architecture experiment before fitting
    if DATASETS != DEFAULT_DATASETS:
        raise ValueError("All four required datasets must retain their supplied order")
    if EVAL_EVERY < 1 or PATIENCE < 1:
        raise ValueError("Evaluation interval and patience must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset NextItNet training on {device}", flush=True)
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    manifest = {
        "family": "nextitnet",
        "architecture": "nextitnet_zero_residual_depth_v3",
        "datasets": DEFAULT_DATASETS,
        "training_seed": SEED,
        "preprocessing": "host-supplied fixed splits; no preprocessing performed",
        "configuration": "same_position_adaptive",
        "prefix_conditioning": False,
        "depth_norm_eps": 1e-5,
        "depth_norm_affine": False,
        "alpha_initial": 0.0,
        "alpha_constraint": "unconstrained",
        "maxlen": MAXLEN,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "dilations": ([1, 2, 4, 8] * math.ceil(NUM_BLOCKS / 4))[:NUM_BLOCKS],
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "BPR softplus over valid targets",
        "maximum_epochs": NUM_EPOCHS,
        "batches_per_epoch": NUM_BATCHES,
        "checkpoint_epochs": [
            epoch
            for epoch in range(1, NUM_EPOCHS + 1)
            if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS
        ],
        "stopping_patience_epochs": PATIENCE,
        "checkpoint_tie_break": "earliest epoch",
        "selection_metric": "four-category macro validation HR@10",
        "models_per_category": 1,
        "comparison_scope": "one configuration per host trial; no internal sweep",
        "shared_remaining_gpu_seconds": None,
        "budget_accounting": "host-managed; remaining allowance unavailable to this script",
        "unexecuted_controls": ["final_only", "static_depth_weights"],
    }
    manifest_path = os.path.join(working_dir, "experiment_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            if json.load(f) != manifest:
                raise RuntimeError("Experiment manifest differs; use a fresh trial")
    else:
        with open(manifest_path, "x") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    scores = []
    ### <<< Self_EvolveRec-BLOCK-END
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
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
    ### >>> Self_EvolveRec-BLOCK-START: Evidence-limited temporal diagnosis interpretation
    - Distinguish position-weighted training separation, training-user score sensitivity,
      in-sample training-prefix ranking audits, and held-out validation ranking.
      Held-out ranking is unavailable through this adapter.
    - Under nested negative sets, target rank cannot improve and hit-at-10 cannot
      increase; a strict decline is not inevitable and does not establish poor
      generalization. Exact hypergeometric expectations describe
      uniform sampling conditional on the fitted scores and declared tie order.
    - Candidate repetitions are not independent users. Paired bootstrap intervals
      condition on repetition zero; repetition SD measures candidate variability,
      and SD/sqrt(10) is only Monte Carlo error with users and models fixed.
    - Report depth eligibility, common-user counts, strata and exclusions. Raw
      tenth-negative margins depend on score scale. Training-prefix targets may
      have been optimized during training; do not infer validation gains, current
      intent, causality, memorization or exposure bias from these diagnostics.
    - This category report cannot establish a four-category macro result. Combine
      categories equally only when all four category reports are available.
    - Cite eligible counts, controls, missingness, and uncertainty for every finding.
    - A score change is not evidence of relevance, improved HR, user intent, simulator
      traits, or feedback effects. Angular alignment and covariance rank do not establish
      useful ranking. The margin threshold 0.5 is scale dependent.
    - Stale-content hypotheses require corroboration across deletion, geometry controls,
      zero replacement, and frequency-matched nonzero replacement. Otherwise label them
      exploratory. Interior zeros and replacement items introduce distribution shifts.
    - Bootstrap intervals condition on fitted models, users, and fixed candidates;
      they exclude training-seed, candidate-sampling, and adaptive-search uncertainty.
      Do not infer p-values or confirmatory significance from these intervals.
    - Missing categories, validation interventions, or unavailable metadata are unavailable
      evidence. Do not invent measurements or recommend ensembles or fitted score fusion.
    - BPR score derivatives are deterministic functions of margins. Their magnitude
      supplies no independent evidence of sampling failure or prospective added value.
      Separate raw derivatives from coefficients divided by the candidate count.
    - Distinguish small derivatives from positive/negative feature-contribution cancellation.
      Local feature gradients are coordinate dependent and omit the encoder Jacobian,
      tied input paths and optimizer state. They do not measure backbone conflict,
      clipping, Adam updates, learning progress, or validation improvement.
    - Use only verified scorer/loss panels for derivative findings. Report failed checks,
      finite counts, pool exclusions, user counts and tiny strata. Null is unavailable.
      Endpoint expectations do not reproduce the complete position-weighted training loss.
    - Do not infer ownership, compatibility, latent intent or proven saturation mechanisms.
      This audit alone cannot justify changing training or choosing a sampler.
    ### <<< Self_EvolveRec-BLOCK-END
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the family-specific ModelView boundary
    def __init__(self, view, device):
        if not isinstance(view.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.view = view
        self.model = view
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = view.user_train

    ### <<< Self_EvolveRec-BLOCK-END

    def _get_categories(self, item_id: int):
        key = str(int(item_id))
        info = self.meta_data.get(key, None)
        if not isinstance(info, dict):
            return []
        cats = info.get("categories", [])
        if isinstance(cats, list):
            return [c for c in cats if isinstance(c, str)]
        return []

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            n = emb.size(0)
            if n <= 2:
                return 0.0

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    ### >>> Self_EvolveRec-BLOCK-START: Restore the public valid-position margin probe
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self, dataloader, num_batches=10, eps=0.5, topk_categories=20
    ):
        from itertools import islice

        network = self.view.model
        modes = [(module, module.training) for module in network.modules()]
        margins, positions, nonfinite, batches = [], 0, 0, 0
        try:
            network.eval()
            for batch in islice(iter(dataloader), max(0, int(num_batches))):
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    raise ValueError("Expected (None, sequence, positive, negative)")
                seq, pos, neg = [
                    torch.as_tensor(x, dtype=torch.long, device=self.device)
                    for x in batch[1:]
                ]
                if seq.ndim != 2 or pos.shape != seq.shape or neg.shape != seq.shape:
                    raise ValueError("Diagnostic tensors must share shape B x L")
                if any(
                    bool(((x < 0) | (x > network.itemnum)).any())
                    for x in (seq, pos, neg)
                ):
                    raise ValueError("Diagnostic item IDs exceed the model vocabulary")
                features = self.view.log2feats(seq)
                ps = self.view.predict_score(features, pos)
                ns = self.view.predict_score(features, neg)
                if ps.shape != seq.shape or ns.shape != seq.shape:
                    raise ValueError("Position scores must have shape B x L")
                valid = (seq != 0) & (pos != 0) & (neg != 0)
                values = (ps.float() - ns.float())[valid]
                finite = torch.isfinite(values)
                positions += values.numel()
                nonfinite += int((~finite).sum())
                batches += 1
                margins.append(values[finite].cpu())
        finally:
            for module, training in modes:
                module.training = training
        values = torch.cat(margins) if margins else torch.empty(0)
        count = values.numel()
        return {
            "mean": float(values.mean()) if count else None,
            "p10": (
                float(values.kthvalue(max(1, int(np.ceil(0.1 * count)))).values)
                if count
                else None
            ),
            "neg_beats_pos_rate": float((values < 0).float().mean()) if count else None,
            "small_margin_rate": (
                float((values < eps).float().mean()) if count else None
            ),
            "eps": float(eps),
            "counts": {
                "batches": batches,
                "valid_positions": positions,
                "finite_positions": count,
                "nonfinite_positions": nonfinite,
            },
        }, None  # Item categories are unavailable.

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Bounded loss-faithful endpoint signal audit
    def _local_signal_audit(self):
        import os
        import time

        network, view = self.view.model, self.view
        expected = {
            "scorer": "tied_item_dot_product",
            "objective": "softplus(negative_score-positive_score)",
            "mask": "sequence_positive_negative_nonzero",
            "reduction": "mean_over_valid_positions",
            "pair_weight": 1.0,
            "negative_distribution": "uniform_integer_catalog",
            "negative_exclusions": "complete_training_history",
            "repeated_targets": "allowed",
            "regularization": "optimizer_weight_decay_excluded_from_audit",
        }
        if getattr(network, "local_audit_spec", None) != expected:
            return {
                "value": None,
                "reason": "Explicit supported scorer/loss/sampler contract unavailable",
            }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        histories = self.user_train.get("History", self.user_train)
        rng = np.random.default_rng(42)
        itemnum, maxlen = int(network.itemnum), int(network.maxlen)
        if not isinstance(histories, dict) or itemnum < 1 or maxlen < 1:
            return {
                "value": None,
                "reason": "Invalid mapped histories or model dimensions",
            }
        frequencies, eligible, excluded = Counter(), [], Counter()
        for uid in sorted(histories, key=lambda u: (type(u).__name__, str(u))):
            h = np.asarray(histories[uid], dtype=np.int64)
            if h.ndim != 1 or np.any((h < 1) | (h > itemnum)):
                excluded["invalid_history"] += 1
                continue
            frequencies.update(h.tolist())
            if len(h) >= 2 and len(set(h.tolist())) < itemnum:
                eligible.append(uid)
            else:
                excluded["not_sampler_eligible"] += 1
        chosen = sorted(
            rng.choice(len(eligible), min(64, len(eligible)), replace=False).tolist()
        )
        frequency_values = np.asarray(list(frequencies.values()), dtype=np.float64)
        cuts = (
            np.quantile(frequency_values, [1 / 3, 2 / 3]).tolist()
            if len(frequency_values)
            else []
        )
        records, gradients = [], []
        counts = Counter()

        def summary(value):
            a = np.asarray(value, dtype=np.float64).reshape(-1)
            finite = np.isfinite(a)
            result = {
                "count": int(a.size),
                "finite": int(finite.sum()),
                "nonfinite": int((~finite).sum()),
            }
            a = a[finite]
            result.update(
                mean=float(a.mean()) if len(a) else None,
                quantiles=(
                    np.quantile(a, [0, 0.1, 0.5, 0.9, 1]).tolist() if len(a) else None
                ),
            )
            return result

        category = str(
            getattr(view, "dataset_name", getattr(network, "dataset_name", "unknown"))
        )
        for index in chosen:
            uid = eligible[index]
            history = np.asarray(histories[uid], dtype=np.int64)
            pool = np.setdiff1d(np.arange(1, itemnum + 1), history, assume_unique=False)
            if len(pool) < 99:
                excluded["selected_pool_below_99"] += 1
                continue
            negatives = rng.choice(pool, 99, replace=False)
            target = int(history[-1])
            retained = history[:-1][-maxlen:]
            seq = torch.zeros((1, maxlen), dtype=torch.long, device=self.device)
            seq[0, -len(retained) :] = torch.as_tensor(retained, device=self.device)
            items = torch.as_tensor(
                np.concatenate(([target], negatives)),
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                feature = view.log2feats(seq)[
                    0, torch.where(seq[0] != 0)[0][-1]
                ].detach()
                reference = view.predict_score(feature, items).detach().reshape(-1)
                embeddings = view.item_emb(items).detach()
            counts["encoder_forwards"] += 1
            counts["reference_scorer_calls"] += 1
            row = {
                "user_id": str(uid),
                "target": target,
                "negative_ids": negatives.tolist(),
                "original_length": len(history),
                "retained_prefix_length": len(retained),
                "pool_size": len(pool),
                "repeated_target": bool(target in history[:-1]),
                "target_frequency": frequencies[target],
                "frequency_bin": int(
                    np.searchsorted(cuts, frequencies[target], side="left")
                ),
                "length_bin": (
                    "1-10"
                    if len(history) <= 10
                    else "11-25" if len(history) <= 25 else "26+"
                ),
            }
            with torch.enable_grad():
                local = feature.clone().requires_grad_(True)
                scores = local @ embeddings.T
                if (
                    reference.shape != scores.shape
                    or not torch.isfinite(reference).all()
                    or not torch.isfinite(scores).all()
                    or not torch.isfinite(feature).all()
                    or not torch.isfinite(embeddings).all()
                ):
                    row.update(value=None, reason="Invalid or nonfinite scorer outputs")
                elif not torch.allclose(scores, reference, atol=1e-6, rtol=1e-5):
                    row.update(
                        value=None,
                        reason="Detached scorer failed reference verification",
                    )
                else:
                    losses = F.softplus(scores[1:] - scores[0])
                    raw = torch.sigmoid(scores[1:] - scores[0]).detach()
                    ds, gradient = torch.autograd.grad(losses.mean(), (scores, local))
                    gp = -raw[:, None] * embeddings[0]
                    gn = raw[:, None] * embeddings[1:]
                    pair = gp + gn
                    # Compute geometry in float64 to avoid squared-norm
                    # overflow/underflow obscuring small BPR contributions.
                    gp64, gn64 = gp.double(), gn.double()
                    pair_norm = (gp64 + gn64).norm(dim=-1)
                    np_, nn_ = gp64.norm(dim=-1), gn64.norm(dim=-1)
                    denominator = np_ + nn_
                    cosine_denominator = np_ * nn_
                    ratio = torch.where(
                        denominator > 0, pair_norm / denominator, torch.nan
                    )
                    cosine = torch.where(
                        cosine_denominator > 0,
                        ((gp64 * gn64).sum(-1) / cosine_denominator).clamp(-1, 1),
                        torch.nan,
                    )
                    expected_ds = torch.cat((-raw.mean().view(1), raw / 99))
                    verified = torch.allclose(ds, expected_ds, atol=1e-6, rtol=1e-5)
                    verified &= torch.allclose(
                        gradient, pair.mean(0), atol=1e-6, rtol=1e-5
                    )
                    verified &= bool(
                        torch.isfinite(gradient).all() and torch.isfinite(pair).all()
                    )
                    if not verified:
                        row.update(
                            value=None,
                            reason="Derivative identity or finite check failed",
                        )
                    else:
                        hard = torch.argsort(
                            scores[1:].detach(), descending=True, stable=True
                        )[:10]
                        arrays = {
                            "positive_logits": scores[0].detach().expand(99),
                            "negative_logits": scores[1:].detach(),
                            "margins": scores[0].detach() - scores[1:].detach(),
                            "pair_loss": losses.detach(),
                            "raw_positive_score_derivative": -raw,
                            "raw_negative_score_derivative": raw,
                            "averaged_negative_score_coefficient": ds[1:].detach(),
                            "positive_contribution_norm": np_,
                            "negative_contribution_norm": nn_,
                            "pair_gradient_norm": pair_norm,
                            "contribution_cosine": cosine,
                            "cancellation_ratio": ratio,
                            "negative_embedding_norm": embeddings[1:]
                            .double()
                            .norm(dim=-1),
                        }
                        row["summaries"] = {
                            k: summary(v.detach().double().cpu().numpy())
                            for k, v in arrays.items()
                        }
                        row["hardest_ten_negative_summaries"] = {
                            k: summary(v[hard].detach().double().cpu().numpy())
                            for k, v in arrays.items()
                        }
                        row.update(
                            positive_score_coefficient=float(ds[0]),
                            feature_norm=float(feature.double().norm()),
                            positive_embedding_norm=float(
                                embeddings[0].double().norm()
                            ),
                            averaged_gradient_norm=float(gradient.double().norm()),
                            ties=int((scores[1:] == scores[0]).sum()),
                            derivative_identity_max_error=float(
                                (ds - expected_ds).abs().max()
                            ),
                            scorer_max_error=float(
                                (scores.detach() - reference).abs().max()
                            ),
                        )
                        # Pair-level paired values permit later analysis without rerunning the model.
                        row["paired_values"] = {
                            k: [
                                float(x) if np.isfinite(x) else None
                                for x in v.detach().double().cpu().numpy()
                            ]
                            for k, v in arrays.items()
                        }
                        gradients.append(gradient.detach().double().cpu())
            records.append(row)
        valid = [r for r in records if "summaries" in r]
        result = {
            "specification": expected,
            "category": category,
            "selected_users": len(chosen),
            "eligible_users": len(eligible),
            "verified_users": len(valid),
            "exclusions": dict(excluded),
            "failed_users": len(records) - len(valid),
            "frequency_cutpoints": cuts,
            "frequency_tie_rule": "lower bin on equality; bins may be empty",
            "precision": str(view.item_emb.weight.dtype),
            "operation_counts": dict(counts),
            "mean_user_gradient_norm": (
                float(torch.stack(gradients).norm(dim=1).mean()) if gradients else None
            ),
            "norm_of_mean_user_gradient": (
                float(torch.stack(gradients).mean(0).norm()) if gradients else None
            ),
            "full_training_loss_equivalence": {
                "value": None,
                "reason": "Endpoint-only evaluation excludes other positions and training dropout",
            },
            "bce_threshold_fractions": {
                "value": None,
                "reason": "Supplied objective is BPR, not BCE",
            },
            "prospective_added_value": {
                "value": None,
                "reason": "No prospective outcomes supplied; score derivatives are deterministic margin functions",
            },
            "paired_user_records": records,
        }
        result["strata"] = {
            field: {
                str(v): {
                    "users": sum(r[field] == v for r in valid),
                    "gradient_norm": summary(
                        [r["averaged_gradient_norm"] for r in valid if r[field] == v]
                    ),
                }
                for v in sorted({r[field] for r in valid}, key=str)
            }
            for field in ("frequency_bin", "length_bin")
        }
        safe_category = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in category
        )
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, safe_category + "_local_signal_audit.json")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["artifact_path"] = path
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        return result

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Validated candidate-depth calibration with bounded bootstrap memory
    @torch.no_grad()
    def _candidate_depth_audit(self, exact_limit=50):
        import hashlib
        import math

        # DEBUG: Replace unavailable SciPy with the exact hypergeometric CDF
        # using standard-library integer combinations. This preserves the
        # conditional sampling expectation without floating-point underflow.
        def hypergeom_cdf(k, N, M, m):
            if not (0 <= M <= N and 0 <= m <= N):
                raise ValueError("Invalid hypergeometric population or sample size")
            lower = max(0, m - (N - M))
            upper = min(m, M)
            cutoff = math.floor(k)
            if cutoff < lower:
                return 0.0
            if cutoff >= upper:
                return 1.0
            favorable = sum(
                math.comb(M, successes) * math.comb(N - M, m - successes)
                for successes in range(lower, cutoff + 1)
            )
            return favorable / math.comb(N, m)

        view, network = self.view, self.view.model
        histories = self.user_train.get("History", self.user_train)
        if not isinstance(histories, dict):
            raise TypeError("Candidate audit requires mapped training histories")
        if int(network.maxlen) <= 0:
            raise ValueError("Candidate audit requires a positive model.maxlen")
        if not isinstance(exact_limit, (int, np.integer)) or exact_limit < 0:
            raise ValueError("exact_limit must be a nonnegative integer")
        category = str(
            getattr(view, "dataset_name", getattr(network, "dataset_name", "unknown"))
        )
        categories = [
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        ]
        category_index = categories.index(category) if category in categories else 4

        def local_rng(uid, repetition):
            digest = hashlib.sha256(str(uid).encode("utf-8")).digest()
            return np.random.default_rng(
                np.random.SeedSequence(
                    [
                        42,
                        category_index,
                        int.from_bytes(digest[:8], "little"),
                        repetition,
                    ]
                )
            )

        real_limit = min(int(network.itemnum), len(view.item_emb.weight) - 1)
        frequencies, eligible, exclusions = Counter(), [], Counter()
        ordered = sorted(histories, key=lambda uid: (type(uid).__name__, str(uid)))
        for uid in ordered:
            history = np.asarray(histories[uid], dtype=np.int64)
            if history.ndim != 1 or np.any((history < 1) | (history > real_limit)):
                exclusions["unmapped_or_invalid_history"] += 1
                continue
            frequencies.update(history.tolist())
            if len(history) < 2:
                exclusions["no_training_prefix"] += 1
            else:
                eligible.append(uid)
        selection_rng = np.random.default_rng(
            np.random.SeedSequence([42, category_index])
        )
        chosen = sorted(
            selection_rng.choice(
                len(eligible), min(200, len(eligible)), replace=False
            ).tolist()
        )
        users = [eligible[i] for i in chosen]
        exact_indices = set(
            selection_rng.choice(
                len(users), min(max(0, exact_limit), 50, len(users)), replace=False
            ).tolist()
        )
        catalog = np.asarray(sorted(frequencies), dtype=np.int64)
        by_frequency = sorted(frequencies, key=lambda item: (-frequencies[item], item))
        bands = {
            item: (
                "head"
                if rank < int(np.ceil(0.2 * len(catalog)))
                else "middle" if rank < int(np.ceil(0.5 * len(catalog))) else "tail"
            )
            for rank, item in enumerate(by_frequency)
        }
        depths, records, exact, unavailable = (99, 499, 999), [], [], Counter()
        recommended, recommendation_frequencies = set(), []

        def scores(feature, items):
            chunks = []
            for start in range(0, len(items), 4096):
                item_tensor = torch.as_tensor(
                    items[start : start + 4096], device=self.device, dtype=torch.long
                )
                value = (
                    view.predict_score(feature, item_tensor)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                if len(value) != len(item_tensor) or not np.isfinite(value).all():
                    raise RuntimeError(
                        "Candidate audit requires one finite score per item"
                    )
                chunks.append(value)
            return np.concatenate(chunks) if chunks else np.empty(0)

        def preceding(items, values, target, positive):
            return (values > positive) | ((values == positive) & (items < target))

        for user_index, uid in enumerate(users):
            history = np.asarray(histories[uid], dtype=np.int64)
            prefix, target = history[:-1], int(history[-1])
            sequence = torch.zeros(
                (1, int(network.maxlen)), dtype=torch.long, device=self.device
            )
            retained = prefix[-sequence.shape[1] :]
            sequence[0, -len(retained) :] = torch.as_tensor(
                retained, device=self.device
            )
            endpoint = torch.where(sequence[0] != 0)[0][-1]
            feature = view.log2feats(sequence)[0, endpoint]
            positive = float(scores(feature, np.asarray([target]))[0])
            pool = catalog[~np.isin(catalog, history)]
            N = len(pool)
            row = {
                "user_id": str(uid),
                "prefix_length": len(prefix),
                "retained_length": len(retained),
                "cutoff_training_index": len(history) - 1,
                "target": target,
                "pool_size": N,
                "frequency_band": bands[target],
                "length_bin": (
                    "1-5"
                    if len(prefix) <= 5
                    else "6-20" if len(prefix) <= 20 else "21+"
                ),
                "repeated_target": bool(target in prefix),
                "repetitions": [],
            }
            for repetition in range(10):
                negatives = local_rng(uid, repetition).choice(
                    pool, min(999, N), replace=False
                )
                negative_scores = scores(feature, negatives)
                measurements = {}
                previous_rank = 1
                for depth in depths:
                    if N < depth:
                        unavailable[str(depth) + ":insufficient_unique_negatives"] += 1
                        continue
                    rank = 1 + int(
                        preceding(
                            negatives[:depth], negative_scores[:depth], target, positive
                        ).sum()
                    )
                    if rank < previous_rank:
                        raise RuntimeError(
                            "Nested candidate ranks must be nondecreasing"
                        )
                    previous_rank = rank
                    measurements[str(depth)] = {
                        "rank": rank,
                        "hit_at_10": int(rank <= 10),
                        "margin_to_tenth_negative": positive
                        - float(np.partition(negative_scores[:depth], -10)[-10]),
                    }
                row["repetitions"].append(measurements)
            records.append(row)
            if user_index in exact_indices:
                M, top_items, top_scores = 0, np.empty(0, dtype=np.int64), np.empty(0)
                for start in range(0, N, 4096):
                    items = pool[start : start + 4096]
                    values = scores(feature, items)
                    M += int(preceding(items, values, target, positive).sum())
                    top_items, top_scores = np.concatenate(
                        (top_items, items)
                    ), np.concatenate((top_scores, values))
                    keep = np.lexsort((top_items, -top_scores))[:10]
                    top_items, top_scores = top_items[keep], top_scores[keep]
                recommended.update(top_items.tolist())
                recommendation_frequencies.extend(
                    frequencies[int(item)] for item in top_items
                )
                exact.append(
                    {
                        "user_id": str(uid),
                        "N": N,
                        "M": M,
                        "full_eligible_pool_rank": 1 + M,
                        "expected": {
                            str(m): {
                                # DEBUG: Execute the dependency-free exact CDF.
                                "hit_at_10": hypergeom_cdf(9, N, M, m),
                                "rank": 1 + m * M / N,
                            }
                            for m in depths
                            if m <= N
                        },
                        "top10_eligible_negative_items": top_items.tolist(),
                    }
                )
        fields = ("hit_at_10", "rank", "margin_to_tenth_negative")
        bootstrap_weights = {}

        def aggregate(rows, requested):
            common = [
                r for r in rows if all(str(d) in r["repetitions"][0] for d in requested)
            ]
            if not common:
                return {
                    "users": 0,
                    "estimates": None,
                    "reason": "No common eligible users",
                }
            values = np.asarray(
                [
                    [
                        [r["repetitions"][rep][str(d)][field] for field in fields]
                        for d in requested
                    ]
                    for r in common
                    for rep in range(10)
                ]
            )
            values = values.reshape(len(common), 10, len(requested), len(fields))
            n = len(common)
            if n not in bootstrap_weights:
                draws = np.random.default_rng(42).integers(n, size=(2000, n))
                counts = np.zeros((2000, n), dtype=np.float64)
                np.add.at(counts, (np.arange(2000)[:, None], draws), 1.0)
                bootstrap_weights[n] = counts / n
            # Multiplicities reproduce the same paired user resamples without
            # materializing a resample x user x depth x field tensor.
            bootstrap = (bootstrap_weights[n] @ values[:, 0].reshape(n, -1)).reshape(
                2000, len(requested), len(fields)
            )
            repetition_means = values.mean(axis=0)
            estimates, contrasts = {}, {}
            for j, depth in enumerate(requested):
                estimates[str(depth)] = {
                    field: {
                        "mean_over_ten_repetitions": float(values[:, :, j, k].mean()),
                        "repetition_zero_mean": float(values[:, 0, j, k].mean()),
                        "repetition_zero_user_ci95": np.quantile(
                            bootstrap[:, j, k], [0.025, 0.975]
                        ).tolist(),
                        "repetition_sd": float(repetition_means[:, j, k].std(ddof=1)),
                        "monte_carlo_se_of_ten_repetition_mean": float(
                            repetition_means[:, j, k].std(ddof=1) / np.sqrt(10)
                        ),
                    }
                    for k, field in enumerate(fields)
                }
                if j:
                    contrasts[str(depth) + "_minus_" + str(requested[0])] = {
                        field: {
                            "repetition_zero_mean_difference": float(
                                (values[:, 0, j, k] - values[:, 0, 0, k]).mean()
                            ),
                            "paired_user_ci95": np.quantile(
                                bootstrap[:, j, k] - bootstrap[:, 0, k], [0.025, 0.975]
                            ).tolist(),
                        }
                        for k, field in enumerate(fields)
                    }
            return {
                "users": len(common),
                "estimates": estimates,
                "paired_contrasts": contrasts,
            }

        by_id = {r["user_id"]: r for r in records}
        calibration = {}
        for depth in depths:
            key = str(depth)
            cohort = [e for e in exact if key in e["expected"]]
            calibration[key] = {
                "users": len(cohort),
                "expected_hit_at_10": None,
                "observed_mean_hit_at_10": None,
            }
            if cohort:
                expected = float(
                    np.mean([e["expected"][key]["hit_at_10"] for e in cohort])
                )
                observed = float(
                    np.mean(
                        [
                            by_id[e["user_id"]]["repetitions"][rep][key]["hit_at_10"]
                            for e in cohort
                            for rep in range(10)
                        ]
                    )
                )
                calibration[key].update(
                    expected_hit_at_10=expected,
                    observed_mean_hit_at_10=observed,
                    observed_minus_expected=observed - expected,
                )
        return {
            "category": category,
            "category_seed_index": category_index,
            "eligible_training_users": len(eligible),
            "selected_users": len(users),
            "selected_fraction_of_eligible": (
                len(users) / len(eligible) if eligible else None
            ),
            "selected_user_ids": [str(uid) for uid in users],
            "exclusions": dict(exclusions),
            "unavailable_user_repetition_depths": dict(unavailable),
            "nested_monotonicity_violations": 0,
            "depth_specific": {str(d): aggregate(records, [d]) for d in depths},
            "common_eligible_comparisons": {
                str(a) + ":" + str(b): aggregate(records, [a, b])
                for a, b in ((99, 499), (99, 999), (499, 999))
            },
            "all_depth_common_users": aggregate(records, depths),
            "strata": {
                field: {
                    str(value): {
                        "selected_users": sum(r[field] == value for r in records),
                        "depths": {
                            str(d): aggregate(
                                [r for r in records if r[field] == value], [d]
                            )
                            for d in depths
                        },
                    }
                    for value in sorted(set(r[field] for r in records), key=str)
                }
                for field in ("frequency_band", "length_bin", "repeated_target")
            },
            "exact_pool_records": exact,
            "sampling_calibration": calibration,
            "eligible_pool_recommendations": {
                "audited_users": len(exact),
                "unique_top10_items": len(recommended),
                "training_catalog_size": len(catalog),
                "catalog_coverage": (
                    len(recommended) / len(catalog) if len(catalog) else None
                ),
                "mean_training_frequency": (
                    float(np.mean(recommendation_frequencies))
                    if recommendation_frequencies
                    else None
                ),
            },
            "paired_user_records": records,
        }

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Padding-aware diagnostics with bounded paired bootstrap memory and synchronized timing
    def run_full_diagnosis(self, train_loader):
        from itertools import islice
        import time

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        diagnostic_started = time.perf_counter()
        view, network = self.view, self.view.model
        rng = np.random.default_rng(42)
        user_rng = np.random.default_rng(42)
        candidate_rng = np.random.default_rng(42)
        previous_modes = [(module, module.training) for module in network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        metrics = {}
        definitions = {
            "training_positions": "Descriptive position-weighted training statistics from at most ten batches, requiring nonzero sequence, positive and negative IDs. No independent-user inference.",
            "pos_neg_margin_mean": "Mean positive-minus-negative training score; scale dependent.",
            "pos_neg_margin_p10": "Nearest-rank tenth percentile of training margins.",
            "pos_neg_neg_beats_pos_rate": "Fraction of valid training margins below zero.",
            "pos_neg_small_margin_rate": "Fraction of valid training margins below 0.5; this threshold is scale dependent.",
            "pairwise_accuracy": "P(positive score > negative score) + 0.5 P(tie), conditional on sampled negative difficulty; invariant to strictly increasing score transformations.",
            "embedding_angular_alignment": "(Mean off-diagonal cosine + 1)/2; approximately 0.5 means zero mean cosine. Not a ranking-quality or semantic-diversity metric.",
            "representations": "Centered covariance eigenvalues for raw and unit-normalized vectors; total variance, largest eigenvalue share and exp(-sum(p log p)), where p is covariance eigenvalue mass. This is not RankMe. Zero variance has null effective rank.",
            "structure": "Inspection of actual NextItNet residual blocks, temporal convolutions, positionwise norms, biases, explicit causal padding and masks. Reachability is conditional on this recognized graph and absence of unrecognized operations.",
            "checks": "Feature comparisons in eval mode: atol=1e-6, rtol=1e-5; report maximum absolute difference. Width checks retain item content but may change absolute positions.",
            "training_endpoint_interventions": "Fixed positive/negative endpoint pair, largest nonzero index, descriptive loader-row margin changes; rows are not identified independent users.",
            "user_score_sensitivity": "Identified public training users, fixed 99 training-observed unseen candidates per user. Mean absolute score change and signed mean score change; no relevance labels, HR or target ranks.",
            "temporal_geometry": "Identified-user strengths 0.25 and 0.5 use k=max(1,floor(fraction*original_nonpadding_length)), before baseline truncation. Interventions are unavailable when retained history cannot support the requested size or preserve the newest k items. Loader-row probes use their observed nonpadding lengths because full histories are unidentified. Deletions compact and left-pad; replacements preserve positions; shuffles preserve identities and multiplicities. Unchanged draws are excluded and counted, so shuffle estimates condition on changed draws.",
            "user_intervals": "Pointwise 95% percentile intervals from 2000 seed-42 user bootstrap resamples. Each complete paired vector is resampled together within the current category and original-length stratum. Conditional on fitted model, fixed users/candidates; excludes training-seed, candidate-sampling and adaptive-search uncertainty. Exploratory, no p-values.",
            "missingness": "Unavailable observations are null with reasons, never zero-effect evidence. Cross-category macro estimates require all four categories.",
        }

        def summary(values):
            x = np.asarray(values, dtype=np.float64).reshape(-1)
            if not x.size:
                return {"n": 0, "mean": None, "reason": "No eligible observations"}
            if not np.isfinite(x).all():
                return {
                    "n": int(x.size),
                    "mean": None,
                    "reason": "Nonfinite observations",
                }
            return {
                "n": int(x.size),
                "mean": float(x.mean()),
                "p10": float(np.quantile(x, 0.1)),
                "p50": float(np.quantile(x, 0.5)),
                "p90": float(np.quantile(x, 0.9)),
                "min": float(x.min()),
                "max": float(x.max()),
            }

        def representation(x):
            x = x.detach().float().cpu().double()
            if not torch.isfinite(x).all():
                return {
                    "count": int(x.shape[0]),
                    "value": None,
                    "reason": "Nonfinite representation values; covariance unavailable",
                }
            result = {
                "count": int(x.shape[0]),
                "norms": summary(x.norm(dim=-1).numpy()),
            }
            for name, z in (("raw", x), ("unit_normalized", F.normalize(x, dim=-1))):
                if len(z) < 2:
                    result[name] = {
                        "total_variance": None,
                        "effective_rank": None,
                        "reason": "At least two vectors required",
                    }
                    continue
                z = z - z.mean(0, keepdim=True)
                eig = torch.linalg.eigvalsh(z.T @ z / (len(z) - 1)).clamp_min(0)
                total = float(eig.sum())
                entry = {
                    "eigenvalues": eig.tolist(),
                    "total_variance": total,
                    "leading_share": None,
                    "effective_rank": None,
                }
                if total > 0:
                    p = eig[eig > 0] / total
                    entry.update(
                        leading_share=float(eig[-1] / total),
                        effective_rank=float(torch.exp(-(p * p.log()).sum())),
                    )
                else:
                    entry["reason"] = "Zero centered variance"
                result[name] = entry
            return result

        def encode(seq):
            return view.log2feats(
                torch.as_tensor(seq, dtype=torch.long, device=self.device)
            )

        def endpoint(seq):
            positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
            return positions.masked_fill(seq == 0, -1).max(1).values

        def final_features(sequences):
            seq = torch.as_tensor(
                (
                    sequences
                    if isinstance(sequences, torch.Tensor)
                    else np.asarray(sequences)
                ),
                dtype=torch.long,
                device=self.device,
            )
            if seq.ndim != 2 or seq.shape[1] == 0:
                raise ValueError("Diagnostic sequences must have nonempty B x L shape")
            last = endpoint(seq)
            if (last < 0).any():
                raise ValueError("All-padding history cannot be scored by diagnostics")
            feats = encode(seq)
            return feats[torch.arange(len(seq), device=self.device), last]

        def score_vectors(sequences, candidates):
            output = []
            for start in range(0, len(sequences), 32):
                feats = final_features(sequences[start : start + 32])
                # One feature vector and one candidate list per call preserves
                # ModelView implementations accepting arbitrary item-list shapes.
                output.extend(
                    view.predict_score(f, candidates).detach().float().cpu().numpy()
                    for f in feats
                )
            return np.asarray(output)

        def compare(a, b):
            return {
                "max_absolute_difference": float((a - b).abs().max()),
                "within_tolerance": bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5)),
            }

        def inspect_structure():
            convs = [
                {
                    "name": name,
                    "kernel": list(m.kernel_size),
                    "dilation": list(m.dilation),
                    "stride": list(m.stride),
                    "padding": list(m.padding),
                    "bias": m.bias is not None,
                }
                for name, m in network.named_modules()
                if isinstance(m, torch.nn.Conv1d)
            ]
            norms = [
                {
                    "name": name,
                    "type": type(m).__name__,
                    "normalized_shape": (
                        list(m.normalized_shape)
                        if isinstance(m, torch.nn.LayerNorm)
                        else None
                    ),
                }
                for name, m in network.named_modules()
                if isinstance(
                    m,
                    (
                        torch.nn.LayerNorm,
                        torch.nn.BatchNorm1d,
                        torch.nn.InstanceNorm1d,
                        torch.nn.GroupNorm,
                    ),
                )
            ]
            recognized = type(network).__name__ == "NextItNetModel"
            recognized &= hasattr(network, "blocks") and hasattr(network, "depth_norms")
            offsets, all_depths = {0}, set()
            if recognized:
                recognized &= len(convs) == 2 * len(network.blocks)
                for block in network.blocks:
                    recognized &= type(block).__name__ == "NextItNetBlock"
                    branch = set(offsets)
                    for conv in (block.conv1, block.conv2):
                        recognized &= (
                            conv.stride == (1,)
                            and conv.padding == (0,)
                            and conv.kernel_size == (block.kernel_size,)
                            and conv.dilation == (block.dilation,)
                        )
                        branch = {
                            o + j * conv.dilation[0]
                            for o in branch
                            for j in range(conv.kernel_size[0])
                        }
                    offsets |= branch
                    all_depths |= offsets
                recognized &= all(
                    n["type"] == "LayerNorm"
                    and n["normalized_shape"] == [network.hidden_units]
                    for n in norms
                )
                # The supplied gate includes a cumulative prefix mean. Its
                # endpoint can depend on every retained earlier input token.
                if hasattr(network, "prefix_norm") and hasattr(network, "depth_gate"):
                    all_depths.update(range(int(network.maxlen)))
            return {
                "convolutions": convs,
                "normalizations": norms,
                "recognized_baseline_graph": bool(recognized),
                "reachable_endpoint_offsets": (
                    sorted(all_depths) if recognized else None
                ),
                "receptive_field_span": (
                    max(all_depths) + 1 if recognized and all_depths else None
                ),
                "padding_and_masks": (
                    "Recognized baseline explicitly left-pads each convolution and masks before/after convolutions, normalization and residual paths"
                    if recognized
                    else None
                ),
                "locality_caveat": "Conditional on supplied baseline forward graph; empirical checks below corroborate but do not prove locality",
            }

        try:
            network.eval()
            with torch.no_grad():
                metrics["structure"] = inspect_structure()
                itemnum = int(network.itemnum)
                weights = view.item_emb.weight.detach()
                real_count = min(itemnum, len(weights) - 1)
                indices = rng.choice(
                    np.arange(1, real_count + 1), min(2048, real_count), replace=False
                )
                sample = weights[
                    torch.as_tensor(indices, dtype=torch.long, device=weights.device)
                ]
                metrics["item_representations"] = representation(sample)
                if len(sample) > 1:
                    unit = F.normalize(sample.float(), dim=-1)
                    cosine = (unit.sum(0).square().sum() - unit.square().sum()) / (
                        len(unit) * (len(unit) - 1)
                    )
                    metrics["embedding_angular_alignment"] = float(
                        (cosine.clamp(-1, 1) + 1) / 2
                    )
                else:
                    metrics["embedding_angular_alignment"] = None
                margins, positives, negatives, final_states = [], [], [], []
                endpoint_effects = []
                batch_count, valid_count = 0, 0
                for batch in islice(iter(train_loader), 10):
                    _, sequence, positive, negative = batch
                    seq, pos, neg = [
                        torch.as_tensor(x, dtype=torch.long, device=self.device)
                        for x in (sequence, positive, negative)
                    ]
                    feats = encode(seq)
                    # Score the two B x L tensors separately: the actual NextItNet
                    # scorer multiplies B x L x D features and item embeddings.
                    ps, ns = view.predict_score(feats, pos), view.predict_score(
                        feats, neg
                    )
                    valid = (seq != 0) & (pos != 0) & (neg != 0)
                    margins.extend((ps - ns)[valid].float().cpu().tolist())
                    positives.extend(ps[valid].float().cpu().tolist())
                    negatives.extend(ns[valid].float().cpu().tolist())
                    valid_count += int(valid.sum())
                    batch_count += 1
                    last = endpoint(seq)
                    rows = torch.where(last >= 0)[0]
                    final_states.append(feats[rows, last[rows]].detach().cpu())
                    if batch_count == 1 and len(rows):
                        checks = {
                            "identity": compare(feats, encode(seq)),
                            "batch_reordering": compare(
                                feats, encode(seq.flip(0)).flip(0)
                            ),
                            "input_width": int(seq.shape[1]),
                        }
                        row = int(rows[0])
                        nonzero = torch.where(seq[row] != 0)[0]
                        checks["retained_history"] = int(len(nonzero))
                        if len(nonzero) >= 2 and real_count >= 2:
                            pivot = int(nonzero[len(nonzero) // 2 - 1])
                            altered = seq[row : row + 1].clone()
                            future = (
                                torch.arange(seq.shape[1], device=self.device) > pivot
                            )
                            future &= altered[0] != 0
                            altered[0, future] = altered[0, future] % real_count + 1
                            checks["future_token_causality"] = dict(
                                compare(
                                    feats[row, : pivot + 1],
                                    encode(altered)[0, : pivot + 1],
                                ),
                                pivot=pivot,
                                changed_tokens=int(future.sum()),
                            )
                        offsets = metrics["structure"]["reachable_endpoint_offsets"]
                        if offsets is not None and real_count >= 2:
                            end = int(last[row])
                            outside = [
                                int(i) for i in nonzero if end - int(i) not in offsets
                            ]
                            if outside:
                                altered = seq[row : row + 1].clone()
                                altered[0, outside] = (
                                    altered[0, outside] % real_count + 1
                                )
                                checks["outside_reachable_offsets"] = dict(
                                    compare(feats[row, end], encode(altered)[0, end]),
                                    changed_lags=[end - i for i in outside],
                                )
                            else:
                                checks["outside_reachable_offsets"] = {
                                    "value": None,
                                    "reason": "No retained tokens outside reachable offsets",
                                }
                        width_checks = []
                        first = int(nonzero[0])
                        for remove in sorted({1, first // 2, first}):
                            if not 0 < remove <= first:
                                continue
                            shortened = seq[row : row + 1, remove:]
                            try:
                                width_checks.append(
                                    dict(
                                        compare(
                                            feats[row, last[row]],
                                            final_features(shortened.cpu().numpy())[0],
                                        ),
                                        input_width=int(shortened.shape[1]),
                                        removed_leading_padding=remove,
                                        absolute_positions_shifted=hasattr(
                                            network, "pos_emb"
                                        ),
                                    )
                                )
                            except (RuntimeError, ValueError, IndexError) as exc:
                                width_checks.append(
                                    {
                                        "value": None,
                                        "input_width": int(shortened.shape[1]),
                                        "reason": str(exc),
                                    }
                                )
                        checks["leading_padding_widths"] = width_checks or {
                            "value": None,
                            "reason": "No removable leading padding",
                        }
                        metrics["checks"] = checks
                        for r in rows[:8].tolist():
                            e = int(last[r])
                            if not bool(valid[r, e]):
                                continue
                            history = seq[r][seq[r] != 0].cpu().numpy()
                            if len(history) < 2:
                                continue
                            pair = torch.stack((pos[r, e], neg[r, e]))
                            baseline = float(ps[r, e] - ns[r, e])
                            for fraction in (0.25, 0.5):
                                k = max(1, int(fraction * len(history)))
                                for region, kept in (
                                    ("old", history[k:]),
                                    ("recent", history[:-k]),
                                ):
                                    altered = np.zeros(seq.shape[1], dtype=np.int64)
                                    altered[-len(kept) :] = kept
                                    scores = score_vectors([altered], pair)[0]
                                    endpoint_effects.append(
                                        {
                                            "fraction": fraction,
                                            "region": region,
                                            "retained_length": len(history),
                                            "deleted": k,
                                            "margin_change": float(
                                                scores[0] - scores[1] - baseline
                                            ),
                                        }
                                    )
                m = np.asarray(margins)
                metrics["training_positions"] = {
                    "batches": batch_count,
                    "valid_positions": valid_count,
                    "margins": summary(m),
                    "positive_scores": summary(positives),
                    "negative_scores": summary(negatives),
                }
                metrics["pos_neg_margin_mean"] = float(m.mean()) if len(m) else None
                metrics["pos_neg_margin_p10"] = (
                    float(np.sort(m)[max(0, int(np.ceil(0.1 * len(m))) - 1)])
                    if len(m)
                    else None
                )
                metrics["pos_neg_neg_beats_pos_rate"] = (
                    float((m < 0).mean()) if len(m) else None
                )
                metrics["pos_neg_small_margin_rate"] = (
                    float((m < 0.5).mean()) if len(m) else None
                )
                metrics["pairwise_accuracy"] = (
                    float(((m > 0) + 0.5 * (m == 0)).mean()) if len(m) else None
                )
                metrics["training_endpoint_interventions"] = endpoint_effects
                metrics["loader_endpoint_representations"] = (
                    representation(torch.cat(final_states))
                    if final_states
                    else {"count": 0, "reason": "No nonpadding rows"}
                )

                histories = self.user_train.get("History", self.user_train)
                if not isinstance(histories, dict):
                    raise TypeError(
                        "ModelView user_train must expose mapped item histories"
                    )
                # Histories must already use the baseline's integer item mapping.
                ordered = sorted(histories, key=lambda x: (type(x).__name__, str(x)))
                chosen = sorted(
                    user_rng.choice(
                        len(ordered), min(200, len(ordered)), replace=False
                    ).tolist()
                )
                frequencies = Counter(
                    int(i)
                    for h in histories.values()
                    for i in h
                    if 1 <= int(i) <= real_count
                )
                observed = np.asarray(sorted(frequencies), dtype=np.int64)
                ordered_frequency = sorted(
                    observed.tolist(), key=lambda i: (frequencies[i], i)
                )
                decile = {
                    item: min(9, 10 * rank // max(1, len(ordered_frequency)))
                    for rank, item in enumerate(ordered_frequency)
                }
                pools = {
                    d: np.asarray(
                        [i for i in observed if decile[int(i)] == d], dtype=np.int64
                    )
                    for d in range(10)
                }
                maxlen = int(network.maxlen)
                user_rows, skips, intact_user_states = [], Counter(), []

                def padded(history):
                    history = np.asarray(history, dtype=np.int64)
                    if history.ndim != 1 or not 0 < len(history) <= maxlen:
                        raise ValueError(
                            "History must be nonempty and fit the diagnostic input width"
                        )
                    array = np.zeros(maxlen, dtype=np.int64)
                    array[-len(history) :] = history
                    return array

                for index in chosen:
                    uid = ordered[index]
                    original = np.asarray(histories[uid], dtype=np.int64)
                    if not len(original) or np.any(
                        (original < 1) | (original > real_count)
                    ):
                        skips["empty_or_unmapped_history"] += 1
                        continue
                    h = original[-maxlen:]
                    unseen = observed[~np.isin(observed, original)]
                    if len(unseen) < 99:
                        skips["fewer_than_99_training_observed_unseen_candidates"] += 1
                        continue
                    candidates = candidate_rng.choice(unseen, 99, replace=False)
                    cand_t = torch.as_tensor(
                        candidates, dtype=torch.long, device=self.device
                    )
                    intact = padded(h)
                    intact_features = final_features([intact])[0]
                    intact_user_states.append(intact_features.detach().cpu())
                    baseline = (
                        view.predict_score(intact_features, cand_t)
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                    if baseline.shape != (99,) or not np.isfinite(baseline).all():
                        raise RuntimeError(
                            "Diagnostic scorer must return 99 finite candidate scores"
                        )
                    variants, labels, geometry = (
                        [intact.copy()],
                        ["identity"],
                        [{"changed": 0}],
                    )
                    unavailable = Counter()

                    def add(label, history, metadata):
                        array = padded(history)
                        if np.array_equal(array, intact):
                            unavailable[label + ":unchanged"] += 1
                            return
                        variants.append(array)
                        labels.append(label)
                        geometry.append(metadata)

                    for fraction in (0.25, 0.5):
                        n = len(h)
                        k = max(1, int(fraction * len(original)))
                        prefix = str(fraction) + ":"
                        if n <= k:
                            unavailable[prefix + "insufficient_remaining_history"] += 1
                            continue
                        regions = {"recent": np.arange(n - k, n)}
                        if n - k >= k:
                            regions["old"] = np.arange(k)
                        else:
                            unavailable[prefix + "old:cannot_preserve_newest_k"] += 1
                        if n - k >= k:
                            regions["random"] = np.sort(
                                rng.choice(n - k, k, replace=False)
                            )
                        else:
                            unavailable[prefix + "random:cannot_preserve_newest_k"] += 1
                        if n - k - 1 >= k:
                            regions["interior"] = np.sort(
                                rng.choice(np.arange(1, n - k), k, replace=False)
                            )
                        else:
                            unavailable[
                                prefix + "interior:cannot_preserve_newest_k"
                            ] += 1
                        for name, positions in regions.items():
                            metadata = {
                                "k": k,
                                "original_length": len(original),
                                "retained_length": n,
                                "input_width": maxlen,
                                "lags": (n - 1 - positions).tolist(),
                                "newest_k_preserved": bool(np.all(positions < n - k)),
                            }
                            add(
                                prefix + name + "_delete",
                                np.delete(h, positions),
                                metadata,
                            )
                            zeros = h.copy()
                            zeros[positions] = 0
                            add(prefix + name + "_zero", zeros, metadata)
                            replacements = h.copy()
                            unmatched = 0
                            for p in positions:
                                forbidden = np.concatenate((h, candidates))
                                options = pools[decile[int(h[p])]]
                                options = options[~np.isin(options, forbidden)]
                                if not len(options):
                                    unmatched += 1
                                    options = observed[~np.isin(observed, forbidden)]
                                if not len(options):
                                    break
                                replacements[p] = rng.choice(options)
                            else:
                                add(
                                    prefix + name + "_replace",
                                    replacements,
                                    dict(
                                        metadata, unmatched_frequency_decile=unmatched
                                    ),
                                )
                                continue
                            unavailable[prefix + name + "_replace:no_alternative"] += 1
                        for name, positions in (
                            ("recent", np.arange(n - k, n)),
                            ("old", np.arange(n - k)),
                        ):
                            if len(positions) < 2 or len(np.unique(h[positions])) < 2:
                                unavailable[
                                    prefix
                                    + name
                                    + "_shuffle:insufficient_distinct_items"
                                ] += 1
                                continue
                            for draw in range(20):
                                altered = h.copy()
                                altered[positions] = rng.permutation(h[positions])
                                add(
                                    prefix + name + "_shuffle",
                                    altered,
                                    {
                                        "k": k,
                                        "region_size": len(positions),
                                        "draw": draw,
                                        "lags": (n - 1 - positions).tolist(),
                                        "retained_length": n,
                                        "input_width": maxlen,
                                    },
                                )
                    scores = score_vectors(variants, cand_t)
                    if (
                        scores.shape != (len(variants), 99)
                        or not np.isfinite(scores).all()
                    ):
                        raise RuntimeError(
                            "Diagnostic intervention scores have invalid shape or nonfinite values"
                        )
                    effects = {}
                    for label in dict.fromkeys(labels):
                        indices = [
                            i for i, label_i in enumerate(labels) if label_i == label
                        ]
                        delta = scores[indices] - baseline
                        effects[label] = {
                            "mean_absolute_score_change": float(np.abs(delta).mean()),
                            "signed_mean_score_change": float(delta.mean()),
                            "changed_draws": len(indices),
                            "geometry": [geometry[i] for i in indices],
                        }
                    user_rows.append(
                        {
                            "user_id": str(uid),
                            "original_length": len(original),
                            "retained_length": len(h),
                            "effects": effects,
                            "ineligible": dict(unavailable),
                        }
                    )

                metrics["intact_user_endpoint_representations"] = (
                    representation(torch.stack(intact_user_states))
                    if intact_user_states
                    else {
                        "count": 0,
                        "value": None,
                        "reason": "No eligible identified training users",
                    }
                )
                definitions["intact_user_endpoint_representations"] = (
                    "Raw and unit-normalized covariance statistics and norm quantiles for "
                    "one intact final-nonpadding feature per identified training user "
                    "eligible for the fixed-candidate sensitivity probe. Loader endpoint "
                    "statistics remain separate and may contain repeated users."
                )
                definitions["paired_contrasts"] = (
                    "Within-user difference between each named method and the first method "
                    "in its declared comparison family, computed on common eligible users. "
                    "Bootstrap intervals resample each complete paired vector together. "
                    "These are score-sensitivity contrasts, not ranking improvements."
                )

                sensitivity_bootstrap_weights = {}

                def aggregate(rows, names):
                    if not rows:
                        return {
                            "eligible_users": 0,
                            "estimates": None,
                            "reason": "No common eligible users",
                        }
                    columns = [
                        (name, field)
                        for name in names
                        for field in (
                            "mean_absolute_score_change",
                            "signed_mean_score_change",
                        )
                    ]
                    values = np.asarray(
                        [
                            [r["effects"][name][field] for name, field in columns]
                            for r in rows
                        ]
                    )
                    # Reuse deterministic user multiplicities by cohort size.
                    # Every column still shares the same paired resample.
                    n = len(rows)
                    if n not in sensitivity_bootstrap_weights:
                        draws = np.random.default_rng(42).integers(n, size=(2000, n))
                        counts = np.zeros((2000, n), dtype=np.float64)
                        np.add.at(counts, (np.arange(2000)[:, None], draws), 1.0)
                        sensitivity_bootstrap_weights[n] = counts / n
                    means = sensitivity_bootstrap_weights[n] @ values
                    bounds = np.quantile(means, [0.025, 0.975], axis=0)
                    contrasts = {}
                    for j, (name, field) in enumerate(columns):
                        if name == names[0]:
                            continue
                        reference = columns.index((names[0], field))
                        paired_difference = values[:, j] - values[:, reference]
                        bootstrap_difference = means[:, j] - means[:, reference]
                        contrasts[name + "_minus_" + names[0] + ":" + field] = {
                            "mean": float(paired_difference.mean()),
                            "ci95": np.quantile(
                                bootstrap_difference, [0.025, 0.975]
                            ).tolist(),
                        }
                    return {
                        "eligible_users": len(rows),
                        "paired_contrasts": contrasts,
                        "estimates": {
                            name
                            + ":"
                            + field: {
                                "mean": float(values[:, j].mean()),
                                "ci95": bounds[:, j].tolist(),
                            }
                            for j, (name, field) in enumerate(columns)
                        },
                    }

                groups = {
                    "all": lambda r: True,
                    "1-10": lambda r: r["original_length"] <= 10,
                    "11-25": lambda r: 11 <= r["original_length"] <= 25,
                    "26+": lambda r: r["original_length"] >= 26,
                }
                methods = ["identity"] + [
                    str(fraction) + ":" + suffix
                    for fraction in (0.25, 0.5)
                    for suffix in (
                        "old_delete",
                        "recent_delete",
                        "random_delete",
                        "interior_delete",
                        "old_zero",
                        "recent_zero",
                        "random_zero",
                        "interior_zero",
                        "old_replace",
                        "recent_replace",
                        "random_replace",
                        "interior_replace",
                        "old_shuffle",
                        "recent_shuffle",
                    )
                ]
                aggregates, common = {}, {}
                for group, predicate in groups.items():
                    rows = [r for r in user_rows if predicate(r)]
                    aggregates[group] = {
                        name: aggregate(
                            [r for r in rows if name in r["effects"]], [name]
                        )
                        for name in methods
                    }
                    common[group] = {}
                    for fraction in (0.25, 0.5):
                        prefix = str(fraction) + ":"
                        families = {
                            "deletions": [
                                "old_delete",
                                "recent_delete",
                                "random_delete",
                                "interior_delete",
                            ],
                            "old_geometry_controls": [
                                "old_delete",
                                "old_zero",
                                "old_replace",
                            ],
                            "recent_geometry_controls": [
                                "recent_delete",
                                "recent_zero",
                                "recent_replace",
                            ],
                            "random_geometry_controls": [
                                "random_delete",
                                "random_zero",
                                "random_replace",
                            ],
                            "interior_geometry_controls": [
                                "interior_delete",
                                "interior_zero",
                                "interior_replace",
                            ],
                            "shuffles": ["old_shuffle", "recent_shuffle"],
                        }
                        for family, suffixes in families.items():
                            names = [prefix + suffix for suffix in suffixes]
                            paired = [
                                r
                                for r in rows
                                if all(name in r["effects"] for name in names)
                            ]
                            common[group][prefix + family] = aggregate(paired, names)
                metrics["user_score_sensitivity"] = {
                    "category": str(
                        getattr(
                            view,
                            "dataset_name",
                            getattr(network, "dataset_name", "current_category"),
                        )
                    ),
                    "selected_users": len(chosen),
                    "eligible_users": len(user_rows),
                    "skipped_users": dict(skips),
                    "maxlen": maxlen,
                    "candidate_source": "99 fixed training-observed unseen IDs per identified user",
                    "by_original_length": aggregates,
                    "common_eligible_comparisons": common,
                    "paired_user_records": user_rows,
                }
                metrics["local_signal_audit"] = self._local_signal_audit()
                definitions["local_signal_audit"] = (
                    "At most 64 identified sampler-eligible training users selected using a local seed-42 generator "
                    "after deterministic ID sorting. Target is the final training interaction; repeated targets are allowed. "
                    "Negatives are 99 unique uniform IDs from the entire integer vocabulary excluding the complete "
                    "training history, matching the declared baseline rejection-sampler support, including any "
                    "training-unobserved vocabulary IDs. This panel is separate from the training-observed candidate-depth panel. "
                    "A unique uniform sample mean is unbiased for the one-negative conditional endpoint BPR data loss, "
                    "not for the full valid-position batch loss or dropout-dependent training computation. Equal-user "
                    "endpoint summaries do not reproduce variable-length position weighting. Optimizer weight decay is excluded. "
                    "Endpoints use the largest nonpadding index in evaluation mode. Detached dot-product scores must match "
                    "the ModelView scorer at atol=1e-6, rtol=1e-5. Only local feature copies require gradients; autograd.grad "
                    "does not populate model parameter gradients. Raw pair derivatives are +/-sigmoid(negative-positive); "
                    "negative coefficients divide by 99 and the shared positive coefficient is the mean raw positive derivative. "
                    "Unweighted pair contributions are -w*positive_embedding and w*negative_embedding. Cosine is undefined "
                    "when either contribution vanishes; cancellation ratio is ||gp+gn||/(||gp||+||gn||), undefined at zero denominator. "
                    "Undefined values are null in paired records and counted as nonfinite in summaries. Quantiles are "
                    "[minimum,p10,median,p90,maximum]. Hard-tail summaries use the ten largest negative logits per user. "
                    "Derivative identities and averaged feature gradients are independently checked by autograd. "
                    "Gradient norms depend on feature coordinates and embedding norms; they omit the encoder Jacobian "
                    "and tied input-embedding pathways. Reciprocal scoring-coordinate rescaling preserves logits but changes "
                    "local gradient norms. No full-backbone conflict, optimizer update, clipping or learning-progress inference is supported. "
                    "Training-derived frequency cutpoints are catalog frequency tertiles with ties assigned to the lower bin; "
                    "original-history length bins are 1-10,11-25,26+. Cell counts are descriptive and small cells are unstable. "
                    "Paired records preserve logits, margins, derivatives and geometry for identical examples; derivative identity "
                    "checks establish redundancy with margins, not prospective diagnostic value beyond logits or candidate ranks. "
                    "Prospective added value is unavailable without later prespecified validation evidence. "
                    "Runtime includes synchronized audit computation; enclosing diagnostic timing also includes artifact writing. "
                    "No independent predictor is fitted and training, checkpoints and production inference are unchanged."
                )
                metrics["candidate_depth_audit"] = self._candidate_depth_audit()
                definitions["candidate_depth_audit"] = (
                    "In-sample exploratory training-prefix audit; no validation labels. "
                    "Up to 200 uniformly sampled eligible users without replacement, sorted "
                    "before selection; seed 42 plus declared category index and stable hashed "
                    "user ID for ten independent candidate repetitions. Unknown category uses "
                    "index 4 and must not be treated as a named category macro report. Each "
                    "target is the final permitted training interaction; its cutoff is a zero-based "
                    "training index, and it may have been optimized during fitting. Prefixes are "
                    "left-padded/truncated to model.maxlen and read at the largest nonzero index. "
                    "The catalog contains only mapped training-observed items. Negative pools "
                    "exclude every full training-history item, including target; this is retrospective "
                    "hindsight rather than a prospective exclusion policy. Repeated targets are "
                    "recorded before exclusion. Ten uniform ordered samples without replacement "
                    "share nested prefixes of 99/499/999 negatives. Missing depths are unavailable, "
                    "with counts in user-repetition-depth units. Ties favor ascending item ID. "
                    "Rank includes the positive; hit means rank<=10. Margin subtracts the tenth "
                    "largest negative score and is scale dependent. Nested ranks cannot improve "
                    "with depth; declining hit rates alone imply nothing about generalization. "
                    "Users receive equal weight; depth-specific populations may differ. Paired "
                    "comparisons use common eligible users and 2000 seed-42 user-cluster bootstrap "
                    "resamples conditional on candidate repetition zero. Intervals exclude training "
                    "seed, adaptive search and candidate-resampling uncertainty. Repetition SD is "
                    "computed across ten category-level summaries; SD/sqrt(10) is solely Monte "
                    "Carlo error of their average, with model and users fixed. Repetitions are never "
                    "independent users. Frequency bands use all valid permitted training interactions, "
                    "descending frequency then ascending item ID: first ceil(20%) head, next through "
                    "ceil(50%) middle, remainder tail. Length bins use prefixes before truncation. "
                    "At most 50 randomly selected audit users have exact full-eligible-pool ranks; "
                    "scores stream in chunks<=4096 and retain only the top ten negative items. "
                    "N is eligible negative count and M counts negatives preceding the target. "
                    "Conditional expected hit is hypergeom.cdf(9,N,M,m); expected rank is "
                    "1+m*M/N, applicable only to declared uniform sampling and tie order. "
                    "Expected/observed residuals measure sampling variability, not generalization. "
                    "Recommendation popularity and catalog coverage describe top-ten scores on "
                    "these same restricted negative pools, excluding known training items and the "
                    "target; they are not unrestricted-catalog recommendation quality. Existing "
                    "controlled shuffle/truncation probes remain descriptive controls. Only one "
                    "category is measured here; four-category macro summaries require all four "
                    "reports and equal category weights. No auxiliary scorer or model is fitted."
                )
                metrics["four_category_macro"] = {
                    "value": None,
                    "reason": "This ModelView contains one category; aggregate four category reports externally only when all four have eligible observations",
                }
                metrics["validation_interventions"] = {
                    "value": None,
                    "reason": "No authorized pinned validation user/target/candidate interface supplied",
                }
                metrics["pos_neg_margin_category_breakdown"] = {
                    "value": None,
                    "reason": "Item categories are unavailable in this benchmark",
                }
                metrics["unobserved_traits"] = [
                    "ratings",
                    "review text",
                    "prices",
                    "stores",
                    "item categories",
                    "simulator diversity",
                    "simulator conformity",
                ]
                metrics["diagnostic_scope"] = (
                    "Exploratory diagnostics only; no model, checkpoint, sampling, training, predictor or evaluator changes"
                )
        finally:
            for module, mode in previous_modes:
                module.training = mode
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics["diagnostic_elapsed_seconds"] = time.perf_counter() - diagnostic_started
        definitions["diagnostic_elapsed_seconds"] = (
            "Wall-clock seconds for all probes, including exact eligible-pool scoring "
            "and final CUDA synchronization; prior queued CUDA work is excluded by "
            "synchronizing before timing. Diagnostic execution consumes the shared "
            "experiment budget."
        )
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
