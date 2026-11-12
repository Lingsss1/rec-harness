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


