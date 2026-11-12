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

### >>> Self_EvolveRec-BLOCK-START: Preserve required data boundary and training seed
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


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    ### >>> Self_EvolveRec-BLOCK-START: Prevent rejection sampling from hanging on exhausted catalogs
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    if not users:
        raise RuntimeError("no eligible training users with an available negative item")
    ### <<< Self_EvolveRec-BLOCK-END
    batch_seq, batch_pos, batch_neg = [], [], []
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
    ### >>> Self_EvolveRec-BLOCK-START: Reuse deterministic validation candidates without consuming training RNG
    rng = random.Random(SEED)
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
            candidates = [target]
            ### >>> Self_EvolveRec-BLOCK-START: Bound unique validation sampling by available catalog size
            candidate_count = 1 + min(99, max(0, itemnum - len(interacted - {0})))
            while len(candidates) < candidate_count:
                neg_item = rng.randint(1, itemnum)
                ### <<< Self_EvolveRec-BLOCK-END
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
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
    ### >>> Self_EvolveRec-BLOCK-START: Compute numerically stable BPR on supported prediction positions
    pos_scores, neg_scores = model(seq, pos, neg)
    valid = seq.ne(0) & pos.ne(0) & neg.ne(0)
    margins = (pos_scores - neg_scores)[valid]
    loss = nn.functional.softplus(-margins).sum()
    return loss / valid.sum().clamp(min=1)
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
### >>> Self_EvolveRec-BLOCK-START: Honor host epoch defaults within the matched training budget
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

    ### >>> Self_EvolveRec-BLOCK-START: Mask every causal convolution sublayer to prevent padding contamination
    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(-1).to(x.dtype)
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
    ### >>> Self_EvolveRec-BLOCK-START: Serialize single-network depth aggregation configuration
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
        aggregation: str = "learned",
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("NextItNet requires at least one residual block")
        if aggregation not in ("learned", "uniform", "final"):
            raise ValueError(f"unknown aggregation: {aggregation}")
        self.aggregation = aggregation
        self.itemnum = itemnum
        ### <<< Self_EvolveRec-BLOCK-END
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
        self.dropout = nn.Dropout(dropout_rate)
        ### >>> Self_EvolveRec-BLOCK-START: Define shared depth normalization and dimensionally consistent causal gates
        self.norm = nn.LayerNorm(hidden_units, elementwise_affine=False)
        self.depth_norm = nn.LayerNorm(
            hidden_units, eps=self.norm.eps, elementwise_affine=True
        )
        # The encoder contracts each D-dimensional depth value with one gate row.
        self.depth_gate = (
            nn.Linear(hidden_units, num_blocks) if aggregation == "learned" else None
        )
        if self.depth_gate is not None:
            nn.init.zeros_(self.depth_gate.weight)
            nn.init.zeros_(self.depth_gate.bias)
        ### <<< Self_EvolveRec-BLOCK-END
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Aggregate causal depth values through a common normalized scoring pathway
    def decompose_permitted_input(self, seq: torch.Tensor):
        batch, length = seq.shape
        valid = seq.ne(0)
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * valid.long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x) * mask
        states = []
        for block in self.blocks:
            x = block(x, valid)
            states.append(self.depth_norm(x) * mask)
        values = torch.stack(states, dim=2)
        if self.aggregation == "final":
            weights = values.new_zeros(values.shape[:-1])
            weights[..., -1] = 1.0
        elif self.depth_gate is None:
            weights = values.new_full(values.shape[:-1], 1.0 / len(states))
        else:
            # Each depth's logit uses only its own current-position value.
            # Softmax still couples routing across all causal depths.
            logits = torch.einsum("blkd,kd->blk", values, self.depth_gate.weight)
            weights = (logits + self.depth_gate.bias).softmax(dim=-1)
        return values, weights

    def mix_depth_features(
        self, values: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        mixed = (values * weights.unsqueeze(-1)).sum(dim=2)
        return self.norm(mixed) * valid.unsqueeze(-1).to(mixed.dtype)

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        values, weights = self.decompose_permitted_input(seq)
        return self.mix_depth_features(values, weights, seq.ne(0))

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == 0:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == 1:
            # A shared candidate list: D -> C, B,D -> B,C, B,L,D -> B,L,C.
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.shape == features.shape[:-1]:
            # One aligned target per feature, including B,L training targets.
            return (features * embeddings).sum(dim=-1)
        if items.shape[:-1] == features.shape[:-1]:
            # Candidate lists attached to each feature.
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        if features.ndim == 3 and items.ndim == 2 and features.size(0) == items.size(0):
            # Per-user candidates shared across sequence positions.
            return torch.einsum("bld,bcd->blc", features, embeddings)
        raise ValueError(
            f"unsupported scoring shapes: features={tuple(features.shape)}, "
            f"items={tuple(items.shape)}"
        )

    ### <<< Self_EvolveRec-BLOCK-END

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        ### >>> Self_EvolveRec-BLOCK-START: Read final nonpadding positions for every row with explicit empty-history scores
        rows = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[rows, last]
        user_emb = user_emb * seq.ne(0).any(dim=1, keepdim=True).to(out.dtype)
        scores = self.predict_score(user_emb, candidates)
        return scores[0] if seq.size(0) == 1 else scores
        ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Seed each independently trained category model consistently
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)
    ### <<< Self_EvolveRec-BLOCK-END
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = NextItNetModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Train one learned aggregation network per category
        maxlen=MAXLEN,
        aggregation="learned",
    ).to(device)
    receptive_field = 1 + sum(
        (conv.kernel_size[0] - 1) * conv.dilation[0]
        for block in model.blocks
        for conv in (block.conv1, block.conv2)
    )
    print(
        f"  [{dataset_name}] aggregation={model.aggregation}"
        f" parameters={sum(p.numel() for p in model.parameters())}"
        f" receptive_field={receptive_field}",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist compact gate and normalization architecture with the checkpoint
                "dropout_rate": DROPOUT_RATE,
                "aggregation": model.aggregation,
                "architecture": "nextitnet_per_depth_gate_v2",
                "depth_norm_eps": model.depth_norm.eps,
                "depth_norm_affine": model.depth_norm.elementwise_affine,
                "output_norm_eps": model.norm.eps,
                "output_norm_affine": model.norm.elementwise_affine,
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
        ### >>> Self_EvolveRec-BLOCK-START: Require one trained checkpoint for every routed category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing NextItNet artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Validate and reload one consistent compact-gate model per category
        if meta.get("architecture") != "nextitnet_per_depth_gate_v2":
            raise ValueError(f"Incompatible NextItNet architecture for {ds}")
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            aggregation=meta["aggregation"],
        )
        if (
            model.depth_norm.eps != meta.get("depth_norm_eps")
            or model.depth_norm.elementwise_affine != meta.get("depth_norm_affine")
            or model.norm.eps != meta.get("output_norm_eps")
            or model.norm.elementwise_affine != meta.get("output_norm_affine")
        ):
            raise ValueError(f"Incompatible NextItNet normalization metadata for {ds}")
        model = model.to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject incorrect dataset routing instead of silently using another vocabulary
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple category models are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"Unknown or unloaded dataset: {dataset_name!r}")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset NextItNet training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


