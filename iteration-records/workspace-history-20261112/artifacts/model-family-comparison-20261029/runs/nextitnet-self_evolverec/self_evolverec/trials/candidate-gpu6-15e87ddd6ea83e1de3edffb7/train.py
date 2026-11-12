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
### >>> Self_EvolveRec-BLOCK-START: Preserve required datasets, data boundary and training seed
DATASETS = list(DEFAULT_DATASETS)

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
    ### >>> Self_EvolveRec-BLOCK-START: Cache immutable training sampling data without changing sampling order or randomness
    # Retain the source object itself to prevent accidental cache hits after ID reuse.
    cache = getattr(sample_batch, "_sampling_cache", None)
    if cache is None or cache[0] is not user_train or cache[1] != itemnum:
        if itemnum < 1:
            raise ValueError("Training requires a positive item vocabulary")
        users = [u for u, items in user_train.items() if len(items) >= 2]
        if not users:
            raise RuntimeError("no users with at least two training interactions")
        observed_items = {}
        for user in users:
            observed = frozenset(user_train[user])
            if any(item < 1 or item > itemnum for item in observed):
                raise ValueError(f"Training user {user} has invalid item IDs")
            if len(observed) >= itemnum:
                raise ValueError(f"Training user {user} has no eligible negative items")
            observed_items[user] = observed
        cache = (user_train, itemnum, users, observed_items)
        sample_batch._sampling_cache = cache
    users, observed_items = cache[2], cache[3]
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
        ### >>> Self_EvolveRec-BLOCK-START: Reuse cached training-only negative exclusions
        interacted = observed_items[user]
        ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible checkpoint validation without consuming training randomness
    rng = random.Random(SEED)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    ### >>> Self_EvolveRec-BLOCK-START: Preserve the caller's model mode during validation
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    was_training = model.training
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            ### >>> Self_EvolveRec-BLOCK-START: Fixed unique validation negatives and explicit infeasible-pool errors
            interacted = set(history) | {target, 0}
            excluded_count = sum(1 for item in interacted if 1 <= item <= itemnum)
            if itemnum - excluded_count < 99:
                raise ValueError(
                    f"Validation user {user} has fewer than 99 eligible unique negatives"
                )
            candidates = [target]
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
    ### >>> Self_EvolveRec-BLOCK-START: Restore the caller's model mode after validation
    n = float(len(val_users))
    model.train(was_training)
    ### <<< Self_EvolveRec-BLOCK-END
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
    mask = (pos != 0).float()
    ### >>> Self_EvolveRec-BLOCK-START: Stable BPR gradients for difficult negative pairs
    loss = nn.functional.softplus(neg_scores - pos_scores)
    ### <<< Self_EvolveRec-BLOCK-END
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# Hyper-parameters
MAXLEN = int(os.environ.get("NEXTITNET_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("NEXTITNET_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("NEXTITNET_BLOCKS", "8"))
KERNEL_SIZE = int(os.environ.get("NEXTITNET_KERNEL", "3"))
DROPOUT_RATE = float(os.environ.get("NEXTITNET_DROPOUT", "0.2"))
LR = float(os.environ.get("NEXTITNET_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("NEXTITNET_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("NEXTITNET_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Fixed training budget and single-configuration residual-gate controls
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1,
        int(os.environ.get("NEXTITNET_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
# One invocation trains exactly one configuration and one model per category.
# Separate host trials may compare these controls; their predictions are never fused.
GATE_MODE = os.environ.get("NEXTITNET_GATE_MODE", "contextual")
if GATE_MODE not in {"fixed_one", "scalar", "bias_only", "contextual"}:
    raise ValueError(f"Unknown residual gate mode: {GATE_MODE}")
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("NEXTITNET_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("NEXTITNET_PATIENCE", "20"))


### >>> Self_EvolveRec-BLOCK-START: Explicit residual-gate configuration for each causal block
class NextItNetBlock(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        dilation: int,
        kernel_size: int,
        dropout_rate: float,
        gate_mode: str = "contextual",
    ):
        ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Baseline-equivalent contextual and static residual gates
        self.activation = nn.GELU()
        if gate_mode not in {"fixed_one", "scalar", "bias_only", "contextual"}:
            raise ValueError(f"Unknown residual gate mode: {gate_mode}")
        self.gate_mode = gate_mode
        if gate_mode == "contextual":
            # CPU construction consumes no net RNG state, preserving subsequent
            # convolution and embedding initialization across gate controls.
            with torch.random.fork_rng(devices=[]):
                self.gate = nn.Linear(2 * hidden_units, hidden_units)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, 2.0)
        elif gate_mode in {"scalar", "bias_only"}:
            width = 1 if gate_mode == "scalar" else hidden_units
            self.gate_bias = nn.Parameter(torch.full((width,), 2.0))
        ### <<< Self_EvolveRec-BLOCK-END

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))
        return conv(x)

    ### >>> Self_EvolveRec-BLOCK-START: Modulate completed causal updates with baseline-equivalent gates
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x * mask
        y = (self.norm1(residual) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y)) * mask
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        y = self.dropout(y) * mask
        if self.gate_mode == "fixed_one":
            return (residual + y) * mask
        if self.gate_mode == "contextual":
            logits = self.gate(torch.cat((residual, y), dim=-1))
        else:
            logits = self.gate_bias
        numerator = torch.sigmoid(logits)
        denominator = torch.sigmoid(numerator.new_tensor(2.0))
        gate = numerator / denominator
        return (residual + gate * y) * mask

    ### <<< Self_EvolveRec-BLOCK-END


class NextItNetModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        ### >>> Self_EvolveRec-BLOCK-START: Serialize the residual-gate mode explicitly
        dropout_rate: float = 0.2,
        maxlen: int = 128,
        gate_mode: str = "contextual",
    ):
        super().__init__()
        self.gate_mode = gate_mode
        self.itemnum = itemnum
        ### <<< Self_EvolveRec-BLOCK-END
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList(
            [
                ### >>> Self_EvolveRec-BLOCK-START: Apply the selected gate control throughout the original dilation schedule
                NextItNetBlock(
                    hidden_units, dilations[i], kernel_size, dropout_rate, gate_mode
                )
                ### <<< Self_EvolveRec-BLOCK-END
                for i in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Masked gated causal features with explicit diagnostic scoring shapes
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        valid = seq != 0
        mask = valid.unsqueeze(-1).to(dtype=self.item_emb.weight.dtype)
        # Apply this padding correction identically to any ungated comparison.
        pos = valid.long().cumsum(dim=1) * valid.long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x * mask) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x) * mask

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        # Preserve aligned scores: [B,L,H] with [B,L], or [B,H] with [B].
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        # A shared candidate vector returns [..., N]. For the ambiguous B == N
        # case, pass shared candidates as [1,N] to request a candidate axis.
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        # Per-feature candidate pools: [...,H] with [...,N]. Shared [1,N]
        # pools also broadcast over the feature batch dimensions.
        if items.ndim == features.ndim:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        raise ValueError(
            f"Unsupported scoring shapes: features={tuple(features.shape)}, "
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
        user_emb = out[0, last[0]]
        cand_emb = self.item_emb(candidates)
        return torch.matmul(cand_emb, user_emb)


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Independently initialize each category with training seed 42
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
        ### >>> Self_EvolveRec-BLOCK-START: Train one explicit gate configuration with unregularized gate biases
        maxlen=MAXLEN,
        gate_mode=GATE_MODE,
    ).to(device)
    gate_biases = []
    regularized_parameters = []
    for name, parameter in model.named_parameters():
        if name.endswith(".gate.bias") or name.endswith(".gate_bias"):
            gate_biases.append(parameter)
        else:
            regularized_parameters.append(parameter)
    parameter_groups = [
        {"params": regularized_parameters, "weight_decay": WEIGHT_DECAY}
    ]
    if gate_biases:
        parameter_groups.append({"params": gate_biases, "weight_decay": 0.0})
    # Preserve Adam and the baseline decay policy for every existing parameter.
    optimizer = torch.optim.Adam(parameter_groups, lr=LR, weight_decay=WEIGHT_DECAY)
    print(
        f"  [{dataset_name}] gate_mode={model.gate_mode}"
        f" parameters={sum(p.numel() for p in model.parameters())}"
        f" max_epochs={NUM_EPOCHS} batches_per_epoch={NUM_BATCHES}"
        f" eval_every={EVAL_EVERY} patience={PATIENCE}"
        f" checkpoint_rule=strict_HR10_improvement_earliest_tie",
        flush=True,
    )
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

    ### >>> Self_EvolveRec-BLOCK-START: Restore the selected checkpoint for consistent diagnostics and deployment
    if best_epoch == 0 or not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"No validation-selected checkpoint was produced for {dataset_name}"
        )
    # Diagnostics must observe the same selected weights loaded by predict.py,
    # rather than the final optimizer iterate after early stopping.
    selected_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(selected_state, strict=True)
    del selected_state
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "kernel_size": KERNEL_SIZE,
                ### >>> Self_EvolveRec-BLOCK-START: Record gate controls and exact single-model training configuration
                "dropout_rate": DROPOUT_RATE,
                "architecture": "nextitnet_positionwise_gate_v2",
                "gate_mode": model.gate_mode,
                "gate_bias_init": 2.0,
                "gate_normalization": "sigmoid_2",
                "gate_initialization": "CPU_fork_rng_zero_weights_bias_2",
                "gate_bias_weight_decay": 0.0,
                "backbone_weight_decay": WEIGHT_DECAY,
                "position_indexing": "nonpadding_cumsum",
                "loss": "stable_softplus_BPR",
                "seed": SEED,
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "batches_per_epoch": NUM_BATCHES,
                "max_epochs": NUM_EPOCHS,
                "actual_epochs": epoch,
                "optimizer_updates": epoch * NUM_BATCHES,
                "best_epoch": best_epoch,
                "eval_every": EVAL_EVERY,
                "patience": PATIENCE,
                "checkpoint_rule": "strict_HR10_improvement_earliest_tie",
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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four predictor dataset routes
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Require one checkpoint with a fully specified residual-gate architecture
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing trained checkpoint or metadata for {ds}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if (
            meta.get("architecture") != "nextitnet_positionwise_gate_v2"
            or meta.get("gate_mode") not in {"fixed_one", "scalar", "bias_only", "contextual"}
            or meta.get("gate_bias_init") != 2.0
            or meta.get("gate_normalization") != "sigmoid_2"
            or meta.get("position_indexing") != "nonpadding_cumsum"
        ):
            raise ValueError(f"Incompatible gated NextItNet metadata for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the exact trained residual-gate configuration
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            gate_mode=meta["gate_mode"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Prevent silent cross-category checkpoint routing
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset route: {dataset_name}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END
    elif _models:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        return [0.0] * len(candidates)

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


