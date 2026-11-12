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


### >>> Self_EvolveRec-BLOCK-START: Stable BPR restricted to valid sequence-target pairs
def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    pos_scores, neg_scores = model(seq, pos, neg)
    valid = (seq != 0) & (pos != 0) & (neg != 0)
    if not bool(valid.any()):
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
### >>> Self_EvolveRec-BLOCK-START: One fixed tap configuration per independent host trial
TAP_MODE = os.environ.get("NEXTITNET_TAP_MODE", "adaptive")
TAP_EPS = 1e-5
if TAP_MODE not in ("ones", "static", "adaptive"):
    raise ValueError("NEXTITNET_TAP_MODE must be ones, static or adaptive")
### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Identity-initialized causal tap gates with shared gate projections and independent unit-gate checks
class CausalTapGateConv(nn.Module):
    def __init__(self, conv, mode="adaptive", eps=1e-5):
        super().__init__()
        if not isinstance(conv, nn.Conv1d):
            raise TypeError("Expected Conv1d")
        if (
            conv.stride != (1,)
            or conv.groups != 1
            or conv.padding != (0,)
            or conv.padding_mode != "zeros"
        ):
            raise ValueError(
                "Expected stride-one ungrouped explicitly padded convolution"
            )
        if mode not in ("ones", "static", "adaptive") or eps <= 0:
            raise ValueError("Invalid tap gate configuration")
        self.conv = conv
        self.mode = mode
        self.eps = float(eps)
        self.kernel_size = conv.kernel_size[0]
        self.dilation = conv.dilation[0]
        historical = self.kernel_size - 1
        if mode == "adaptive":
            self.current_gate = nn.Parameter(
                conv.weight.new_zeros(historical, conv.in_channels)
            )
            self.history_gate = nn.Parameter(
                conv.weight.new_zeros(historical, conv.in_channels)
            )
        else:
            self.register_parameter("current_gate", None)
            self.register_parameter("history_gate", None)
        if mode in ("static", "adaptive"):
            self.gate_bias = nn.Parameter(conv.weight.new_zeros(historical))
        else:
            self.register_parameter("gate_bias", None)

    def forward(self, h, sequence, force_unit_gates=False):
        if h.ndim != 3 or tuple(h.shape[:2]) != tuple(sequence.shape):
            raise ValueError("Expected B x L x C states and B x L sequence")
        if h.shape[-1] != self.conv.in_channels or h.shape[1] < 1:
            raise ValueError("Invalid convolution input dimensions")
        valid = sequence.ne(0)
        x = h.masked_fill(~valid.unsqueeze(-1), 0)
        padding = self.dilation * (self.kernel_size - 1)
        if self.mode == "ones" and not force_unit_gates:
            result = self.conv(
                nn.functional.pad(x.transpose(1, 2), (padding, 0))
            ).transpose(1, 2)
            return result.masked_fill(~valid.unsqueeze(-1), 0)

        adaptive = self.mode == "adaptive" and not force_unit_gates
        current_logits = history_logits = None
        if adaptive and self.kernel_size > 1:
            # Affine-free channel normalization commutes with temporal shifting.
            # Compute it once, then project all historical gates together.
            normalized = nn.functional.layer_norm(x, (x.shape[-1],), eps=self.eps)
            current_logits = nn.functional.linear(normalized, self.current_gate)
            history_logits = nn.functional.linear(normalized, self.history_gate)

        result = x.new_zeros(x.shape[0], x.shape[1], self.conv.out_channels)
        for tap in range(self.kernel_size):
            lag = (self.kernel_size - 1 - tap) * self.dilation
            if lag == 0:
                contribution = x
            else:
                if lag >= x.shape[1]:
                    continue
                past = nn.functional.pad(x[:, :-lag, :], (0, 0, lag, 0))
                if force_unit_gates:
                    # Exercise the actual tap decomposition, rather than calling
                    # the reference Conv1d and making equivalence checks tautological.
                    contribution = past
                else:
                    past_valid = nn.functional.pad(
                        valid[:, :-lag], (lag, 0), value=False
                    )
                    logits = self.gate_bias[tap]
                    if adaptive:
                        historical = nn.functional.pad(
                            history_logits[:, :-lag, tap], (lag, 0)
                        )
                        logits = current_logits[:, :, tap] + historical + logits
                    gate = 2.0 * torch.sigmoid(logits)
                    gate = gate * (valid & past_valid).to(x.dtype)
                    contribution = past * gate.unsqueeze(-1)
            result = result + nn.functional.linear(
                contribution, self.conv.weight[:, :, tap], bias=None
            )
        if self.conv.bias is not None:
            result = result + self.conv.bias
        return result.masked_fill(~valid.unsqueeze(-1), 0)


### <<< Self_EvolveRec-BLOCK-END


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

    ### >>> Self_EvolveRec-BLOCK-START: Apply tap gates without duplicate causal padding
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = mask.squeeze(-1).ne(0)
        residual = x * mask
        y = self.norm1(residual) * mask
        y = self.conv1(y, sequence) * mask
        y = self.dropout(self.activation(y)) * mask
        y = self.norm2(y) * mask
        y = self.conv2(y, sequence) * mask
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
        ### >>> Self_EvolveRec-BLOCK-START: Explicit checkpointed tap configuration
        maxlen: int = 128,
        tap_mode: str = "adaptive",
        tap_eps: float = 1e-5,
        ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Restore final-only backbone for the isolated tap-gating experiment
        if num_blocks < 1:
            raise ValueError("NextItNet requires at least one residual block")
        if tap_mode not in ("ones", "static", "adaptive") or tap_eps <= 0:
            raise ValueError("Invalid tap configuration")
        self.tap_mode = tap_mode
        self.tap_eps = float(tap_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        ### <<< Self_EvolveRec-BLOCK-END
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()
        ### >>> Self_EvolveRec-BLOCK-START: Wrap initialized convolutions before optimizer creation without random draws
        for block in self.blocks:
            block.conv1 = CausalTapGateConv(
                block.conv1, mode=self.tap_mode, eps=self.tap_eps
            )
            block.conv2 = CausalTapGateConv(
                block.conv2, mode=self.tap_mode, eps=self.tap_eps
            )
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Final-only causal encoding and shape-compatible tied scoring
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        valid = seq != 0
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        pos = valid.long().cumsum(dim=1).clamp_max(self.maxlen)
        pos = pos * valid.long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x * mask) * mask
        for block in self.blocks:
            x = block(x, mask) * mask
        return self.norm(x) * mask

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

    ### >>> Self_EvolveRec-BLOCK-START: Score every history at its final nonpadding position
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.size(1) == 0:
            raise ValueError("seq must have shape B x L with L > 0")
        if candidates.ndim not in (1, 2):
            raise ValueError("candidates must have shape N or B x N")
        if candidates.ndim == 2 and candidates.size(0) != seq.size(0):
            raise ValueError("Candidate and history batch sizes must match")
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1)
        rows = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[rows, last.clamp_min(0)]
        user_emb = user_emb * (last >= 0).unsqueeze(-1)
        scores = self.predict_score(user_emb, candidates)
        expected = (seq.size(0), candidates.size(-1))
        if tuple(scores.shape) != expected:
            raise RuntimeError(
                f"Expected candidate scores {expected}, got {tuple(scores.shape)}"
            )
        # Preserve the deployed single-history list-of-scores contract.
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
        ### >>> Self_EvolveRec-BLOCK-START: Invoke the selected tap configuration during training
        maxlen=MAXLEN,
        tap_mode=TAP_MODE,
        tap_eps=TAP_EPS,
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)
    ### >>> Self_EvolveRec-BLOCK-START: Check initialized tap outputs and gradients without consuming sampler randomness
    check_report = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        for kernel, dilation, width in ((1, 1, 3), (3, 1, 7), (3, 4, 3), (3, 2, 9)):
            conv = nn.Conv1d(
                4, 5, kernel, dilation=dilation, padding=0, bias=True
            ).double()
            adapter = CausalTapGateConv(conv, mode=TAP_MODE, eps=TAP_EPS)
            sequence = torch.ones((3, width), dtype=torch.long)
            sequence[0, : width // 2] = 0
            sequence[1, 1::2] = 0
            sequence[2] = 0
            h = torch.randn(3, width, 4, dtype=torch.float64, requires_grad=True)
            valid = sequence.ne(0).unsqueeze(-1)
            masked = h.masked_fill(~valid, 0)
            reference = (
                conv(
                    nn.functional.pad(
                        masked.transpose(1, 2), ((kernel - 1) * dilation, 0)
                    )
                )
                .transpose(1, 2)
                .masked_fill(~valid, 0)
            )
            upstream = torch.randn_like(reference)
            parameters = (h, conv.weight, conv.bias)
            reference_gradients = torch.autograd.grad(reference, parameters, upstream)
            for label, forced in (("initialized", False), ("forced_unit", True)):
                actual = adapter(h, sequence, force_unit_gates=forced)
                gradients = torch.autograd.grad(actual, parameters, upstream)
                comparisons = [(actual, reference)] + list(
                    zip(gradients, reference_gradients)
                )
                errors = [
                    float((a.detach() - b.detach()).abs().max()) for a, b in comparisons
                ]
                if not all(
                    torch.allclose(a, b, atol=1e-10, rtol=1e-8) for a, b in comparisons
                ):
                    raise RuntimeError(
                        f"Tap output/gradient check failed: "
                        f"kernel={kernel}, dilation={dilation}, width={width}, "
                        f"mode={TAP_MODE}, path={label}, errors={errors}"
                    )
                check_report[f"{kernel}:{dilation}:{width}:{label}"] = errors
    print(
        f"  [{dataset_name}] CPU float64 tap checks "
        f"(output,input,weight,bias max errors): {check_report}",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ### <<< Self_EvolveRec-BLOCK-END

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
                ### >>> Self_EvolveRec-BLOCK-START: Persist the single gated-tap architecture
                "dropout_rate": DROPOUT_RATE,
                "architecture": "nextitnet_causal_tap_gate_v1",
                "prefix_conditioning": False,
                "depth_correction": False,
                "position_encoding": "observed_prefix_count",
                "tap_mode": model.tap_mode,
                "tap_eps": model.tap_eps,
                "tap_implementation": "per_tap_linear",
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
        ### >>> Self_EvolveRec-BLOCK-START: Reconstruct exactly one matching gated-tap checkpoint per category
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if (
            meta.get("architecture") != "nextitnet_causal_tap_gate_v1"
            or meta.get("prefix_conditioning") is not False
            or meta.get("depth_correction") is not False
            or meta.get("position_encoding") != "observed_prefix_count"
            or meta.get("tap_mode") not in ("ones", "static", "adaptive")
            or meta.get("tap_eps") != 1e-5
            or meta.get("tap_implementation") != "per_tap_linear"
        ):
            raise RuntimeError(f"Incompatible NextItNet architecture for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            dropout_rate=0.0,
            ### >>> Self_EvolveRec-BLOCK-START: Use saved tap settings rather than inference environment defaults
            maxlen=meta["maxlen"],
            tap_mode=meta["tap_mode"],
            tap_eps=meta["tap_eps"],
            ### <<< Self_EvolveRec-BLOCK-END
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


### >>> Self_EvolveRec-BLOCK-START: Record the isolated four-category experiment before fitting
def main() -> None:
    if DATASETS != DEFAULT_DATASETS:
        raise ValueError("Retain all four required categories in their supplied order")
    if EVAL_EVERY < 1 or PATIENCE < 1:
        raise ValueError("Evaluation interval and patience must be positive")
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    manifest = {
        "family": "nextitnet",
        "architecture": "nextitnet_causal_tap_gate_v1",
        "datasets": DATASETS,
        "training_seed": SEED,
        "preprocessing": "unchanged host splits, preprocessing seed 0",
        "tap_mode": TAP_MODE,
        "tap_eps": TAP_EPS,
        "tap_implementation": "per_tap_linear",
        "gate_initial_value": 1.0,
        "current_tap": "unchanged",
        "depth_correction": False,
        "prefix_conditioning": False,
        "maxlen": MAXLEN,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "kernel_size": KERNEL_SIZE,
        "dilations": ([1, 2, 4, 8] * math.ceil(NUM_BLOCKS / 4))[:NUM_BLOCKS],
        "dropout": DROPOUT_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "unchanged position-weighted BPR softplus",
        "maximum_epochs": NUM_EPOCHS,
        "batches_per_epoch": NUM_BATCHES,
        "checkpoint_epochs": [
            epoch
            for epoch in range(1, NUM_EPOCHS + 1)
            if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS
        ],
        "patience_epochs": PATIENCE,
        "checkpoint_tie_break": "earliest epoch",
        "selection_metric": "four-category macro validation HR@10",
        "models_per_category": 1,
        "configuration_scope": "one configuration per host trial; no internal sweep",
        "control_modes": ["ones", "static", "adaptive"],
        "configuration_tie_order": ["ones", "static", "adaptive"],
        "shared_remaining_gpu_seconds": None,
        "budget_accounting": "host-managed; remaining balance unavailable",
        "execution_status": "configuration recorded; no measurements asserted",
    }
    path = os.path.join(working_dir, "experiment_manifest.json")
    if os.path.exists(path):
        with open(path) as handle:
            if json.load(handle) != manifest:
                raise RuntimeError("Configuration differs from existing trial manifest")
    else:
        with open(path, "x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### <<< Self_EvolveRec-BLOCK-END
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


