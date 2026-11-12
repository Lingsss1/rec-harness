"""Per-dataset GRU4Rec baseline for Amazon Reviews.

Trains one independent GRU4Rec checkpoint per dataset while sharing one default
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
### >>> Self_EvolveRec-BLOCK-START: Preserve the fixed four-category task and training seed
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
    # DEBUG: A catalog-saturated user cannot supply an unseen negative. Exclude
    # such users before sampling instead of randomly aborting an entire category.
    if itemnum <= 0 or batch_size <= 0 or maxlen <= 0:
        raise ValueError("itemnum, batch_size and maxlen must be positive")
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2
        and len({item for item in items if 1 <= item <= itemnum}) < itemnum
    ]
    if not users:
        raise RuntimeError(
            "no training users with at least two interactions and an unseen negative"
        )
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
        # DEBUG: Prevent an infinite rejection loop when no negative item exists.
        if sum(1 <= item <= itemnum for item in interacted) >= itemnum:
            raise RuntimeError(
                f"Training user {user} has no unseen items available for negative sampling"
            )
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


# DEBUG: Keep validation implementation separate so mode restoration also
# occurs when candidate construction or model scoring raises an exception.
def _evaluate_val_impl(
    model: nn.Module,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Make checkpoint validation reproducible without consuming training randomness
    rng = random.Random(SEED)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    ### >>> Self_EvolveRec-BLOCK-START: Preserve the model mode around validation
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    was_training = model.training
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            ### >>> Self_EvolveRec-BLOCK-START: Use reproducible unique validation negatives and reject impossible candidate sets
            interacted = set(history) | {target, 0}
            available = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available < 99:
                raise RuntimeError(
                    f"Validation user {user} has only {available} unseen negatives; "
                    "99 unique negatives are required"
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
    ### >>> Self_EvolveRec-BLOCK-START: Restore the mode present before validation
    n = float(len(val_users))
    model.train(was_training)
    ### <<< Self_EvolveRec-BLOCK-END
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


def evaluate_val(
    model: nn.Module,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int,
    num_users: int = 200,
) -> dict[str, float]:
    # DEBUG: Restore the caller's mode on every exit without suppressing errors.
    was_training = model.training
    try:
        return _evaluate_val_impl(
            model, user_train, user_valid, itemnum, device, maxlen, num_users
        )
    finally:
        model.train(was_training)


def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Enforce the supplied training budget limits
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))


### >>> Self_EvolveRec-BLOCK-START: Jointly trained item-and-state soft retention before GRU updates
class SoftRetentionGRU(nn.GRU):
    """Preserve PyTorch GRU equations, parameters, and padding updates."""

    def __init__(
        self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0
    ):
        super().__init__(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.retention_input = nn.ModuleList(
            [
                nn.Linear(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )
        self.retention_hidden = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)]
        )
        for projection, recurrent in zip(self.retention_input, self.retention_hidden):
            nn.init.zeros_(projection.weight)
            nn.init.constant_(projection.bias, math.log(99.0))
            nn.init.zeros_(recurrent.weight)

    def forward(self, inputs, hx=None, exact_identity_gate=False):
        if inputs.ndim != 3 or not self.batch_first:
            raise ValueError("SoftRetentionGRU expects batch-first B x L x H inputs")
        batch_size, length, _ = inputs.shape
        if length == 0:
            raise ValueError("GRU input must contain at least one timestep")
        if hx is None:
            hx = inputs.new_zeros(self.num_layers, batch_size, self.hidden_size)
        layer_inputs = inputs
        final_states = []
        for layer in range(self.num_layers):
            # Hoist input projections out of the sequential recurrence.
            projected = torch.nn.functional.linear(
                layer_inputs,
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"bias_ih_l{layer}"),
            )
            gate_inputs = (
                None
                if exact_identity_gate
                else self.retention_input[layer](layer_inputs)
            )
            previous = hx[layer]
            outputs = []
            for timestep in range(length):
                if exact_identity_gate:
                    retained = previous
                else:
                    retention = torch.sigmoid(
                        gate_inputs[:, timestep]
                        + self.retention_hidden[layer](previous)
                    )
                    retained = retention * previous
                recurrent = torch.nn.functional.linear(
                    retained,
                    getattr(self, f"weight_hh_l{layer}"),
                    getattr(self, f"bias_hh_l{layer}"),
                )
                input_reset, input_update, input_new = projected[:, timestep].chunk(
                    3, -1
                )
                hidden_reset, hidden_update, hidden_new = recurrent.chunk(3, -1)
                reset = torch.sigmoid(input_reset + hidden_reset)
                update = torch.sigmoid(input_update + hidden_update)
                candidate = torch.tanh(input_new + reset * hidden_new)
                previous = candidate + update * (retained - candidate)
                # The supplied baseline updates states even at padding positions.
                outputs.append(previous)
            layer_inputs = torch.stack(outputs, dim=1)
            final_states.append(previous)
            if layer + 1 < self.num_layers and self.dropout:
                layer_inputs = torch.nn.functional.dropout(
                    layer_inputs,
                    p=self.dropout,
                    training=self.training,
                )
        return layer_inputs, torch.stack(final_states, dim=0)


def _verify_retention_identity():
    """Disposable CPU control; does not use data or consume training RNG state."""
    with torch.random.fork_rng(devices=[]):
        baseline = nn.GRU(4, 4, num_layers=2, batch_first=True, dropout=0.0)
        gated = SoftRetentionGRU(4, 4, num_layers=2, dropout=0.0)
        incompatible = gated.load_state_dict(baseline.state_dict(), strict=False)
        assert not incompatible.unexpected_keys
        assert all(name.startswith("retention_") for name in incompatible.missing_keys)
        baseline.eval()
        gated.eval()
        sequence = torch.tensor([[0, 0, 1, 2], [3, 1, 2, 3], [0, 0, 0, 0]])
        embedding = torch.randn(4, 4)
        embedding[0].zero_()
        x_base = embedding[sequence].clone().requires_grad_(True)
        x_gate = x_base.detach().clone().requires_grad_(True)
        out_base, state_base = baseline(x_base)
        out_gate, state_gate = gated(x_gate, exact_identity_gate=True)

        def close(actual, expected):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

        close(out_gate, out_base)
        close(state_gate, state_base)
        positions = torch.arange(sequence.size(1)).expand_as(sequence)
        last = positions.masked_fill(sequence == 0, -1).max(1).values.clamp(min=0)
        rows = torch.arange(sequence.size(0))
        close(out_gate[rows, last], out_base[rows, last])
        candidates = torch.randn(5, 4)
        close(out_gate[rows, last] @ candidates.T, out_base[rows, last] @ candidates.T)
        positive, negative = torch.randn_like(out_base), torch.randn_like(out_base)
        mask = sequence.ne(0).float()

        def loss(outputs):
            margin = (outputs * (positive - negative)).sum(-1)
            terms = -torch.log(torch.sigmoid(margin) + 1e-8)
            return (terms * mask).sum() / mask.sum().clamp(min=1.0)

        loss_base, loss_gate = loss(out_base), loss(out_gate)
        close(loss_gate, loss_base)
        loss_base.backward()
        loss_gate.backward()
        close(x_gate.grad, x_base.grad)
        gated_parameters = dict(gated.named_parameters())
        for name, parameter in baseline.named_parameters():
            close(gated_parameters[name].grad, parameter.grad)


### <<< Self_EvolveRec-BLOCK-END
class GRU4RecModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        dropout = dropout_rate if num_layers > 1 else 0.0
        ### >>> Self_EvolveRec-BLOCK-START: Apply soft retention in every recurrent layer
        self.gru = SoftRetentionGRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        ### <<< Self_EvolveRec-BLOCK-END
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Share gated features across training inference and family diagnostics
    def log2feats(
        self, sequence: torch.Tensor, exact_identity_gate=False
    ) -> torch.Tensor:
        # DEBUG: Accept NumPy histories from family diagnostic adapters as well as tensors.
        sequence = torch.as_tensor(
            sequence, dtype=torch.long, device=self.item_emb.weight.device
        )
        if sequence.ndim != 2 or sequence.size(1) == 0:
            raise ValueError("log2feats expects a nonempty B x L sequence")
        emb = self.dropout(self.item_emb(sequence))
        out, _ = self.gru(emb, exact_identity_gate=exact_identity_gate)
        return out

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        # DEBUG: Normalize adapter item inputs and preserve aligned training scoring.
        # Also support shared candidates and explicit per-position candidate axes.
        # The generated predictor imports this same model class from train.py.
        items = torch.as_tensor(
            items, dtype=torch.long, device=self.item_emb.weight.device
        )
        # DEBUG: One-dimensional IDs always denote shared candidates. Check this
        # before aligned scoring so C == B cannot silently drop the candidate axis.
        # Training IDs remain B x L and retain their aligned scoring behavior.
        if items.ndim == 1:
            return torch.matmul(features, self.item_emb(items).transpose(0, 1))
        if items.shape == features.shape[:-1]:
            return (features * self.item_emb(items)).sum(dim=-1)
        if items.ndim == features.ndim and items.shape[:-1] == features.shape[:-1]:
            return (features.unsqueeze(-2) * self.item_emb(items)).sum(dim=-1)
        raise ValueError(
            "predict_score expects aligned item IDs, shared candidates, "
            "or one per-position candidate axis: "
            f"features={tuple(features.shape)}, items={tuple(items.shape)}"
        )

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        # DEBUG: Normalize the readout input too. Conversion inside log2feats
        # leaves this caller's NumPy input or CPU tensor unchanged.
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        out = self.log2feats(seq)
        # Preserve the final nonpadding index and baseline index-zero empty fallback.
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    ### <<< Self_EvolveRec-BLOCK-END

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        # DEBUG: Normalize candidates through the same architecture-aware scorer
        # used by diagnostics; Python lists and CPU IDs now work on CUDA too.
        # Preserve the existing single-history, one-score-per-candidate contract.
        features = self.encode(seq)
        if features.size(0) != 1:
            raise ValueError("score_candidates expects exactly one history")
        candidates = torch.as_tensor(
            candidates, dtype=torch.long, device=self.item_emb.weight.device
        )
        if candidates.ndim != 1:
            raise ValueError(
                "score_candidates expects a one-dimensional candidate list"
            )
        return self.predict_score(features, candidates)[0]


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Independently initialize each category with the fixed training seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    ### <<< Self_EvolveRec-BLOCK-END
    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

    ### >>> Self_EvolveRec-BLOCK-START: Verify selected-checkpoint serialization using the same category model
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    selected_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(selected_state, strict=True)
    model.eval()
    probe_user = next(iter(user_train))
    probe_sequence = _history_to_tensor(user_train[probe_user], MAXLEN, device)
    probe_items = torch.arange(1, min(itemnum, 100) + 1, device=device)
    with torch.no_grad():
        expected_scores = model.score_candidates(probe_sequence, probe_items).clone()
        # Reload into the same module; no additional fitted model is created.
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True), strict=True
        )
        reloaded_scores = model.score_candidates(probe_sequence, probe_items)
        torch.testing.assert_close(
            reloaded_scores, expected_scores, atol=1e-6, rtol=1e-5
        )
    del selected_state
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                ### >>> Self_EvolveRec-BLOCK-START: Record the complete retention architecture contract
                "dropout_rate": DROPOUT_RATE,
                "architecture": "gru4rec_soft_retention_v1",
                "retention_initial_bias": math.log(99.0),
                "padding_behavior": "update",
                "empty_readout_index": 0,
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
    script = r'''"""Per-dataset GRU4Rec predict function."""
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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four prediction routes
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END
_models = {}
_metas = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"gru4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


def _ensure_loaded():
    global _device
    # DEBUG: A partially populated cache does not mean all four routes loaded.
    if all(ds in _models and ds in _metas for ds in DATASETS):
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    # DEBUG: Publish the cache only after every required category loads correctly.
    loaded_models = {}
    loaded_metas = {}
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Require a checkpoint for every fixed category route
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing required category artifacts: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Reload the identical single gated GRU architecture
        if meta.get("architecture") != "gru4rec_soft_retention_v1":
            raise RuntimeError(f"Unexpected architecture for {ds}")
        if meta.get("padding_behavior") != "update":
            raise RuntimeError(f"Unexpected padding behavior for {ds}")
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            dropout_rate=meta["dropout_rate"],
        ).to(_device)
        ### <<< Self_EvolveRec-BLOCK-END
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        loaded_models[ds] = model
        loaded_metas[ds] = meta
    # DEBUG: Failed initialization cannot leave apparently usable partial routing.
    _models.clear()
    _metas.clear()
    _models.update(loaded_models)
    _metas.update(loaded_metas)


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes instead of scoring with another category
    ds = DEFAULT_DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError(f"Unknown or unavailable dataset route: {ds!r}")
    model = _models[ds]
    meta = _metas[ds]
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
    ### >>> Self_EvolveRec-BLOCK-START: Verify exact-identity recurrence before fitting category models
    _verify_retention_identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset soft-retention GRU4Rec training on {device}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


