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
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
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
    ### >>> Self_EvolveRec-BLOCK-START: Make local checkpoint validation reproducible without consuming training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Use fixed unique validation negatives and reject impossible candidate counts
            available = itemnum - sum(1 <= item <= itemnum for item in interacted)
            if available < 99:
                raise ValueError(
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
### >>> Self_EvolveRec-BLOCK-START: Preserve the fixed training budget
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


class GRU4RecModel(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Configure learned retention and exact identity control
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        architecture_variant: str = "learned_pre_gate",
        retention_init: float = 0.999,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        ### >>> Self_EvolveRec-BLOCK-START: Independent layer gates conditioned on undropped observed item embeddings
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
        if architecture_variant not in {
            "baseline",
            "identity_control",
            "learned_pre_gate",
        }:
            raise ValueError(f"Unsupported architecture: {architecture_variant}")
        if not 0.0 < retention_init < 1.0:
            raise ValueError("retention_init must lie strictly between zero and one")
        self.architecture_config = {
            "itemnum": itemnum,
            "hidden_units": hidden_units,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "architecture_variant": architecture_variant,
            "retention_init": retention_init,
        }
        # Preserve the CPU RNG stream after constructing the baseline weights.
        # These layers are initialized deterministically below.
        with torch.random.fork_rng(devices=[]):
            self.retention_x = nn.ModuleList(
                [
                    nn.Linear(hidden_units, hidden_units, bias=True)
                    for _ in range(num_layers)
                ]
            )
            self.retention_h = nn.ModuleList(
                [
                    nn.Linear(hidden_units, hidden_units, bias=False)
                    for _ in range(num_layers)
                ]
            )
        for gate_x, gate_h in zip(self.retention_x, self.retention_h):
            nn.init.zeros_(gate_x.weight)
            nn.init.constant_(
                gate_x.bias, math.log(retention_init / (1.0 - retention_init))
            )
            nn.init.zeros_(gate_h.weight)
        # Called once per timestep with layers x batch x hidden tensors.
        self.retention_gate = nn.Sigmoid()
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve PyTorch GRU algebra with explicit padding and shared scoring
    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        raw_emb = self.item_emb(seq)
        emb = self.dropout(raw_emb)
        occupied = seq.ne(0).any(dim=0).nonzero(as_tuple=False).flatten()
        if occupied.numel() == 0:
            return emb * 0.0
        start = int(occupied[0].item())
        variant = self.architecture_config["architecture_variant"]

        if variant == "baseline":
            # Run one packed GRU call for all nonempty histories. Compaction
            # freezes states on padding, including internal padding positions.
            indices = [
                seq[row].ne(0).nonzero(as_tuple=False).flatten()
                for row in range(seq.size(0))
            ]
            active = [row for row, index in enumerate(indices) if index.numel()]
            packed = nn.utils.rnn.pack_sequence(
                [emb[row].index_select(0, indices[row]) for row in active],
                enforce_sorted=False,
            )
            packed_output, _ = self.gru(packed)
            compact, lengths = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            active_position = {row: position for position, row in enumerate(active)}
            rows = []
            for row, index in enumerate(indices):
                output = emb[row] * 0.0
                if row in active_position:
                    position = active_position[row]
                    output = output.index_copy(
                        0, index, compact[position, : int(lengths[position])]
                    )
                rows.append(output)
            return torch.stack(rows, dim=0)

        projected = None
        if variant == "learned_pre_gate":
            projected = [gate(raw_emb[:, start:]) for gate in self.retention_x]
        hidden = [
            emb.new_zeros(seq.size(0), self.hidden_units)
            for _ in range(self.gru.num_layers)
        ]
        # The first layer's input projection is independent of recurrence.
        first_input = torch.nn.functional.linear(
            emb[:, start:], self.gru.weight_ih_l0, self.gru.bias_ih_l0
        )
        outputs = []
        for offset, t in enumerate(range(start, seq.size(1))):
            valid = seq[:, t].ne(0)[:, None]
            if projected is not None:
                gates = self.retention_gate(
                    torch.stack(
                        [
                            projected[layer][:, offset]
                            + self.retention_h[layer](hidden[layer])
                            for layer in range(self.gru.num_layers)
                        ],
                        dim=0,
                    )
                )
            x = emb[:, t]
            for layer in range(self.gru.num_layers):
                previous = hidden[layer]
                recurrent = previous if projected is None else gates[layer] * previous
                input_gates = (
                    first_input[:, offset]
                    if layer == 0
                    else (
                        torch.nn.functional.linear(
                            x,
                            getattr(self.gru, f"weight_ih_l{layer}"),
                            getattr(self.gru, f"bias_ih_l{layer}"),
                        )
                    )
                )
                hidden_gates = torch.nn.functional.linear(
                    recurrent,
                    getattr(self.gru, f"weight_hh_l{layer}"),
                    getattr(self.gru, f"bias_hh_l{layer}"),
                )
                ir, iz, inn = input_gates.chunk(3, dim=-1)
                hr, hz, hn = hidden_gates.chunk(3, dim=-1)
                reset = torch.sigmoid(ir + hr)
                update = torch.sigmoid(iz + hz)
                candidate = torch.tanh(inn + reset * hn)
                proposal = (1.0 - update) * candidate + update * recurrent
                hidden[layer] = torch.where(valid, proposal, previous)
                x = hidden[layer]
                if layer + 1 < self.gru.num_layers:
                    x = torch.nn.functional.dropout(
                        x, p=self.gru.dropout, training=self.training
                    )
            outputs.append(torch.where(valid, x, torch.zeros_like(x)))
        suffix = torch.stack(outputs, dim=1)
        prefix = emb.new_zeros(seq.size(0), start, self.hidden_units)
        return torch.cat((prefix, suffix), dim=1)

    def _readout(self, out: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) == 0:
            return out.new_zeros(seq.size(0), self.hidden_units)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return torch.where(last[:, None].ge(0), features, torch.zeros_like(features))

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self._readout(self.log2feats(seq), seq)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        item_features = self.item_emb(items)
        if items.ndim == 1:
            # A shared catalog/candidate list for any leading feature dimensions.
            return torch.matmul(features, item_features.transpose(0, 1))
        if item_features.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * item_features).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    @torch.inference_mode()
    def validate_execution(self) -> None:
        device = self.item_emb.weight.device
        a, b = 1, min(2, self.itemnum)
        seq = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, a],
                [0, a, 0, b],
                [a, b, a, b],
                [0, a, 0, 0],
            ],
            dtype=torch.long,
            device=device,
        )
        expected_last = torch.tensor([-1, 3, 3, 3, 1], device=device)
        positions = torch.arange(4, device=device).expand_as(seq)
        actual_last = positions.masked_fill(seq.eq(0), -1).amax(dim=1)
        torch.testing.assert_close(actual_last, expected_last, rtol=0, atol=0)
        was_training = self.training
        variant = self.architecture_config["architecture_variant"]
        self.eval()
        try:
            # Both paths use exactly the same embedding and recurrent weights.
            self.architecture_config["architecture_variant"] = "baseline"
            reference = self.log2feats(seq)
            self.architecture_config["architecture_variant"] = "identity_control"
            control = self.log2feats(seq)
            torch.testing.assert_close(control, reference, rtol=2e-5, atol=2e-6)
            self.architecture_config["architecture_variant"] = variant
            features = self.log2feats(seq)
            independent = torch.stack(
                [
                    (
                        features[row, expected_last[row]]
                        if expected_last[row] >= 0
                        else features.new_zeros(self.hidden_units)
                    )
                    for row in range(seq.size(0))
                ]
            )
            torch.testing.assert_close(
                self._readout(features, seq), independent, rtol=0, atol=0
            )
            encoded = self.encode(seq)
            padded = torch.cat((torch.zeros_like(seq[:, :2]), seq), dim=1)
            torch.testing.assert_close(
                self.encode(padded), encoded, rtol=2e-5, atol=2e-6
            )
            for row in range(seq.size(0)):
                torch.testing.assert_close(
                    self.encode(seq[row : row + 1]),
                    encoded[row : row + 1],
                    rtol=2e-5,
                    atol=2e-6,
                )
            empty = seq[:, :0]
            torch.testing.assert_close(
                self.encode(empty), torch.zeros_like(encoded), rtol=0, atol=0
            )
            candidates = torch.tensor([a, b], device=device)
            torch.testing.assert_close(
                self.score_candidates(seq, candidates),
                self.predict_score(encoded, candidates),
                rtol=0,
                atol=0,
            )
            expanded = candidates.expand(seq.size(0), seq.size(1), -1)
            torch.testing.assert_close(
                self.predict_score(features, expanded),
                self.predict_score(features, candidates),
                rtol=2e-5,
                atol=2e-6,
            )
            # Check aligned training targets separately from shared candidates.
            aligned = seq.clamp_min(1)
            torch.testing.assert_close(
                self.predict_score(features, aligned),
                (features * self.item_emb(aligned)).sum(dim=-1),
                rtol=0,
                atol=0,
            )
            per_row_candidates = candidates.expand(seq.size(0), -1)
            torch.testing.assert_close(
                self.score_candidates(seq, per_row_candidates),
                self.predict_score(encoded, candidates),
                rtol=2e-5,
                atol=2e-6,
            )
            torch.testing.assert_close(
                self.score_candidates(seq[1:2], candidates),
                self.predict_score(self.encode(seq[1:2]), candidates)[0],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                self.log2feats(seq[:1]),
                torch.zeros_like(features[:1]),
                rtol=0,
                atol=0,
            )
        finally:
            self.architecture_config["architecture_variant"] = variant
            self.train(was_training)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Route candidate scoring through the shared feature scoring implementation
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        scores = self.predict_score(self.encode(seq), candidates)
        # Preserve the existing single-history predictor return contract.
        return scores[0] if seq.size(0) == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Select one explicit architecture per trial and verify execution before training
    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        architecture_variant=os.environ.get("GRU4REC_ARCHITECTURE", "learned_pre_gate"),
        retention_init=0.999,
    ).to(device)
    model.validate_execution()
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
                ### >>> Self_EvolveRec-BLOCK-START: Save one selected model with its architecture configuration
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "architecture_config": model.architecture_config,
                    },
                    ckpt_path,
                )
                ### <<< Self_EvolveRec-BLOCK-END
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    ### >>> Self_EvolveRec-BLOCK-START: Preserve checkpoint format in the fallback save path
    if not os.path.exists(ckpt_path):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture_config": model.architecture_config,
            },
            ckpt_path,
        )
    ### <<< Self_EvolveRec-BLOCK-END

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                ### >>> Self_EvolveRec-BLOCK-START: Record inherited training settings and retention architecture
                "dropout_rate": DROPOUT_RATE,
                "architecture_config": model.architecture_config,
                "training_config": {
                    "seed": SEED,
                    "max_epochs": NUM_EPOCHS,
                    "batches_per_epoch": NUM_BATCHES,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "optimizer": "Adam",
                    "loss": "baseline_sampled_bpr",
                    "eval_every": EVAL_EVERY,
                    "patience": PATIENCE,
                    "best_epoch": best_epoch,
                    "gradient_clip_norm": 5.0,
                },
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
        f"gru4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


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
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the single selected retention-gated checkpoint exactly
        checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=True)
        config = checkpoint["architecture_config"]
        if config != meta["architecture_config"]:
            raise RuntimeError(f"Checkpoint/metadata architecture mismatch for {ds}")
        model = ModelClass(**config).to(_device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.validate_execution()
        model.eval()
        ### <<< Self_EvolveRec-BLOCK-END
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing category checkpoints instead of silently scoring with another category
    _ensure_loaded()
    if dataset_name is not None:
        if dataset_name not in _models:
            raise RuntimeError(f"No trained GRU4Rec checkpoint for {dataset_name!r}")
        ds = dataset_name
    elif len(_models) == 1:
        ds = next(iter(_models))
    else:
        raise ValueError("dataset_name is required when routing multiple category models")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset GRU4Rec training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


