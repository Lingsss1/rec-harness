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


### >>> Self_EvolveRec-BLOCK-START: Cache immutable training exclusions without changing sampling draws
_SAMPLER_CACHE = {}


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    if itemnum < 1 or batch_size < 1 or maxlen < 1:
        raise ValueError("itemnum, batch_size and maxlen must be positive")
    if (
        _SAMPLER_CACHE.get("source") is not user_train
        or _SAMPLER_CACHE.get("itemnum") != itemnum
    ):
        users = [u for u, items in user_train.items() if len(items) >= 2]
        if not users:
            raise RuntimeError("no users with at least two training interactions")
        exclusions = {}
        for user in users:
            observed = set(user_train[user])
            if any(item < 1 or item > itemnum for item in observed):
                raise ValueError(f"Invalid training item ID for user {user}")
            if len(observed) >= itemnum:
                raise ValueError(f"Training user {user} has no eligible negative")
            exclusions[user] = observed
        _SAMPLER_CACHE.clear()
        _SAMPLER_CACHE.update(
            source=user_train, itemnum=itemnum, users=users, exclusions=exclusions
        )

    users = _SAMPLER_CACHE["users"]
    batch_seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    batch_pos = np.zeros_like(batch_seq)
    batch_neg = np.zeros_like(batch_seq)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        seq, pos, neg = batch_seq[row], batch_pos[row], batch_neg[row]
        nxt = history[-1]
        idx = maxlen - 1
        interacted = _SAMPLER_CACHE["exclusions"][user]
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
    return batch_seq, batch_pos, batch_neg


### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Explicit retention architecture configuration
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        architecture_variant: str = "item_carry_gate_v1",
        retention_init: float = 0.99,
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
        ### >>> Self_EvolveRec-BLOCK-START: Item-only retention on layer-zero carry with preserved baseline initialization
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
        if architecture_variant != "item_carry_gate_v1":
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
        # Construct on CPU without advancing sampling or dropout RNG streams.
        # This adds exactly H*(input_size+1) jointly trained parameters.
        with torch.random.fork_rng(devices=[]):
            self.retention_x = nn.Linear(hidden_units, hidden_units, bias=True)
        nn.init.zeros_(self.retention_x.weight)
        nn.init.constant_(
            self.retention_x.bias, math.log(retention_init / (1.0 - retention_init))
        )
        # Hooks observe B x suffix_length x H gates. Padding must be excluded
        # from diagnostic summaries using the corresponding sequence mask.
        self.retention_gate = nn.Sigmoid()
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: First-layer carry-only recurrence with masked padding and shared final readout
    def _gru_step(self, input_gates, previous, layer, retention=1.0):
        recurrent = torch.nn.functional.linear(
            previous,
            getattr(self.gru, f"weight_hh_l{layer}"),
            getattr(self.gru, f"bias_hh_l{layer}"),
        )
        i_r, i_z, i_n = input_gates.chunk(3, dim=-1)
        h_r, h_z, h_n = recurrent.chunk(3, dim=-1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        # PyTorch reset-after-linear: reset also multiplies candidate bias.
        candidate = torch.tanh(i_n + reset * h_n)
        return update * retention * previous + (1.0 - update) * candidate

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError("sequence must have shape B x L")
        emb = self.dropout(self.item_emb(seq))
        occupied = seq.ne(0).any(dim=0).nonzero(as_tuple=False).flatten()
        if occupied.numel() == 0:
            return emb * 0.0
        start = int(occupied[0].item())
        inputs = emb[:, start:]
        # Compute item-dependent projections once across the whole suffix.
        retention = self.retention_gate(self.retention_x(inputs))
        input_gates = torch.nn.functional.linear(
            inputs, self.gru.weight_ih_l0, self.gru.bias_ih_l0
        )
        hidden = [
            emb.new_zeros(seq.size(0), self.hidden_units)
            for _ in range(self.gru.num_layers)
        ]
        outputs = []
        for offset, t in enumerate(range(start, seq.size(1))):
            valid = seq[:, t].ne(0).unsqueeze(-1)
            proposed = self._gru_step(
                input_gates[:, offset], hidden[0], 0, retention[:, offset]
            )
            hidden[0] = torch.where(valid, proposed, hidden[0])
            for layer in range(1, self.gru.num_layers):
                # Preserve ordinary higher-layer recurrence and interlayer
                # dropout; no additional retention gate is applied here.
                x = torch.nn.functional.dropout(
                    hidden[layer - 1], p=self.gru.dropout, training=self.training
                )
                gates = torch.nn.functional.linear(
                    x,
                    getattr(self.gru, f"weight_ih_l{layer}"),
                    getattr(self.gru, f"bias_ih_l{layer}"),
                )
                proposed = self._gru_step(gates, hidden[layer], layer)
                hidden[layer] = torch.where(valid, proposed, hidden[layer])
            outputs.append(torch.where(valid, hidden[-1], torch.zeros_like(hidden[-1])))
        suffix = torch.stack(outputs, dim=1)
        prefix = emb.new_zeros(seq.size(0), start, self.hidden_units)
        return torch.cat((prefix, suffix), dim=1)

    def _readout(self, out: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) == 0:
            return out.new_zeros(seq.size(0), self.hidden_units)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1)
        features = out[torch.arange(seq.size(0), device=seq.device), last.clamp_min(0)]
        return torch.where(last[:, None].ge(0), features, torch.zeros_like(features))

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self._readout(self.log2feats(seq), seq)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        item_features = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, item_features.transpose(0, 1))
        if item_features.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * item_features).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def validate_carry_equivalence(self):
        # Implementation check only: the reference is never trained or deployed.
        # Check each layer's ordinary equations and gradients with retention=1.
        reference_weight = self.gru.weight_ih_l0
        device, dtype = reference_weight.device, reference_weight.dtype
        with torch.random.fork_rng(devices=[]):
            reference = nn.GRU(self.hidden_units, self.hidden_units, batch_first=True)
        # cuDNN requires training-mode RNN execution to retain the workspace
        # needed by backward. This single-layer reference has zero dropout,
        # so training mode preserves its deterministic mathematical behavior.
        reference = reference.to(device=device, dtype=dtype).train()
        names = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
        with torch.enable_grad():
            for layer in range(self.gru.num_layers):
                parameters = [getattr(self.gru, f"{name}_l{layer}") for name in names]
                with torch.no_grad():
                    for name, parameter in zip(names, parameters):
                        getattr(reference, f"{name}_l0").copy_(parameter)
                values = torch.arange(
                    2 * 4 * self.hidden_units, device=device, dtype=dtype
                ).reshape(2, 4, self.hidden_units)
                x = (0.1 * torch.sin(values)).requires_grad_()
                initial = (0.1 * torch.cos(values[:, 0])).requires_grad_()
                ref_x = x.detach().clone().requires_grad_()
                ref_initial = initial.detach().clone().requires_grad_()
                expected, expected_hidden = reference(ref_x, ref_initial.unsqueeze(0))
                gates = torch.nn.functional.linear(x, parameters[0], parameters[2])
                state, outputs = initial, []
                for t in range(x.size(1)):
                    state = self._gru_step(gates[:, t], state, layer, retention=1.0)
                    outputs.append(state)
                actual = torch.stack(outputs, dim=1)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(
                    state, expected_hidden[0], atol=1e-5, rtol=1e-4
                )
                upstream = torch.cos(values)
                end_upstream = torch.sin(values[:, 0])
                actual_grads = torch.autograd.grad(
                    (actual * upstream).sum() + (state * end_upstream).sum(),
                    [x, initial] + parameters,
                )
                expected_grads = torch.autograd.grad(
                    (expected * upstream).sum()
                    + (expected_hidden[0] * end_upstream).sum(),
                    [ref_x, ref_initial]
                    + [getattr(reference, f"{name}_l0") for name in names],
                )
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, atol=1e-5, rtol=1e-4
                    )

    @torch.inference_mode()
    def validate_execution(self):
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            device = self.item_emb.weight.device
            a, b = 1, min(2, self.itemnum)
            seq = torch.tensor(
                [[0, 0, 0, 0], [0, 0, 0, a], [0, a, 0, b], [a, b, a, b], [0, a, 0, 0]],
                dtype=torch.long,
                device=device,
            )
            features = self.log2feats(seq)
            if not torch.isfinite(features).all():
                raise FloatingPointError("Nonfinite execution-check features")
            independent = torch.stack(
                [
                    (
                        features[row, position]
                        if position >= 0
                        else features.new_zeros(self.hidden_units)
                    )
                    for row, position in enumerate([-1, 3, 3, 3, 1])
                ]
            )
            torch.testing.assert_close(
                self._readout(features, seq), independent, atol=0, rtol=0
            )

            def close(actual, expected):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

            close(features[seq.eq(0)], torch.zeros_like(features[seq.eq(0)]))
            close(self.log2feats(seq), features)
            close(self.encode(seq), independent)
            padded = torch.cat((torch.zeros_like(seq[:, :2]), seq), dim=1)
            close(self.log2feats(padded)[:, 2:], features)
            order = torch.arange(seq.size(0) - 1, -1, -1, device=device)
            close(self.log2feats(seq[order]), features[order])
            candidates = torch.tensor([a, b], device=device)
            scores = self.predict_score(independent, candidates)
            close(self.score_candidates(seq, candidates), scores)
            close(
                self.predict_score(independent, candidates.expand(seq.size(0), -1)),
                scores,
            )
            ids = candidates.view(1, 1, -1).expand(seq.size(0), seq.size(1), -1)
            position_scores = self.predict_score(features, ids)
            close(position_scores, self.predict_score(features, candidates))
            close(
                self.predict_score(features, seq),
                (features * self.item_emb(seq)).sum(-1),
            )
            for row in range(seq.size(0)):
                close(self.log2feats(seq[row : row + 1]), features[row : row + 1])
                compact = seq[row][seq[row].ne(0)].unsqueeze(0)
                close(self.encode(compact)[0], independent[row])
                close(
                    self.score_candidates(seq[row : row + 1], candidates), scores[row]
                )
                for t in range(seq.size(1)):
                    close(
                        position_scores[row, t],
                        self.predict_score(features[row, t], candidates),
                    )
            close(self.log2feats(seq[:1]), torch.zeros_like(features[:1]))
            close(self.encode(seq[:, :0]), torch.zeros_like(independent))
        finally:
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Share candidate scoring across training adapters and exported prediction
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        scores = self.predict_score(self.encode(seq), candidates)
        return scores[0] if seq.size(0) == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    ### >>> Self_EvolveRec-BLOCK-START: Seed each independent category fit and audit carry-only execution before optimization
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    model.dataset_name = dataset_name
    model.validate_carry_equivalence()
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

    ### >>> Self_EvolveRec-BLOCK-START: Reload and verify the validation-selected category checkpoint
    if best_epoch == 0:
        raise RuntimeError(f"No validation-selected checkpoint for {dataset_name}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if checkpoint["architecture_config"] != model.architecture_config:
        raise RuntimeError(f"Checkpoint architecture mismatch for {dataset_name}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.validate_execution()
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
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required category artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing required GRU4Rec artifacts for {ds}: "
                f"metadata={meta_path}, checkpoint={ckpt_path}"
            )
        # Static review summary: checked CUDA reference-backward requirements,
        # sampler termination and unchanged RNG draw order, category routing,
        # carry-only equations, padding readout and strict checkpoint loading.
        # No training, execution or validation improvement is claimed.
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Load the saved item-carry architecture without reinitializing trained gates
        checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=True)
        config = checkpoint["architecture_config"]
        if config != meta["architecture_config"]:
            raise RuntimeError(f"Checkpoint/metadata architecture mismatch for {ds}")
        model = ModelClass(**config).to(_device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.dataset_name = ds
        model.eval()
        model.validate_execution()
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve all category routes and verify emitted predictor scores against selected checkpoints
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError("This benchmark requires exactly the four supplied categories")
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)

    import importlib.util

    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        spec = importlib.util.spec_from_file_location(
            "current_trial_predictor_audit", os.path.join(working_dir, "predict.py")
        )
        exported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exported)
        for dataset_name in DATASETS:
            checkpoint = torch.load(
                os.path.join(working_dir, f"{dataset_name}_model.pt"),
                map_location=device,
                weights_only=True,
            )
            with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "rb") as f:
                metadata = pickle.load(f)
            model = GRU4RecModel(**checkpoint["architecture_config"]).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            exported_class = exported._load_model_class()
            restored = exported_class(**checkpoint["architecture_config"]).to(device)
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            restored.eval()
            exported._models = {dataset_name: restored}
            exported._metas = {dataset_name: metadata}
            exported._device = device
            histories, _, _ = load_dataset(dataset_name, DATA_DIR)
            panel = [[], [1], [1] * MAXLEN]
            panel.extend(list(histories.values())[:3])
            candidates = list(range(1, min(16, model.itemnum) + 1))
            with torch.inference_mode():
                for history in panel:
                    sequence = _history_to_tensor(history, MAXLEN, device)
                    expected = model.score_candidates(
                        sequence, torch.tensor(candidates, device=device)
                    ).cpu()
                    actual = torch.tensor(
                        exported.predict(
                            0, history, candidates, dataset_name=dataset_name
                        ),
                        dtype=expected.dtype,
                    )
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            exported._models.clear()
            exported._metas.clear()
            del model, restored
    ### <<< Self_EvolveRec-BLOCK-END
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


