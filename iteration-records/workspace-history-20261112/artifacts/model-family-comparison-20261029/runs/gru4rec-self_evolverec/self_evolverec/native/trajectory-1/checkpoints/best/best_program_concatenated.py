# === main_code.py ===
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound history-filtered negative sampling without changing its distribution
            if nxt != 0 and len(interacted) < itemnum:
                neg_item = random.randint(1, itemnum)
                for _attempt in range(32):
                    if neg_item not in interacted:
                        break
                    neg_item = random.randint(1, itemnum)
                if neg_item in interacted:
                    pool = [i for i in range(1, itemnum + 1) if i not in interacted]
                    neg_item = random.choice(pool)
                neg[idx] = neg_item
            ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Freeze internal validation sampling across checkpoints
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
            ### >>> Self_EvolveRec-BLOCK-START: Use deterministic unique validation negatives with finite catalog handling
            candidates = [target]
            candidate_count = 1 + min(
                99, itemnum - sum(1 <= item <= itemnum for item in interacted)
            )
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve pairwise BPR with stable arithmetic and valid negative masking
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = seq.ne(0) & pos.ne(0) & neg.ne(0)
    loss = -torch.nn.functional.logsigmoid(pos_scores.float() - neg_scores.float())
    return loss.masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1)
    ### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Honor host epoch override and fixed training step limits
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1, int(os.environ.get("SASREC_EPOCHS", os.environ.get("GRU4REC_EPOCHS", "200")))
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))


### >>> Self_EvolveRec-BLOCK-START: Content-conditioned GRU retention with aligned leading-padding work elimination
class GRU4RecModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GRU4Rec requires at least one recurrent layer")
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        # DEBUG: retain an explicit nn.GRU backbone in the trained model.
        # Each recurrent layer belongs to this single jointly trained network.
        self.gru = nn.GRU(hidden_units, hidden_units, batch_first=True)
        self.upper_grus = nn.ModuleList(
            [
                nn.GRU(hidden_units, hidden_units, batch_first=True)
                for _ in range(num_layers - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.interlayer_dropout = nn.Dropout(dropout_rate)
        # Initialize embeddings before new gate parameters consume RNG draws.
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
        self.retention = nn.ModuleList(
            [nn.Linear(2 * hidden_units, 1) for _ in range(num_layers)]
        )
        for gate in self.retention:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, 3.0)

    def _recur(self, seq: torch.Tensor, bypass_retention: bool = False):
        valid = seq.ne(0)
        emb = self.dropout(self.item_emb(seq))
        states = [
            emb.new_zeros(seq.size(0), self.hidden_units)
            for _ in range(self.num_layers)
        ]
        # Skip shared leading padding while preserving embedding-dropout draws,
        # full output alignment, and a separate state for each recurrent layer.
        occupied_columns = valid.any(dim=0).nonzero(as_tuple=True)[0]
        first = (
            int(occupied_columns[0].item()) if occupied_columns.numel() else seq.size(1)
        )
        outputs = [emb.new_zeros(seq.size(0), self.hidden_units)] * first
        for t in range(first, seq.size(1)):
            active = valid[:, t, None]
            x = emb[:, t]
            # DEBUG: execute registered GRU layers with gated initial states;
            # a one-step nn.GRU implements the same GRU update as GRUCell.
            recurrent_layers = (self.gru, *self.upper_grus)
            for layer, recurrent_layer in enumerate(recurrent_layers):
                previous = states[layer]
                if bypass_retention:
                    recurrent = previous
                else:
                    alpha = torch.sigmoid(
                        self.retention[layer](torch.cat((previous, x), dim=-1))
                    )
                    recurrent = alpha * previous
                _, proposed_state = recurrent_layer(
                    x.unsqueeze(1), recurrent.unsqueeze(0).contiguous()
                )
                proposed = proposed_state[0]
                states[layer] = torch.where(active, proposed, previous)
                x = states[layer]
                if layer + 1 < self.num_layers:
                    x = self.interlayer_dropout(x)
            outputs.append(torch.where(active, states[-1], torch.zeros_like(x)))
        features = (
            torch.stack(outputs, dim=1)
            if outputs
            else emb.new_zeros(seq.size(0), 0, self.hidden_units)
        )
        return features, torch.stack(states, dim=0)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        return self._recur(seq)[0]

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.size(1) == 0:
            return self.item_emb.weight.new_zeros(seq.size(0), self.hidden_units)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values
        feature = out[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return feature * last.ge(0).unsqueeze(-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_score(self.encode(seq)[0], candidates)

    @torch.inference_mode()
    def audit_cell_parity(self):
        # A temporary copied-weight reference checks implementation semantics only.
        # It is never trained, saved, or used to produce deployed scores.
        reference = nn.GRU(
            self.hidden_units,
            self.hidden_units,
            self.num_layers,
            batch_first=True,
            dropout=self.interlayer_dropout.p if self.num_layers > 1 else 0.0,
        ).to(device=self.item_emb.weight.device, dtype=self.item_emb.weight.dtype)
        reference.eval()
        # DEBUG: copy the actual registered GRU weights into the temporary
        # parity reference; it remains outside the deployed model.
        recurrent_layers = (self.gru, *self.upper_grus)
        for layer, recurrent_layer in enumerate(recurrent_layers):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(reference, f"{name}_l{layer}").copy_(
                    getattr(recurrent_layer, f"{name}_l0")
                )
        prior_modes = [(module, module.training) for module in self.modules()]
        self.eval()
        max_difference = 0.0
        ranking_checks = []
        try:
            device = self.item_emb.weight.device
            candidates = torch.arange(1, min(self.itemnum, 16) + 1, device=device)
            for history in ([], [1], [1, min(2, self.itemnum), 1]):
                for width in (3, 6, 10):
                    seq = torch.zeros(1, width, dtype=torch.long, device=device)
                    if history:
                        seq[0, -len(history) :] = torch.tensor(history, device=device)
                    actual, states = self._recur(seq, bypass_retention=True)
                    expected = torch.zeros_like(actual)
                    expected_states = torch.zeros_like(states)
                    if history:
                        compact = seq[:, -len(history) :]
                        with torch.backends.cudnn.flags(enabled=False):
                            output, expected_states = reference(self.item_emb(compact))
                        expected[:, -len(history) :] = output
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(
                        states, expected_states, atol=1e-5, rtol=1e-4
                    )
                    a = self.predict_score(actual[0, -1], candidates)
                    b = self.predict_score(expected[0, -1], candidates)
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                    max_difference = max(
                        max_difference, float((actual - expected).abs().max())
                    )
                    ranking_checks.append(
                        bool(
                            torch.equal(
                                a.argsort(descending=True), b.argsort(descending=True)
                            )
                        )
                    )
            return {
                "bypass_parity": "passed",
                "atol": 1e-5,
                "rtol": 1e-4,
                "max_feature_abs_difference": max_difference,
                "ranking_agreement": all(ranking_checks),
                "ranking_note": "Numerically tied scores can have different orders",
            }
        finally:
            for module, training in prior_modes:
                module.training = training


### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Reset the required training seed for each independent category model
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    ### >>> Self_EvolveRec-BLOCK-START: Audit explicit recurrence and record the unchanged optimization configuration
    # Reference construction and auditing must not change training RNG streams.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        parity = model.audit_cell_parity()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    print(f"  [{dataset_name}] recurrence_audit={json.dumps(parity)}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    trial_config = {
        "variant": "content_conditioned_scalar_retention_gru4rec",
        "seed": SEED,
        "maxlen": MAXLEN,
        "embedding_dim": HIDDEN_UNITS,
        "hidden_units": HIDDEN_UNITS,
        "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "dropout_placement": "embedding and between recurrent layers",
        "input_output_projections": None,
        "retention_bias_initialization": 3.0,
        "retention_weight_initialization": 0.0,
        "objective": "pairwise_BPR",
        "sampler": "uniform excluding observed training history",
        "negative_count": 1,
        "optimizer": "Adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "epochs": NUM_EPOCHS,
        "gradient_clip_norm": 5.0,
        "eval_every": EVAL_EVERY,
        "patience": PATIENCE,
        "checkpoint_selection": "strict improvement in internal validation HR@10",
        "recurrence_audit": parity,
    }
    print(
        f"  [{dataset_name}] config={json.dumps(trial_config, sort_keys=True)}",
        flush=True,
    )

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Persist resolved retention experiment settings beneath the trial directory
    with open(os.path.join(working_dir, f"{dataset_name}_config.json"), "w") as f:
        json.dump(trial_config, f, indent=2, sort_keys=True)
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                "dropout_rate": DROPOUT_RATE,
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


### >>> Self_EvolveRec-BLOCK-START: Load the model definition embedded by the predictor writer
def _load_model_class():
    return GRU4RecModel
### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    ### >>> Self_EvolveRec-BLOCK-START: Require every category checkpoint before completing predictor loading
    if all(ds in _models for ds in DATASETS):
        return
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on missing category artifacts
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing GRU4Rec artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the identical checkpoint architecture before evaluation mode
            dropout_rate=meta["dropout_rate"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Preserve category routing without silently substituting another trained model
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")
        ds = dataset_name
    elif _models:
        ds = next(iter(_models))
    else:
        raise RuntimeError("No GRU4Rec category models were loaded")
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
    ### >>> Self_EvolveRec-BLOCK-START: Freeze the trained architecture into the generated predictor
    import inspect

    model_source = inspect.getsource(GRU4RecModel)
    import_anchor = "import torch\n"
    if script.count(import_anchor) != 1:
        raise RuntimeError("Predictor template must contain exactly one torch import")
    script = script.replace(
        import_anchor,
        import_anchor + "import torch.nn as nn\n\n" + model_source + "\n",
        1,
    )
    compile(script, os.path.join(working_dir, "predict.py"), "exec")
    predict_path = os.path.join(working_dir, "predict.py")
    ### <<< Self_EvolveRec-BLOCK-END
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


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic iteration without consuming an extra training batch
from collections import Counter
from itertools import islice

### <<< Self_EvolveRec-BLOCK-END


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
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    - Every statement must be grounded in the provided metric values/definitions.
    ### >>> Self_EvolveRec-BLOCK-START: Constrain interpretation to observed training evidence
    - Training margins measure sampled training separability, not validation accuracy.
    - Embedding concentration does not measure recommendation diversity.
    - Do not infer unavailable ratings, reviews, categories, prices, stores or simulator traits.
    - Null metrics are unavailable evidence, not zero-valued observations.
    - This probe does not verify generated predictor routing; do not claim predictor parity from these metrics.
    - Use STABLE only when measured evidence supports it, and acknowledge missing checks.
    ### <<< Self_EvolveRec-BLOCK-END
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the host ModelView contract and unavailable metadata boundary
    def __init__(self, view, device):
        if not isinstance(view.model, torch.nn.Module):
            raise TypeError("view.model must be the actual trained nn.Module")
        self.model = view
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.model.item_emb.weight.detach()
            emb = emb[
                1 : self.model.model.itemnum + 1
            ]  # exclude padding and reserved mask tokens

            ### >>> Self_EvolveRec-BLOCK-START: Report undefined concentration without fabricated observations
            n = emb.size(0)
            if n < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            ### >>> Self_EvolveRec-BLOCK-START: Measure directional concentration with bounded memory
            x = emb.index_select(0, idx).float()
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError("Nonfinite sampled item embeddings")
            x = x[x.norm(dim=-1).gt(0)]
            k = x.size(0)
            if k < 2:
                return None
            x = F.normalize(x, p=2, dim=-1)
            offdiag_sum = x.sum(dim=0).square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            if not np.isfinite(mean_cos):
                raise RuntimeError("Nonfinite embedding concentration")
            ### <<< Self_EvolveRec-BLOCK-END

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the actual recurrent module through ModelView
        self.model.model.eval()
        ### <<< Self_EvolveRec-BLOCK-END

        margins_all = []
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Enforce the diagnostic loader contract and exact batch limit
        for batch in islice(dataloader, num_batches):
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError(
                    "Diagnostic loader must yield (None, sequence, positive, negative)"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Pass aligned device tensors to the GRU feature interface
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            log_feats = self.model.log2feats(seq_t)
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Validate aligned diagnostic tensors before scoring
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic tensors must have matching B x L shapes")
            expected = (*seq_t.shape, self.model.item_emb.embedding_dim)
            if tuple(log_feats.shape) != expected:
                raise RuntimeError("Diagnostic features must retain B x L alignment")
            all_items = torch.stack([pos_t, neg_t], dim=-1)
            ### <<< Self_EvolveRec-BLOCK-END
            all_logits = self.model.predict_score(log_feats, all_items)
            margin = all_logits[..., 0] - all_logits[..., 1]

            ### >>> Self_EvolveRec-BLOCK-START: Exclude padding and missing negative targets from measured margins
            valid = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if margin.shape != valid.shape:
                raise RuntimeError("Diagnostic score alignment mismatch")
            if not torch.isfinite(margin[valid]).all():
                raise RuntimeError("Nonfinite diagnostic margins")
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use identical validity masks for summaries and breakdowns
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            for b, l in np.argwhere(low_mask):
                low_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    low_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    low_neg_cat[c] += 1

            for b, l in np.argwhere(wrong_mask):
                wrong_total += 1
                pid = int(pos_cpu[b, l])
                nid = int(neg_cpu[b, l])
                for c in self._get_categories(pid):
                    wrong_pos_cat[c] += 1
                for c in self._get_categories(nid):
                    wrong_neg_cat[c] += 1

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Preserve undefined margin statistics as null
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        ### >>> Self_EvolveRec-BLOCK-START: Report measured support and unavailable category attributes
        support = sum(m.numel() for m in margins_all)
        breakdown = {
            "valid_positions": int(support),
            "undefined_reason": None if support else "No valid aligned training pairs",
            "category_availability": "Unavailable in the shared task; breakdown lists are empty",
            "eps": float(eps),
            ### <<< Self_EvolveRec-BLOCK-END
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        return margin_metrics, breakdown

    def run_full_diagnosis(self, train_loader):
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        ### >>> Self_EvolveRec-BLOCK-START: Isolate diagnostic randomness and restore recurrent module modes
        module = self.model.model
        modes = [(child, child.training) for child in module.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            module.eval()
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=False
            ):
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        train_loader, num_batches=10, eps=eps, topk_categories=20
                    )
                )
        finally:
            for child, training in modes:
                child.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        ### <<< Self_EvolveRec-BLOCK-END
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Interpret concentration without claiming recommendation diversity
                    "Mean off-diagonal cosine mapped to [0,1], excluding padding and zero-norm embeddings. "
                    "Directional concentration only, not recommendation diversity. Null with fewer than two observations."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                ### >>> Self_EvolveRec-BLOCK-START: Define training-only margin support and missing evidence
                "pos_neg_margin_mean": "Mean positive-minus-negative score where input, positive and negative are nonzero; sampled training separability, not validation accuracy. Null without support.",
                ### <<< Self_EvolveRec-BLOCK-END
                "pos_neg_margin_p10": "10th percentile of (pos_logit - neg_logit); lower suggests many hard/confusing cases.",
                "pos_neg_neg_beats_pos_rate": "Fraction where neg_logit > pos_logit (margin < 0); lower is better.",
                "pos_neg_small_margin_rate": "Fraction where (pos_logit - neg_logit) < eps; higher means weak separation.",
                "pos_neg_margin_category_breakdown": (
                    ### >>> Self_EvolveRec-BLOCK-START: Describe unavailable metadata honestly
                    "Counts valid training positions, margins below eps=0.5 and margins below zero. "
                    "Item categories are unavailable; category lists remain empty."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
            },
        }
