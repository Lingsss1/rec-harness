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

# DEBUG: FP32 tensor dtype alone does not prevent TF32 arithmetic. Use full
# FP32 precision in training and in the model imported by the predictor.
# Retrain all category checkpoints after applying this numerical-path repair.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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


### >>> Self_EvolveRec-BLOCK-START: Reuse eligible training users while preserving diagnostic sampling
def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users: bool = False,
    eligible_users=None,
):
    users = eligible_users
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    ### <<< Self_EvolveRec-BLOCK-END
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    ### >>> Self_EvolveRec-BLOCK-START: Track sampled users for training-history exclusions
    batch_seq, batch_pos, batch_neg = [], [], []
    batch_users = []
    for _ in range(batch_size):
        user = random.choice(users)
        batch_users.append(user)
        ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic negative sampling and bypass unused training draws
            if nxt != 0 and not return_users and len(interacted) < itemnum:
                neg_item = random.randint(1, itemnum)
                for _attempt in range(32):
                    if neg_item not in interacted:
                        break
                    neg_item = random.randint(1, itemnum)
                if neg_item in interacted:
                    eligible = [i for i in range(1, itemnum + 1) if i not in interacted]
                    neg_item = random.choice(eligible)
                neg[idx] = neg_item
            ### <<< Self_EvolveRec-BLOCK-END
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    ### >>> Self_EvolveRec-BLOCK-START: Preserve default sampler return contract
    result = (np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg))
    if return_users:
        return np.asarray(batch_users, dtype=np.int64), result
    return result
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
    ### >>> Self_EvolveRec-BLOCK-START: Make checkpoint validation repeatable without consuming training RNG
    rng = random.Random(SEED)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        ### <<< Self_EvolveRec-BLOCK-END
        val_users = rng.sample(val_users, num_users)
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            ### >>> Self_EvolveRec-BLOCK-START: Reuse fixed unique validation candidates and bound small catalogs
            candidates = [target]
            candidate_count = 1 + min(
                99, itemnum - len({i for i in interacted if 1 <= i <= itemnum})
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
### >>> Self_EvolveRec-BLOCK-START: Respect fixed training limits and host epoch override
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
        self.gru = nn.GRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Correct padding-state updates and expose shared GRU scoring for loss-only control
    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        valid = seq.ne(0)
        lengths = valid.sum(dim=1)
        width = seq.size(1)
        positions = torch.arange(width, device=seq.device).expand_as(seq)
        # Stable compaction also handles empty histories without updating a real state.
        order = positions.masked_fill(~valid, width).argsort(dim=1, stable=True)
        compact = seq.gather(1, order)
        emb = self.dropout(self.item_emb(compact))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        # DEBUG: Avoid batch-shape-dependent cuDNN RNN numerical paths in the
        # strict padding/batch audit. Training, ModelView and the generated
        # predictor all execute this same packed, causal GRU implementation.
        # Keep the audit tolerances unchanged and preserve checkpoint keys.
        with torch.backends.cudnn.flags(enabled=False):
            packed_out, _ = self.gru(packed)
        compact_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=width
        )
        compact_index = (valid.long().cumsum(dim=1) - 1).clamp(min=0)
        out = compact_out.gather(
            1, compact_index.unsqueeze(-1).expand(-1, -1, self.hidden_units)
        )
        return out * valid.unsqueeze(-1)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_score(self.encode(seq)[0], candidates)

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: BPR-max-only attribution trial with exact history-filtered replacement sampling
class TrainingNegativeSampler:
    def __init__(self, user_train, itemnum, device):
        self.itemnum = itemnum
        self.device = device
        self.user_index = {u: i for i, u in enumerate(user_train)}
        self.histories = [
            sorted({i for i in items if 1 <= i <= itemnum})
            for items in user_train.values()
        ]
        stride = itemnum + 1
        keys = [
            u * stride + item
            for u, history in enumerate(self.histories)
            for item in history
        ]
        self.keys = torch.tensor(sorted(keys), dtype=torch.long, device=device)
        self.available = torch.tensor(
            [itemnum - len(h) for h in self.histories], dtype=torch.long, device=device
        )
        self.skipped_positions = 0

    def sample(self, users, targets, count=32):
        stride = self.itemnum + 1

        def forbidden(draws, row_users, row_targets):
            query = row_users[:, None] * stride + draws
            if self.keys.numel():
                index = torch.searchsorted(self.keys, query)
                seen = self.keys[index.clamp(max=self.keys.numel() - 1)].eq(query)
                seen &= index.lt(self.keys.numel())
            else:
                seen = torch.zeros_like(draws, dtype=torch.bool)
            return seen | draws.eq(row_targets[:, None])

        # Supplied positive targets are training interactions and thus already excluded.
        eligible = self.available[users].gt(0)
        self.skipped_positions += int((~eligible).sum().item())
        users = users[eligible]
        targets = targets[eligible]
        draws = torch.randint(
            1, self.itemnum + 1, (users.numel(), count), device=self.device
        )
        bad = forbidden(draws, users, targets)
        for _ in range(8):
            if not bool(bad.any()):
                break
            replacements = torch.randint(
                1, self.itemnum + 1, draws.shape, device=self.device
            )
            draws = torch.where(bad, replacements, draws)
            bad = forbidden(draws, users, targets)
        if bool(bad.any()):
            unresolved_rows = bad.any(dim=1)
            for user in users[unresolved_rows].unique().cpu().tolist():
                rows = unresolved_rows & users.eq(user)
                allowed = torch.ones(
                    self.itemnum + 1, dtype=torch.bool, device=self.device
                )
                allowed[0] = False
                history = torch.tensor(
                    self.histories[user], dtype=torch.long, device=self.device
                )
                allowed[history] = False
                pool = allowed.nonzero(as_tuple=True)[0]
                row_bad = bad[rows]
                replacements = pool[
                    torch.randint(pool.numel(), row_bad.shape, device=self.device)
                ]
                draws[rows] = torch.where(row_bad, replacements, draws[rows])
        return draws, eligible


def bpr_max_backward(model, seq, pos, batch_users, sampler):
    mask = seq.ne(0) & pos.ne(0)
    users = torch.tensor(
        [sampler.user_index[int(u)] for u in batch_users],
        dtype=torch.long,
        device=seq.device,
    )
    position_users = users[:, None].expand_as(pos)[mask]
    targets = pos[mask]
    negatives, eligible = sampler.sample(position_users, targets)
    if not bool(eligible.any()):
        return 0.0
    features = model.log2feats(seq)[mask][eligible]
    targets = targets[eligible]
    # Accumulate feature gradients separately so each score chunk releases its
    # item-embedding graph, then backpropagate through the GRU exactly once.
    score_features = features.detach().requires_grad_(True)
    total = 0.0
    n = targets.numel()
    for start in range(0, n, 1024):
        end = min(start + 1024, n)
        f = score_features[start:end]
        positive_scores = model.predict_score(f, targets[start:end]).float()
        negative_scores = model.predict_score(f, negatives[start:end]).float()
        log_weights = torch.log_softmax(negative_scores, dim=-1)
        ranking = -torch.logsumexp(
            log_weights
            + torch.nn.functional.logsigmoid(
                positive_scores[:, None] - negative_scores
            ),
            dim=-1,
        )
        penalty = 0.01 * (log_weights.exp() * negative_scores.square()).sum(dim=-1)
        loss = (ranking + penalty).sum() / n
        loss.backward()
        total += float(loss.detach())
    features.backward(score_features.grad)
    return total


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Seed each independent category model and cache eligible users
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    eligible_users = [u for u, items in user_train.items() if len(items) >= 2]
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
    ### >>> Self_EvolveRec-BLOCK-START: Initialize one training-only sampler and record resolved loss-control settings
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    negative_sampler = TrainingNegativeSampler(user_train, itemnum, device)
    trial_config = dict(
        variant="bpr_max_only",
        seed=SEED,
        maxlen=MAXLEN,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        optimizer="Adam",
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        batches_per_epoch=NUM_BATCHES,
        epochs=NUM_EPOCHS,
        eval_every=EVAL_EVERY,
        patience=PATIENCE,
        negative_count=32,
        negative_score_regularization=0.01,
        sampling="uniform_with_replacement_excluding_training_history",
        recent_residual=False,
    )
    print(
        f"  [{dataset_name}] config={json.dumps(trial_config, sort_keys=True)}",
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
            ### >>> Self_EvolveRec-BLOCK-START: Use cached users in the BPR-max-only attribution workflow
            batch_users, (seq, pos, _) = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                return_users=True,
                eligible_users=eligible_users,
            )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_value = bpr_max_backward(
                model, seq_t, pos_t, batch_users, negative_sampler
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss_value
            ### <<< Self_EvolveRec-BLOCK-END

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

    ### >>> Self_EvolveRec-BLOCK-START: Persist trial configuration and measured skipped-target count
    trial_config["skipped_positions_no_eligible_negative"] = (
        negative_sampler.skipped_positions
    )
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

# DEBUG: Match the training numerical policy in each generated predictor
# process, including when the host has previously enabled TF32.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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
    ### >>> Self_EvolveRec-BLOCK-START: Require complete category loading before using the predictor cache
    if all(ds in _models for ds in DATASETS):
        return
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Surface missing category artifacts during predictor loading
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing GRU4Rec artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the selected GRU configuration through the shared model definition
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
    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid dataset routing instead of scoring with another category
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


