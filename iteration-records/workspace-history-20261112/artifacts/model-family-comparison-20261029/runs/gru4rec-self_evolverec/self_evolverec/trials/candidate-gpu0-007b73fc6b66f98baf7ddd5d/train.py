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
### >>> Self_EvolveRec-BLOCK-START: Preserve the four-category task and fixed training seed
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


### >>> Self_EvolveRec-BLOCK-START: Unique uniform negatives with training-history exclusions and diagnostic compatibility
def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    negative_count: int = 1,
):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    if negative_count < 1:
        raise ValueError("negative_count must be positive")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros((maxlen, negative_count), dtype=np.int64)
        interacted = {int(i) for i in history if 1 <= i <= itemnum}
        available = itemnum - len(interacted)
        count = min(negative_count, available)
        # Dense exclusions use an explicit eligible pool; sparse exclusions use
        # rejection sampling. Both give uniform samples without replacement.
        eligible = None
        if count and (
            available <= 2 * negative_count or len(interacted) > itemnum // 2
        ):
            eligible = [i for i in range(1, itemnum + 1) if i not in interacted]
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0 and count:
                if eligible is not None:
                    sampled = random.sample(eligible, count)
                else:
                    sampled = []
                    selected = set()
                    while len(sampled) < count:
                        candidate = random.randint(1, itemnum)
                        if candidate not in interacted and candidate not in selected:
                            sampled.append(candidate)
                            selected.add(candidate)
                neg[idx, :count] = sampled
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg[:, 0] if negative_count == 1 else neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


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
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible local validation without perturbing training sampling
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound unique validation candidates for small eligible catalogs
            available = itemnum - sum(1 for i in interacted if 1 <= i <= itemnum)
            candidate_count = 1 + min(99, available)
            while len(candidates) < candidate_count:
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


### >>> Self_EvolveRec-BLOCK-START: Memory-bounded BPR-max with stable pairwise loss and differentiable weights
def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
):
    from torch.utils.checkpoint import checkpoint

    if neg.ndim == 2:
        valid = (pos != 0) & (neg != 0)
    elif neg.ndim == 3:
        valid = (pos != 0) & (neg != 0).any(dim=-1)
    else:
        raise ValueError("neg must have shape B x L or B x L x K")
    supervised = pos != 0
    model.loss_positions += int(supervised.sum().item())
    model.omitted_positions += int((supervised & ~valid).sum().item())
    if not bool(valid.any()):
        return None
    if neg.ndim == 2:
        pos_scores, neg_scores = model(seq, pos, neg)
        return torch.nn.functional.softplus(
            neg_scores[valid] - pos_scores[valid]
        ).mean()

    features = model.log2feats(seq)[valid]
    positives = pos[valid]
    negatives = neg[valid]

    def chunk_loss(f, positive_ids, negative_ids):
        p = model.predict_score(f, positive_ids)
        n = model.predict_score(f, negative_ids)
        mask = negative_ids != 0
        finite_n = n.masked_fill(~mask, 0.0)
        log_w = torch.log_softmax(finite_n.masked_fill(~mask, -torch.inf), dim=-1)
        log_probability = torch.logsumexp(
            log_w + torch.nn.functional.logsigmoid(p.unsqueeze(-1) - finite_n),
            dim=-1,
        )
        ranking = -torch.logaddexp(
            log_probability, log_probability.new_tensor(math.log(1e-24))
        )
        penalty = (log_w.exp() * finite_n.square()).sum(dim=-1)
        return (ranking + BPRMAX_LAMBDA * penalty).sum()

    total = features.new_zeros(())
    for start in range(0, features.size(0), 256):
        args = (
            features[start : start + 256],
            positives[start : start + 256],
            negatives[start : start + 256],
        )
        # Ordinary chunking retains every chunk's embedding activations until
        # backward. Recompute these deterministic scores to bound that memory.
        if torch.is_grad_enabled():
            total = total + checkpoint(
                chunk_loss,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            total = total + chunk_loss(*args)
    return total / features.size(0)


### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Predeclared single-network ablations within fixed training limits
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get("GRU4REC_EPOCHS", "200"))))
EVAL_EVERY = max(1, int(os.environ.get("GRU4REC_EVAL_EVERY", "5")))
PATIENCE = max(1, int(os.environ.get("GRU4REC_PATIENCE", "20")))
EXPERIMENTS = {
    "baseline": {"event_decay": False, "bpr_max": False},
    "bpr_max": {"event_decay": False, "bpr_max": True},
    "event_decay": {"event_decay": True, "bpr_max": False},
    "combined": {"event_decay": True, "bpr_max": True},
}
EXPERIMENT = os.environ.get("GRU4REC_EXPERIMENT", "combined")
if EXPERIMENT not in EXPERIMENTS:
    raise ValueError(f"Unknown GRU4REC_EXPERIMENT: {EXPERIMENT}")
CONFIG = EXPERIMENTS[EXPERIMENT]
NEGATIVE_COUNT = 64 if CONFIG["bpr_max"] else 1
BPRMAX_LAMBDA = 1.0
### <<< Self_EvolveRec-BLOCK-END


class GRU4RecModel(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Explicit checkpointed event-retention architecture
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        event_decay: bool = False,
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
        ### >>> Self_EvolveRec-BLOCK-START: Jointly learned scalar retention initialized to 0.99
        self.dropout = nn.Dropout(dropout_rate)
        self.event_decay = bool(event_decay)
        if self.event_decay:
            initial_a = math.log(math.expm1(-math.log(0.99)))
            self.retention_logit = nn.Parameter(
                torch.tensor(initial_a, dtype=torch.float32)
            )
        else:
            self.register_parameter("retention_logit", None)
        self.loss_positions = 0
        self.omitted_positions = 0
        ### <<< Self_EvolveRec-BLOCK-END
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Causal GRU features with padding-invariant event retention
    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.item_emb(seq))
        if not self.event_decay:
            # Preserve the supplied baseline path for controlled ablations.
            out, _ = self.gru(emb)
            return out
        hidden = emb.new_zeros((self.gru.num_layers, seq.size(0), self.hidden_units))
        rho = torch.exp(-torch.nn.functional.softplus(self.retention_logit))
        outputs = []
        # Columns that are padding for the entire batch require no recurrent work.
        occupied = (seq != 0).any(dim=0).nonzero(as_tuple=False).flatten()
        first = int(occupied[0].item()) if occupied.numel() else seq.size(1)
        if first:
            outputs.append(emb.new_zeros((seq.size(0), first, self.hidden_units)))
        for t in range(first, seq.size(1)):
            _, proposed = self.gru(emb[:, t : t + 1], hidden * rho)
            active = (seq[:, t] != 0).view(1, -1, 1)
            hidden = torch.where(active, proposed, hidden)
            outputs.append(hidden[-1].unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == features.ndim:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.encode(seq)[0]
        return torch.matmul(self.item_emb(candidates), user_emb)

    ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Independent category initialization at seed 42 and resolved configuration logging
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "experiment": EXPERIMENT,
                "seed": SEED,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT_RATE,
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "batches_per_epoch": NUM_BATCHES,
                "max_epochs": NUM_EPOCHS,
                "negative_count": NEGATIVE_COUNT,
                "bpr_max_lambda": BPRMAX_LAMBDA,
                "score_activation": "identity",
                **CONFIG,
            }
        ),
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = GRU4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the declared event-retention configuration
        dropout_rate=DROPOUT_RATE,
        event_decay=CONFIG["event_decay"],
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ### >>> Self_EvolveRec-BLOCK-START: Train the selected objective and omit positions without eligible negatives
        total_loss = 0.0
        updates = 0
        model.loss_positions = 0
        model.omitted_positions = 0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, negative_count=NEGATIVE_COUNT
            )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            if loss is None:
                continue
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Nonfinite loss for {dataset_name}, epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            updates += 1

        avg_loss = total_loss / max(updates, 1)
        print(
            f"  [{dataset_name}] epoch={epoch} updates={updates}/{NUM_BATCHES}"
            f" supervised_positions={model.loss_positions}"
            f" omitted_no_negatives={model.omitted_positions}",
            flush=True,
        )
        ### <<< Self_EvolveRec-BLOCK-END
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
                "num_layers": NUM_LAYERS,
                ### >>> Self_EvolveRec-BLOCK-START: Save architecture and objective configuration alongside the single checkpoint
                "dropout_rate": DROPOUT_RATE,
                "event_decay": CONFIG["event_decay"],
                "experiment": EXPERIMENT,
                "negative_count": NEGATIVE_COUNT,
                "bpr_max_lambda": BPRMAX_LAMBDA,
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
### >>> Self_EvolveRec-BLOCK-START: Retain all four category models in the generated predictor
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
                f"{meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the exact trained GRU architecture from checkpoint metadata
            dropout_rate=meta["dropout_rate"],
            event_decay=meta.get("event_decay", False),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Prevent silently routing a category to another category's checkpoint
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained GRU4Rec checkpoint for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif len(_models) == 1:
        ds = next(iter(_models))
        model = _models[ds]
        meta = _metas[ds]
    else:
        raise ValueError("dataset_name is required when multiple category models are loaded")
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


