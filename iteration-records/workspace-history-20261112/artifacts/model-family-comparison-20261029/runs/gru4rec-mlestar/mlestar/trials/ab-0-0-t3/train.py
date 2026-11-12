"""Per-category GRU4Rec with output dropout disabled."""
from __future__ import annotations

import importlib.util
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
    "Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"
]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
HIDDEN_DROPOUT_RATE = 0.0
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))))
EVAL_EVERY = 5
PATIENCE = 20


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path: str) -> dict[int, list[int]]:
    data = defaultdict(list)
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
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name: str, data_dir: str):
    train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train: dict[int, list[int]], itemnum: int,
                 batch_size: int, maxlen: int):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError("no unseen training negatives for sampled user")
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


def _history_to_tensor(history: list[int], maxlen: int,
                       device: torch.device) -> torch.Tensor:
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum: int, hidden_units: int = 128,
                 num_layers: int = 1, dropout_rate: float = 0.1,
                 hidden_dropout_rate: float = 0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(
            hidden_units, hidden_units, num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dropout = nn.Dropout(hidden_dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.item_emb(sequence))
        out, _ = self.gru(emb)
        return self.hidden_dropout(out)

    def predict_score(self, features: torch.Tensor,
                      items: torch.Tensor) -> torch.Tensor:
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq: torch.Tensor, pos: torch.Tensor,
                neg: torch.Tensor):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.log2feats(seq)
        # Histories are LEFT padded: select the final nonpadding position.
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp_min(0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def score_candidates(self, seq: torch.Tensor,
                         candidates: torch.Tensor) -> torch.Tensor:
        return self.predict_score(self.encode(seq)[0], candidates)


class ModelView(nn.Module):
    """Expose the same feature and item-scoring interface for diagnostics."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def item_emb(self):
        return self.model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)

    def forward(self, seq, pos, neg):
        return self.model(seq, pos, neg)


def bpr_sequence_loss(model: nn.Module, seq: torch.Tensor,
                      pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def evaluate_val(model, user_train, user_valid, itemnum, device,
                 maxlen, num_users=200):
    # Keep this diagnostic candidate set fixed across checkpoint comparisons.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {"HR@10": 0.0, "HR@20": 0.0,
              "NDCG@10": 0.0, "NDCG@20": 0.0}
    if not users:
        return totals
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            available = itemnum - sum(1 for x in excluded if 1 <= x <= itemnum)
            count = min(99, available)
            while len(candidates) < count + 1:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f"HR@{k}"] += 1.0
                    totals[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / len(users) for key, value in totals.items()}


def train_one_dataset(dataset_name: str, device: torch.device,
                      data_dir: str) -> float:
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"[{dataset_name}] users={len(user_train)} items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")
    model = GRU4RecModel(
        itemnum=itemnum, hidden_units=HIDDEN_UNITS,
        num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
        hidden_dropout_rate=HIDDEN_DROPOUT_RATE,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid,
                                   itemnum, device, MAXLEN)
            hr10 = metrics["HR@10"]
            print(
                f"[{dataset_name}] epoch {epoch}/{NUM_EPOCHS} "
                f"loss={total_loss / NUM_BATCHES:.4f} "
                f"val_score: {hr10:.4f} HR@20={metrics['HR@20']:.4f} "
                f"NDCG@10={metrics['NDCG@10']:.4f}", flush=True,
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f"[{dataset_name}] early stopping at epoch {epoch}", flush=True)
                break
    if best_epoch == 0:
        torch.save(model.state_dict(), ckpt_path)
    meta = {
        "itemnum": itemnum, "maxlen": MAXLEN,
        "hidden_units": HIDDEN_UNITS, "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "hidden_dropout_rate": HIDDEN_DROPOUT_RATE,
    }
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[{dataset_name}] best validation HR@10={max(best_hr10, 0.0):.4f}", flush=True)
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""One GRU4Rec checkpoint per Amazon category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = None


def _load_model_class():
    path = os.path.join(_HERE, "..", "train.py")
    spec = importlib.util.spec_from_file_location(
        f"gru4rec_output_dropout_train_{os.getpid()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GRU4RecModel


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_layers=meta["num_layers"],
            dropout_rate=meta["dropout_rate"],
            hidden_dropout_rate=meta["hidden_dropout_rate"],
        ).to(_device)
        state = torch.load(os.path.join(_HERE, f"{ds}_model.pt"),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f"Unknown dataset_name: {ds}")
    model = _models[ds]
    meta = _metas[ds]
    maxlen = int(meta["maxlen"])
    seq_arr = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    model.eval()
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    path = os.path.join(working_dir, "predict.py")
    with open(path, "w") as f:
        f.write(script)
    print(f"predict.py written -> {path}", flush=True)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"GRU4Rec input dropout={DROPOUT_RATE}, output dropout={HIDDEN_DROPOUT_RATE}, device={device}", flush=True)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    write_predict_script(working_dir)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f"Average validation HR@10 = {sum(scores) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
