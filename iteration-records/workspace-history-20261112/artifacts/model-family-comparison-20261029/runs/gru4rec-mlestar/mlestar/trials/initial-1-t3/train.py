"""Per-category GRU4Rec with independent input and output dropout."""
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
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42

MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
HIDDEN_DROPOUT_RATE = 0.2
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get(
    "GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))))
EVAL_EVERY = 5
PATIENCE = 20


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir, dataset_name):
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


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError("uniform negative sampling requires an unseen item")
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                negative = random.randint(1, itemnum)
                while negative in interacted:
                    negative = random.randint(1, itemnum)
                neg[idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1,
                 dropout_rate=0.1, hidden_dropout_rate=0.2):
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

    def log2feats(self, sequence):
        emb = self.dropout(self.item_emb(sequence))
        out, _ = self.gru(emb)
        return self.hidden_dropout(out)

    def predict_score(self, features, items):
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp_min(0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = pos.ne(0).float()
    losses = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}
    if not users:
        return totals
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            available = itemnum - sum(1 for i in excluded if 1 <= i <= itemnum)
            nneg = min(99, available)
            candidates = [target]
            while len(candidates) < nneg + 1:
                negative = rng.randint(1, itemnum)
                if negative not in excluded:
                    excluded.add(negative)
                    candidates.append(negative)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for cutoff in (10, 20):
                if rank < cutoff:
                    totals[f"HR@{cutoff}"] += 1.0
                    totals[f"NDCG@{cutoff}"] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / len(users) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"[{dataset_name}] users={len(user_train)} items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or vocabulary: {dataset_name}")
    model = GRU4RecModel(
        itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE, HIDDEN_DROPOUT_RATE,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    os.makedirs(working_dir, exist_ok=True)
    checkpoint = os.path.join(working_dir, f"{dataset_name}_model.pt")
    best_hr10, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics["HR@10"]
            print(
                f"[{dataset_name}] epoch {epoch}/{NUM_EPOCHS} "
                f"loss={total_loss / NUM_BATCHES:.4f} val_score: {hr10:.4f} "
                f"HR@20={metrics['HR@20']:.4f} NDCG@10={metrics['NDCG@10']:.4f}",
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                print(f"[{dataset_name}] early stopping at {epoch}", flush=True)
                break
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump({
            "itemnum": itemnum,
            "maxlen": MAXLEN,
            "hidden_units": HIDDEN_UNITS,
            "num_layers": NUM_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "hidden_dropout_rate": HIDDEN_DROPOUT_RATE,
            "best_epoch": best_epoch,
        }, f)
    print(f"[{dataset_name}] best validation HR@10={best_hr10:.4f}", flush=True)
    return best_hr10


def write_predict_script(working_dir):
    # Embed the exact trained architecture so prediction is self-contained.
    import inspect
    architecture = inspect.getsource(GRU4RecModel)
    header = '''"""Single-checkpoint GRU4Rec prediction with category routing."""
from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

'''
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for ds in DATASETS:
        with open(os.path.join(_HERE, f"{ds}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = GRU4RecModel(
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
        _metas[ds] = meta
        _models[ds] = model


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f"Unknown dataset: {ds}")
    if not candidates:
        return []
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
    with open(os.path.join(working_dir, "predict.py"), "w") as f:
        f.write(header + architecture + "\n" + runtime)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"GRU4Rec input_dropout=0.1 output_dropout=0.2 device={device}", flush=True)
    scores = []
    for dataset in DATASETS:
        scores.append(train_one_dataset(dataset, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    write_predict_script(working_dir)
    print(f"Average validation HR@10 = {sum(scores) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
