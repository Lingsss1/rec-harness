"""Per-category GRU4Rec with independent input and output dropout."""
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

DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = 128
NUM_LAYERS = 1
INPUT_DROPOUT = 0.2
OUTPUT_DROPOUT = 0.1
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
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


def _read_split(path):
    result = defaultdict(list)
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(path):
        with open(path) as f:
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
            raise RuntimeError("no unobserved training negative available")
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
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
                 dropout_rate=0.1, input_dropout=0.2, output_dropout=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.input_dropout = nn.Dropout(input_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        embedded = self.input_dropout(self.item_emb(sequence))
        features, _ = self.gru(embedded)
        # Output dropout does not feed dropped states back into the GRU.
        return self.output_dropout(features)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == 1 and features.ndim >= 2:
            return torch.matmul(features, embeddings.transpose(0, 1))
        return (features * embeddings).sum(dim=-1)

    def encode(self, seq):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        return self.predict_score(features, candidates)[0]


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = (pos != 0).float()
    losses = -torch.log(torch.sigmoid(positive - negative) + 1e-8)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # Fixed validation candidates; this local RNG cannot perturb training sampling.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}
    if not users:
        return totals
    model.eval()
    with torch.no_grad():
        for user in users:
            history = user_train[user]
            target = user_valid[user][0]
            excluded = set(history) | {target, 0}
            candidates = [target]
            available = itemnum - sum(1 <= x <= itemnum for x in excluded)
            count = min(99, available)
            while len(candidates) < count + 1:
                negative = rng.randint(1, itemnum)
                if negative not in excluded:
                    excluded.add(negative)
                    candidates.append(negative)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f"HR@{k}"] += 1.0
                    totals[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)
    model.train()
    return {key: value / len(users) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"[{dataset_name}] users={len(user_train)} items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or vocabulary: {dataset_name}")
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS,
                         input_dropout=INPUT_DROPOUT,
                         output_dropout=OUTPUT_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
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
            print(f"[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.4f} "
                  f"val_score: {hr10:.4f} NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f"[{dataset_name}] early stopping at {epoch}", flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError("training completed without checkpoint selection")
    metadata = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                    num_layers=NUM_LAYERS, dropout_rate=0.1,
                    input_dropout=INPUT_DROPOUT, output_dropout=OUTPUT_DROPOUT,
                    seed=SEED, best_epoch=best_epoch)
    with open(os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"[{dataset_name}] best validation HR@10={best_hr10:.4f}", flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""Single-checkpoint per-category GRU4Rec prediction."""
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
_model_class = None


def _load_model_class():
    global _model_class
    if _model_class is None:
        path = os.path.join(_HERE, "..", "train.py")
        spec = importlib.util.spec_from_file_location(f"gru4rec_dropout_train_{os.getpid()}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.GRU4RecModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    if _device is None:
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(_HERE, f"{dataset_name}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    model = _load_model_class()(
        itemnum=meta["itemnum"], hidden_units=meta["hidden_units"],
        num_layers=meta["num_layers"], dropout_rate=meta["dropout_rate"],
        input_dropout=meta["input_dropout"], output_dropout=meta["output_dropout"],
    ).to(_device)
    state = torch.load(os.path.join(_HERE, f"{dataset_name}_model.pt"),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    if not candidates:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = meta["maxlen"]
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, "predict.py"), "w") as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    write_predict_script(working_dir)
    print(f"GRU4Rec input_dropout={INPUT_DROPOUT} output_dropout={OUTPUT_DROPOUT} device={device}", flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f"Average validation HR@10 = {sum(scores) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
