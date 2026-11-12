"""Per-category GRU4Rec with tempered-frequency training negatives."""
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
import torch.nn.functional as F

DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42

MAXLEN = int(os.environ.get("GRU4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN", "128"))
NUM_LAYERS = int(os.environ.get("GRU4REC_LAYERS", "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT", "0.1"))
LR = float(os.environ.get("GRU4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("GRU4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("GRU4REC_BATCH", "128"))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get("GRU4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))))
EVAL_EVERY = int(os.environ.get("GRU4REC_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("GRU4REC_PATIENCE", "20"))
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


seed_all()


def _read_split(path):
    result = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.split()
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
                    parts = line.split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    return train, valid, _load_itemnum(data_dir, dataset_name)


# Only one active training category is sampled at a time. No sampler artifacts
# are saved or consulted by prediction.
_sampler_source = None
_sampler_itemnum = None
_sampler_cdf = None


def prepare_sampler(user_train, itemnum):
    global _sampler_source, _sampler_itemnum, _sampler_cdf
    if itemnum <= 0:
        raise RuntimeError("Empty item vocabulary")
    counts = np.zeros(itemnum + 1, dtype=np.float64)
    for history in user_train.values():
        np.add.at(counts, np.asarray(history, dtype=np.int64), 1)
    weights = counts[1:] ** 0.5
    if weights.sum() == 0:
        weights[:] = 1
    q = 0.5 / itemnum + 0.5 * weights / weights.sum()
    cdf = np.cumsum(q)
    cdf[-1] = 1.0
    _sampler_source = user_train
    _sampler_itemnum = itemnum
    _sampler_cdf = cdf


def draw_negative(interacted):
    if _sampler_cdf is None:
        raise RuntimeError("Training sampler has not been initialized")
    if len(interacted) >= _sampler_itemnum:
        raise RuntimeError("No unobserved item available for negative sampling")
    while True:
        item = int(np.searchsorted(_sampler_cdf, np.random.random(), side="right")) + 1
        if item not in interacted:
            return item


def sample_batch(user_train, itemnum, batch_size, maxlen):
    # Lazy initialization also supports standalone family-specific diagnostics.
    if _sampler_source is not user_train or _sampler_itemnum != itemnum:
        prepare_sampler(user_train, itemnum)
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
                neg[idx] = draw_negative(interacted)
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long,
                                       device=self.item_emb.weight.device)
        emb = self.dropout(self.item_emb(sequence))
        out, _ = self.gru(emb)
        return out

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long,
                                    device=self.item_emb.weight.device)
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(self, seq, candidates):
        user_emb = self.encode(seq)[0]
        return torch.matmul(self.item_emb(candidates), user_emb)


class ModelView(nn.Module):
    """Diagnostic view of the same single GRU network."""
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


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    losses = F.softplus(neg_scores - pos_scores)
    return losses[pos.ne(0)].mean()


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # A private RNG keeps validation candidates fixed across epochs and prevents
    # evaluation from changing the training sampling stream.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            wanted = min(99, itemnum - len(excluded - {0}))
            while len(candidates) < wanted + 1:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f"HR@{k}"] += 1.0
                    totals[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(len(users), 1) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"[{dataset_name}] users={len(user_train)} items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")
    prepare_sampler(user_train, itemnum)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    os.makedirs(WORKING_DIR, exist_ok=True)
    ckpt_path = os.path.join(WORKING_DIR, f"{dataset_name}_model.pt")
    best_hr10, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            loss = bpr_sequence_loss(model, seq, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics["HR@10"]
            print(f"[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} loss={total_loss / NUM_BATCHES:.4f} "
                  f"val_score: {hr10:.4f} HR@20={metrics['HR@20']:.4f} "
                  f"NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f"[{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(f"[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.4f}", flush=True)
    with open(os.path.join(WORKING_DIR, f"{dataset_name}_meta.pkl"), "wb") as f:
        pickle.dump({"itemnum": itemnum, "maxlen": MAXLEN,
                     "hidden_units": HIDDEN_UNITS, "num_layers": NUM_LAYERS,
                     "dropout_rate": DROPOUT_RATE}, f)
    print(f"[{dataset_name}] Best val HR@10={best_hr10:.4f}", flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""Route each category to exactly one trained GRU4Rec checkpoint."""
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
_ModelClass = None


def _ensure_loaded(dataset_name):
    global _device, _ModelClass
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _ModelClass is None:
        spec = importlib.util.spec_from_file_location(
            f"gru4rec_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.GRU4RecModel
    with open(os.path.join(_HERE, f"{dataset_name}_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    model = _ModelClass(meta["itemnum"], meta["hidden_units"],
                        meta["num_layers"], meta["dropout_rate"]).to(_device)
    state = torch.load(os.path.join(_HERE, f"{dataset_name}_model.pt"),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset_name = dataset_name or DATASETS[0]
    _ensure_loaded(dataset_name)
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = int(meta["maxlen"])
    seq_arr = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, cand).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    path = os.path.join(working_dir, "predict.py")
    with open(path, "w") as f:
        f.write(script)
    print(f"predict.py written -> {path}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset GRU4Rec training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    write_predict_script(WORKING_DIR)
    print(f"All datasets done. Average val HR@10 = {sum(scores) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
