"""Per-category GRU4Rec with tempered training-negative sampling."""
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

DEFAULT_DATASETS = [
    'Movies_and_TV', 'Industrial_and_Scientific',
    'Electronics', 'CDs_and_Vinyl',
]
DATASETS = [p.strip() for p in os.environ.get(
    'GAGC_DATASETS', ','.join(DEFAULT_DATASETS)).split(',') if p.strip()]
if len(DATASETS) != 4 or set(DATASETS) != set(DEFAULT_DATASETS):
    raise ValueError('All four benchmark categories must be retained')
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
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
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class TrainingNegativeSampler:
    def __init__(self, user_train, itemnum):
        if itemnum <= 0:
            raise ValueError('itemnum must be positive')
        self.itemnum = itemnum
        counts = np.zeros(itemnum + 1, dtype=np.float64)
        for history in user_train.values():
            np.add.at(counts, np.asarray(history, dtype=np.int64), 1)
        weights = counts[1:] ** 0.5
        if weights.sum() == 0:
            weights[:] = 1
        q = 0.5 / itemnum + 0.5 * weights / weights.sum()
        self.cdf = np.cumsum(q)
        self.cdf[-1] = 1.0
        self.users = [u for u, history in user_train.items() if len(history) >= 2]
        if not self.users:
            raise RuntimeError('no users with at least two training interactions')

    def draw_negative(self, interacted):
        if len(interacted) >= self.itemnum:
            raise RuntimeError('No unobserved item available for negative sampling')
        while True:
            item = int(np.searchsorted(self.cdf, np.random.random(), side='right')) + 1
            if item not in interacted:
                return item


# Keep sample_batch's diagnostic signature and [B,T] outputs unchanged.
# Retain a reference to the training dictionary to prevent identity reuse.
_SAMPLER_CACHE = {}


def _get_sampler(user_train, itemnum):
    key = (id(user_train), itemnum)
    cached = _SAMPLER_CACHE.get(key)
    if cached is None or cached[0] is not user_train:
        cached = (user_train, TrainingNegativeSampler(user_train, itemnum))
        _SAMPLER_CACHE[key] = cached
    return cached[1]


def sample_batch(user_train, itemnum, batch_size, maxlen):
    sampler = _get_sampler(user_train, itemnum)
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(sampler.users)
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
                neg[idx] = sampler.draw_negative(interacted)
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
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq[maxlen - 1 - i] = item
    return torch.tensor(seq[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = random.Random(SEED).sample(val_users, num_users)
    if not val_users:
        return {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            candidates = [target]
            while len(candidates) < 100 and itemnum > 0:
                neg_item = random.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
            seq = _history_to_tensor(history, maxlen, device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).detach().cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += float(rank < 10)
            hr20 += float(rank < 20)
            ndcg10 += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
            ndcg20 += 1.0 / math.log2(rank + 2) if rank < 20 else 0.0
    n = float(len(val_users))
    model.train()
    return {'HR@10': hr10 / n, 'HR@20': hr20 / n,
            'NDCG@10': ndcg10 / n, 'NDCG@20': ndcg20 / n}


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    losses = F.softplus(neg_scores - pos_scores)
    return losses[pos.ne(0)].mean()


MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
LR = float(os.environ.get('GRU4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('GRU4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = 200
EVAL_EVERY = int(os.environ.get('GRU4REC_EVAL_EVERY', '5'))
PATIENCE = int(os.environ.get('GRU4REC_PATIENCE', '20'))


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long,
                                       device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        out, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return out

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        else:
            items = items.to(device=features.device, dtype=torch.long)
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(self, seq, candidates):
        user_emb = self.encode(seq)[0]
        return torch.matmul(self.item_emb(candidates), user_emb)


class ModelView(nn.Module):
    """Family-specific diagnostic view without additional parameters."""
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


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f'  [{dataset_name}] users={len(user_train)}  items={itemnum}', flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'empty dataset or item vocabulary for {dataset_name}')
    # Build this category's training-only CDF once, before sampling batches.
    _get_sampler(user_train, itemnum)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
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
        avg_loss = total_loss / NUM_BATCHES
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(f'  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}'
                  f'  loss={avg_loss:.4f}  val_score: {hr10:.4f}'
                  f"  HR@20={metrics['HR@20']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}",
                  flush=True)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'  [{dataset_name}] Early stopping at epoch {epoch}', flush=True)
                break
        elif epoch % 10 == 0:
            print(f'  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}',
                  flush=True)
    if best_hr10 < 0:
        torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump({'itemnum': itemnum, 'maxlen': MAXLEN,
                     'hidden_units': HIDDEN_UNITS, 'num_layers': NUM_LAYERS,
                     'dropout_rate': DROPOUT_RATE}, f)
    _SAMPLER_CACHE.pop((id(user_train), itemnum), None)
    print(f'  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}',
          flush=True)
    return max(best_hr10, 0.0)


def write_predict_script(working_dir):
    script = r'''"""One trained GRU4Rec checkpoint per benchmark category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _load_model_class():
    path = os.path.join(_HERE, '..', 'train.py')
    spec = importlib.util.spec_from_file_location(f'gru4rec_train_{os.getpid()}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelClass = _load_model_class()
    loaded = {}
    metas = {}
    for ds in DATASETS:
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = ModelClass(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                           num_layers=meta['num_layers'], dropout_rate=0.0).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        loaded[ds] = model
        metas[ds] = meta
    _models.update(loaded)
    _metas.update(metas)


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is None:
        dataset_name = DATASETS[0]
    if dataset_name not in _models:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, cand).detach().cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    path = os.path.join(working_dir, 'predict.py')
    with open(path, 'w') as f:
        f.write(script)
    print(f'predict.py written -> {path}', flush=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Per-dataset GRU4Rec training on {device}', flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    avg = sum(scores) / len(scores)
    print(f'\nAll datasets done. Average val HR@10 = {avg:.4f}', flush=True)


if __name__ == '__main__':
    main()
