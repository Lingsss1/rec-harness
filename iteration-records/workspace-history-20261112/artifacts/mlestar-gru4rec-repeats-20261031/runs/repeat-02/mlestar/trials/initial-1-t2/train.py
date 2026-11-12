"""Per-category GRU4Rec with tempered-popularity negative sampling.

One independently trained checkpoint per category. Training counts affect only
negative sampling, never prediction scores. Uses the supplied train/valid splits.
"""
from __future__ import annotations

import inspect
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
    'Movies_and_TV',
    'Industrial_and_Scientific',
    'Electronics',
    'CDs_and_Vinyl',
]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
LR = float(os.environ.get('GRU4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('GRU4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get(
    'GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NEGATIVE_ALPHA = 0.5
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_split(path):
    result = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def _load_itemnum(data_dir, dataset_name):
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
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


class TemperedSampler:
    def __init__(self, user_train, itemnum, alpha=0.5):
        self.user_train = user_train
        self.itemnum = int(itemnum)
        self.rng = np.random.RandomState(SEED)
        self.seen = {u: set(h) for u, h in user_train.items() if len(h) >= 2}
        self.users = [u for u in self.seen
                      if len({i for i in self.seen[u] if 1 <= i <= itemnum}) < itemnum]
        if not self.users:
            raise RuntimeError('No training users have both a transition and an eligible negative')
        counts = np.zeros(itemnum + 1, dtype=np.float64)
        for history in user_train.values():
            ids = np.asarray(history, dtype=np.int64)
            if ids.size and (ids.min() < 1 or ids.max() > itemnum):
                raise ValueError('Training item outside declared vocabulary')
            np.add.at(counts, ids, 1.0)
        self.q = np.maximum(counts, 1.0) ** alpha
        self.q[0] = 0.0
        self.q /= self.q.sum()
        self.buffer = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.complements = {}

    def draw(self):
        if self.cursor >= self.buffer.size:
            self.buffer = self.rng.choice(
                self.itemnum + 1, size=65536, replace=True, p=self.q)
            self.cursor = 0
        value = int(self.buffer[self.cursor])
        self.cursor += 1
        return value

    def negative(self, user):
        seen = self.seen[user]
        for _ in range(64):
            value = self.draw()
            if value not in seen:
                return value
        # Exact conditional distribution for dense histories; no endless rejection.
        if user not in self.complements:
            allowed = np.fromiter(
                (i for i in range(1, self.itemnum + 1) if i not in seen),
                dtype=np.int64)
            if not allowed.size:
                raise RuntimeError('No eligible negative for user')
            probabilities = self.q[allowed].copy()
            probabilities /= probabilities.sum()
            self.complements[user] = (allowed, probabilities)
        allowed, probabilities = self.complements[user]
        return int(self.rng.choice(allowed, p=probabilities))

    def batch(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        for row in range(batch_size):
            user = random.choice(self.users)
            history = self.user_train[user]
            nxt = history[-1]
            idx = maxlen - 1
            for item in reversed(history[:-1]):
                seq[row, idx] = item
                pos[row, idx] = nxt
                neg[row, idx] = self.negative(user)
                nxt = item
                idx -= 1
                if idx < 0:
                    break
        return seq, pos, neg


_ACTIVE_SAMPLER = None


def sample_batch(user_train, itemnum, batch_size, maxlen):
    global _ACTIVE_SAMPLER
    if (_ACTIVE_SAMPLER is None or
            _ACTIVE_SAMPLER.user_train is not user_train or
            _ACTIVE_SAMPLER.itemnum != itemnum):
        _ACTIVE_SAMPLER = TemperedSampler(user_train, itemnum, NEGATIVE_ALPHA)
    return _ACTIVE_SAMPLER.batch(batch_size, maxlen)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(
            hidden_units, hidden_units, num_layers=num_layers, batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(
                sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        out, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return out

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True).to(features.dtype)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        emb = self.item_emb(items)
        if features.ndim == 2 and items.ndim == 1:
            return features @ emb.transpose(0, 1)
        if features.ndim == 2 and items.ndim == 2:
            return (features.unsqueeze(1) * emb).sum(dim=-1)
        return (features * emb).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return (features * self.item_emb(pos)).sum(-1), (features * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        scores = self.predict_score(self.encode(seq), candidates)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView(nn.Module):
    """Expose the family-specific diagnostic feature/scoring interface."""
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
    positive, negative = model(seq, pos, neg)
    mask = pos.ne(0)
    return F.softplus(negative - positive)[mask].mean()


def _history_to_tensor(history, maxlen, device):
    sequence = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    return torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # Uniform, unique evaluation negatives. This RNG is independent of training.
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = random.Random(SEED).sample(val_users, num_users)
    rng = random.Random(SEED)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            available = itemnum - sum(1 for i in excluded if 1 <= i <= itemnum)
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
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {name: value / max(1, len(val_users)) for name, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    global _ACTIVE_SAMPLER
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} negative_alpha={NEGATIVE_ALPHA}', flush=True)
    _ACTIVE_SAMPLER = TemperedSampler(user_train, itemnum, NEGATIVE_ALPHA)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            batch = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss for {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} '
                  f'val_score: {hr:.4f} HR@20={metrics["HR@20"]:.4f} '
                  f'NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = {
        'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
        'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
        'negative_alpha': NEGATIVE_ALPHA, 'seed': SEED,
        'best_epoch': best_epoch,
    }
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    _ACTIVE_SAMPLER = None
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained class so inference does not import training or data.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    model_source = inspect.getsource(GRU4RecModel)
    predictor_source = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(
        itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
        num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta['maxlen'])
    tail = list(history[-maxlen:])
    seq = [0] * (maxlen - len(tail)) + tail
    sequence = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(sequence, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + model_source + '\n' + predictor_source)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'All datasets done. Average val HR@10 = {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
