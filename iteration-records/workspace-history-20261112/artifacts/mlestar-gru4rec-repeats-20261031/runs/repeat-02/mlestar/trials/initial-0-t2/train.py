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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
NEGATIVE_ALPHA = 0.5


def seed_all():
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


class TemperedSampler:
    def __init__(self, user_train, itemnum, alpha=NEGATIVE_ALPHA, buffer_size=262144):
        self.user_train = user_train
        self.itemnum = itemnum
        counts = np.zeros(itemnum + 1, dtype=np.float64)
        self.seen = {}
        self.users = []
        for user, history in user_train.items():
            values = np.asarray(history, dtype=np.int64)
            if np.any(values < 1) or np.any(values > itemnum):
                raise ValueError('Training item outside declared vocabulary')
            np.add.at(counts, values, 1.0)
            seen = set(history)
            self.seen[user] = seen
            if len(history) >= 2 and len(seen) < itemnum:
                self.users.append(user)
        if not self.users:
            raise RuntimeError('No training users with a next-item pair and an eligible negative')
        self.q = np.maximum(counts, 1.0) ** alpha
        self.q[0] = 0.0
        self.q /= self.q.sum()
        self.rng = np.random.RandomState(SEED)
        self.buffer_size = buffer_size
        self.buffer = np.empty(0, dtype=np.int64)
        self.offset = 0

    def draw(self):
        if self.offset >= len(self.buffer):
            self.buffer = self.rng.choice(self.itemnum + 1, size=self.buffer_size, p=self.q)
            self.offset = 0
        value = int(self.buffer[self.offset])
        self.offset += 1
        return value

    def negative(self, user):
        seen = self.seen[user]
        for _ in range(64):
            value = self.draw()
            if value not in seen:
                return value
        # Exact conditional sampling fallback prevents unbounded rejection loops.
        weights = self.q.copy()
        weights[np.fromiter(seen, dtype=np.int64)] = 0.0
        total = weights.sum()
        if total <= 0:
            raise RuntimeError('User has no eligible negative')
        weights /= total
        return int(self.rng.choice(self.itemnum + 1, p=weights))

    def batch(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        for row in range(batch_size):
            user = random.choice(self.users)
            history = self.user_train[user]
            nxt = history[-1]
            index = maxlen - 1
            for item in reversed(history[:-1]):
                seq[row, index] = item
                pos[row, index] = nxt
                neg[row, index] = self.negative(user)
                nxt = item
                index -= 1
                if index < 0:
                    break
        return seq, pos, neg


_sampler_cache = None


def sample_batch(user_train, itemnum, batch_size, maxlen):
    global _sampler_cache
    if (_sampler_cache is None or _sampler_cache.user_train is not user_train
            or _sampler_cache.itemnum != itemnum):
        _sampler_cache = TemperedSampler(user_train, itemnum)
    return _sampler_cache.batch(batch_size, maxlen)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.dropout(self.item_emb(sequence))
        features, _ = self.gru(embeddings)
        return features

    def encode(self, seq):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(dim=-1),
                (features * self.item_emb(neg)).sum(dim=-1))

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq), candidates)[0]


class ModelView(nn.Module):
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
    mask = (pos != 0).float()
    losses = -torch.log(torch.sigmoid(positive - negative) + 1e-8)
    return (losses * mask).sum() / mask.sum().clamp(min=1.0)


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


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
            if itemnum - len(interacted - {0}) < 99:
                raise RuntimeError('Insufficient unique negatives for validation protocol')
            while len(candidates) < 100 and itemnum > 0:
                negative = random.randint(1, itemnum)
                if negative not in interacted:
                    interacted.add(negative)
                    candidates.append(negative)
            seq = _history_to_tensor(history, maxlen, device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += float(rank < 10)
            hr20 += float(rank < 20)
            ndcg10 += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
            ndcg20 += 1.0 / math.log2(rank + 2) if rank < 20 else 0.0
    n = float(len(val_users))
    model.train()
    return {'HR@10': hr10 / n, 'HR@20': hr20 / n,
            'NDCG@10': ndcg10 / n, 'NDCG@20': ndcg20 / n}


def train_one_dataset(dataset_name, device, data_dir):
    global _sampler_cache
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} alpha={NEGATIVE_ALPHA}', flush=True)
    _sampler_cache = TemperedSampler(user_train, itemnum)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            batch = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(x, dtype=torch.long, device=device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val={metrics}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_hr < 0:
        raise RuntimeError('Training completed without validation checkpoint')
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump({'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
                     'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
                     'negative_alpha': NEGATIVE_ALPHA, 'best_epoch': best_epoch}, f)
    _sampler_cache = None
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture so inference does not import training code.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    architecture = inspect.getsource(GRU4RecModel)
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds in DATASETS:
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], 0.0).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f'Unknown dataset: {ds}')
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        return [float(x) for x in model.score_candidates(seq, items).cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n' + runtime)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = []
    for ds in DATASETS:
        scores.append(train_one_dataset(ds, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
