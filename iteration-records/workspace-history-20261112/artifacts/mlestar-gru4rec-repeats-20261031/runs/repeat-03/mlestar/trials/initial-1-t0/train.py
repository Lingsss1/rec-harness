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
from torch import nn
from torch.nn import functional as F

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 25
NUM_NEGATIVES = 64
BPR_REG = 0.1
_SAMPLE_CACHE = {}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                data[int(fields[0])].append(int(fields[1]))
    return dict(data)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((i for split in (train, valid) for h in split.values() for i in h), default=0)
    return train, valid, itemnum


def _sampling_state(user_train, itemnum):
    key = (id(user_train), itemnum)
    cached = _SAMPLE_CACHE.get(key)
    if cached is not None and cached[0] is user_train:
        return cached[1:]
    users, seen, complements = [], {}, {}
    for u, h in user_train.items():
        if len(h) < 2:
            continue
        forbidden = set(h)
        if len(forbidden) >= itemnum:
            continue
        users.append(u)
        seen[u] = forbidden
        if len(forbidden) > itemnum // 2:
            complements[u] = np.asarray([i for i in range(1, itemnum + 1) if i not in forbidden], dtype=np.int64)
    if not users:
        raise RuntimeError('No eligible training histories')
    _SAMPLE_CACHE[key] = (user_train, users, seen, complements)
    return users, seen, complements


def _draw_negatives(itemnum, forbidden, count, complement=None):
    if complement is not None:
        return np.random.choice(complement, size=count, replace=True)
    result = []
    for _ in range(8):
        draws = np.random.randint(1, itemnum + 1, size=max(2 * (count - len(result)), 16))
        result.extend(int(i) for i in draws if int(i) not in forbidden)
        if len(result) >= count:
            return np.asarray(result[:count], dtype=np.int64)
    eligible = np.asarray([i for i in range(1, itemnum + 1) if i not in forbidden], dtype=np.int64)
    if not len(eligible):
        raise RuntimeError('No eligible negative items')
    result.extend(np.random.choice(eligible, size=count - len(result), replace=True).tolist())
    return np.asarray(result, dtype=np.int64)


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    users, seen, complements = _sampling_state(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    if num_negatives == 1:
        neg = np.zeros_like(seq)
    else:
        pools = np.zeros((batch_size, num_negatives), dtype=np.int64)
    for b in range(batch_size):
        u = random.choice(users)
        h = user_train[u]
        length = min(len(h) - 1, maxlen)
        seq[b, -length:] = h[-length - 1:-1]
        pos[b, -length:] = h[-length:]
        count = length if num_negatives == 1 else num_negatives
        sampled = _draw_negatives(itemnum, seen[u], count, complements.get(u))
        if num_negatives == 1:
            neg[b, -length:] = sampled
        else:
            pools[b] = sampled
    if num_negatives != 1:
        # One independently sampled pool per user; all shifted targets belong
        # to that user's forbidden history. Reuse pools across valid positions.
        neg = np.broadcast_to(pools[:, None, :], (batch_size, maxlen, num_negatives))
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1).clamp_min(0)
        return features[torch.arange(sequence.size(0), device=sequence.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(-1)
        if items.ndim == 1:
            return features @ embeddings.T
        if items.ndim == features.ndim:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        raise ValueError('Unsupported feature/item shapes')

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.ndim == 3:
            negative = torch.bmm(features, self.item_emb(neg[:, -1, :]).transpose(1, 2))
        else:
            negative = (features * self.item_emb(neg)).sum(-1)
        return positive, negative

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq), candidates)[0]


def bpr_max(p, n, reg=0.1):
    weights = torch.softmax(n, dim=-1)
    log_win = torch.logsumexp(F.log_softmax(n, dim=-1) + F.logsigmoid(p[:, None] - n), dim=-1)
    return (-log_win + reg * (weights * n.square()).sum(-1)).mean()


def bpr_sequence_loss(model, seq, pos, neg):
    p, n = model(seq, pos, neg)
    mask = pos.ne(0)
    if n.ndim == p.ndim:
        n = n.unsqueeze(-1)
    return bpr_max(p[mask], n[mask], BPR_REG)


def make_validation_cases(train, valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        target = valid[u][0]
        forbidden = set(train[u]) | {0, target}
        if itemnum - len(forbidden - {0}) < 99:
            raise RuntimeError('Validation requires 99 unique eligible negatives')
        cand = [target]
        while len(cand) < 100:
            i = rng.randint(1, itemnum)
            if i not in forbidden:
                cand.append(i)
                forbidden.add(i)
        history = train[u][-maxlen:]
        sequences.append([0] * (maxlen - len(history)) + history)
        candidates.append(cand)
    if not sequences:
        raise RuntimeError('No validation users')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    ranks = []
    sequences, candidates = cases
    for start in range(0, len(sequences), 128):
        seq = torch.as_tensor(sequences[start:start + 128], device=device)
        cand = torch.as_tensor(candidates[start:start + 128], device=device)
        features = model.encode(seq)
        scores = model.predict_score(features, cand)
        # The positive occupies index zero, matching stable candidate ordering.
        ranks.extend((scores[:, 1:] > scores[:, :1]).sum(-1).cpu().tolist())
    model.train()
    return float(np.mean(np.asarray(ranks) < 10))


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    _SAMPLE_CACHE.clear()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    cases = make_validation_cases(train, valid, itemnum, MAXLEN)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, dataset_name + '_model.pt')
    best, best_epoch = -1.0, 0
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN, NUM_NEGATIVES)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            # Materialize only the negative pool; expand its time dimension.
            pool = torch.as_tensor(np.array(neg[:, -1, :], copy=True), device=device)
            neg = pool[:, None, :].expand(-1, MAXLEN, -1)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr = evaluate_val(model, cases, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.4f}', flush=True)
            if hr > best:
                best, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                loss='bpr-max', negatives=NUM_NEGATIVES, score_reg=BPR_REG,
                seed=SEED, best_epoch=best_epoch)
    with open(os.path.join(working, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best:.4f}', flush=True)
    return best


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nfrom torch import nn\n\n'
    body = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
        model.load_state_dict(torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True))
        model.eval()
        _models[dataset_name] = model
        _metas[dataset_name] = meta
    return _models[dataset_name], _metas[dataset_name]


def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model, meta = _load(ds)
    maxlen = meta['maxlen']
    history = list(history)[-maxlen:]
    sequence = [0] * (maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, cand)
    return scores.cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(GRU4RecModel) + '\n' + body)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    write_predict_script(working)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={np.mean(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
