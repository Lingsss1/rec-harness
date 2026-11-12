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

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
K = 256
CHUNK_SIZE = 512
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_USER_CACHE = {}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


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
    if itemnum <= 0:
        itemnum = max((i for split in (train, valid) for history in split.values() for i in history), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    cached = _USER_CACHE.get(key)
    if cached is None or cached[0] is not user_train:
        users = [u for u, history in user_train.items() if len(history) >= 2]
        _USER_CACHE[key] = (user_train, users)
    else:
        users = cached[1]
    if not users or itemnum < 2:
        raise RuntimeError('Need training sequences and at least two items')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        inputs = history[:-1][-maxlen:]
        targets = history[1:][-maxlen:]
        seq[row, -len(inputs):] = inputs
        pos[row, -len(targets):] = targets
    # Independent draws with replacement; exclude only the current positive.
    neg = np.zeros((batch_size, maxlen, K), dtype=np.int64)
    targets = pos[pos != 0]
    sampled = np.random.randint(1, itemnum + 1, size=(len(targets), K), dtype=np.int64)
    collisions = sampled == targets[:, None]
    while collisions.any():
        sampled[collisions] = np.random.randint(1, itemnum + 1, size=int(collisions.sum()))
        collisions = sampled == targets[:, None]
    neg[pos != 0] = sampled
    return seq, pos, neg


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
        return self.gru(self.dropout(self.item_emb(sequence)))[0]

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(sequence.size(0), device=sequence.device), last]

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.ndim == seq.ndim + 1:
            negative = (features.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        else:
            negative = (features * self.item_emb(neg)).sum(-1)
        return positive, negative

    def score_candidates(self, sequence, candidates):
        return torch.matmul(self.item_emb(candidates), self.encode(sequence)[0])


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


def sampled_ce_step(model, optimizer, seq, pos, neg):
    optimizer.zero_grad(set_to_none=True)
    mask = pos.ne(0)
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    negatives = neg[mask]
    # Give every nonempty history equal mass. Within a history, retain all
    # targets while mildly emphasizing those nearest the final valid position.
    with torch.no_grad():
        positions = torch.arange(pos.size(1), device=pos.device).expand_as(pos)
        last = positions.masked_fill(~mask, -1).max(dim=1).values
        age = (last.unsqueeze(1) - positions).clamp(min=0).to(features.dtype)
        weights = (0.5 + 0.5 * torch.exp(-age / 10.0)) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        weights = weights / mask.any(dim=1).sum().clamp(min=1)
        weights = weights[mask]
    # Accumulate output-embedding gradients chunk by chunk, then propagate
    # the complete weighted feature derivative through the recurrent graph.
    feature_grad = torch.empty_like(features)
    loss_value = 0.0
    for start in range(0, total, CHUNK_SIZE):
        end = min(total, start + CHUNK_SIZE)
        h = features[start:end].detach().requires_grad_(True)
        y = targets[start:end]
        n = negatives[start:end]
        positive = (h * model.item_emb(y)).sum(-1, keepdim=True)
        negative = torch.einsum('nd,nkd->nk', h, model.item_emb(n))
        logits = torch.cat([positive, negative], dim=1)
        labels = torch.zeros(end - start, dtype=torch.long, device=h.device)
        losses = F.cross_entropy(logits, labels, reduction='none')
        loss = (losses * weights[start:end]).sum()
        loss.backward()
        feature_grad[start:end].copy_(h.grad)
        loss_value += float(loss.detach())
    features.backward(feature_grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss_value


def validation_cases(train, valid, itemnum):
    users = [u for u in valid if u in train and valid[u]]
    rng = random.Random(SEED)
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | {0, target}
        available = itemnum - len({i for i in excluded if 1 <= i <= itemnum})
        count = min(99, available)
        candidates = [target]
        while len(candidates) < count + 1:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                excluded.add(candidate)
                candidates.append(candidate)
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            sequence[-len(history):] = history
        cases.append((sequence, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hits = ndcg = 0.0
    for sequence, candidates in cases:
        sequence = torch.as_tensor(sequence, device=device).unsqueeze(0)
        candidates = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(sequence, candidates)
        # Stable target-first tie breaking, matching descending score sorting.
        rank = int((scores[1:] > scores[0]).sum().item())
        if rank < 10:
            hits += 1.0
            ndcg += 1.0 / math.log2(rank + 2)
    return hits / max(len(cases), 1), ndcg / max(len(cases), 1)


def train_one_dataset(dataset_name, device):
    train, valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    if not train or itemnum < 2:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            total_loss += sampled_ce_step(model, optimizer, seq, pos, neg)
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, cases, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                         num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                         objective='sampled_cross_entropy', negatives=K), f)
    _USER_CACHE.pop(id(train), None)
    print(f'[{dataset_name}] best_val_HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained model definition so inference is self-contained.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    tail = r'''
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
    for dataset in DATASETS:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'],
                             meta['num_layers'], meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else DATASETS[0]
    if dataset not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _models[dataset]
    meta = _metas[dataset]
    maxlen = int(meta['maxlen'])
    sequence = [0] * maxlen
    history = list(history[-maxlen:])
    if history:
        sequence[-len(history):] = history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(GRU4RecModel) + '\n' + tail)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(dataset, device) for dataset in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
