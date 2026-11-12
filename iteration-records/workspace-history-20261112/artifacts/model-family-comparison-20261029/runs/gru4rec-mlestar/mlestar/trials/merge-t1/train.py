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


class TrainingHistories(dict):
    def __init__(self, data):
        super().__init__(data)
        self.eligible_users = [u for u, history in data.items() if len(history) >= 2]


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = getattr(user_train, 'eligible_users', None)
    if users is None:
        users = [u for u, history in user_train.items() if len(history) >= 2]
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
    # Independent uniform draws, rejecting only the positive at this position.
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
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    negatives = neg[mask]
    total = targets.numel()
    if total == 0:
        raise RuntimeError('Training batch has no valid targets')
    # Chunk output embedding computation while preserving exact gradients.
    # Recurrent features are differentiated once after collecting each chunk.
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
        loss = F.cross_entropy(logits, labels, reduction='sum') / total
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
        rank = int((scores[1:] > scores[0]).sum().item())
        if rank < 10:
            hits += 1.0
            ndcg += 1.0 / math.log2(rank + 2)
    return hits / max(len(cases), 1), ndcg / max(len(cases), 1)


def train_one_dataset(dataset_name, device):
    train, valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    train = TrainingHistories(train)
    if not train.eligible_users or itemnum < 2:
        raise RuntimeError('Empty or insufficient dataset: ' + dataset_name)
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
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                objective='uniform_sampled_cross_entropy', num_negatives=K,
                seed=SEED, best_epoch=best_epoch)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best_val_HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'],
                         meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = DATASETS[0] if dataset_name is None else dataset_name
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if len(candidates) == 0:
        return []
    _ensure_loaded(dataset)
    model = _models[dataset]
    meta = _metas[dataset]
    maxlen = int(meta['maxlen'])
    recent = list(history[-maxlen:])
    sequence = [0] * (maxlen - len(recent)) + recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(GRU4RecModel) + '\n' + runtime)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    # Preserve the stronger initial solution's single seeded RNG stream.
    scores = [train_one_dataset(dataset, device) for dataset in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
