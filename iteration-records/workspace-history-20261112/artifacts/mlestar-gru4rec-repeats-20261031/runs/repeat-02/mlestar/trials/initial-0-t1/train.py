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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
SEED = 42
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '50'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = 1
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.2'))
LR = float(os.environ.get('GRU4REC_LR', '0.001'))
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '64'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('SASREC_EPOCHS', '200')), int(os.environ.get('GRU4REC_EPOCHS', '200')))
EVAL_EVERY = 5
PATIENCE = 25
SOFTMAX_CHUNK = int(os.environ.get('GRU4REC_SOFTMAX_CHUNK', '128'))
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
    if not itemnum:
        itemnum = max((max(items, default=0) for split in (train, valid) for items in split.values()), default=0)
    return train, valid, itemnum


_SAMPLE_CACHE = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    if key not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[key] = [u for u, items in user_train.items() if len(items) >= 2]
    users = _SAMPLE_CACHE[key]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length-1:-1]
        pos[row, -length:] = history[-length:]
        # Preserve the family-specific diagnostic sampling interface.
        seen = set(history)
        if len(seen) < itemnum:
            for column in range(maxlen - length, maxlen):
                candidate = random.randint(1, itemnum)
                while candidate in seen:
                    candidate = random.randint(1, itemnum)
                neg[row, column] = candidate
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        return self.gru(self.dropout(self.item_emb(sequence)))[0]

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).amax(1).clamp_min(0)
        return features[torch.arange(sequence.shape[0], device=sequence.device), last]

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        if items.ndim == 1 and features.ndim == 2:
            return features @ embeddings.T
        return (features * embeddings).sum(-1)

    def forward(self, sequence, pos, neg):
        features = self.log2feats(sequence)
        return (features * self.item_emb(pos)).sum(-1), (features * self.item_emb(neg)).sum(-1)

    def score_candidates(self, sequence, candidates):
        return self.predict_score(self.encode(sequence), candidates).squeeze(0)


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

    def forward(self, *args):
        return self.model(*args)


def exact_cross_entropy_backward(model, sequence, targets):
    # Split the chain rule at the selected GRU features. Each softmax chunk
    # immediately releases its catalog logits, while accumulating exact output
    # embedding gradients and feature gradients. Then backpropagate through
    # the GRU once, including the input side of the tied embedding matrix.
    features = model.log2feats(sequence)
    mask = targets.ne(0)
    selected = features[mask]
    labels = targets[mask] - 1
    count = labels.numel()
    if count == 0:
        raise RuntimeError('Empty target batch')
    leaf = selected.detach().requires_grad_(True)
    total = 0.0
    for start in range(0, count, SOFTMAX_CHUNK):
        end = min(start + SOFTMAX_CHUNK, count)
        logits = leaf[start:end] @ model.item_emb.weight[1:].T
        loss = F.cross_entropy(logits.float(), labels[start:end], reduction='sum') / count
        loss.backward()
        total += float(loss.detach())
        del logits, loss
    selected.backward(leaf.grad)
    return total


def make_validation(train, valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    rows = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | {0, target}
        required = min(99, itemnum - len(excluded - {0}))
        candidates = [target]
        while len(candidates) - 1 < required:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                excluded.add(candidate)
                candidates.append(candidate)
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            sequence[-len(history):] = history
        rows.append((sequence, candidates))
    return rows


def evaluate_val(model, rows, device):
    hits = 0.0
    ndcg = 0.0
    model.eval()
    with torch.no_grad():
        for sequence, candidates in rows:
            seq = torch.as_tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand)
            # Stable descending ordering gives the target its original position
            # when scores tie, matching a stable candidate ranking.
            rank = int(torch.argsort(scores, descending=True, stable=True).eq(0).nonzero()[0, 0])
            if rank < 10:
                hits += 1
                ndcg += 1 / math.log2(rank + 2)
    model.train()
    n = max(1, len(rows))
    return hits / n, ndcg / n


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not train:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    rows = make_validation(train, valid, itemnum)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            sequence, targets, _ = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            total_loss += exact_cross_entropy_backward(model, sequence, targets)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, rows, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if (hr, ndcg) > (best_hr, best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    if not os.path.isfile(checkpoint):
        torch.save(model.state_dict(), checkpoint)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                         num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE), f)
    _SAMPLE_CACHE.pop(id(train), None)
    print(f'[{dataset_name}] selected_epoch={best_epoch} best_val_HR@10={best_hr:.4f}', flush=True)
    return max(best_hr, 0.0)


def write_predict_script(working_dir):
    script = r'''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('gru4rec_trained_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset in _DATASETS:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = module.GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = int(meta['maxlen'])
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
