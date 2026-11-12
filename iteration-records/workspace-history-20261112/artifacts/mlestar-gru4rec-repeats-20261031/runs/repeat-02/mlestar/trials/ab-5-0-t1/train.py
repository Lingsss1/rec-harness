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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
SEED = 42
MAXLEN = 50
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 64
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('SASREC_EPOCHS', '200')), int(os.environ.get('GRU4REC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 30
EXACT_CATALOG_LIMIT = 8000
NEGATIVE_COUNT = 512
POSITION_CHUNK = 128
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
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
    itemnum = 0
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((i for split in (train, valid) for items in split.values() for i in items), default=0)
    return train, valid, itemnum


class BatchSampler:
    def __init__(self, train, itemnum):
        self.histories = [items for items in train.values() if len(items) >= 2]
        self.itemnum = itemnum
        if not self.histories:
            raise RuntimeError('No training histories with at least two interactions')

    def sample(self, batch_size, maxlen, need_neg=False):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        for row in range(batch_size):
            history = random.choice(self.histories)
            n = min(maxlen, len(history) - 1)
            seq[row, -n:] = history[-n-1:-1]
            pos[row, -n:] = history[-n:]
            if need_neg:
                excluded = set(history)
                if len(excluded) >= self.itemnum:
                    raise RuntimeError('No available negative item')
                for col in range(maxlen-n, maxlen):
                    item = random.randint(1, self.itemnum)
                    while item in excluded:
                        item = random.randint(1, self.itemnum)
                    neg[row, col] = item
        return seq, pos, neg


def sample_batch(user_train, itemnum, batch_size, maxlen):
    return BatchSampler(user_train, itemnum).sample(batch_size, maxlen, need_neg=True)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.dropout = nn.Identity()
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        return self.gru(self.dropout(self.item_emb(sequence)))[0]

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1).clamp_min(0)
        result = features[torch.arange(sequence.size(0), device=sequence.device), last]
        return result * sequence.ne(0).any(dim=1, keepdim=True)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    def forward(self, sequence, pos, neg):
        features = self.log2feats(sequence)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, sequence, candidates):
        features = self.encode(sequence)
        if candidates.ndim == 1:
            scores = features @ self.item_emb(candidates).T
        else:
            scores = self.predict_score(features, candidates)
        return scores.squeeze(0)


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


def categorical_backward(model, sequence, targets, exact):
    selected = model.log2feats(sequence)[targets.ne(0)]
    labels = targets[targets.ne(0)]
    count = labels.numel()
    if not count:
        raise RuntimeError('Empty target batch')
    # Accumulate output gradients per chunk and traverse the recurrent graph once.
    leaf = selected.detach().requires_grad_(True)
    if not exact:
        negatives = torch.randint(1, model.itemnum + 1, (NEGATIVE_COUNT,), device=sequence.device)
        log_kq = math.log(NEGATIVE_COUNT / model.itemnum)
    total = 0.0
    for start in range(0, count, POSITION_CHUNK):
        end = min(start + POSITION_CHUNK, count)
        h = leaf[start:end]
        y = labels[start:end]
        if exact:
            logits = h @ model.item_emb.weight[1:].T
            loss = F.cross_entropy(logits.float(), y - 1, reduction='sum') / count
        else:
            # Draws are IID uniform with replacement; retain duplicate negatives.
            # Rebuild embedding lookups per chunk to release each output graph.
            positive = (h * model.item_emb(y)).sum(-1, keepdim=True)
            negative = h @ model.item_emb(negatives).T - log_kq
            negative = negative.masked_fill(y[:, None].eq(negatives[None, :]), -torch.inf)
            logits = torch.cat((positive, negative), dim=1)
            loss = F.cross_entropy(logits.float(), torch.zeros(y.size(0), dtype=torch.long, device=y.device), reduction='sum') / count
        total += float(loss.detach())
        loss.backward()
        del logits, loss
    selected.backward(leaf.grad)
    return total


def validation_cases(train, valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | {target, 0}
        candidates = [target]
        required = min(99, itemnum - sum(1 <= i <= itemnum for i in excluded))
        while len(candidates) < required + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            sequence[-len(history):] = history
        cases.append((sequence, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hr = ndcg = 0.0
    for sequence, candidates in cases:
        seq = torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)
        cand = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, cand).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        if rank < 10:
            hr += 1
            ndcg += 1 / math.log2(rank + 2)
    model.train()
    n = max(1, len(cases))
    return hr / n, ndcg / n


def train_one_dataset(dataset_name, device):
    seed_all()
    train, valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    if itemnum <= 0:
        raise RuntimeError('Empty vocabulary for ' + dataset_name)
    sampler = BatchSampler(train, itemnum)
    cases = validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError('No validation cases for ' + dataset_name)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    exact = itemnum <= EXACT_CATALOG_LIMIT
    loss_name = 'exact_ce' if exact else 'sampled_ce_512'
    print(f'[{dataset_name}] users={len(train)} items={itemnum} loss={loss_name}', flush=True)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best = (-1.0, -1.0)
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sampler.sample(BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = categorical_backward(model, seq, pos, exact)
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite categorical loss')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total / NUM_BATCHES:.5f} val_HR@10={metrics[0]:.5f} val_NDCG@10={metrics[1]:.5f}', flush=True)
            if metrics > best:
                best = metrics
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                         num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                         loss=loss_name, best_epoch=best_epoch), f)
    print(f'[{dataset_name}] selected_epoch={best_epoch} best_val_HR@10={best[0]:.5f}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best[0]


def write_predict_script(working_dir):
    model_source = inspect.getsource(GRU4RecModel)
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    footer = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds in _DATASETS:
        with open(os.path.join(_HERE, ds + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, ds + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _models:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + model_source + '\n' + footer)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
