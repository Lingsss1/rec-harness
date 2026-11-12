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
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 25
FULL_CATALOG_LIMIT = 30000
NEGATIVE_SAMPLES = 2048
POSITION_CHUNK = 128
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_USERS_CACHE = {}


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


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    if key not in _USERS_CACHE:
        _USERS_CACHE[key] = [u for u, items in user_train.items() if len(items) >= 2]
    users = _USERS_CACHE[key]
    if not users:
        raise RuntimeError('No users with two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        # Preserve the family-specific diagnostic batch interface.
        seen = set(history)
        if len(seen) < itemnum:
            for col in range(maxlen - length, maxlen):
                candidate = random.randint(1, itemnum)
                while candidate in seen:
                    candidate = random.randint(1, itemnum)
                neg[row, col] = candidate
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
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
        output, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return output

    def encode(self, seq):
        output = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp_min(0)
        return output[torch.arange(seq.shape[0], device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embedding = self.item_emb(items)
        if items.ndim == 1:
            return features @ embedding.transpose(0, 1)
        return (features.unsqueeze(-2) * embedding).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

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


def categorical_backward(model, seq, pos):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    count = targets.numel()
    if count == 0:
        raise RuntimeError('Empty shifted-target batch')
    # Retain the original token objective, with a small user-balanced
    # recency component to emphasize prediction near the history endpoint.
    with torch.no_grad():
        positions = torch.arange(pos.shape[1], device=pos.device).expand_as(pos)
        last = positions.masked_fill(~mask, -1).amax(dim=1, keepdim=True)
        age = (last - positions).clamp_min(0).float()
        recent = torch.exp(-math.log(2.0) * age / 16.0) * mask.float()
        recent = recent / recent.sum(dim=1, keepdim=True).clamp_min(1e-12)
        active_users = mask.any(dim=1).sum().clamp_min(1)
        weights = 0.8 / count + 0.2 * recent[mask] / active_users
    # Chunk output-layer graphs, then traverse the recurrent graph once.
    proxy = features.detach().requires_grad_(True)
    sampled = model.itemnum > FULL_CATALOG_LIMIT
    if sampled:
        # q(i)=1/itemnum for every nonpadding item, sampled with replacement.
        negatives = torch.randint(1, model.itemnum + 1, (NEGATIVE_SAMPLES,), device=seq.device)
        correction = math.log(NEGATIVE_SAMPLES / model.itemnum)
    total = 0.0
    for start in range(0, count, POSITION_CHUNK):
        stop = min(start + POSITION_CHUNK, count)
        h = proxy[start:stop]
        target = targets[start:stop]
        if not sampled:
            logits = h @ model.item_emb.weight[1:].T
            per_position = F.cross_entropy(logits, target - 1, reduction='none')
        else:
            positive = (h * model.item_emb(target)).sum(-1, keepdim=True)
            negative_logits = h @ model.item_emb(negatives).T
            negative_logits = negative_logits - correction
            negative_logits = negative_logits.masked_fill(target[:, None].eq(negatives[None, :]), float('-inf'))
            logits = torch.cat((positive, negative_logits), dim=1)
            per_position = F.cross_entropy(logits, torch.zeros(stop - start, dtype=torch.long, device=seq.device), reduction='none')
        loss = (per_position * weights[start:stop]).sum()
        loss.backward()
        total += float(loss.detach())
    features.backward(proxy.grad)
    return total


def build_validation(user_train, user_valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    records = []
    for user in users:
        target = user_valid[user][0]
        history = user_train[user]
        excluded = set(history) | {0, target}
        candidates = [target]
        needed = min(99, itemnum - len(excluded - {0}))
        while len(candidates) < needed + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                candidates.append(item)
                excluded.add(item)
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        if history:
            tail = history[-MAXLEN:]
            sequence[-len(tail):] = tail
        records.append((sequence, candidates))
    return records


@torch.no_grad()
def evaluate_val(model, records, device):
    model.eval()
    hits = ndcg = 0.0
    for sequence, candidates in records:
        seq = torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items)
        rank = int((scores[1:] > scores[0]).sum().item())
        hits += rank < 10
        if rank < 10:
            ndcg += 1.0 / math.log2(rank + 2)
    n = max(1, len(records))
    model.train()
    return hits / n, ndcg / n


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} loss=' + ('full CE' if itemnum <= FULL_CATALOG_LIMIT else 'corrected sampled CE'), flush=True)
    records = build_validation(user_train, user_valid, itemnum)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            total_loss += categorical_backward(model, seq, pos)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, records, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if (hr, ndcg) > (best_hr, best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint selected')
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                         num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                         best_epoch=best_epoch), f)
    _USERS_CACHE.pop(id(user_train), None)
    print(f'[{dataset_name}] selected epoch={best_epoch} HR@10={best_hr:.4f}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture to avoid import-time training dependencies.
    prefix = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    suffix = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_loaded(dataset_name):
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    dataset_name = dataset_name or DATASETS[0]
    _ensure_loaded(dataset_name)
    if len(candidates) == 0:
        return []
    model = _models[dataset_name]
    maxlen = _metas[dataset_name]['maxlen']
    tail = list(history[-maxlen:])
    sequence = [0] * (maxlen - len(tail)) + tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    return [float(x) for x in model.score_candidates(seq, items).cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(prefix + inspect.getsource(GRU4RecModel) + '\n' + suffix)


def main():
    torch.set_num_threads(8)
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    write_predict_script(WORKING_DIR)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
