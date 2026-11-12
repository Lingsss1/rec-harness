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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = float(os.environ.get('NEXTITNET_DROPOUT', '0.2'))
LR = float(os.environ.get('NEXTITNET_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('NEXTITNET_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('NEXTITNET_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = max(1, int(os.environ.get('NEXTITNET_EVAL_EVERY', '5')))
PATIENCE = int(os.environ.get('NEXTITNET_PATIENCE', '20'))
MULTI_NEGATIVE_DATASETS = {'Movies_and_TV', 'Industrial_and_Scientific'}
_SAMPLER_CACHE = {}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    maximum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, dataset_name + '_' + split + '.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    fields = line.split()
                    if len(fields) == 2:
                        maximum = max(maximum, int(fields[1]))
    return maximum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        users = [u for u, h in user_train.items() if len(h) >= 2]
        seen = {u: np.asarray(sorted(set(user_train[u])), dtype=np.int64) for u in users}
        users = [u for u in users if len(seen[u]) < itemnum]
        _SAMPLER_CACHE[key] = (user_train, users, seen)
    _, users, seen = _SAMPLER_CACHE[key]
    if not users:
        raise RuntimeError('No eligible training users with available negatives')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        count = min(maxlen, len(history) - 1)
        seq[row, -count:] = history[-count - 1:-1]
        pos[row, -count:] = history[-count:]
        draws = np.random.randint(1, itemnum + 1, size=(count, num_negatives))
        rejected = np.isin(draws, seen[user])
        while rejected.any():
            draws[rejected] = np.random.randint(1, itemnum + 1, size=int(rejected.sum()))
            rejected = np.isin(draws, seen[user])
        neg[row, -count:] = draws
    return seq, pos, neg[..., 0] if num_negatives == 1 else neg


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(nn.functional.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
            for i in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device)[None].expand(batch, -1) + 1
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        encoded = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = encoded[torch.arange(seq.shape[0], device=seq.device), last]
        if candidates.ndim == 1:
            return (features @ self.item_emb(candidates).T).squeeze(0)
        return self.predict_score(features, candidates)


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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def ranking_terms(positive, negative):
    if negative.ndim == 1:
        pairwise = F.softplus(negative - positive)
        pointwise = F.softplus(-positive) + F.softplus(negative)
        return 0.5 * pairwise + 0.5 * pointwise
    weights = torch.softmax(negative.detach() / 2.0, dim=-1)
    pairwise = (weights * F.softplus(negative - positive[:, None])).sum(-1)
    pointwise = F.softplus(-positive) + (weights * F.softplus(negative)).sum(-1)
    return 0.5 * pairwise + 0.5 * pointwise


def bpr_sequence_loss(model, seq, pos, neg):
    features = model.log2feats(seq)
    valid = pos != 0
    features = features[valid]
    return ranking_terms(model.predict_score(features, pos[valid]),
                         model.predict_score(features, neg[valid])).mean()


def train_batch(model, optimizer, seq, pos, neg):
    optimizer.zero_grad(set_to_none=True)
    if neg.ndim == 2:
        loss = bpr_sequence_loss(model, seq, pos, neg)
        loss.backward()
        total = float(loss.detach())
    else:
        # Accumulate the exact mean objective in scoring chunks to bound memory.
        encoded = model.log2feats(seq)
        valid = pos != 0
        features = encoded[valid]
        detached = features.detach().requires_grad_(True)
        positive_ids, negative_ids = pos[valid], neg[valid]
        count = features.shape[0]
        total = 0.0
        for start in range(0, count, 512):
            end = min(start + 512, count)
            f = detached[start:end]
            p = model.predict_score(f, positive_ids[start:end])
            n = model.predict_score(f, negative_ids[start:end])
            loss = ranking_terms(p, n).sum() / count
            loss.backward()
            total += float(loss.detach())
        features.backward(detached.grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return total


def _history_to_tensor(history, maxlen, device):
    array = np.zeros(maxlen, dtype=np.int64)
    recent = history[-maxlen:]
    if recent:
        array[-len(recent):] = recent
    return torch.as_tensor(array[None], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        size = min(100, 1 + itemnum - len(excluded - {0}))
        candidates = [target]
        while len(candidates) < size:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        cases.append((user, candidates))
    return cases


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen,
                 num_users=200, cases=None):
    if cases is None:
        cases = make_validation_cases(user_train, user_valid, itemnum, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user, candidates in cases:
            seq = _history_to_tensor(user_train[user], maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for cutoff in (10, 20):
                if rank < cutoff:
                    totals['HR@' + str(cutoff)] += 1
                    totals['NDCG@' + str(cutoff)] += 1 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty training dataset: ' + dataset_name)
    negatives = 64 if dataset_name in MULTI_NEGATIVE_DATASETS else 1
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, dataset_name + '_model.pt')
    cases = make_validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    best_hr, best_epoch = -1.0, 0
    print(f'[{dataset_name}] users={len(train)} items={itemnum} negatives={negatives}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN, negatives)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            total += train_batch(model, optimizer, seq, pos, neg)
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, train, valid, itemnum, device, MAXLEN, cases=cases)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total / NUM_BATCHES:.5f} validation={metrics}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(working, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    _SAMPLER_CACHE.clear()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    prefix = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
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
    for name in DATASETS:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = NextItNetModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[name] = model
        _metas[name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    name = DATASETS[0] if dataset_name is None else dataset_name
    if name not in _models:
        raise ValueError('Unknown dataset_name: ' + str(name))
    if not candidates:
        return []
    model = _models[name]
    maxlen = int(_metas[name]['maxlen'])
    sequence = [0] * maxlen
    recent = list(history[-maxlen:])
    if recent:
        sequence[-len(recent):] = recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(prefix + architecture + '\n\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working)
    scores = [train_one_dataset(name, device, DATA_DIR) for name in DATASETS]
    print(f'Macro-average validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
