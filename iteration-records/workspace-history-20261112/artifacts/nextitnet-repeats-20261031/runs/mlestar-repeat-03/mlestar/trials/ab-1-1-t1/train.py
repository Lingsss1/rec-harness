from __future__ import annotations
import os
import math
import json
import pickle
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = DEFAULT_DATASETS[:]
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 40
NEGATIVES = 64
CALIBRATION = 0.5


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
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


class TrainingData(dict):
    def __init__(self, data, itemnum):
        super().__init__(data)
        self.eligible = [u for u, h in data.items() if len(h) >= 2 and len(set(h)) < itemnum]
        self.seen = {u: set(data[u]) for u in self.eligible}


def sample_batch(user_train, itemnum, batch_size, maxlen, return_users=False):
    users = getattr(user_train, 'eligible', None)
    if users is None:
        users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
    if not users:
        raise RuntimeError('No eligible training users')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    chosen = []
    for b in range(batch_size):
        u = random.choice(users)
        chosen.append(u)
        h = user_train[u]
        seen = user_train.seen[u] if isinstance(user_train, TrainingData) else set(h)
        inputs = h[:-1][-maxlen:]
        targets = h[1:][-maxlen:]
        start = maxlen - len(inputs)
        seq[b, start:] = inputs
        pos[b, start:] = targets
        for j in range(start, maxlen):
            x = random.randint(1, itemnum)
            while x in seen:
                x = random.randint(1, itemnum)
            neg[b, j] = x
    if return_users:
        return seq, pos, neg, np.asarray(chosen, dtype=np.int64)
    return seq, pos, neg


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
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self._causal_conv(self.conv1, self.norm1(x).transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self._causal_conv(self.conv2, self.norm2(y).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        ds = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, ds[i], kernel_size, dropout_rate) for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        mask = seq.ne(0).unsqueeze(-1)
        positions = torch.arange(1, seq.shape[1] + 1, device=seq.device)[None, :].expand_as(seq)
        positions = positions * seq.ne(0)
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        emb = self.item_emb(items)
        if items.ndim == 1:
            return features @ emb.T
        return (features.unsqueeze(-2) * emb).sum(-1)

    def forward(self, seq, pos, neg):
        out = self.encode_sequence(seq)
        return (out * self.item_emb(pos)).sum(-1), (out * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(1).values.clamp(min=0)
        features = out[torch.arange(seq.shape[0], device=seq.device), last]
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.shape[0] == 1 else scores


class ModelView:
    def __init__(self, model):
        self.model = model
        self.item_emb = model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


class NegativeSampler:
    def __init__(self, train, itemnum, device):
        self.itemnum = itemnum
        self.stride = itemnum + 1
        keys = [u * self.stride + i for u in train.eligible for i in train.seen[u]]
        self.keys = torch.tensor(sorted(keys), dtype=torch.long, device=device)

    @torch.no_grad()
    def sample(self, users, targets, count):
        negatives = torch.randint(1, self.itemnum + 1, (len(targets), count), device=targets.device)
        while True:
            encoded = users[:, None] * self.stride + negatives
            idx = torch.searchsorted(self.keys, encoded.reshape(-1)).reshape_as(negatives)
            seen = self.keys[idx.clamp(max=len(self.keys) - 1)].eq(encoded)
            reject = seen | negatives.eq(targets[:, None])
            n = int(reject.sum().item())
            if n == 0:
                return negatives
            negatives[reject] = torch.randint(1, self.itemnum + 1, (n,), device=targets.device)


def gbce_sequence_loss(model, seq, pos, users, sampler):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    owners = users[:, None].expand_as(pos)[mask]
    alpha = NEGATIVES / max(model.itemnum - 1, 1)
    beta = 1.0 - CALIBRATION + alpha * CALIBRATION
    total = features.new_zeros((), dtype=torch.float64)
    for start in range(0, len(targets), 512):
        f = features[start:start + 512]
        p = targets[start:start + 512]
        negatives = sampler.sample(owners[start:start + 512], p, NEGATIVES)
        positive_scores = (f * model.item_emb(p)).sum(-1).double()
        negative_scores = torch.einsum('nd,nkd->nk', f, model.item_emb(negatives)).double()
        a = beta * F.logsigmoid(positive_scores)
        positive_loss = -a - torch.log(-torch.expm1(a))
        calibrated_loss = positive_loss + F.softplus(negative_scores).sum(-1)
        total = total + calibrated_loss.sum()
    return total / ((NEGATIVES + 1) * max(len(targets), 1))


def make_validation(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for u in users:
        target = valid[u][0]
        excluded = set(train[u]) | {target, 0}
        candidates = [target]
        count = min(99, itemnum - len(excluded - {0}))
        while len(candidates) < count + 1:
            x = rng.randint(1, itemnum)
            if x not in excluded:
                candidates.append(x)
                excluded.add(x)
        sequence = [0] * MAXLEN
        h = train[u][-MAXLEN:]
        if h:
            sequence[-len(h):] = h
        examples.append((sequence, candidates))
    return examples


@torch.no_grad()
def evaluate_val(model, examples, device):
    model.eval()
    hits = ndcg = 0.0
    for sequence, candidates in examples:
        seq = torch.tensor([sequence], dtype=torch.long, device=device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items)
        rank = int(torch.argsort(scores, descending=True, stable=True).eq(0).nonzero()[0].item())
        if rank < 10:
            hits += 1
            ndcg += 1 / math.log2(rank + 2)
    n = max(len(examples), 1)
    return hits / n, ndcg / n


def train_one_dataset(dataset_name, device):
    seed_all()
    raw, valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    train = TrainingData(raw, itemnum)
    if not train.eligible:
        raise RuntimeError('Empty eligible dataset: ' + dataset_name)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sampler = NegativeSampler(train, itemnum, device)
    validation = make_validation(train, valid, itemnum)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg, users = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN, return_users=True)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            users = torch.as_tensor(users, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = gbce_sequence_loss(model, seq, pos, users, sampler)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite calibrated loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, validation, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total / NUM_BATCHES:.6f} HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(working, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    return best_hr


def write_predict_script(working_dir):
    script = '''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('trained_nextitnet_definition', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ds in DATASETS:
        with open(os.path.join(_HERE, ds + '_meta.pkl'), 'rb') as f:
            config = pickle.load(f)
        model = module.NextItNetModel(**config).to(_device)
        model.load_state_dict(torch.load(os.path.join(_HERE, ds + '_model.pt'), map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = config


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError('Unknown dataset: ' + str(ds))
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
    sequence = [0] * maxlen
    h = history[-maxlen:]
    if h:
        sequence[-len(h):] = h
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        return [float(x) for x in model.score_candidates(seq, items).cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    torch.set_num_threads(4)
    seed_all()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    write_predict_script(working)
    scores = [train_one_dataset(ds, device) for ds in DATASETS]
    print('Macro validation HR@10 = %.6f' % (sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
