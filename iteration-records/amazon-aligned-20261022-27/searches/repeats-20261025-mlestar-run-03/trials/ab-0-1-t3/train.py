from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN = int(os.environ.get('SASREC_HIDDEN', '64'))
BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
DROPOUT = float(os.environ.get('SASREC_DROPOUT', '0.3'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WD = float(os.environ.get('SASREC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('SASREC_BATCH', '128'))
NUM_BATCHES = int(os.environ.get('SASREC_BATCHES', '200'))
NUM_EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path, 'r') as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    maximum = max((max(items) for items in train.values() if items), default=0)
    maximum = max(maximum, max((max(items) for items in valid.values() if items), default=0))
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path, 'r') as handle:
            stats = json.load(handle)
        if name in stats:
            maximum = max(maximum, int(stats[name]['itemnum']))
    return train, valid, maximum


class FrequencyFilter(nn.Module):
    def __init__(self, maxlen, hidden, dropout):
        super().__init__()
        self.weight = nn.Parameter(0.02 * torch.randn(1, maxlen // 2 + 1, hidden, 2))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden, eps=1e-8)

    def forward(self, x):
        spectrum = torch.fft.rfft(x.float(), dim=1, norm='ortho')
        weight = torch.view_as_complex(self.weight.contiguous())
        filtered = torch.fft.irfft(spectrum * weight, n=x.shape[1], dim=1, norm='ortho')
        return self.norm(x + self.dropout(filtered.to(x.dtype)))


class FeedForward(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.dense1 = nn.Linear(hidden, 4 * hidden)
        self.dense2 = nn.Linear(4 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden, eps=1e-8)

    def forward(self, x):
        y = self.dense2(F.gelu(self.dense1(x)))
        return self.norm(x + self.dropout(y))


class FrequencyBlock(nn.Module):
    def __init__(self, maxlen, hidden, dropout):
        super().__init__()
        self.filter = FrequencyFilter(maxlen, hidden, dropout)
        self.ffn = FeedForward(hidden, dropout)

    def forward(self, x, mask):
        x = self.filter(x) * mask
        return self.ffn(x) * mask


class FMLPRec(nn.Module):
    def __init__(self, itemnum, hidden=64, num_blocks=2, maxlen=128, dropout=0.3):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.input_norm = nn.LayerNorm(hidden, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([FrequencyBlock(maxlen, hidden, dropout) for _ in range(num_blocks)])
        self.register_buffer('positions', torch.arange(maxlen), persistent=False)
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.item_emb(seq) + self.pos_emb(self.positions).unsqueeze(0)
        x = self.dropout(self.input_norm(x)) * mask
        for block in self.blocks:
            x = block(x, mask)
        return x[:, -1, :]

    def forward(self, seq, positive, negative):
        feature = self.encode(seq)
        pos_logits = (feature * self.item_emb(positive)).sum(-1)
        neg_logits = (feature * self.item_emb(negative)).sum(-1)
        return pos_logits, neg_logits

    def predict(self, seq, candidates):
        feature = self.encode(seq)
        return (self.item_emb(candidates) * feature.unsqueeze(1)).sum(-1)


class PrefixSampler:
    def __init__(self, train, itemnum, maxlen, seed=42):
        self.histories = [np.asarray(items, dtype=np.int64) for items in train.values() if 2 <= len(items) and len(set(items)) < itemnum]
        if not self.histories:
            raise ValueError('No users with valid prefix training examples')
        self.lengths = np.asarray([len(h) for h in self.histories], dtype=np.int64)
        self.offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(self.lengths)[:-1]])
        self.flat = np.concatenate(self.histories)
        self.seen = [set(h.tolist()) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.rng = np.random.RandomState(seed)
        self.relative = np.arange(-maxlen, 0, dtype=np.int64)

    def sample(self, batch_size):
        rows = self.rng.randint(0, len(self.histories), size=batch_size)
        lengths = self.lengths[rows]
        cutoffs = 1 + (self.rng.random_sample(batch_size) * (lengths - 1)).astype(np.int64)
        relative_indices = cutoffs[:, None] + self.relative[None, :]
        indices = self.offsets[rows, None] + np.maximum(relative_indices, 0)
        seq = self.flat[indices].copy()
        seq[relative_indices < 0] = 0
        positive = self.flat[self.offsets[rows] + cutoffs]
        negative = self.rng.randint(1, self.itemnum + 1, size=batch_size)
        for j, row in enumerate(rows):
            while int(negative[j]) in self.seen[row]:
                negative[j] = self.rng.randint(1, self.itemnum + 1)
        return seq, positive, negative


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        target = valid[u][0]
        seen = set(train[u]) | set(valid[u])
        if itemnum - len(seen) < 99:
            raise ValueError('Validation requires 99 unique unseen negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in seen:
                negatives.add(item)
        hist = train[u][-maxlen:]
        sequences.append([0] * (maxlen - len(hist)) + hist)
        candidates.append([target] + list(negatives))
    return np.asarray(sequences, dtype=np.int64).reshape(-1, maxlen), np.asarray(candidates, dtype=np.int64).reshape(-1, 100)


@torch.no_grad()
def evaluate(model, validation, device):
    sequences, candidates = validation
    model.eval()
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(sequences), 128):
        seq = torch.as_tensor(sequences[start:start + 128], device=device)
        cand = torch.as_tensor(candidates[start:start + 128], device=device)
        scores = model.predict(seq, cand).cpu().numpy()
        order = np.argsort(-scores, axis=1, kind='stable')
        ranks = np.argmax(order == 0, axis=1)
        hits += float((ranks < 10).sum())
        ndcg += float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0).sum())
    n = max(len(sequences), 1)
    return hits / n, ndcg / n


def train_one_dataset(name, device):
    train, valid, itemnum = load_dataset(name)
    print(f'[{name}] users={len(train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden=HIDDEN, num_blocks=BLOCKS, maxlen=MAXLEN, dropout=DROPOUT)
    model = FMLPRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD)
    sampler = PrefixSampler(train, itemnum, MAXLEN, seed=42)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best = (-1.0, -1.0)
    stale_checks = 0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for step in range(NUM_BATCHES):
            seq, pos, neg = sampler.sample(BATCH_SIZE)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            positive, negative = model(seq, pos, neg)
            loss = F.softplus(-positive).mean() + F.softplus(negative).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Ablation: gradient norm clipping is disabled.
            optimizer.step()
            total_loss += loss.detach()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={total_loss.item() / max(NUM_BATCHES, 1):.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                stale_checks = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale_checks += 1
            if stale_checks >= PATIENCE:
                break
    if best_epoch == 0:
        best = evaluate(model, validation, device)
        torch.save(model.state_dict(), checkpoint)
    with open(os.path.join(WORKING, name + '_meta.pkl'), 'wb') as handle:
        pickle.dump({'architecture': 'FMLPRec', 'config': config, 'best_epoch': best_epoch}, handle)
    print(f'[{name}] best_epoch={best_epoch} best_val_HR@10={best[0]:.5f}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best[0]


def write_predict_script(working_dir):
    script = '''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _load_class():
    global _model_class
    if _model_class is None:
        path = os.path.abspath(os.path.join(_HERE, '..', 'train.py'))
        spec = importlib.util.spec_from_file_location('fmlprec_architecture_' + str(os.getpid()), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.FMLPRec
    return _model_class


def _ensure_loaded(dataset):
    global _device
    if dataset in _models:
        return
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    config = dict(meta['config'])
    config['dropout'] = 0.0
    model = _load_class()(**config).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta


@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    _ensure_loaded(dataset)
    model = _models[dataset]
    maxlen = _metas[dataset]['config']['maxlen']
    hist = [int(item) for item in history[-maxlen:]]
    hist = [item if 0 <= item <= model.itemnum else 0 for item in hist]
    seq = torch.tensor([[0] * (maxlen - len(hist)) + hist], dtype=torch.long, device=_device)
    ids = torch.tensor([list(candidates)], dtype=torch.long, device=_device)
    valid = (ids > 0) & (ids <= model.itemnum)
    scores = model.predict(seq, ids.clamp(0, model.itemnum))
    scores = scores.masked_fill(~valid, float('-inf'))
    return scores[0].cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(script)


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    scores = []
    for name in DATASETS:
        scores.append(train_one_dataset(name, device))
    write_predict_script(WORKING)
    print(f'Macro validation HR@10={float(np.mean(scores)):.6f}', flush=True)


if __name__ == '__main__':
    main()
