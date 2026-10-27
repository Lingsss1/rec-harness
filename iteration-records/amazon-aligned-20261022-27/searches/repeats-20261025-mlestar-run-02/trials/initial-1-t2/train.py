from __future__ import annotations

import os
import json
import math
import pickle
import random
import inspect
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
WEIGHT_DECAY = float(os.environ.get('SASREC_WD', '0.0'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
STEPS = int(os.environ.get('SASREC_BATCHES', '200'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
NEGATIVES = 32
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    itemnum = max((max(items) for items in train.values() if items), default=0)
    itemnum = max(itemnum, max((max(items) for items in valid.values() if items), default=0))
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = max(itemnum, int(stats[name]['itemnum']))
    return train, valid, itemnum


class FilterLayer(nn.Module):
    def __init__(self, maxlen, hidden_units, dropout_rate):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(maxlen // 2 + 1, hidden_units, 2) * 0.02)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, x):
        # FFT must stay float32, including under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=1, norm='ortho')
            weight = torch.view_as_complex(self.complex_weight.float().contiguous())
            filtered = torch.fft.irfft(spectrum * weight.unsqueeze(0), n=x.shape[1], dim=1, norm='ortho')
        return self.norm(x + self.dropout(filtered.to(x.dtype)))


class Intermediate(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.dense1 = nn.Linear(hidden_units, 4 * hidden_units)
        self.dense2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, x):
        y = self.dense2(F.gelu(self.dense1(x)))
        return self.norm(x + self.dropout(y))


class FMLPRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, dropout_rate=0.3, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.input_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.input_dropout = nn.Dropout(dropout_rate)
        self.filters = nn.ModuleList([FilterLayer(maxlen, hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.intermediates = nn.ModuleList([Intermediate(hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1)
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.item_emb(seq) + self.pos_emb(positions).unsqueeze(0)
        x = self.input_dropout(self.input_norm(x)) * mask
        for frequency_filter, intermediate in zip(self.filters, self.intermediates):
            x = intermediate(frequency_filter(x)) * mask
        return x[:, -1, :]

    def forward(self, seq, positive, negatives):
        features = self.encode(seq)
        positive_logits = (features * self.item_emb(positive)).sum(-1)
        negative_logits = (features.unsqueeze(1) * self.item_emb(negatives)).sum(-1)
        return positive_logits, negative_logits

    def predict(self, seq, candidates):
        features = self.encode(seq)
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


class PrefixSampler:
    def __init__(self, train, itemnum, seed=42):
        self.itemnum = itemnum
        self.rng = np.random.RandomState(seed)
        self.histories = []
        self.forbidden = []
        self.allowed = []
        for user, history in train.items():
            seen = set(history)
            if len(history) < 2 or len(seen) >= itemnum:
                continue
            self.histories.append(np.asarray(history, dtype=np.int64))
            self.forbidden.append(np.asarray(sorted(seen), dtype=np.int64))
            # Avoid rejection-sampling stalls for unusually dense users.
            if len(seen) > itemnum // 2:
                self.allowed.append(np.setdiff1d(np.arange(1, itemnum + 1), self.forbidden[-1]))
            else:
                self.allowed.append(None)
        if not self.histories:
            raise ValueError('No eligible training prefixes')

    def batch(self):
        seq = np.zeros((BATCH, MAXLEN), dtype=np.int64)
        pos = np.empty(BATCH, dtype=np.int64)
        neg = self.rng.randint(1, self.itemnum + 1, size=(BATCH, NEGATIVES))
        users = self.rng.randint(0, len(self.histories), size=BATCH)
        for row, index in enumerate(users):
            history = self.histories[index]
            cutoff = self.rng.randint(1, len(history))
            prefix = history[max(0, cutoff - MAXLEN):cutoff]
            seq[row, -len(prefix):] = prefix
            pos[row] = history[cutoff]
            allowed = self.allowed[index]
            if allowed is not None:
                neg[row] = self.rng.choice(allowed, NEGATIVES, replace=True)
            else:
                bad = np.isin(neg[row], self.forbidden[index])
                while bad.any():
                    neg[row, bad] = self.rng.randint(1, self.itemnum + 1, size=int(bad.sum()))
                    bad = np.isin(neg[row], self.forbidden[index])
        return seq, pos, neg


def make_validation(train, valid, itemnum):
    # Local RNG keeps the benchmark candidates fixed without changing training RNG.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            sequence[-len(history):] = history
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError('Validation requires 99 distinct eligible negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(sequence)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError('No validation users')
    return np.stack(sequences), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    ranks = []
    for start in range(0, len(sequences), 128):
        seq = torch.as_tensor(sequences[start:start + 128], device=device)
        cand = torch.as_tensor(candidates[start:start + 128], device=device)
        scores = model.predict(seq, cand)
        # Target is first; strict comparison implements stable tie handling.
        ranks.extend((scores[:, 1:] > scores[:, :1]).sum(-1).cpu().tolist())
    ranks = np.asarray(ranks)
    hr = float(np.mean(ranks < 10))
    ndcg = float(np.mean(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0)))
    model.train()
    return hr, ndcg


def train_one_dataset(name, device):
    seed_everything(42)
    train, valid, itemnum = load_dataset(name)
    sampler = PrefixSampler(train, itemnum, seed=42)
    validation = make_validation(train, valid, itemnum)
    meta = dict(architecture='fmlp', itemnum=itemnum, hidden_units=HIDDEN,
                num_blocks=BLOCKS, dropout_rate=DROPOUT, maxlen=MAXLEN)
    model = FMLPRec(itemnum, HIDDEN, BLOCKS, DROPOUT, MAXLEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best = (-1.0, -1.0)
    best_epoch = 0
    print(f'[{name}] users={len(train)} items={itemnum} architecture=fmlp', flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step in range(STEPS):
            seq, pos, neg = sampler.batch()
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            optimizer.zero_grad(set_to_none=True)
            positive_logits, negative_logits = model(seq, pos, neg)
            loss = F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={total_loss / STEPS:.5f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{name}] early stop; best epoch={best_epoch}', flush=True)
                break
    with open(os.path.join(WORKING, name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best[0]


def write_predict_script():
    # Embed architecture code so inference needs no training data or train.py import.
    source = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n'
    source += inspect.getsource(FilterLayer) + '\n\n'
    source += inspect.getsource(Intermediate) + '\n\n'
    source += inspect.getsource(FMLPRec) + '\n\n'
    source += '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_METAS = {}
_DEVICE = None


def _load(dataset_name):
    global _DEVICE
    if dataset_name in _MODELS:
        return _MODELS[dataset_name], _METAS[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _DEVICE is None:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    if meta['architecture'] != 'fmlp':
        raise ValueError('Unsupported checkpoint architecture')
    model = FMLPRec(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                    num_blocks=meta['num_blocks'], dropout_rate=meta['dropout_rate'],
                    maxlen=meta['maxlen']).to(_DEVICE)
    weights = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                         map_location=_DEVICE, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    _MODELS[dataset_name] = model
    _METAS[dataset_name] = meta
    return model, meta


@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model, meta = _load(dataset_name)
    if len(candidates) == 0:
        return []
    maxlen = meta['maxlen']
    hist = [int(item) for item in history[-maxlen:]]
    seq = [0] * (maxlen - len(hist)) + hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_t, cand_t)[0].cpu().tolist()
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as f:
        f.write(source)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything(42)
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    write_predict_script()
    scores = []
    for name in DATASETS:
        scores.append(train_one_dataset(name, device))
    print('Macro validation HR@10 = {:.5f}'.format(sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
