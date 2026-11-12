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
import torch.nn as nn
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
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_training():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    itemnum = max((max(items) for split in (train, valid) for items in split.values() if items), default=0)
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path, 'r') as handle:
            stats = json.load(handle)
        if name in stats:
            itemnum = max(itemnum, int(stats[name]['itemnum']))
    return train, valid, itemnum


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


class FilterBlock(nn.Module):
    def __init__(self, maxlen, hidden, dropout):
        super().__init__()
        self.filter = FrequencyFilter(maxlen, hidden, dropout)
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden, eps=1e-8)

    def forward(self, x, mask):
        x = self.filter(x) * mask
        residual = self.fc2(F.gelu(self.fc1(x)))
        return self.norm(x + self.dropout(residual)) * mask


class FMLPRec(nn.Module):
    def __init__(self, itemnum, hidden=64, blocks=2, maxlen=128, dropout=0.3):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.emb_norm = nn.LayerNorm(hidden, eps=1e-8)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([FilterBlock(maxlen, hidden, dropout) for _ in range(blocks)])
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.item_emb(seq) + self.pos_emb(positions).unsqueeze(0)
        x = self.emb_dropout(self.emb_norm(x)) * mask
        for block in self.blocks:
            x = block(x, mask)
        return x[:, -1, :]

    def forward(self, seq, positive, negative):
        features = self.encode(seq)
        pos_logits = (features * self.item_emb(positive)).sum(-1)
        neg_logits = (features * self.item_emb(negative)).sum(-1)
        return pos_logits, neg_logits

    def predict(self, seq, candidates):
        features = self.encode(seq)
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


class PrefixSampler:
    def __init__(self, train, itemnum, maxlen):
        histories = [items for items in train.values() if len(items) >= 2 and len(set(items)) < itemnum]
        if not histories:
            raise ValueError('No eligible training histories')
        self.lengths = np.asarray([len(items) for items in histories], dtype=np.int64)
        self.offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(self.lengths)[:-1]])
        self.flat = np.asarray([item for items in histories for item in items], dtype=np.int64)
        self.seen = [set(items) for items in histories]
        self.relative = np.arange(maxlen, dtype=np.int64) - maxlen
        self.itemnum = itemnum
        self.rng = np.random.default_rng(42)
        self.complements = {}

    def sample(self, size):
        users = self.rng.integers(len(self.lengths), size=size)
        lengths = self.lengths[users]
        # A cutoff c uses only history[:c] and predicts history[c].
        cutoffs = self.rng.integers(np.ones(size, dtype=np.int64), lengths)
        relative = cutoffs[:, None] + self.relative[None, :]
        positions = self.offsets[users, None] + np.maximum(relative, 0)
        seq = self.flat[positions].copy()
        seq[relative < 0] = 0
        positives = self.flat[self.offsets[users] + cutoffs]
        negatives = self.rng.integers(1, self.itemnum + 1, size=size)
        for index, user in enumerate(users):
            seen = self.seen[user]
            if len(seen) > self.itemnum // 2:
                if user not in self.complements:
                    self.complements[user] = np.asarray([i for i in range(1, self.itemnum + 1) if i not in seen], dtype=np.int64)
                negatives[index] = self.rng.choice(self.complements[user])
            else:
                while int(negatives[index]) in seen:
                    negatives[index] = self.rng.integers(1, self.itemnum + 1)
        return seq, positives, negatives


def validation_examples(train, valid, itemnum, maxlen):
    # Local RNG ensures validation never changes training's random stream.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        seq = [0] * maxlen
        hist = train[user][-maxlen:]
        if hist:
            seq[-len(hist):] = hist
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError('Validation requires 99 unique negative candidates')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError('No validation examples')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, examples, device):
    model.eval()
    sequences, candidates = examples
    ranks = []
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start + 64], device=device)
        cand = torch.as_tensor(candidates[start:start + 64], device=device)
        scores = model.predict(seq, cand).cpu().numpy()
        for row in scores:
            # Stable ties favor the target, matching candidate-order ranking.
            ranks.append(int(np.sum(row[1:] > row[0])))
    ranks = np.asarray(ranks)
    hr = float(np.mean(ranks < 10))
    ndcg = float(np.mean(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0)))
    return hr, ndcg


def train_one(name, device):
    seed_training()
    train, valid, itemnum = load_dataset(name)
    print(f'[{name}] users={len(train)} items={itemnum}', flush=True)
    sampler = PrefixSampler(train, itemnum, MAXLEN)
    examples = validation_examples(train, valid, itemnum, MAXLEN)
    meta = dict(itemnum=itemnum, hidden=HIDDEN, blocks=BLOCKS, maxlen=MAXLEN, dropout=DROPOUT)
    model = FMLPRec(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD)
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best = (-1.0, -1.0)
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for step in range(NUM_BATCHES):
            arrays = sampler.sample(BATCH_SIZE)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            pos_logits, neg_logits = model(seq, pos, neg)
            loss = F.softplus(-pos_logits).mean() + F.softplus(neg_logits).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {name} at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.detach().item()
        if epoch % 5 == 0 or epoch == EPOCHS:
            metric = evaluate(model, examples, device)
            print(f'[{name}] epoch={epoch} loss={total / NUM_BATCHES:.5f} val_HR@10={metric[0]:.4f} val_NDCG@10={metric[1]:.4f}', flush=True)
            if metric > best:
                best = metric
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{name}] early stopping at epoch {epoch}', flush=True)
                break
    with open(os.path.join(WORKING, name + '_meta.pkl'), 'wb') as handle:
        pickle.dump(meta, handle)
    print(f'[{name}] best validation HR@10={best[0]:.4f} epoch={best_epoch}', flush=True)
    del optimizer, model
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
_MODELS = {}
_METAS = {}
_CLASS = None
_DEVICE = None


def _load(dataset):
    global _CLASS, _DEVICE
    if dataset in _MODELS:
        return
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location('fmlprec_trial_train', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.FMLPRec
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = _CLASS(**meta).to(_DEVICE)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODELS[dataset] = model
    _METAS[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    _load(dataset)
    if len(candidates) == 0:
        return []
    model = _MODELS[dataset]
    meta = _METAS[dataset]
    maxlen = meta['maxlen']
    hist = list(history)[-maxlen:]
    seq = [0] * maxlen
    if hist:
        seq[-len(hist):] = [int(i) if 0 < int(i) <= meta['itemnum'] else 0 for i in hist]
    cand = [int(i) for i in candidates]
    valid = [0 < i <= meta['itemnum'] for i in cand]
    safe = [i if ok else 0 for i, ok in zip(cand, valid)]
    with torch.inference_mode():
        seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
        cand_t = torch.tensor([safe], dtype=torch.long, device=_DEVICE)
        scores = model.predict(seq_t, cand_t)[0].cpu().tolist()
    return [float(score) if ok else -1e9 for score, ok in zip(scores, valid)]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(script)


def main():
    seed_training()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    print(f'FMLP-Rec device={device}; fixed provided splits; training seed=42', flush=True)
    scores = [train_one(name, device) for name in DATASETS]
    write_predict_script(WORKING)
    print(f'Macro validation HR@10={np.mean(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
