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
from torch import nn
from torch.nn import functional as F

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
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '30'))
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
    with open(path, 'r') as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    itemnum = max((max(items) for split in (train, valid) for items in split.values() if items), default=0)
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as handle:
            stats = json.load(handle)
        if name in stats:
            itemnum = max(itemnum, int(stats[name]['itemnum']))
    if itemnum < 1:
        raise ValueError('Empty item vocabulary: ' + name)
    return train, valid, itemnum


class FilterLayer(nn.Module):
    def __init__(self, width, maxlen, dropout):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.empty(1, maxlen // 2 + 1, width, 2))
        nn.init.normal_(self.complex_weight, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width, eps=1e-12)

    def forward(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            spectrum = torch.fft.rfft(inputs.float(), dim=1, norm='ortho')
            weight = torch.view_as_complex(self.complex_weight.float().contiguous())
            filtered = torch.fft.irfft(spectrum * weight, n=inputs.shape[1], dim=1, norm='ortho')
        return self.norm(inputs + self.dropout(filtered.to(inputs.dtype)))


class Intermediate(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.dense1 = nn.Linear(width, 4 * width)
        self.dense2 = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width, eps=1e-12)

    def forward(self, inputs):
        outputs = self.dense2(F.gelu(self.dense1(inputs)))
        return self.norm(inputs + self.dropout(outputs))


class FMLPRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, maxlen=128, dropout_rate=0.3):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.filters = nn.ModuleList([FilterLayer(hidden_units, maxlen, dropout_rate) for _ in range(num_blocks)])
        self.intermediates = nn.ModuleList([Intermediate(hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.apply(self._initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        mask = seq.ne(0).unsqueeze(-1)
        positions = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        x = self.emb_dropout(self.emb_norm(self.item_emb(seq) + self.pos_emb(positions)))
        x = x * mask
        for frequency_filter, feed_forward in zip(self.filters, self.intermediates):
            x = frequency_filter(x)
            x = feed_forward(x) * mask
        return x[:, -1, :]

    def forward(self, seq, positive, negative):
        features = self.encode(seq)
        positive_logits = (features * self.item_emb(positive)).sum(-1)
        negative_logits = torch.einsum('bd,bkd->bk', features, self.item_emb(negative))
        return positive_logits, negative_logits

    def predict(self, seq, candidates):
        features = self.encode(seq)
        return torch.einsum('bd,bkd->bk', features, self.item_emb(candidates))


class PrefixSampler:
    def __init__(self, train, itemnum, seed=42):
        self.itemnum = itemnum
        self.rng = np.random.RandomState(seed)
        self.histories = []
        self.seen = []
        self.complements = {}
        for items in train.values():
            seen = set(items)
            if len(items) >= 2 and len(seen) < itemnum:
                self.histories.append(np.asarray(items, dtype=np.int64))
                self.seen.append(seen)
        if not self.histories:
            raise ValueError('No eligible training prefixes')

    def sample(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        positive = np.empty(batch_size, dtype=np.int64)
        negative = np.empty((batch_size, NEGATIVES), dtype=np.int64)
        indices = self.rng.randint(len(self.histories), size=batch_size)
        for row, index in enumerate(indices):
            history = self.histories[index]
            # Noncausal frequency mixing supervises only a target after the prefix.
            cutoff = self.rng.randint(1, len(history))
            prefix = history[max(0, cutoff - maxlen):cutoff]
            seq[row, -len(prefix):] = prefix
            positive[row] = history[cutoff]
            seen = self.seen[index]
            if len(seen) > self.itemnum // 2:
                if index not in self.complements:
                    self.complements[index] = np.asarray([item for item in range(1, self.itemnum + 1) if item not in seen], dtype=np.int64)
                negative[row] = self.rng.choice(self.complements[index], size=NEGATIVES, replace=True)
            else:
                values = self.rng.randint(1, self.itemnum + 1, size=NEGATIVES)
                bad = np.asarray([item in seen for item in values])
                while bad.any():
                    values[bad] = self.rng.randint(1, self.itemnum + 1, size=int(bad.sum()))
                    bad = np.asarray([item in seen for item in values])
                negative[row] = values
        return seq, positive, negative


def validation_data(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [user for user in valid if user in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user] or not train[user]:
            continue
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError('Validation requires 99 distinct eligible negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        prefix = train[user][-maxlen:]
        sequences.append([0] * (maxlen - len(prefix)) + prefix)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError('No validation users')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, validation, device):
    sequences, candidates = validation
    model.eval()
    hits = 0
    ndcg = 0.0
    for offset in range(0, len(sequences), 128):
        seq = torch.as_tensor(sequences[offset:offset + 128], dtype=torch.long, device=device)
        cand = torch.as_tensor(candidates[offset:offset + 128], dtype=torch.long, device=device)
        scores = model.predict(seq, cand)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1).cpu().tolist()
        for rank in ranks:
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    return hits / len(sequences), ndcg / len(sequences)


def train_one_dataset(name, device):
    seed_everything(42)
    train, valid, itemnum = load_dataset(name)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS, maxlen=MAXLEN, dropout_rate=DROPOUT)
    model = FMLPRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    sampler = PrefixSampler(train, itemnum, 42)
    validation = validation_data(train, valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    metadata = dict(config, architecture='fmlp')
    with open(os.path.join(WORKING, name + '_meta.pkl'), 'wb') as handle:
        pickle.dump(metadata, handle)
    best = (-1.0, -1.0)
    best_epoch = 0
    print(f'[{name}] users={len(train)} items={itemnum} architecture=fmlp', flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for _ in range(STEPS):
            arrays = sampler.sample(BATCH, MAXLEN)
            seq, positive, negative = [torch.as_tensor(array, device=device, dtype=torch.long) for array in arrays]
            optimizer.zero_grad(set_to_none=True)
            pos_logits, neg_logits = model(seq, positive, negative)
            loss = F.softplus(-pos_logits).mean() + F.softplus(neg_logits).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
            loss_sum += loss.detach()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={loss_sum.item() / max(STEPS, 1):.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise ValueError('Training produced no validated checkpoint')
    print(f'[{name}] best_val_HR@10={best[0]:.5f} best_epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best[0]


def write_predict_script():
    # Embed the exact trained architecture; inference does not import train.py.
    source = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'
    for cls in (FilterLayer, Intermediate, FMLPRec):
        source += inspect.getsource(cls) + '\n\n'
    source += '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_METAS = {}
_DEVICE = None


def _load(dataset):
    global _DEVICE
    if dataset in _MODELS:
        return _MODELS[dataset], _METAS[dataset]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _DEVICE is None:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    if meta['architecture'] != 'fmlp':
        raise ValueError('Unsupported model architecture')
    config = {key: meta[key] for key in ('itemnum', 'hidden_units', 'num_blocks', 'maxlen', 'dropout_rate')}
    model = FMLPRec(**config).to(_DEVICE)
    weights = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_DEVICE, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    _MODELS[dataset] = model
    _METAS[dataset] = meta
    return model, meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model, meta = _load(dataset)
    prefix = list(history)[-meta['maxlen']:]
    prefix = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in prefix]
    sequence = [0] * (meta['maxlen'] - len(prefix)) + prefix
    safe_candidates = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in candidates]
    seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand = torch.tensor([safe_candidates], dtype=torch.long, device=_DEVICE)
    scores = model.predict(seq, cand)[0].cpu().tolist()
    return [score if item != 0 else -1e9 for score, item in zip(scores, safe_candidates)]
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as handle:
        handle.write(source)


def main():
    if min(HIDDEN, BLOCKS, MAXLEN, BATCH, STEPS, EPOCHS) < 1:
        raise ValueError('Model and training dimensions must be positive')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script()
    scores = [train_one_dataset(name, device) for name in DATASETS]
    print('Macro validation HR@10 = %.6f' % (sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
