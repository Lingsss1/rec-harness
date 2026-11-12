from __future__ import annotations

import inspect
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = float(os.environ.get('NEXTITNET_DROPOUT', '0.2'))
LR = 0.001
BATCH_SIZE = int(os.environ.get('NEXTITNET_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 32
BPR_MAX_REG = 0.01
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_SAMPLER_CACHE = {}


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
        itemnum = max((i for split in (train, valid) for hist in split.values() for i in hist), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        users = [u for u, h in user_train.items() if len(h) >= 2]
        if not users:
            raise RuntimeError('No users with at least two training interactions')
        excluded = {u: np.unique(np.asarray(user_train[u], dtype=np.int64)) for u in users}
        if any(len(excluded[u]) >= itemnum for u in users):
            raise RuntimeError('A training user has no eligible negative items')
        _SAMPLER_CACHE[key] = (users, excluded)
    users, excluded = _SAMPLER_CACHE[key]
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
    for b in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        length = min(maxlen, len(history) - 1)
        seq[b, -length:] = history[-length - 1:-1]
        pos[b, -length:] = history[-length:]
        draws = np.random.randint(1, itemnum + 1, size=(length, NUM_NEGATIVES))
        bad = np.isin(draws, excluded[u])
        while bad.any():
            draws[bad] = np.random.randint(1, itemnum + 1, size=int(bad.sum()))
            bad = np.isin(draws, excluded[u])
        neg[b, -length:] = draws
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
        return conv(nn.functional.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3, dropout_rate=0.2, maxlen=128):
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
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * seq.ne(0).long()
        mask = seq.ne(0).unsqueeze(-1).to(self.item_emb.weight.dtype)
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
        if emb.ndim == features.ndim:
            return (features * emb).sum(-1)
        if emb.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * emb).sum(-1)
        if features.ndim == 1 and emb.ndim == 2:
            return emb @ features
        raise ValueError('Incompatible feature and item shapes')

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(1).values.clamp(min=0)
        rows = torch.arange(seq.size(0), device=seq.device)
        final = features[rows, last]
        emb = self.item_emb(candidates)
        if candidates.ndim == 1:
            result = final @ emb.transpose(0, 1)
        else:
            result = (final.unsqueeze(1) * emb).sum(-1)
        return result[0] if seq.size(0) == 1 else result


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


def bpr_max_loss(p, n, reg=BPR_MAX_REG):
    p, n = p.float(), n.float()
    log_weights = F.log_softmax(n, dim=-1)
    ranking = -torch.logsumexp(log_weights + F.logsigmoid(p[:, None] - n), dim=-1)
    penalty = (log_weights.exp() * n.square()).sum(-1)
    return (ranking + reg * penalty).mean()


def bpr_sequence_loss(model, seq, pos, neg):
    valid = pos.ne(0)
    features = model.log2feats(seq)[valid]
    p = model.predict_score(features, pos[valid])
    n = model.predict_score(features, neg[valid])
    return bpr_max_loss(p, n)


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        target = user_valid[u][0]
        history = user_train[u]
        seen = set(history) | {0, target}
        if itemnum - len(seen - {0}) < 99:
            raise RuntimeError('Insufficient unique validation negatives')
        cand = [target]
        while len(cand) < 100:
            item = rng.randint(1, itemnum)
            if item not in seen:
                seen.add(item)
                cand.append(item)
        sequence = [0] * maxlen
        suffix = history[-maxlen:]
        if suffix:
            sequence[-len(suffix):] = suffix
        sequences.append(sequence)
        candidates.append(cand)
    if not users:
        raise RuntimeError('No validation users')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    sequences, candidates = cases
    hits = 0.0
    ndcg = 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start + 64], device=device)
        cand = torch.as_tensor(candidates[start:start + 64], device=device)
        scores = model.score_candidates(seq, cand)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        rank = order.eq(0).long().argmax(1)
        hit = rank.lt(10)
        hits += float(hit.sum().item())
        ndcg += float((hit.float() / torch.log2(rank.float() + 2)).sum().item())
    return {'HR@10': hits / len(sequences), 'NDCG@10': ndcg / len(sequences)}


def train_one_dataset(dataset_name, device):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    if not user_train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_total = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite BPR-max loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach().item())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={loss_total / NUM_BATCHES:.5f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'[{dataset_name}] selected_epoch={best_epoch} val_HR@10={best_hr:.4f}', flush=True)
    _SAMPLER_CACHE.clear()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the actual training architecture so inference has no dependency on
    # training configuration, environment variables, or a separately edited copy.
    header = 'from __future__ import annotations\nimport math\nimport json\nimport os\nimport torch\nimport torch.nn as nn\n\n'
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
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
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.json')) as f:
        meta = json.load(f)
    model = NextItNetModel(**meta).to(_device)
    checkpoint = os.path.join(_HERE, dataset_name + '_model.pt')
    model.load_state_dict(torch.load(checkpoint, map_location=_device, weights_only=True))
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    if not candidates:
        return []
    _ensure_loaded(ds)
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    suffix = list(history[-maxlen:])
    if suffix:
        sequence[-len(suffix):] = suffix
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    scores = _models[ds].score_candidates(seq, cand)
    return [float(s) for s in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    print(f'NextItNet BPR-max: device={device}, K={NUM_NEGATIVES}, reg={BPR_MAX_REG}', flush=True)
    scores = [train_one_dataset(ds, device) for ds in DATASETS]
    print(f'Macro-average validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
