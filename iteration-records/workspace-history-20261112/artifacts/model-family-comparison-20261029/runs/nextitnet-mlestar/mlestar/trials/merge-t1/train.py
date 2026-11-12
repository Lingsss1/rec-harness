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
from torch.utils.checkpoint import checkpoint

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
EVAL_EVERY = 5
PATIENCE = 40
NUM_NEGATIVES = 64
CALIBRATION = 0.75
LOSS_CHUNK = 512
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
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
    largest = 0
    for split in ('train', 'valid'):
        for items in _read_split(os.path.join(data_dir, f'{dataset_name}_{split}.txt')).values():
            if items:
                largest = max(largest, max(items))
    return largest


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, h in user_train.items() if len(h) >= 2]
        self.excluded = {u: np.unique(np.asarray(user_train[u], dtype=np.int64)) for u in self.users}
        if not self.users:
            raise RuntimeError('No eligible training users')
        if any(len(self.excluded[u]) >= itemnum for u in self.users):
            raise RuntimeError('A training user has no unseen negative items')

    def sample(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
        for b in range(batch_size):
            u = random.choice(self.users)
            history = self.user_train[u]
            length = min(maxlen, len(history) - 1)
            seq[b, -length:] = history[-length-1:-1]
            pos[b, -length:] = history[-length:]
            draws = np.random.randint(1, self.itemnum + 1, size=(length, NUM_NEGATIVES))
            invalid = np.isin(draws, self.excluded[u])
            while invalid.any():
                draws[invalid] = np.random.randint(1, self.itemnum + 1, size=int(invalid.sum()))
                invalid = np.isin(draws, self.excluded[u])
            neg[b, -length:] = draws
        return seq, pos, neg


_ACTIVE_SAMPLER = None


def sample_batch(user_train, itemnum, batch_size, maxlen):
    global _ACTIVE_SAMPLER
    if (_ACTIVE_SAMPLER is None or _ACTIVE_SAMPLER.user_train is not user_train
            or _ACTIVE_SAMPLER.itemnum != itemnum):
        _ACTIVE_SAMPLER = TrainingSampler(user_train, itemnum)
    return _ACTIVE_SAMPLER.sample(batch_size, maxlen)


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        residual = x
        y = self._causal_conv(self.conv1, self.norm1(x).transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self._causal_conv(self.conv2, self.norm2(y).transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(y)


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
        mask = seq.ne(0)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * mask.long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x) * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(-1)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        raise ValueError('Incompatible feature and candidate shapes')

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.ndim == seq.ndim:
            negative = (features * self.item_emb(neg)).sum(-1)
        else:
            negative = (features.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        return positive, negative

    def score_candidates(self, seq, candidates):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        users = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(users, candidates)
        return scores[0] if seq.size(0) == 1 else scores


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


def generalized_bce_loss(model, seq, pos, neg):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    negatives = neg[mask]
    count = targets.numel()
    if count == 0:
        raise RuntimeError('Training batch has no targets')
    k = negatives.size(-1)
    alpha = k / float(max(model.itemnum - 1, 1))
    beta = 1.0 - CALIBRATION + alpha * CALIBRATION
    positive_logits = (features * model.item_emb(targets)).sum(-1)
    total = -beta * F.logsigmoid(positive_logits).sum()

    def negative_sum(chunk_features, chunk_ids):
        embeddings = model.item_emb(chunk_ids)
        logits = (chunk_features.unsqueeze(1) * embeddings).sum(-1)
        return F.softplus(logits).sum()

    for start in range(0, count, LOSS_CHUNK):
        end = min(start + LOSS_CHUNK, count)
        total = total + checkpoint(negative_sum, features[start:end], negatives[start:end],
                                   use_reentrant=False, preserve_rng_state=False)
    return total / (count * (k + 1))


def _history_to_tensor(history, maxlen, device):
    clipped = list(history[-maxlen:])
    arr = [0] * (maxlen - len(clipped)) + clipped
    return torch.tensor([arr], dtype=torch.long, device=device)


def build_validation(user_train, user_valid, itemnum, maxlen, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    rows = []
    for u in users:
        target = user_valid[u][0]
        history = user_train[u]
        excluded = set(history) | {target, 0}
        candidates = [target]
        required = min(99, itemnum - len({i for i in excluded if 1 <= i <= itemnum}))
        while len(candidates) < required + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                candidates.append(item)
                excluded.add(item)
        clipped = history[-maxlen:]
        sequence = [0] * (maxlen - len(clipped)) + clipped
        rows.append((sequence, candidates))
    return rows


def evaluate_rows(model, rows, device):
    was_training = model.training
    model.eval()
    hits = ndcg = 0.0
    with torch.no_grad():
        for sequence, candidates in rows:
            seq = torch.tensor([sequence], dtype=torch.long, device=device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    n = max(len(rows), 1)
    return {'HR@10': hits / n, 'NDCG@10': ndcg / n}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    rows = build_validation(user_train, user_valid, itemnum, maxlen, num_users)
    return evaluate_rows(model, rows, device)


def train_one_dataset(dataset_name, device, data_dir):
    global _ACTIVE_SAMPLER
    seed_all()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 1 or not user_train:
        raise RuntimeError(f'Empty dataset or invalid vocabulary: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} K={NUM_NEGATIVES} t={CALIBRATION}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    _ACTIVE_SAMPLER = TrainingSampler(user_train, itemnum)
    rows = build_validation(user_train, user_valid, itemnum, MAXLEN)
    if not rows:
        raise RuntimeError(f'No validation users for {dataset_name}')
    os.makedirs(WORKING_DIR, exist_ok=True)
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = generalized_bce_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_rows(model, rows, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={loss_sum / NUM_BATCHES:.6f} '
                  f'val_HR@10={hr:.4f} val_NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stop at {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint selected')
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] selected epoch={best_epoch} validation HR@10={best_hr:.4f}', flush=True)
    _ACTIVE_SAMPLER = None
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
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
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = NextItNetModel(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    if len(candidates) == 0:
        return []
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    clipped = list(history[-maxlen:])
    seq = [0] * (maxlen - len(clipped)) + clipped
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(value) for value in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    path = os.path.join(working_dir, 'predict.py')
    with open(path, 'w') as f:
        f.write(header + architecture + '\n\n' + runtime)
    print(f'Wrote {path}', flush=True)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
