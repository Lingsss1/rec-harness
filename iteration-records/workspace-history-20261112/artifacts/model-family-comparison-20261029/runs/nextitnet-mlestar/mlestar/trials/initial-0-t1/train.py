from __future__ import annotations

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

DEFAULT_DATASETS = [
    'Movies_and_TV', 'Industrial_and_Scientific',
    'Electronics', 'CDs_and_Vinyl',
]
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
NUM_EPOCHS = min(200, max(1, int(os.environ.get(
    'NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 40
NUM_NEGATIVES = 64
CALIBRATION_T = 0.75
NEGATIVE_CHUNK = 1024


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


seed_everything()


def _read_split(path):
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir, dataset_name):
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


_sampler_cache = None


def sample_batch(user_train, itemnum, batch_size, maxlen):
    global _sampler_cache
    if _sampler_cache is None or _sampler_cache[0] is not user_train:
        users = [u for u, h in user_train.items() if len(h) >= 2]
        seen = {u: np.asarray(sorted(set(user_train[u])), dtype=np.int64)
                for u in users}
        if not users:
            raise RuntimeError('No users with at least two training interactions')
        if any(len(seen[u]) >= itemnum for u in users):
            raise RuntimeError('A training user has no unseen negative items')
        _sampler_cache = (user_train, users, seen)
    _, users, seen = _sampler_cache
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
    for row in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length-1:-1]
        pos[row, -length:] = history[-length:]
        # IID uniform draws over real item IDs, conditional only on train history.
        draws = np.random.randint(1, itemnum + 1,
                                  size=(length, NUM_NEGATIVES), dtype=np.int64)
        bad = np.isin(draws, seen[u])
        while bad.any():
            draws[bad] = np.random.randint(1, itemnum + 1, size=int(bad.sum()))
            bad = np.isin(draws, seen[u])
        neg[row, -length:] = draws
    return seq, pos, neg


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size,
                               dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size,
                               dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        residual = x
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        y = self.dropout(y)
        return residual + y


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8,
                 kernel_size=3, dropout_rate=0.2, maxlen=128):
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
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x)
            x = x * mask
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long,
                                   device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long,
                                device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.ndim == seq.ndim:
            negative = (features * self.item_emb(neg)).sum(-1)
        else:
            negative = (features.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        return positive, negative

    def score_candidates(self, seq, candidates):
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(features, candidates)
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
    mask = pos != 0
    # Encode once, then score only supervised positions.
    features = model.log2feats(seq)[mask]
    positive_ids = pos[mask]
    negative_ids = neg[mask]
    count = features.size(0)
    if count == 0:
        raise RuntimeError('Batch contains no supervised positions')
    k = negative_ids.size(-1)
    alpha = k / float(max(model.itemnum - 1, 1))
    beta = 1.0 - CALIBRATION_T + alpha * CALIBRATION_T
    positive_logits = (features * model.item_emb(positive_ids)).sum(-1)
    loss_sum = -beta * F.logsigmoid(positive_logits).sum()

    def negative_term(chunk_features, chunk_ids):
        embeddings = model.item_emb(chunk_ids)
        logits = (chunk_features.unsqueeze(1) * embeddings).sum(-1)
        return F.softplus(logits).sum()

    # Recompute these embedding lookups during backward rather than retaining
    # an [all valid positions, K, hidden] activation tensor.
    for start in range(0, count, NEGATIVE_CHUNK):
        stop = min(start + NEGATIVE_CHUNK, count)
        loss_sum = loss_sum + checkpoint(
            negative_term, features[start:stop], negative_ids[start:stop],
            use_reentrant=False, preserve_rng_state=False)
    return loss_sum / (count * (k + 1))


def _history_to_tensor(history, maxlen, device):
    arr = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        arr[-len(tail):] = tail
    return torch.as_tensor(arr, dtype=torch.long, device=device).unsqueeze(0)


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    # Private validation RNG: training and evaluation never perturb one another.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        nneg = min(99, itemnum - len({i for i in excluded if 1 <= i <= itemnum}))
        candidates = [target]
        while len(candidates) < nneg + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        cases.append((user_train[user], candidates))
    return cases


def evaluate_cases(model, cases, device, maxlen):
    was_training = model.training
    model.eval()
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    with torch.no_grad():
        for history, candidates in cases:
            seq = _history_to_tensor(history, maxlen, device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            if rank < 10:
                hr10 += 1
                ndcg10 += 1.0 / math.log2(rank + 2)
            if rank < 20:
                hr20 += 1
                ndcg20 += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    n = max(len(cases), 1)
    return {'HR@10': hr10 / n, 'HR@20': hr20 / n,
            'NDCG@10': ndcg10 / n, 'NDCG@20': ndcg20 / n}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(
        user_train, user_valid, itemnum, num_users), device, maxlen)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 1 or not user_train:
        raise RuntimeError(f'Empty dataset or invalid vocabulary: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                  num_blocks=NUM_BLOCKS, kernel_size=KERNEL_SIZE,
                  dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation users for {dataset_name}')
    best_hr10, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = generalized_bce_loss(model, seq_t, pos_t, neg_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} '
                  f'loss={total_loss / NUM_BATCHES:.6f} '
                  f'val_score={hr10:.4f} HR@20={metrics["HR@20"]:.4f} '
                  f'NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at {epoch}', flush=True)
                break
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr10:.4f} '
          f'epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr10


def write_predict_script(working_dir):
    script = r'''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _load_model_class():
    global _model_class
    if _model_class is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('nextitnet_gbce_train', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.NextItNetModel
    return _model_class


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
    model = _load_model_class()(**meta).to(_device)
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
    if not candidates:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta['maxlen'])
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'NextItNet direct gBCE K={NUM_NEGATIVES} t={CALIBRATION_T} device={device}', flush=True)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'All datasets done. Average val HR@10 = {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
