"""Merged single-model NextItNet with mixed causal-prefix sampling."""
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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = DEFAULT_DATASETS[:]
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
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
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


seed_everything = seed_all


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
    itemnum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    fields = line.split()
                    if len(fields) == 2:
                        itemnum = max(itemnum, int(fields[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('no users with at least two training interactions')
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        complete_history = user_train[user]
        interacted = set(complete_history)
        if len(interacted) >= itemnum:
            raise RuntimeError('user has no unseen training negatives')
        if random.random() < 0.5:
            history = complete_history
        else:
            history = complete_history[:random.randint(2, len(complete_history))]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            neg[idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


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
        residual = x
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
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
        mask = (seq != 0)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * mask.long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if features.ndim == 2:
            return (features.unsqueeze(1) * embeddings).sum(-1)
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(-1)
        return torch.einsum('blh,bkh->blk', features, embeddings)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView(nn.Module):
    """Diagnostic view sharing the trained network and its scoring head."""
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


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos != 0
    return F.softplus(negative - positive)[mask].mean()


def _history_to_tensor(history, maxlen, device):
    arr = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if len(tail):
        arr[-len(tail):] = tail
    return torch.as_tensor(arr[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # A fresh local RNG makes every checkpoint use identical validation candidates
    # without advancing the training sampler's random stream.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            available = itemnum - len({i for i in excluded if 1 <= i <= itemnum})
            count = min(99, max(0, available))
            candidates = [target]
            while len(candidates) < count + 1:
                negative = rng.randint(1, itemnum)
                if negative not in excluded:
                    excluded.add(negative)
                    candidates.append(negative)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(users)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'empty dataset or vocabulary: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr10, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(x, dtype=torch.long, device=device) for x in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} loss={total_loss / NUM_BATCHES:.4f} '
                  f'val_score: {hr10:.4f} HR@20={metrics["HR@20"]:.4f} '
                  f'NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    meta['sampling'] = 'half_full_half_uniform_causal_prefix'
    meta['seed'] = SEED
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr10:.4f} epoch={best_epoch}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""One trained NextItNet checkpoint per Amazon category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_ModelClass = None


def _ensure_loaded(dataset_name):
    global _device, _ModelClass
    if dataset_name not in DATASETS:
        raise ValueError(f'unknown dataset_name: {dataset_name}')
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _ModelClass is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location(f'nextitnet_training_{os.getpid()}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    kwargs = {key: meta[key] for key in ('itemnum', 'hidden_units', 'num_blocks', 'kernel_size', 'dropout_rate', 'maxlen')}
    model = _ModelClass(**kwargs).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(dataset)
    model, meta = _models[dataset], _metas[dataset]
    maxlen = int(meta['maxlen'])
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
        f.write(script)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'NextItNet merged causal-prefix training on {device}', flush=True)
    write_predict_script(WORKING_DIR)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f'Average local validation HR@10 = {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
