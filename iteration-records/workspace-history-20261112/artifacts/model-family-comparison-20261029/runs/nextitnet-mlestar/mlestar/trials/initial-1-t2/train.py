"""Single-model per-category NextItNet with BPR-max training."""
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
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = int(os.environ.get('NEXTITNET_EVAL_EVERY', '5'))
PATIENCE = int(os.environ.get('NEXTITNET_PATIENCE', '20'))
NUM_NEGATIVES = 64
BPR_MAX_REG = 0.1
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_split(path):
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
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
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class BatchSampler:
    def __init__(self, user_train, itemnum):
        self.itemnum = itemnum
        self.users = []
        self.histories = {}
        self.excluded = {}
        for user, history in user_train.items():
            seen = np.unique(np.asarray(history, dtype=np.int64))
            if len(history) >= 2 and len(seen) < itemnum:
                self.users.append(user)
                self.histories[user] = history
                self.excluded[user] = seen
        if not self.users:
            raise RuntimeError('No eligible training users')

    def sample(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
        for row in range(batch_size):
            user = random.choice(self.users)
            history = self.histories[user]
            length = min(maxlen, len(history) - 1)
            seq[row, -length:] = history[-length-1:-1]
            pos[row, -length:] = history[-length:]
            seen = self.excluded[user]
            shape = (length, NUM_NEGATIVES)
            if len(seen) > self.itemnum // 2:
                eligible = np.setdiff1d(np.arange(1, self.itemnum + 1), seen, assume_unique=True)
                values = np.random.choice(eligible, size=shape, replace=True)
            else:
                values = np.random.randint(1, self.itemnum + 1, size=shape)
                rejected = np.isin(values, seen)
                while rejected.any():
                    values[rejected] = np.random.randint(1, self.itemnum + 1, size=int(rejected.sum()))
                    rejected = np.isin(values, seen)
            neg[row, -length:] = values
        return seq, pos, neg


def sample_batch(user_train, itemnum, batch_size, maxlen):
    return BatchSampler(user_train, itemnum).sample(batch_size, maxlen)


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
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = seq.ne(0).unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        if items.ndim == 1 and features.ndim == 2:
            return features @ embeddings.transpose(0, 1)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(-1)
        if neg.ndim == pos.ndim:
            neg_scores = (out * self.item_emb(neg)).sum(-1)
        else:
            neg_scores = (out.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        return pos_scores, neg_scores

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(1).values.clamp(min=0)
        selected = features[torch.arange(seq.shape[0], device=seq.device), last]
        scores = self.predict_score(selected, candidates)
        return scores[0] if seq.shape[0] == 1 else scores


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

    def forward(self, sequence, pos, neg):
        return self.model(sequence, pos, neg)


def bpr_max_sequence_loss(model, seq, pos, neg):
    features = model.log2feats(seq)
    mask = pos.ne(0)
    features = features[mask]
    positives = pos[mask]
    negatives = neg[mask]
    if features.shape[0] == 0:
        return features.sum() * 0.0
    # Chunking bounds temporary memory; every valid position has equal weight.
    total = features.new_zeros(())
    for start in range(0, features.shape[0], 1024):
        end = start + 1024
        f = features[start:end]
        p = (f * model.item_emb(positives[start:end])).sum(-1)
        n = (f[:, None, :] * model.item_emb(negatives[start:end])).sum(-1)
        logw = torch.log_softmax(n, dim=-1)
        ranking = -torch.logsumexp(logw + F.logsigmoid(p[:, None] - n), dim=-1)
        penalty = (logw.exp() * n.square()).sum(-1)
        total = total + (ranking + BPR_MAX_REG * penalty).sum()
    return total / features.shape[0]


def bpr_sequence_loss(model, seq, pos, neg):
    return bpr_max_sequence_loss(model, seq, pos, neg)


def _history_to_tensor(history, maxlen, device):
    array = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        array[-len(tail):] = tail
    return torch.as_tensor(array[None, :], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    user_rng = random.Random(SEED)
    if len(users) > 200:
        users = user_rng.sample(users, 200)
    negative_rng = random.Random(SEED)
    cases = []
    for user in users:
        target = user_valid[user][0]
        history = user_train[user]
        excluded = set(history) | {0, target}
        count = min(99, itemnum - sum(1 for i in excluded if 1 <= i <= itemnum))
        candidates = [target]
        used = set(excluded)
        while len(candidates) < count + 1:
            item = negative_rng.randint(1, itemnum)
            if item not in used:
                candidates.append(item)
                used.add(item)
        seq = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        cases.append((seq, np.asarray(candidates, dtype=np.int64)))
    return cases


def evaluate_cases(model, cases, device):
    was_training = model.training
    model.eval()
    metrics = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    with torch.no_grad():
        for array, candidates in cases:
            seq = torch.as_tensor(array[None, :], dtype=torch.long, device=device)
            cand = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for cutoff in (10, 20):
                if rank < cutoff:
                    metrics[f'HR@{cutoff}'] += 1.0
                    metrics[f'NDCG@{cutoff}'] += 1.0 / math.log2(rank + 2)
    if cases:
        metrics = {key: value / len(cases) for key, value in metrics.items()}
    model.train(was_training)
    return metrics


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    cases = make_validation_cases(user_train, user_valid, itemnum, maxlen)
    return evaluate_cases(model, cases[:num_users], device)


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    sampler = BatchSampler(user_train, itemnum)
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    if not cases:
        raise RuntimeError(f'No validation users: {dataset_name}')
    model = NextItNetModel(itemnum, HIDDEN_UNITS, NUM_BLOCKS, KERNEL_SIZE, DROPOUT_RATE, MAXLEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
                loss='bpr-max', negatives=NUM_NEGATIVES, bpr_max_reg=BPR_MAX_REG)
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sampler.sample(BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_max_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss for {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val={json.dumps(metrics)}', flush=True)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('Training completed without validation checkpoint')
    meta['best_epoch'] = best_epoch
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr10:.4f} epoch={best_epoch}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""Route each category to its single trained NextItNet checkpoint."""
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


def _load_model_class():
    global _ModelClass
    if _ModelClass is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('nextitnet_bprmax_architecture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    return _ModelClass


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _load_model_class()(
        itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
        num_blocks=meta['num_blocks'], kernel_size=meta['kernel_size'],
        dropout_rate=meta['dropout_rate'], maxlen=meta['maxlen']
    ).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    seq_arr = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f'NextItNet BPR-max on {device}; K=64; penalty=0.1; seed=42', flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_predict_script(WORKING_DIR)
    print(f'Macro best local validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
