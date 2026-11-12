"""Single-model, per-category NextItNet with multi-negative BPR-max."""
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

DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = float(os.environ.get('NEXTITNET_DROPOUT', '0.2'))
LR = 0.001
WEIGHT_DECAY = 0.0
BATCH_SIZE = int(os.environ.get('NEXTITNET_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 32
BPR_MAX_REG = 0.01
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
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class TrainingData(dict):
    """Cache training-only sampling metadata without modifying histories."""
    def __init__(self, histories, itemnum):
        super().__init__(histories)
        self.users = [u for u, h in histories.items() if len(h) >= 2 and len(set(h)) < itemnum]
        self.seen = {u: np.asarray(sorted(set(histories[u])), dtype=np.int64) for u in self.users}


def _uniform_unseen(seen, itemnum, shape):
    # Rejection sampling is uniform over the user's unobserved training items.
    # Use an explicit complement only for exceptionally dense histories.
    if len(seen) >= itemnum:
        raise RuntimeError('No unseen training items available for negative sampling')
    if len(seen) > itemnum // 2:
        allowed = np.setdiff1d(np.arange(1, itemnum + 1, dtype=np.int64), seen, assume_unique=True)
        return np.random.choice(allowed, size=shape, replace=True)
    values = np.random.randint(1, itemnum + 1, size=shape, dtype=np.int64)
    invalid = np.isin(values, seen)
    while invalid.any():
        values[invalid] = np.random.randint(1, itemnum + 1, size=int(invalid.sum()), dtype=np.int64)
        invalid = np.isin(values, seen)
    return values


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=NUM_NEGATIVES):
    users = getattr(user_train, 'users', None)
    if users is None:
        users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
    if not users:
        raise RuntimeError('No eligible users with training transitions and unseen items')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
    cached_seen = getattr(user_train, 'seen', None)
    for row in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        seen = cached_seen[u] if cached_seen is not None else np.asarray(sorted(set(history)), dtype=np.int64)
        neg[row, -length:] = _uniform_unseen(seen, itemnum, (length, num_negatives))
    return seq, pos, neg


def _history_to_tensor(history, maxlen, device):
    arr = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        arr[-len(tail):] = tail
    return torch.as_tensor(arr[None, :], dtype=torch.long, device=device)


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
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return residual + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate) for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        mask = (seq != 0).unsqueeze(-1)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x) * mask.to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        emb = self.item_emb(items)
        if emb.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * emb).sum(-1)
        return (features * emb).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        final = features[torch.arange(seq.shape[0], device=seq.device), last]
        if candidates.ndim == 1:
            scores = final @ self.item_emb(candidates).T
        else:
            scores = (final.unsqueeze(1) * self.item_emb(candidates)).sum(-1)
        return scores[0] if seq.shape[0] == 1 else scores


class ModelView(nn.Module):
    """Family-compatible view for encoder and tied-embedding diagnostics."""
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


def bpr_max_loss(p, n, reg=BPR_MAX_REG):
    p, n = p.float(), n.float()
    log_weights = F.log_softmax(n, dim=-1)
    ranking = -torch.logsumexp(log_weights + F.logsigmoid(p[:, None] - n), dim=-1)
    penalty = (log_weights.exp() * n.square()).sum(-1)
    return (ranking + reg * penalty).mean()


def bpr_sequence_loss(model, seq, pos, neg):
    # Encode once, and embed negatives only at supervised positions.
    features = model.log2feats(seq)
    valid = pos != 0
    f = features[valid]
    p = model.predict_score(f, pos[valid])
    n = model.predict_score(f, neg[valid])
    return bpr_max_loss(p, n)


def build_validation_cases(user_train, user_valid, itemnum, num_users=200):
    # Dedicated RNG keeps validation candidates fixed and leaves training RNG intact.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for u in users:
        target = user_valid[u][0]
        excluded = set(user_train[u]) | {target, 0}
        count = min(99, itemnum - sum(1 for x in excluded if 1 <= x <= itemnum))
        candidates = [target]
        if count > 0:
            if len(excluded) > itemnum // 2:
                pool = [i for i in range(1, itemnum + 1) if i not in excluded]
                candidates.extend(rng.sample(pool, count))
            else:
                while len(candidates) < count + 1:
                    candidate = rng.randint(1, itemnum)
                    if candidate not in excluded:
                        excluded.add(candidate)
                        candidates.append(candidate)
        cases.append((u, candidates))
    return cases


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200, cases=None):
    if cases is None:
        cases = build_validation_cases(user_train, user_valid, itemnum, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for u, candidates in cases:
            seq = _history_to_tensor(user_train[u], maxlen, device)
            cand = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).float().cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    raw_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not raw_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    user_train = TrainingData(raw_train, itemnum)
    cases = build_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation cases: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} negatives={NUM_NEGATIVES}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    os.makedirs(WORKING_DIR, exist_ok=True)
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite BPR-max loss in {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN, cases=cases)
            hr = metrics['HR@10']
            ndcg = metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={loss_sum / NUM_BATCHES:.5f} val={metrics}', flush=True)
            if (hr, ndcg) > (best_hr, best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    meta.update(best_epoch=best_epoch, validation_hr10=best_hr, num_negatives=NUM_NEGATIVES, bpr_max_reg=BPR_MAX_REG)
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    script = r'''"""Route each Amazon category to its single trained NextItNet checkpoint."""
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
        spec = importlib.util.spec_from_file_location('nextitnet_bprmax_model_definition', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    return _ModelClass


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    kwargs = {k: meta[k] for k in ('itemnum', 'hidden_units', 'num_blocks', 'kernel_size', 'dropout_rate', 'maxlen')}
    model = _load_model_class()(**kwargs).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    if not candidates:
        return []
    model = _models[ds]
    meta = _metas[ds]
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).float().cpu().tolist()
    return [float(s) for s in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
