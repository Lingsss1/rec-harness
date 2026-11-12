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
NUM_EPOCHS = max(1, min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 50
NUM_NEGATIVES = 64
CALIBRATION_T = 0.5
_SAMPLING_CACHE = {}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
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
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    largest = max(largest, int(parts[1]))
    return largest


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def _sampling_data(user_train, itemnum):
    key = id(user_train)
    if key not in _SAMPLING_CACHE:
        users = [u for u, h in user_train.items() if len(h) >= 2]
        seen = {u: np.asarray(sorted(set(user_train[u])), dtype=np.int64) for u in users}
        for u in users:
            if np.count_nonzero((seen[u] >= 1) & (seen[u] <= itemnum)) >= itemnum:
                raise RuntimeError(f'User {u} has no available training negatives')
        _SAMPLING_CACHE[key] = (user_train, users, seen)
    _, users, seen = _SAMPLING_CACHE[key]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    return users, seen


def _draw_negatives(itemnum, forbidden, shape):
    values = np.random.randint(1, itemnum + 1, size=shape, dtype=np.int64)
    for _ in range(8):
        invalid = np.isin(values, forbidden)
        if not invalid.any():
            return values
        values[invalid] = np.random.randint(1, itemnum + 1, size=int(invalid.sum()))
    invalid = np.isin(values, forbidden)
    if invalid.any():
        available = np.setdiff1d(np.arange(1, itemnum + 1, dtype=np.int64), forbidden, assume_unique=True)
        if len(available) == 0:
            raise RuntimeError('No available negative items')
        values[invalid] = np.random.choice(available, size=int(invalid.sum()), replace=True)
    return values


def sample_batch(user_train, itemnum, batch_size, maxlen, *, return_histories=False):
    # The default three-array causal contract is retained for diagnosis.
    users, seen = _sampling_data(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    histories = []
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        n = min(maxlen, len(history) - 1)
        seq[row, -n:] = history[-n-1:-1]
        pos[row, -n:] = history[-n:]
        forbidden = seen[user]
        neg[row, -n:] = _draw_negatives(itemnum, forbidden, (n,))
        histories.append(forbidden)
    if return_histories:
        return seq, pos, neg, histories
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
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

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
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if features.ndim == embeddings.ndim - 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        out = self.encode_sequence(seq)
        return (out * self.item_emb(pos)).sum(-1), (out * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        return self.predict_score(features, candidates)[0]


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


def gbce_sequence_loss(model, seq, pos, neg, histories):
    # All forbidden sets come exclusively from this batch's training users.
    mask = pos.ne(0)
    counts = mask.sum(1).detach().cpu().tolist()
    negative_parts = [
        _draw_negatives(model.itemnum, forbidden, (int(count), NUM_NEGATIVES))
        for forbidden, count in zip(histories, counts) if count
    ]
    if not negative_parts:
        return model.item_emb.weight.sum() * 0.0
    sampled = torch.as_tensor(np.concatenate(negative_parts, axis=0), dtype=torch.long, device=seq.device)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    alpha = NUM_NEGATIVES / max(model.itemnum - 1, 1)
    beta = 1.0 - CALIBRATION_T + alpha * CALIBRATION_T
    total = features.new_zeros((), dtype=torch.float64)
    # Chunk the scoring operation to limit temporary allocations.
    for start in range(0, features.shape[0], 1024):
        end = min(start + 1024, features.shape[0])
        h = features[start:end]
        positive_scores = (h * model.item_emb(targets[start:end])).sum(-1)
        negative_scores = torch.bmm(model.item_emb(sampled[start:end]), h.unsqueeze(-1)).squeeze(-1)
        a = beta * F.logsigmoid(positive_scores.double())
        # Stable requested calibrated positive log-odds expression.
        # Clamping only protects expm1 against exact underflow to signed zero.
        a = a.clamp(max=-torch.finfo(torch.float64).tiny)
        positive_loss = -a - torch.log(-torch.expm1(a))
        negative_loss = F.softplus(negative_scores.double()).sum(-1)
        total = total + ((positive_loss + negative_loss) / (NUM_NEGATIVES + 1)).sum()
    return total / features.shape[0]


def bpr_sequence_loss(model, seq, pos, neg):
    # Compatibility helper for callers diagnosing the original pairwise interface.
    positive, negative = model(seq, pos, neg)
    mask = pos.ne(0)
    return F.softplus(negative - positive)[mask].mean()


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    # Fixed validation users and unique negatives; never consume training RNG.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        forbidden = set(history) | {target, 0}
        if itemnum - sum(1 <= x <= itemnum for x in forbidden) < 99:
            raise RuntimeError('Validation requires 99 unique unseen negative candidates')
        candidates = [target]
        while len(candidates) < 100:
            candidate = rng.randint(1, itemnum)
            if candidate not in forbidden:
                candidates.append(candidate)
                forbidden.add(candidate)
        seq = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        cases.append((seq, np.asarray(candidates, dtype=np.int64)))
    return cases


@torch.no_grad()
def evaluate_cases(model, cases, device):
    if not cases:
        raise RuntimeError('No validation users')
    was_training = model.training
    model.eval()
    hits = ndcg = 0.0
    for seq, candidates in cases:
        seq_t = torch.as_tensor(seq[None, :], dtype=torch.long, device=device)
        cand_t = torch.as_tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq_t, cand_t).cpu().numpy()
        rank = int(np.flatnonzero(np.argsort(-scores, kind='stable') == 0)[0])
        if rank < 10:
            hits += 1
            ndcg += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {'HR@10': hits / len(cases), 'NDCG@10': ndcg / len(cases)}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, maxlen), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    _SAMPLING_CACHE.clear()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 1 or not user_train:
        raise RuntimeError(f'Empty dataset or invalid vocabulary: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg, histories = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN, return_histories=True)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = gbce_sequence_loss(model, seq_t, pos_t, neg_t, histories)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite gBCE loss in {dataset_name} at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} loss={total_loss / NUM_BATCHES:.6f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del optimizer, model
    _SAMPLING_CACHE.clear()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    script = r'''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(_DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('nextitnet_gbce_trained_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset in _DATASETS:
        if dataset in _models:
            continue
        with open(os.path.join(_HERE, f'{dataset}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = module.NextItNetModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, f'{dataset}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _models:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not candidates:
        return []
    model = _models[dataset]
    maxlen = _metas[dataset]['maxlen']
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq_t = torch.tensor([sequence], dtype=torch.long, device=_device)
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
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'NextItNet gBCE K={NUM_NEGATIVES} t={CALIBRATION_T} device={device}', flush=True)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    write_predict_script(working_dir)
    print(f'Macro-average best validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
