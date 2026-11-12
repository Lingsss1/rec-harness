"""Independent per-category GRU4Rec with multi-negative BPR-max."""
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
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
LR = 1e-3
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('SASREC_EPOCHS', '200')), int(os.environ.get('GRU4REC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
LOSS_CHUNK = 1024
SCORE_REG = 0.1
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_SAMPLE_CACHE = None


def seed_training():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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
    largest = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        largest = max(largest, int(parts[1]))
    return largest


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is None or _SAMPLE_CACHE[0] is not user_train or _SAMPLE_CACHE[1] != itemnum:
        users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        seen = {u: np.asarray(list(set(user_train[u])), dtype=np.int64) for u in users}
        _SAMPLE_CACHE = (user_train, itemnum, users, seen)
    users, seen = _SAMPLE_CACHE[2:]
    if not users:
        raise RuntimeError('No eligible users with available training negatives')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length-1:-1]
        pos[row, -length:] = history[-length:]
        draws = np.random.randint(1, itemnum + 1, size=(length, NUM_NEGATIVES))
        rejected = np.isin(draws, seen[user])
        while rejected.any():
            draws[rejected] = np.random.randint(1, itemnum + 1, size=int(rejected.sum()))
            rejected = np.isin(draws, seen[user])
        neg[row, -length:] = draws
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        embedded = self.dropout(self.item_emb(sequence))
        output, _ = self.gru(embedded)
        return output

    def predict_score(self, features, items):
        embedded = self.item_emb(items)
        while features.ndim < embedded.ndim:
            features = features.unsqueeze(-2)
        return (features * embedded).sum(dim=-1)

    def encode(self, seq):
        output = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        features = output[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


def bpr_sequence_loss(model, seq, pos, neg, backward=False):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    negatives = neg[mask]
    count = targets.numel()
    if count == 0:
        raise RuntimeError('Batch contains no valid targets')
    # In training, accumulate output-embedding and feature gradients one chunk
    # at a time, then propagate the combined feature gradient through the GRU.
    # This avoids retaining all K-negative embedding graphs simultaneously.
    chunk_features = features.detach().requires_grad_(True) if backward else features
    total = 0.0 if backward else features.sum() * 0.0
    for start in range(0, count, LOSS_CHUNK):
        stop = min(start + LOSS_CHUNK, count)
        h = chunk_features[start:stop]
        s_pos = model.predict_score(h, targets[start:stop])
        s_neg = model.predict_score(h, negatives[start:stop])
        log_w = F.log_softmax(s_neg, dim=-1)
        ranking = -torch.logsumexp(log_w + F.logsigmoid(s_pos[:, None] - s_neg), dim=-1)
        regularizer = SCORE_REG * (log_w.exp() * s_neg.square()).sum(dim=-1)
        weighted_loss = (ranking + regularizer).sum() / count
        if backward:
            weighted_loss.backward()
            total += float(weighted_loss.detach())
        else:
            total = total + weighted_loss
    if backward:
        features.backward(chunk_features.grad)
    return total


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    # This local RNG never advances training's Python, NumPy, or Torch RNG.
    rng = random.Random(42)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        candidates = [target]
        wanted = min(99, itemnum - len({i for i in excluded if 1 <= i <= itemnum}))
        while len(candidates) < wanted + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        cases.append((user, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200, cases=None):
    if cases is None:
        cases = make_validation_cases(user_train, user_valid, itemnum, num_users)
    was_training = model.training
    model.eval()
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    for user, candidates in cases:
        seq = _history_to_tensor(user_train[user], maxlen, device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        for k in (10, 20):
            if rank < k:
                totals[f'HR@{k}'] += 1.0
                totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(len(cases), 1) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_training()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model_config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                        num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    model = GRU4RecModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation users for {dataset_name}')
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            total_loss += bpr_sequence_loss(model, seq, pos, neg, backward=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN, cases=cases)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.4f} val_NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    metadata = dict(model_config=model_config, maxlen=MAXLEN,
                    best_epoch=best_epoch, validation_hr10=best_hr,
                    num_negatives=NUM_NEGATIVES, score_regularization=SCORE_REG)
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print(f'[{dataset_name}] selected_epoch={best_epoch} val_HR@10={best_hr:.4f}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    script = r'''"""Route each category to its single selected GRU4Rec checkpoint."""
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
_model_class = None


def _load_model_class():
    global _model_class
    if _model_class is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('gru4rec_bprmax_train', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.GRU4RecModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _load_model_class()(**meta['model_config']).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in DATASETS:
        raise ValueError(f'Unknown dataset_name: {ds}')
    _ensure_loaded(ds)
    if not candidates:
        return []
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(value) for value in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_training()
    print(f'GRU4Rec BPR-max K={NUM_NEGATIVES} device={device}', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    write_predict_script(WORKING_DIR)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
