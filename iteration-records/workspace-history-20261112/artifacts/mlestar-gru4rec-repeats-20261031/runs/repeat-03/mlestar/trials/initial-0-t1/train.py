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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = 200
EVAL_EVERY = 5
PATIENCE = 25
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
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
    highest = 0
    for split in ('train', 'valid'):
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    highest = max(highest, int(parts[1]))
    return highest


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


_SAMPLE_CACHE = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    cached = _SAMPLE_CACHE.get(key)
    if cached is None or cached[0] is not user_train:
        users = [u for u, history in user_train.items() if len(history) >= 2]
        interacted = {u: set(user_train[u]) for u in users}
        cached = (user_train, users, interacted)
        _SAMPLE_CACHE[key] = cached
    _, users, interacted = cached
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length-1:-1]
        pos[row, -length:] = history[-length:]
        seen = interacted[user]
        if len(seen) < itemnum:
            for col in range(maxlen - length, maxlen):
                candidate = random.randint(1, itemnum)
                while candidate in seen:
                    candidate = random.randint(1, itemnum)
                neg[row, col] = candidate
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        embedded = self.dropout(self.item_emb(sequence))
        output, _ = self.gru(embedded)
        return output

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).max(dim=1).values
        result = features[torch.arange(sequence.size(0), device=sequence.device), last.clamp(min=0)]
        return result * last.ge(0).unsqueeze(-1).to(result.dtype)

    def predict_score(self, features, items):
        embedded = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embedded.transpose(0, 1))
        return (features.unsqueeze(-2) * embedded).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        scores = self.predict_score(self.encode(seq), candidates)
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

    def forward(self, seq, pos, neg):
        return self.model(seq, pos, neg)


def cross_entropy_sequence_loss(model, seq, pos, neg=None):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask] - 1
    if targets.numel() == 0:
        return features.sum() * 0.0
    chunk_size = max(1, min(256, 4000000 // model.itemnum))
    loss = features.sum() * 0.0
    for start in range(0, targets.numel(), chunk_size):
        stop = min(start + chunk_size, targets.numel())
        logits = features[start:stop] @ model.item_emb.weight[1:].T
        loss = loss + F.cross_entropy(logits, targets[start:stop], reduction='sum') / targets.numel()
    return loss


def bpr_sequence_loss(model, seq, pos, neg):
    # Compatibility entry point for diagnostics; the trial objective is categorical CE.
    return cross_entropy_sequence_loss(model, seq, pos, neg)


def backward_full_catalogue_ce(model, seq, pos):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask] - 1
    count = targets.numel()
    if not count:
        raise RuntimeError('Batch contains no shifted targets')
    # Detach only at the feature boundary. Accumulate the exact CE gradients
    # for both the output embedding and feature leaf, then traverse the GRU
    # graph once. Tied input/output embeddings receive both contributions.
    feature_leaf = features.detach().requires_grad_(True)
    chunk_size = max(1, min(256, 4000000 // model.itemnum))
    total = 0.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        logits = feature_leaf[start:stop] @ model.item_emb.weight[1:].T
        chunk_loss = F.cross_entropy(logits, targets[start:stop], reduction='sum') / count
        chunk_loss.backward()
        total += chunk_loss.detach().item()
        del logits, chunk_loss
    features.backward(feature_leaf.grad)
    return total


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {0, target}
        if itemnum - len(excluded - {0}) < 99:
            raise RuntimeError('Fewer than 99 distinct eligible validation negatives')
        items = [target]
        while len(items) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                items.append(item)
        seq = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        sequences.append(seq)
        candidates.append(items)
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_cases(model, cases, device):
    sequences, candidates = cases
    was_training = model.training
    model.eval()
    ranks = []
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start+64], device=device)
        items = torch.as_tensor(candidates[start:start+64], device=device)
        scores = model.predict_score(model.encode(seq), items)
        # Stable candidate ordering puts the target first for equal scores.
        rank = scores[:, 1:].gt(scores[:, :1]).sum(dim=1)
        ranks.extend(rank.cpu().tolist())
    model.train(was_training)
    if not ranks:
        return {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0}
    n = len(ranks)
    return {
        'HR@10': sum(r < 10 for r in ranks) / n,
        'HR@20': sum(r < 20 for r in ranks) / n,
        'NDCG@10': sum(1.0 / math.log2(r + 2) for r in ranks if r < 10) / n,
    }


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, maxlen), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_all()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = make_validation_cases(train, valid, itemnum, MAXLEN)
    if len(cases[0]) == 0:
        raise RuntimeError(f'No validation users for {dataset_name}')
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_value = backward_full_catalogue_ce(model, seq_t, pos_t)
            if not math.isfinite(loss_value):
                raise RuntimeError(f'Nonfinite CE loss for {dataset_name}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
            total_loss += loss_value
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} ce={total_loss / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg = hr, ndcg
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stop at epoch {epoch}', flush=True)
                break
        elif epoch == 1:
            print(f'[{dataset_name}] epoch=1 ce={total_loss / NUM_BATCHES:.5f}', flush=True)
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                objective='full_catalogue_cross_entropy', best_epoch=best_epoch,
                seed=SEED)
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    _SAMPLE_CACHE.clear()
    del optimizer, model, train, valid
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact training architecture so inference is self-contained.
    header = '''from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

'''
    body = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(itemnum=meta['itemnum'],
                             hidden_units=meta['hidden_units'],
                             num_layers=meta['num_layers'],
                             dropout_rate=meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f'Unknown dataset_name: {ds}')
    if not candidates:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq_t = torch.tensor([sequence], dtype=torch.long, device=_device)
    items_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, items_t)
    return [float(s) for s in scores.cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(GRU4RecModel) + '\n' + body)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    print(f'GRU4Rec full-catalogue CE on {device}; seed=42; epochs<=200; batches=200', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
