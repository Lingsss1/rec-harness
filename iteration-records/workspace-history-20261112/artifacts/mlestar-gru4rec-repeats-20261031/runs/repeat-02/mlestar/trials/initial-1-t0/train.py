"""Per-category GRU4Rec with multi-negative BPR-max."""
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

DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
BETA = 0.1
LOSS_CHUNK = 512
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
                parts = line.split()
                if len(parts) == 2:
                    highest = max(highest, int(parts[1]))
    return highest


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def _sampler_data(user_train, itemnum):
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        users = [u for u, items in user_train.items() if len(items) >= 2]
        seen = {u: set(user_train[u]) for u in users}
        users = [u for u in users if len(seen[u]) < itemnum]
        if not users:
            raise RuntimeError('No users with a training pair and eligible negative')
        # Holding the dictionary prevents accidental identity reuse in diagnostics.
        _SAMPLER_CACHE[key] = (user_train, users, seen, {})
    return _SAMPLER_CACHE[key]


def _sample_eligible(excluded, itemnum, count, pools, user):
    if len(excluded) > itemnum // 2:
        if user not in pools:
            pools[user] = np.asarray([i for i in range(1, itemnum + 1) if i not in excluded], dtype=np.int64)
        return np.random.choice(pools[user], size=count, replace=True)
    result = np.random.randint(1, itemnum + 1, size=count, dtype=np.int64)
    bad = np.asarray([int(i) in excluded for i in result], dtype=bool)
    while bad.any():
        result[bad] = np.random.randint(1, itemnum + 1, size=int(bad.sum()))
        bad = np.asarray([int(i) in excluded for i in result], dtype=bool)
    return result


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    _, users, seen, pools = _sampler_data(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    # Default output remains the baseline's aligned [batch, length] negatives.
    neg = np.zeros_like(seq) if num_negatives == 1 else np.empty((batch_size, num_negatives), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        window = history[-(maxlen + 1):]
        length = len(window) - 1
        seq[row, -length:] = window[:-1]
        pos[row, -length:] = window[1:]
        if num_negatives == 1:
            neg[row, -length:] = _sample_eligible(seen[user], itemnum, length, pools, user)
        else:
            # One independently sampled bank per user, reused across its positions.
            # Every aligned target belongs to the excluded full training history.
            neg[row] = _sample_eligible(seen[user], itemnum, num_negatives, pools, user)
    if num_negatives != 1:
        neg = np.broadcast_to(neg[:, None, :], (batch_size, maxlen, num_negatives))
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
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        out, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return out

    def encode(self, seq):
        if not torch.is_tensor(seq):
            seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            seq = seq.to(device=self.item_emb.weight.device, dtype=torch.long)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        emb = self.item_emb(items)
        if emb.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * emb).sum(dim=-1)
        return (features * emb).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        candidates = torch.as_tensor(candidates, dtype=torch.long, device=features.device)
        if candidates.ndim == 1:
            scores = features @ self.item_emb(candidates).T
        else:
            scores = (features[:, None, :] * self.item_emb(candidates)).sum(-1)
        return scores[0] if features.size(0) == 1 else scores


def bpr_sequence_loss(model, seq, pos, neg):
    features = model.log2feats(seq)
    valid = pos.ne(0)
    features = features[valid]
    positives = pos[valid]
    negatives = neg[valid]
    if negatives.ndim == 1:
        negatives = negatives[:, None]
    if not positives.numel():
        return features.sum() * 0.0

    def chunk_sum(feat, positive_ids, negative_ids):
        p = (feat * model.item_emb(positive_ids)).sum(-1).float()
        n = (feat[:, None, :] * model.item_emb(negative_ids)).sum(-1).float()
        log_w = F.log_softmax(n, dim=-1)
        ranking = -torch.logsumexp(log_w + F.logsigmoid(p[:, None] - n), dim=-1)
        regularizer = (log_w.exp() * n.square()).sum(-1)
        return (ranking + BETA * regularizer).sum()

    total = features.new_zeros((), dtype=torch.float32)
    for start in range(0, positives.numel(), LOSS_CHUNK):
        end = start + LOSS_CHUNK
        # Recompute item-level logits during backward to bound embedding memory.
        total = total + checkpoint(chunk_sum, features[start:end], positives[start:end],
                                   negatives[start:end], use_reentrant=False)
    return total / positives.numel()


def _history_to_tensor(history, maxlen, device):
    sequence = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    return torch.as_tensor(sequence[None], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    # Fixed validation draws do not consume or reset the training RNG.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        count = min(99, max(0, itemnum - len(excluded - {0})))
        candidates = [target]
        if count and itemnum - len(excluded - {0}) < 198:
            pool = [i for i in range(1, itemnum + 1) if i not in excluded]
            candidates.extend(rng.sample(pool, count))
        else:
            while len(candidates) < count + 1:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
        cases.append((user, candidates))
    return cases


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200, cases=None):
    if cases is None:
        cases = make_validation_cases(user_train, user_valid, itemnum, num_users)
    metrics = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user, candidates in cases:
            seq = _history_to_tensor(user_train[user], maxlen, device)
            ids = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, ids).cpu().numpy()
            rank = int(np.flatnonzero(np.argsort(-scores, kind='stable') == 0)[0])
            for k in (10, 20):
                if rank < k:
                    metrics[f'HR@{k}'] += 1.0
                    metrics[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(len(cases), 1) for key, value in metrics.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum < 1:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint_path = os.path.join(working, f'{dataset_name}_model.pt')
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation users: {dataset_name}')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        accumulated = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN, NUM_NEGATIVES)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            # Materialize the broadcast view before tensor conversion.
            neg_t = torch.as_tensor(np.array(neg, copy=True), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {dataset_name} epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            accumulated += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN, cases=cases)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={accumulated / NUM_BATCHES:.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint_path)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = {'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
            'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
            'best_epoch': best_epoch, 'validation_hr10': best_hr}
    with open(os.path.join(working, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    _SAMPLER_CACHE.clear()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact training architecture, avoiding imports of training data code.
    header = '''from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

'''
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load(dataset_name):
    if dataset_name not in _DATASETS:
        raise ValueError("Unknown dataset: " + str(dataset_name))
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + "_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta["itemnum"], meta["hidden_units"], meta["num_layers"], meta["dropout_rate"]).to(_device)
        state = torch.load(os.path.join(_HERE, dataset_name + "_model.pt"), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset_name] = model
        _metas[dataset_name] = meta
    return _models[dataset_name], _metas[dataset_name]


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    model, meta = _load(ds)
    maxlen = int(meta["maxlen"])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    ids = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, ids).cpu().tolist()
    return [float(value) for value in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(GRU4RecModel) + '\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print('Selected-checkpoint validation macro HR@10 = %.6f' % float(np.mean(scores)), flush=True)


if __name__ == '__main__':
    main()
