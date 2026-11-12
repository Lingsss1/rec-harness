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

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
LR = float(os.environ.get('GRU4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('GRU4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
CACHE_SIZE = 1048576
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


seed_all()


def _read_split(path):
    result = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as handle:
            stats = json.load(handle)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    maximum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 2:
                        maximum = max(maximum, int(parts[1]))
    return maximum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class CachedNegativeSampler:
    def __init__(self, user_train, itemnum, device):
        self.device = device
        self.itemnum = itemnum
        self.users = [u for u, h in user_train.items() if len(h) >= 2]
        if not self.users:
            raise RuntimeError('No users with at least two training interactions')
        self.user_train = user_train
        self.user_to_row = {u: row for row, u in enumerate(self.users)}
        counts = np.zeros(itemnum + 1, dtype=np.float64)
        for history in user_train.values():
            np.add.at(counts, np.asarray(history, dtype=np.int64), 1)
        weights = np.sqrt(counts[1:])
        if weights.sum() <= 0:
            raise RuntimeError('No training item counts')
        q = 0.5 / itemnum + 0.5 * weights / weights.sum()
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(SEED)
        self.cache = torch.empty(0, dtype=torch.long, device=device)
        self.cursor = 0
        keys = []
        self.histories = []
        for row, user in enumerate(self.users):
            history_set = set(user_train[user])
            history_set.discard(0)
            if len(history_set) >= itemnum:
                raise RuntimeError(f'User {user} has no eligible training negatives')
            self.histories.append(history_set)
            keys.extend(row * (itemnum + 1) + item for item in history_set)
        self.keys = torch.tensor(sorted(keys), dtype=torch.long, device=device)

    def draw(self, count):
        pieces = []
        remaining = count
        while remaining:
            if self.cursor == self.cache.numel():
                self.cache = torch.multinomial(self.q, CACHE_SIZE, replacement=True,
                                               generator=self.generator) + 1
                self.cursor = 0
            take = min(remaining, self.cache.numel() - self.cursor)
            pieces.append(self.cache[self.cursor:self.cursor + take])
            self.cursor += take
            remaining -= take
        return torch.cat(pieces) if len(pieces) > 1 else pieces[0].clone()

    def forbidden(self, rows, positives, negatives):
        encoded = rows[:, None] * (self.itemnum + 1) + negatives
        locations = torch.searchsorted(self.keys, encoded)
        locations = locations.clamp(max=self.keys.numel() - 1)
        return (self.keys[locations] == encoded) | (negatives == positives[:, None])

    @torch.no_grad()
    def sample(self, rows, positives, k=NUM_NEGATIVES):
        negatives = self.draw(rows.numel() * k).reshape(-1, k)
        for _ in range(8):
            bad = self.forbidden(rows, positives, negatives)
            count = int(bad.sum().item())
            if count == 0:
                return negatives
            negatives[bad] = self.draw(count)
        bad = self.forbidden(rows, positives, negatives)
        if bad.any():
            # Bounded rejection: use the exact eligible distribution for remaining rows.
            for row in rows[bad.any(dim=1)].unique().tolist():
                row_mask = (rows == row)[:, None] & bad
                count = int(row_mask.sum().item())
                eligible_q = self.q.clone()
                excluded = torch.tensor(sorted(self.histories[row]), dtype=torch.long,
                                        device=self.device) - 1
                eligible_q[excluded] = 0
                # Positives normally belong to the training history; exclude explicitly too.
                positive_ids = positives[rows == row].unique() - 1
                eligible_q[positive_ids] = 0
                if not bool(eligible_q.sum() > 0):
                    raise RuntimeError('No eligible negative after rejection')
                negatives[row_mask] = torch.multinomial(
                    eligible_q, count, replacement=True, generator=self.generator) + 1
        return negatives


def sample_batch(user_train, itemnum, batch_size, maxlen, sampler=None,
                 return_rows=False):
    # Keep the family-specific diagnostic interface: sequence, positive, negative.
    if sampler is None:
        sampler = CachedNegativeSampler(user_train, itemnum, torch.device('cpu'))
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    rows = np.empty(batch_size, dtype=np.int64)
    for b in range(batch_size):
        user = random.choice(sampler.users)
        rows[b] = sampler.user_to_row[user]
        history = user_train[user]
        n = min(maxlen, len(history) - 1)
        sequences[b, -n:] = history[-n - 1:-1]
        positives[b, -n:] = history[-n:]
    if return_rows:
        return sequences, positives, rows
    negative = np.zeros_like(positives)
    valid = positives != 0
    expanded_rows = np.broadcast_to(rows[:, None], positives.shape)[valid].copy()
    sampled = sampler.sample(
        torch.tensor(expanded_rows, device=sampler.device),
        torch.tensor(positives[valid], device=sampler.device), k=1)
    negative[valid] = sampled[:, 0].cpu().numpy()
    return sequences, positives, negative


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
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
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long,
                                       device=self.item_emb.weight.device)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.size(1), device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(sequence.size(0), device=sequence.device), last]

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long,
                                    device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        raise ValueError('Unsupported feature/item shapes')

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        if candidates.ndim == 1:
            scores = features @ self.item_emb(candidates).T
        else:
            scores = self.predict_score(features, candidates)
        return scores[0] if scores.shape[0] == 1 else scores


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


def bpr_sequence_loss(model, seq, pos, neg):
    positive_scores, negative_scores = model(seq, pos, neg)
    return F.softplus(negative_scores[pos != 0] - positive_scores[pos != 0]).mean()


def multi_negative_loss(model, seq, pos, rows, sampler):
    valid = pos != 0
    features = model.log2feats(seq)[valid]
    targets = pos[valid]
    target_rows = rows[:, None].expand_as(pos)[valid]
    negatives = sampler.sample(target_rows, targets)

    def chunk_loss(feat, target, negative):
        positive_scores = (feat * model.item_emb(target)).sum(-1)
        negative_scores = (feat[:, None, :] * model.item_emb(negative)).sum(-1)
        return F.softplus(negative_scores - positive_scores[:, None]).sum()

    # Recompute embedding products during backward to bound the K=64 memory cost.
    total = features.new_zeros(())
    for start in range(0, targets.numel(), 512):
        end = start + 512
        total = total + checkpoint(chunk_loss, features[start:end], targets[start:end],
                                   negatives[start:end], use_reentrant=False)
    return total / (targets.numel() * NUM_NEGATIVES)


def make_validation(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        target = user_valid[user][0]
        history = user_train[user]
        excluded = set(history) | {0, target}
        if itemnum - len(excluded - {0}) < 99:
            raise RuntimeError('Insufficient unique validation negatives')
        selected = [target]
        for _ in range(10000):
            if len(selected) == 100:
                break
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                selected.append(item)
        if len(selected) < 100:
            eligible = [i for i in range(1, itemnum + 1) if i not in excluded]
            selected.extend(rng.sample(eligible, 100 - len(selected)))
        sequence = [0] * maxlen
        tail = history[-maxlen:]
        if tail:
            sequence[-len(tail):] = tail
        sequences.append(sequence)
        candidates.append(selected)
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_cached(model, validation, device):
    sequences, candidates = validation
    if len(sequences) == 0:
        return 0.0, 0.0
    model.eval()
    hits = ndcg = 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start + 64], device=device)
        cand = torch.as_tensor(candidates[start:start + 64], device=device)
        features = model.encode(seq)
        scores = model.predict_score(features, cand)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        hits += (ranks < 10).sum().item()
        ndcg += torch.where(ranks < 10, 1.0 / torch.log2(ranks.float() + 2), 0.0).sum().item()
    model.train()
    return hits / len(sequences), ndcg / len(sequences)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    hr, ndcg = evaluate_cached(model, make_validation(user_train, user_valid, itemnum, maxlen), device)
    return {'HR@10': hr, 'NDCG@10': ndcg}


def train_one_dataset(dataset_name, device, data_dir):
    seed_all()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = CachedNegativeSampler(user_train, itemnum, device)
    validation = make_validation(user_train, user_valid, itemnum, MAXLEN)
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, rows = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN,
                                          sampler=sampler, return_rows=True)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = multi_negative_loss(model, seq_t, pos_t, rows_t, sampler)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'Nonfinite training loss: {dataset_name}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_cached(model, validation, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} '
                  f'internal_val_HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint selected')
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as handle:
        pickle.dump({'itemnum': itemnum, 'maxlen': MAXLEN,
                     'hidden_units': HIDDEN_UNITS, 'num_layers': NUM_LAYERS,
                     'dropout_rate': DROPOUT_RATE, 'best_epoch': best_epoch,
                     'negative_count': NUM_NEGATIVES}, handle)
    del sampler, optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture, making inference independent of importing train.py.
    import inspect
    architecture = inspect.getsource(GRU4RecModel)
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name in _models:
        return _models[dataset_name], _metas[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'],
                        meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta
    return model, meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model, meta = _load(dataset_name)
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq_t = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t)
    return [float(score) for score in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(header + architecture + '\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Internal validation macro HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
