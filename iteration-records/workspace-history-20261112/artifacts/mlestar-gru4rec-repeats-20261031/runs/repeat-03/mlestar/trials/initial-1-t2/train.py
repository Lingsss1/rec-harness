"""GRU4Rec with cached frequency-mixture sampling and 64-negative BPR."""
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
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
LR = float(os.environ.get('GRU4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('GRU4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
CACHE_SIZE = 1048576
LOSS_CHUNK = 512


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
    maximum = 0
    for split in ('train', 'valid'):
        data = _read_split(os.path.join(data_dir, f'{dataset_name}_{split}.txt'))
        for items in data.values():
            maximum = max(maximum, max(items, default=0))
    return maximum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def _sample_sequences(user_train, users, batch_size, maxlen):
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    chosen = []
    for row in range(batch_size):
        user = random.choice(users)
        chosen.append(user)
        history = user_train[user]
        prefix = history[:-1][-maxlen:]
        targets = history[1:][-maxlen:]
        seq[row, -len(prefix):] = prefix
        pos[row, -len(targets):] = targets
    return seq, pos, chosen


def sample_batch(user_train, itemnum, batch_size, maxlen):
    """Family-compatible diagnostic batch with one eligible negative per target."""
    users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
    if not users:
        raise RuntimeError('No trainable users with eligible negatives')
    seq, pos, chosen = _sample_sequences(user_train, users, batch_size, maxlen)
    neg = np.zeros_like(pos)
    for row, user in enumerate(chosen):
        forbidden = set(user_train[user])
        fallback = None
        for col in np.flatnonzero(pos[row]):
            for attempt in range(16):
                item = random.randint(1, itemnum)
                if item not in forbidden:
                    break
            else:
                if fallback is None:
                    fallback = np.asarray([i for i in range(1, itemnum + 1) if i not in forbidden], dtype=np.int64)
                item = int(fallback[random.randrange(len(fallback))])
            neg[row, col] = item
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
        sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        out, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return out

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        items = items.to(device=features.device, dtype=torch.long)
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == features.ndim:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return (features * self.item_emb(pos)).sum(-1), (features * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        embeddings = self.item_emb(candidates)
        if candidates.ndim == 1:
            return torch.matmul(embeddings, features[0])
        return (features[:, None, :] * embeddings).sum(-1)


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
    positive, negative = model(seq, pos, neg)
    mask = pos.ne(0)
    return F.softplus(negative[mask] - positive[mask]).mean()


class CachedNegativeSampler:
    """Only training interactions determine q and per-user exclusion sets."""
    def __init__(self, user_train, itemnum, device):
        self.itemnum = itemnum
        self.stride = itemnum + 1
        self.device = device
        self.users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        if not self.users:
            raise RuntimeError('No trainable users with eligible negatives')
        self.user_to_row = {u: r for r, u in enumerate(self.users)}
        self.histories = [np.asarray(sorted(set(user_train[u])), dtype=np.int64) for u in self.users]
        counts = np.zeros(itemnum + 1, dtype=np.float64)
        for history in user_train.values():
            np.add.at(counts, np.asarray(history, dtype=np.int64), 1)
        root = np.sqrt(counts[1:])
        if root.sum() <= 0:
            raise RuntimeError('Empty training-frequency distribution')
        q = 0.5 / itemnum + 0.5 * root / root.sum()
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
        keys = np.concatenate([h + row * self.stride for row, h in enumerate(self.histories)])
        self.forbidden = torch.as_tensor(keys, dtype=torch.long, device=device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(SEED)
        self.cache = None
        self.offset = 0
        self.eligible_cache = {}

    def draw(self, count):
        pieces = []
        remaining = count
        while remaining:
            if self.cache is None or self.offset == self.cache.numel():
                self.cache = torch.multinomial(self.q, CACHE_SIZE, replacement=True, generator=self.generator) + 1
                self.offset = 0
            take = min(remaining, self.cache.numel() - self.offset)
            pieces.append(self.cache[self.offset:self.offset + take])
            self.offset += take
            remaining -= take
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def invalid(self, negatives, row_ids, positives):
        keys = row_ids[:, None] * self.stride + negatives
        indices = torch.searchsorted(self.forbidden, keys.reshape(-1)).reshape_as(keys)
        bounded = indices.clamp(max=self.forbidden.numel() - 1)
        in_history = (indices < self.forbidden.numel()) & (self.forbidden[bounded] == keys)
        return in_history | negatives.eq(positives[:, None])

    @torch.no_grad()
    def sample(self, row_ids, positives):
        negatives = self.draw(row_ids.numel() * NUM_NEGATIVES).reshape(-1, NUM_NEGATIVES).clone()
        for _ in range(8):
            bad = self.invalid(negatives, row_ids, positives)
            count = int(bad.sum().item())
            if count == 0:
                return negatives
            negatives[bad] = self.draw(count)
        bad = self.invalid(negatives, row_ids, positives)
        if bool(bad.any().item()):
            coordinates = bad.nonzero(as_tuple=False)
            flat_rows = coordinates[:, 0]
            user_rows = row_ids[flat_rows]
            for user_row in torch.unique(user_rows).cpu().tolist():
                if user_row not in self.eligible_cache:
                    eligible = np.setdiff1d(np.arange(1, self.itemnum + 1, dtype=np.int64),
                                            self.histories[user_row], assume_unique=True)
                    self.eligible_cache[user_row] = torch.as_tensor(eligible, dtype=torch.long, device=self.device)
                eligible = self.eligible_cache[user_row]
                selected = coordinates[user_rows == user_row]
                # Positives belong to the user's training history and are excluded here.
                weights = self.q[eligible - 1]
                choices = torch.multinomial(weights, selected.size(0), replacement=True, generator=self.generator)
                negatives[selected[:, 0], selected[:, 1]] = eligible[choices]
        return negatives


def backward_multineg_bpr(model, seq, pos, chosen, sampler):
    valid = pos.ne(0)
    features = model.log2feats(seq)[valid]
    positives = pos[valid]
    batch_rows = torch.tensor([sampler.user_to_row[u] for u in chosen], dtype=torch.long, device=seq.device)
    row_ids = batch_rows[:, None].expand_as(pos)[valid]
    negatives = sampler.sample(row_ids, positives)
    # Accumulate exact feature and item gradients in chunks to bound activation memory.
    proxy = features.detach().requires_grad_(True)
    total = features.size(0)
    loss_value = 0.0
    for start in range(0, total, LOSS_CHUNK):
        end = min(start + LOSS_CHUNK, total)
        feat = proxy[start:end]
        positive_scores = (feat * model.item_emb(positives[start:end])).sum(-1)
        negative_scores = torch.einsum('bd,bkd->bk', feat, model.item_emb(negatives[start:end]))
        loss = F.softplus(negative_scores - positive_scores[:, None]).mean()
        weighted = loss * ((end - start) / total)
        weighted.backward()
        loss_value += float(weighted.detach())
    features.backward(proxy.grad)
    return loss_value


def make_validation(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        forbidden = set(history) | {0, target}
        available = itemnum - len({i for i in forbidden if 1 <= i <= itemnum})
        wanted = min(99, available)
        items = [target]
        for _ in range(wanted):
            for attempt in range(100):
                item = rng.randint(1, itemnum)
                if item not in forbidden:
                    break
            else:
                eligible = [i for i in range(1, itemnum + 1) if i not in forbidden]
                item = rng.choice(eligible)
            forbidden.add(item)
            items.append(item)
        sequence = [0] * maxlen
        tail = history[-maxlen:]
        if tail:
            sequence[-len(tail):] = tail
        sequences.append(sequence)
        candidates.append(items)
    return sequences, candidates


@torch.no_grad()
def evaluate_cached(model, validation, device):
    sequences, candidates = validation
    if not sequences:
        return {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0}
    model.eval()
    hits10, hits20, ndcg10 = 0.0, 0.0, 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.tensor(sequences[start:start + 64], dtype=torch.long, device=device)
        group = candidates[start:start + 64]
        width = max(map(len, group))
        padded = [items + [0] * (width - len(items)) for items in group]
        cand = torch.tensor(padded, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, cand).masked_fill(cand.eq(0), -torch.inf)
        # Strict comparison gives the target its original first-candidate tie order.
        ranks = (scores > scores[:, :1]).sum(-1).cpu().tolist()
        for rank in ranks:
            hits10 += rank < 10
            hits20 += rank < 20
            if rank < 10:
                ndcg10 += 1.0 / math.log2(rank + 2)
    n = len(sequences)
    model.train()
    return {'HR@10': hits10 / n, 'HR@20': hits20 / n, 'NDCG@10': ndcg10 / n}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cached(model, make_validation(user_train, user_valid, itemnum, maxlen), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_all()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} negatives={NUM_NEGATIVES}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = CachedNegativeSampler(user_train, itemnum, device)
    validation = make_validation(user_train, user_valid, itemnum, MAXLEN)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, f'{dataset_name}_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, chosen = _sample_sequences(user_train, sampler.users, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            total_loss += backward_multineg_bpr(model, seq, pos, chosen, sampler)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cached(model, validation, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} validation={json.dumps(metrics)}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at {epoch}', flush=True)
                break
    with open(os.path.join(working, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump({'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
                     'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
                     'best_epoch': best_epoch, 'num_negatives': NUM_NEGATIVES}, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer, sampler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained class so prediction does not import training or data.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    model_source = inspect.getsource(GRU4RecModel)
    predictor = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(_DATASETS):
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in _DATASETS:
        if dataset in _models:
            continue
        with open(os.path.join(_HERE, f'{dataset}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
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
    if len(candidates) == 0:
        return []
    model, meta = _models[dataset], _metas[dataset]
    maxlen = int(meta['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + model_source + '\n' + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro-average validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
