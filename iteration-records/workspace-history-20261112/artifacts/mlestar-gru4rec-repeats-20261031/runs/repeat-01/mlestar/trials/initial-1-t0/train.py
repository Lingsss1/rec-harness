"""One independently trained GRU4Rec model per Amazon category, with BPR-max."""
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
DROPOUT_RATE = 0.1
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 33
BPR_MAX_COEFFICIENT = 0.1
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
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
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def _sampler_state(user_train, itemnum):
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        users = []
        forbidden = {}
        complements = {}
        for user, history in user_train.items():
            if len(history) < 2:
                continue
            seen = set(int(x) for x in history if 1 <= x <= itemnum)
            if len(seen) >= itemnum:
                continue
            users.append(user)
            forbidden[user] = seen
            if len(seen) > itemnum // 2:
                complements[user] = np.asarray([x for x in range(1, itemnum + 1) if x not in seen], dtype=np.int64)
        if not users:
            raise RuntimeError('No eligible causal training sequences with negative items')
        _SAMPLER_CACHE[key] = (users, forbidden, complements)
    return _SAMPLER_CACHE[key]


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    """Default three-array interface is unchanged; training requests 33 negatives.

    Each positive is the item immediately following its causal input. Negatives
    are drawn with replacement from items outside that user's training history.
    """
    users, forbidden, complements = _sampler_state(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        inputs = history[:-1][-maxlen:]
        targets = history[1:][-maxlen:]
        length = len(inputs)
        seq[row, -length:] = inputs
        pos[row, -length:] = targets
        shape = (length, num_negatives)
        if user in complements:
            pool = complements[user]
            draws = pool[np.random.randint(0, len(pool), size=shape)]
        else:
            draws = np.random.randint(1, itemnum + 1, size=shape)
            # Vectorized rejection has high acceptance for sparse histories.
            seen = np.fromiter(forbidden[user], dtype=np.int64)
            invalid = np.isin(draws, seen)
            while invalid.any():
                draws[invalid] = np.random.randint(1, itemnum + 1, size=int(invalid.sum()))
                invalid = np.isin(draws, seen)
        neg[row, -length:, :] = draws
    return seq, pos, neg[:, :, 0] if num_negatives == 1 else neg


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
        output, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return output

    def encode(self, seq):
        if not torch.is_tensor(seq):
            seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            seq = seq.to(device=self.item_emb.weight.device, dtype=torch.long)
        output = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        features = output[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        pos_scores = (features * self.item_emb(pos)).sum(dim=-1)
        if neg.ndim == seq.ndim:
            neg_scores = (features * self.item_emb(neg)).sum(dim=-1)
        else:
            neg_scores = (features.unsqueeze(-2) * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq), candidates)[0]


class ModelView(nn.Module):
    """Transparent family-specific adapter for diagnosis."""
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


def bpr_max_loss(pos_scores, neg_scores, coefficient=0.1):
    log_weights = F.log_softmax(neg_scores.float(), dim=-1)
    log_win = F.logsigmoid(pos_scores.float()[:, None] - neg_scores.float())
    ranking = -torch.logsumexp(log_weights + log_win, dim=-1)
    regularizer = (log_weights.exp() * neg_scores.float().square()).sum(dim=-1)
    return (ranking + coefficient * regularizer).mean()


def training_loss(model, seq, pos, neg):
    # Select only valid positions before negative embedding lookup or loss.
    features = model.log2feats(seq)
    mask = pos.ne(0)
    features = features[mask]
    positive_ids = pos[mask]
    negative_ids = neg[mask]
    if not features.size(0):
        return features.sum() * 0.0
    # Chunking bounds the temporary [positions, negatives, hidden] tensor.
    total = features.sum() * 0.0
    count = features.size(0)
    for start in range(0, count, 512):
        end = min(start + 512, count)
        hidden = features[start:end]
        positives = (hidden * model.item_emb(positive_ids[start:end])).sum(-1)
        negatives = (hidden[:, None, :] * model.item_emb(negative_ids[start:end])).sum(-1)
        total = total + bpr_max_loss(positives, negatives, BPR_MAX_COEFFICIENT) * ((end - start) / count)
    return total


def _history_to_tensor(history, maxlen, device):
    result = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        result[-len(tail):] = tail
    return torch.as_tensor(result[None, :], dtype=torch.long, device=device)


def build_validation_cases(user_train, user_valid, itemnum):
    # Fixed seed, 200 users, and 99 distinct unseen negative candidates.
    # This local validation never accesses test labels; the host is authoritative.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        available = itemnum - sum(1 for x in excluded if 1 <= x <= itemnum)
        if available < 99:
            raise RuntimeError('Validation requires 99 unique eligible negatives')
        candidates = [target]
        while len(candidates) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        cases.append((user_train[user], candidates))
    return cases


def evaluate_val(model, cases, device):
    if not cases:
        raise RuntimeError('No validation users')
    hr10 = hr20 = ndcg10 = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cases), 64):
            chunk = cases[start:start + 64]
            sequences = np.zeros((len(chunk), MAXLEN), dtype=np.int64)
            for row, (history, _) in enumerate(chunk):
                tail = history[-MAXLEN:]
                if tail:
                    sequences[row, -len(tail):] = tail
            seq = torch.as_tensor(sequences, dtype=torch.long, device=device)
            candidates = torch.tensor([c for _, c in chunk], dtype=torch.long, device=device)
            scores = model.predict_score(model.encode(seq), candidates)
            ranks = torch.argsort(torch.argsort(-scores, dim=1, stable=True), dim=1, stable=True)[:, 0].cpu().tolist()
            for rank in ranks:
                hr10 += float(rank < 10)
                hr20 += float(rank < 20)
                ndcg10 += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
    model.train(was_training)
    return {'HR@10': hr10 / len(cases), 'HR@20': hr20 / len(cases), 'NDCG@10': ndcg10 / len(cases)}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    cases = build_validation_cases(user_train, user_valid, itemnum)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    checkpoint_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN, NUM_NEGATIVES)
            seq, pos, neg = [torch.as_tensor(x, dtype=torch.long, device=device) for x in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite BPR-max loss in {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} local_validation={json.dumps(metrics)}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump({'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
                     'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
                     'loss': 'bpr-max', 'num_negatives': NUM_NEGATIVES,
                     'coefficient': BPR_MAX_COEFFICIENT, 'selected_epoch': best_epoch}, f)
    _SAMPLER_CACHE.clear()
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture, so prediction requires no import of
    # train.py, no training inputs, and no environment-dependent reconstruction.
    architecture = inspect.getsource(GRU4RecModel)
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    predictor = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(x) for x in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n' + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    write_predict_script(WORKING_DIR)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f'Local validation macro HR@10: {sum(scores) / len(scores):.6f}; host validation is authoritative.', flush=True)


if __name__ == '__main__':
    main()
