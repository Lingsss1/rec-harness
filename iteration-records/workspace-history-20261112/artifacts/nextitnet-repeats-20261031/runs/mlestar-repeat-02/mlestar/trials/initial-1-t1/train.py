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
DATASETS = [s.strip() for s in os.environ.get('GAGC_DATASETS', ','.join(DEFAULT_DATASETS)).split(',') if s.strip()]
if len(DATASETS) != 4 or set(DATASETS) != set(DEFAULT_DATASETS):
    raise ValueError('All four Amazon categories must be retained')
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
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


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if itemnum == 0:
        itemnum = max((i for split in (train, valid) for history in split.values() for i in history), default=0)
    return train, valid, itemnum


def _sampler_state(user_train, itemnum):
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        users = [u for u, h in user_train.items() if len(h) >= 2]
        if not users:
            raise RuntimeError('No training users with at least two interactions')
        seen = {u: set(user_train[u]) for u in users}
        _SAMPLER_CACHE[key] = (user_train, users, seen, {})
    return _SAMPLER_CACHE[key]


def _uniform_negatives(itemnum, excluded, k, dense_cache=None, user=None):
    available = itemnum - sum(1 for i in excluded if 1 <= i <= itemnum)
    if available < k:
        raise RuntimeError(f'Need {k} distinct unseen negatives; only {available} available')
    if available < itemnum // 2:
        if dense_cache is not None and user in dense_cache:
            pool = dense_cache[user]
        else:
            pool = np.asarray([i for i in range(1, itemnum + 1) if i not in excluded], dtype=np.int64)
            if dense_cache is not None:
                dense_cache[user] = pool
        return np.random.choice(pool, size=k, replace=False)
    chosen = set()
    values = []
    while len(values) < k:
        draws = np.random.randint(1, itemnum + 1, size=max(16, 2 * (k - len(values))))
        for value in draws:
            value = int(value)
            if value not in excluded and value not in chosen:
                chosen.add(value)
                values.append(value)
                if len(values) == k:
                    break
    return np.asarray(values, dtype=np.int64)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    # Sequences and positive targets retain the baseline's left-padded layout.
    # Each row of negatives is a distinct uniform set shared across positions.
    _, users, seen, dense_cache = _sampler_state(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.empty((batch_size, NUM_NEGATIVES), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        neg[row] = _uniform_negatives(itemnum, seen[user], NUM_NEGATIVES, dense_cache, user)
    return seq, pos, neg


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
        # Retain original absolute indices when empty leading columns are trimmed.
        positions = torch.arange(self.maxlen - length + 1, self.maxlen + 1, device=seq.device)
        positions = positions.unsqueeze(0).expand(batch, -1) * seq.ne(0).long()
        mask = seq.ne(0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
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
        if features.ndim == embeddings.ndim and features.shape[:-1] == embeddings.shape[:-1]:
            return (features * embeddings).sum(-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        negative = torch.bmm(features, self.item_emb(neg).transpose(1, 2))
        return positive, negative

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(1).values.clamp_min(0)
        final = features[torch.arange(seq.shape[0], device=seq.device), last]
        embeddings = self.item_emb(candidates)
        if candidates.ndim == 1:
            scores = final @ embeddings.t()
        else:
            scores = (final.unsqueeze(1) * embeddings).sum(-1)
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

    def forward(self, *args):
        return self.model(*args)


def sampled_categorical_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos.ne(0)
    logits = torch.cat((positive.unsqueeze(-1), negative), dim=-1)[mask]
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    # Sampled categorical cross-entropy, not exact full-vocabulary softmax.
    return F.cross_entropy(logits, labels)


def _history_to_tensor(history, maxlen, device):
    array = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        array[-len(tail):] = tail
    return torch.as_tensor(array[None], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, maxlen, num_users=200):
    rng = random.Random(42)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    sequences, candidates = [], []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {0, target}
        if itemnum - sum(1 for x in excluded if 1 <= x <= itemnum) < 99:
            raise RuntimeError('Validation requires 99 distinct unseen candidates')
        row = [target]
        while len(row) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                row.append(item)
        history = user_train[user][-maxlen:]
        sequences.append([0] * (maxlen - len(history)) + history)
        candidates.append(row)
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


def evaluate_cases(model, cases, device):
    sequences, candidates = cases
    if len(sequences) == 0:
        return {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0}
    totals = np.zeros(3, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), 64):
            seq = torch.as_tensor(sequences[start:start + 64], device=device)
            cand = torch.as_tensor(candidates[start:start + 64], device=device)
            scores = model.score_candidates(seq, cand)
            if scores.ndim == 1:
                scores = scores.unsqueeze(0)
            # Stable candidate ordering: the positive occupies index zero.
            ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
            totals[0] += ranks.lt(10).sum().item()
            totals[1] += ranks.lt(20).sum().item()
            totals[2] += (ranks.lt(10).float() / torch.log2(ranks.float() + 2)).sum().item()
    model.train()
    totals /= len(sequences)
    return dict(zip(('HR@10', 'HR@20', 'NDCG@10'), map(float, totals)))


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, maxlen, num_users), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= NUM_NEGATIVES:
        raise RuntimeError('Empty or insufficient training vocabulary: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model = NextItNetModel(itemnum, HIDDEN_UNITS, NUM_BLOCKS, KERNEL_SIZE, DROPOUT_RATE, MAXLEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    ckpt_path = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            # Keep one empty boundary column when trimming; blocks mask padding.
            occupied = np.flatnonzero(np.any(seq != 0, axis=0))
            first = max(0, int(occupied[0]) - 1)
            seq_t = torch.as_tensor(seq[:, first:].copy(), dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos[:, first:].copy(), dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = sampled_categorical_loss(model, seq_t, pos_t, neg_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.detach()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} sampled_ce={epoch_loss.item()/NUM_BATCHES:.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early_stop={epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint selected')
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS, kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best_validation_HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    _SAMPLER_CACHE.clear()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the actual trained architecture so inference has no train.py import
    # side effects and cannot drift from the checkpoint definition.
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    tail = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in _DATASETS:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = NextItNetModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if len(candidates) == 0:
        return []
    model = _models[dataset]
    maxlen = int(_metas[dataset]['maxlen'])
    history = list(history)[-maxlen:]
    sequence = [0] * (maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + tail)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    write_predict_script(WORKING_DIR)
    print('Macro-average validation HR@10 = %.6f' % (sum(scores) / 4), flush=True)


if __name__ == '__main__':
    main()
