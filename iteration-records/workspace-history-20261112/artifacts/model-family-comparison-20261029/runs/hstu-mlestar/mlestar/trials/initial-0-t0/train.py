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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN_UNITS = 64
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT_RATE = 0.2
MAXLEN = 128
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('HSTU_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
LR = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE = 20
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_split(path):
    result = defaultdict(list)
    with open(path, 'r') as stream:
        for line in stream:
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
        with open(stats_path, 'r') as stream:
            stats = json.load(stream)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((item for split in (train, valid) for history in split.values() for item in history), default=0)
    if not train or itemnum < 2:
        raise ValueError('Missing or invalid training data for ' + dataset_name)
    return train, valid, itemnum


_sampler_cache = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    cached = _sampler_cache.get(key)
    if cached is None or cached[0] is not user_train:
        users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        if not users:
            raise ValueError('No eligible training users')
        seen = {u: set(user_train[u]) for u in users}
        cached = (user_train, users, seen)
        _sampler_cache[key] = cached
    _, users, seen = cached
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(len(history) - 1, maxlen)
        seq[row, -length:] = history[-length-1:-1]
        pos[row, -length:] = history[-length:]
        for col in range(maxlen - length, maxlen):
            candidate = random.randint(1, itemnum)
            while candidate in seen[user]:
                candidate = random.randint(1, itemnum)
            neg[row, col] = candidate
    return seq, pos, neg


class HSTUBlock(nn.Module):
    """Causal UVQK transduction with SiLU attention and multiplicative gating."""
    def __init__(self, hidden_units, num_heads, dropout_rate, maxlen):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.maxlen = maxlen
        self.input_norm = nn.LayerNorm(hidden_units)
        self.uvqk = nn.Linear(hidden_units, 4 * hidden_units)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * maxlen - 1))
        self.attention_norm = nn.LayerNorm(hidden_units)
        self.output = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        positions = torch.arange(maxlen)
        relative = positions[:, None] - positions[None, :] + maxlen - 1
        self.register_buffer('relative_index', relative, persistent=False)
        self.register_buffer('causal', torch.ones(maxlen, maxlen, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x, valid):
        batch, length, width = x.shape
        u, v, q, k = F.silu(self.uvqk(self.input_norm(x))).chunk(4, dim=-1)
        def split_heads(t):
            return t.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        bias = self.relative_bias[:, self.relative_index[:length, :length]]
        weights = F.silu(torch.matmul(q, k.transpose(-1, -2)) + bias.unsqueeze(0)) / float(self.maxlen)
        allowed = self.causal[:length, :length][None, None, :, :]
        allowed = allowed & valid[:, None, None, :] & valid[:, None, :, None]
        weights = weights * allowed.to(weights.dtype)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
        update = self.output(u * self.attention_norm(attended))
        return (x + self.dropout(update)) * valid.unsqueeze(-1).to(x.dtype)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('Hidden size must be divisible by number of heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            HSTUBlock(hidden_units, num_heads, dropout_rate, maxlen)
            for _ in range(num_blocks)
        ])
        self.last_norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        valid = sequence.ne(0)
        length = sequence.shape[1]
        positions = torch.arange(1, length + 1, device=sequence.device)[None, :] * valid.long()
        x = self.item_emb(sequence) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.emb_dropout(x) * valid.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, valid)
        return self.last_norm(x) * valid.unsqueeze(-1).to(x.dtype)

    def forward(self, sequence):
        return self.log2feats(sequence)

    def final_features(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
        last = torch.where(sequence.ne(0), positions, -1).amax(dim=1)
        result = features[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return result * last.ge(0).unsqueeze(-1).to(result.dtype)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)


class ModelView(nn.Module):
    """Expose the family model interface for training diagnostics."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def item_emb(self):
        return self.model.item_emb

    def forward(self, sequence):
        return self.model(sequence)

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def build_validation(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in user_valid if user_valid[u] and user_train.get(u)]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | set(user_valid[user])
        needed = min(99, itemnum - len(excluded))
        negatives = []
        chosen = set()
        if itemnum - len(excluded) < 200:
            pool = [i for i in range(1, itemnum + 1) if i not in excluded]
            negatives = rng.sample(pool, needed)
        else:
            while len(negatives) < needed:
                item = rng.randint(1, itemnum)
                if item not in excluded and item not in chosen:
                    chosen.add(item)
                    negatives.append(item)
        seq = [0] * maxlen
        tail = history[-maxlen:]
        seq[-len(tail):] = tail
        sequences.append(seq)
        candidates.append([target] + negatives)
    return sequences, candidates


@torch.no_grad()
def evaluate_val(model, validation, device):
    model.eval()
    sequences, candidates = validation
    if not sequences:
        raise ValueError('No validation users')
    hits = 0
    ndcg = 0.0
    for start in range(0, len(sequences), 64):
        features = model.final_features(torch.tensor(sequences[start:start + 64], device=device))
        for offset, feature in enumerate(features):
            items = torch.tensor(candidates[start + offset], dtype=torch.long, device=device)
            scores = model.predict_score(feature, items)
            order = torch.argsort(scores, descending=True, stable=True)
            rank = int((order == 0).nonzero(as_tuple=False)[0, 0].item())
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    return {'HR@10': hits / len(sequences), 'NDCG@10': ndcg / len(sequences)}


def train_one_dataset(dataset_name, device):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    validation = build_validation(user_train, user_valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, dataset_name + '_model.pt')
    best = (-1.0, -1.0)
    best_epoch = 0
    stale = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            features = model.log2feats(seq)
            mask = pos.ne(0)
            active = features[mask]
            positive = (active * model.item_emb(pos[mask])).sum(dim=-1)
            negative = (active * model.item_emb(neg[mask])).sum(dim=-1)
            loss = F.softplus(negative - positive).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss for ' + dataset_name)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, validation, device)
            score = (metrics['HR@10'], metrics['NDCG@10'])
            print(json.dumps(dict(dataset=dataset_name, epoch=epoch,
                                  loss=total_loss / NUM_BATCHES, local_validation=metrics)), flush=True)
            if score > best:
                best = score
                best_epoch = epoch
                stale = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    with open(os.path.join(WORKING, dataset_name + '_meta.pkl'), 'wb') as stream:
        pickle.dump(dict(config=config, architecture='native_silu_hstu_v1',
                         best_epoch=best_epoch, validation_hr10=best[0]), stream)
    print(dataset_name, 'selected epoch', best_epoch, 'local validation HR@10', best[0], flush=True)
    _sampler_cache.clear()
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best[0]


def write_predict_script(working_dir):
    header = '''from __future__ import annotations
import math
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = None


def _load(dataset_name):
    global _device
    if dataset_name in _models:
        return _models[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
        meta = pickle.load(stream)
    if meta['architecture'] != 'native_silu_hstu_v1':
        raise ValueError('Architecture mismatch')
    model = HSTURec(**meta['config']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = os.environ.get('GAGC_DATASET')
        if dataset_name is None:
            configured = [s.strip() for s in os.environ.get('GAGC_DATASETS', '').split(',') if s.strip()]
            if len(configured) == 1:
                dataset_name = configured[0]
            else:
                raise ValueError('dataset_name is required for four-category routing')
    model = _load(dataset_name)
    seq = [0] * model.maxlen
    tail = [int(i) for i in history[-model.maxlen:]]
    if any(i < 0 or i > model.itemnum for i in tail):
        raise ValueError('History item outside dataset vocabulary')
    if tail:
        seq[-len(tail):] = tail
    with torch.inference_mode():
        features = model.final_features(torch.tensor([seq], dtype=torch.long, device=_device))
        items = torch.tensor(candidates, dtype=torch.long, device=_device)
        valid = (items >= 1) & (items <= model.itemnum)
        scores = model.predict_score(features, items.clamp(0, model.itemnum))[0]
        scores = scores.masked_fill(~valid, float('-inf'))
        return scores.float().cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as stream:
        stream.write(header + '\n' + inspect.getsource(HSTUBlock) + '\n' +
                     inspect.getsource(HSTURec) + '\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Native SiLU HSTU; training seed=42; supplied splits unchanged; device=' + str(device), flush=True)
    write_predict_script(WORKING)
    scores = [train_one_dataset(dataset_name, device) for dataset_name in DATASETS]
    print(json.dumps({'local_validation_macro_HR@10': sum(scores) / len(DATASETS)}), flush=True)


if __name__ == '__main__':
    main()
