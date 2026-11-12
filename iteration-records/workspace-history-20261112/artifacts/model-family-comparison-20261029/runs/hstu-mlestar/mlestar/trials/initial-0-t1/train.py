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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN_UNITS = 64
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT_RATE = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = 200
MAXLEN = 128
NUM_NEGATIVES = 128
TEMPERATURE = 0.05
LOSS_CHUNK = 512
PATIENCE = 100
EVAL_USERS = 200
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as stream:
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
        with open(stats_path) as stream:
            stats = json.load(stream)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if itemnum <= 0:
        itemnum = max((item for split in (train, valid) for history in split.values() for item in history), default=0)
    if itemnum <= 0:
        raise ValueError('Empty catalog for ' + dataset_name)
    return train, valid, itemnum


class TrainingSampler:
    def __init__(self, train, itemnum):
        self.train = train
        self.itemnum = itemnum
        self.users = [u for u, history in train.items() if len(history) >= 2]
        self.histories = [train[u] for u in self.users]
        self.seen = [set(history) for history in self.histories]
        if not self.users:
            raise ValueError('No users with two training interactions')
        if any(len(seen) >= itemnum for seen in self.seen):
            raise ValueError('A training user has no eligible negative items')
        self.forbidden = None

    def sample(self, batch_size, maxlen, return_users=False):
        rows = np.random.randint(len(self.users), size=batch_size)
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        for b, row in enumerate(rows):
            history = self.histories[row]
            n = min(len(history) - 1, maxlen)
            seq[b, -n:] = history[-n-1:-1]
            pos[b, -n:] = history[-n:]
            if not return_users:
                for j in range(maxlen - n, maxlen):
                    candidate = random.randint(1, self.itemnum)
                    while candidate in self.seen[row]:
                        candidate = random.randint(1, self.itemnum)
                    neg[b, j] = candidate
        if return_users:
            return seq, pos, neg, rows
        return seq, pos, neg

    def prepare_forbidden(self, device):
        stride = self.itemnum + 1
        encoded = [row * stride + item for row, seen in enumerate(self.seen) for item in seen]
        self.forbidden = torch.tensor(sorted(encoded), dtype=torch.long, device=device)

    def sample_negatives(self, user_rows, count):
        shape = (user_rows.numel(), count)
        negatives = torch.randint(1, self.itemnum + 1, shape, device=user_rows.device)
        offsets = user_rows[:, None] * (self.itemnum + 1)
        while True:
            keys = offsets + negatives
            indices = torch.searchsorted(self.forbidden, keys)
            invalid = self.forbidden[indices.clamp(max=self.forbidden.numel() - 1)] == keys
            if not bool(invalid.any()):
                return negatives
            negatives[invalid] = torch.randint(1, self.itemnum + 1, (int(invalid.sum()),), device=user_rows.device)


_SAMPLER_CACHE = {}


def sample_batch(user_train, itemnum, batch_size, maxlen, return_users=False):
    key = (id(user_train), itemnum)
    sampler = _SAMPLER_CACHE.get(key)
    if sampler is None:
        sampler = TrainingSampler(user_train, itemnum)
        _SAMPLER_CACHE[key] = sampler
    return sampler.sample(batch_size, maxlen, return_users)


class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(self, seq_emb, attention_mask, padding_mask):
        normed = self.norm(seq_emb)
        attended, _ = self.attn(normed, normed, normed, attn_mask=attention_mask, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask[..., None], 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.3, maxlen=128, temperature=0.05, normalize_scores=True):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.temperature = float(temperature)
        self.normalize_scores = bool(normalize_scores)
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([HSTUBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        seq = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch_size, length = seq.shape
        padding = seq.eq(0)
        positions = torch.arange(1, length + 1, device=seq.device)[None, :].expand(batch_size, -1)
        positions = positions.masked_fill(padding, 0)
        hidden = self.item_emb(seq) * math.sqrt(self.hidden_units)
        hidden = self.emb_dropout(hidden + self.pos_emb(positions))
        hidden = hidden.masked_fill(padding[..., None], 0.0)
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
        allowed = causal[None, :, :] & (~padding[:, None, :])
        # Padded query rows attend only to themselves, avoiding all-masked attention.
        diagonal = torch.eye(length, dtype=torch.bool, device=seq.device)[None, :, :]
        allowed = allowed | (padding[:, :, None] & diagonal)
        attention_mask = (~allowed).repeat_interleave(self.num_heads, dim=0)
        for block in self.blocks:
            hidden = block(hidden, attention_mask, padding)
        return self.last_norm(hidden).masked_fill(padding[..., None], 0.0)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if self.normalize_scores:
            features = F.normalize(features, dim=-1, eps=1e-8)
            embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(-1, -2)) / self.temperature
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1) / self.temperature

    def readout(self, seq, features=None):
        if features is None:
            features = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device)[None, :].expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values
        selected = features[torch.arange(seq.shape[0], device=seq.device), last.clamp(min=0)]
        return selected * last.ge(0).unsqueeze(-1)


class ModelView:
    def __init__(self, model):
        self.model = model
        self.item_emb = model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def validation_cases(train, valid, itemnum, maxlen, num_users=200):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        needed = min(99, itemnum - len(excluded))
        negatives = []
        used = set(excluded)
        while len(negatives) < needed:
            candidate = rng.randint(1, itemnum)
            if candidate not in used:
                used.add(candidate)
                negatives.append(candidate)
        seq = np.zeros(maxlen, dtype=np.int64)
        history = train[user][-maxlen:]
        seq[-len(history):] = history
        cases.append((seq, [target] + negatives))
    return cases


@torch.no_grad()
def evaluate_cases(model, cases, device):
    model.eval()
    hits = 0.0
    ndcg = 0.0
    for start in range(0, len(cases), 64):
        batch = cases[start:start + 64]
        seq = torch.as_tensor(np.stack([case[0] for case in batch]), dtype=torch.long, device=device)
        features = model.readout(seq)
        for row, (_, candidates) in enumerate(batch):
            scores = model.predict_score(features[row], candidates)
            ranking = torch.argsort(scores, descending=True, stable=True)
            rank = int((ranking == 0).nonzero(as_tuple=False)[0, 0])
            if rank < 10:
                hits += 1.0
                ndcg += 1.0 / math.log2(rank + 2)
    return {'HR@10': hits / max(1, len(cases)), 'NDCG@10': ndcg / max(1, len(cases))}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen=128, num_users=200):
    cases = validation_cases(user_train, user_valid, itemnum, maxlen, num_users)
    return evaluate_cases(model, cases, device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything(42)
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
                  temperature=TEMPERATURE, normalize_scores=True)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = TrainingSampler(train, itemnum)
    _SAMPLER_CACHE[(id(train), itemnum)] = sampler
    sampler.prepare_forbidden(device)
    cases = validation_cases(train, valid, itemnum, MAXLEN, EVAL_USERS)
    if not cases:
        raise ValueError('No validation cases for ' + dataset_name)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_key = (-1.0, -1.0)
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _, user_rows = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN, return_users=True)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            user_rows = torch.as_tensor(user_rows, dtype=torch.long, device=device)
            valid_mask = pos.ne(0)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.log2feats(seq)
            selected = outputs[valid_mask]
            targets = pos[valid_mask]
            rows = user_rows[:, None].expand_as(pos)[valid_mask]
            # Accumulate the loss gradient at a detached feature boundary. This
            # frees each negative-embedding graph immediately and traverses the
            # shared HSTU backbone once per batch, with one optimizer step.
            feature_leaf = selected.detach().requires_grad_(True)
            count = targets.numel()
            batch_loss = 0.0
            for start in range(0, count, LOSS_CHUNK):
                end = min(start + LOSS_CHUNK, count)
                negatives = sampler.sample_negatives(rows[start:end], NUM_NEGATIVES)
                candidates = torch.cat((targets[start:end, None], negatives), dim=1)
                logits = model.predict_score(feature_leaf[start:end], candidates)
                labels = torch.zeros(end - start, dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, labels, reduction='sum') / count
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite sampled-softmax loss')
                loss.backward()
                batch_loss += float(loss.detach())
            selected.backward(feature_leaf.grad)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += batch_loss
        if epoch % 5 == 0:
            metrics = evaluate_cases(model, cases, device)
            key = (metrics['HR@10'], metrics['NDCG@10'])
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={key[0]:.5f} val_NDCG@10={key[1]:.5f}', flush=True)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
                with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as stream:
                    pickle.dump(dict(config=config, best_epoch=epoch, training_seed=42,
                                     num_negatives=NUM_NEGATIVES, scoring='normalized_dot_product'), stream)
            if epoch - best_epoch >= PATIENCE:
                break
    del _SAMPLER_CACHE[(id(train), itemnum)]
    del sampler, optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_key[0]


PREDICTOR = r'''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_model_class = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_loaded(dataset_name):
    global _model_class
    if dataset_name in _models:
        return
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('trained_hstu_architecture', os.path.join(_HERE, 'hstu_architecture.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.HSTURec
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
        meta = pickle.load(stream)
    model = _model_class(**meta['config']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    _ensure_loaded(dataset_name)
    model = _models[dataset_name]
    history = list(history)[-model.maxlen:]
    if any(int(item) < 0 or int(item) > model.itemnum for item in history):
        raise ValueError('History item outside dataset catalog')
    if any(int(item) <= 0 or int(item) > model.itemnum for item in candidates):
        raise ValueError('Candidate outside dataset catalog')
    seq = torch.zeros((1, model.maxlen), dtype=torch.long, device=_device)
    if history:
        seq[0, -len(history):] = torch.tensor(history, dtype=torch.long, device=_device)
    with torch.inference_mode():
        features = model.readout(seq)
        scores = model.predict_score(features, candidates)[0]
    return scores.cpu().tolist()
'''


def write_predict_script(working_dir):
    # Embed the actual model definitions so the predictor remains consistent
    # with training without importing training configuration or touching data.
    import inspect
    architecture = 'import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture += inspect.getsource(HSTUBlock) + '\n\n' + inspect.getsource(HSTURec) + '\n'
    with open(os.path.join(working_dir, 'hstu_architecture.py'), 'w') as stream:
        stream.write(architecture)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as stream:
        stream.write(PREDICTOR)


def main():
    seed_everything(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    print(f'HSTU normalized sampled softmax: device={device}, negatives={NUM_NEGATIVES}, temperature={TEMPERATURE}', flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f'Macro-average selected validation HR@10={sum(scores) / len(DATASETS):.6f}', flush=True)


if __name__ == '__main__':
    main()
