from __future__ import annotations
import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN_UNITS = 64
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT_RATE = 0.3
MAXLEN = 128
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('HSTU_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
LR = 1e-3
WEIGHT_DECAY = 1e-6
K = 128
CALIBRATION = 0.75
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            fields = line.split()
            if len(fields) == 2:
                user, item = map(int, fields)
                result[user].append(item)
    return dict(result)


def load_dataset(name, data_dir):
    train = _read_split(os.path.join(data_dir, name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as stream:
            stats = json.load(stream)
        itemnum = int(stats[name]['itemnum']) if name in stats else 0
    else:
        itemnum = 0
    if itemnum <= 0:
        itemnum = max((item for split in (train, valid) for items in split.values() for item in items), default=0)
    if itemnum <= 1 or not train:
        raise ValueError('Missing or invalid data for ' + name)
    return train, valid, itemnum


_sampler_cache = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    if key not in _sampler_cache:
        users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        forbidden = {u: np.asarray(list(set(user_train[u])), dtype=np.int64) for u in users}
        if not users:
            raise ValueError('No eligible training users')
        _sampler_cache[key] = (users, forbidden)
    users, forbidden = _sampler_cache[key]
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, K), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length-1:-1]
        pos[row, -length:] = history[-length:]
        samples = np.random.randint(1, itemnum + 1, size=(length, K), dtype=np.int64)
        rejected = np.isin(samples, forbidden[user])
        while rejected.any():
            samples[rejected] = np.random.randint(1, itemnum + 1, size=int(rejected.sum()))
            rejected = np.isin(samples, forbidden[user])
        neg[row, -length:] = samples
    return seq, pos, neg


class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(nn.Linear(hidden_units, hidden_units * 4), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_units * 4, hidden_units), nn.Dropout(dropout_rate))

    def forward(self, x, attention_mask, padding_mask):
        normalized = self.norm(x)
        attended, _ = self.attn(normalized, normalized, normalized, attn_mask=attention_mask, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        x = x + self.dropout(value * F.silu(gate))
        x = x + self.ffn(self.ffn_norm(x))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2, dropout_rate=0.3, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('Hidden dimension must be divisible by heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([HSTUBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        batch, length = sequence.shape
        padding = sequence.eq(0)
        positions = torch.arange(1, length + 1, device=sequence.device)[None, :].expand(batch, -1)
        positions = positions.masked_fill(padding, 0)
        x = self.emb_dropout(self.item_emb(sequence) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        x = x.masked_fill(padding.unsqueeze(-1), 0.0)
        # Every padded query may attend to itself, preventing fully masked rows.
        # Real queries see only real keys at the same or earlier position.
        causal = torch.ones((length, length), dtype=torch.bool, device=sequence.device).triu(1)
        forbidden = causal[None, :, :] | padding[:, None, :]
        diagonal = torch.eye(length, dtype=torch.bool, device=sequence.device)[None, :, :]
        forbidden = forbidden & ~(padding[:, :, None] & diagonal)
        attention_mask = forbidden.repeat_interleave(self.num_heads, dim=0)
        for block in self.blocks:
            x = block(x, attention_mask, padding)
        return self.last_norm(x).masked_fill(padding.unsqueeze(-1), 0.0)

    def forward(self, sequence):
        return self.log2feats(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        if features.ndim == 3:
            features = features[:, -1, :]
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)


class ModelView:
    def __init__(self, model):
        self.model = model
        self.item_emb = model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def gbce_loss(model, features, positives, negatives):
    alpha = K / (model.itemnum - 1)
    beta = 1.0 - CALIBRATION + alpha * CALIBRATION
    positive_scores = (features * model.item_emb(positives)).sum(-1)
    positive_loss = beta * F.softplus(-positive_scores).sum()

    def negative_sum(feature_chunk, negative_chunk):
        scores = (feature_chunk[:, None, :] * model.item_emb(negative_chunk)).sum(-1)
        return F.softplus(scores).sum()

    negative_loss = features.new_zeros(())
    for start in range(0, len(features), 256):
        stop = start + 256
        # Recompute large negative embedding tensors during backward.
        negative_loss = negative_loss + checkpoint(negative_sum, features[start:stop], negatives[start:stop], use_reentrant=False)
    return (positive_loss + negative_loss) / ((K + 1) * len(features))


def validation_batches(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        excluded = set(train[user]) | set(valid[user])
        available = itemnum - len(excluded)
        count = min(99, available)
        if count < 99:
            pool = [i for i in range(1, itemnum + 1) if i not in excluded]
            negatives = rng.sample(pool, count)
        else:
            negatives, seen = [], set()
            while len(negatives) < count:
                item = rng.randint(1, itemnum)
                if item not in excluded and item not in seen:
                    negatives.append(item)
                    seen.add(item)
        history = train[user][-maxlen:]
        sequences.append([0] * (maxlen - len(history)) + history)
        candidates.append([valid[user][0]] + negatives)
    return sequences, candidates


@torch.no_grad()
def evaluate_val(model, batches, device):
    sequences, candidates = batches
    model.eval()
    hits = ndcg = 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start+64], dtype=torch.long, device=device)
        features = model.log2feats(seq)[:, -1, :]
        for offset, row in enumerate(candidates[start:start+64]):
            scores = model.predict_score(features[offset], row)
            rank = int((scores[1:] >= scores[0]).sum().item())
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    denominator = max(1, len(sequences))
    return hits / denominator, ndcg / denominator


def train_one_dataset(name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(name, data_dir)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batches = validation_batches(train, valid, itemnum, MAXLEN)
    best = (-1.0, -1.0)
    best_epoch = 0
    ckpt_path = os.path.join(WORKING, name + '_model.pt')
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            active = pos.ne(0)
            negatives = torch.as_tensor(neg, device=device)[active]
            optimizer.zero_grad(set_to_none=True)
            features = model.log2feats(seq)[active]
            loss = gbce_loss(model, features, pos[active], negatives)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite gBCE loss: ' + name)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
                model.pos_emb.weight[0].zero_()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, batches, device)
            print(f'[{name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.6f} validation_HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= 100:
                break
    with open(os.path.join(WORKING, name + '_meta.pkl'), 'wb') as stream:
        pickle.dump(config, stream)
    _sampler_cache.pop(id(train), None)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best[0]


def write_predict_script(working_dir):
    script = '''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_class = None


def _load(name):
    global _model_class
    if name in _models:
        return _models[name], _metas[name]
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('hstu_gbce_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.HSTURec
    with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as stream:
        meta = pickle.load(stream)
    model = _model_class(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[name] = model
    _metas[name] = meta
    return model, meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    model, meta = _load(dataset_name)
    maxlen = meta['maxlen']
    history = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in history[-maxlen:]]
    sequence = [0] * (maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    legal = (items > 0) & (items <= meta['itemnum'])
    safe_items = items.clamp(0, meta['itemnum'])
    features = model.log2feats(seq)
    # Locate the final actual item, including histories with interior padding.
    positions = torch.arange(maxlen, device=_device)
    final = int(positions.masked_fill(seq[0].eq(0), -1).max().item())
    if final < 0:
        return [0.0 if bool(ok) else -1e9 for ok in legal.cpu().tolist()]
    scores = model.predict_score(features[0, final], safe_items)
    return scores.masked_fill(~legal, -1e9).cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as stream:
        stream.write(script)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script(WORKING)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'HSTU gBCE K={K} t={CALIBRATION} device={device}', flush=True)
    scores = []
    for name in DATASETS:
        scores.append(train_one_dataset(name, device, DATA_DIR))
    print('Macro-average validation HR@10: {:.4f}'.format(sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
