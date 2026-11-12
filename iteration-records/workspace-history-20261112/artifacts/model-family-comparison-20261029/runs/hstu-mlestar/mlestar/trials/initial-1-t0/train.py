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
NUM_EPOCHS = min(200, int(os.environ.get('SASREC_EPOCHS', '200')), int(os.environ.get('HSTU_EPOCHS', '200')))
LR = 0.001
WEIGHT_DECAY = 1e-6
PATIENCE = 20
EVAL_USERS = 200
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def _read_split(path):
    data = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            parts = line.split()
            if len(parts) == 2:
                user, item = map(int, parts)
                data[user].append(item)
    return dict(data)


def load_dataset(dataset_name, data_dir):
    # Consume the supplied seed-0 preprocessing without changing its splits.
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as stream:
            stats = json.load(stream)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    # Never open test files, even for an item-count fallback.
    observed = max((max(items, default=0) for split in (train, valid) for items in split.values()), default=0)
    itemnum = max(itemnum, observed)
    if itemnum < 2 or not train:
        raise ValueError('Missing training data for ' + dataset_name)
    return train, valid, itemnum


_SAMPLER_CACHE = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    if key not in _SAMPLER_CACHE:
        eligible = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        if not eligible:
            raise ValueError('No eligible training users')
        _SAMPLER_CACHE[key] = (eligible, {u: set(user_train[u]) for u in eligible})
    users, seen = _SAMPLER_CACHE[key]
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        inputs = history[:-1][-maxlen:]
        targets = history[1:][-maxlen:]
        start = maxlen - len(inputs)
        seq[row, start:] = inputs
        pos[row, start:] = targets
        for col in range(start, maxlen):
            item = random.randint(1, itemnum)
            while item in seen[user]:
                item = random.randint(1, itemnum)
            neg[row, col] = item
    return seq, pos, neg


class HSTUBlock(nn.Module):
    """HSTU sequential transduction with native SiLU attention and U gating."""

    def __init__(self, hidden_units, num_heads, dropout_rate, maxlen):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.maxlen = maxlen
        self.norm = nn.LayerNorm(hidden_units)
        self.uvqk = nn.Linear(hidden_units, 4 * hidden_units)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * maxlen - 1))
        self.attn_norm = nn.LayerNorm(hidden_units)
        self.output = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        offsets = torch.arange(maxlen)[:, None] - torch.arange(maxlen)[None, :]
        self.register_buffer('relative_index', offsets + maxlen - 1, persistent=False)
        self.register_buffer('causal', torch.ones(maxlen, maxlen, dtype=torch.bool).tril(), persistent=False)
        nn.init.xavier_uniform_(self.uvqk.weight)
        nn.init.zeros_(self.uvqk.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, seq_emb, valid):
        batch, length, hidden = seq_emb.shape
        u, v, q, k = F.silu(self.uvqk(self.norm(seq_emb))).chunk(4, dim=-1)
        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        bias = self.relative_bias[:, self.relative_index[:length, :length]]
        logits = torch.matmul(q, k.transpose(-1, -2)) + bias.unsqueeze(0)
        weights = F.silu(logits) / float(self.maxlen)
        allowed = self.causal[:length, :length][None, None, :, :]
        allowed = allowed & valid[:, None, :, None] & valid[:, None, None, :]
        weights = weights * allowed.to(weights.dtype)
        attended = torch.matmul(self.dropout(weights), v)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        update = self.output(u * self.attn_norm(attended))
        result = seq_emb + self.dropout(update)
        return result * valid.unsqueeze(-1).to(result.dtype)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
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
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        valid = sequence.ne(0)
        length = sequence.shape[1]
        position = torch.arange(1, length + 1, device=sequence.device).unsqueeze(0)
        position = position * valid.long()
        features = self.item_emb(sequence) * math.sqrt(self.hidden_units)
        features = self.emb_dropout(features + self.pos_emb(position))
        features = features * valid.unsqueeze(-1).to(features.dtype)
        for block in self.blocks:
            features = block(features, valid)
        return self.last_norm(features) * valid.unsqueeze(-1).to(features.dtype)

    def forward(self, sequence):
        return self.log2feats(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        else:
            items = items.to(device=features.device, dtype=torch.long)
        if features.ndim == 3:
            features = features[:, -1, :]
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(-1, -2))
        return (features.unsqueeze(-2) * embeddings).sum(-1)


class ModelView(nn.Module):
    """Diagnostic adapter exposing the HSTU feature and scoring interfaces."""

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

    def forward(self, sequence):
        return self.log2feats(sequence)


def make_validation(train, valid, itemnum, maxlen):
    # Fixed local validation candidates: 99 unique negatives and seed 42.
    # The host independently computes the authoritative benchmark score.
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > EVAL_USERS:
        users = rng.sample(users, EVAL_USERS)
    sequences, candidates = [], []
    for user in users:
        history = train[user][-maxlen:]
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        available = itemnum - len(excluded)
        count = min(99, available)
        negatives, used = [], set()
        if available <= 198:
            pool = [i for i in range(1, itemnum + 1) if i not in excluded]
            negatives = rng.sample(pool, count)
        else:
            while len(negatives) < count:
                item = rng.randint(1, itemnum)
                if item not in excluded and item not in used:
                    used.add(item)
                    negatives.append(item)
        sequences.append([0] * (maxlen - len(history)) + history)
        candidates.append([target] + negatives)
    return sequences, candidates


def evaluate_val(model, validation, device):
    sequences, candidates = validation
    if not sequences:
        return {'HR@10': 0.0, 'NDCG@10': 0.0}
    hits, ndcg = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), 64):
            seq = torch.tensor(sequences[start:start + 64], dtype=torch.long, device=device)
            features = model.log2feats(seq)[:, -1, :]
            for offset, items in enumerate(candidates[start:start + 64]):
                scores = model.predict_score(features[offset], items)
                rank = int((scores[1:] >= scores[0]).sum().item())
                if rank < 10:
                    hits += 1
                    ndcg += 1.0 / math.log2(rank + 2)
    return {'HR@10': hits / len(sequences), 'NDCG@10': ndcg / len(sequences)}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    ckpt_path = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            features = model.log2feats(seq)
            active = pos.ne(0)
            features = features[active]
            positives = model.item_emb(pos[active])
            negatives = model.item_emb(neg[active])
            margin = (features * (positives - negatives)).sum(-1)
            loss = F.softplus(-margin).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite HSTU loss in ' + dataset_name)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, validation, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if (hr, ndcg) > (best_hr, best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE * 5:
                break
    if best_epoch == 0:
        raise RuntimeError('No validation checkpoint was selected')
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as stream:
        pickle.dump(dict(config=config, family='hstu', architecture='native_silu_uvqk_v1',
                         best_epoch=best_epoch, validation_hr10=best_hr), stream)
    print(f'[{dataset_name}] selected epoch={best_epoch} HR@10={best_hr:.5f}', flush=True)
    _SAMPLER_CACHE.clear()
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained classes so inference cannot drift from training.
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    classes = inspect.getsource(HSTUBlock) + '\n\n' + inspect.getsource(HSTURec)
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
            meta = pickle.load(stream)
        if meta['family'] != 'hstu' or meta['architecture'] != 'native_silu_uvqk_v1':
            raise ValueError('Incompatible HSTU architecture')
        model = HSTURec(**meta['config']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset_name] = model
    return _models[dataset_name]


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model = _load(dataset_name)
    history = [int(i) if 0 < int(i) <= model.itemnum else 0 for i in history[-model.maxlen:]]
    sequence = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    ids = torch.tensor(candidates, dtype=torch.long, device=_device)
    supported = (ids > 0) & (ids <= model.itemnum)
    safe_ids = ids.clamp(0, model.itemnum)
    with torch.inference_mode():
        # Left-padded histories end at the final position.
        features = model.log2feats(seq)[:, -1, :]
        scores = model.predict_score(features, safe_ids).squeeze(0)
        scores = scores.masked_fill(~supported, -1e9)
    return scores.cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as stream:
        stream.write(header + classes + '\n' + runtime)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    print(f'HSTU native SiLU UVQK: device={device}, hidden=64, blocks=2, heads=2, dropout=0.2, epochs={NUM_EPOCHS}, batches=200', flush=True)
    scores = [train_one_dataset(name, device, DATA_DIR) for name in DATASETS]
    print('Local validation macro HR@10 = ' + str(sum(scores) / len(DATASETS)), flush=True)


if __name__ == '__main__':
    main()
