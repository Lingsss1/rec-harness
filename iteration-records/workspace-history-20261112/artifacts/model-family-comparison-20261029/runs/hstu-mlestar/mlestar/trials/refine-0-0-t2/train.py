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
from torch import nn
from torch.nn import functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN = int(os.environ.get('HSTU_HIDDEN', '64'))
BLOCKS = int(os.environ.get('HSTU_BLOCKS', '2'))
HEADS = int(os.environ.get('HSTU_HEADS', '2'))
DROPOUT = float(os.environ.get('HSTU_DROPOUT', '0.3'))
MAXLEN = int(os.environ.get('HSTU_MAXLEN', '128'))
BATCH_SIZE = int(os.environ.get('HSTU_BATCH', '128'))
EPOCHS = min(200, int(os.environ.get('HSTU_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
BATCHES = 200
LR = float(os.environ.get('HSTU_LR', '0.001'))
WD = float(os.environ.get('HSTU_WD', '0.000001'))
K = 128
CALIBRATION = 0.75
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def read_split(path):
    result = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_dataset(name, directory):
    train = read_split(os.path.join(directory, name + '_train.txt'))
    valid = read_split(os.path.join(directory, name + '_valid.txt'))
    stats_path = os.path.join(directory, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as stream:
            stats = json.load(stream)
        if name in stats:
            itemnum = int(stats[name]['itemnum'])
    if not itemnum:
        itemnum = max((item for split in (train, valid) for history in split.values() for item in history), default=0)
    if itemnum < 2:
        raise ValueError('Insufficient item vocabulary for ' + name)
    return train, valid, itemnum


class BatchSampler:
    def __init__(self, train, itemnum):
        self.train = train
        self.itemnum = itemnum
        self.seen = {u: set(items) for u, items in train.items()}
        self.users = [u for u, items in train.items() if len(items) >= 2 and len(self.seen[u]) < itemnum]
        self.complements = {}
        if not self.users:
            raise ValueError('No eligible training histories')

    def negatives(self, user, count):
        seen = self.seen[user]
        if len(seen) > self.itemnum // 2:
            if user not in self.complements:
                self.complements[user] = np.asarray([i for i in range(1, self.itemnum + 1) if i not in seen], dtype=np.int64)
            return np.random.choice(self.complements[user], size=count, replace=True)
        result = np.random.randint(1, self.itemnum + 1, size=count, dtype=np.int64)
        bad = np.fromiter((int(i) in seen for i in result), dtype=bool, count=count)
        while bad.any():
            result[bad] = np.random.randint(1, self.itemnum + 1, size=int(bad.sum()))
            bad = np.fromiter((int(i) in seen for i in result), dtype=bool, count=count)
        return result

    def sample(self, batch_size, maxlen, num_negatives=None):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        # Default retains the diagnostic sampler's per-position negative shape.
        neg = np.zeros_like(seq) if num_negatives is None else np.empty((batch_size, num_negatives), dtype=np.int64)
        for row in range(batch_size):
            user = random.choice(self.users)
            history = self.train[user]
            length = min(len(history) - 1, maxlen)
            seq[row, -length:] = history[-length-1:-1]
            pos[row, -length:] = history[-length:]
            if num_negatives is None:
                neg[row, -length:] = self.negatives(user, length)
            else:
                neg[row] = self.negatives(user, num_negatives)
        return seq, pos, neg


_sampler_cache = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = (id(user_train), itemnum)
    if key not in _sampler_cache:
        _sampler_cache[key] = BatchSampler(user_train, itemnum)
    return _sampler_cache[key].sample(batch_size, maxlen)


class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(nn.Linear(hidden_units, hidden_units * 4), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_units * 4, hidden_units), nn.Dropout(dropout_rate))

    def forward(self, x, attention_mask, valid):
        z = self.norm(x)
        attended, _ = self.attn(z, z, z, attn_mask=attention_mask, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        x = x + self.dropout(value * F.silu(gate))
        x = x + self.ffn(self.ffn_norm(x))
        return x * valid


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2, dropout_rate=0.3, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('Hidden width must be divisible by heads')
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
        seq = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch, length = seq.shape
        valid_bool = seq.ne(0)
        valid = valid_bool.unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0) * valid_bool
        x = self.emb_dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)) * valid
        causal = torch.ones((length, length), dtype=torch.bool, device=seq.device).tril()
        allowed = causal.unsqueeze(0) & valid_bool.unsqueeze(1)
        # Padded queries attend only to themselves, preventing fully masked rows.
        diagonal = torch.eye(length, dtype=torch.bool, device=seq.device).unsqueeze(0)
        allowed = (allowed & valid_bool.unsqueeze(2)) | (diagonal & ~valid_bool.unsqueeze(2))
        attention_mask = (~allowed).repeat_interleave(self.num_heads, dim=0)
        for block in self.blocks:
            x = block(x, attention_mask, valid)
        return self.last_norm(x) * valid

    def forward(self, seq):
        return self.log2feats(seq)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if features.ndim == embeddings.ndim - 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def readout(self, sequence):
        feats = self.log2feats(sequence)
        indices = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0).expand_as(sequence)
        last = indices.masked_fill(sequence.eq(0), -1).amax(dim=1)
        result = feats[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return result * last.ge(0).unsqueeze(-1)


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

    def forward(self, sequence):
        return self.log2feats(sequence)


def validation_cases(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        count = min(99, itemnum - len(excluded))
        negatives = []
        chosen = set()
        if itemnum - len(excluded) < 198:
            negatives = rng.sample([i for i in range(1, itemnum + 1) if i not in excluded], count)
        else:
            while len(negatives) < count:
                item = rng.randint(1, itemnum)
                if item not in excluded and item not in chosen:
                    negatives.append(item)
                    chosen.add(item)
        seq = [0] * MAXLEN
        hist = train[user][-MAXLEN:]
        seq[-len(hist):] = hist
        cases.append((seq, [target] + negatives))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hits = 0
    ndcg = 0.0
    # Candidate counts can differ for tiny vocabularies.
    for start in range(0, len(cases), 64):
        group = cases[start:start + 64]
        seq = torch.tensor([c[0] for c in group], dtype=torch.long, device=device)
        features = model.readout(seq)
        for index, (_, candidates) in enumerate(group):
            scores = model.predict_score(features[index], candidates)
            ranking = torch.argsort(scores, descending=True, stable=True)
            rank = int((ranking == 0).nonzero(as_tuple=True)[0].item())
            if rank < 10:
                hits += 1
                ndcg += 1 / math.log2(rank + 2)
    return hits / max(1, len(cases)), ndcg / max(1, len(cases))


def gbce_loss(model, features, positives, negatives, beta):
    valid = positives.ne(0)
    positive_scores = model.predict_score(features, positives).float()
    log_partition = positive_scores
    negative_penalty = torch.zeros_like(positive_scores)
    # Reuse each history's negatives without a [batch, length, K, hidden] tensor.
    # Streaming logsumexp keeps the sampled-softmax denominator stable.
    for start in range(0, negatives.shape[1], 32):
        embeddings = model.item_emb(negatives[:, start:start + 32])
        scores = torch.bmm(features, embeddings.transpose(1, 2)).float()
        log_partition = torch.logaddexp(log_partition, torch.logsumexp(scores, dim=-1))
        negative_penalty = negative_penalty + F.softplus(scores).sum(-1)
    ranking_loss = log_partition - positive_scores
    calibrated_loss = (beta * F.softplus(-positive_scores) + negative_penalty) / (negatives.shape[1] + 1)
    losses = ranking_loss + 0.05 * calibrated_loss
    # The sampler draws users uniformly; give each sampled history equal weight.
    counts = valid.sum(dim=1)
    per_history = losses.masked_fill(~valid, 0.0).sum(dim=1) / counts.clamp_min(1)
    active = counts.gt(0).to(per_history.dtype)
    return (per_history * active).sum() / active.sum().clamp_min(1)


def train_one_dataset(name, device):
    seed_all()
    train, valid, itemnum = load_dataset(name, DATA_DIR)
    sampler = BatchSampler(train, itemnum)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS, num_heads=HEADS, dropout_rate=DROPOUT, maxlen=MAXLEN)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    beta = 1.0 - CALIBRATION + (K / (itemnum - 1)) * CALIBRATION
    cases = validation_cases(train, valid, itemnum)
    if not cases:
        raise ValueError('No validation users for ' + name)
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best = (-1.0, -1.0)
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for _ in range(BATCHES):
            seq, pos, neg = sampler.sample(BATCH_SIZE, MAXLEN, K)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            optimizer.zero_grad(set_to_none=True)
            features = model.log2feats(seq)
            loss = gbce_loss(model, features, pos, neg, beta)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite gBCE loss for ' + name)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.detach().item()
        if epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate_val(model, cases, device)
            print(f'[{name}] epoch={epoch} loss={running_loss/BATCHES:.6f} HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= 100:
                break
    with open(os.path.join(WORKING, name + '_meta.pkl'), 'wb') as stream:
        pickle.dump(config, stream)
    print(f'[{name}] best validation HR@10={best[0]:.4f} epoch={best_epoch}', flush=True)
    return best[0]


def write_predict_script(directory):
    # Embed the exact architecture used for training, independent of train.py imports.
    preamble = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'
    architecture = inspect.getsource(HSTUBlock) + '\n\n' + inspect.getsource(HSTURec)
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _load(name):
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _models:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as stream:
            config = pickle.load(stream)
        model = HSTURec(**config).to(_device)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[name] = model
    return _models[name]

@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model = _load(dataset_name)
    if not candidates:
        return []
    seq = [0] * model.maxlen
    history = [int(item) for item in history if 0 < int(item) <= model.itemnum][-model.maxlen:]
    if history:
        seq[-len(history):] = history
    sequence = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.as_tensor(candidates, dtype=torch.long, device=_device)
    valid = (items > 0) & (items <= model.itemnum)
    scores = model.predict_score(model.readout(sequence)[0], items.clamp(0, model.itemnum))
    return scores.masked_fill(~valid, -1e9).cpu().tolist()
'''
    with open(os.path.join(directory, 'predict.py'), 'w') as stream:
        stream.write(preamble + architecture + '\n' + runtime)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script(WORKING)
    scores = []
    for name in DATASETS:
        print('Training ' + name + ' with HSTU gBCE K=128 t=0.75', flush=True)
        scores.append(train_one_dataset(name, device))
    print('Macro validation HR@10 = %.6f' % (sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
