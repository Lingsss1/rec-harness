from __future__ import annotations

import importlib.util
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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN = int(os.environ.get('SASREC_HIDDEN', '64'))
BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT = float(os.environ.get('SASREC_DROPOUT', '0.5'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WD = float(os.environ.get('SASREC_WD', '0.0'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
BATCHES = int(os.environ.get('SASREC_BATCHES', '200'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
NEGATIVES = int(os.environ.get('GSASREC_NEGATIVES', '64'))
GBCE_T = float(os.environ.get('GSASREC_T', '0.75'))
NEG_CHUNK = int(os.environ.get('GSASREC_CHUNK', '512'))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    maximum = max((max(items) for split in (train, valid) for items in split.values() if items), default=0)
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        maximum = max(maximum, int(stats.get(name, {}).get('itemnum', 0)))
    return train, valid, maximum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        x = self.dropout2(self.conv2(F.relu(x)))
        return inputs + x.transpose(-1, -2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0.0, 1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def log2feats(self, seq):
        batch, length = seq.shape
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        x = self.emb_dropout(x + self.pos_emb(positions))
        keep = (seq != 0).unsqueeze(-1)
        x = x * keep
        mask = torch.ones(length, length, device=seq.device, dtype=torch.bool).triu(1)
        for norm, attn, fnorm, feed in zip(self.attention_layernorms, self.attention_layers,
                                           self.forward_layernorms, self.forward_layers):
            x = x.transpose(0, 1)
            q = norm(x)
            y, _ = attn(q, x, x, attn_mask=mask, need_weights=False)
            x = (q + y).transpose(0, 1)
            x = feed(fnorm(x)) * keep
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)

    def negative_loss(self, features, negatives):
        embeddings = self.item_emb(negatives)
        logits = (features.unsqueeze(1) * embeddings).sum(-1)
        return F.softplus(logits.float()).sum()

    def training_loss(self, seq, positive, negatives, t):
        valid = positive != 0
        features = self.log2feats(seq)[valid]
        positive_logits = (features * self.item_emb(positive[valid])).sum(-1).float()
        k = negatives.shape[1]
        beta = 1.0 - t + (k / max(self.itemnum - 1, 1)) * t
        total = beta * F.softplus(-positive_logits).sum()
        for start in range(0, features.shape[0], NEG_CHUNK):
            total = total + checkpoint(self.negative_loss,
                                       features[start:start + NEG_CHUNK],
                                       negatives[start:start + NEG_CHUNK],
                                       use_reentrant=False)
        return total / (features.shape[0] * (k + 1))


class Sampler:
    def __init__(self, train, itemnum, maxlen):
        self.itemnum = itemnum
        self.users = [u for u, h in train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        if not self.users:
            raise ValueError('No users have both training transitions and available negative items')
        self.seq = np.zeros((len(self.users), maxlen), dtype=np.int32)
        self.pos = np.zeros_like(self.seq)
        self.seen = []
        self.counts = []
        for row, u in enumerate(self.users):
            history = train[u]
            n = min(maxlen, len(history) - 1)
            self.seq[row, -n:] = history[-n-1:-1]
            self.pos[row, -n:] = history[-n:]
            self.seen.append(np.asarray(sorted(set(history)), dtype=np.int64))
            self.counts.append(n)

    def sample(self):
        rows = np.random.randint(0, len(self.users), size=BATCH)
        negative_parts = []
        for row in rows:
            shape = (self.counts[row], NEGATIVES)
            neg = np.random.randint(1, self.itemnum + 1, size=shape)
            blocked = np.isin(neg, self.seen[row])
            while blocked.any():
                neg[blocked] = np.random.randint(1, self.itemnum + 1, size=int(blocked.sum()))
                blocked = np.isin(neg, self.seen[row])
            negative_parts.append(neg)
        return self.seq[rows], self.pos[rows], np.concatenate(negative_parts, axis=0)


def validation_cases(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        seq = np.zeros(MAXLEN, dtype=np.int64)
        hist = train[u][-MAXLEN:]
        if hist:
            seq[-len(hist):] = hist
        interacted = set(train[u]) | set(valid[u])
        if itemnum - len(interacted) < 99:
            raise ValueError('Pinned validation protocol requires 99 available unique negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in interacted:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[u][0]] + list(negatives))
    if not sequences:
        raise ValueError('No validation users')
    return np.stack(sequences), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, cases, device):
    model.eval()
    seqs, candidates = cases
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(seqs), 32):
        seq = torch.as_tensor(seqs[start:start+32], dtype=torch.long, device=device)
        cand = torch.as_tensor(candidates[start:start+32], dtype=torch.long, device=device)
        scores = model.predict(seq, cand)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
        hit = ranks < 10
        hits += hit.float().sum().item()
        ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
    model.train()
    return hits / len(seqs), ndcg / len(seqs)


def train_one_dataset(name, device):
    train, valid, itemnum = load_dataset(name)
    print(f'[{name}] users={len(train)} items={itemnum}', flush=True)
    sampler = Sampler(train, itemnum, MAXLEN)
    cases = validation_cases(train, valid, itemnum)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS,
                num_heads=HEADS, dropout_rate=DROPOUT, maxlen=MAXLEN)
    model = SASRec(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD)
    path = os.path.join(WORKING, name + '_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for _ in range(BATCHES):
            seq, pos, neg = sampler.sample()
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(seq, pos, neg, GBCE_T)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss in {name}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach()
        if epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, cases, device)
            print(f'[{name}] epoch={epoch} loss={loss_sum.item()/BATCHES:.6f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), path)
            if epoch - best_epoch >= PATIENCE * 5:
                break
    if best_hr < 0:
        best_hr, _ = evaluate(model, cases, device)
        torch.save(model.state_dict(), path)
    with open(os.path.join(WORKING, name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script():
    script = '''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None

def _load(dataset):
    global _device, _model_class
    if dataset in _models:
        return
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('gsasrec_trial_model', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.SASRec
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _model_class(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta

def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    _load(dataset)
    meta = _metas[dataset]
    maxlen = meta['maxlen']
    hist = list(history)[-maxlen:]
    seq = [0] * (maxlen - len(hist)) + hist
    with torch.no_grad():
        sequence = torch.tensor([seq], dtype=torch.long, device=_device)
        items = torch.tensor([list(candidates)], dtype=torch.long, device=_device)
        return _models[dataset].predict(sequence, items)[0].float().cpu().tolist()
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    if NEGATIVES < 1 or NEG_CHUNK < 1 or EPOCHS < 1 or BATCHES < 1:
        raise ValueError('Training counts must be positive')
    seed_all()
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device={device}; negatives={NEGATIVES}; gBCE_t={GBCE_T}; epochs={EPOCHS}', flush=True)
    write_predict_script()
    scores = {}
    for name in DATASETS:
        scores[name] = train_one_dataset(name, device)
    macro = sum(scores.values()) / len(DATASETS)
    with open(os.path.join(WORKING, 'validation_summary.json'), 'w') as f:
        json.dump({'per_dataset_validation_HR@10': scores, 'macro_validation_HR@10': macro}, f, indent=2)
    print(f'Macro validation HR@10={macro:.6f}', flush=True)


if __name__ == '__main__':
    main()
