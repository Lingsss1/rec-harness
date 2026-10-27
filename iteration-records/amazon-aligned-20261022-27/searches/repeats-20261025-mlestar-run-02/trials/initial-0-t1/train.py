from __future__ import annotations

import json
import math
import os
import random
import inspect
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN = int(os.environ.get('SASREC_HIDDEN', '64'))
BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT = float(os.environ.get('SASREC_DROPOUT', '0.3'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WD = float(os.environ.get('SASREC_WD', '0.0'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
STEPS = int(os.environ.get('SASREC_BATCHES', '200'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
TEMPERATURE = float(os.environ.get('DUOREC_TEMPERATURE', '0.2'))
DROP_WEIGHT = float(os.environ.get('DUOREC_DROPOUT_WEIGHT', '0.1'))
SEM_WEIGHT = float(os.environ.get('DUOREC_SEMANTIC_WEIGHT', '0.1'))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as stream:
            stats = json.load(stream)
        if name in stats:
            itemnum = int(stats[name]['itemnum'])
    if not itemnum:
        itemnum = max((max(items) for split in (train, valid) for items in split.values() if items), default=0)
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout1(self.conv1(inputs.transpose(1, 2)))
        outputs = F.relu(outputs)
        outputs = self.dropout2(self.conv2(outputs)).transpose(1, 2)
        return inputs + outputs


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.3, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0.0, 1.0)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1), persistent=False)

    def log2feats(self, seq):
        length = seq.size(1)
        positions = torch.arange(self.maxlen - length, self.maxlen, device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions).unsqueeze(0))
        active = seq.ne(0).unsqueeze(-1)
        x = x * active
        for norm1, attention, norm2, ffn in zip(self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers):
            queries = norm1(x)
            out, _ = attention(queries, x, x, attn_mask=self.causal_mask[:length, :length], need_weights=False)
            x = queries + out
            x = ffn(norm2(x)) * active
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


class PrefixSampler:
    def __init__(self, train, itemnum, maxlen, seed=42):
        self.histories = [np.asarray(items, dtype=np.int64) for items in train.values() if len(items) >= 2]
        if not self.histories:
            raise ValueError('No training sequences with at least two items')
        self.seen = [np.unique(h) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.rng = np.random.default_rng(seed)
        counts = np.asarray([len(h) - 1 for h in self.histories], dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(counts)))
        self.targets = np.concatenate([h[1:] for h in self.histories])
        self.order = np.argsort(self.targets, kind='stable')
        frequencies = np.bincount(self.targets, minlength=itemnum + 1)
        self.offsets = np.concatenate(([0], np.cumsum(frequencies)))

    def decode(self, example):
        user = int(np.searchsorted(self.starts, example, side='right') - 1)
        t = int(example - self.starts[user]) + 1
        return user, t

    def draw(self, size):
        ids = self.rng.integers(0, len(self.targets), size=size)
        labels = self.targets[ids]
        semantic_ids = ids.copy()
        available = np.zeros(size, dtype=bool)
        for row, (example, target) in enumerate(zip(ids, labels)):
            start, end = int(self.offsets[target]), int(self.offsets[target + 1])
            if end - start > 1:
                alternative = int(self.order[self.rng.integers(start, end)])
                while alternative == example:
                    alternative = int(self.order[self.rng.integers(start, end)])
                semantic_ids[row] = alternative
                available[row] = True
        decoded = [self.decode(int(e)) for e in ids]
        sem_decoded = [self.decode(int(semantic_ids[i])) for i in np.flatnonzero(available)]
        length = min(self.maxlen, max(t for _, t in decoded + sem_decoded))
        seq = np.zeros((size, length), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        for row, (user, t) in enumerate(decoded):
            history = self.histories[user]
            n = min(length, t)
            seq[row, -n:] = history[t - n:t]
            pos[row, -n:] = history[t - n + 1:t + 1]
            sampled = self.rng.integers(1, self.itemnum + 1, size=n)
            if len(self.seen[user]) >= self.itemnum:
                raise ValueError('No unobserved training negative for a user')
            bad = np.isin(sampled, self.seen[user])
            while bad.any():
                sampled[bad] = self.rng.integers(1, self.itemnum + 1, size=int(bad.sum()))
                bad = np.isin(sampled, self.seen[user])
            neg[row, -n:] = sampled
        semantic = np.zeros((len(sem_decoded), length), dtype=np.int64)
        for row, (user, t) in enumerate(sem_decoded):
            n = min(length, t)
            semantic[row, -n:] = self.histories[user][t - n:t]
        return seq, pos, neg, labels, semantic, available


def symmetric_infonce(first, second, targets):
    n = first.size(0)
    if n < 2:
        return (first.sum() + second.sum()) * 0.0
    views = F.normalize(torch.cat((first, second), dim=0).float(), dim=-1)
    logits = views @ views.t() / TEMPERATURE
    row = torch.arange(2 * n, device=views.device)
    paired = (row + n) % (2 * n)
    labels = torch.cat((targets, targets))
    excluded = labels[:, None].eq(labels[None, :])
    excluded[row, paired] = False
    logits = logits.masked_fill(excluded, float('-inf'))
    return F.cross_entropy(logits, paired)


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        hist = train[user][-maxlen:]
        seq = [0] * (maxlen - len(hist)) + hist
        interacted = set(train[user]) | set(valid[user])
        needed = min(99, itemnum - len(interacted))
        if needed < 0:
            raise ValueError('Invalid item vocabulary')
        negatives = set()
        while len(negatives) < needed:
            item = rng.randint(1, itemnum)
            if item not in interacted:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    return sequences, candidates


@torch.no_grad()
def evaluate(model, validation, device):
    sequences, candidates = validation
    model.eval()
    hits = 0.0
    ndcg = 0.0
    # Usually every row has exactly 100 candidates; grouping also handles tiny vocabularies.
    groups = defaultdict(list)
    for index, row in enumerate(candidates):
        groups[len(row)].append(index)
    for indices in groups.values():
        for start in range(0, len(indices), 128):
            chosen = indices[start:start + 128]
            seq = torch.tensor([sequences[i] for i in chosen], dtype=torch.long, device=device)
            cand = torch.tensor([candidates[i] for i in chosen], dtype=torch.long, device=device)
            scores = model.predict(seq, cand)
            ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
            hit = ranks < 10
            hits += hit.sum().item()
            ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
    denominator = max(1, len(sequences))
    return hits / denominator, ndcg / denominator


def train_one(name, device):
    seed_all(42)
    train, valid, itemnum = load_dataset(name)
    sampler = PrefixSampler(train, itemnum, MAXLEN, 42)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS,
                  num_heads=HEADS, dropout_rate=DROPOUT, maxlen=MAXLEN)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    print(f'[{name}] users={len(train)} items={itemnum} training_prefixes={len(sampler.targets)}', flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = np.zeros(3, dtype=np.float64)
        for _ in range(STEPS):
            seq, pos, neg, targets, semantic, available = sampler.draw(BATCH)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            targets = torch.as_tensor(targets, device=device)
            eligible = torch.as_tensor(available, device=device)
            optimizer.zero_grad(set_to_none=True)
            feats = model.log2feats(seq)
            active = pos.ne(0)
            active_feats = feats[active]
            pos_logits = (active_feats * model.item_emb(pos[active])).sum(-1)
            neg_logits = (active_feats * model.item_emb(neg[active])).sum(-1)
            recommendation = F.softplus(-pos_logits).mean() + F.softplus(neg_logits).mean()
            dropout_loss = feats.sum() * 0.0
            semantic_loss = feats.sum() * 0.0
            if DROP_WEIGHT:
                second = model.log2feats(seq)[:, -1]
                dropout_loss = symmetric_infonce(feats[:, -1], second, targets)
            if SEM_WEIGHT and int(available.sum()) >= 2:
                semantic = torch.as_tensor(semantic, device=device)
                sem_features = model.log2feats(semantic)[:, -1]
                semantic_loss = symmetric_infonce(feats[eligible, -1], sem_features, targets[eligible])
            loss = recommendation + DROP_WEIGHT * dropout_loss + SEM_WEIGHT * semantic_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses += [recommendation.detach().item(), dropout_loss.detach().item(), semantic_loss.detach().item()]
        if epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} rec={losses[0]/STEPS:.4f} dropout_nce={losses[1]/STEPS:.4f} semantic_nce={losses[2]/STEPS:.4f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save({'config': config, 'state_dict': model.state_dict()}, checkpoint)
            if epoch - best_epoch >= PATIENCE * 5:
                break
    if not os.path.isfile(checkpoint):
        torch.save({'config': config, 'state_dict': model.state_dict()}, checkpoint)
    print(f'[{name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script():
    # Export the actual encoder classes so inference cannot silently use an older architecture.
    model_source = 'import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    model_source += inspect.getsource(PointWiseFeedForward) + '\n\n' + inspect.getsource(SASRec)
    with open(os.path.join(WORKING, 'model_def.py'), 'w') as stream:
        stream.write(model_source)
    predictor = '''from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_SPEC = importlib.util.spec_from_file_location('duorec_exported_encoder', os.path.join(_HERE, 'model_def.py'))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load(dataset):
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if dataset not in _MODELS:
        checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location='cpu', weights_only=True)
        model = _MODULE.SASRec(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(_DEVICE).eval()
        _MODELS[dataset] = model
    return _MODELS[dataset]


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(dataset)
    hist = [int(item) for item in history[-model.maxlen:]]
    seq = [0] * (model.maxlen - len(hist)) + hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    with torch.inference_mode():
        scores = model.predict(seq_t, cand_t)[0]
    return scores.cpu().tolist()
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as stream:
        stream.write(predictor)


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_predict_script()
    scores = []
    for name in DATASETS:
        scores.append(train_one(name, device))
    print(f'Macro-average validation HR@10={sum(scores)/len(scores):.6f}', flush=True)
    with open(os.path.join(WORKING, 'validation_summary.json'), 'w') as stream:
        json.dump({'validation_HR@10': dict(zip(DATASETS, scores)), 'macro_validation_HR@10': sum(scores)/len(scores)}, stream, indent=2)


if __name__ == '__main__':
    main()
