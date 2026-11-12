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
DATASETS = DEFAULT_DATASETS[:]
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = float(os.environ.get('NEXTITNET_DROPOUT', '0.2'))
LR = float(os.environ.get('NEXTITNET_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('NEXTITNET_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('NEXTITNET_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
CL_WEIGHT = 0.0
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    if dataset_name in stats:
        itemnum = int(stats[dataset_name]['itemnum'])
    else:
        itemnum = max((i for split in (train, valid) for items in split.values() for i in items), default=0)
    return train, valid, itemnum


_SAMPLER_USERS = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    cached = _SAMPLER_USERS.get(key)
    if cached is None or cached[0] is not user_train:
        users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        _SAMPLER_USERS[key] = (user_train, users)
    else:
        users = cached[1]
    if not users:
        raise RuntimeError('No eligible training histories')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for b in range(batch_size):
        history = user_train[random.choice(users)]
        interacted = set(history)
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[b, idx] = item
            pos[b, idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            neg[b, idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
    return seq, pos, neg


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self._causal_conv(self.conv1, self.norm1(x).transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self._causal_conv(self.conv2, self.norm2(y).transpose(1, 2)).transpose(1, 2)
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
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate) for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        seq = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        if seq.ndim == 1:
            seq = seq.unsqueeze(0)
        batch, length = seq.shape
        mask = (seq != 0).unsqueeze(-1)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * (seq != 0).long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def encode_sequence(self, seq):
        return self.log2feats(seq)

    def final_features(self, features, seq):
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        emb = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, emb.transpose(0, 1))
        if features.ndim == emb.ndim - 1:
            return (features.unsqueeze(-2) * emb).sum(dim=-1)
        return (features * emb).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return (features * self.item_emb(pos)).sum(-1), (features * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        features = self.final_features(self.log2feats(seq), seq)
        return self.predict_score(features, candidates)[0]


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

    def forward(self, seq, pos, neg):
        return self.model(seq, pos, neg)


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    return F.softplus(negative - positive)[pos != 0].mean()


def contrastive_loss(model, seq, pos, first_features):
    second_features = model.log2feats(seq)
    first = F.normalize(model.final_features(first_features, seq), dim=-1)
    second = F.normalize(model.final_features(second_features, seq), dim=-1)
    z = torch.cat((first, second), dim=0)
    batch = seq.size(0)
    logits = z.float() @ z.float().T / 0.2
    _, history_ids = torch.unique(seq, dim=0, return_inverse=True)
    positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
    last = positions.masked_fill(seq == 0, -1).max(1).values.clamp(min=0)
    targets = pos[torch.arange(batch, device=seq.device), last]
    false_negative = (history_ids[:, None] == history_ids[None, :]) | (targets[:, None] == targets[None, :])
    excluded = false_negative.repeat(2, 2)
    rows = torch.arange(2 * batch, device=seq.device)
    paired = (rows + batch) % (2 * batch)
    excluded[rows, paired] = False
    excluded.fill_diagonal_(True)
    return F.cross_entropy(logits.masked_fill(excluded, float('-inf')), paired)


def training_loss(model, seq, pos, neg, cl_weight=0.0):
    features = model.log2feats(seq)
    positive = (features * model.item_emb(pos)).sum(-1)
    negative = (features * model.item_emb(neg)).sum(-1)
    bpr = F.softplus(negative - positive)[pos != 0].mean()
    cl = contrastive_loss(model, seq, pos, features) if cl_weight > 0 else bpr.new_zeros(())
    return bpr + cl_weight * cl, bpr.detach(), cl.detach()


def make_validation_cases(train, valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | {target, 0}
        candidates = [target]
        required = min(99, itemnum - len(excluded - {0}))
        while len(candidates) < 1 + required:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            sequence[-len(history):] = history
        cases.append((sequence, candidates))
    return cases


def evaluate_val(model, cases, device):
    model.eval()
    hits = 0
    ndcg = 0.0
    with torch.inference_mode():
        for sequence, candidates in cases:
            seq = torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().numpy()
            rank = int(np.where(np.argsort(-scores, kind='stable') == 0)[0][0])
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    model.train()
    return hits / max(1, len(cases)), ndcg / max(1, len(cases))


def train_one_dataset(dataset_name, device, data_dir):
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        cl_weight = CL_WEIGHT * min(1.0, max(0.0, (epoch - 10) / 10.0))
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss, bpr, cl = training_loss(model, seq, pos, neg, cl_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [loss.item(), bpr.item(), cl.item()]
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, cases, device)
            means = totals / NUM_BATCHES
            print(f'[{dataset_name}] epoch={epoch} loss={means[0]:.4f} bpr={means[1]:.4f} contrastive={means[2]:.4f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best:
                best = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    _SAMPLER_USERS.pop(id(train), None)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset):
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if dataset not in _models:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            config = pickle.load(f)
        model = NextItNetModel(**config).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = config
    return _models[dataset], _metas[dataset]


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model, meta = _load(dataset)
    if len(candidates) == 0:
        return []
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    recent = list(history[-maxlen:])
    if recent:
        sequence[-len(recent):] = recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(s) for s in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
