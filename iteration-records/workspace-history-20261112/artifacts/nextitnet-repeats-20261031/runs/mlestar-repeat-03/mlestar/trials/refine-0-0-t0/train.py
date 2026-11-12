from __future__ import annotations

import inspect
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 64
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 1]
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
SECONDS_PER_DATASET = float(os.environ.get('NEXTITNET_SECONDS_PER_DATASET', '1800'))
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


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
    if not itemnum:
        itemnum = max((max(items, default=0) for split in (train, valid) for items in split.values()), default=0)
    return train, valid, itemnum


_SAMPLER_CACHE = None


def sample_batch(user_train, itemnum, batch_size, maxlen):
    global _SAMPLER_CACHE
    if _SAMPLER_CACHE is None or _SAMPLER_CACHE[0] is not user_train:
        eligible = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        if not eligible:
            raise RuntimeError('No training histories with valid negative items')
        arrays = {u: np.asarray(user_train[u], dtype=np.int64) for u in eligible}
        seen = {u: np.unique(arrays[u]) for u in eligible}
        _SAMPLER_CACHE = (user_train, eligible, arrays, seen)
    _, eligible, arrays, seen = _SAMPLER_CACHE
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        u = random.choice(eligible)
        history = arrays[u]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length-1:-1]
        pos[row, -length:] = history[-length:]
        samples = np.random.randint(1, itemnum + 1, size=length)
        bad = np.isin(samples, seen[u])
        while bad.any():
            samples[bad] = np.random.randint(1, itemnum + 1, size=int(bad.sum()))
            bad = np.isin(samples, seen[u])
        neg[row, -length:] = samples
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

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self._causal_conv(self.conv1, self.norm1(x).transpose(1, 2)).transpose(1, 2)
        y = self.dropout(F.gelu(y))
        y = self._causal_conv(self.conv2, self.norm2(y).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128, dilations=None):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 1]
        if len(dilations) != num_blocks:
            raise ValueError('Dilation count must equal residual block count')
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, d, kernel_size, dropout_rate) for d in dilations
        ])
        self.input_norm = nn.LayerNorm(hidden_units)
        self.residual_scales = nn.Parameter(torch.full((num_blocks, hidden_units), 0.1))
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        mask = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(1, seq.size(1) + 1, device=seq.device).unsqueeze(0)
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(self.input_norm(x)) * mask
        for index, block in enumerate(self.blocks):
            residual = block(x) - x
            x = (x + self.residual_scales[index].view(1, 1, -1) * residual) * mask
        return self.norm(x) * mask

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if features.ndim == 1:
            return torch.matmul(embeddings, features)
        if features.ndim == 2 and embeddings.ndim == 2:
            return features @ embeddings.transpose(0, 1)
        if features.ndim == 2 and embeddings.ndim == 3:
            return (features.unsqueeze(1) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 and candidates.ndim == 1 else scores


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos != 0
    return F.softplus(negative[mask] - positive[mask]).mean()


def make_validation(train, valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | {target, 0}
        if itemnum - len(excluded - {0}) < 99:
            raise RuntimeError('Insufficient unique validation negatives')
        choices = [target]
        while len(choices) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                choices.append(item)
        seq = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            seq[-len(history):] = history
        sequences.append(seq)
        candidates.append(choices)
    if not sequences:
        raise RuntimeError('No validation users')
    return np.asarray(sequences), np.asarray(candidates)


@torch.no_grad()
def evaluate_val(model, validation, device):
    model.eval()
    sequences, candidates = validation
    ranks = []
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start+64], device=device)
        cand = torch.as_tensor(candidates[start:start+64], device=device)
        scores = model.score_candidates(seq, cand)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        ranks.append((order == 0).long().argmax(dim=-1).cpu())
    ranks = torch.cat(ranks)
    model.train()
    return {'HR@10': float((ranks < 10).float().mean()),
            'NDCG@10': float(torch.where(ranks < 10, 1.0 / torch.log2(ranks.float() + 2), 0.0).mean())}


def train_one_dataset(dataset_name, device):
    global _SAMPLER_CACHE
    seed_everything()
    _SAMPLER_CACHE = None
    train, valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    validation = make_validation(train, valid, itemnum)
    kwargs = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=8,
                  kernel_size=3, dropout_rate=0.2, maxlen=MAXLEN, dilations=DILATIONS)
    model = NextItNetModel(**kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    started = time.monotonic()
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            batch = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss for ' + dataset_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        timed_out = SECONDS_PER_DATASET > 0 and time.monotonic() - started >= SECONDS_PER_DATASET
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS or timed_out:
            metrics = evaluate_val(model, validation, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_score: {hr:.5f} NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if timed_out or epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(dict(model_kwargs=kwargs, best_epoch=best_epoch, validation_hr10=best_hr), f)
    print(f'[{dataset_name}] selected epoch={best_epoch} HR@10={best_hr:.5f}', flush=True)
    del optimizer, model
    _SAMPLER_CACHE = None
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    predictor = r'''
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
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = NextItNetModel(**meta['model_kwargs']).to(_device)
    checkpoint = os.path.join(_HERE, dataset_name + '_model.pt')
    model.load_state_dict(torch.load(checkpoint, map_location=_device, weights_only=True))
    model.eval()
    _models[dataset_name] = model
    return model


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(ds)
    sequence = [0] * model.maxlen
    recent = list(history[-model.maxlen:])
    if recent:
        sequence[-len(recent):] = recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = [train_one_dataset(ds, device) for ds in DATASETS]
    print(f'Macro validation HR@10 = {sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
