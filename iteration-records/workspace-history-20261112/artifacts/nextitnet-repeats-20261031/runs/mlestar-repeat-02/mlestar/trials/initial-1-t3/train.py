from __future__ import annotations
import os
import math
import json
import pickle
import random
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
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
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
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
        itemnum = max((i for split in (train, valid) for items in split.values() for i in items), default=0)
    return train, valid, itemnum


class TrainingHistories(dict):
    def __init__(self, histories, itemnum):
        super().__init__(histories)
        self.users = [u for u, h in self.items() if len(h) >= 2]
        self.seen = {u: set(self[u]) for u in self.users}
        self.users = [u for u in self.users if len(self.seen[u]) < itemnum]


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = getattr(user_train, 'users', None)
    if users is None:
        users = [u for u, h in user_train.items() if len(h) >= 2 and len(set(h)) < itemnum]
    if not users:
        raise RuntimeError('No eligible training histories')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    seen_cache = getattr(user_train, 'seen', {})
    for b in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        seen = seen_cache.get(u)
        if seen is None:
            seen = set(history)
        nxt = history[-1]
        index = maxlen - 1
        for item in reversed(history[:-1]):
            seq[b, index] = item
            pos[b, index] = nxt
            negative = random.randint(1, itemnum)
            while negative in seen:
                negative = random.randint(1, itemnum)
            neg[b, index] = negative
            nxt = item
            index -= 1
            if index < 0:
                break
    return seq, pos, neg


def last_positions(seq):
    positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
    return positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp_min(0)


def final_features(features, seq):
    return features[torch.arange(seq.shape[0], device=seq.device), last_positions(seq)]


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
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
                                     for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        mask = seq.ne(0)
        positions = torch.arange(1, seq.shape[1] + 1, device=seq.device).unsqueeze(0)
        positions = positions * mask.long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = final_features(self.encode_sequence(seq), seq)
        embeddings = self.item_emb(candidates)
        if candidates.ndim == 1:
            return (features @ embeddings.T).squeeze(0)
        return (features.unsqueeze(1) * embeddings).sum(dim=-1)


def contrastive_loss(first, second, seq, pos):
    batch = seq.shape[0]
    rows = torch.arange(batch, device=seq.device)
    last = last_positions(seq)
    target = pos[rows, last]
    a = F.normalize(first[rows, last].float(), dim=-1)
    b = F.normalize(second[rows, last].float(), dim=-1)
    z = torch.cat([a, b], dim=0)
    logits = z @ z.T / 0.2
    # Equal visible histories and equal final shifted targets are false negatives.
    _, history_ids = torch.unique(seq, dim=0, return_inverse=True)
    duplicate = history_ids[:, None].eq(history_ids[None, :])
    same_target = target[:, None].eq(target[None, :])
    excluded = (duplicate | same_target).repeat(2, 2)
    index = torch.arange(2 * batch, device=seq.device)
    paired = (index + batch) % (2 * batch)
    excluded[index, paired] = False
    excluded[index, index] = True
    logits = logits.masked_fill(excluded, float('-inf'))
    return F.cross_entropy(logits, paired)


def bpr_sequence_loss(model, seq, pos, neg):
    first = model.log2feats(seq)
    positive = model.predict_score(first, pos)
    negative = model.predict_score(first, neg)
    mask = pos.ne(0)
    recommendation = F.softplus(negative - positive)[mask].mean()
    second = model.log2feats(seq)
    regularizer = contrastive_loss(first, second, seq, pos)
    return recommendation + 0.05 * regularizer


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq, device=device).unsqueeze(0)


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for u in users:
        target = valid[u][0]
        excluded = set(train[u]) | {0, target}
        candidates = [target]
        count = min(99, itemnum - len(excluded - {0}))
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        examples.append((train[u], candidates))
    return examples


@torch.no_grad()
def evaluate_examples(model, examples, device, maxlen):
    model.eval()
    hits = 0
    ndcg = 0.0
    for history, candidates in examples:
        seq = _history_to_tensor(history, maxlen, device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        if rank < 10:
            hits += 1
            ndcg += 1 / math.log2(rank + 2)
    n = max(1, len(examples))
    model.train()
    return {'HR@10': hits / n, 'NDCG@10': ndcg / n}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_examples(model, make_validation(user_train, user_valid, itemnum, maxlen), device, maxlen)


def train_one_dataset(dataset_name, device, data_dir):
    seed_all()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    train = TrainingHistories(train, itemnum)
    examples = make_validation(train, valid, itemnum, MAXLEN)
    if not examples:
        raise RuntimeError('No validation users: ' + dataset_name)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint = os.path.join(WORKING, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        for _ in range(NUM_BATCHES):
            batch = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_examples(model, examples, device, MAXLEN)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total / NUM_BATCHES:.5f} val_HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    script = r'''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_Model = None


def _ensure_loaded(dataset):
    global _device, _Model
    if dataset in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        spec = importlib.util.spec_from_file_location('nextitnet_trial_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _Model = module.NextItNetModel
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _Model(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    _ensure_loaded(dataset)
    meta = _metas[dataset]
    maxlen = meta['maxlen']
    seq = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = _models[dataset].score_candidates(seq_t, items)
    return [float(x) for x in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script(WORKING)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print('Macro validation HR@10 = {:.6f}'.format(sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
