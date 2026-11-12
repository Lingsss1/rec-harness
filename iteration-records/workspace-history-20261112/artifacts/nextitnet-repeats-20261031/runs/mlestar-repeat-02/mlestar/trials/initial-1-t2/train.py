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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 32
_SAMPLERS = {}


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
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    if dataset_name in stats:
        itemnum = int(stats[dataset_name]['itemnum'])
    else:
        itemnum = max((i for split in (train, valid) for h in split.values() for i in h), default=0)
    return train, valid, itemnum


class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.histories = user_train
        self.itemnum = itemnum
        self.seen = {u: set(h) for u, h in user_train.items() if len(h) >= 2}
        self.users = [u for u, seen in self.seen.items() if itemnum - len(seen) > 0]
        if not self.users:
            raise RuntimeError('No training users with shifted targets and eligible negatives')
        self.complements = {}

    def negatives(self, user):
        seen = self.seen[user]
        available = self.itemnum - len(seen)
        k = min(NUM_NEGATIVES, available)
        if available <= 4 * NUM_NEGATIVES or available < self.itemnum // 4:
            if user not in self.complements:
                self.complements[user] = [i for i in range(1, self.itemnum + 1) if i not in seen]
            return random.sample(self.complements[user], k)
        chosen = set()
        result = []
        while len(result) < k:
            item = random.randint(1, self.itemnum)
            if item not in seen and item not in chosen:
                chosen.add(item)
                result.append(item)
        return result

    def batch(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, NUM_NEGATIVES), dtype=np.int64)
        for b in range(batch_size):
            user = random.choice(self.users)
            history = self.histories[user]
            inputs = history[:-1][-maxlen:]
            targets = history[1:][-maxlen:]
            seq[b, -len(inputs):] = inputs
            pos[b, -len(targets):] = targets
            sampled = self.negatives(user)
            neg[b, :len(sampled)] = sampled
        return seq, pos, neg


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = (id(user_train), itemnum)
    if key not in _SAMPLERS:
        _SAMPLERS[key] = TrainingSampler(user_train, itemnum)
    return _SAMPLERS[key].batch(batch_size, maxlen)


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
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        mask = (seq != 0).unsqueeze(-1)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * (seq != 0).long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if features.ndim == 2:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        out = self.encode_sequence(seq)
        positive = (out * self.item_emb(pos)).sum(-1)
        if neg.ndim == 2:
            negative = torch.bmm(out, self.item_emb(neg).transpose(1, 2))
        else:
            negative = (out.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        return positive, negative

    def last_features(self, seq):
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def score_candidates(self, seq, candidates):
        scores = self.predict_score(self.last_features(seq), candidates)
        return scores[0] if seq.size(0) == 1 else scores


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

    def forward(self, *args):
        return self.model(*args)


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    eligible = (neg != 0).unsqueeze(1) if neg.ndim == 2 else (neg != 0)
    log_w = F.log_softmax(negative.masked_fill(~eligible, float('-inf')), dim=-1)
    ranking_loss = -torch.logsumexp(log_w + F.logsigmoid(positive.unsqueeze(-1) - negative), dim=-1)
    regularization = 0.1 * (log_w.exp() * negative.square()).sum(-1)
    valid = pos != 0
    return (ranking_loss + regularization)[valid].mean()


def build_validation(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    records = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        count = min(99, itemnum - len(excluded - {0}))
        candidates = [target]
        if count < 99 or itemnum - len(excluded - {0}) < itemnum // 4:
            available = [i for i in range(1, itemnum + 1) if i not in excluded]
            candidates.extend(rng.sample(available, count))
        else:
            while len(candidates) < 100:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
        seq = np.zeros(maxlen, dtype=np.int64)
        recent = history[-maxlen:]
        if recent:
            seq[-len(recent):] = recent
        records.append((seq, candidates))
    return records


@torch.no_grad()
def evaluate_records(model, records, device):
    model.eval()
    hits = 0.0
    ndcg = 0.0
    for start in range(0, len(records), 128):
        chunk = records[start:start + 128]
        seq = torch.as_tensor(np.stack([r[0] for r in chunk]), dtype=torch.long, device=device)
        width = max(len(r[1]) for r in chunk)
        candidates = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        valid = torch.zeros_like(candidates, dtype=torch.bool)
        for b, (_, items) in enumerate(chunk):
            candidates[b, :len(items)] = torch.tensor(items, device=device)
            valid[b, :len(items)] = True
        scores = model.predict_score(model.last_features(seq), candidates).masked_fill(~valid, float('-inf'))
        ranks = (scores[:, 1:] > scores[:, :1]).sum(-1)
        hits += (ranks < 10).float().sum().item()
        ndcg += torch.where(ranks < 10, 1.0 / torch.log2(ranks.float() + 2), 0.0).sum().item()
    model.train()
    n = max(1, len(records))
    return {'HR@10': hits / n, 'NDCG@10': ndcg / n}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_records(model, build_validation(user_train, user_valid, itemnum, maxlen), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_all()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty training data for ' + dataset_name)
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS, kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    sampler = TrainingSampler(train, itemnum)
    records = build_validation(train, valid, itemnum, MAXLEN)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, dataset_name + '_model.pt')
    best = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            batch = sampler.batch(BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(x, dtype=torch.long, device=device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite BPR-max loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_records(model, records, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_score: {hr:.4f} NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best:
                best = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(working, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'[{dataset_name}] best validation HR@10={best:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best


def write_predict_script(working_dir):
    # Embed the exact trained architecture so inference is self-contained.
    import inspect
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    predictor = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loaded = {}
    for dataset in _DATASETS:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            config = pickle.load(f)
        model = NextItNetModel(**config).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        loaded[dataset] = model
    _models.update(loaded)


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if not candidates:
        return []
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    model = _models[dataset_name]
    sequence = [0] * model.maxlen
    recent = list(history[-model.maxlen:])
    if recent:
        sequence[-len(recent):] = recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(s) for s in scores.cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + predictor)


def main():
    torch.set_num_threads(4)
    seed_all()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
