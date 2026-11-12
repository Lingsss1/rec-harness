"""Per-category NextItNet with 64-negative sampled categorical cross-entropy."""
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

DEFAULT_DATASETS = [
    'Movies_and_TV',
    'Industrial_and_Scientific',
    'Electronics',
    'CDs_and_Vinyl',
]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = 0.2
LR = 1e-3
BATCH_SIZE = 128
NUM_NEGATIVES = 64
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get(
    'NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
_SAMPLERS = {}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_split(path):
    result = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as handle:
            stats = json.load(handle)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    maximum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 2:
                        maximum = max(maximum, int(parts[1]))
    return maximum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, history in user_train.items() if len(history) >= 2]
        if not self.users:
            raise RuntimeError('No users with at least two training interactions')
        self.seen = {u: set(user_train[u]) | {0} for u in self.users}
        for u in self.users:
            available = itemnum - sum(1 <= i <= itemnum for i in self.seen[u])
            if available < NUM_NEGATIVES:
                raise RuntimeError('Cannot draw 64 distinct unseen training negatives')

    def sample(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.empty((batch_size, NUM_NEGATIVES), dtype=np.int64)
        for row in range(batch_size):
            user = random.choice(self.users)
            history = self.user_train[user]
            length = min(maxlen, len(history) - 1)
            seq[row, -length:] = history[-length - 1:-1]
            pos[row, -length:] = history[-length:]
            # Rejection sampling without replacement gives a uniform subset
            # of the complement of the complete training history.
            selected = set()
            forbidden = self.seen[user]
            count = 0
            while count < NUM_NEGATIVES:
                item = random.randint(1, self.itemnum)
                if item not in forbidden and item not in selected:
                    selected.add(item)
                    neg[row, count] = item
                    count += 1
        return seq, pos, neg


def sample_batch(user_train, itemnum, batch_size, maxlen):
    """Return left-padded sequences/targets and one B x 64 negative set."""
    key = (id(user_train), itemnum)
    sampler = _SAMPLERS.get(key)
    if sampler is None or sampler.user_train is not user_train:
        sampler = TrainingSampler(user_train, itemnum)
        _SAMPLERS[key] = sampler
    return sampler.sample(batch_size, maxlen)


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size,
                               dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size,
                               dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(nn.functional.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        residual = x
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        y = self.dropout(y)
        return residual + y


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8,
                 kernel_size=3, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
            for i in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        mask = seq.ne(0)
        positions = positions * mask.long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        return (features * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        positive_scores = self.predict_score(features, pos)
        # Training negatives are shared across each user's sequence positions.
        negative_scores = torch.bmm(features, self.item_emb(neg).transpose(1, 2))
        return positive_scores, negative_scores

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        features = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView:
    """Expose the NextItNet diagnostic feature and scoring interfaces."""
    def __init__(self, model):
        self.model = model

    @property
    def item_emb(self):
        return self.model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def sampled_ce_loss(model, seq, pos, neg):
    positive_logits, negative_logits = model(seq, pos, neg)
    valid = pos.ne(0)
    logits = torch.cat((positive_logits.unsqueeze(-1), negative_logits), dim=-1)[valid]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def _history_to_tensor(history, maxlen, device):
    sequence = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    return torch.as_tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # Local RNG makes checkpoint comparisons stable and leaves training RNG untouched.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            history = user_train[user]
            target = user_valid[user][0]
            excluded = set(history) | {target, 0}
            count = min(99, itemnum - sum(1 <= i <= itemnum for i in excluded))
            candidates = [target]
            while len(candidates) < count + 1:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for cutoff in (10, 20):
                if rank < cutoff:
                    totals[f'HR@{cutoff}'] += 1.0
                    totals[f'NDCG@{cutoff}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(users)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model_kwargs = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                        num_blocks=NUM_BLOCKS, kernel_size=KERNEL_SIZE,
                        dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    checkpoint_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = sampled_ce_loss(model, seq, pos, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} '
                  f'sampled_ce={total_loss / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr:.5f} val_NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError(f'No validation checkpoint saved for {dataset_name}')
    metadata = dict(model_kwargs)
    metadata.update(best_epoch=best_epoch, best_validation_hr10=best_hr,
                    objective='sampled_categorical_cross_entropy', negatives=NUM_NEGATIVES,
                    seed=SEED)
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as handle:
        pickle.dump(metadata, handle)
    _SAMPLERS.pop((id(user_train), itemnum), None)
    print(f'[{dataset_name}] best_epoch={best_epoch} best_val_HR@10={best_hr:.5f}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture so inference has no dependency on
    # importing train.py or executing its environment configuration.
    architecture = '\n\n'.join(inspect.getsource(cls) for cls in
                                (NextItNetBlock, NextItNetModel))
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    predictor = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    kwargs = {key: meta[key] for key in
              ('itemnum', 'hidden_units', 'num_blocks', 'kernel_size', 'dropout_rate', 'maxlen')}
    model = NextItNetModel(**kwargs).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset_name = dataset_name or DATASETS[0]
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if not candidates:
        return []
    _ensure_loaded(dataset_name)
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(header + architecture + '\n\n' + predictor)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f'Macro-average selected validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
