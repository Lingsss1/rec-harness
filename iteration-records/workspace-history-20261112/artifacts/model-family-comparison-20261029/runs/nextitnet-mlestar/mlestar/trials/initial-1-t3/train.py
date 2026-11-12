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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = 0.2
LR = float(os.environ.get('NEXTITNET_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('NEXTITNET_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('NEXTITNET_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
CL_WEIGHT = 0.05
CL_TEMPERATURE = 0.2


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen, return_metadata=False,
                 eligible_users=None, interacted_by_user=None):
    users = eligible_users
    if users is None:
        users = [u for u, items in user_train.items()
                 if len(items) >= 2 and len(set(items)) < itemnum]
    if not users:
        raise RuntimeError('No eligible training users')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    user_ids = np.empty(batch_size, dtype=np.int64)
    final_targets = np.empty(batch_size, dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        interacted = (interacted_by_user[user] if interacted_by_user is not None
                      else set(history))
        user_ids[row] = user
        final_targets[row] = history[-1]
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[row, idx] = item
            pos[row, idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            neg[row, idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
    if return_metadata:
        return seq, pos, neg, user_ids, final_targets
    return seq, pos, neg


def last_nonpadding(seq):
    positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
    return positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)


def gather_last(features, seq):
    return features[torch.arange(seq.size(0), device=seq.device), last_nonpadding(seq)]


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
        positions = torch.arange(1, length + 1, device=seq.device).expand(batch, -1)
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = seq.ne(0).unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if features.ndim == 2 and items.ndim == 2:
            return (features.unsqueeze(1) * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        features = gather_last(self.encode_sequence(seq), seq)
        scores = torch.matmul(features, self.item_emb(candidates).transpose(0, 1))
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos.ne(0)
    return F.softplus(negative - positive)[mask].mean()


def dropout_view_loss(first, second, seq, user_ids, final_targets):
    first = F.normalize(gather_last(first, seq), p=2, dim=-1)
    second = F.normalize(gather_last(second, seq), p=2, dim=-1)
    z = torch.cat((first, second), dim=0)
    batch = first.size(0)
    rows = torch.arange(2 * batch, device=z.device)
    targets = (rows + batch) % (2 * batch)
    users = torch.cat((user_ids, user_ids))
    labels = torch.cat((final_targets, final_targets))
    false_negative = users[:, None].eq(users[None, :]) | labels[:, None].eq(labels[None, :])
    false_negative[rows, targets] = False
    false_negative[rows, rows] = True
    logits = (z.float() @ z.float().T) / CL_TEMPERATURE
    logits = logits.masked_fill(false_negative, float('-inf'))
    return F.cross_entropy(logits, targets)


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros((1, maxlen), dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[0, -len(tail):] = tail
    return torch.as_tensor(seq, dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    metrics = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    if not users:
        return metrics
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            negative_count = min(99, itemnum - len([i for i in excluded if 1 <= i <= itemnum]))
            while len(candidates) < negative_count + 1:
                candidate = rng.randint(1, itemnum)
                if candidate not in excluded:
                    excluded.add(candidate)
                    candidates.append(candidate)
            sequence = _history_to_tensor(history, maxlen, device)
            items = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(sequence, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    metrics[f'HR@{k}'] += 1.0
                    metrics[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / len(users) for key, value in metrics.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    interacted = {u: set(items) for u, items in user_train.items()}
    eligible = [u for u, items in user_train.items()
                if len(items) >= 2 and len(interacted[u]) < itemnum]
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN,
                                  return_metadata=True, eligible_users=eligible,
                                  interacted_by_user=interacted)
            seq, pos, neg, users, targets = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            first = model.encode_sequence(seq)
            positive = (first * model.item_emb(pos)).sum(-1)
            negative = (first * model.item_emb(neg)).sum(-1)
            bpr = F.softplus(negative - positive)[pos.ne(0)].mean()
            second = model.encode_sequence(seq)
            contrastive = dropout_view_loss(first, second, seq, users, targets)
            loss = bpr + CL_WEIGHT * contrastive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [loss.item(), bpr.item(), contrastive.item()]
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            averages = totals / NUM_BATCHES
            print(f'[{dataset_name}] epoch={epoch} loss={averages[0]:.4f} '
                  f'bpr={averages[1]:.4f} cl={averages[2]:.4f} '
                  f'val_score: {hr10:.4f} NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'[{dataset_name}] best validation HR@10={max(0.0, best_hr10):.4f}', flush=True)
    return max(0.0, best_hr10)


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
_ModelClass = None


def _ensure_loaded(dataset_name):
    global _device, _ModelClass
    if dataset_name in _models:
        return
    if dataset_name not in _DATASETS:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _ModelClass is None:
        spec = importlib.util.spec_from_file_location(
            f'nextitnet_contrastive_train_{os.getpid()}', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _ModelClass(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    _ensure_loaded(ds)
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
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
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f'Macro validation HR@10 = {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
