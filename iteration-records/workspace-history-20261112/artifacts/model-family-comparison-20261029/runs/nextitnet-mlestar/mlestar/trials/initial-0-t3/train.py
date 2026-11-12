"""NextItNet with dropout-view contrastive regularization, one model per category."""
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

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
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
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
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
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
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


def sample_batch(user_train, itemnum, batch_size, maxlen, return_users=False,
                 eligible_users=None, interacted_sets=None):
    users = eligible_users if eligible_users is not None else [u for u, h in user_train.items() if len(h) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)
    sampled_users = np.empty(batch_size, dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        sampled_users[row] = user
        history = user_train[user]
        interacted = interacted_sets[user] if interacted_sets is not None else set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('No unobserved training negative available')
        nxt = history[-1]
        index = maxlen - 1
        for item in reversed(history[:-1]):
            sequences[row, index] = item
            positives[row, index] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            negatives[row, index] = negative
            nxt = item
            index -= 1
            if index < 0:
                break
    if return_users:
        return sequences, positives, negatives, sampled_users
    return sequences, positives, negatives


def final_indices(seq):
    positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
    return positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp_min(0)


def final_features(features, seq):
    return features[torch.arange(seq.size(0), device=seq.device), final_indices(seq)]


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
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128):
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
        mask = seq.ne(0).unsqueeze(-1)
        positions = (torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1) * seq.ne(0).long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        out = self.encode_sequence(seq)
        return (out * self.item_emb(pos)).sum(-1), (out * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        features = final_features(self.encode_sequence(seq), seq)
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView:
    def __init__(self, model):
        self.model = model

    @property
    def item_emb(self):
        return self.model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos.ne(0)
    return F.softplus(negative - positive)[mask].mean()


def dropout_contrastive_loss(first, second, seq, users, targets):
    a = F.normalize(final_features(first, seq), dim=-1)
    b = F.normalize(final_features(second, seq), dim=-1)
    z = torch.cat((a, b), dim=0)
    batch = seq.size(0)
    indices = torch.arange(2 * batch, device=z.device)
    labels = (indices + batch) % (2 * batch)
    all_users = users.repeat(2)
    all_targets = targets.repeat(2)
    excluded = all_users[:, None].eq(all_users[None, :]) | all_targets[:, None].eq(all_targets[None, :])
    excluded[indices, labels] = False
    excluded[indices, indices] = True
    logits = (z @ z.T) / CL_TEMPERATURE
    return F.cross_entropy(logits.masked_fill(excluded, float('-inf')), labels)


def make_validation_cases(train, valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | {target, 0}
        candidates = [target]
        desired = min(99, itemnum - len(excluded - {0}))
        while len(candidates) < desired + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        sequence = np.zeros(maxlen, dtype=np.int64)
        history = train[user][-maxlen:]
        if history:
            sequence[-len(history):] = history
        cases.append((sequence, candidates))
    return cases


@torch.no_grad()
def evaluate_cases(model, cases, device):
    model.eval()
    hits = 0.0
    ndcg = 0.0
    for sequence, candidates in cases:
        seq = torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)
        items = torch.as_tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items)
        order = torch.argsort(scores, descending=True, stable=True)
        rank = int((order == 0).nonzero(as_tuple=False)[0, 0])
        if rank < 10:
            hits += 1
            ndcg += 1 / math.log2(rank + 2)
    model.train()
    return {'HR@10': hits / max(1, len(cases)), 'NDCG@10': ndcg / max(1, len(cases))}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, maxlen), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    users = [u for u, history in train.items() if len(history) >= 2]
    interacted = {u: set(train[u]) for u in users}
    cases = make_validation_cases(train, valid, itemnum, MAXLEN)
    if not cases:
        raise RuntimeError(f'No validation users: {dataset_name}')
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, f'{dataset_name}_model.pt')
    with open(os.path.join(working, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    best_hr = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_bpr = total_cl = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg, ids = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN,
                                             return_users=True, eligible_users=users,
                                             interacted_sets=interacted)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            ids = torch.as_tensor(ids, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            first = model.encode_sequence(seq)
            second = model.encode_sequence(seq)
            positive = (first * model.item_emb(pos)).sum(-1)
            negative = (first * model.item_emb(neg)).sum(-1)
            bpr = F.softplus(negative - positive)[pos.ne(0)].mean()
            targets = pos[torch.arange(seq.size(0), device=device), final_indices(seq)]
            cl = dropout_contrastive_loss(first, second, seq, ids, targets)
            loss = bpr + CL_WEIGHT * cl
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss in {dataset_name}, epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_bpr += bpr.detach().item()
            total_cl += cl.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} BPR={total_bpr / NUM_BATCHES:.4f} CL={total_cl / NUM_BATCHES:.4f} val_HR@10={hr:.4f} val_NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f}, epoch={best_epoch}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    script = r'''"""Single-checkpoint per-category NextItNet inference."""
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


def _load_model_class():
    global _model_class
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('nextitnet_contrastive_train', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.NextItNetModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _load_model_class()(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    _ensure_loaded(ds)
    if len(candidates) == 0:
        return []
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = _models[ds].score_candidates(seq, items)
    return [float(value) for value in scores.cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro-average selected validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
