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
PATIENCE = 25
NUM_NEGATIVES = 16
HARD_WEIGHT = 0.25
SCORE_REG = 0.01
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
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
    largest = 0
    for split in ('train', 'valid'):
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 2:
                    largest = max(largest, int(parts[1]))
    return largest


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.train = user_train
        self.itemnum = itemnum
        self.seen = {u: set(items) for u, items in user_train.items()}
        self.users = [u for u, items in user_train.items()
                      if len(items) >= 2 and len(self.seen[u]) < itemnum]
        if not self.users:
            raise RuntimeError('No eligible training users')
        self.complements = {}
        for u in self.users:
            if len(self.seen[u]) > itemnum // 2:
                self.complements[u] = np.asarray([i for i in range(1, itemnum + 1)
                                                 if i not in self.seen[u]], dtype=np.int64)

    def batch(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros_like(seq)
        for row in range(batch_size):
            user = random.choice(self.users)
            history = self.train[user]
            inputs = history[:-1][-maxlen:]
            targets = history[1:][-maxlen:]
            length = len(inputs)
            seq[row, -length:] = inputs
            pos[row, -length:] = targets
            if user in self.complements:
                negatives = np.random.choice(self.complements[user], size=length, replace=True)
            else:
                negatives = np.random.randint(1, self.itemnum + 1, size=length)
                for j in range(length):
                    while int(negatives[j]) in self.seen[user]:
                        negatives[j] = np.random.randint(1, self.itemnum + 1)
            neg[row, -length:] = negatives
        return seq, pos, neg


_sampler_cache = None


def sample_batch(user_train, itemnum, batch_size, maxlen):
    global _sampler_cache
    if _sampler_cache is None or _sampler_cache.train is not user_train or _sampler_cache.itemnum != itemnum:
        _sampler_cache = TrainingSampler(user_train, itemnum)
    return _sampler_cache.batch(batch_size, maxlen)


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
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
                                     for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        mask = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * (seq != 0).long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x) * mask

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if features.ndim == 2 and items.ndim == 2:
            return torch.einsum('bd,bkd->bk', features, embeddings)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(1).values.clamp(min=0)
        final = features[torch.arange(seq.shape[0], device=seq.device), last]
        scores = self.predict_score(final, candidates)
        return scores[0] if seq.shape[0] == 1 else scores


def bpr_sequence_loss(model, seq, pos, neg):
    features = model.log2feats(seq)
    positive = (features * model.item_emb(pos)).sum(-1)
    negative = (features * model.item_emb(neg)).sum(-1)
    valid = pos != 0
    if not valid.any():
        return positive.sum() * 0.0
    pairwise = F.softplus(negative - positive)
    # Pool only this user's sampled unseen negatives; padding receives no weight.
    pool = neg[:, -min(NUM_NEGATIVES, neg.shape[1]):]
    multi = torch.einsum('bld,bkd->blk', features, model.item_emb(pool))
    pool_mask = (pool != 0).unsqueeze(1)
    # Temperature 2 retains hard-negative emphasis without concentrating all
    # gradient on a single high-scoring sampled item.
    logits = (multi.detach() / 2.0).masked_fill(~pool_mask, -1e9)
    weights = F.softmax(logits, dim=-1) * pool_mask.to(multi.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    hard_ranking = (weights * F.softplus(multi - positive.unsqueeze(-1))).sum(-1)
    # Penalize both score signs so negative-only shrinkage cannot shift the
    # entire score distribution upward without a corresponding cost.
    negative_penalty = ((1.0 - HARD_WEIGHT) * negative.square()
                        + HARD_WEIGHT * (weights * multi.square()).sum(-1))
    penalty = 0.5 * (positive.square() + negative_penalty)
    loss = (1.0 - HARD_WEIGHT) * pairwise + HARD_WEIGHT * hard_ranking + SCORE_REG * penalty
    return loss[valid].mean()


def _history_to_tensor(history, maxlen, device):
    values = [0] * maxlen
    clipped = history[-maxlen:]
    if clipped:
        values[-len(clipped):] = clipped
    return torch.tensor([values], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        available = itemnum - sum(1 for i in excluded if 1 <= i <= itemnum)
        count = min(99, available)
        if count < 0:
            raise RuntimeError('Invalid validation vocabulary')
        if available <= 2 * count:
            eligible = [i for i in range(1, itemnum + 1) if i not in excluded]
            negatives = rng.sample(eligible, count)
        else:
            negatives = []
            while len(negatives) < count:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    negatives.append(item)
        cases.append((user_train[user], [target] + negatives))
    return cases


@torch.no_grad()
def evaluate_cases(model, cases, device, maxlen):
    was_training = model.training
    model.eval()
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    for history, candidates in cases:
        seq = _history_to_tensor(history, maxlen, device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items)
        rank = int((scores[1:] > scores[0]).sum().item())
        for cutoff in (10, 20):
            if rank < cutoff:
                totals[f'HR@{cutoff}'] += 1.0
                totals[f'NDCG@{cutoff}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, num_users), device, maxlen)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation users: {dataset_name}')
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total / NUM_BATCHES:.5f} validation={metrics}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as handle:
        pickle.dump(config, handle)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
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
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('nextitnet_trial_model', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as handle:
            config = pickle.load(handle)
        model = module.NextItNetModel(**config).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = config


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f'Unknown dataset_name: {ds}')
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
    seq = [0] * maxlen
    clipped = history[-maxlen:]
    if clipped:
        seq[-len(clipped):] = clipped
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro-average validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
