"""One NextItNet per Amazon category, trained with BPR-max."""
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

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DEFAULT_DATASETS = DATASETS[:]
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
NUM_NEGATIVES = 64
BPR_MAX_REG = 0.01


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


def _eligible_uniform_negatives(history, itemnum, shape):
    excluded = np.unique(np.asarray(history, dtype=np.int64))
    excluded = excluded[(excluded >= 1) & (excluded <= itemnum)]
    if len(excluded) >= itemnum:
        raise RuntimeError('User has no eligible training negatives')
    # Rejection sampling is exactly uniform over the eligible catalog.
    if len(excluded) > itemnum // 2:
        eligible = np.setdiff1d(np.arange(1, itemnum + 1), excluded, assume_unique=True)
        return np.random.choice(eligible, size=shape, replace=True)
    values = np.random.randint(1, itemnum + 1, size=shape)
    invalid = np.isin(values, excluded)
    while invalid.any():
        values[invalid] = np.random.randint(1, itemnum + 1, size=int(invalid.sum()))
        invalid = np.isin(values, excluded)
    return values


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1, eligible_users=None):
    # The default interface retains the original [batch, length] negatives.
    users = eligible_users if eligible_users is not None else [u for u, h in user_train.items() if len(h) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
    for b in range(batch_size):
        history = user_train[random.choice(users)]
        count = min(maxlen, len(history) - 1)
        seq[b, -count:] = history[-count-1:-1]
        pos[b, -count:] = history[-count:]
        neg[b, -count:, :] = _eligible_uniform_negatives(history, itemnum, (count, num_negatives))
    if num_negatives == 1:
        neg = neg[..., 0]
    return seq, pos, neg


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
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
        return residual + self.dropout(y)


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
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
            x = x * (seq != 0).unsqueeze(-1).float()
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        emb = self.item_emb(items)
        if emb.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * emb).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        final = features[torch.arange(seq.size(0), device=seq.device), last]
        if candidates.ndim == 1:
            scores = final @ self.item_emb(candidates).T
        else:
            scores = self.predict_score(final, candidates)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView(nn.Module):
    """Family-specific diagnostic interface for the same trained network."""
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


def bpr_max_sequence_loss(model, seq, pos, neg):
    # Gather valid positions before looking up K embeddings per position.
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    positive = model.predict_score(features, pos[mask])
    negative_ids = neg[mask]
    if negative_ids.ndim == 1:
        negative_ids = negative_ids.unsqueeze(-1)
    negative = model.predict_score(features, negative_ids)
    log_w = F.log_softmax(negative, dim=-1)
    ranking_loss = -torch.logsumexp(log_w + F.logsigmoid(positive.unsqueeze(-1) - negative), dim=-1)
    score_regularizer = (log_w.exp() * negative.square()).sum(dim=-1)
    return (ranking_loss + BPR_MAX_REG * score_regularizer).mean()


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    # Local RNG makes checkpoint comparisons independent of training randomness.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        if itemnum - len({i for i in excluded if 1 <= i <= itemnum}) < 99:
            raise RuntimeError('Validation requires 99 unique eligible negatives')
        candidates = [target]
        while len(candidates) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        seq = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        cases.append((seq, candidates))
    return cases


def evaluate_cases(model, cases, device):
    totals = np.zeros(4, dtype=np.float64)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for seq, candidates in cases:
            seq_t = torch.as_tensor(seq[None, :], dtype=torch.long, device=device)
            candidates_t = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, candidates_t).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: (-scores[i], i)).index(0)
            totals[0] += rank < 10
            totals[1] += rank < 20
            totals[2] += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
            totals[3] += 1.0 / math.log2(rank + 2) if rank < 20 else 0.0
    model.train(was_training)
    totals /= max(len(cases), 1)
    return dict(zip(['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'], totals.tolist()))


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, maxlen), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    users = [u for u, h in user_train.items() if len(h) >= 2]
    if itemnum <= 0 or not users:
        raise RuntimeError(f'Empty training data for {dataset_name}')
    model_kwargs = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                        kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    best_hr10, best_ndcg, best_epoch = -1.0, -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN,
                                         num_negatives=NUM_NEGATIVES, eligible_users=users)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_max_sequence_loss(model, seq_t, pos_t, neg_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite BPR-max loss in {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr10, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} loss={total_loss / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr10:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if (hr10, ndcg) > (best_hr10, best_ndcg):
                best_hr10, best_ndcg, best_epoch = hr10, ndcg, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('No validation checkpoint was saved')
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(dict(model_kwargs, best_epoch=best_epoch, objective='bpr-max',
                         num_negatives=NUM_NEGATIVES, score_regularizer=BPR_MAX_REG), f)
    print(f'[{dataset_name}] best validation HR@10={best_hr10:.5f} epoch={best_epoch}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr10


def write_predict_script(working_dir):
    # Embed the exact model definitions used for training; no dependency on importing train.py.
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    model_source = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    predictor = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in DATASETS:
        with open(os.path.join(_HERE, f'{dataset}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        keys = ('itemnum', 'hidden_units', 'num_blocks', 'kernel_size', 'dropout_rate', 'maxlen')
        model = NextItNetModel(**{k: meta[k] for k in keys}).to(_device)
        state = torch.load(os.path.join(_HERE, f'{dataset}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    dataset = dataset_name if dataset_name is not None else DATASETS[0]
    if dataset not in _models:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not candidates:
        return []
    model, meta = _models[dataset], _metas[dataset]
    maxlen = int(meta['maxlen'])
    seq = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + model_source + '\n\n' + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    print(f'NextItNet BPR-max K={NUM_NEGATIVES} lambda={BPR_MAX_REG} device={device}', flush=True)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro-average best validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
