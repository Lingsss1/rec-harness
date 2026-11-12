"""Independent per-category NextItNet with BPR-max and 64 uniform negatives."""
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

DEFAULT_DATASETS = [
    'Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl'
]
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


# Cache only training interaction indices; never use validation interactions to
# choose training negatives. The public sampler keeps its three-array interface.
_sampler_cache = {}


def _sampling_index(user_train, itemnum):
    key = (id(user_train), itemnum)
    cached = _sampler_cache.get(key)
    if cached is not None and cached[0] is user_train:
        return cached[1], cached[2]
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('no users with at least two training interactions')
    excluded = {u: np.asarray(sorted(set(user_train[u])), dtype=np.int64) for u in users}
    for u in users:
        if np.count_nonzero((excluded[u] >= 1) & (excluded[u] <= itemnum)) >= itemnum:
            raise RuntimeError(f'user {u} has no eligible training negatives')
    _sampler_cache[key] = (user_train, users, excluded)
    return users, excluded


def _uniform_eligible(shape, itemnum, excluded):
    values = np.random.randint(1, itemnum + 1, size=shape, dtype=np.int64)
    invalid = np.isin(values, excluded)
    while invalid.any():
        values[invalid] = np.random.randint(1, itemnum + 1, size=int(invalid.sum()), dtype=np.int64)
        invalid = np.isin(values, excluded)
    return values


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    users, excluded = _sampling_index(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(len(history) - 1, maxlen)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        # Independent draws for every target, uniform over the complement of
        # this user's entire training history, with replacement across draws.
        neg[row, -length:, :] = _uniform_eligible(
            (length, num_negatives), itemnum, excluded[user]
        )
    if num_negatives == 1:
        neg = neg[:, :, 0]
    return seq, pos, neg


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


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
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        residual = x
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return residual + self.dropout(y)


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
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x)
            x = x * mask
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        positive = self.predict_score(features, pos)
        if neg.ndim == pos.ndim + 1:
            negative = self.predict_score(features.unsqueeze(-2), neg)
        else:
            negative = self.predict_score(features, neg)
        return positive, negative

    def score_candidates(self, seq, candidates):
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        if candidates.ndim == 1:
            scores = features @ self.item_emb(candidates).transpose(0, 1)
        else:
            scores = (features.unsqueeze(1) * self.item_emb(candidates)).sum(-1)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView(nn.Module):
    """Expose the same causal features and item scorer to diagnostics."""
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


def bpr_max_sequence_loss(model, seq, pos, neg):
    features = model.log2feats(seq)
    mask = pos != 0
    if not mask.any():
        return features.sum() * 0.0
    features = features[mask]
    positives = pos[mask]
    negatives = neg[mask]
    if negatives.ndim == 1:
        negatives = negatives.unsqueeze(-1)
    p = model.predict_score(features, positives)
    # Both ranking terms train the same causal network and item embeddings.
    total = features.new_zeros(())
    for start in range(0, features.shape[0], 1024):
        end = min(start + 1024, features.shape[0])
        n = model.predict_score(features[start:end].unsqueeze(1), negatives[start:end])
        positive = p[start:end].unsqueeze(-1)
        log_w = F.log_softmax(n, dim=-1)
        ranking_loss = -torch.logsumexp(
            log_w + F.logsigmoid(positive - n), dim=-1
        )
        # A stable listwise loss maintains pressure to beat the entire sampled
        # candidate set, including when pairwise sigmoid probabilities saturate.
        logits = torch.cat((positive, n), dim=-1)
        listwise_loss = torch.logsumexp(logits, dim=-1) - positive.squeeze(-1)
        score_regularizer = (log_w.exp() * n.square()).sum(dim=-1)
        total = total + (
            0.75 * ranking_loss + 0.25 * listwise_loss
            + BPR_MAX_REG * score_regularizer
        ).sum()
    return total / features.shape[0]


def bpr_sequence_loss(model, seq, pos, neg):
    # Keep the family-specific loss entry point usable by diagnostics as well.
    return bpr_max_sequence_loss(model, seq, pos, neg)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    # Local RNG prevents evaluations from changing training samples and fixes
    # validation users and unique candidate sets across checkpoints.
    rng = random.Random(SEED)
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    if not val_users:
        return {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    totals = np.zeros(4, dtype=np.float64)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            candidates = [target]
            eligible_count = itemnum - sum(1 for i in interacted if 1 <= i <= itemnum)
            required = min(99, eligible_count)
            while len(candidates) < required + 1:
                item = rng.randint(1, itemnum)
                if item not in interacted:
                    interacted.add(item)
                    candidates.append(item)
            seq_t = _history_to_tensor(history, maxlen, device)
            cand_t = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            gain = 1.0 / math.log2(rank + 2)
            totals += [float(rank < 10), float(rank < 20), gain if rank < 10 else 0.0, gain if rank < 20 else 0.0]
    model.train(was_training)
    totals /= len(val_users)
    return dict(zip(('HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'), totals.tolist()))


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'empty dataset or item vocabulary for {dataset_name}')
    model = NextItNetModel(
        itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
        kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, num_negatives=NUM_NEGATIVES
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_max_sequence_loss(model, seq_t, pos_t, neg_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(
                f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} '
                f'loss={total_loss / NUM_BATCHES:.5f} val_score: {hr10:.4f} '
                f'HR@20={metrics["HR@20"]:.4f} NDCG@10={metrics["NDCG@10"]:.4f}',
                flush=True
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('no validation checkpoint was selected')
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump({
            'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
            'num_blocks': NUM_BLOCKS, 'kernel_size': KERNEL_SIZE,
            'dropout_rate': DROPOUT_RATE, 'best_epoch': best_epoch,
            'objective': 'bpr-max', 'num_negatives': NUM_NEGATIVES,
            'score_regularization': BPR_MAX_REG
        }, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr10:.4f} epoch={best_epoch}', flush=True)
    _sampler_cache.clear()
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""One NextItNet checkpoint per category; no score fusion."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_ModelClass = None


def _load_model_class():
    global _ModelClass
    if _ModelClass is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('nextitnet_bprmax_training_definition', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.NextItNetModel
    return _ModelClass


def _ensure_loaded(dataset_name=None):
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    requested = DATASETS if dataset_name is None else [dataset_name]
    ModelClass = _load_model_class()
    for ds in requested:
        if ds in _models:
            continue
        if ds not in DATASETS:
            raise ValueError(f'unknown dataset: {ds}')
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
            num_blocks=meta['num_blocks'], kernel_size=meta['kernel_size'],
            dropout_rate=meta['dropout_rate'], maxlen=meta['maxlen']
        ).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = DATASETS[0] if dataset_name is None else dataset_name
    _ensure_loaded(ds)
    if not candidates:
        return []
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
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'NextItNet BPR-max on {device}; K={NUM_NEGATIVES}, lambda={BPR_MAX_REG}', flush=True)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    write_predict_script(working_dir)
    print(f'All datasets done. Average val HR@10 = {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
