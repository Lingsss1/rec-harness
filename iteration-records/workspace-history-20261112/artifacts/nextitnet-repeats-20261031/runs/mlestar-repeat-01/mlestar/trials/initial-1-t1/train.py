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
from torch.utils.checkpoint import checkpoint

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
NUM_EPOCHS = max(1, min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
CALIBRATION = 0.75
LOSS_CHUNK = 256
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_SAMPLER_CACHE = {}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if itemnum <= 0:
        itemnum = max((max(items) for split in (train, valid) for items in split.values() if items), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    # Negatives depend exclusively on training histories. Repeats across draws
    # are allowed, as for independent uniform sampling with replacement.
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        entries = []
        for history in user_train.values():
            if len(history) < 2:
                continue
            excluded = np.unique(np.asarray(history, dtype=np.int64))
            excluded = excluded[(excluded >= 1) & (excluded <= itemnum)]
            if len(excluded) >= itemnum:
                continue
            allowed = None
            if len(excluded) > itemnum // 2:
                allowed = np.setdiff1d(np.arange(1, itemnum + 1, dtype=np.int64), excluded, assume_unique=True)
            entries.append((np.asarray(history, dtype=np.int64), excluded, allowed))
        if not entries:
            raise RuntimeError('No eligible training users with an available negative item')
        _SAMPLER_CACHE[key] = (user_train, entries)
    entries = _SAMPLER_CACHE[key][1]
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
    for row, index in enumerate(np.random.randint(len(entries), size=batch_size)):
        history, excluded, allowed = entries[index]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        shape = (length, NUM_NEGATIVES)
        if allowed is not None:
            sampled = allowed[np.random.randint(len(allowed), size=shape)]
        else:
            sampled = np.random.randint(1, itemnum + 1, size=shape)
            bad = np.isin(sampled, excluded)
            while bad.any():
                sampled[bad] = np.random.randint(1, itemnum + 1, size=int(bad.sum()))
                bad = np.isin(sampled, excluded)
        neg[row, -length:, :] = sampled
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
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
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

    def log2feats(self, seq):
        batch, length = seq.shape
        mask = (seq != 0).unsqueeze(-1)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * (seq != 0).long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask.to(x.dtype)
        return self.norm(x) * mask.to(x.dtype)

    def encode_sequence(self, seq):
        return self.log2feats(seq)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.ndim == 3:
            negative = (features.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        else:
            negative = (features * self.item_emb(neg)).sum(-1)
        return positive, negative

    def score_candidates(self, seq, candidates):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        final = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(final, candidates)
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


def gbce_loss(p, n, itemnum, t=0.75):
    k = n.size(-1)
    alpha = k / max(itemnum - 1, 1)
    beta = 1.0 - t + alpha * t
    return ((beta * F.softplus(-p.float()) + F.softplus(n.float()).sum(-1)) / (k + 1)).mean()


def gbce_sequence_loss(model, seq, pos, neg):
    features = model.log2feats(seq)
    mask = pos != 0
    features = features[mask]
    positives = pos[mask]
    negatives = neg[mask]
    count = features.size(0)
    if count == 0:
        return features.sum() * 0.0

    def chunk_loss(feats, positive_ids, negative_ids):
        p = (feats * model.item_emb(positive_ids)).sum(-1)
        # Only this chunk's [positions, K, H] embedding is materialized.
        n = torch.bmm(model.item_emb(negative_ids), feats.unsqueeze(-1)).squeeze(-1)
        return gbce_loss(p, n, model.itemnum, CALIBRATION)

    loss = features.new_zeros(())
    for start in range(0, count, LOSS_CHUNK):
        end = min(start + LOSS_CHUNK, count)
        # Recompute sampled embeddings in backward rather than retaining all
        # chunks' K-by-H activations until the encoder's backward pass.
        part = checkpoint(chunk_loss, features[start:end], positives[start:end],
                          negatives[start:end], use_reentrant=False,
                          preserve_rng_state=False)
        loss = loss + part * ((end - start) / count)
    return loss


def build_validation(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        available = itemnum - sum(1 for i in excluded if 1 <= i <= itemnum)
        if available < 99:
            raise RuntimeError('Validation requires 99 unique available negatives')
        row = [target]
        while len(row) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                row.append(item)
        seq = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        sequences.append(seq)
        candidates.append(row)
    if not sequences:
        raise RuntimeError('No validation users')
    return np.asarray(sequences), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, validation, device):
    model.eval()
    sequences, candidates = validation
    ranks = []
    for start in range(0, len(sequences), 32):
        seq = torch.as_tensor(sequences[start:start + 32], device=device)
        cand = torch.as_tensor(candidates[start:start + 32], device=device)
        scores = model.score_candidates(seq, cand)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        # Stable descending ordering puts the target first among equal scores.
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        ranks.extend((order == 0).long().argmax(-1).cpu().tolist())
    ranks = np.asarray(ranks)
    result = {'HR@10': float(np.mean(ranks < 10)),
              'HR@20': float(np.mean(ranks < 20)),
              'NDCG@10': float(np.mean(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0)))}
    model.train()
    return result


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    validation = build_validation(user_train, user_valid, itemnum, MAXLEN)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ckpt_path = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = torch.zeros((), device=device)
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = gbce_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.detach()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, validation, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={running_loss.item()/NUM_BATCHES:.6f} '
                  f'HR@10={hr:.4f} HR@20={metrics["HR@20"]:.4f} '
                  f'NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at {epoch}', flush=True)
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    _SAMPLER_CACHE.clear()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture so inference does not depend on
    # importing train.py or on training-only environment settings.
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    config = dict(meta)
    config['dropout_rate'] = 0.0
    model = NextItNetModel(**config).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset_name = dataset_name or DATASETS[0]
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t)
    return [float(value) for value in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + runtime)


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'NextItNet gBCE: device={device}, K={NUM_NEGATIVES}, t={CALIBRATION}, '
          f'epochs<={NUM_EPOCHS}, batches={NUM_BATCHES}, seed={SEED}', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro-average best validation HR@10={sum(scores)/len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
