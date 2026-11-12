"""Per-category NextItNet with training-only dropout-view contrastive loss."""
from __future__ import annotations

import inspect
import json
import math
import os
import random
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
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
CONTRASTIVE_WEIGHT = 0.05
TEMPERATURE = 0.2
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_training():
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
            parts = line.strip().split()
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
        itemnum = max((item for split in (train, valid) for items in split.values() for item in items), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen, users=None):
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2 and len(set(items)) < itemnum]
    if not users:
        raise RuntimeError('No eligible training histories')
    seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(seqs)
    negatives = np.zeros_like(seqs)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        seen = set(history)
        nxt = history[-1]
        for idx, item in zip(range(maxlen - 1, -1, -1), reversed(history[:-1])):
            seqs[row, idx] = item
            positives[row, idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in seen:
                negative = random.randint(1, itemnum)
            negatives[row, idx] = negative
            nxt = item
    return seqs, positives, negatives


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

    def encode_sequence(self, seq):
        batch, length = seq.shape
        mask = seq.ne(0)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * mask.long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if features.ndim == embeddings.ndim:
            return (features * embeddings).sum(-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(1).clamp_min(0)
        final = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = torch.matmul(final, self.item_emb(candidates).transpose(-1, -2))
        return scores[0] if seq.size(0) == 1 else scores


class ModelView(nn.Module):
    """Compatibility view for family-specific diagnostics; shares the trained model."""
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


def last_features(features, seq):
    positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
    last = positions.masked_fill(seq.eq(0), -1).amax(1).clamp_min(0)
    return features[torch.arange(seq.size(0), device=seq.device), last]


def dropout_contrastive_loss(features1, features2, seq, pos):
    nonempty = seq.ne(0).any(dim=1)
    if int(nonempty.sum()) < 2:
        return features1.sum() * 0.0
    seq = seq[nonempty]
    pos = pos[nonempty]
    a = F.normalize(last_features(features1[nonempty], seq), dim=-1)
    b = F.normalize(last_features(features2[nonempty], seq), dim=-1)
    batch = a.size(0)
    z = torch.cat((a, b), dim=0)
    logits = torch.matmul(z, z.T) / TEMPERATURE
    with torch.no_grad():
        _, history_ids = torch.unique(seq, dim=0, return_inverse=True)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(1).clamp_min(0)
        targets = pos[torch.arange(batch, device=seq.device), last]
        duplicate = history_ids[:, None].eq(history_ids[None, :]) | targets[:, None].eq(targets[None, :])
        excluded = duplicate.repeat(2, 2)
        rows = torch.arange(2 * batch, device=seq.device)
        positive_indices = (rows + batch) % (2 * batch)
        excluded[rows, positive_indices] = False
        excluded.fill_diagonal_(True)
    return F.cross_entropy(logits.masked_fill(excluded, float('-inf')), positive_indices)


def training_loss(model, seq, pos, neg):
    # Both stochastic views use this single model and optimizer.
    features1 = model.log2feats(seq)
    positive_scores = model.predict_score(features1, pos)
    negative_scores = model.predict_score(features1, neg)
    mask = pos.ne(0)
    recommendation = F.softplus(negative_scores - positive_scores)[mask].mean()
    features2 = model.log2feats(seq)
    contrastive = dropout_contrastive_loss(features1, features2, seq, pos)
    return recommendation + CONTRASTIVE_WEIGHT * contrastive, recommendation.detach(), contrastive.detach()


def bpr_sequence_loss(model, seq, pos, neg):
    positive_scores, negative_scores = model(seq, pos, neg)
    mask = pos.ne(0)
    return F.softplus(negative_scores - positive_scores)[mask].mean()


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    recent = history[-maxlen:]
    if recent:
        seq[-len(recent):] = recent
    return torch.as_tensor(seq, dtype=torch.long, device=device).unsqueeze(0)


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    # Validation never advances training RNGs; candidates stay fixed across epochs.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        seen = set(user_train[user]) | {target, 0}
        candidates = [target]
        available = itemnum - sum(1 for item in seen if 1 <= item <= itemnum)
        count = min(99, available)
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in seen:
                seen.add(item)
                candidates.append(item)
        cases.append((user_train[user], candidates))
    return cases


def evaluate_cases(model, cases, device, maxlen):
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    model.eval()
    with torch.no_grad():
        for history, candidates in cases:
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train()
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, num_users), device, maxlen)


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty training data for {dataset_name}')
    users = [u for u, history in user_train.items() if len(history) >= 2 and len(set(history)) < itemnum]
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    ckpt_path = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.json'), 'w') as f:
        json.dump(config, f)
    best_hr = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = rec_sum = cl_sum = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN, users)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss, rec, cl = training_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss in {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
            rec_sum += rec.item()
            cl_sum += cl.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={loss_sum / NUM_BATCHES:.4f} '
                  f'bpr={rec_sum / NUM_BATCHES:.4f} contrastive={cl_sum / NUM_BATCHES:.4f} '
                  f'val_HR@10={hr:.4f} val_NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture for a standalone, consistent predictor.
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    header = '''from __future__ import annotations
import json
import math
import os
import torch
import torch.nn as nn

'''
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _load_dataset(dataset_name):
    global _device
    if dataset_name in _models:
        return _models[dataset_name], _metas[dataset_name]
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.json')) as f:
        meta = json.load(f)
    model = NextItNetModel(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta
    return model, meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset_name = DATASETS[0] if dataset_name is None else dataset_name
    model, meta = _load_dataset(dataset_name)
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    recent = list(history[-maxlen:])
    if recent:
        seq_arr[-len(recent):] = recent
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n' + runtime)


def main():
    seed_training()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'NextItNet dropout-view training on {device}; seed={SEED}', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
