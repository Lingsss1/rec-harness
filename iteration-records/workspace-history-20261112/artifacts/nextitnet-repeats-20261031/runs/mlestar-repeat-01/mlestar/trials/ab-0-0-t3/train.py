"""Single-checkpoint per-category NextItNet with conservative dropout regularization."""
from __future__ import annotations
import json
import math
import os
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
MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
CONTRASTIVE_WEIGHT = 0.0
TEMPERATURE = 0.2


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
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
        itemnum = max((i for split in (train, valid) for history in split.values() for i in history), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, history in user_train.items() if len(history) >= 2]
    if not users:
        raise RuntimeError('No eligible training users')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('No eligible negative items')
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
    return seq, pos, neg


def last_features(features, seq):
    positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
    last = positions.masked_fill(seq.eq(0), -1).amax(1).clamp_min(0)
    return features[torch.arange(seq.size(0), device=seq.device), last]


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
        dilations = ([1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4)))[:num_blocks]
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, d, kernel_size, dropout_rate) for d in dilations])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq) + 1
        mask = seq.ne(0)
        positions = positions * mask.long()
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x) * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        seq = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(seq)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        emb = self.item_emb(items)
        if items.ndim == 1:
            return features @ emb.transpose(-1, -2)
        if features.ndim == emb.ndim:
            return (features * emb).sum(-1)
        if features.ndim == 2 and emb.ndim == 3:
            return (features.unsqueeze(1) * emb).sum(-1)
        return torch.matmul(features, emb.transpose(-1, -2))

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return (features * self.item_emb(pos)).sum(-1), (features * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        features = last_features(self.encode_sequence(seq), seq)
        scores = self.predict_score(features, candidates)
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


def bpr_sequence_loss(model, seq, pos, neg, features=None):
    if features is None:
        features = model.encode_sequence(seq)
    positive = (features * model.item_emb(pos)).sum(-1)
    negative = (features * model.item_emb(neg)).sum(-1)
    mask = pos.ne(0)
    return F.softplus(negative - positive)[mask].mean()


def dropout_contrastive_loss(first, second, seq, pos):
    first = last_features(first, seq)
    second = last_features(second, seq)
    targets = last_features(pos.unsqueeze(-1), seq).squeeze(-1)
    batch = first.size(0)
    if batch < 2:
        return first.sum() * 0.0
    z = torch.cat([F.normalize(first.float(), dim=-1), F.normalize(second.float(), dim=-1)], dim=0)
    logits = z @ z.T / TEMPERATURE
    same_history = seq.unsqueeze(1).eq(seq.unsqueeze(0)).all(-1)
    same_target = targets[:, None].eq(targets[None, :])
    excluded = (same_history | same_target).repeat(2, 2)
    rows = torch.arange(2 * batch, device=seq.device)
    positives = (rows + batch) % (2 * batch)
    excluded[rows, positives] = False
    excluded[rows, rows] = True
    return F.cross_entropy(logits.masked_fill(excluded, float('-inf')), positives)


def make_validation_batches(user_train, user_valid, itemnum, maxlen):
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = random.Random(SEED).sample(users, 200)
    rng = random.Random(SEED)
    sequences, candidates = [], []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        if itemnum - len(excluded - {0}) < 99:
            raise RuntimeError('Insufficient unique validation negatives')
        choices = [target]
        while len(choices) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                choices.append(item)
        sequence = [0] * maxlen
        tail = history[-maxlen:]
        if tail:
            sequence[-len(tail):] = tail
        sequences.append(sequence)
        candidates.append(choices)
    return sequences, candidates


@torch.no_grad()
def evaluate_val(model, validation, device):
    sequences, candidates = validation
    if not sequences:
        raise RuntimeError('No validation users')
    model.eval()
    hits = ndcg = 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.tensor(sequences[start:start + 64], dtype=torch.long, device=device)
        items = torch.tensor(candidates[start:start + 64], dtype=torch.long, device=device)
        features = last_features(model.encode_sequence(seq), seq)
        scores = model.predict_score(features, items)
        ranks = scores.argsort(dim=1, descending=True, stable=True).eq(0).long().argmax(1)
        hits += ranks.lt(10).float().sum().item()
        ndcg += torch.where(ranks < 10, 1.0 / torch.log2(ranks.float() + 2), 0.0).sum().item()
    model.train()
    return {'HR@10': hits / len(sequences), 'NDCG@10': ndcg / len(sequences)}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    validation = make_validation_batches(user_train, user_valid, itemnum, MAXLEN)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, dataset_name + '_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        recommendation_total = contrastive_total = 0.0
        weight = CONTRASTIVE_WEIGHT * min(1.0, max(0.0, (epoch - 10) / 10.0))
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            first = model.encode_sequence(seq)
            recommendation = bpr_sequence_loss(model, seq, pos, neg, first)
            contrastive = first.new_zeros(())
            if weight > 0:
                second = model.encode_sequence(seq)
                contrastive = dropout_contrastive_loss(first, second, seq, pos)
            loss = recommendation + weight * contrastive
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            recommendation_total += recommendation.detach().item()
            contrastive_total += contrastive.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, validation, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} bpr={recommendation_total / NUM_BATCHES:.5f} '
                  f'contrastive={contrastive_total / NUM_BATCHES:.5f} weight={weight:.5f} '
                  f'val_HR@10={hr:.5f} val_NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(working, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best_validation_HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    script = r'''"""One trained NextItNet checkpoint per dataset."""
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _ensure_loaded(dataset):
    global _device, _model_class
    if dataset in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        spec = importlib.util.spec_from_file_location('nextitnet_training_definition', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.NextItNetModel
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _model_class(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = DATASETS[0] if dataset_name is None else dataset_name
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    _ensure_loaded(dataset)
    if len(candidates) == 0:
        return []
    model = _models[dataset]
    maxlen = _metas[dataset]['maxlen']
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        return [float(v) for v in model.score_candidates(seq, items).cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro-average best validation HR@10: {sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
