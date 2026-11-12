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

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
PREPROCESSING_SEED = 0
MAXLEN = 128
HIDDEN_UNITS = 64
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 1]
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


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
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as handle:
            stats = json.load(handle)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if itemnum <= 0:
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
            raise RuntimeError('No unobserved item available for negative sampling')
        nxt = history[-1]
        index = maxlen - 1
        for item in reversed(history[:-1]):
            seq[row, index] = item
            pos[row, index] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            neg[row, index] = negative
            nxt = item
            index -= 1
            if index < 0:
                break
    return seq, pos, neg


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
        return conv(nn.functional.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 1]
        if len(dilations) != num_blocks:
            raise ValueError('Dilation count must equal block count')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, d, kernel_size, dropout_rate) for d in dilations
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        mask = (seq != 0).unsqueeze(-1)
        positions = torch.arange(1, seq.size(1) + 1, device=seq.device).unsqueeze(0)
        positions = positions.expand_as(seq) * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x) * mask
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x) * mask

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(dim=-1),
                (features * self.item_emb(neg)).sum(dim=-1))

    def score_candidates(self, seq, candidates):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 else scores


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos != 0
    losses = nn.functional.softplus(negative - positive)
    return losses[mask].mean()


def make_validation_cases(user_train, user_valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        available = itemnum - sum(1 for item in excluded if 1 <= item <= itemnum)
        count = min(99, available)
        candidates = [target]
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        seq = np.zeros(MAXLEN, dtype=np.int64)
        tail = history[-MAXLEN:]
        if tail:
            seq[-len(tail):] = tail
        cases.append((seq, candidates))
    if not cases:
        raise RuntimeError('No validation users')
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hits = 0
    ndcg = 0.0
    for start in range(0, len(cases), 64):
        group = cases[start:start + 64]
        seq = torch.as_tensor(np.stack([case[0] for case in group]), device=device)
        lengths = [len(case[1]) for case in group]
        width = max(lengths)
        candidates = np.zeros((len(group), width), dtype=np.int64)
        for row, (_, items) in enumerate(group):
            candidates[row, :len(items)] = items
        candidate_t = torch.as_tensor(candidates, device=device)
        scores = model.score_candidates(seq, candidate_t)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        for row, length in enumerate(lengths):
            values = scores[row, :length]
            rank = int((values[1:] > values[0]).sum().item())
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    model.train()
    return {'HR@10': hits / len(cases), 'NDCG@10': ndcg / len(cases)}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    architecture = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                        num_blocks=len(DILATIONS), kernel_size=KERNEL_SIZE,
                        dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
                        dilations=list(DILATIONS))
    model = NextItNetModel(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} dilations={DILATIONS}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.5f} val_NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as handle:
        pickle.dump(architecture, handle)
    print(f'[{dataset_name}] selected_epoch={best_epoch} val_HR@10={best_hr:.5f}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = '''from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

'''
    model_source = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    predictor = '''

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = NextItNetModel(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    dataset_name = dataset_name if dataset_name is not None else _DATASETS[0]
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    meta = _metas[dataset_name]
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = _models[dataset_name].score_candidates(seq, items)
    return [float(value) for value in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(header + model_source + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
