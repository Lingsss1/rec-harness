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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 64
DILATIONS = [1, 2, 4, 8, 16, 1, 2, 4]
KERNEL_SIZE = 3
DROPOUT_RATE = 0.0
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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
    if not itemnum:
        itemnum = max((i for split in (train, valid) for items in split.values() for i in items), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(seqs)
    negatives = np.zeros_like(seqs)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('No unseen negative item for a training user')
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seqs[row, idx] = item
            positives[row, idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            negatives[row, idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
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
        y = self._causal_conv(self.conv1, self.norm1(x).transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self._causal_conv(self.conv2, self.norm2(y).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128, dilations=None):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 1, 2, 4]
        self.dilations = list(dilations)
        if len(self.dilations) != num_blocks:
            raise ValueError('Dilation schedule must specify every block')
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, d, kernel_size, dropout_rate) for d in self.dilations
        ])
        self.residual_scale = nn.Parameter(torch.full((num_blocks,), 0.5))
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        nonpadding = seq.ne(0)
        positions = nonpadding.long().cumsum(dim=1) * nonpadding.long()
        mask = nonpadding.unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)) * mask
        for index, block in enumerate(self.blocks):
            residual = block(x) - x
            x = (x + self.residual_scale[index] * residual) * mask
        return self.norm(x) * mask

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        final = features[torch.arange(seq.shape[0], device=seq.device), last]
        if candidates.ndim == 1:
            scores = self.predict_score(final[:, None, :], candidates[None, :])
        else:
            scores = self.predict_score(final[:, None, :], candidates)
        return scores[0] if seq.shape[0] == 1 else scores


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
    return nn.functional.softplus(negative - positive)[mask].mean()


def build_validation(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        if itemnum - len(excluded - {0}) < 99:
            raise RuntimeError('Validation requires 99 unique unseen negatives')
        candidates = [target]
        while len(candidates) < 100:
            negative = rng.randint(1, itemnum)
            if negative not in excluded:
                excluded.add(negative)
                candidates.append(negative)
        seq = [0] * maxlen
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        examples.append((seq, candidates))
    return examples


@torch.no_grad()
def evaluate_val(model, examples, device):
    model.eval()
    hits = 0
    ndcg = 0.0
    for seq, candidates in examples:
        seq_t = torch.tensor([seq], dtype=torch.long, device=device)
        cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        if rank < 10:
            hits += 1
            ndcg += 1.0 / math.log2(rank + 2)
    n = max(1, len(examples))
    return {'HR@10': hits / n, 'NDCG@10': ndcg / n}


def train_one_dataset(dataset_name, device, data_dir):
    seed_all()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    examples = build_validation(user_train, user_valid, itemnum, MAXLEN)
    if not examples:
        raise RuntimeError('No validation users: ' + dataset_name)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=len(DILATIONS),
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE,
                maxlen=MAXLEN, dilations=list(DILATIONS))
    model = NextItNetModel(**meta).to(device)
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
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, examples, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} local_val_HR@10={hr:.5f} NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    model_source = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    predictor = r'''

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for dataset in _DATASETS:
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = NextItNetModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if not candidates:
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
        scores = model.score_candidates(seq, items)
    return [float(s) for s in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + model_source + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    write_predict_script(WORKING_DIR)
    print(f'Macro local validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
