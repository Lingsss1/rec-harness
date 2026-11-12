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
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 256
CHUNK_SIZE = 512
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                data[int(fields[0])].append(int(fields[1]))
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
        itemnum = max((item for split in (train, valid) for history in split.values() for item in history), default=0)
    return train, valid, itemnum


class TrainingHistories(dict):
    def __init__(self, data):
        super().__init__(data)
        self.eligible_users = [u for u, history in data.items() if len(history) >= 2]


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = getattr(user_train, 'eligible_users', None)
    if users is None:
        users = [u for u, history in user_train.items() if len(history) >= 2]
    if not users or itemnum < 2:
        raise RuntimeError('Need training histories of length >= 2 and at least two items')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        inputs = history[:-1][-maxlen:]
        targets = history[1:][-maxlen:]
        seq[row, -len(inputs):] = inputs
        pos[row, -len(targets):] = targets
    # Uniform draws with replacement, rejecting only the current target.
    # Other history items deliberately remain eligible for this objective.
    neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
    valid = pos != 0
    targets = pos[valid, None]
    draws = np.random.randint(1, itemnum + 1, size=(len(targets), NUM_NEGATIVES), dtype=np.int64)
    collision = draws == targets
    while collision.any():
        draws[collision] = np.random.randint(1, itemnum + 1, size=int(collision.sum()))
        collision = draws == targets
    neg[valid] = draws
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        embeddings = self.dropout(self.item_emb(sequence))
        output, _ = self.gru(embeddings)
        return output

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def encode(self, seq):
        output = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        return output[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        output = self.log2feats(seq)
        positives = (output * self.item_emb(pos)).sum(-1)
        if neg.ndim == seq.ndim + 1:
            negatives = (output.unsqueeze(-2) * self.item_emb(neg)).sum(-1)
        else:
            negatives = (output * self.item_emb(neg)).sum(-1)
        return positives, negatives

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


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

    def forward(self, seq, pos, neg):
        return self.model(seq, pos, neg)


def sampled_ce_backward(model, seq, pos, neg):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    negatives = neg[mask]
    count = targets.numel()
    if count == 0:
        raise RuntimeError('Training batch has no valid targets')
    # Accumulate output-embedding gradients chunk by chunk. A detached leaf
    # collects the exact feature gradients, followed by one recurrent backward.
    leaf = features.detach().requires_grad_(True)
    total_loss = 0.0
    for start in range(0, count, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, count)
        h = leaf[start:stop]
        y = targets[start:stop]
        n = negatives[start:stop]
        positive = (h * model.item_emb(y)).sum(-1, keepdim=True)
        negative = torch.einsum('nd,nkd->nk', h, model.item_emb(n))
        logits = torch.cat([positive, negative], dim=1)
        labels = torch.zeros(stop - start, dtype=torch.long, device=h.device)
        loss = F.cross_entropy(logits, labels, reduction='sum') / count
        loss.backward()
        total_loss += float(loss.detach())
    features.backward(leaf.grad)
    return total_loss


def _history_to_tensor(history, maxlen, device):
    sequence = np.zeros(maxlen, dtype=np.int64)
    recent = history[-maxlen:]
    if recent:
        sequence[-len(recent):] = recent
    return torch.as_tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    # Local RNG keeps checkpoint comparisons fixed and independent of training.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {0, target}
        candidates = [target]
        target_count = 1 + min(99, itemnum - sum(1 for x in excluded if 1 <= x <= itemnum))
        while len(candidates) < target_count:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        history = user_train[user][-maxlen:]
        sequence = [0] * (maxlen - len(history)) + history
        cases.append((sequence, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hits = 0.0
    ndcg = 0.0
    for sequence, candidates in cases:
        seq = torch.tensor([sequence], dtype=torch.long, device=device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items)
        # Stable target-first handling for exact ties.
        rank = int((scores[1:] > scores[0]).sum().item())
        if rank < 10:
            hits += 1.0
            ndcg += 1.0 / math.log2(rank + 2)
    return {'HR@10': hits / max(1, len(cases)), 'NDCG@10': ndcg / max(1, len(cases))}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    user_train = TrainingHistories(user_train)
    if not user_train.eligible_users or itemnum < 2:
        raise RuntimeError('Empty or insufficient dataset: ' + dataset_name)
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            total_loss += sampled_ce_backward(model, seq, pos, neg)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} '
                  f'validation HR@10={hr:.5f} NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = {'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
            'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
            'objective': 'uniform_sampled_cross_entropy', 'num_negatives': NUM_NEGATIVES,
            'seed': SEED, 'best_epoch': best_epoch}
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact training architecture to keep inference self-contained.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    architecture = inspect.getsource(GRU4RecModel)
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
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'],
                         meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    if not candidates:
        return []
    _ensure_loaded(ds)
    model = _models[ds]
    meta = _metas[ds]
    maxlen = int(meta['maxlen'])
    recent = list(history[-maxlen:])
    sequence = [0] * (maxlen - len(recent)) + recent
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(value) for value in scores.cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n' + runtime)


def main():
    torch.set_num_threads(8)
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Single-model GRU4Rec sampled-CE training on {device}', flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    write_predict_script(WORKING_DIR)
    print(f'Macro-average best validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
