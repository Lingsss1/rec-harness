"""Single-model GRU4Rec merge: pairwise BPR and hard-negative BPR-max."""
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
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


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
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    itemnum = 0
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((max(items, default=0) for split in (train, valid) for items in split.values()), default=0)
    return train, valid, itemnum


_SAMPLER_CACHE = {}


def _sampler_data(user_train, itemnum):
    key = (id(user_train), itemnum)
    cached = _SAMPLER_CACHE.get(key)
    if cached is None or cached[0] is not user_train:
        seen = {u: set(h) for u, h in user_train.items() if len(h) >= 2}
        users = [u for u in seen if len(seen[u]) < itemnum]
        if not users:
            raise RuntimeError('No eligible training users')
        cached = (user_train, users, seen)
        _SAMPLER_CACHE[key] = cached
    return cached[1], cached[2]


def _sample_negatives(seen, itemnum, count):
    accepted = []
    for _ in range(8):
        draws = np.random.randint(1, itemnum + 1, size=max(16, 2 * (count - len(accepted))))
        accepted.extend(int(x) for x in draws if int(x) not in seen)
        if len(accepted) >= count:
            return np.asarray(accepted[:count], dtype=np.int64)
    eligible = np.asarray([i for i in range(1, itemnum + 1) if i not in seen], dtype=np.int64)
    if not len(eligible):
        raise RuntimeError('No eligible negative items')
    remaining = np.random.choice(eligible, size=count - len(accepted), replace=True)
    return np.asarray(accepted + remaining.tolist(), dtype=np.int64)


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    users, seen_by_user = _sampler_data(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    if num_negatives == 1:
        neg = np.zeros_like(seq)
    else:
        neg = np.empty((batch_size, num_negatives), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        if num_negatives == 1:
            neg[row, -length:] = _sample_negatives(seen_by_user[user], itemnum, length)
        else:
            neg[row] = _sample_negatives(seen_by_user[user], itemnum, num_negatives)
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
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, seq):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        positive = (features * self.item_emb(pos)).sum(-1)
        if neg.shape == seq.shape:
            negative = (features * self.item_emb(neg)).sum(-1)
        else:
            negative = torch.einsum('bld,bkd->blk', features, self.item_emb(neg))
        return positive, negative

    def score_candidates(self, seq, candidates):
        scores = self.predict_score(self.encode(seq), candidates)
        return scores[0] if seq.size(0) == 1 else scores


def bpr_sequence_loss(model, seq, pos, neg, max_weight=0.75):
    positive, negative = model(seq, pos, neg)
    mask = pos != 0
    p = positive[mask]
    if negative.ndim == 2:
        return F.softplus(negative[mask] - p).mean()
    n = negative[mask]
    # Softer detached hardness weights spread supervision across plausible
    # competitors while retaining a stronger BPR-max contribution.
    pairwise = F.softplus(n - p[:, None]).mean(dim=-1)
    log_weights = F.log_softmax(n.detach() / 2.0, dim=-1)
    weights = log_weights.exp()
    log_win = torch.logsumexp(log_weights + F.logsigmoid(p[:, None] - n), dim=-1)
    hard = -log_win + 0.01 * (weights * n.square()).sum(dim=-1)
    return ((1.0 - max_weight) * pairwise + max_weight * hard).mean()


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
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
        attempts = 0
        while len(candidates) < 100 and attempts < 10000:
            item = rng.randint(1, itemnum)
            attempts += 1
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        if len(candidates) < 100:
            eligible = [i for i in range(1, itemnum + 1) if i not in excluded]
            candidates.extend(rng.sample(eligible, 100 - len(candidates)))
        sequence = [0] * maxlen
        tail = history[-maxlen:]
        if tail:
            sequence[-len(tail):] = tail
        cases.append((sequence, candidates))
    if not cases:
        raise RuntimeError('No validation users')
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hits = 0
    ndcg = 0.0
    for start in range(0, len(cases), 64):
        batch = cases[start:start + 64]
        seq = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
        candidates = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
        scores = model.predict_score(model.encode(seq), candidates).cpu().numpy()
        for row in scores:
            rank = int(np.flatnonzero(np.argsort(-row, kind='stable') == 0)[0])
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    model.train()
    return {'HR@10': hits / len(cases), 'NDCG@10': ndcg / len(cases)}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    negatives = 1 if dataset_name == 'Electronics' else 64
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    ckpt_path = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN, negatives)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} '
                  f'validation_HR@10={hr:.4f} NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                loss='bpr' if negatives == 1 else 'bpr-plus-bpr-max', negatives=negatives,
                training_seed=SEED, preprocessing_seed=0, best_epoch=best_epoch)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    _SAMPLER_CACHE.clear()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    model_source = inspect.getsource(GRU4RecModel)
    predictor = r'''
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
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if dataset_name is None:
        dataset_name = DATASETS[0]
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + model_source + '\n' + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    write_predict_script(WORKING_DIR)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
