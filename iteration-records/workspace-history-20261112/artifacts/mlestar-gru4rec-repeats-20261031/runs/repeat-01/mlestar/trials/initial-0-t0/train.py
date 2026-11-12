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
NUM_EPOCHS = max(1, min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
ADDITIONAL_NEGATIVES = 32
NEGATIVE_COUNT = 1 + ADDITIONAL_NEGATIVES


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
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
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
    if itemnum <= 0:
        itemnum = max((item for split in (train, valid) for items in split.values() for item in items), default=0)
    return train, valid, itemnum


# Cache only the currently supplied training dictionary and its exclusion sets.
_sampling_cache = None


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    global _sampling_cache
    if (_sampling_cache is None or _sampling_cache[0] is not user_train
            or _sampling_cache[1] != itemnum):
        excluded = {u: np.asarray(sorted(set(items)), dtype=np.int64)
                    for u, items in user_train.items() if len(items) >= 2}
        users = [u for u in excluded if len(excluded[u]) < itemnum]
        _sampling_cache = (user_train, itemnum, users, excluded)
    _, _, users, excluded = _sampling_cache
    if not users:
        raise RuntimeError('No training users with causal targets and eligible negatives')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        forbidden = excluded[user]
        shape = (length, num_negatives)
        if len(forbidden) > itemnum // 2:
            allowed = np.setdiff1d(np.arange(1, itemnum + 1), forbidden, assume_unique=True)
            draws = np.random.choice(allowed, size=shape, replace=True)
        else:
            draws = np.random.randint(1, itemnum + 1, size=shape)
            invalid = np.isin(draws, forbidden)
            while invalid.any():
                draws[invalid] = np.random.randint(1, itemnum + 1, size=int(invalid.sum()))
                invalid = np.isin(draws, forbidden)
        neg[row, -length:] = draws
    # Preserve the supplied diagnostic sampler's default shapes.
    if num_negatives == 1:
        neg = neg[..., 0]
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        out, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return out

    def encode(self, seq):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        pos_scores = self.predict_score(features, pos)
        if neg.ndim == pos.ndim + 1:
            neg_scores = self.predict_score(features.unsqueeze(-2), neg)
        else:
            neg_scores = self.predict_score(features, neg)
        return pos_scores, neg_scores

    def score_candidates(self, seq, candidates):
        features = self.encode(seq)
        if candidates.ndim == 1:
            return self.item_emb(candidates) @ features[0]
        return self.predict_score(features.unsqueeze(1), candidates)


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

    def forward(self, *args):
        return self.model(*args)


def bpr_max_loss(pos_scores, neg_scores, coefficient=0.1):
    log_weights = torch.log_softmax(neg_scores.float(), dim=-1)
    log_win = F.logsigmoid(pos_scores.float()[:, None] - neg_scores.float())
    ranking = -torch.logsumexp(log_weights + log_win, dim=-1)
    regularizer = (log_weights.exp() * neg_scores.float().square()).sum(-1)
    return (ranking + coefficient * regularizer).mean()


def bpr_sequence_loss(model, seq, pos, neg):
    # Select valid positions before materializing the multi-negative embeddings.
    valid = pos != 0
    features = model.log2feats(seq)[valid]
    positives = pos[valid]
    negatives = neg[valid]
    if negatives.ndim == 1:
        negatives = negatives[:, None]
    pos_scores = model.predict_score(features, positives)
    neg_scores = model.predict_score(features[:, None, :], negatives)
    return bpr_max_loss(pos_scores, neg_scores)


def _history_to_tensor(history, maxlen, device):
    array = np.zeros((1, maxlen), dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        array[0, -len(tail):] = tail
    return torch.as_tensor(array, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # Reset evaluation's own RNG on every call; never consume training randomness.
    rng = random.Random(42)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            count = min(99, itemnum - sum(1 <= i <= itemnum for i in excluded))
            while len(candidates) < count + 1:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train()
    return {key: value / max(1, len(users)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty training data for {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    checkpoint = os.path.join(working, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN, NEGATIVE_COUNT)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss for {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.6f} validation={json.dumps(metrics)}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                selected_epoch=best_epoch, validation_hr10=best_hr,
                negative_count=NEGATIVE_COUNT, loss='bpr-max', coefficient=0.1)
    with open(os.path.join(working, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] selected_epoch={best_epoch} validation_HR@10={best_hr:.6f}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture, avoiding import-time training dependencies.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    architecture = inspect.getsource(GRU4RecModel)
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_loaded(dataset_name):
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset_name = dataset_name or DATASETS[0]
    _ensure_loaded(dataset_name)
    if len(candidates) == 0:
        return []
    model = _models[dataset_name]
    maxlen = _metas[dataset_name]['maxlen']
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(x) for x in scores.cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n' + runtime)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro-average selected validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
