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
NEGATIVE_COUNT = 33
BPR_MAX_COEFFICIENT = 0.1
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_sampling_cache = None


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
    itemnum = 0
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if itemnum <= 0:
        itemnum = max((item for split in (train, valid) for items in split.values() for item in items), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=1):
    global _sampling_cache
    if (_sampling_cache is None or _sampling_cache[0] is not user_train
            or _sampling_cache[1] != itemnum):
        excluded = {u: np.asarray(sorted(set(items)), dtype=np.int64)
                    for u, items in user_train.items() if len(items) >= 2}
        users = [u for u in excluded if len(excluded[u]) < itemnum]
        complements = {
            u: np.setdiff1d(np.arange(1, itemnum + 1), excluded[u], assume_unique=True)
            for u in users if len(excluded[u]) > itemnum // 2
        }
        _sampling_cache = (user_train, itemnum, users, excluded, complements)
    _, _, users, excluded, complements = _sampling_cache
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
        shape = (length, num_negatives)
        if user in complements:
            draws = np.random.choice(complements[user], size=shape, replace=True)
        else:
            draws = np.random.randint(1, itemnum + 1, size=shape)
            invalid = np.isin(draws, excluded[user])
            while invalid.any():
                draws[invalid] = np.random.randint(1, itemnum + 1, size=int(invalid.sum()))
                invalid = np.isin(draws, excluded[user])
        neg[row, -length:] = draws
    return seq, pos, neg[..., 0] if num_negatives == 1 else neg


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
        self.input_residual = nn.Linear(hidden_units, hidden_units, bias=False)
        self.residual_gate = nn.Parameter(torch.full((hidden_units,), -2.0))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.eye_(self.input_residual.weight)
        # A scoring-head intercept trained jointly with the recurrent network.
        # Preserve the backbone's initialization and subsequent RNG stream.
        self.item_bias = nn.Embedding.from_pretrained(
            torch.zeros(itemnum + 1, 1), freeze=False, padding_idx=0)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch_size, width = sequence.shape
        mask = sequence.ne(0)
        lengths = mask.sum(dim=1)
        positions = torch.arange(width, device=sequence.device).expand_as(sequence)
        # Move left-padded histories to the beginning before recurrence so that
        # GRU biases cannot accumulate hidden state on leading padding.
        offsets = width - lengths
        compact_indices = (positions + offsets[:, None]).clamp(max=width - 1)
        compact = sequence.gather(1, compact_indices)
        compact_mask = positions < lengths[:, None]
        compact = compact.masked_fill(~compact_mask, 0)
        embedded = self.item_emb(compact)
        out, _ = self.gru(self.dropout(embedded))
        out = out + torch.sigmoid(self.residual_gate) * self.input_residual(embedded)
        # Restore the original positions for causal per-position training.
        restore_indices = (positions - offsets[:, None]).clamp(min=0)
        out = out.gather(1, restore_indices.unsqueeze(-1).expand(batch_size, width, self.hidden_units))
        return out * mask.unsqueeze(-1)

    def encode(self, seq):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        features = out[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        return (features * self.item_emb(items)).sum(dim=-1) + self.item_bias(items).squeeze(-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        pos_scores = self.predict_score(features, pos)
        if neg.ndim == pos.ndim + 1:
            neg_scores = self.predict_score(features.unsqueeze(-2), neg)
        else:
            neg_scores = self.predict_score(features, neg)
        return pos_scores, neg_scores

    def score_candidates(self, seq, candidates):
        candidates = torch.as_tensor(candidates, dtype=torch.long, device=self.item_emb.weight.device)
        features = self.encode(seq)
        if candidates.ndim == 1:
            bias = self.item_bias(candidates).squeeze(-1)
            if features.size(0) == 1:
                return self.item_emb(candidates) @ features[0] + bias
            return features @ self.item_emb(candidates).transpose(0, 1) + bias
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def bpr_max_loss(pos_scores, neg_scores, coefficient=BPR_MAX_COEFFICIENT):
    log_weights = F.log_softmax(neg_scores.float(), dim=-1)
    log_win = F.logsigmoid(pos_scores.float()[:, None] - neg_scores.float())
    ranking = -torch.logsumexp(log_weights + log_win, dim=-1)
    regularizer = (log_weights.exp() * neg_scores.float().square()).sum(dim=-1)
    return (ranking + coefficient * regularizer).mean()


def bpr_sequence_loss(model, seq, pos, neg):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    positive_ids = pos[mask]
    negative_ids = neg[mask]
    if negative_ids.ndim == 1:
        negative_ids = negative_ids[:, None]
    count = features.size(0)
    if count == 0:
        return features.sum() * 0.0
    total = features.sum() * 0.0
    for start in range(0, count, 512):
        end = min(start + 512, count)
        hidden = features[start:end]
        positives = model.predict_score(hidden, positive_ids[start:end])
        negatives = model.predict_score(hidden[:, None, :], negative_ids[start:end])
        total = total + bpr_max_loss(positives, negatives) * ((end - start) / count)
    return total


def _history_to_tensor(history, maxlen, device):
    array = np.zeros((1, maxlen), dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        array[0, -len(tail):] = tail
    return torch.as_tensor(array, dtype=torch.long, device=device)


def build_validation_cases(user_train, user_valid, itemnum, maxlen=MAXLEN, num_users=200):
    rng = random.Random(42)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    if not users:
        raise RuntimeError('No validation users')
    sequences = np.zeros((len(users), maxlen), dtype=np.int64)
    candidates = np.zeros((len(users), 100), dtype=np.int64)
    for row, user in enumerate(users):
        history = user_train[user]
        target = user_valid[user][0]
        tail = history[-maxlen:]
        if tail:
            sequences[row, -len(tail):] = tail
        excluded = set(history) | {target, 0}
        available = itemnum - sum(1 <= i <= itemnum for i in excluded)
        if available < 99:
            raise RuntimeError('Validation requires 99 unique eligible negatives')
        items = [target]
        while len(items) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                items.append(item)
        candidates[row] = items
    return sequences, candidates


def evaluate_cases(model, cases, device):
    sequences, candidates = cases
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), 64):
            seq = torch.as_tensor(sequences[start:start + 64], dtype=torch.long, device=device)
            items = torch.as_tensor(candidates[start:start + 64], dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items)
            # Target occupies index zero, so stable descending ties favor it.
            ranks = (scores[:, 1:] > scores[:, :1]).sum(dim=1).cpu().tolist()
            for rank in ranks:
                for k in (10, 20):
                    if rank < k:
                        totals[f'HR@{k}'] += 1.0
                        totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / len(sequences) for key, value in totals.items()}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    cases = build_validation_cases(user_train, user_valid, itemnum, maxlen, num_users)
    return evaluate_cases(model, cases, device)


def train_one_dataset(dataset_name, device, data_dir):
    global _sampling_cache
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty training data for {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    cases = build_validation_cases(user_train, user_valid, itemnum)
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
            metrics = evaluate_cases(model, cases, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.6f} local_validation={json.dumps(metrics)}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                selected_epoch=best_epoch, validation_hr10=best_hr,
                negative_count=NEGATIVE_COUNT, loss='bpr-max',
                coefficient=BPR_MAX_COEFFICIENT)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] selected_epoch={best_epoch} local_validation_HR@10={best_hr:.6f}', flush=True)
    _sampling_cache = None
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
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
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro-average selected local validation HR@10={sum(scores) / len(scores):.6f}; host validation is authoritative.', flush=True)


if __name__ == '__main__':
    main()
