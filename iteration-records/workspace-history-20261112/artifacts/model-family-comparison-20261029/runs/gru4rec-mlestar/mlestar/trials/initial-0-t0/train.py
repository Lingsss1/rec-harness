"""One GRU4Rec model per category, trained with multi-negative BPR-max."""
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
from torch import nn
import torch.nn.functional as F

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
PREPROCESSING_SEED = 0
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
SCORE_REGULARIZATION = 0.1
CHUNK_SIZE = 1024
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_SAMPLE_CACHE = {}


def seed_training():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as handle:
        for line in handle:
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
        with open(stats_path) as handle:
            stats = json.load(handle)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((i for split in (train, valid) for items in split.values() for i in items), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    """Shifted causal targets; negatives exclude training interactions only."""
    key = id(user_train)
    if key not in _SAMPLE_CACHE:
        entries = []
        for user, history in user_train.items():
            if len(history) < 2:
                continue
            excluded = np.asarray(sorted(set(history)), dtype=np.int64)
            if len(excluded) >= itemnum:
                continue
            entries.append((user, history, excluded))
        _SAMPLE_CACHE[key] = (user_train, entries)
    entries = _SAMPLE_CACHE[key][1]
    if not entries:
        raise RuntimeError('No eligible training histories with available negatives')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
    for row in range(batch_size):
        _, history, excluded = random.choice(entries)
        count = min(maxlen, len(history) - 1)
        seq[row, -count:] = history[-count - 1:-1]
        pos[row, -count:] = history[-count:]
        sampled = np.random.randint(1, itemnum + 1, size=(count, NUM_NEGATIVES), dtype=np.int64)
        rejected = np.isin(sampled, excluded)
        while rejected.any():
            sampled[rejected] = np.random.randint(1, itemnum + 1, size=int(rejected.sum()))
            rejected = np.isin(sampled, excluded)
        neg[row, -count:, :] = sampled
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
        outputs, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return outputs

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        while features.ndim < embeddings.ndim:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    def encode(self, seq):
        outputs = self.log2feats(seq)
        positions = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        return outputs[torch.arange(seq.shape[0], device=seq.device), last]

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


class ModelView(nn.Module):
    """Expose the same causal feature and scoring interface for diagnostics."""
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


def bpr_sequence_loss(model, seq, pos, neg, backward=False):
    """Backpropagate count-weighted chunks, retaining only the shared GRU graph.

    The training caller zeros gradients before this call and steps once after it.
    The default returns a differentiable scalar for diagnostic consumers.
    """
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    negatives = neg[mask]
    count = targets.numel()
    if not count:
        raise RuntimeError('Batch contains no valid shifted targets')
    total = features.new_zeros(())
    for start in range(0, count, CHUNK_SIZE):
        end = min(count, start + CHUNK_SIZE)
        h = features[start:end]
        positive_scores = model.predict_score(h, targets[start:end])
        negative_scores = model.predict_score(h, negatives[start:end])
        log_weights = F.log_softmax(negative_scores, dim=-1)
        ranking = -torch.logsumexp(
            log_weights + F.logsigmoid(positive_scores[:, None] - negative_scores), dim=-1)
        regularizer = SCORE_REGULARIZATION * (
            log_weights.exp() * negative_scores.square()).sum(dim=-1)
        weighted_loss = (ranking + regularizer).sum() / count
        if backward:
            weighted_loss.backward(retain_graph=end < count)
            total = total + weighted_loss.detach()
        else:
            total = total + weighted_loss
    return total


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    # A private RNG fixes users and unique negatives without touching training RNG.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        available = itemnum - len([x for x in excluded if 1 <= x <= itemnum])
        required = min(99, available)
        candidates = [target]
        while len(candidates) < required + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        history = user_train[user][-MAXLEN:]
        if history:
            sequence[-len(history):] = history
        cases.append((sequence, candidates))
    return cases


def evaluate_val(model, cases, device):
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for sequence, candidates in cases:
            seq = torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)
            ids = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, ids).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += float(rank < 10)
            hr20 += float(rank < 20)
            ndcg10 += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
            ndcg20 += 1.0 / math.log2(rank + 2) if rank < 20 else 0.0
    model.train(was_training)
    n = max(1, len(cases))
    return {'HR@10': hr10 / n, 'HR@20': hr20 / n,
            'NDCG@10': ndcg10 / n, 'NDCG@20': ndcg20 / n}


def train_one_dataset(dataset_name, device, data_dir):
    seed_training()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr10 = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq_t, pos_t, neg_t, backward=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr10:.5f} val_NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = {'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
            'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
            'num_negatives': NUM_NEGATIVES, 'score_regularization': SCORE_REGULARIZATION,
            'best_epoch': best_epoch, 'seed': SEED, 'preprocessing_seed': PREPROCESSING_SEED}
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as handle:
        pickle.dump(meta, handle)
    _SAMPLE_CACHE.clear()
    print(f'[{dataset_name}] selected epoch={best_epoch} val_HR@10={best_hr10:.5f}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""Route each category to its single validation-selected GRU4Rec model."""
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


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location(
        'trained_gru4rec_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, ds + '_meta.pkl'), 'rb') as handle:
            meta = pickle.load(handle)
        model = module.GRU4RecModel(
            itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
            num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, ds + '_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError('Unknown dataset_name: ' + str(ds))
    model = _models[ds]
    meta = _metas[ds]
    if not candidates:
        return []
    maxlen = int(meta['maxlen'])
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
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(script)


def main():
    if NUM_EPOCHS < 1:
        raise ValueError('Epoch limit must be positive')
    torch.set_num_threads(4)
    seed_training()
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
