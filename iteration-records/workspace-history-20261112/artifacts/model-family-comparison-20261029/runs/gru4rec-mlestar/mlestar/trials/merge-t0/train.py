"""Independent per-category GRU4Rec with memory-efficient BPR-max training."""
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
import torch.nn.functional as F

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
PREPROCESSING_SEED = 0
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
LR = 1e-3
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('SASREC_EPOCHS', '200')), int(os.environ.get('GRU4REC_EPOCHS', '200')))
EVAL_EVERY = 5
PATIENCE = 20
NUM_NEGATIVES = 64
LOSS_CHUNK = 1024
SCORE_REG = 0.1
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_SAMPLE_CACHE = None


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
        itemnum = max((i for split in (train, valid) for history in split.values() for i in history), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    """Causal shifted targets, with negatives excluding training interactions."""
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is None or _SAMPLE_CACHE[0] is not user_train or _SAMPLE_CACHE[1] != itemnum:
        entries = []
        for user, history in user_train.items():
            excluded = np.asarray(sorted(set(history)), dtype=np.int64)
            if len(history) >= 2 and len(excluded) < itemnum:
                entries.append((history, excluded))
        _SAMPLE_CACHE = (user_train, itemnum, entries)
    entries = _SAMPLE_CACHE[2]
    if not entries:
        raise RuntimeError('No eligible training histories with available negatives')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((batch_size, maxlen, NUM_NEGATIVES), dtype=np.int64)
    for row in range(batch_size):
        history, excluded = random.choice(entries)
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        draws = np.random.randint(1, itemnum + 1, size=(length, NUM_NEGATIVES))
        rejected = np.isin(draws, excluded)
        while rejected.any():
            draws[rejected] = np.random.randint(1, itemnum + 1, size=int(rejected.sum()))
            rejected = np.isin(draws, excluded)
        neg[row, -length:] = draws
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
        output, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return output

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        while features.ndim < embeddings.ndim:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    def encode(self, seq):
        output = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        features = output[torch.arange(seq.size(0), device=seq.device), last]
        return features * seq.ne(0).any(dim=1, keepdim=True)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


class ModelView(nn.Module):
    """Expose the trained causal model through the diagnostic interface."""
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
    """Chunk output gradients and backpropagate through the GRU once.

    Default mode returns a differentiable scalar for diagnostic callers.
    Training mode accumulates both scoring and recurrent gradients before the
    caller clips gradients and performs a single optimizer step.
    """
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    negatives = neg[mask]
    count = targets.numel()
    if not count:
        raise RuntimeError('Batch contains no valid shifted targets')
    chunk_features = features.detach().requires_grad_(True) if backward else features
    total = features.new_zeros(())
    for start in range(0, count, LOSS_CHUNK):
        end = min(start + LOSS_CHUNK, count)
        h = chunk_features[start:end]
        positive_scores = model.predict_score(h, targets[start:end])
        negative_scores = model.predict_score(h, negatives[start:end])
        log_weights = F.log_softmax(negative_scores, dim=-1)
        ranking = -torch.logsumexp(
            log_weights + F.logsigmoid(positive_scores[:, None] - negative_scores), dim=-1)
        regularizer = SCORE_REG * (log_weights.exp() * negative_scores.square()).sum(dim=-1)
        weighted_loss = (ranking + regularizer).sum() / count
        if backward:
            weighted_loss.backward()
            total = total + weighted_loss.detach()
        else:
            total = total + weighted_loss
    if backward:
        features.backward(chunk_features.grad)
    return total


def make_validation_cases(user_train, user_valid, itemnum, num_users=200, maxlen=MAXLEN):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        available = itemnum - sum(1 for i in excluded if 1 <= i <= itemnum)
        required = min(99, available)
        candidates = [target]
        while len(candidates) < required + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        sequence = np.zeros(maxlen, dtype=np.int64)
        tail = user_train[user][-maxlen:]
        if tail:
            sequence[-len(tail):] = tail
        cases.append((sequence, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    was_training = model.training
    model.eval()
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    for sequence, candidates in cases:
        seq = torch.as_tensor(sequence[None], dtype=torch.long, device=device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        for k in (10, 20):
            if rank < k:
                totals[f'HR@{k}'] += 1.0
                totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(len(cases), 1) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    global _SAMPLE_CACHE
    seed_training()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                  num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    model = GRU4RecModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            total_loss += bpr_sequence_loss(model, seq, pos, neg, backward=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss.item() / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr:.5f} val_NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    metadata = dict(model_config=config, maxlen=MAXLEN, best_epoch=best_epoch,
                    validation_hr10=best_hr, seed=SEED, preprocessing_seed=PREPROCESSING_SEED,
                    num_negatives=NUM_NEGATIVES, score_regularization=SCORE_REG)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as handle:
        pickle.dump(metadata, handle)
    _SAMPLE_CACHE = None
    print(f'[{dataset_name}] selected_epoch={best_epoch} val_HR@10={best_hr:.5f}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    script = r'''"""Route each category to its single selected GRU4Rec checkpoint."""
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
_model_class = None


def _load_model_class():
    global _model_class
    if _model_class is None:
        spec = importlib.util.spec_from_file_location(
            'trained_gru4rec_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.GRU4RecModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = _load_model_class()(**meta['model_config']).to(_device)
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
        raise ValueError('Unknown dataset_name: ' + str(ds))
    if not candidates:
        return []
    _ensure_loaded(ds)
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = _models[ds].score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(script)


def main():
    if NUM_EPOCHS < 1 or MAXLEN < 1:
        raise ValueError('Epoch limit and maximum history length must be positive')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_training()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    write_predict_script(WORKING_DIR)
    print(f'GRU4Rec BPR-max K={NUM_NEGATIVES} device={device}', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
