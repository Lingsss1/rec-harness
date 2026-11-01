"""Per-category BERT4Rec with shared-weight dropout-view contrastive training."""
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
import torch.nn as nn
import torch.nn.functional as F

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('BERT4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('BERT4REC_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('BERT4REC_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('BERT4REC_HEADS', '2'))
DROPOUT_RATE = 0.2
LR = float(os.environ.get('BERT4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('BERT4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('BERT4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('BERT4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
TEMPERATURE = 0.2
CONTRASTIVE_WEIGHT = 0.05


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as handle:
            stats = json.load(handle)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, items in user_train.items() if len(items) >= 2 and len(set(items)) < itemnum]
        self.seen = {u: set(user_train[u]) for u in self.users}
        if not self.users:
            raise RuntimeError('No eligible users for masked-target training')

    def sample(self, batch_size, maxlen):
        sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
        positives = np.zeros_like(sequences)
        negatives = np.zeros_like(sequences)
        for row in range(batch_size):
            user = random.choice(self.users)
            items = self.user_train[user][-maxlen:]
            sequences[row, -len(items):] = items
            eligible = range(maxlen - len(items), maxlen)
            selected = [i for i in eligible if random.random() < 0.2]
            if not selected:
                selected = [random.choice(eligible)]
            if random.random() < 0.5 and maxlen - 1 not in selected:
                selected.append(maxlen - 1)
            for column in selected:
                positives[row, column] = sequences[row, column]
                sequences[row, column] = self.itemnum + 1
                negative = random.randint(1, self.itemnum)
                while negative in self.seen[user]:
                    negative = random.randint(1, self.itemnum)
                negatives[row, column] = negative
        return sequences, positives, negatives


# Preserve the family-specific diagnostic sampling interface.
def sample_batch(user_train, itemnum, batch_size, maxlen):
    return TrainingSampler(user_train, itemnum).sample(batch_size, maxlen)


def _history_to_tensor(history, maxlen, device):
    array = np.zeros(maxlen, dtype=np.int64)
    items = history[-maxlen:]
    if items:
        array[-len(items):] = items
    return torch.as_tensor(array, dtype=torch.long, device=device).unsqueeze(0)


class BERT4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2,
                 num_heads=2, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_token = itemnum + 1
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=num_heads,
            dim_feedforward=hidden_units * 4, dropout=dropout_rate,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks,
                                             enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        padding = seq.eq(0)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if features.ndim == embeddings.ndim:
            return (features * embeddings).sum(dim=-1)
        if features.ndim == 2 and embeddings.ndim == 3:
            return (features.unsqueeze(1) * embeddings).sum(dim=-1)
        raise ValueError('Incompatible feature and candidate dimensions')

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        # Left padding is retained; prediction always reads the appended mask.
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        features = self.log2feats(prediction_seq)[:, -1, :]
        if candidates.ndim == 1:
            return (self.item_emb(candidates) @ features[0]) if seq.shape[0] == 1 else features @ self.item_emb(candidates).T
        return self.predict_score(features, candidates)


class ModelView(nn.Module):
    """Expose the same encoder and scoring primitives for diagnostics."""
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


def bpr_sequence_loss(model, seq, pos, neg):
    # Both stochastic views use exactly the same parameters and masked inputs.
    first = model.log2feats(seq)
    second = model.log2feats(seq)
    selected = pos.ne(0)
    positive_embeddings = model.item_emb(pos[selected])
    negative_embeddings = model.item_emb(neg[selected])
    ranking = []
    for features in (first, second):
        margin = (features[selected] * (positive_embeddings - negative_embeddings)).sum(dim=-1)
        ranking.append(F.softplus(-margin).mean())
    recommendation_loss = 0.5 * (ranking[0] + ranking[1])

    # Pool supervised mask positions only, excluding all padding.
    mask = selected.unsqueeze(-1).to(first.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    first_view = F.normalize((first * mask).sum(dim=1) / denominator, dim=-1)
    second_view = F.normalize((second * mask).sum(dim=1) / denominator, dim=-1)
    logits = first_view @ second_view.T / TEMPERATURE
    identical = seq[:, None, :].eq(seq[None, :, :]).all(dim=-1)
    diagonal = torch.eye(seq.shape[0], device=seq.device, dtype=torch.bool)
    logits = logits.masked_fill(identical & ~diagonal, float('-inf'))
    labels = torch.arange(seq.shape[0], device=seq.device)
    contrastive_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return recommendation_loss + CONTRASTIVE_WEIGHT * contrastive_loss


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    # Local RNG keeps checkpoint comparisons fixed and training RNG untouched.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        if itemnum - len(excluded - {0}) < 99:
            raise RuntimeError('Fewer than 99 unique validation negatives available')
        candidates = [target]
        while len(candidates) < 100:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                excluded.add(candidate)
                candidates.append(candidate)
        cases.append((user, candidates))
    return cases


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200, cases=None):
    if cases is None:
        cases = make_validation_cases(user_train, user_valid, itemnum, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for user, candidates in cases:
            seq = _history_to_tensor(user_train[user], maxlen, device)
            items = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for cutoff in (10, 20):
                if rank < cutoff:
                    totals[f'HR@{cutoff}'] += 1.0
                    totals[f'NDCG@{cutoff}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    sampler = TrainingSampler(user_train, itemnum)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation users: {dataset_name}')
    model = BERT4RecModel(itemnum, HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS,
                          DROPOUT_RATE, MAXLEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    checkpoint_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sampler.sample(BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss for {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum,
                                   device, MAXLEN, cases=cases)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} loss={total_loss / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr10:.4f} val_NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    metadata = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                    num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                    dropout_rate=DROPOUT_RATE, best_epoch=best_epoch,
                    contrastive_temperature=TEMPERATURE,
                    contrastive_weight=CONTRASTIVE_WEIGHT)
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as handle:
        pickle.dump(metadata, handle)
    print(f'[{dataset_name}] best validation HR@10={best_hr10:.4f}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""Single-checkpoint per-category BERT4Rec inference."""
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
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('bert4rec_contrastive_architecture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.BERT4RecModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    model = _load_model_class()(
        itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
        num_blocks=meta['num_blocks'], num_heads=meta['num_heads'],
        dropout_rate=meta['dropout_rate'], maxlen=meta['maxlen'],
    ).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if dataset_name is None:
        dataset_name = DATASETS[0]
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    array = [0] * maxlen
    items = list(history[-maxlen:])
    if items:
        array[-len(items):] = items
    sequence = torch.tensor([array], dtype=torch.long, device=_device)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(sequence, candidate_tensor).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(script)


def main():
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    print(f'BERT4Rec dropout-view training on {device}', flush=True)
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    print(f'Macro-average best local validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
