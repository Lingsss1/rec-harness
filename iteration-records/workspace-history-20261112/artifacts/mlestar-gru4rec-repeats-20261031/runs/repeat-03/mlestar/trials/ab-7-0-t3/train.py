"""Single-model, per-category GRU4Rec with learned output bias."""
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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = 1
DROPOUT_RATE = 0.1
OUTPUT_DROPOUT = 0.0
LR = 0.001
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
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
            parts = line.split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('No eligible training histories')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for b in range(batch_size):
        history = user_train[random.choice(users)]
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('No unseen item available for negative sampling')
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[b, idx] = item
            pos[b, idx] = nxt
            sampled = random.randint(1, itemnum)
            while sampled in interacted:
                sampled = random.randint(1, itemnum)
            neg[b, idx] = sampled
            nxt = item
            idx -= 1
            if idx < 0:
                break
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1,
                 dropout_rate=0.1, output_dropout=0.2):
        super().__init__()
        if num_layers != 1:
            raise ValueError('This trajectory uses one GRU layer')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(output_dropout)
        self.state_norm = nn.LayerNorm(hidden_units)
        self.residual_gate = nn.Parameter(torch.full((hidden_units,), -1.0))
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch_size, width = sequence.shape
        mask = sequence.ne(0)
        lengths = mask.sum(dim=1)
        # Pack real histories from the start, then restore left-padded positions.
        positions = torch.arange(width, device=sequence.device).unsqueeze(0)
        starts = width - lengths
        compact_indices = (positions + starts.unsqueeze(1)).clamp(max=width - 1)
        compact_sequence = sequence.gather(1, compact_indices)
        compact_mask = positions < lengths.unsqueeze(1)
        compact_sequence = compact_sequence.masked_fill(~compact_mask, 0)
        raw_emb = self.item_emb(compact_sequence)
        emb = self.dropout(raw_emb)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.clamp(min=1).detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        compact_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=width,
        )
        gate = torch.sigmoid(self.residual_gate)
        compact_features = self.state_norm(compact_out) + gate * raw_emb
        restore_indices = (positions - starts.unsqueeze(1)).clamp(min=0)
        out = compact_features.gather(
            1, restore_indices.unsqueeze(-1).expand(batch_size, width, self.hidden_units),
        )
        out = out * mask.unsqueeze(-1).to(out.dtype)
        return self.output_dropout(out)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        emb = self.item_emb(items)
        bias = self.item_bias(items).squeeze(-1)
        if features.ndim == 2 and items.ndim == 1:
            return features @ emb.transpose(0, 1) + bias
        if items.ndim == features.ndim:
            features = features.unsqueeze(-2)
        return (features * emb).sum(dim=-1) + bias

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

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


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos != 0
    return torch.nn.functional.softplus(negative[mask] - positive[mask]).mean()


def _history_to_tensor(history, maxlen, device):
    sequence = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    return torch.as_tensor(sequence[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            required = min(99, itemnum - len(excluded - {0}))
            while len(candidates) < required + 1:
                negative = rng.randint(1, itemnum)
                if negative not in excluded:
                    excluded.add(negative)
                    candidates.append(negative)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(len(users), 1) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE, OUTPUT_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            batch = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(x, dtype=torch.long, device=device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val={metrics}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('Training completed without validation checkpoint')
    metadata = {
        'itemnum': itemnum,
        'maxlen': MAXLEN,
        'hidden_units': HIDDEN_UNITS,
        'num_layers': NUM_LAYERS,
        'dropout_rate': DROPOUT_RATE,
        'output_dropout': OUTPUT_DROPOUT,
        'item_bias': True,
        'architecture': 'tied_gru4rec_output_bias_v1',
    }
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print(f'[{dataset_name}] best_epoch={best_epoch} best_val_HR@10={best_hr:.4f}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    script = r'''"""One jointly trained GRU4Rec checkpoint per category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _load_model_class():
    global _model_class
    if _model_class is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('gru4rec_bias_training_definition', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.GRU4RecModel
    return _model_class


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in _DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    if not meta.get('item_bias'):
        raise RuntimeError('Checkpoint architecture does not include output bias')
    model = _load_model_class()(
        itemnum=meta['itemnum'],
        hidden_units=meta['hidden_units'],
        num_layers=meta['num_layers'],
        dropout_rate=meta['dropout_rate'],
        output_dropout=meta['output_dropout'],
    ).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset_name = dataset_name or _DATASETS[0]
    _ensure_loaded(dataset_name)
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = meta['maxlen']
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(value) for value in scores.cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    write_predict_script(WORKING_DIR)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
