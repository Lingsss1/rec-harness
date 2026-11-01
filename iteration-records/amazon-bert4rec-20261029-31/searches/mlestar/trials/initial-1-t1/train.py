"""Per-category BERT4Rec with random-prefix and terminal-mask training."""
from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('BERT4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('BERT4REC_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('BERT4REC_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('BERT4REC_HEADS', '2'))
DROPOUT_RATE = float(os.environ.get('BERT4REC_DROPOUT', '0.2'))
LR = float(os.environ.get('BERT4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('BERT4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('BERT4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('BERT4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
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
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
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
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, h in user_train.items() if len(h) >= 2]
    if not users:
        raise RuntimeError('no eligible training users')
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        end = len(history) if random.random() < 0.5 else random.randint(2, len(history))
        window = history[max(0, end - maxlen):end]
        start = maxlen - len(window)
        sequences[row, start:] = window
        if random.random() < 0.5:
            selected = [maxlen - 1]
        else:
            selected = [j for j in range(start, maxlen) if random.random() < 0.2]
            if not selected:
                selected = [random.randrange(start, maxlen)]
        observed = set(history)
        if len(observed) >= itemnum:
            raise RuntimeError('no unobserved negative available')
        for j in selected:
            positives[row, j] = sequences[row, j]
            sequences[row, j] = itemnum + 1
            negative = random.randint(1, itemnum)
            while negative in observed:
                negative = random.randint(1, itemnum)
            negatives[row, j] = negative
    return sequences, positives, negatives


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


class BERT4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.2, maxlen=128):
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
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device)[None, :].expand(batch, -1) + 1
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        padding = seq.eq(0)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        if features.ndim == 2 and items.ndim == 1:
            return features @ embeddings.T
        if features.ndim == 3 and items.ndim == 1:
            return features @ embeddings.T
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        features = self.encode_sequence(prediction_seq)[:, -1, :]
        return (features @ self.item_emb(candidates).T).squeeze(0)


class ModelView:
    """Diagnostic access to the same trained BERT4Rec network."""
    def __init__(self, model):
        self.model = model
        self.item_emb = model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    selected = pos.ne(0)
    return torch.nn.functional.softplus(negative[selected] - positive[selected]).mean()


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        history = user_train[user]
        excluded = set(history) | {target, 0}
        candidates = [target]
        count = min(99, itemnum - len({i for i in excluded if 1 <= i <= itemnum}))
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                candidates.append(item)
                excluded.add(item)
        cases.append((history, candidates))
    return cases


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200, cases=None):
    if cases is None:
        cases = make_validation_cases(user_train, user_valid, itemnum, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for history, candidates in cases:
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not train:
        raise RuntimeError(f'empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    model = BERT4RecModel(itemnum, HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS,
                          DROPOUT_RATE, MAXLEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError(f'no validation users: {dataset_name}')
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss in {dataset_name} at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, train, valid, itemnum, device, MAXLEN, cases=cases)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={loss_sum / NUM_BATCHES:.4f} '
                  f'val HR@10={hr:.4f} NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE)
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    return best_hr


def write_predict_script(working_dir):
    script = r'''"""Route predictions to one BERT4Rec checkpoint per category."""
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
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location(
        'bert4rec_prefix_training', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ds in DATASETS:
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = module.BERT4RecModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f'Unknown dataset: {ds}')
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
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
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Random-prefix BERT4Rec on {device}; seed={SEED}', flush=True)
    scores = []
    for dataset in DATASETS:
        scores.append(train_one_dataset(dataset, device, DATA_DIR))
    write_predict_script(WORKING_DIR)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
