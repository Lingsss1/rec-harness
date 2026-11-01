"""One BERT4Rec per category, merging full-history and random-prefix masking."""
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
EVAL_EVERY = max(1, int(os.environ.get('BERT4REC_EVAL_EVERY', '5')))
PATIENCE = int(os.environ.get('BERT4REC_PATIENCE', '20'))
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


class TrainingHistories(dict):
    def __init__(self, histories, prefix_training=False):
        super().__init__(histories)
        self.prefix_training = prefix_training
        self.eligible_users = [u for u, h in histories.items() if len(h) >= 2]
        self.observed = {u: set(histories[u]) for u in self.eligible_users}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = getattr(user_train, 'eligible_users', None)
    if users is None:
        users = [u for u, h in user_train.items() if len(h) >= 2]
    if not users:
        raise RuntimeError('no eligible training users')
    prefix_training = getattr(user_train, 'prefix_training', False)
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        end = len(history)
        if prefix_training and random.random() >= 0.5:
            end = random.randint(2, len(history))
        window = history[max(0, end - maxlen):end]
        start = maxlen - len(window)
        sequences[row, start:] = window
        if prefix_training and random.random() < 0.5:
            selected = [maxlen - 1]
        else:
            selected = [j for j in range(start, maxlen) if random.random() < 0.2]
            if not selected:
                selected = [random.randrange(start, maxlen)]
            if not prefix_training and random.random() < 0.5 and maxlen - 1 not in selected:
                selected.append(maxlen - 1)
        observed = user_train.observed[user] if hasattr(user_train, 'observed') else set(history)
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
        device = self.item_emb.weight.device
        seq = torch.as_tensor(seq, dtype=torch.long, device=device)
        batch, length = seq.shape
        positions = torch.arange(1, length + 1, device=device).unsqueeze(0).expand(batch, -1)
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        padding = seq.eq(0)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if embeddings.ndim == 2 and features.ndim in (2, 3):
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(dim=-1),
                (features * self.item_emb(neg)).sum(dim=-1))

    def score_candidates(self, seq, candidates):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        # Appended MASK is the final nonpadding position in each left-padded row.
        features = self.encode_sequence(prediction_seq)[:, -1]
        scores = self.predict_score(features, candidates)
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

    def forward(self, seq, pos, neg):
        return self.model(seq, pos, neg)


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    selected = pos.ne(0)
    return F.softplus(negative[selected] - positive[selected]).mean()


def _history_to_tensor(history, maxlen, device):
    arr = np.zeros(maxlen, dtype=np.int64)
    items = history[-maxlen:]
    if items:
        arr[-len(items):] = items
    return torch.as_tensor(arr, dtype=torch.long, device=device).unsqueeze(0)


def make_validation_cases(user_train, user_valid, itemnum, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        candidates = [target]
        count = min(99, itemnum - sum(1 <= x <= itemnum for x in excluded))
        for _ in range(max(0, count)):
            negative = rng.randint(1, itemnum)
            while negative in excluded:
                negative = rng.randint(1, itemnum)
            excluded.add(negative)
            candidates.append(negative)
        cases.append((user, candidates))
    return cases


def evaluate_cases(model, user_train, cases, device, maxlen):
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user, candidates in cases:
            seq = _history_to_tensor(user_train[user], maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda j: -scores[j]).index(0)
            for cutoff in (10, 20):
                if rank < cutoff:
                    totals[f'HR@{cutoff}'] += 1.0
                    totals[f'NDCG@{cutoff}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    cases = make_validation_cases(user_train, user_valid, itemnum, num_users)
    return evaluate_cases(model, user_train, cases, device, maxlen)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    prefix_training = dataset_name in ('Industrial_and_Scientific', 'Electronics')
    train = TrainingHistories(train, prefix_training=prefix_training)
    if itemnum <= 0 or not train.eligible_users:
        raise RuntimeError(f'empty training data: {dataset_name}')
    print(f'[{dataset_name}] users={len(train)} items={itemnum} prefix_training={prefix_training}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = BERT4RecModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError(f'no validation cases: {dataset_name}')
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            batch = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss: {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, train, cases, device, MAXLEN)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} loss={total_loss / NUM_BATCHES:.5f} validation={metrics}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if PATIENCE > 0 and epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best local validation HR@10={best_hr:.5f}, epoch={best_epoch}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    body = r'''
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
        raise ValueError(f'unknown dataset_name: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = BERT4RecModel(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
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
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(BERT4RecModel) + '\n' + body)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Merged BERT4Rec on {device}, seed={SEED}, batches={NUM_BATCHES}, max_epochs={NUM_EPOCHS}', flush=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro-average best local validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
