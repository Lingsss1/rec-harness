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
from torch.utils.checkpoint import checkpoint

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.2
LR = 0.001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 25
KL_COEFFICIENT = 0.1
CHUNK_SIZE = 64
_USER_CACHE = {}


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
    key = id(user_train)
    if key not in _USER_CACHE:
        _USER_CACHE[key] = [u for u, items in user_train.items() if len(items) >= 2]
    users = _USER_CACHE[key]
    if not users:
        raise RuntimeError('No eligible training users')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        n = min(maxlen, len(history) - 1)
        seq[row, -n:] = history[-n-1:-1]
        pos[row, -n:] = history[-n:]
        # Preserve the family-specific diagnostic sampler contract.
        excluded = np.asarray(list(set(history)), dtype=np.int64)
        if len(excluded) < itemnum:
            draws = np.random.randint(1, itemnum + 1, size=n)
            bad = np.isin(draws, excluded)
            while bad.any():
                draws[bad] = np.random.randint(1, itemnum + 1, size=int(bad.sum()))
                bad = np.isin(draws, excluded)
            neg[row, -n:] = draws
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        self.output_norm = nn.LayerNorm(hidden_units)
        self.item_bias = nn.Parameter(torch.zeros(itemnum + 1))
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        # Compact LEFT-padded histories for a GRU that starts at the first item.
        # Restore features to their original positions for sequence supervision.
        valid = sequence.ne(0)
        lengths = valid.sum(dim=1)
        batch, steps = sequence.shape
        offsets = steps - lengths
        positions = torch.arange(steps, device=sequence.device).unsqueeze(0)
        compact_indices = (positions + offsets.unsqueeze(1)).clamp(max=steps - 1)
        compact = sequence.gather(1, compact_indices)
        compact = compact.masked_fill(positions >= lengths.unsqueeze(1), 0)
        emb = self.dropout(self.item_emb(compact))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=steps)
        restore_indices = (positions - offsets.unsqueeze(1)).clamp(min=0)
        output = output.gather(1, restore_indices.unsqueeze(-1).expand(-1, -1, self.hidden_units))
        return self.output_norm(output) * valid.unsqueeze(-1)

    def encode(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(sequence.shape[0], device=sequence.device), last]

    def predict_score(self, features, items):
        if features.ndim == 3:
            features = features[:, -1, :]
        embedding = self.item_emb(items)
        if items.ndim == 1:
            return features @ embedding.T + self.item_bias[items]
        return (features.unsqueeze(1) * embedding).sum(-1) + self.item_bias[items]

    def score_candidates(self, sequence, candidates):
        return self.predict_score(self.encode(sequence), candidates).squeeze(0)

    def forward(self, sequence, pos, neg):
        features = self.log2feats(sequence)
        positive = (features * self.item_emb(pos)).sum(-1) + self.item_bias[pos]
        negative = (features * self.item_emb(neg)).sum(-1) + self.item_bias[neg]
        return positive, negative


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


def rdrop_loss(a, b, targets, coefficient=0.1):
    a, b = a.float(), b.float()
    la, lb = F.log_softmax(a, dim=-1), F.log_softmax(b, dim=-1)
    pa, pb = la.exp(), lb.exp()
    symmetric_kl = 0.5 * ((pa * (la - lb)).sum(-1) + (pb * (lb - la)).sum(-1)).mean()
    supervised = 0.5 * (F.cross_entropy(a, targets) + F.cross_entropy(b, targets))
    return supervised + coefficient * symmetric_kl


def _chunk_loss(a_features, b_features, weight, bias, targets):
    logits_a = F.linear(a_features, weight, bias)
    logits_b = F.linear(b_features, weight, bias)
    return rdrop_loss(logits_a, logits_b, targets, KL_COEFFICIENT)


def training_loss(model, sequence, positive):
    valid = positive.ne(0)
    # The latest target plus one random valid position per history keeps the
    # full-vocabulary objective affordable while supervising earlier states.
    selection = torch.zeros_like(valid)
    positions = torch.arange(positive.shape[1], device=positive.device).expand_as(positive)
    last = positions.masked_fill(~valid, -1).max(1).values
    rows = torch.arange(positive.shape[0], device=positive.device)
    selection[rows, last.clamp(min=0)] = last.ge(0)
    earlier = valid & ~selection
    random_scores = torch.rand(positive.shape, device=positive.device).masked_fill(~earlier, -1)
    chosen = random_scores.argmax(1)
    selection[rows, chosen] |= earlier.any(1)
    indices = selection.flatten().nonzero(as_tuple=False).squeeze(1)
    targets = positive.flatten()[indices] - 1
    # Exactly the same positions and targets in both shared-parameter passes.
    a = model.log2feats(sequence).reshape(-1, model.hidden_units)[indices]
    b = model.log2feats(sequence).reshape(-1, model.hidden_units)[indices]
    weight = model.item_emb.weight[1:]
    bias = model.item_bias[1:]
    count = len(targets)
    total = a.sum() * 0.0
    for start in range(0, count, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, count)
        # Recompute vocabulary logits during backward instead of retaining all
        # chunks' distributions. There is still one backward/update per batch.
        part = checkpoint(_chunk_loss, a[start:end], b[start:end], weight, bias,
                          targets[start:end], use_reentrant=False)
        total = total + part * ((end - start) / count)
    return total


def _history_to_tensor(history, maxlen, device):
    sequence = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    return torch.as_tensor(sequence, device=device).unsqueeze(0)


def make_validation_cases(user_train, user_valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {0, target}
        candidates = [target]
        needed = min(99, itemnum - len(excluded - {0}))
        while len(candidates) < needed + 1:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                excluded.add(candidate)
                candidates.append(candidate)
        cases.append((user_train[user], candidates))
    return cases


@torch.no_grad()
def evaluate_cases(model, cases, device):
    model.eval()
    hits = ndcg = 0.0
    for history, candidates in cases:
        sequence = _history_to_tensor(history, MAXLEN, device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(sequence, items)
        # Stable candidate-order tie breaking, with the target at index zero.
        rank = int((scores[1:] > scores[0]).sum().item())
        hits += float(rank < 10)
        ndcg += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
    n = max(1, len(cases))
    return {'HR@10': hits / n, 'NDCG@10': ndcg / n}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen=MAXLEN, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_all()
    _USER_CACHE.clear()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                  num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    model = GRU4RecModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation users for {dataset_name}')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            sequence, positive, _negative = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            sequence = torch.as_tensor(sequence, device=device)
            positive = torch.as_tensor(positive, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, sequence, positive)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss: {dataset_name}, epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
            total_loss += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} '
                  f'validation HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('Training did not produce a selected checkpoint')
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump({'model_config': config, 'maxlen': MAXLEN,
                     'best_epoch': best_epoch, 'validation_hr10': best_hr}, f)
    print(f'[{dataset_name}] selected epoch={best_epoch}, validation HR@10={best_hr:.4f}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the actual trained class so predictor and checkpoint cannot diverge.
    preamble = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(**meta['model_config']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    _ensure_loaded(ds)
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
    tail = history[-maxlen:]
    seq = [0] * (maxlen - len(tail)) + list(tail)
    sequence = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    scores = model.score_candidates(sequence, items)
    return [float(value) for value in scores.cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(preamble + inspect.getsource(GRU4RecModel) + '\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro-average selected validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
