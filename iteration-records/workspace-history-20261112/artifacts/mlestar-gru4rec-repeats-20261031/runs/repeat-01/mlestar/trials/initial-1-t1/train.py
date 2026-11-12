from __future__ import annotations

import inspect
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = DEFAULT_DATASETS[:]
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
MAX_POSITIONS = 512
LOGIT_CHUNK = 64


def _read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


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


_sampler_cache = None


def sample_batch(user_train, itemnum, batch_size, maxlen):
    # Preserve the family-specific (sequence, positive, negative) contract.
    # The CE objective does not use negatives; construct them only for diagnostics.
    global _sampler_cache
    if _sampler_cache is None or _sampler_cache[0] is not user_train:
        users = [u for u, h in user_train.items() if len(h) >= 2]
        if not users:
            raise RuntimeError('No users with at least two training interactions')
        _sampler_cache = (user_train, users)
    users = _sampler_cache[1]
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
    # Vectorized uniform negatives keep diagnosis interfaces usable without
    # spending Python time on samples that the training objective ignores.
    valid = pos != 0
    neg[valid] = np.random.randint(1, itemnum + 1, size=int(valid.sum()))
    for row in range(batch_size):
        history_set = set(user_train[users[0]]) if False else set()
        # Negatives need only be valid item IDs for this CE training path.
        # They are never consumed by the objective.
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
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
        embeddings = self.dropout(self.item_emb(sequence))
        features, _ = self.gru(embeddings)
        return features

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if features.ndim == 1:
            return (embeddings * features).sum(dim=-1)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def encode(self, seq):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

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


def categorical_sequence_loss(model, seq, pos):
    features = model.log2feats(seq)
    flat_targets = pos.reshape(-1)
    selected = torch.nonzero(flat_targets != 0, as_tuple=False).flatten()
    if selected.numel() > MAX_POSITIONS:
        selected = selected[torch.randperm(selected.numel(), device=seq.device)[:MAX_POSITIONS]]
    if selected.numel() == 0:
        raise RuntimeError('Training batch has no shifted targets')
    features = features.reshape(-1, model.hidden_units).index_select(0, selected)
    targets = flat_targets.index_select(0, selected) - 1
    weight = model.item_emb.weight[1:].float()
    total = features.new_zeros((), dtype=torch.float32)
    for start in range(0, selected.numel(), LOGIT_CHUNK):
        logits = features[start:start + LOGIT_CHUNK].float() @ weight.T
        total = total + F.cross_entropy(logits, targets[start:start + LOGIT_CHUNK], reduction='sum')
    return total / selected.numel()


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    # Local RNG prevents evaluation from advancing training randomness.
    rng = random.Random(42)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {0, target}
        count = min(99, itemnum - len({i for i in excluded if 1 <= i <= itemnum}))
        candidates = [target]
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        seq = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        cases.append((seq, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hr = ndcg = 0.0
    for start in range(0, len(cases), 64):
        batch = cases[start:start + 64]
        sequences = torch.as_tensor(np.stack([c[0] for c in batch]), device=device)
        features = model.encode(sequences)
        for feature, (_, candidates) in zip(features, batch):
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.predict_score(feature, items)
            # Target is first; stable ranking breaks equal scores by candidate order.
            rank = int((scores[1:] > scores[0]).sum().item())
            if rank < 10:
                hr += 1
                ndcg += 1 / math.log2(rank + 2)
    model.train()
    return {'HR@10': hr / max(1, len(cases)), 'NDCG@10': ndcg / max(1, len(cases))}


def train_one_dataset(dataset_name, device, data_dir):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = categorical_sequence_loss(model, seq, pos)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr = metrics['HR@10']
            elapsed = time.perf_counter() - started
            print(f'[{dataset_name}] epoch={epoch} loss={loss_sum / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr:.5f} val_NDCG@10={metrics["NDCG@10"]:.5f} '
                  f'elapsed_device_seconds={elapsed:.2f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('No validation checkpoint saved')
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    record = dict(dataset=dataset_name, best_epoch=best_epoch, validation_hr10=best_hr,
                  elapsed_device_seconds=elapsed, device=str(device),
                  timing_note='Synchronized elapsed time including sampling, training and validation; host GPU accounting is authoritative.')
    with open(os.path.join(working_dir, f'{dataset_name}_training.json'), 'w') as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record), flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture; inference does not import train.py.
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    model_source = inspect.getsource(GRU4RecModel)
    predictor = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = GRU4RecModel(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                             num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset_name] = model
        _metas[dataset_name] = meta
    return _models[dataset_name], _metas[dataset_name]


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    model, meta = _load(ds)
    maxlen = int(meta['maxlen'])
    tail = list(history[-maxlen:])
    seq = [0] * (maxlen - len(tail)) + tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + model_source + '\n' + predictor)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    started = time.perf_counter()
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f} '
          f'total_elapsed_device_seconds={time.perf_counter() - started:.2f}', flush=True)


if __name__ == '__main__':
    main()
