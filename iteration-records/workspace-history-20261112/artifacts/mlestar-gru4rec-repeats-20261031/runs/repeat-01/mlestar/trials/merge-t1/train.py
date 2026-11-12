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
from torch.utils.checkpoint import checkpoint

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DEFAULT_DATASETS = DATASETS[:]
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
MAX_POSITIONS = 512
LOSS_CHUNK = 64
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
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
                fields = line.split()
                if len(fields) == 2:
                    itemnum = max(itemnum, int(fields[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen, users=None, need_negatives=True):
    if users is None:
        users = [u for u, h in user_train.items() if len(h) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        size = min(maxlen, len(history) - 1)
        seq[row, -size:] = history[-size - 1:-1]
        pos[row, -size:] = history[-size:]
        if need_negatives:
            seen = set(history)
            if len(seen) >= itemnum:
                raise RuntimeError('Cannot sample an unseen negative item')
            for col in range(maxlen - size, maxlen):
                item = random.randint(1, itemnum)
                while item in seen:
                    item = random.randint(1, itemnum)
                neg[row, col] = item
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
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, seq):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if features.ndim == 1:
            return (features * embeddings).sum(dim=-1)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        if items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

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


def _chunk_cross_entropy(features, weights, targets):
    logits = features.float() @ weights.float().transpose(0, 1)
    return F.cross_entropy(logits, targets - 1, reduction='sum')


def categorical_sequence_loss(model, seq, pos):
    features = model.log2feats(seq)
    flat_targets = pos.reshape(-1)
    selected = torch.nonzero(flat_targets != 0, as_tuple=False).flatten()
    if selected.numel() == 0:
        raise RuntimeError('Batch contains no shifted training targets')
    if selected.numel() > MAX_POSITIONS:
        selected = selected[torch.randperm(selected.numel(), device=seq.device)[:MAX_POSITIONS]]
    features = features.reshape(-1, model.hidden_units).index_select(0, selected)
    targets = flat_targets.index_select(0, selected)
    weights = model.item_emb.weight[1:]
    losses = []
    for start in range(0, selected.numel(), LOSS_CHUNK):
        losses.append(checkpoint(_chunk_cross_entropy,
                                 features[start:start + LOSS_CHUNK], weights,
                                 targets[start:start + LOSS_CHUNK],
                                 use_reentrant=False))
    return torch.stack(losses).sum() / selected.numel()


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {0, target}
        desired = min(99, itemnum - sum(1 <= i <= itemnum for i in excluded))
        candidates = [target]
        while len(candidates) < desired + 1:
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
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    for start in range(0, len(cases), 64):
        batch = cases[start:start + 64]
        sequences = torch.as_tensor(np.stack([case[0] for case in batch]), dtype=torch.long, device=device)
        features = model.encode(sequences)
        for feature, (_, candidates) in zip(features, batch):
            scores = model.predict_score(feature, torch.tensor(candidates, dtype=torch.long, device=device))
            rank = int((scores[1:] > scores[0]).sum().item())
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train()
    return {key: value / max(1, len(cases)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    users = [u for u, h in user_train.items() if len(h) >= 2]
    if itemnum <= 0 or not users:
        raise RuntimeError(f'Empty training data for {dataset_name}')
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    if not cases:
        raise RuntimeError(f'No validation users for {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} valid_users={len(cases)}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = best_ndcg = -1.0
    best_epoch = 0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.monotonic()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN,
                                       users=users, need_negatives=False)
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = categorical_sequence_loss(model, seq_t, pos_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {dataset_name} at epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} loss={loss_sum / NUM_BATCHES:.5f} '
                  f'val_score: {hr:.4f} NDCG@10={ndcg:.4f} '
                  f'elapsed_single_gpu_seconds={time.monotonic() - started:.1f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('No validation checkpoint selected')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                best_epoch=best_epoch, best_validation_hr10=best_hr,
                best_validation_ndcg10=best_ndcg,
                elapsed_single_gpu_seconds=elapsed if device.type == 'cuda' else 0.0,
                elapsed_wall_seconds=elapsed, objective='full_vocabulary_cross_entropy')
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best HR@10={best_hr:.4f} epoch={best_epoch} seconds={elapsed:.1f}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr, meta


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                         num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    if len(candidates) == 0:
        return []
    _ensure_loaded(ds)
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta['maxlen'])
    tail = list(history[-maxlen:])
    seq = [0] * (maxlen - len(tail)) + tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t)
    return [float(score) for score in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(GRU4RecModel) + '\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    print(f'Full-vocabulary CE GRU4Rec on {device}; seed={SEED}; batches/epoch={NUM_BATCHES}', flush=True)
    results = {}
    scores = []
    for dataset_name in DATASETS:
        score, meta = train_one_dataset(dataset_name, device, DATA_DIR)
        scores.append(score)
        results[dataset_name] = meta
    results['macro_validation_hr10'] = sum(scores) / len(scores)
    results['total_elapsed_single_gpu_seconds'] = sum(results[ds]['elapsed_single_gpu_seconds'] for ds in DATASETS)
    with open(os.path.join(WORKING_DIR, 'training_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)
    print(f"Total single-GPU allocation seconds={results['total_elapsed_single_gpu_seconds']:.1f}", flush=True)


if __name__ == '__main__':
    main()
