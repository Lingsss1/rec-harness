"""Single-model, per-category GRU4Rec with full-vocabulary cross-entropy."""
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

DEFAULT_DATASETS = [
    'Movies_and_TV',
    'Industrial_and_Scientific',
    'Electronics',
    'CDs_and_Vinyl',
]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
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
LOSS_CHUNK = 64
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 2:
                data[int(fields[0])].append(int(fields[1]))
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
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) == 2:
                        itemnum = max(itemnum, int(fields[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen, users=None, need_negatives=True):
    # Preserve the family-specific (sequence, shifted positives, negatives) contract.
    if users is None:
        users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        inputs = history[:-1][-maxlen:]
        targets = history[1:][-maxlen:]
        size = len(inputs)
        seq[row, -size:] = inputs
        pos[row, -size:] = targets
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
        self.gru = nn.GRU(
            hidden_units, hidden_units, num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        features, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return features

    def encode(self, seq):
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        if items.ndim == features.ndim - 1 and items.shape == features.shape[:-1]:
            return (features * embeddings).sum(dim=-1)
        if features.ndim == 2 and items.ndim == 2:
            return torch.einsum('bd,bkd->bk', features, embeddings)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        out = self.log2feats(seq)
        return (out * self.item_emb(pos)).sum(-1), (out * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq), candidates)[0]


def _chunk_cross_entropy(features, weights, targets):
    logits = features.float() @ weights.float().transpose(0, 1)
    return F.cross_entropy(logits, targets - 1, reduction='sum')


def categorical_sequence_loss(model, seq, pos):
    features = model.log2feats(seq)
    indices = torch.nonzero(pos.reshape(-1) != 0, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError('Batch contains no shifted training targets')
    if indices.numel() > MAX_POSITIONS:
        order = torch.randperm(indices.numel(), device=indices.device)[:MAX_POSITIONS]
        indices = indices[order]
    selected_features = features.reshape(-1, model.hidden_units)[indices]
    targets = pos.reshape(-1)[indices]
    # Checkpoint each scalar loss so full-vocabulary logits are recomputed during
    # backward instead of retaining all 512 rows of softmax activations at once.
    weights = model.item_emb.weight[1:]
    losses = []
    for start in range(0, indices.numel(), LOSS_CHUNK):
        losses.append(checkpoint(
            _chunk_cross_entropy,
            selected_features[start:start + LOSS_CHUNK],
            weights,
            targets[start:start + LOSS_CHUNK],
            use_reentrant=False,
        ))
    return torch.stack(losses).sum() / indices.numel()


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros((1, maxlen), dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[0, -len(tail):] = tail
    return torch.as_tensor(seq, dtype=torch.long, device=device)


def build_validation_cases(user_train, user_valid, itemnum):
    # An isolated RNG keeps evaluation fixed without perturbing training sampling.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        candidates = [target]
        desired = min(99, max(0, itemnum - len(excluded - {0})))
        while len(candidates) < desired + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        cases.append((user, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, user_train, cases, device):
    model.eval()
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    for user, candidates in cases:
        seq = _history_to_tensor(user_train[user], MAXLEN, device)
        candidates_t = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, candidates_t).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        for k in (10, 20):
            if rank < k:
                totals[f'HR@{k}'] += 1.0
                totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    count = max(1, len(cases))
    model.train()
    return {key: value / count for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    users = [u for u, history in user_train.items() if len(history) >= 2]
    if itemnum <= 0 or not users:
        raise RuntimeError(f'Empty training data for {dataset_name}')
    cases = build_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation users for {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum} valid_users={len(cases)}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr10 = -1.0
    best_ndcg10 = -1.0
    best_epoch = 0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.monotonic()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN,
                users=users, need_negatives=False,
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = categorical_sequence_loss(model, seq_t, pos_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {dataset_name} at epoch {epoch}')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, cases, device)
            hr10 = metrics['HR@10']
            ndcg10 = metrics['NDCG@10']
            elapsed = time.monotonic() - started
            print(
                f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} '
                f'loss={loss_sum / NUM_BATCHES:.5f} '
                f'val_score: {hr10:.4f} NDCG@10={ndcg10:.4f} '
                f'elapsed_single_gpu_seconds={elapsed:.1f}', flush=True,
            )
            if hr10 > best_hr10 or (hr10 == best_hr10 and ndcg10 > best_ndcg10):
                best_hr10, best_ndcg10 = hr10, ndcg10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint was selected; check the epoch configuration')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    meta = {
        'itemnum': itemnum, 'maxlen': MAXLEN, 'hidden_units': HIDDEN_UNITS,
        'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE,
        'best_epoch': best_epoch, 'best_validation_hr10': best_hr10,
        'elapsed_single_gpu_seconds': elapsed if device.type == 'cuda' else 0.0,
        'elapsed_wall_seconds': elapsed,
        'objective': 'full_vocabulary_cross_entropy',
    }
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best HR@10={best_hr10:.4f} epoch={best_epoch} seconds={elapsed:.1f}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr10, meta


def write_predict_script(working_dir):
    # Embed the exact trained architecture; inference never imports training data.
    header = '''"""One GRU4Rec checkpoint per Amazon category."""
from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

'''
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
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(
        itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
        num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate'],
    ).to(_device)
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
        raise ValueError(f'Unknown dataset: {ds}')
    if len(candidates) == 0:
        return []
    _ensure_loaded(ds)
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta['maxlen'])
    seq = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq[-len(tail):] = tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t)
    return [float(score) for score in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header)
        f.write(inspect.getsource(GRU4RecModel))
        f.write('\n')
        f.write(runtime)


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
    results['total_elapsed_single_gpu_seconds'] = sum(
        results[ds]['elapsed_single_gpu_seconds'] for ds in DATASETS
    )
    with open(os.path.join(WORKING_DIR, 'training_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)
    print(f"Total single-GPU allocation seconds={results['total_elapsed_single_gpu_seconds']:.1f}", flush=True)


if __name__ == '__main__':
    main()
