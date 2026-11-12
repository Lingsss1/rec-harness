from __future__ import annotations

import inspect
import json
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('GRU4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
LR = float(os.environ.get('GRU4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('GRU4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 25
NUM_NEGATIVES = 32
HARD_WEIGHT = 0.25
BPR_MAX_BETA = 0.1
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_SAMPLER_CACHE = {}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _read_split(path):
    result = defaultdict(list)
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
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def _sampler_state(user_train, itemnum):
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        users = [u for u, h in user_train.items() if len(h) >= 2]
        if not users:
            raise RuntimeError('No users with at least two training interactions')
        forbidden = {u: np.asarray(sorted(set(user_train[u])), dtype=np.int64) for u in users}
        _SAMPLER_CACHE[key] = (user_train, users, forbidden, {})
    return _SAMPLER_CACHE[key]


def _sample_allowed(itemnum, forbidden, count, eligible_cache, user):
    forbidden = forbidden[(forbidden >= 1) & (forbidden <= itemnum)]
    if len(forbidden) >= itemnum:
        raise RuntimeError('No eligible training negative')
    if user in eligible_cache:
        return np.random.choice(eligible_cache[user], count, replace=True)
    if len(forbidden) > itemnum // 4:
        eligible = np.setdiff1d(np.arange(1, itemnum + 1, dtype=np.int64), forbidden, assume_unique=True)
        eligible_cache[user] = eligible
        return np.random.choice(eligible, count, replace=True)
    result = np.random.randint(1, itemnum + 1, size=count, dtype=np.int64)
    for _ in range(12):
        bad = np.isin(result, forbidden)
        if not bad.any():
            return result
        result[bad] = np.random.randint(1, itemnum + 1, size=int(bad.sum()), dtype=np.int64)
    eligible = np.setdiff1d(np.arange(1, itemnum + 1, dtype=np.int64), forbidden, assume_unique=True)
    eligible_cache[user] = eligible
    bad = np.isin(result, forbidden)
    result[bad] = np.random.choice(eligible, int(bad.sum()), replace=True)
    return result


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=None):
    # The default retains the baseline's aligned-negative diagnostic interface.
    _, users, forbidden, eligible_cache = _sampler_state(user_train, itemnum)
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq) if num_negatives is None else np.zeros((batch_size, num_negatives), dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        count = length if num_negatives is None else num_negatives
        sampled = _sample_allowed(itemnum, forbidden[user], count, eligible_cache, user)
        if num_negatives is None:
            neg[row, -length:] = sampled
        else:
            neg[row] = sampled
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
        embeddings = self.dropout(self.item_emb(sequence))
        features, _ = self.gru(embeddings)
        return features

    def encode(self, seq):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values
        result = features[torch.arange(seq.size(0), device=seq.device), last.clamp(min=0)]
        return result * (last >= 0).unsqueeze(-1).to(result.dtype)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(-1)
        return (features * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return (features * self.item_emb(pos)).sum(-1), (features * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq), candidates)[0]


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos != 0
    return F.softplus(negative[mask].float() - positive[mask].float()).mean()


def merged_sequence_loss(model, seq, pos, neg):
    # One GRU, one item table, and one jointly optimized loss. Uniform BPR
    # retains broad ranking pressure while BPR-max emphasizes hard negatives.
    features = model.log2feats(seq)
    total = features.new_zeros((), dtype=torch.float32)
    count = (pos != 0).sum().clamp(min=1)
    for start in range(0, seq.size(0), 16):
        end = min(start + 16, seq.size(0))
        h = features[start:end]
        target = pos[start:end]
        mask = target != 0
        positive = (h * model.item_emb(target)).sum(-1)[mask].float()
        negatives = torch.bmm(h, model.item_emb(neg[start:end]).transpose(1, 2))[mask].float()
        for offset in range(0, positive.numel(), 2048):
            p = positive[offset:offset + 2048]
            n = negatives[offset:offset + 2048]
            pairwise = F.softplus(n - p[:, None]).mean(-1)
            log_weights = F.log_softmax(n, dim=-1)
            hard = -torch.logsumexp(log_weights + F.logsigmoid(p[:, None] - n), dim=-1)
            regularizer = (log_weights.exp() * n.square()).sum(-1)
            total = total + ((1.0 - HARD_WEIGHT) * pairwise + HARD_WEIGHT * (hard + BPR_MAX_BETA * regularizer)).sum()
    return total / count


def make_validation(user_train, user_valid, itemnum, maxlen, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    sequences, candidates = [], []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        blocked = set(history) | {target, 0}
        available = itemnum - len({i for i in blocked if 1 <= i <= itemnum})
        if available < 99:
            raise RuntimeError('Validation requires 99 unique eligible negatives')
        chosen = [target]
        if available < itemnum // 4:
            pool = [i for i in range(1, itemnum + 1) if i not in blocked]
            chosen.extend(rng.sample(pool, 99))
        else:
            while len(chosen) < 100:
                item = rng.randint(1, itemnum)
                if item not in blocked:
                    blocked.add(item)
                    chosen.append(item)
        sequence = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            sequence[-len(tail):] = tail
        sequences.append(sequence)
        candidates.append(chosen)
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


def evaluate_fixed(model, validation, device):
    sequences, candidates = validation
    if not len(sequences):
        raise RuntimeError('No validation users')
    was_training = model.training
    model.eval()
    ranks = []
    with torch.no_grad():
        for start in range(0, len(sequences), 64):
            seq = torch.as_tensor(sequences[start:start + 64], device=device)
            items = torch.as_tensor(candidates[start:start + 64], device=device)
            scores = model.predict_score(model.encode(seq), items)
            ranks.extend((scores[:, 1:] > scores[:, :1]).sum(-1).cpu().tolist())
    model.train(was_training)
    ranks = np.asarray(ranks)
    return {
        'HR@10': float(np.mean(ranks < 10)),
        'HR@20': float(np.mean(ranks < 20)),
        'NDCG@10': float(np.mean(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0))),
    }


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    return evaluate_fixed(model, make_validation(user_train, user_valid, itemnum, maxlen, num_users), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    validation = make_validation(user_train, user_valid, itemnum, MAXLEN)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN, NUM_NEGATIVES)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = merged_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_fixed(model, validation, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.6f} val_score: {hr:.4f} HR@20={metrics["HR@20"]:.4f} NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = {
        'itemnum': itemnum, 'maxlen': MAXLEN,
        'hidden_units': HIDDEN_UNITS, 'num_layers': NUM_LAYERS,
        'dropout_rate': DROPOUT_RATE, 'best_epoch': best_epoch,
        'validation_hr10': best_hr, 'num_negatives': NUM_NEGATIVES,
        'hard_weight': HARD_WEIGHT, 'bpr_max_beta': BPR_MAX_BETA,
    }
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as handle:
        pickle.dump(meta, handle)
    print(f'[{dataset_name}] selected epoch={best_epoch} HR@10={best_hr:.4f}', flush=True)
    _SAMPLER_CACHE.clear()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    architecture = inspect.getsource(GRU4RecModel)
    predictor = r'''
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
    loaded_models, loaded_metas = {}, {}
    for ds in DATASETS:
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as handle:
            meta = pickle.load(handle)
        with torch.random.fork_rng(devices=[]):
            model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate'])
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(_device)
        model.eval()
        loaded_models[ds] = model
        loaded_metas[ds] = meta
    _models.update(loaded_models)
    _metas.update(loaded_metas)


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f'Unknown dataset route: {ds}')
    if not candidates:
        return []
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items)
    return [float(score) for score in scores.cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(header + architecture + '\n' + predictor)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'GRU4Rec merged BPR training: device={device}, seed=42, batches=200, max_epochs={NUM_EPOCHS}', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    write_predict_script(WORKING_DIR)
    print(f'Macro validation HR@10 = {sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
