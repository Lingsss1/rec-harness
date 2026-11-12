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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
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
PATIENCE = 25
NEGATIVE_SAMPLES = 2048
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
_SAMPLE_CACHE = {}


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
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    itemnum = 0
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((max(items, default=0) for split in (train, valid) for items in split.values()), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    cached = _SAMPLE_CACHE.get(key)
    if cached is None or cached[0] is not user_train:
        users = [u for u, history in user_train.items() if len(history) >= 2]
        cached = (user_train, users, {u: set(user_train[u]) for u in users})
        _SAMPLE_CACHE[key] = cached
    _, users, interacted = cached
    if not users:
        raise RuntimeError('No users with two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        length = min(maxlen, len(history) - 1)
        seq[row, -length:] = history[-length - 1:-1]
        pos[row, -length:] = history[-length:]
        seen = interacted[user]
        if len(seen) < itemnum:
            for col in range(maxlen - length, maxlen):
                candidate = random.randint(1, itemnum)
                while candidate in seen:
                    candidate = random.randint(1, itemnum)
                neg[row, col] = candidate
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
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
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        output, _ = self.gru(self.dropout(self.item_emb(sequence)))
        return output

    def encode(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        output = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1)
        result = output[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return result * last.ge(0).unsqueeze(-1).to(result.dtype)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embedding = self.item_emb(items)
        if items.ndim == 1:
            return features @ embedding.transpose(0, 1)
        return (features.unsqueeze(-2) * embedding).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        scores = self.predict_score(self.encode(seq), candidates)
        return scores[0] if scores.shape[0] == 1 else scores


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


def cross_entropy_sequence_loss(model, seq, pos, neg=None):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask] - 1
    if not targets.numel():
        return features.sum() * 0.0
    chunk_size = max(1, min(256, 4000000 // model.itemnum))
    loss = features.sum() * 0.0
    for start in range(0, targets.numel(), chunk_size):
        stop = min(start + chunk_size, targets.numel())
        logits = features[start:stop] @ model.item_emb.weight[1:].T
        loss = loss + F.cross_entropy(logits, targets[start:stop], reduction='sum') / targets.numel()
    return loss


def bpr_sequence_loss(model, seq, pos, neg):
    return cross_entropy_sequence_loss(model, seq, pos, neg)


def categorical_backward(model, seq, pos, sampled=False):
    mask = pos.ne(0)
    features = model.log2feats(seq)[mask]
    targets = pos[mask]
    count = targets.numel()
    if not count:
        raise RuntimeError('Empty shifted-target batch')
    # Accumulate output embedding and feature gradients in bounded-size chunks,
    # then backpropagate through the recurrent graph once.
    proxy = features.detach().requires_grad_(True)
    if sampled:
        negatives = torch.randint(1, model.itemnum + 1, (NEGATIVE_SAMPLES,), device=seq.device)
        correction = math.log(NEGATIVE_SAMPLES / model.itemnum)
        chunk_size = 128
    else:
        chunk_size = max(1, min(256, 4000000 // model.itemnum))
    total = 0.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        h = proxy[start:stop]
        target = targets[start:stop]
        if sampled:
            positive = (h * model.item_emb(target)).sum(-1, keepdim=True)
            negative_logits = h @ model.item_emb(negatives).T
            negative_logits = negative_logits - correction
            negative_logits = negative_logits.masked_fill(target[:, None].eq(negatives[None, :]), float('-inf'))
            logits = torch.cat((positive, negative_logits), dim=1)
            labels = torch.zeros(stop - start, dtype=torch.long, device=seq.device)
        else:
            logits = h @ model.item_emb.weight[1:].T
            labels = target - 1
        loss = F.cross_entropy(logits, labels, reduction='sum') / count
        loss.backward()
        total += loss.detach().item()
        del logits, loss
    features.backward(proxy.grad)
    return total


def backward_full_catalogue_ce(model, seq, pos):
    return categorical_backward(model, seq, pos, sampled=False)


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {0, target}
        if itemnum - len(excluded - {0}) < 99:
            raise RuntimeError('Fewer than 99 distinct eligible validation negatives')
        items = [target]
        while len(items) < 100:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                items.append(item)
        sequence = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            sequence[-len(tail):] = tail
        sequences.append(sequence)
        candidates.append(items)
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_cases(model, cases, device):
    sequences, candidates = cases
    was_training = model.training
    model.eval()
    ranks = []
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start + 64], dtype=torch.long, device=device)
        items = torch.as_tensor(candidates[start:start + 64], dtype=torch.long, device=device)
        scores = model.predict_score(model.encode(seq), items)
        ranks.extend(scores[:, 1:].gt(scores[:, :1]).sum(dim=1).cpu().tolist())
    model.train(was_training)
    n = max(1, len(ranks))
    return {
        'HR@10': sum(r < 10 for r in ranks) / n,
        'HR@20': sum(r < 20 for r in ranks) / n,
        'NDCG@10': sum(1.0 / math.log2(r + 2) for r in ranks if r < 10) / n,
    }


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen=MAXLEN, num_users=200):
    return evaluate_cases(model, make_validation_cases(user_train, user_valid, itemnum, maxlen), device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    sampled = dataset_name == 'Electronics' and itemnum > 30000
    objective = 'corrected_sampled_ce' if sampled else 'full_catalogue_ce'
    print(f'[{dataset_name}] users={len(train)} items={itemnum} objective={objective}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = make_validation_cases(train, valid, itemnum, MAXLEN)
    if not len(cases[0]):
        raise RuntimeError('No validation users: ' + dataset_name)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_value = categorical_backward(model, seq, pos, sampled=sampled)
            if not math.isfinite(loss_value):
                raise RuntimeError('Nonfinite loss: ' + dataset_name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
            total_loss += loss_value
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if (hr, ndcg) > (best_hr, best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
        elif epoch == 1:
            print(f'[{dataset_name}] epoch=1 loss={total_loss / NUM_BATCHES:.5f}', flush=True)
    if not best_epoch:
        raise RuntimeError('No checkpoint selected: ' + dataset_name)
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                objective=objective, best_epoch=best_epoch, seed=SEED)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] selected epoch={best_epoch} HR@10={best_hr:.4f}', flush=True)
    _SAMPLE_CACHE.clear()
    del optimizer, model, train, valid
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
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _ensure_loaded(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name in _models:
        return
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    dataset_name = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(dataset_name)
    if len(candidates) == 0:
        return []
    model = _models[dataset_name]
    maxlen = _metas[dataset_name]['maxlen']
    tail = list(history[-maxlen:])
    sequence = [0] * (maxlen - len(tail)) + tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    return [float(x) for x in model.score_candidates(seq, items).cpu().tolist()]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(GRU4RecModel) + '\n' + body)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
