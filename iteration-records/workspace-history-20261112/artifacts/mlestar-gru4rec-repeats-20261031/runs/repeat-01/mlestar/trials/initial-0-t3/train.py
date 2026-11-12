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
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 64
HIDDEN_UNITS = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.2
LR = 0.001
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 25
CONSISTENCY = 0.1
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as handle:
            stats = json.load(handle)
        itemnum = int(stats.get(dataset_name, {}).get('itemnum', 0))
    if not itemnum:
        itemnum = max((item for split in (train, valid) for hist in split.values() for item in hist), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    # Keep the baseline's family-specific (sequence, positive, negative) contract.
    users = [u for u, history in user_train.items() if len(history) >= 2]
    if not users:
        raise RuntimeError('No users with two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        inputs = history[:-1][-maxlen:]
        targets = history[1:][-maxlen:]
        seq[row, -len(inputs):] = inputs
        pos[row, -len(targets):] = targets
        # Negatives are unused by full-vocabulary CE, but remain available to diagnostics.
        excluded = set(history)
        if len(excluded) < itemnum:
            draws = np.random.randint(1, itemnum + 1, size=len(inputs))
            for j in range(len(draws)):
                while int(draws[j]) in excluded:
                    draws[j] = np.random.randint(1, itemnum + 1)
            neg[row, -len(inputs):] = draws
    return seq, pos, neg


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.output_bias = nn.Parameter(torch.zeros(itemnum + 1))
        nn.init.normal_(self.item_emb.weight, std=0.05)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        lengths = (sequence != 0).sum(dim=1)
        width = sequence.shape[1]
        positions = torch.arange(width, device=sequence.device).unsqueeze(0)
        # Compact LEFT-padded histories before recurrence, then restore alignment.
        compact_indices = (positions + width - lengths[:, None]).clamp(0, width - 1)
        compact = sequence.gather(1, compact_indices)
        compact = compact.masked_fill(positions >= lengths[:, None], 0)
        embedded = self.dropout(self.item_emb(compact))
        packed = pack_padded_sequence(embedded, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=width)
        restored_indices = (positions - (width - lengths[:, None])).clamp(0, width - 1)
        out = out.gather(1, restored_indices.unsqueeze(-1).expand(-1, -1, self.hidden_units))
        return out.masked_fill((sequence == 0).unsqueeze(-1), 0.0)

    def encode(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(sequence)
        last = positions.masked_fill(sequence == 0, -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(sequence.shape[0], device=sequence.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.T + self.output_bias[items]
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(-1) + self.output_bias[items]
        return (features.unsqueeze(-2) * embeddings).sum(-1) + self.output_bias[items]

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1) + self.output_bias[pos],
                (features * self.item_emb(neg)).sum(-1) + self.output_bias[neg])

    def score_candidates(self, sequence, candidates):
        return self.predict_score(self.encode(sequence), candidates)[0]


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

    def forward(self, *args):
        return self.model(*args)


def rdrop_loss(a, b, targets, coefficient=0.1):
    a, b = a.float(), b.float()
    la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    pa, pb = la.exp(), lb.exp()
    symmetric_kl = 0.5 * ((pa * (la - lb)).sum(-1) + (pb * (lb - la)).sum(-1)).mean()
    supervised = 0.5 * (F.cross_entropy(a, targets) + F.cross_entropy(b, targets))
    return supervised + coefficient * symmetric_kl


def training_loss(model, seq, pos):
    # One supervised position per sampled user controls full-vocabulary compute.
    # Half emphasize the latest transition; half sample uniformly over the history.
    lengths = (pos != 0).sum(1)
    random_offsets = (torch.rand(seq.shape[0], device=seq.device) * lengths).long()
    selected = seq.shape[1] - lengths + random_offsets
    use_last = torch.rand(seq.shape[0], device=seq.device) < 0.5
    selected = torch.where(use_last, torch.full_like(selected, seq.shape[1] - 1), selected)
    rows = torch.arange(seq.shape[0], device=seq.device)
    targets = pos[rows, selected] - 1
    first = model.log2feats(seq)[rows, selected]
    second = model.log2feats(seq)[rows, selected]

    def chunk_loss(a, b, labels, weight, bias):
        logits_a = F.linear(a, weight, bias)
        logits_b = F.linear(b, weight, bias)
        return rdrop_loss(logits_a, logits_b, labels, CONSISTENCY)

    loss = first.new_zeros(())
    for start in range(0, len(targets), 64):
        end = min(start + 64, len(targets))
        # Recompute vocabulary-sized activations during the single backward pass.
        part = checkpoint(chunk_loss, first[start:end], second[start:end], targets[start:end],
                          model.item_emb.weight[1:], model.output_bias[1:], use_reentrant=False)
        loss = loss + part * ((end - start) / len(targets))
    return loss


def make_validation(train, valid, itemnum):
    # Local RNG keeps candidates fixed across epochs and leaves training RNG untouched.
    rng = random.Random(SEED)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | {target, 0}
        candidates = [target]
        wanted = min(99, itemnum - len(excluded - {0}))
        while len(candidates) < wanted + 1:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                excluded.add(candidate)
                candidates.append(candidate)
        seq = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            seq[-len(history):] = history
        cases.append((seq, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hits = ndcg = 0.0
    for seq, candidates in cases:
        sequence = torch.as_tensor(seq[None, :], device=device)
        candidates = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(sequence, candidates)
        # Stable ordering places the positive first when scores tie.
        rank = int((scores[1:] > scores[0]).sum().item())
        if rank < 10:
            hits += 1.0
            ndcg += 1.0 / math.log2(rank + 2)
    count = max(1, len(cases))
    return {'HR@10': hits / count, 'NDCG@10': ndcg / count}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    cases = make_validation(train, valid, itemnum)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    print(f'[{dataset_name}] users={len(train)} items={itemnum}', flush=True)
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    checkpoint_path = os.path.join(WORKING, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, _ = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, seq, pos)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total / NUM_BATCHES:.5f} HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint_path)
            if epoch - best_epoch >= PATIENCE:
                break
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    with open(os.path.join(WORKING, dataset_name + '_meta.pkl'), 'wb') as handle:
        pickle.dump(meta, handle)
    print(f'[{dataset_name}] selected_epoch={best_epoch} validation_HR@10={best_hr:.4f}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    script = r'''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(_DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('rdrop_gru_training_definition', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dataset in _DATASETS:
        if dataset in _models:
            continue
        with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
            meta = pickle.load(handle)
        model = module.GRU4RecModel(meta['itemnum'], meta['hidden_units'], meta['num_layers'], meta['dropout_rate']).to(_device)
        weights = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _models:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if not candidates:
        return []
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    maxlen = meta['maxlen']
    seq = [0] * maxlen
    recent = list(history[-maxlen:])
    if recent:
        seq[-len(recent):] = recent
    sequence = torch.tensor([seq], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    return [float(value) for value in model.score_candidates(sequence, items).cpu().tolist()]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as handle:
        handle.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = [train_one_dataset(dataset, device, DATA_DIR) for dataset in DATASETS]
    write_predict_script(WORKING)
    print(f'Macro validation HR@10: {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
