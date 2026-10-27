from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN_UNITS = int(os.environ.get('SASREC_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT_RATE = float(os.environ.get('SASREC_DROPOUT', '0.5'))
LR = float(os.environ.get('SASREC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('SASREC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('SASREC_BATCH', '128'))
NUM_BATCHES = int(os.environ.get('SASREC_BATCHES', '200'))
NUM_EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
K = 32
CALIBRATION = 0.75
NEGATIVE_CHUNK = 1024
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    itemnum = 0
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats.get(name, {}).get('itemnum', 0))
    observed_max = max((max(items) for split in (train, valid)
                        for items in split.values() if items), default=0)
    return train, valid, max(itemnum, observed_max)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.conv1(inputs.transpose(-1, -2))
        x = self.relu(self.dropout1(x))
        x = self.dropout2(self.conv2(x)).transpose(-1, -2)
        return inputs + x


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.item_emb.weight.data.normal_(0.0, 1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def log2feats(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        x = self.emb_dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        keep = (seq != 0).unsqueeze(-1)
        x = x * keep
        causal = torch.ones(length, length, device=seq.device, dtype=torch.bool).triu(1)
        for norm_a, attention, norm_f, feedforward in zip(
                self.attention_layernorms, self.attention_layers,
                self.forward_layernorms, self.forward_layers):
            x = x.transpose(0, 1)
            queries = norm_a(x)
            output, _ = attention(queries, x, x, attn_mask=causal, need_weights=False)
            x = (queries + output).transpose(0, 1)
            x = feedforward(norm_f(x)) * keep
        return self.last_layernorm(x)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


class TrainingSampler:
    def __init__(self, train, itemnum, device):
        self.itemnum = itemnum
        self.device = device
        histories = [v for v in train.values() if len(v) >= 2 and len(set(v)) < itemnum]
        if not histories:
            raise ValueError('No eligible training histories')
        self.seq = np.zeros((len(histories), MAXLEN), dtype=np.int32)
        self.pos = np.zeros_like(self.seq)
        encoded = []
        for row, history in enumerate(histories):
            length = min(len(history) - 1, MAXLEN)
            self.seq[row, -length:] = history[-length-1:-1]
            self.pos[row, -length:] = history[-length:]
            encoded.append(np.asarray(sorted(set(history)), dtype=np.int64) + row * (itemnum + 1))
        self.keys = torch.as_tensor(np.concatenate(encoded), device=device, dtype=torch.long)
        self.rng = np.random.default_rng(42)

    def batch(self):
        rows = self.rng.integers(0, len(self.seq), size=BATCH_SIZE)
        seq = torch.as_tensor(self.seq[rows].astype(np.int64), device=self.device)
        pos = torch.as_tensor(self.pos[rows].astype(np.int64), device=self.device)
        active = pos != 0
        row_t = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        owners = row_t[:, None].expand_as(pos)[active]
        targets = pos[active]
        negatives = torch.randint(1, self.itemnum + 1, (targets.numel(), K), device=self.device)
        offsets = owners[:, None] * (self.itemnum + 1)
        while True:
            query = negatives + offsets
            indices = torch.searchsorted(self.keys, query.contiguous())
            invalid = self.keys[indices.clamp_max(self.keys.numel() - 1)] == query
            count = int(invalid.sum().item())
            if count == 0:
                break
            negatives[invalid] = torch.randint(1, self.itemnum + 1, (count,), device=self.device)
        return seq, active, targets, negatives


def prepare_validation(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        history = train[user][-MAXLEN:]
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError('Validation requires 99 unique unobserved negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append([0] * (MAXLEN - len(history)) + history)
        candidates.append([valid[user][0]] + list(negatives))
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, validation, device):
    sequences, candidates = validation
    was_training = model.training
    model.eval()
    hits, ndcg = 0, 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start+64], device=device)
        cand = torch.as_tensor(candidates[start:start+64], device=device)
        scores = model.predict(seq, cand)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(-1)
        hit = ranks < 10
        hits += int(hit.sum().item())
        ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
    model.train(was_training)
    count = max(len(sequences), 1)
    return hits / count, ndcg / count


def gbce_loss(model, seq, active, targets, negatives):
    features = model(seq)[active]
    positive_logits = (features * model.item_emb(targets)).sum(-1)
    alpha = K / max(model.itemnum - 1, 1)
    beta = 1.0 - CALIBRATION * (1.0 - alpha)
    total = beta * F.softplus(-positive_logits.float()).sum()
    for start in range(0, targets.numel(), NEGATIVE_CHUNK):
        stop = start + NEGATIVE_CHUNK
        logits = (model.item_emb(negatives[start:stop]) * features[start:stop, None, :]).sum(-1)
        total = total + F.softplus(logits.float()).sum()
    return total / (targets.numel() * (K + 1))


def train_one_dataset(name, device, deadline):
    seed_all(42)
    train, valid, itemnum = load_dataset(name)
    validation = prepare_validation(train, valid, itemnum)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                 betas=(0.9, 0.98))
    sampler = TrainingSampler(train, itemnum, device)
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best = (-1.0, -1.0)
    best_epoch = 0
    print(f'[{name}] users={len(train)} items={itemnum} negatives={K}', flush=True)

    def save():
        torch.save(dict(config=config, state_dict=model.state_dict(), objective='gBCE',
                        negative_count=K, calibration=CALIBRATION,
                        validation_hr10=best[0], validation_ndcg10=best[1],
                        epoch=best_epoch), checkpoint)

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss, steps = 0.0, 0
        model.train()
        for _ in range(NUM_BATCHES):
            if time.monotonic() >= deadline - 15.0:
                break
            seq, active, targets, negatives = sampler.batch()
            optimizer.zero_grad(set_to_none=True)
            loss = gbce_loss(model, seq, active, targets, negatives)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite training loss for {name}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        expired = time.monotonic() >= deadline - 15.0
        if epoch == 1 or epoch % 5 == 0 or epoch == NUM_EPOCHS or expired:
            hr, ndcg = evaluate_val(model, validation, device)
            print(f'[{name}] epoch={epoch} steps={steps} loss={total_loss/max(steps,1):.6f} '
                  f'val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                save()
            if expired or epoch - best_epoch >= PATIENCE * 5:
                break
    if best[0] < 0:
        best = evaluate_val(model, validation, device)
        save()
    print(f'[{name}] best validation HR@10={best[0]:.4f} epoch={best_epoch}', flush=True)
    result = dict(hr10=best[0], ndcg10=best[1], best_epoch=best_epoch)
    del sampler, optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def write_predict_script(working_dir):
    script = '''from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = None
_CLASS = None


def _load(dataset):
    global _DEVICE, _CLASS
    if dataset in _MODELS:
        return _MODELS[dataset]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _CLASS is None:
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location('merged_sasrec_trial_model', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.SASRec
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    config['dropout_rate'] = 0.0
    model = _CLASS(**config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(_DEVICE).eval()
    _MODELS[dataset] = model
    return model


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if len(candidates) == 0:
        return []
    model = _load(dataset)
    hist = list(history[-model.maxlen:])
    seq = [0] * (model.maxlen - len(hist)) + hist
    with torch.inference_mode():
        seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
        cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
        return model.predict(seq_t, cand_t)[0].cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    started = time.monotonic()
    global_deadline = started + 6900.0
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script(WORKING)
    results = {}
    for index, name in enumerate(DATASETS):
        now = time.monotonic()
        share = max(0.0, global_deadline - now) / (len(DATASETS) - index)
        results[name] = train_one_dataset(name, device, min(global_deadline, now + share))
    macro_hr = float(np.mean([results[name]['hr10'] for name in DATASETS]))
    summary = dict(per_dataset=results, macro_validation_hr10=macro_hr,
                   elapsed_seconds=time.monotonic() - started)
    with open(os.path.join(WORKING, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Macro validation HR@10={macro_hr:.6f}', flush=True)


if __name__ == '__main__':
    main()
