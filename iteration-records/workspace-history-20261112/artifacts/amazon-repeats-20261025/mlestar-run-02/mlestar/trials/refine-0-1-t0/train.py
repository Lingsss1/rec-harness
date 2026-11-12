from __future__ import annotations

import importlib.util
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
NEGATIVE_CHUNK = 1024
DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']


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
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = int(stats[name]['itemnum'])
    if not itemnum:
        itemnum = max((max(items) for split in (train, valid) for items in split.values() if items), default=0)
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        return outputs.transpose(-1, -2) + inputs


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum
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
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        x = self.emb_dropout(x + self.pos_emb(positions))
        keep = (seq != 0).unsqueeze(-1)
        x = x * keep
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).triu(1)
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
        features = self.log2feats(seq)[:, -1, :]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


class Sampler:
    def __init__(self, train, itemnum):
        self.rng = np.random.RandomState(42)
        self.itemnum = itemnum
        self.histories = []
        self.excluded = []
        for items in train.values():
            if len(items) >= 2:
                history = np.asarray(items, dtype=np.int64)
                excluded = np.unique(history)
                if len(excluded) >= itemnum:
                    raise ValueError('No unobserved training negatives for a user')
                self.histories.append(history)
                self.excluded.append(excluded)
        if not self.histories:
            raise ValueError('Dataset contains no users with two training interactions')

    def sample(self):
        seq = np.zeros((BATCH_SIZE, MAXLEN), dtype=np.int64)
        pos = np.zeros_like(seq)
        negatives = []
        users = self.rng.randint(len(self.histories), size=BATCH_SIZE)
        for row, index in enumerate(users):
            history = self.histories[index]
            length = min(len(history) - 1, MAXLEN)
            seq[row, -length:] = history[-length-1:-1]
            pos[row, -length:] = history[-length:]
            neg = self.rng.randint(1, self.itemnum + 1, size=(length, K))
            invalid = np.isin(neg, self.excluded[index])
            while invalid.any():
                neg[invalid] = self.rng.randint(1, self.itemnum + 1, size=int(invalid.sum()))
                invalid = np.isin(neg, self.excluded[index])
            negatives.append(neg)
        return seq, pos, np.concatenate(negatives, axis=0)


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
        seq = np.zeros(MAXLEN, dtype=np.int64)
        if history:
            seq[-len(history):] = history
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError('Validation requires 99 unique unobserved negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, validation, device):
    sequences, candidates = validation
    model.eval()
    hits = 0
    ndcg = 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start+64], device=device)
        cand = torch.as_tensor(candidates[start:start+64], device=device)
        scores = model.predict(seq, cand)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(-1)
        hit = ranks < 10
        hits += int(hit.sum().item())
        ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
    count = max(len(sequences), 1)
    model.train()
    return hits / count, ndcg / count


def gbce_loss(model, sequences, positive_items, negative_items):
    mask = positive_items != 0
    features = model(sequences)[mask]
    positive_logits = (features * model.item_emb(positive_items[mask])).sum(-1).float()
    alpha = K / max(model.itemnum - 1, 1)
    beta = 1.0 - 0.75 * (1.0 - alpha)
    total = beta * F.softplus(-positive_logits).sum()
    ranking_total = positive_logits.new_zeros(())
    for start in range(0, features.shape[0], NEGATIVE_CHUNK):
        embeddings = model.item_emb(negative_items[start:start+NEGATIVE_CHUNK])
        logits = (features[start:start+NEGATIVE_CHUNK].unsqueeze(1) * embeddings).sum(-1).float()
        total = total + F.softplus(logits).sum()
        positive_chunk = positive_logits[start:start+NEGATIVE_CHUNK].unsqueeze(1)
        ranking_total = ranking_total + F.softplus(logits - positive_chunk).sum()
    count = features.shape[0]
    return total / (count * (K + 1)) + 0.05 * ranking_total / (count * K)


def train_one_dataset(name, device, working_dir, deadline):
    train, valid, itemnum = load_dataset(name)
    print(f'[{name}] users={len(train)} items={itemnum} negatives={K}', flush=True)
    validation = prepare_validation(train, valid, itemnum)
    sampler = Sampler(train, itemnum)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                 betas=(0.9, 0.98))
    checkpoint = os.path.join(working_dir, name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        steps = 0
        model.train()
        for _ in range(NUM_BATCHES):
            if time.monotonic() >= deadline - 15:
                break
            seq, pos, neg = sampler.sample()
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = gbce_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss for {name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        timed_out = time.monotonic() >= deadline - 15
        if epoch % 5 == 0 or epoch == NUM_EPOCHS or timed_out:
            hr, ndcg = evaluate_val(model, validation, device)
            print(f'[{name}] epoch={epoch} steps={steps} loss={total_loss/max(steps,1):.6f} '
                  f'val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if timed_out or epoch - best_epoch >= PATIENCE * 5:
                break
    if best_hr < 0:
        best_hr, _ = evaluate_val(model, validation, device)
        torch.save(model.state_dict(), checkpoint)
    with open(os.path.join(working_dir, name + '_meta.pkl'), 'wb') as f:
        pickle.dump(dict(config=config, objective='gBCE', negatives=K, calibration_t=0.75,
                         best_epoch=best_epoch, validation_hr10=best_hr), f)
    print(f'[{name}] best validation HR@10={best_hr:.4f}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    script = '''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _load_dataset(name):
    global _device, _model_class
    if name in _models:
        return
    if name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if _device is None:
        torch.set_num_threads(1)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        spec = importlib.util.spec_from_file_location('gsasrec_trial_model', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.SASRec
    with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    config = dict(meta['config'])
    config['dropout_rate'] = 0.0
    model = _model_class(**config).to(_device)
    state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[name] = model
    _metas[name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    name = dataset_name if dataset_name is not None else DATASETS[0]
    _load_dataset(name)
    if len(candidates) == 0:
        return []
    config = _metas[name]['config']
    maxlen = config['maxlen']
    hist = list(history[-maxlen:])
    seq = [0] * (maxlen - len(hist)) + hist
    with torch.inference_mode():
        seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
        cand_t = torch.tensor([candidates], dtype=torch.long, device=_device)
        return _models[name].predict(seq_t, cand_t)[0].cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    started = time.monotonic()
    # Reserve 100 seconds within the 7,200-second allowance for final I/O and overhead.
    global_deadline = started + 7100
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    write_predict_script(working_dir)
    scores = []
    for index, name in enumerate(DATASETS):
        now = time.monotonic()
        share = max(0.0, (global_deadline - now) / (len(DATASETS) - index))
        deadline = min(global_deadline, now + share)
        scores.append(train_one_dataset(name, device, working_dir, deadline))
    with open(os.path.join(working_dir, 'validation_summary.json'), 'w') as f:
        json.dump({'per_dataset_hr10': dict(zip(DATASETS, scores)),
                   'macro_validation_hr10': float(np.mean(scores)),
                   'elapsed_seconds': time.monotonic() - started}, f, indent=2)
    print(f'Macro validation HR@10={np.mean(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
