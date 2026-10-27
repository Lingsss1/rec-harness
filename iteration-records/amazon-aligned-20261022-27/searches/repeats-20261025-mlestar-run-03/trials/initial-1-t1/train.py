from __future__ import annotations

import importlib.util
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
HIDDEN_UNITS = int(os.environ.get('SASREC_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT_RATE = float(os.environ.get('SASREC_DROPOUT', '0.2'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WEIGHT_DECAY = float(os.environ.get('SASREC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('SASREC_BATCH', '128'))
NUM_BATCHES = int(os.environ.get('SASREC_BATCHES', '200'))
NUM_EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
NUM_NEGATIVES = int(os.environ.get('SASREC_NEGATIVES', '128'))
TEMPERATURE = float(os.environ.get('SASREC_TEMPERATURE', '0.1'))
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


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
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
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
        itemnum = max((max(x) for x in train.values() if x), default=0)
        itemnum = max(itemnum, max((max(x) for x in valid.values() if x), default=0))
    return train, valid, itemnum


class BatchSampler:
    def __init__(self, train, itemnum, maxlen, seed=42):
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.rng = np.random.RandomState(seed)
        self.users = [u for u, h in train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        if not self.users:
            raise ValueError('No eligible training users')
        self.histories = [np.asarray(train[u], dtype=np.int64) for u in self.users]
        self.seen = [np.unique(h) for h in self.histories]

    def sample(self, batch_size, k):
        seq = np.zeros((batch_size, self.maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.empty((batch_size, k), dtype=np.int64)
        selected = self.rng.randint(len(self.users), size=batch_size)
        for row, ix in enumerate(selected):
            history = self.histories[ix]
            length = min(len(history) - 1, self.maxlen)
            seq[row, -length:] = history[-length-1:-1]
            pos[row, -length:] = history[-length:]
            seen = self.seen[ix]
            # One independent uniform pool per sampled user, shared across positions.
            # Rejection excludes all training interactions, including future positives.
            if len(seen) > self.itemnum // 2:
                allowed = np.setdiff1d(np.arange(1, self.itemnum + 1), seen, assume_unique=True)
                neg[row] = self.rng.choice(allowed, size=k, replace=True)
            else:
                filled = 0
                while filled < k:
                    draws = self.rng.randint(1, self.itemnum + 1, size=max(32, 2 * (k - filled)))
                    draws = draws[~np.isin(draws, seen)]
                    take = min(len(draws), k - filled)
                    neg[row, filled:filled+take] = draws[:take]
                    filled += take
        return seq, pos, neg


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = F.relu(self.dropout1(outputs))
        outputs = self.dropout2(self.conv2(outputs)).transpose(-1, -2)
        return outputs + inputs


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.2, maxlen=128, normalize=True, temperature=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.normalize = normalize
        self.temperature = temperature
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1), persistent=False)

    def log2feats(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        features = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        mask = (seq != 0).unsqueeze(-1)
        features = self.emb_dropout(features) * mask
        for norm, attention, ff_norm, ff in zip(self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers):
            queries = norm(features)
            attended, _ = attention(queries, features, features, attn_mask=self.causal_mask[:length, :length], need_weights=False)
            features = queries + attended
            features = ff(ff_norm(features)) * mask
        return self.last_layernorm(features)

    def forward(self, seq, positives, negatives):
        features = self.log2feats(seq).float()
        positive_emb = self.item_emb(positives).float()
        negative_emb = self.item_emb(negatives).float()
        if self.normalize:
            features = F.normalize(features, dim=-1)
            positive_emb = F.normalize(positive_emb, dim=-1)
            negative_emb = F.normalize(negative_emb, dim=-1)
        positive_logits = (features * positive_emb).sum(-1, keepdim=True)
        # [B,L,K] avoids materializing [B,L,K,D] negative embeddings.
        negative_logits = torch.bmm(features, negative_emb.transpose(1, 2))
        logits = torch.cat([positive_logits, negative_logits], dim=-1)
        logits = logits[positives != 0] / self.temperature
        return F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=seq.device))

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :].float()
        embeddings = self.item_emb(candidates).float()
        if self.normalize:
            features = F.normalize(features, dim=-1)
            embeddings = F.normalize(embeddings, dim=-1)
        return torch.bmm(embeddings, features.unsqueeze(-1)).squeeze(-1) / self.temperature


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        history = train[user][-maxlen:]
        sequence = [0] * (maxlen - len(history)) + history
        target = valid[user][0]
        seen = set(train[user]) | set(valid[user])
        negatives = set()
        required = min(99, itemnum - len(seen))
        while len(negatives) < required:
            item = rng.randint(1, itemnum)
            if item not in seen:
                negatives.add(item)
        if required != 99:
            raise ValueError('Validation requires 99 unique unobserved negatives')
        sequences.append(sequence)
        candidates.append([target] + list(negatives))
    if not sequences:
        raise ValueError('No validation examples')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, validation, device):
    model.eval()
    sequences, candidates = validation
    ranks = []
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start+64], device=device)
        cand = torch.as_tensor(candidates[start:start+64], device=device)
        scores = model.predict(seq, cand).cpu().numpy()
        # Stable sorting gives candidate zero the same tie handling as baseline.
        order = np.argsort(-scores, axis=1, kind='stable')
        ranks.extend(np.argmax(order == 0, axis=1).tolist())
    ranks = np.asarray(ranks)
    return {'HR@10': float(np.mean(ranks < 10)),
            'HR@20': float(np.mean(ranks < 20)),
            'NDCG@10': float(np.mean(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0)))}


def train_one_dataset(name, device):
    seed_all(42)
    train, valid, itemnum = load_dataset(name)
    sampler = BatchSampler(train, itemnum, MAXLEN, seed=42)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
                normalize=True, temperature=TEMPERATURE)
    model = SASRec(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    checkpoint = os.path.join(WORKING_DIR, name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    print(f'[{name}] users={len(train)} items={itemnum} negatives={NUM_NEGATIVES} temperature={TEMPERATURE}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sampler.sample(BATCH_SIZE, NUM_NEGATIVES)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(seq, pos, neg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, validation, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{name}] epoch={epoch}/{NUM_EPOCHS} loss={total_loss.item()/NUM_BATCHES:.4f} val_score: {hr:.4f} HR@20={metrics["HR@20"]:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE * 5:
                print(f'[{name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        metrics = evaluate_val(model, validation, device)
        best_hr = metrics['HR@10']
        torch.save(model.state_dict(), checkpoint)
    with open(os.path.join(WORKING_DIR, name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer
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


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location('sampled_softmax_sasrec_runtime', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in DATASETS:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        # These saved options include normalization and training temperature.
        model = module.SASRec(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _models[name] = model
        _metas[name] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    _ensure_loaded()
    if not len(candidates):
        return []
    name = dataset_name if dataset_name is not None else DATASETS[0]
    if name not in _models:
        raise ValueError('Unknown dataset: ' + str(name))
    model, meta = _models[name], _metas[name]
    maxlen, itemnum = meta['maxlen'], meta['itemnum']
    hist = [int(item) if 0 < int(item) <= itemnum else 0 for item in history[-maxlen:]]
    sequence = [0] * (maxlen - len(hist)) + hist
    candidate_ids = [int(item) for item in candidates]
    valid = [0 < item <= itemnum for item in candidate_ids]
    safe_candidates = [item if ok else 0 for item, ok in zip(candidate_ids, valid)]
    with torch.inference_mode():
        seq = torch.tensor([sequence], dtype=torch.long, device=_device)
        cand = torch.tensor([safe_candidates], dtype=torch.long, device=_device)
        scores = model.predict(seq, cand)[0].cpu().tolist()
    return [float(score) if ok else -1e9 for score, ok in zip(scores, valid)]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f'Using {device}; normalized sampled-softmax SASRec', flush=True)
    scores = []
    for name in DATASETS:
        scores.append(train_one_dataset(name, device))
    write_predict_script(WORKING_DIR)
    print(f'All datasets done. Average val HR@10 = {sum(scores)/len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
