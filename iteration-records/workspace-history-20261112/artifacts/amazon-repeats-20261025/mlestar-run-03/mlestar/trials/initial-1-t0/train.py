from __future__ import annotations

import inspect
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
NUM_NEGATIVES = int(os.environ.get('GSASREC_NEGATIVES', '64'))
GBCE_T = float(os.environ.get('GSASREC_T', '0.75'))
SCORE_CHUNK = int(os.environ.get('GSASREC_SCORE_CHUNK', '512'))
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_training(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    itemnum = None
    if os.path.isfile(stats_path):
        with open(stats_path) as stream:
            stats = json.load(stream)
        if name in stats:
            itemnum = int(stats[name]['itemnum'])
    if itemnum is None:
        itemnum = max((max(items) for items in train.values() if items), default=0)
        itemnum = max(itemnum, max((max(items) for items in valid.values() if items), default=0))
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.dropout2(self.conv2(outputs))
        return outputs.transpose(-1, -2) + inputs


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1), persistent=False)

    def log2feats(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions))
        keep = seq.ne(0).unsqueeze(-1)
        x = x * keep
        mask = self.causal_mask[:length, :length]
        for norm, attention, ff_norm, ff in zip(self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers):
            x = x.transpose(0, 1)
            queries = norm(x)
            output, _ = attention(queries, x, x, attn_mask=mask, need_weights=False)
            x = (queries + output).transpose(0, 1)
            x = ff(ff_norm(x)) * keep
        return self.last_layernorm(x)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


class BatchSampler:
    def __init__(self, train, itemnum, maxlen, seed=42):
        self.rng = np.random.default_rng(seed)
        self.itemnum = itemnum
        self.histories = []
        sequences, positives = [], []
        for history in train.values():
            if len(history) < 2:
                continue
            unique = np.unique(np.asarray(history, dtype=np.int64))
            if len(unique) >= itemnum:
                continue
            sequence = np.zeros(maxlen, dtype=np.int64)
            positive = np.zeros(maxlen, dtype=np.int64)
            length = min(len(history) - 1, maxlen)
            sequence[-length:] = history[-length - 1:-1]
            positive[-length:] = history[-length:]
            sequences.append(sequence)
            positives.append(positive)
            self.histories.append(unique)
        if not sequences:
            raise ValueError('No eligible training sequences')
        self.sequences = np.stack(sequences)
        self.positives = np.stack(positives)
        self.valid_counts = np.count_nonzero(self.positives, axis=1)

    def sample(self, batch_size, k):
        indices = self.rng.integers(len(self.sequences), size=batch_size)
        seq = self.sequences[indices]
        pos = self.positives[indices]
        counts = self.valid_counts[indices]
        negatives = np.empty((int(counts.sum()), k), dtype=np.int64)
        offset = 0
        for index, count in zip(indices, counts):
            out = self.rng.integers(1, self.itemnum + 1, size=(int(count), k), dtype=np.int64)
            forbidden = self.histories[int(index)]
            # Reject all training interactions, including the positive target.
            # Repeated independent negatives are allowed by the sampling objective.
            bad = np.isin(out, forbidden)
            while bad.any():
                out[bad] = self.rng.integers(1, self.itemnum + 1, size=int(bad.sum()))
                bad = np.isin(out, forbidden)
            negatives[offset:offset + int(count)] = out
            offset += int(count)
        return seq, pos, negatives


def generalized_bce(model, features, positives, negatives):
    k = negatives.shape[1]
    beta = 1.0 - GBCE_T + (k / max(model.itemnum - 1, 1)) * GBCE_T
    positive_logits = (features * model.item_emb(positives)).sum(-1).float()
    total = beta * F.softplus(-positive_logits).sum()

    def negative_sum(feats, ids):
        embeddings = model.item_emb(ids)
        logits = (embeddings * feats.unsqueeze(1)).sum(-1)
        return F.softplus(logits.float()).sum()

    # Recompute each scoring chunk during backward, retaining only its small
    # inputs rather than N*K*hidden_dim negative embedding activations.
    for start in range(0, len(features), SCORE_CHUNK):
        stop = start + SCORE_CHUNK
        total = total + checkpoint(negative_sum, features[start:stop], negatives[start:stop],
                                   use_reentrant=False, preserve_rng_state=False)
    return total / (len(features) * (k + 1))


def validation_cases(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    seqs, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        seq = np.zeros(maxlen, dtype=np.int64)
        history = train[user][-maxlen:]
        if history:
            seq[-len(history):] = history
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError('Validation requires 99 unique available negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        seqs.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    if not seqs:
        raise ValueError('No validation users')
    return np.stack(seqs), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    sequences, candidates = cases
    ranks = []
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start + 64], device=device)
        cand = torch.as_tensor(candidates[start:start + 64], device=device)
        scores = model.predict(seq, cand).float().cpu().numpy()
        # Stable candidate-order tie breaking keeps the positive at index zero.
        order = np.argsort(-scores, axis=1, kind='stable')
        ranks.extend(np.argmax(order == 0, axis=1).tolist())
    rank = np.asarray(ranks)
    hr = float(np.mean(rank < 10))
    ndcg = float(np.mean(np.where(rank < 10, 1.0 / np.log2(rank + 2), 0.0)))
    return hr, ndcg


def train_one_dataset(name, device):
    seed_training(42)
    train, valid, itemnum = load_dataset(name)
    print(f'[{name}] users={len(train)} items={itemnum}', flush=True)
    sampler = BatchSampler(train, itemnum, MAXLEN, seed=42)
    cases = validation_cases(train, valid, itemnum, MAXLEN)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                 betas=(0.9, 0.98))
    path = os.path.join(WORKING_DIR, name + '_model.pt')
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_total = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sampler.sample(BATCH_SIZE, NUM_NEGATIVES)
            seq_t = torch.as_tensor(seq, device=device)
            pos_t = torch.as_tensor(pos, device=device)
            neg_t = torch.as_tensor(neg, device=device)
            optimizer.zero_grad(set_to_none=True)
            features = model(seq_t)
            mask = pos_t.ne(0)
            loss = generalized_bce(model, features[mask], pos_t[mask], neg_t)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {name}, epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach())
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, cases, device)
            print(f'[{name}] epoch={epoch}/{NUM_EPOCHS} loss={loss_total / NUM_BATCHES:.6f} '
                  f'val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({'config': config, 'state_dict': state, 'epoch': epoch,
                            'validation_hr10': hr}, path)
            if epoch - best_epoch >= PATIENCE * 5:
                print(f'[{name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        hr, _ = evaluate_val(model, cases, device)
        best_hr = hr
        torch.save({'config': config,
                    'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    'epoch': 0, 'validation_hr10': hr}, path)
    print(f'[{name}] best validation HR@10={best_hr:.4f}, epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script():
    # Embed the exact trained architecture so inference has no dependency on
    # importing training configuration or on access to the input data directory.
    header = 'from __future__ import annotations\nimport math\nimport os\nimport torch\nimport torch.nn as nn\n\n'
    architecture = inspect.getsource(PointWiseFeedForward) + '\n\n' + inspect.getsource(SASRec)
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)


def _load(name):
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        path = os.path.join(_HERE, name + '_model.pt')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = SASRec(**payload['config'])
        model.load_state_dict(payload['state_dict'])
        model.to(_DEVICE).eval()
        _MODELS[name] = model
    return _MODELS[name]


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(name)
    hist = list(history)[-model.maxlen:]
    sequence = [0] * (model.maxlen - len(hist)) + hist
    seq_t = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    with torch.inference_mode():
        return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
'''
    with open(os.path.join(WORKING_DIR, 'predict.py'), 'w') as stream:
        stream.write(header + architecture + '\n\n' + runtime)


def main():
    if NUM_NEGATIVES < 1 or SCORE_CHUNK < 1:
        raise ValueError('Negative count and scoring chunk must be positive')
    if not 0.0 <= GBCE_T <= 1.0:
        raise ValueError('GSASREC_T must be in [0, 1]')
    if NUM_EPOCHS < 1 or NUM_BATCHES < 1:
        raise ValueError('Epoch and batch counts must be positive')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_training(42)
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device={device}; negatives={NUM_NEGATIVES}; t={GBCE_T}', flush=True)
    scores = {}
    for name in DATASETS:
        scores[name] = train_one_dataset(name, device)
    write_predict_script()
    macro = sum(scores.values()) / len(DATASETS)
    with open(os.path.join(WORKING_DIR, 'validation_summary.json'), 'w') as stream:
        json.dump({'per_dataset_validation_hr10': scores,
                   'macro_validation_hr10': macro,
                   'training_seed': 42, 'negatives': NUM_NEGATIVES, 'gbce_t': GBCE_T}, stream, indent=2)
    print(f'All four datasets complete. Macro validation HR@10={macro:.4f}', flush=True)


if __name__ == '__main__':
    main()
