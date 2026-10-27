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
HIDDEN = int(os.environ.get('SASREC_HIDDEN', '64'))
BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT = float(os.environ.get('SASREC_DROPOUT', '0.5'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WD = float(os.environ.get('SASREC_WD', '0.0'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
BATCHES = int(os.environ.get('SASREC_BATCHES', '200'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
NEGATIVES = int(os.environ.get('GSASREC_NEGATIVES', '64'))
GBCE_T = float(os.environ.get('GSASREC_T', '0.75'))
NEG_CHUNK = int(os.environ.get('GSASREC_SCORE_CHUNK', os.environ.get('GSASREC_CHUNK', '512')))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


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
    itemnum = max((max(items) for split in (train, valid)
                   for items in split.values() if items), default=0)
    path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as stream:
            stats = json.load(stream)
        itemnum = max(itemnum, int(stats.get(name, {}).get('itemnum', 0)))
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        x = self.dropout2(self.conv2(F.relu(x)))
        return inputs + x.transpose(-1, -2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0.0, 1.0)
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
        self.register_buffer('causal_mask', torch.ones(maxlen, maxlen, dtype=torch.bool).triu(1), persistent=False)

    def log2feats(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions))
        keep = seq.ne(0).unsqueeze(-1)
        x = x * keep
        mask = self.causal_mask[:length, :length]
        for norm, attention, fnorm, feed in zip(self.attention_layernorms, self.attention_layers,
                                               self.forward_layernorms, self.forward_layers):
            x = x.transpose(0, 1)
            queries = norm(x)
            output, _ = attention(queries, x, x, attn_mask=mask, need_weights=False)
            x = (queries + output).transpose(0, 1)
            x = feed(fnorm(x)) * keep
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


def generalized_bce(model, seq, positive, negatives):
    mask = positive.ne(0)
    features = model.log2feats(seq)[mask]
    logits = (features * model.item_emb(positive[mask])).sum(-1).float()
    k = negatives.shape[1]
    beta = 1.0 - GBCE_T + (k / max(model.itemnum - 1, 1)) * GBCE_T
    total = beta * F.softplus(-logits).sum()

    def negative_sum(feats, ids):
        embeddings = model.item_emb(ids)
        scores = (feats.unsqueeze(1) * embeddings).sum(-1)
        return F.softplus(scores.float()).sum()

    for start in range(0, len(features), NEG_CHUNK):
        total = total + checkpoint(negative_sum, features[start:start + NEG_CHUNK],
                                   negatives[start:start + NEG_CHUNK],
                                   use_reentrant=False, preserve_rng_state=False)
    return total / (len(features) * (k + 1))


class Sampler:
    def __init__(self, train, itemnum, maxlen, member):
        self.itemnum = itemnum
        self.rng = np.random if member == 0 else np.random.default_rng(42)
        self.member = member
        histories = [h for h in train.values() if len(h) >= 2 and len(set(h)) < itemnum]
        if not histories:
            raise ValueError('No eligible training sequences')
        self.seq = np.zeros((len(histories), maxlen), dtype=np.int64)
        self.pos = np.zeros_like(self.seq)
        self.seen = []
        self.counts = []
        for row, history in enumerate(histories):
            n = min(maxlen, len(history) - 1)
            self.seq[row, -n:] = history[-n-1:-1]
            self.pos[row, -n:] = history[-n:]
            self.seen.append(np.unique(np.asarray(history, dtype=np.int64)))
            self.counts.append(n)

    def integers(self, low, high, size):
        if self.member == 0:
            return self.rng.randint(low, high, size=size)
        return self.rng.integers(low, high, size=size, dtype=np.int64)

    def sample(self):
        rows = self.integers(0, len(self.seq), BATCH)
        negatives = np.empty((sum(self.counts[row] for row in rows), NEGATIVES), dtype=np.int64)
        offset = 0
        for row in rows:
            count = self.counts[row]
            out = self.integers(1, self.itemnum + 1, (count, NEGATIVES))
            bad = np.isin(out, self.seen[row])
            while bad.any():
                out[bad] = self.integers(1, self.itemnum + 1, int(bad.sum()))
                bad = np.isin(out, self.seen[row])
            negatives[offset:offset + count] = out
            offset += count
        return self.seq[rows], self.pos[rows], negatives


def validation_cases(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        seq = np.zeros(MAXLEN, dtype=np.int64)
        history = train[user][-MAXLEN:]
        if history:
            seq[-len(history):] = history
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError('Validation requires 99 available unique negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError('No validation users')
    return np.stack(sequences), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def validation_scores(model, cases, device):
    model.eval()
    sequences, candidates = cases
    result = []
    for start in range(0, len(sequences), 32):
        seq = torch.as_tensor(sequences[start:start + 32], dtype=torch.long, device=device)
        cand = torch.as_tensor(candidates[start:start + 32], dtype=torch.long, device=device)
        result.append(model.predict(seq, cand).float().cpu().numpy())
    return np.concatenate(result)


def metrics(scores):
    rank = (scores[:, 1:] > scores[:, :1]).sum(axis=1)
    hit = rank < 10
    return float(hit.mean()), float((hit / np.log2(rank + 2)).mean())


def member_path(name, member):
    return os.path.join(WORKING, name + '_member' + str(member) + '.pt')


def train_member(name, member, device):
    if member == 1:
        seed_all()
    train, valid, itemnum = load_dataset(name)
    sampler = Sampler(train, itemnum, MAXLEN, member)
    cases = validation_cases(train, valid, itemnum)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS,
                  num_heads=HEADS, dropout_rate=DROPOUT, maxlen=MAXLEN)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD)
    best_hr, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for _ in range(BATCHES):
            seq, pos, neg = sampler.sample()
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = generalized_bce(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {name}, member={member}, epoch={epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach()
        if epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = metrics(validation_scores(model, cases, device))
            print(f'[{name}/{member}] epoch={epoch} loss={loss_sum.item()/BATCHES:.6f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({'config': config, 'state_dict': state, 'epoch': epoch,
                            'validation_hr10': hr}, member_path(name, member))
            if epoch - best_epoch >= PATIENCE * 5:
                break
    print(f'[{name}/{member}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def merge_dataset(name, device):
    train, valid, itemnum = load_dataset(name)
    cases = validation_cases(train, valid, itemnum)
    payloads, scores = [], []
    for member in (0, 1):
        payload = torch.load(member_path(name, member), map_location='cpu', weights_only=True)
        payloads.append(payload)
        model = SASRec(**payload['config']).to(device)
        model.load_state_dict(payload['state_dict'])
        scores.append(validation_scores(model, cases, device))
        del model
    best_hr, best_weight = -1.0, 0.5
    # Prefer the equal blend when validation HR is tied; endpoints preserve
    # the option to use either original training procedure by itself.
    for weight in (0.5, 0.25, 0.75, 0.0, 1.0):
        hr, ndcg = metrics(weight * scores[0] + (1.0 - weight) * scores[1])
        print(f'[{name}] member0_weight={weight:.2f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
        if hr > best_hr:
            best_hr, best_weight = hr, weight
    weights = [best_weight, 1.0 - best_weight]
    active = [(payload, weight) for payload, weight in zip(payloads, weights) if weight > 0]
    torch.save({'members': [payload for payload, weight in active],
                'weights': [weight for payload, weight in active],
                'validation_hr10': best_hr}, os.path.join(WORKING, name + '_model.pt'))
    return {'validation_hr10': best_hr, 'member0_weight': best_weight,
            'member_validation_hr10': [p['validation_hr10'] for p in payloads]}


def write_predict_script():
    header = 'from __future__ import annotations\nimport math\nimport os\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(PointWiseFeedForward) + '\n\n' + inspect.getsource(SASRec)
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(4, os.cpu_count() or 1))
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _load(name):
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        payload = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location='cpu', weights_only=True)
        members = []
        for member, weight in zip(payload['members'], payload['weights']):
            model = SASRec(**member['config'])
            model.load_state_dict(member['state_dict'])
            model.to(_DEVICE).eval()
            members.append((model, float(weight)))
        _MODELS[name] = members
    return _MODELS[name]


def predict(user_id, history, candidates, *, dataset_name=None):
    items = list(candidates)
    if not items:
        return []
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    members = _load(name)
    history = list(history)
    cand = torch.tensor([items], dtype=torch.long, device=_DEVICE)
    total = None
    with torch.inference_mode():
        for model, weight in members:
            hist = history[-model.maxlen:]
            sequence = [0] * (model.maxlen - len(hist)) + hist
            seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
            scores = weight * model.predict(seq, cand).float()
            total = scores if total is None else total + scores
    return total[0].cpu().tolist()
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as stream:
        stream.write(header + architecture + '\n\n' + runtime)


def main():
    if min(NEGATIVES, NEG_CHUNK, EPOCHS, BATCHES, BATCH, MAXLEN, PATIENCE) < 1:
        raise ValueError('Training counts must be positive')
    if not 0.0 <= GBCE_T <= 1.0:
        raise ValueError('GSASREC_T must be in [0, 1]')
    if HIDDEN < 1 or BLOCKS < 1 or HEADS < 1 or HIDDEN % HEADS:
        raise ValueError('Invalid architecture dimensions')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_all()
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device={device}; negatives={NEGATIVES}; gBCE_t={GBCE_T}; epochs={EPOCHS}; members=2', flush=True)
    # Complete the global RNG trajectory before resetting seeds for member 1.
    for member in (0, 1):
        for name in DATASETS:
            train_member(name, member, device)
    details = {name: merge_dataset(name, device) for name in DATASETS}
    write_predict_script()
    scores = {name: result['validation_hr10'] for name, result in details.items()}
    macro = sum(scores.values()) / len(DATASETS)
    with open(os.path.join(WORKING, 'validation_summary.json'), 'w') as stream:
        json.dump({'per_dataset_validation_HR@10': scores,
                   'macro_validation_HR@10': macro,
                   'ensemble_details': details, 'training_seed': 42,
                   'negatives': NEGATIVES, 'gbce_t': GBCE_T}, stream, indent=2)
    print(f'Macro validation HR@10={macro:.6f}', flush=True)


if __name__ == '__main__':
    main()
