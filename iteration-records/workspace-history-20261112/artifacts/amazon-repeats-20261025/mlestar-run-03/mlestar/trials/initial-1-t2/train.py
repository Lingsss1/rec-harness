from __future__ import annotations
import os
import json
import math
import random
import inspect
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN = int(os.environ.get('SASREC_HIDDEN', '64'))
BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT = float(os.environ.get('SASREC_DROPOUT', '0.4'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WD = float(os.environ.get('SASREC_WD', '0.0'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
BATCHES = int(os.environ.get('SASREC_BATCHES', '100'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '12'))
CL_WEIGHT = float(os.environ.get('DUOREC_WEIGHT', '0.05'))
TEMPERATURE = float(os.environ.get('DUOREC_TAU', '0.2'))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_training():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def read_split(path):
    result = defaultdict(list)
    with open(path) as handle:
        for line in handle:
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
        with open(stats_path) as handle:
            stats = json.load(handle)
        itemnum = int(stats.get(name, {}).get('itemnum', 0))
    if not itemnum:
        itemnum = max((max(items) for split in (train, valid) for items in split.values() if items), default=0)
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.conv1(inputs.transpose(-1, -2))
        x = F.relu(self.dropout1(x))
        x = self.dropout2(self.conv2(x))
        return inputs + x.transpose(-1, -2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1, dropout_rate=0.4, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0.0, 1.0)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
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
        self.register_buffer('causal_mask', torch.ones(maxlen, maxlen, dtype=torch.bool).triu(1), persistent=False)

    def log2feats(self, seq):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device)
        mask = seq.ne(0).unsqueeze(-1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions)) * mask
        for norm, attn, ffn_norm, ffn in zip(self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers):
            q = norm(x)
            attended = attn(q, x, x, attn_mask=self.causal_mask[:length, :length], need_weights=False)[0]
            x = ffn(ffn_norm(q + attended)) * mask
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        final = self.log2feats(seq)[:, -1, :]
        return (self.item_emb(candidates) * final.unsqueeze(1)).sum(-1)


def dropout_nce(a, b, target):
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    logits = a @ b.T / TEMPERATURE
    eye = torch.eye(target.numel(), device=target.device, dtype=torch.bool)
    false_negative = target[:, None].eq(target[None, :]) & ~eye
    logits = logits.masked_fill(false_negative, float('-inf'))
    labels = torch.arange(target.numel(), device=target.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class BatchSampler:
    def __init__(self, train, itemnum):
        users = [u for u, history in train.items() if len(history) >= 2 and len(set(history)) < itemnum]
        if not users:
            raise ValueError('No eligible training users')
        self.seq = np.zeros((len(users), MAXLEN), dtype=np.int64)
        self.pos = np.zeros_like(self.seq)
        self.seen = []
        self.itemnum = itemnum
        for row, u in enumerate(users):
            history = train[u]
            count = min(MAXLEN, len(history) - 1)
            self.seq[row, -count:] = history[-count - 1:-1]
            self.pos[row, -count:] = history[-count:]
            self.seen.append(np.asarray(list(set(history)), dtype=np.int64))

    def sample(self):
        rows = np.random.randint(len(self.seq), size=BATCH)
        seq = self.seq[rows]
        pos = self.pos[rows]
        neg = np.random.randint(1, self.itemnum + 1, size=pos.shape, dtype=np.int64)
        for i, row in enumerate(rows):
            active = pos[i] != 0
            bad = np.isin(neg[i], self.seen[row]) & active
            while bad.any():
                neg[i, bad] = np.random.randint(1, self.itemnum + 1, size=int(bad.sum()))
                bad = np.isin(neg[i], self.seen[row]) & active
        neg[pos == 0] = 0
        return seq, pos, neg


def validation_data(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        if not train[u] or not valid[u]:
            continue
        excluded = set(train[u]) | set(valid[u])
        if itemnum - len(excluded) < 99:
            raise ValueError('Insufficient unique validation negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        history = train[u][-MAXLEN:]
        sequences.append([0] * (MAXLEN - len(history)) + history)
        candidates.append([valid[u][0]] + list(negatives))
    if not sequences:
        raise ValueError('No validation users')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    seq, cand = validation
    ranks = []
    for start in range(0, len(seq), 100):
        s = torch.as_tensor(seq[start:start + 100], device=device)
        c = torch.as_tensor(cand[start:start + 100], device=device)
        scores = model.predict(s, c)
        # Candidate zero is the positive; stable ties favor its original position.
        rank = (scores[:, 1:] > scores[:, :1]).sum(1)
        ranks.extend(rank.cpu().tolist())
    ranks = np.asarray(ranks)
    return float(np.mean(ranks < 10)), float(np.mean(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0.0)))


def train_one_dataset(name, device):
    train, valid, itemnum = load_dataset(name)
    sampler = BatchSampler(train, itemnum)
    validation = validation_data(train, valid, itemnum)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS, num_heads=HEADS, dropout_rate=DROPOUT, maxlen=MAXLEN)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD)
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    print(f'[{name}] users={len(train)} items={itemnum}', flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = np.zeros(2, dtype=np.float64)
        for _ in range(BATCHES):
            arrays = sampler.sample()
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            features = model.log2feats(seq)
            active = pos.ne(0)
            selected = features[active]
            positive = (selected * model.item_emb(pos[active])).sum(-1)
            negative = (selected * model.item_emb(neg[active])).sum(-1)
            rec_loss = F.softplus(-positive).mean() + F.softplus(negative).mean()
            second = model.log2feats(seq)
            cl_loss = dropout_nce(features[:, -1], second[:, -1], pos[:, -1])
            loss = rec_loss + CL_WEIGHT * cl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses += [float(rec_loss.detach()), float(cl_loss.detach())]
        if epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} bce={losses[0]/BATCHES:.4f} cl={losses[1]/BATCHES:.4f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE * 5:
                break
    with open(os.path.join(WORKING, name + '_meta.json'), 'w') as handle:
        json.dump(config, handle)
    print(f'[{name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script():
    # Embed the exact architecture so prediction does not import training or data.
    header = 'from __future__ import annotations\nimport os\nimport json\nimport math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(PointWiseFeedForward) + '\n\n' + inspect.getsource(SASRec)
    predictor = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.json')) as handle:
        meta = json.load(handle)
    model = SASRec(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


@torch.no_grad()
def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    _ensure_loaded(name)
    meta = _metas[name]
    maxlen = meta['maxlen']
    hist = [int(i) if 0 < int(i) <= meta['itemnum'] else 0 for i in history[-maxlen:]]
    seq = torch.tensor([[0] * (maxlen - len(hist)) + hist], dtype=torch.long, device=_device)
    raw_candidates = torch.tensor([candidates], dtype=torch.long, device=_device)
    valid = (raw_candidates > 0) & (raw_candidates <= meta['itemnum'])
    safe = raw_candidates.clamp(0, meta['itemnum'])
    scores = _models[name].predict(seq, safe)
    scores = scores.masked_fill(~valid, float('-inf'))
    return scores[0].cpu().tolist()
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as handle:
        handle.write(header + architecture + '\n' + predictor)


def main():
    seed_training()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device={device}; auxiliary weight={CL_WEIGHT}; tau={TEMPERATURE}', flush=True)
    scores = [train_one_dataset(name, device) for name in DATASETS]
    print(f'Macro validation HR@10={np.mean(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
