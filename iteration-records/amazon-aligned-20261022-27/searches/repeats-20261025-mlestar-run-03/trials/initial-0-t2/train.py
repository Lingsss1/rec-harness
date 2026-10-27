from __future__ import annotations

import inspect
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN = int(os.environ.get('SASREC_HIDDEN', '64'))
BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT = float(os.environ.get('SASREC_DROPOUT', '0.5'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WD = float(os.environ.get('SASREC_WD', '0.0'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
BATCHES = int(os.environ.get('SASREC_BATCHES', '100'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '30'))
CL_WEIGHT = 0.1
TEMPERATURE = 0.2
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything(seed=42):
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
        itemnum = int(stats.get(name, {}).get('itemnum', 0))
    if not itemnum:
        itemnum = max((max(items) for items in train.values() if items), default=0)
        itemnum = max(itemnum, max((max(items) for items in valid.values() if items), default=0))
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.conv1(inputs.transpose(1, 2))
        x = F.relu(self.dropout1(x))
        x = self.dropout2(self.conv2(x)).transpose(1, 2)
        return inputs + x


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate,
                                  batch_first=True) for _ in range(num_blocks)])
        self.forward_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)])
        self.forward_layers = nn.ModuleList([
            PointWiseFeedForward(hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.ones(maxlen, maxlen,
                             dtype=torch.bool).triu(1), persistent=False)

    def log2feats(self, seq):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        keep = seq.ne(0).unsqueeze(-1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions)) * keep
        for norm, attn, fnorm, ff in zip(self.attention_layernorms,
                self.attention_layers, self.forward_layernorms, self.forward_layers):
            q = norm(x)
            attended = attn(q, x, x, attn_mask=self.causal_mask[:length, :length],
                            need_weights=False)[0]
            x = ff(fnorm(q + attended)) * keep
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(-1)


def dropout_nce(a, b, target, tau=TEMPERATURE):
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    logits = a @ b.T / tau
    eye = torch.eye(len(target), dtype=torch.bool, device=target.device)
    false_negative = target[:, None].eq(target[None, :]) & ~eye
    logits = logits.masked_fill(false_negative, float('-inf'))
    labels = torch.arange(len(target), device=target.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class TrainingSampler:
    def __init__(self, train, itemnum):
        histories = [h for h in train.values()
                     if len(h) >= 2 and len(set(h)) < itemnum]
        if not histories:
            raise ValueError('No eligible training sequences')
        self.itemnum = itemnum
        self.rng = np.random.default_rng(42)
        self.seq = np.zeros((len(histories), MAXLEN), dtype=np.int32)
        self.pos = np.zeros_like(self.seq)
        keys = []
        stride = itemnum + 1
        for row, hist in enumerate(histories):
            n = min(len(hist) - 1, MAXLEN)
            self.seq[row, -n:] = hist[-n-1:-1]
            self.pos[row, -n:] = hist[-n:]
            keys.append(np.asarray(sorted(set(hist)), dtype=np.int64) + row * stride)
        self.keys = np.sort(np.concatenate(keys))

    def sample(self):
        rows = self.rng.integers(len(self.seq), size=BATCH)
        seq, pos = self.seq[rows], self.pos[rows]
        neg = self.rng.integers(1, self.itemnum + 1, size=pos.shape, dtype=np.int64)
        active = pos != 0
        offsets = rows[:, None].astype(np.int64) * (self.itemnum + 1)
        while True:
            encoded = offsets + neg
            ix = np.searchsorted(self.keys, encoded)
            bad = active & (self.keys[np.minimum(ix, len(self.keys) - 1)] == encoded)
            if not bad.any():
                break
            neg[bad] = self.rng.integers(1, self.itemnum + 1, size=int(bad.sum()))
        neg[~active] = 0
        return seq, pos, neg


def validation_cases(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        if not train[u] or not valid[u]:
            continue
        history = train[u][-MAXLEN:]
        seq = [0] * (MAXLEN - len(history)) + history
        excluded = set(train[u]) | set(valid[u])
        count = min(99, itemnum - len(excluded))
        negatives = set()
        while len(negatives) < count:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[u][0]] + list(negatives))
    return sequences, candidates


@torch.no_grad()
def evaluate(model, cases, device):
    model.eval()
    hits = ndcg = 0.0
    sequences, candidates = cases
    for start in range(0, len(sequences), 100):
        ss = sequences[start:start + 100]
        cc = candidates[start:start + 100]
        # Real benchmark categories have at least 99 available negatives.
        if len({len(c) for c in cc}) != 1:
            for seq, cand in zip(ss, cc):
                scores = model.predict(torch.tensor([seq], device=device),
                                       torch.tensor([cand], device=device))[0]
                rank = int((scores[1:] > scores[0]).sum().item())
                hits += rank < 10
                ndcg += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
        else:
            scores = model.predict(torch.tensor(ss, dtype=torch.long, device=device),
                                   torch.tensor(cc, dtype=torch.long, device=device))
            ranks = (scores[:, 1:] > scores[:, :1]).sum(1).cpu().tolist()
            for rank in ranks:
                hits += rank < 10
                ndcg += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
    n = max(1, len(sequences))
    return hits / n, ndcg / n


def train_one_dataset(name, device):
    seed_everything(42)
    train, valid, itemnum = load_dataset(name)
    sampler = TrainingSampler(train, itemnum)
    cases = validation_cases(train, valid, itemnum)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS,
                  num_heads=HEADS, dropout_rate=DROPOUT, maxlen=MAXLEN)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98),
                                 weight_decay=WD)
    best = (-1.0, -1.0)
    best_epoch = 0
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    print(f'[{name}] users={len(train)} items={itemnum}', flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = rec_sum = cl_sum = 0.0
        for _ in range(BATCHES):
            seq, pos, neg = sampler.sample()
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            first = model.log2feats(seq)
            active = pos.ne(0)
            features = first[active]
            positive = (features * model.item_emb(pos[active])).sum(-1)
            negative = (features * model.item_emb(neg[active])).sum(-1)
            recommendation_loss = F.softplus(-positive).mean() + F.softplus(negative).mean()
            # The prefix excludes its final training target. Both passes use
            # independent dropout masks; no validation labels enter this loss.
            second = model.log2feats(seq)
            contrastive_loss = dropout_nce(first[:, -1], second[:, -1], pos[:, -1])
            loss = recommendation_loss + CL_WEIGHT * contrastive_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach().item()
            rec_sum += recommendation_loss.detach().item()
            cl_sum += contrastive_loss.detach().item()
        if epoch % 5 == 0 or epoch == EPOCHS:
            score = evaluate(model, cases, device)
            print(f'[{name}] epoch={epoch} loss={loss_sum/BATCHES:.4f} '
                  f'bce={rec_sum/BATCHES:.4f} nce={cl_sum/BATCHES:.4f} '
                  f'val_HR@10={score[0]:.4f} val_NDCG@10={score[1]:.4f}', flush=True)
            if score > best:
                best, best_epoch = score, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        torch.save(model.state_dict(), checkpoint)
    with open(os.path.join(WORKING, name + '_meta.json'), 'w') as f:
        json.dump(config, f)
    print(f'[{name}] best_validation_HR@10={best[0]:.4f} epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best[0]


def write_predict_script():
    # Export the exact trained architecture without importing training data or
    # relying on the caller's Python module search path during inference.
    architecture = ('import math\nimport torch\nfrom torch import nn\n'
                    'from torch.nn import functional as F\n\n')
    architecture += inspect.getsource(PointWiseFeedForward) + '\n\n'
    architecture += inspect.getsource(SASRec) + '\n'
    with open(os.path.join(WORKING, 'sasrec_model.py'), 'w') as f:
        f.write(architecture)
    predictor = '''from __future__ import annotations
import importlib.util
import json
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _load(dataset):
    global _device, _model_class
    if dataset in _models:
        return
    if _device is None:
        torch.set_num_threads(4)
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _model_class is None:
        spec = importlib.util.spec_from_file_location(
            'duorec_architecture', os.path.join(_HERE, 'sasrec_model.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.SASRec
    with open(os.path.join(_HERE, dataset + '_meta.json')) as f:
        meta = json.load(f)
    model = _model_class(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if len(candidates) == 0:
        return []
    _load(dataset)
    meta = _metas[dataset]
    history = list(history)[-meta['maxlen']:]
    seq = [0] * (meta['maxlen'] - len(history)) + history
    with torch.inference_mode():
        scores = _models[dataset].predict(
            torch.tensor([seq], dtype=torch.long, device=_device),
            torch.tensor([list(candidates)], dtype=torch.long, device=_device))
    return scores[0].cpu().tolist()
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as f:
        f.write(predictor)


def main():
    torch.set_num_threads(4)
    seed_everything(42)
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = []
    for name in DATASETS:
        scores.append(train_one_dataset(name, device))
    write_predict_script()
    print(f'Macro-average best validation HR@10: {sum(scores)/len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
