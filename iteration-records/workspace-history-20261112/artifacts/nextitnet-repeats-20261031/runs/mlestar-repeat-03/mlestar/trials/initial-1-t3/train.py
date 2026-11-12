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

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = float(os.environ.get('NEXTITNET_DROPOUT', '0.2'))
LR = float(os.environ.get('NEXTITNET_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('NEXTITNET_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('NEXTITNET_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = 5
PATIENCE = 20
CONTRASTIVE_WEIGHT = 0.05
TEMPERATURE = 0.2


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if itemnum == 0:
        itemnum = max((i for split in (train, valid) for items in split.values() for i in items), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('Cannot sample an unseen negative for a training user')
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            sequences[row, idx] = item
            positives[row, idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            negatives[row, idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
    return sequences, positives, negatives


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate) for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = seq.ne(0).unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x) * mask
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if features.ndim == 3 and items.ndim == 2:
            return (features * embeddings).sum(-1)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)

    @staticmethod
    def final_features(features, seq):
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return (features * self.item_emb(pos)).sum(-1), (features * self.item_emb(neg)).sum(-1)

    def score_candidates(self, seq, candidates):
        features = self.final_features(self.encode_sequence(seq), seq)
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 else scores


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    mask = pos.ne(0)
    return F.softplus(negative - positive)[mask].mean()


def training_loss(model, seq, pos, neg):
    # Both stochastic views encode only causal training inputs using the same parameters.
    view1 = model.encode_sequence(seq)
    positive = (view1 * model.item_emb(pos)).sum(-1)
    negative = (view1 * model.item_emb(neg)).sum(-1)
    bpr = F.softplus(negative - positive)[pos.ne(0)].mean()
    view2 = model.encode_sequence(seq)
    z1 = F.normalize(model.final_features(view1, seq), dim=-1)
    z2 = F.normalize(model.final_features(view2, seq), dim=-1)
    z = torch.cat((z1, z2), dim=0)
    batch = seq.size(0)
    n = 2 * batch
    rows = torch.arange(n, device=seq.device)
    paired = (rows + batch) % n
    positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
    last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
    targets = pos[torch.arange(batch, device=seq.device), last].repeat(2)
    excluded = targets[:, None].eq(targets[None, :])
    # Keep the paired dropout view as the sole positive, even for repeated targets.
    excluded[rows, paired] = False
    excluded[rows, rows] = True
    logits = (z.float() @ z.float().T) / TEMPERATURE
    logits = logits.masked_fill(excluded, float('-inf'))
    contrastive = F.cross_entropy(logits, paired)
    return bpr + CONTRASTIVE_WEIGHT * contrastive, bpr.detach(), contrastive.detach()


def make_validation_cases(user_train, user_valid, itemnum, maxlen):
    # A fixed validation panel avoids changing training RNG or checkpoint comparisons.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        candidates = [target]
        count = min(99, max(0, itemnum - len({i for i in excluded if 1 <= i <= itemnum})))
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                candidates.append(item)
                excluded.add(item)
        seq = np.zeros(maxlen, dtype=np.int64)
        tail = history[-maxlen:]
        if tail:
            seq[-len(tail):] = tail
        cases.append((seq, candidates))
    return cases


def evaluate_val(model, cases, device):
    model.eval()
    hits = 0.0
    ndcg = 0.0
    with torch.no_grad():
        for seq, candidates in cases:
            seq_t = torch.as_tensor(seq[None, :], dtype=torch.long, device=device)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    model.train()
    return hits / max(1, len(cases)), ndcg / max(1, len(cases))


def train_one_dataset(dataset_name, device, working_dir):
    seed_all()
    user_train, user_valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    if not user_train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(user_train, user_valid, itemnum, MAXLEN)
    ckpt_path = os.path.join(working_dir, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        for _ in range(NUM_BATCHES):
            batch = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, bpr, contrastive = training_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [loss.item(), bpr.item(), contrastive.item()]
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, cases, device)
            total, bpr_avg, cl_avg = totals / NUM_BATCHES
            print(f'[{dataset_name}] epoch={epoch} loss={total:.4f} bpr={bpr_avg:.4f} contrastive={cl_avg:.4f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(working_dir, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained class definitions, with no training-module dependency.
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    classes = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = NextItNetModel(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    _ensure_loaded(ds)
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    tail = list(history[-maxlen:])
    seq = [0] * (maxlen - len(tail)) + tail
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
    return [float(s) for s in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + classes + '\n\n' + runtime)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    write_predict_script(working_dir)
    scores = [train_one_dataset(ds, device, working_dir) for ds in DATASETS]
    print(f'Macro validation HR@10: {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
