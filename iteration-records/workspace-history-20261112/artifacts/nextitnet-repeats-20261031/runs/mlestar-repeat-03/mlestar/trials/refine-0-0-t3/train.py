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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
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
CL_WEIGHT = 0.005
CL_TEMPERATURE = 0.2
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 2:
                data[int(fields[0])].append(int(fields[1]))
    return dict(data)


def _load_itemnum(data_dir, dataset_name):
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) == 2:
                    itemnum = max(itemnum, int(fields[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    for row in range(batch_size):
        history = user_train[random.choice(users)]
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('No unseen negative available for sampled user')
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[row, idx] = item
            pos[row, idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            neg[row, idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
    return seq, pos, neg


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        residual = x
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return residual + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
            for i in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        mask = seq.ne(0)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * mask.long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def last_features(self, seq, features):
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp(min=0)
        return features[torch.arange(seq.size(0), device=seq.device), last]

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        features = self.last_features(seq, self.encode_sequence(seq))
        scores = self.predict_score(features, candidates)
        return scores[0] if seq.size(0) == 1 else scores


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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def bpr_sequence_loss(model, seq, pos, neg):
    positive, negative = model(seq, pos, neg)
    return F.softplus(negative - positive)[pos.ne(0)].mean()


def dropout_contrastive_loss(first, second, targets):
    batch = first.size(0)
    z = F.normalize(torch.cat([first, second], dim=0).float(), dim=-1)
    logits = z @ z.transpose(0, 1) / CL_TEMPERATURE
    rows = torch.arange(2 * batch, device=z.device)
    paired = (rows + batch) % (2 * batch)
    labels = torch.cat([targets, targets], dim=0)
    excluded = labels[:, None].eq(labels[None, :])
    excluded[rows, paired] = False
    excluded[rows, rows] = True
    return F.cross_entropy(logits.masked_fill(excluded, float('-inf')), paired)


def training_loss(model, seq, pos, neg, cl_weight):
    first = model.log2feats(seq)
    positive = (first * model.item_emb(pos)).sum(-1).float()
    negative = (first * model.item_emb(neg)).sum(-1).float()
    mask = pos.ne(0)
    positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
    last = positions.masked_fill(seq.eq(0), -1).amax(dim=1).clamp(min=0)
    rows = torch.arange(seq.size(0), device=seq.device)
    # Normalize within each user so long histories do not dominate the batch.
    distance = (last[:, None] - positions).clamp(min=0).float()
    weights = (0.5 + 0.5 * torch.exp(-distance / 16.0)) * mask.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    pairwise = F.softplus(negative - positive)
    pointwise = 0.5 * (F.softplus(-positive) + F.softplus(negative))
    bpr = (weights * pairwise).sum(dim=1).mean()
    recommendation = (weights * (pairwise + 0.25 * pointwise)).sum(dim=1).mean()
    if cl_weight <= 0:
        return recommendation, bpr.detach(), bpr.detach().new_zeros(())
    second = model.log2feats(seq)
    # Targets only mask false negatives; they never enter either causal history.
    cl = dropout_contrastive_loss(first[rows, last], second[rows, last], pos[rows, last])
    return recommendation + 0.2 * cl_weight * cl, bpr.detach(), cl.detach()


def make_validation_cases(user_train, user_valid, itemnum):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        forbidden = set(history) | {target, 0}
        available = itemnum - sum(1 for i in forbidden if 1 <= i <= itemnum)
        count = min(99, available)
        candidates = [target]
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in forbidden:
                forbidden.add(item)
                candidates.append(item)
        sequence = np.zeros(MAXLEN, dtype=np.int64)
        tail = history[-MAXLEN:]
        if tail:
            sequence[-len(tail):] = tail
        cases.append((sequence, candidates))
    return cases


def evaluate_val(model, cases, device):
    was_training = model.training
    model.eval()
    hr10 = hr20 = ndcg10 = 0.0
    with torch.inference_mode():
        for sequence, candidates in cases:
            seq = torch.as_tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += float(rank < 10)
            hr20 += float(rank < 20)
            ndcg10 += 1.0 / math.log2(rank + 2) if rank < 10 else 0.0
    model.train(was_training)
    n = max(1, len(cases))
    return {'HR@10': hr10 / n, 'HR@20': hr20 / n, 'NDCG@10': ndcg10 / n}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if not user_train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    architecture = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                        kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(user_train, user_valid, itemnum)
    if not cases:
        raise RuntimeError(f'No validation cases: {dataset_name}')
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        cl_weight = CL_WEIGHT * min(1.0, max(0.0, (epoch - 5) / 10.0))
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss, bpr, cl = training_loss(model, seq, pos, neg, cl_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [loss.item(), bpr.item(), cl.item()]
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, cases, device)
            hr = metrics['HR@10']
            means = totals / NUM_BATCHES
            print(f'[{dataset_name}] epoch={epoch} loss={means[0]:.4f} '
                  f'bpr={means[1]:.4f} cl={means[2]:.4f} cl_weight={cl_weight:.4f} '
                  f'val_HR@10={hr:.4f} HR@20={metrics["HR@20"]:.4f} '
                  f'NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(architecture, f)
    print(f'[{dataset_name}] best_epoch={best_epoch} best_validation_HR@10={best_hr:.4f}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    classes = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    runtime = r'''
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
    for ds in _DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = NextItNetModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _models:
        raise ValueError(f'Unknown dataset_name: {dataset_name}')
    if len(candidates) == 0:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, cand).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + classes + '\n\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    print(f'NextItNet BPR with gentle dropout-view regularization on {device}', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    write_predict_script(WORKING_DIR)
    print(f'Macro-average validation HR@10 = {sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
