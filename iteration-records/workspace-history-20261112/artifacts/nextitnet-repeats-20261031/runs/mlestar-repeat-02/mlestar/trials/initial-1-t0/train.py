"""Single-model, per-category NextItNet receptive-field experiment."""
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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 8
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2
DILATIONS = [1, 2, 4, 8, 16, 1, 2, 4]
LR = 1e-3
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    if dataset_name in stats:
        itemnum = int(stats[dataset_name]['itemnum'])
    else:
        itemnum = max((i for split in (train, valid) for items in split.values() for i in items), default=0)
    return train, valid, itemnum


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(seqs)
    negatives = np.zeros_like(seqs)
    for b in range(batch_size):
        history = user_train[random.choice(users)]
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('No unobserved items available for negative sampling')
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seqs[b, idx] = item
            positives[b, idx] = nxt
            neg = random.randint(1, itemnum)
            while neg in interacted:
                neg = random.randint(1, itemnum)
            negatives[b, idx] = neg
            nxt = item
            idx -= 1
            if idx < 0:
                break
    return seqs, positives, negatives


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
        return conv(nn.functional.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128, dilations=None):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.dilations = list(dilations if dilations is not None else [1, 2, 4, 8, 16, 1, 2, 4])
        if len(self.dilations) != num_blocks:
            raise ValueError('Dilation schedule must specify every block')
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, d, kernel_size, dropout_rate)
            for d in self.dilations
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        mask = seq.ne(0)
        positions = torch.arange(1, seq.size(1) + 1, device=seq.device).unsqueeze(0)
        positions = positions * mask.long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.norm(x)

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        else:
            sequence = sequence.to(device=self.item_emb.weight.device, dtype=torch.long)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if features.ndim == 1:
            return embeddings @ features
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(dim=-1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        features = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq.eq(0), -1).max(dim=1).values.clamp(min=0)
        readout = features[torch.arange(seq.size(0), device=seq.device), last]
        scores = self.predict_score(readout, candidates)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView(nn.Module):
    """Family-specific diagnostic view of the same trained network."""
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


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = pos.ne(0).float()
    losses = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (losses * mask).sum() / mask.sum().clamp(min=1)


def validation_cases(user_train, user_valid, itemnum):
    # Freeze validation sampling so checkpoint comparisons do not consume training RNG.
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    rng = random.Random(SEED)
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for user in users:
        history = user_train[user]
        target = user_valid[user][0]
        excluded = set(history) | {target, 0}
        count = min(99, itemnum - len({i for i in excluded if 1 <= i <= itemnum}))
        candidates = [target]
        while len(candidates) < count + 1:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                excluded.add(item)
                candidates.append(item)
        seq = np.zeros(MAXLEN, dtype=np.int64)
        tail = history[-MAXLEN:]
        if tail:
            seq[-len(tail):] = tail
        cases.append((seq, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, cases, device):
    model.eval()
    hits = 0
    for seq, candidates in cases:
        seq_t = torch.as_tensor(seq[None, :], dtype=torch.long, device=device)
        cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq_t, cand_t).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        hits += int(rank < 10)
    model.train()
    return hits / max(1, len(cases))


def train_one_dataset(dataset_name, device):
    train, valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    if not train or itemnum <= 0:
        raise RuntimeError('Empty dataset: ' + dataset_name)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE,
                  maxlen=MAXLEN, dilations=list(DILATIONS))
    model = NextItNetModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    cases = validation_cases(train, valid, itemnum)
    if not cases:
        raise RuntimeError('No validation users: ' + dataset_name)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr, best_epoch = -1.0, 0
    print(f'[{dataset_name}] users={len(train)} items={itemnum} dilations={DILATIONS}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            hr = evaluate_val(model, cases, device)
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.5f}', flush=True)
            if hr > best_hr:
                best_hr, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as f:
        pickle.dump(dict(config=config, selected_epoch=best_epoch, validation_hr10=best_hr), f)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact training classes; inference does not import or execute train.py.
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\n\n'
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds in _DATASETS:
        with open(os.path.join(_HERE, ds + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = NextItNetModel(**meta['config']).to(_device)
        state = torch.load(os.path.join(_HERE, ds + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _models:
        raise ValueError('Unknown dataset: ' + str(ds))
    if not candidates:
        return []
    model = _models[ds]
    maxlen = int(_metas[ds]['config']['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + runtime)


def main():
    if NUM_EPOCHS < 1:
        raise ValueError('Epoch limit must be positive')
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device) for ds in DATASETS]
    print('Macro validation HR@10 = {:.6f}'.format(sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
