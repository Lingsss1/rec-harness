from __future__ import annotations
import os
import json
import math
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN = 64
BLOCKS = 2
HEADS = 2
DROPOUT = 0.2
MAXLEN = 128
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAX_STEPS = int(os.environ.get('SASREC_BATCHES', '200'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '20'))
LR = float(os.environ.get('BERT4REC_LR', '0.001'))
MASK_RATE = float(os.environ.get('BERT4REC_MASK_RATE', '0.2'))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                u, i = map(int, fields)
                out[u].append(i)
    return dict(out)


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    itemnum = max((max(s) for s in train.values() if s), default=0)
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = int(stats[name]['itemnum'])
    else:
        itemnum = max(itemnum, max((max(s) for s in valid.values() if s), default=0))
    if itemnum < 1:
        raise ValueError('Empty item vocabulary: ' + name)
    return train, valid, itemnum


class BERT4Rec(nn.Module):
    def __init__(self, itemnum, hidden=64, blocks=2, heads=2, dropout=0.2,
                 maxlen=128, mask_id=None):
        super().__init__()
        self.itemnum = itemnum
        self.mask_id = itemnum + 1 if mask_id is None else mask_id
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 2, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.emb_norm = nn.LayerNorm(hidden, eps=1e-12)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
                dropout=dropout, activation='gelu', batch_first=True,
                norm_first=False, layer_norm_eps=1e-12
            ) for _ in range(blocks)
        ])
        self.output_dense = nn.Linear(hidden, hidden)
        self.output_norm = nn.LayerNorm(hidden, eps=1e-12)
        self.output_bias = nn.Parameter(torch.zeros(itemnum))
        self.apply(self._initialize)
        with torch.no_grad():
            for layer in self.layers:
                nn.init.normal_(layer.self_attn.in_proj_weight, std=0.02)
                if layer.self_attn.in_proj_bias is not None:
                    layer.self_attn.in_proj_bias.zero_()
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        positions = torch.arange(seq.shape[1], device=seq.device)
        padding = seq.eq(0)
        x = self.emb_dropout(self.emb_norm(self.item_emb(seq) + self.pos_emb(positions)[None]))
        x = x.masked_fill(padding.unsqueeze(-1), 0.0)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding)
            x = x.masked_fill(padding.unsqueeze(-1), 0.0)
        return x

    def output_features(self, x):
        return self.output_norm(F.gelu(self.output_dense(x)))

    def predict(self, seq, candidates):
        h = self.output_features(self.encode(seq)[:, -1])
        embeddings = self.item_emb(candidates)
        scores = (embeddings * h[:, None, :]).sum(-1)
        safe_index = (candidates - 1).clamp(0, self.itemnum - 1)
        scores = scores + self.output_bias[safe_index]
        return scores.masked_fill((candidates < 1) | (candidates > self.itemnum), -1e9)


def make_batch(histories, itemnum, rng):
    seq = np.zeros((BATCH, MAXLEN), dtype=np.int64)
    labels = np.zeros_like(seq)
    terminal_batch = rng.random() < 0.5
    for row in range(BATCH):
        history = histories[int(rng.integers(len(histories)))]
        if terminal_batch:
            target_index = int(rng.integers(1, len(history)))
            prefix = history[max(0, target_index - MAXLEN + 1):target_index]
            tokens = prefix + [itemnum + 1]
            seq[row, -len(tokens):] = tokens
            labels[row, -1] = history[target_index]
        else:
            end = len(history)
            if end > MAXLEN:
                end = int(rng.integers(MAXLEN, end + 1))
            tokens = np.asarray(history[max(0, end - MAXLEN):end], dtype=np.int64)
            start = MAXLEN - len(tokens)
            seq[row, start:] = tokens
            selected = np.flatnonzero(rng.random(len(tokens)) < MASK_RATE)
            if len(selected) == 0:
                selected = np.array([int(rng.integers(len(tokens)))])
            positions = selected + start
            labels[row, positions] = tokens[selected]
            corruption = rng.random(len(selected))
            mask_positions = positions[corruption < 0.8]
            random_positions = positions[(corruption >= 0.8) & (corruption < 0.9)]
            seq[row, mask_positions] = itemnum + 1
            seq[row, random_positions] = rng.integers(1, itemnum + 1, size=len(random_positions))
    # Remove columns containing only padding to reduce attention work. Preserve
    # absolute position IDs by retaining MAXLEN here, as inference does.
    return seq, labels


def exact_softmax_backward(model, seq, labels):
    selected = labels.ne(0)
    features = model.output_features(model.encode(seq)[selected])
    targets = labels[selected] - 1
    count = targets.numel()
    if count == 0:
        raise RuntimeError('No supervised positions')
    # Each detached chunk backpropagates immediately into the tied output
    # embedding and bias. Accumulated feature gradients then traverse the
    # encoder once. This is exact cross-entropy, without negative sampling.
    proxy = features.detach().requires_grad_(True)
    chunk_size = max(1, min(64, 4000000 // model.itemnum))
    total = 0.0
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        logits = F.linear(proxy[start:end], model.item_emb.weight[1:model.itemnum + 1], model.output_bias)
        loss = F.cross_entropy(logits, targets[start:end], reduction='sum') / count
        loss.backward()
        total += float(loss.detach())
    features.backward(proxy.grad)
    return total


def make_validation(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        target = valid[u][0]
        excluded = set(train[u]) | set(valid[u])
        available = itemnum - len({i for i in excluded if 1 <= i <= itemnum})
        if available < 99:
            raise ValueError('Fewer than 99 eligible validation negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        tokens = train[u][-(MAXLEN - 1):] + [itemnum + 1]
        sequences.append([0] * (MAXLEN - len(tokens)) + tokens)
        candidates.append([target] + list(negatives))
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, validation, device):
    seq, candidates = validation
    if len(seq) == 0:
        return 0.0, 0.0
    model.eval()
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(seq), 32):
        s = torch.as_tensor(seq[start:start + 32], device=device)
        c = torch.as_tensor(candidates[start:start + 32], device=device)
        scores = model.predict(s, c)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
        hit = ranks < 10
        hits += hit.float().sum().item()
        ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
    model.train()
    return hits / len(seq), ndcg / len(seq)


def train_one(name, device):
    seed_all(42)
    rng = np.random.default_rng(42)
    train, valid, itemnum = load_dataset(name)
    histories = [s for s in train.values() if len(s) >= 2]
    if not histories:
        raise ValueError('No trainable histories: ' + name)
    config = dict(itemnum=itemnum, hidden=HIDDEN, blocks=BLOCKS, heads=HEADS,
                  dropout=DROPOUT, maxlen=MAXLEN, mask_id=itemnum + 1)
    model = BERT4Rec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)
    validation = make_validation(train, valid, itemnum)
    steps = min(MAX_STEPS, max(10, math.ceil(len(histories) / BATCH)))
    if steps < 1:
        raise ValueError('SASREC_BATCHES must be positive')
    best = (-1.0, -1.0)
    best_epoch = 0
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    print(f'[{name}] users={len(train)} items={itemnum} steps_per_epoch={steps}', flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(steps):
            sequences, labels = make_batch(histories, itemnum, rng)
            seq_t = torch.as_tensor(sequences, device=device)
            labels_t = torch.as_tensor(labels, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss_sum += exact_softmax_backward(model, seq_t, labels_t)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={loss_sum / steps:.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                torch.save({'config': config, 'state_dict': model.state_dict(),
                            'epoch': epoch, 'validation_hr10': hr}, checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    if not os.path.isfile(checkpoint):
        torch.save({'config': config, 'state_dict': model.state_dict(), 'epoch': 0}, checkpoint)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return max(best[0], 0.0)


def write_predict_script():
    script = '''from __future__ import annotations
import os
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_CLASS = None
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset):
    global _CLASS
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _CLASS is None:
        path = os.path.join(_HERE, '..', 'train.py')
        spec = importlib.util.spec_from_file_location('bertrec_trained_architecture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.BERT4Rec
    checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = _CLASS(**config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(_DEVICE).eval()
    _MODELS[dataset] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _load(dataset)
    tokens = list(history)[-(model.maxlen - 1):] + [model.mask_id]
    seq = [0] * (model.maxlen - len(tokens)) + tokens
    with torch.no_grad():
        sequence = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
        items = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
        valid = (items >= 1) & (items <= model.itemnum)
        safe_items = items.clamp(1, model.itemnum)
        scores = model.predict(sequence, safe_items).masked_fill(~valid, -1e9)
        return scores[0].cpu().tolist()
'''
    with open(os.path.join(WORKING, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script()
    scores = [train_one(name, device) for name in DATASETS]
    print('Macro-average selected validation HR@10: %.6f' % (sum(scores) / len(scores)), flush=True)


if __name__ == '__main__':
    main()
