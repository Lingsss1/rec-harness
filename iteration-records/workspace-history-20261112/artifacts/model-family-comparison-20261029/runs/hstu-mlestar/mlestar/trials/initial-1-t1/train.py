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

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
HIDDEN_UNITS = int(os.environ.get('HSTU_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('HSTU_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('HSTU_HEADS', '2'))
DROPOUT_RATE = float(os.environ.get('HSTU_DROPOUT', '0.3'))
LR = float(os.environ.get('HSTU_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('HSTU_WD', '1e-6'))
BATCH_SIZE = int(os.environ.get('HSTU_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('HSTU_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
MAXLEN = int(os.environ.get('HSTU_MAXLEN', '128'))
PATIENCE = int(os.environ.get('HSTU_PATIENCE', '20'))
NUM_NEGATIVES = 128
TEMPERATURE = 0.05
LOSS_CHUNK = 256
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def _read_split(path):
    result = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, dataset_name + '_train.txt'))
    valid = _read_split(os.path.join(data_dir, dataset_name + '_valid.txt'))
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as stream:
            stats = json.load(stream)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((item for split in (train, valid) for hist in split.values() for item in hist), default=0)
    if not itemnum or not train:
        raise ValueError('Missing dataset: ' + dataset_name)
    return train, valid, itemnum


class SequenceSampler:
    def __init__(self, train, itemnum):
        self.train = train
        self.itemnum = itemnum
        self.users = [u for u, h in train.items() if len(h) >= 2 and len(set(h)) < itemnum]
        self.seen = {u: np.asarray(sorted(set(train[u])), dtype=np.int64) for u in self.users}
        if not self.users:
            raise ValueError('No eligible training users')

    def sample(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        histories = []
        for row in range(batch_size):
            u = random.choice(self.users)
            hist = self.train[u]
            length = min(len(hist) - 1, maxlen)
            seq[row, -length:] = hist[-length - 1:-1]
            pos[row, -length:] = hist[-length:]
            histories.append(self.seen[u])
        return seq, pos, histories


_SAMPLER_CACHE = {}


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = (id(user_train), itemnum)
    if key not in _SAMPLER_CACHE:
        _SAMPLER_CACHE[key] = SequenceSampler(user_train, itemnum)
    seq, pos, histories = _SAMPLER_CACHE[key].sample(batch_size, maxlen)
    neg = np.zeros_like(pos)
    for row, seen in enumerate(histories):
        where = np.flatnonzero(pos[row])
        values = np.random.randint(1, itemnum + 1, len(where))
        rejected = np.isin(values, seen)
        while rejected.any():
            values[rejected] = np.random.randint(1, itemnum + 1, int(rejected.sum()))
            rejected = np.isin(values, seen)
        neg[row, where] = values
    return seq, pos, neg


class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4), nn.SiLU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate))

    def forward(self, seq_emb, causal_mask, padding_mask):
        batch, length, _ = seq_emb.shape
        # Every padding query may attend to itself, avoiding all-masked rows.
        blocked = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=seq_emb.device).unsqueeze(0)
        blocked = blocked & ~(diagonal & padding_mask.unsqueeze(-1))
        blocked = blocked.unsqueeze(1).expand(batch, self.num_heads, length, length)
        blocked = blocked.reshape(batch * self.num_heads, length, length)
        normed = self.norm(seq_emb)
        attended, _ = self.attn(normed, normed, normed, attn_mask=blocked, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.3, maxlen=128, temperature=0.05,
                 normalize_embeddings=True):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([HSTUBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        if not torch.is_tensor(sequence):
            sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch, length = sequence.shape
        padding = sequence.eq(0)
        positions = torch.arange(1, length + 1, device=sequence.device).unsqueeze(0).expand(batch, -1)
        positions = positions.masked_fill(padding, 0)
        x = self.item_emb(sequence) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.emb_dropout(x).masked_fill(padding.unsqueeze(-1), 0.0)
        causal = torch.ones(length, length, dtype=torch.bool, device=sequence.device).triu(1)
        for block in self.blocks:
            x = block(x, causal, padding)
        return self.last_norm(x).masked_fill(padding.unsqueeze(-1), 0.0)

    def forward(self, sequence):
        return self.log2feats(sequence)

    def readout(self, sequence):
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        last = positions.masked_fill(sequence.eq(0), -1).amax(dim=1)
        selected = features[torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)]
        return selected.masked_fill(last.lt(0).unsqueeze(-1), 0.0)

    def predict_score(self, features, items):
        if not torch.is_tensor(items):
            items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if self.normalize_embeddings:
            features = F.normalize(features, p=2, dim=-1, eps=1e-8)
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        if items.ndim == 1:
            scores = torch.matmul(features, embeddings.transpose(-1, -2))
        else:
            scores = (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return scores / self.temperature


class ModelView:
    def __init__(self, model):
        self.model = model
        self.item_emb = model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if valid[u] and train.get(u)]
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for u in users:
        target = valid[u][0]
        excluded = set(train[u]) | set(valid[u])
        count = min(99, itemnum - len(excluded))
        negatives = []
        used = set(excluded)
        if count > 0 and len(excluded) > itemnum // 2:
            pool = [i for i in range(1, itemnum + 1) if i not in excluded]
            negatives = rng.sample(pool, count)
        else:
            while len(negatives) < count:
                item = rng.randint(1, itemnum)
                if item not in used:
                    used.add(item)
                    negatives.append(item)
        sequence = np.zeros(maxlen, dtype=np.int64)
        history = train[u][-maxlen:]
        sequence[-len(history):] = history
        examples.append((sequence, [target] + negatives))
    return examples


@torch.no_grad()
def evaluate_val(model, examples, device):
    model.eval()
    hits = 0
    ndcg = 0.0
    for start in range(0, len(examples), 64):
        part = examples[start:start + 64]
        seq = torch.as_tensor(np.stack([e[0] for e in part]), device=device)
        features = model.readout(seq)
        for index, (_, candidates) in enumerate(part):
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.predict_score(features[index], items)
            rank = int((scores[1:] >= scores[0]).sum().item())
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    denominator = max(len(examples), 1)
    return {'HR@10': hits / denominator, 'NDCG@10': ndcg / denominator}


def sampled_softmax_backward(model, seq, pos, histories):
    device = seq.device
    valid = pos.ne(0)
    rows = torch.nonzero(valid, as_tuple=False)[:, 0]
    targets = pos[valid]
    features = model.log2feats(seq)[valid]
    # Accumulate feature gradients separately, then traverse the backbone once.
    # Each chunk immediately releases its negative embedding activations.
    leaf = features.detach().requires_grad_(True)
    stride = model.itemnum + 1
    encoded = np.concatenate([history + row * stride for row, history in enumerate(histories)])
    forbidden = torch.as_tensor(encoded, dtype=torch.long, device=device)
    total_positions = targets.numel()
    total_loss = 0.0
    for start in range(0, total_positions, LOSS_CHUNK):
        end = min(start + LOSS_CHUNK, total_positions)
        row_codes = rows[start:end, None] * stride
        negatives = torch.randint(1, model.itemnum + 1, (end - start, NUM_NEGATIVES), device=device)
        while True:
            codes = row_codes + negatives
            locations = torch.searchsorted(forbidden, codes.reshape(-1)).reshape_as(codes)
            rejected = forbidden[locations.clamp_max(forbidden.numel() - 1)].eq(codes)
            if not bool(rejected.any()):
                break
            replacements = torch.randint(1, model.itemnum + 1, negatives.shape, device=device)
            negatives = torch.where(rejected, replacements, negatives)
        candidates = torch.cat([targets[start:end, None], negatives], dim=1)
        logits = model.predict_score(leaf[start:end], candidates)
        labels = torch.zeros(end - start, dtype=torch.long, device=device)
        chunk_loss = F.cross_entropy(logits, labels, reduction='sum') / total_positions
        chunk_loss.backward()
        total_loss += float(chunk_loss.detach())
    features.backward(leaf.grad)
    return total_loss


def train_one_dataset(dataset_name, device):
    seed_all()
    train, valid, itemnum = load_dataset(dataset_name, DATA_DIR)
    sampler = SequenceSampler(train, itemnum)
    examples = make_validation(train, valid, itemnum, MAXLEN)
    if not examples:
        raise ValueError('No validation users for ' + dataset_name)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
                  temperature=TEMPERATURE, normalize_embeddings=True)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint = os.path.join(WORKING, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, histories = sampler.sample(BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            total_loss += sampled_softmax_backward(model, seq, pos, histories)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
                model.pos_emb.weight[0].zero_()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, examples, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE * 5:
                break
    with open(os.path.join(WORKING, dataset_name + '_meta.pkl'), 'wb') as stream:
        pickle.dump({'model_config': config, 'best_epoch': best_epoch,
                     'validation_hr10': best_hr, 'num_negatives': NUM_NEGATIVES,
                     'seed': 42}, stream)
    print(f'[{dataset_name}] best_validation_HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(HSTUBlock) + '\n\n' + inspect.getsource(HSTURec)
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _models:
        with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
            meta = pickle.load(stream)
        model = HSTURec(**meta['model_config']).to(_device)
        state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset_name] = model
    return _models[dataset_name]


def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if len(candidates) == 0:
        return []
    model = _load(dataset_name)
    history = [int(item) for item in history if 0 < int(item) <= model.itemnum][-model.maxlen:]
    sequence = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    valid = items.gt(0) & items.le(model.itemnum)
    with torch.inference_mode():
        features = model.readout(seq)[0]
        scores = model.predict_score(features, items.clamp(0, model.itemnum))
        scores = scores.masked_fill(~valid, -1e9)
    return scores.cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as stream:
        stream.write(header + architecture + '\n\n' + runtime)


def main():
    seed_all()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script(WORKING)
    results = {}
    for dataset_name in DATASETS:
        results[dataset_name] = train_one_dataset(dataset_name, device)
    macro = sum(results.values()) / len(DATASETS)
    with open(os.path.join(WORKING, 'validation_summary.json'), 'w') as stream:
        json.dump({'per_dataset_validation_hr10': results, 'macro_validation_hr10': macro}, stream, indent=2)
    print(f'Macro validation HR@10={macro:.5f}', flush=True)


if __name__ == '__main__':
    main()
