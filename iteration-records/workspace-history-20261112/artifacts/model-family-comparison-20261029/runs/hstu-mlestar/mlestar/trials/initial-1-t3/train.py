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
CL_WEIGHT = 0.05
CL_TEMPERATURE = 0.2


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_split(path):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                data[int(fields[0])].append(int(fields[1]))
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
    observed = max((max(items, default=0) for split in (train, valid) for items in split.values()), default=0)
    itemnum = max(itemnum, observed)
    if itemnum < 2:
        raise ValueError('Insufficient items for ' + dataset_name)
    return train, valid, itemnum


class TrainingData(dict):
    def __init__(self, data):
        super().__init__(data)
        self.users = [u for u, history in data.items() if len(history) >= 2]
        self.seen = {u: set(data[u]) for u in self.users}


def sample_batch(user_train, itemnum, batch_size, maxlen, return_metadata=False):
    users = getattr(user_train, 'users', None)
    if users is None:
        users = [u for u, h in user_train.items() if len(h) >= 2]
    if not users:
        raise ValueError('No users with at least two training interactions')
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    user_ids = np.empty(batch_size, dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        interacted = user_train.seen[user] if isinstance(user_train, TrainingData) else set(history)
        if len(interacted) >= itemnum:
            raise ValueError('Cannot sample an unseen negative for user ' + str(user))
        user_ids[row] = user
        targets[row] = history[-1]
        inputs = history[:-1][-maxlen:]
        labels = history[1:][-maxlen:]
        offset = maxlen - len(inputs)
        seq[row, offset:] = inputs
        pos[row, offset:] = labels
        for col in range(offset, maxlen):
            candidate = random.randint(1, itemnum)
            while candidate in interacted:
                candidate = random.randint(1, itemnum)
            neg[row, col] = candidate
    if return_metadata:
        return seq, pos, neg, user_ids, targets
    return seq, pos, neg


class HSTUBlock(nn.Module):
    """Causal attention, SiLU-gated residual update, and residual FFN."""
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
        # An otherwise fully masked padded query may attend only to itself.
        # Real queries can attend only to nonpadding causal keys.
        blocked = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=seq_emb.device).unsqueeze(0)
        blocked = blocked & ~(padding_mask.unsqueeze(2) & diagonal)
        blocked = blocked.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        blocked = blocked.reshape(batch * self.num_heads, length, length)
        normed = self.norm(seq_emb)
        attended, _ = self.attn(normed, normed, normed, attn_mask=blocked, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.3, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.hidden_units = hidden_units
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([HSTUBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.last_norm = nn.LayerNorm(hidden_units)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()

    def forward(self, seq):
        batch, length = seq.shape
        padding_mask = seq.eq(0)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions.masked_fill(padding_mask, 0)
        features = self.item_emb(seq) * math.sqrt(self.hidden_units)
        features = self.emb_dropout(features + self.pos_emb(positions))
        features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=seq.device).triu(1)
        for block in self.blocks:
            features = block(features, causal_mask, padding_mask)
        return self.last_norm(features).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.forward(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        if features.ndim == 3:
            features = features[:, -1, :]
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        return (features.unsqueeze(-2) * embeddings).sum(dim=-1)


class ModelView(nn.Module):
    """Diagnostic adapter exposing the trained HSTU encoder and scoring head."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def item_emb(self):
        return self.model.item_emb

    def forward(self, sequence):
        return self.model.log2feats(sequence)

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def dropout_contrastive_loss(view_a, view_b, users, targets):
    # Each row has precisely one positive: the same sequence in the other view.
    batch = view_a.shape[0]
    features = F.normalize(torch.cat((view_a, view_b), dim=0).float(), dim=-1)
    logits = features @ features.T / CL_TEMPERATURE
    ids = torch.cat((users, users))
    labels = torch.cat((targets, targets))
    excluded = ids[:, None].eq(ids[None, :]) | labels[:, None].eq(labels[None, :])
    rows = torch.arange(2 * batch, device=features.device)
    positives = (rows + batch) % (2 * batch)
    excluded[rows, positives] = False
    logits = logits.masked_fill(excluded, float('-inf'))
    return F.cross_entropy(logits, positives)


def build_validation(train, valid, itemnum, maxlen):
    # Fixed, training-independent validation candidates: 99 unique negatives.
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        target = valid[user][0]
        forbidden = set(train[user]) | set(valid[user])
        available = itemnum - len(forbidden)
        count = min(99, available)
        if count < 0:
            raise ValueError('Invalid validation item IDs')
        if available < 198:
            pool = [i for i in range(1, itemnum + 1) if i not in forbidden]
            negatives = rng.sample(pool, count)
        else:
            negatives, chosen = [], set()
            while len(negatives) < count:
                item = rng.randint(1, itemnum)
                if item not in forbidden and item not in chosen:
                    chosen.add(item)
                    negatives.append(item)
        history = train[user][-maxlen:]
        sequences.append([0] * (maxlen - len(history)) + history)
        candidates.append([target] + negatives)
    return sequences, candidates


@torch.no_grad()
def evaluate_val(model, validation, device):
    model.eval()
    sequences, candidate_lists = validation
    if not sequences:
        raise ValueError('No validation users')
    hits = 0.0
    ndcg = 0.0
    # Individual candidate sets also support tiny catalogs without padding scores.
    for offset in range(0, len(sequences), 64):
        seq = torch.tensor(sequences[offset:offset + 64], device=device, dtype=torch.long)
        features = model(seq)[:, -1, :]
        for row, candidates in enumerate(candidate_lists[offset:offset + 64]):
            items = torch.tensor(candidates, device=device, dtype=torch.long)
            scores = model.predict_score(features[row:row + 1], items)[0]
            rank = int((scores[1:] >= scores[0]).sum().item())
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    return {'HR@10': hits / len(sequences), 'NDCG@10': ndcg / len(sequences)}


def train_one_dataset(dataset_name, device, data_dir, working_dir):
    seed_everything(42)
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    train = TrainingData(train)
    validation = build_validation(train, valid, itemnum, MAXLEN)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint_path = os.path.join(working_dir, dataset_name + '_model.pt')
    best_hr = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        recommendation_total = contrastive_total = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg, users, targets = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN, return_metadata=True)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            users = torch.as_tensor(users, dtype=torch.long, device=device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            first = model(seq)
            second = model(seq)
            mask = pos.ne(0)
            positive_scores = (first * model.item_emb(pos)).sum(dim=-1)
            negative_scores = (first * model.item_emb(neg)).sum(dim=-1)
            recommendation_loss = F.softplus(-(positive_scores[mask] - negative_scores[mask])).mean()
            contrastive_loss = dropout_contrastive_loss(first[:, -1, :], second[:, -1, :], users, targets)
            loss = recommendation_loss + CL_WEIGHT * contrastive_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss in ' + dataset_name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            recommendation_total += recommendation_loss.item()
            contrastive_total += contrastive_loss.item()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, validation, device)
            hr = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} rec={recommendation_total / NUM_BATCHES:.5f} '
                  f'cl={contrastive_total / NUM_BATCHES:.5f} validation_HR@10={hr:.5f} '
                  f'NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            if hr > best_hr:
                best_hr = hr
                best_epoch = epoch
                torch.save({'config': config, 'state_dict': model.state_dict(),
                            'epoch': epoch, 'validation_hr10': hr,
                            'contrastive_weight': CL_WEIGHT, 'contrastive_temperature': CL_TEMPERATURE}, checkpoint_path)
            if epoch - best_epoch >= PATIENCE * 5:
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint saved')
    print(f'[{dataset_name}] best validation HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture so prediction is independent of train.py imports.
    header = 'from __future__ import annotations\nimport os\nimport math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(HSTUBlock) + '\n\n' + inspect.getsource(HSTURec)
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load(dataset_name):
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if dataset_name not in _models:
        path = os.path.join(_HERE, dataset_name + '_model.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = HSTURec(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.to(_device).eval()
        _models[dataset_name] = model
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = _DATASETS[0]
    model = _load(dataset_name)
    if not candidates:
        return []
    history = [int(item) for item in history if 0 < int(item) <= model.itemnum][-model.maxlen:]
    sequence = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    if bool(((items < 1) | (items > model.itemnum)).any()):
        raise ValueError('Candidate outside category item vocabulary')
    features = model.log2feats(seq)[:, -1, :]
    return model.predict_score(features, items)[0].float().cpu().tolist()
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n' + runtime)


def main():
    if NUM_EPOCHS < 1:
        raise ValueError('At least one training epoch is required')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    write_predict_script(working_dir)
    print(f'HSTU dropout-view contrastive training on {device}; epochs <= {NUM_EPOCHS}; '
          f'batches/epoch={NUM_BATCHES}; CL weight={CL_WEIGHT}; temperature={CL_TEMPERATURE}', flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR, working_dir))
    print(f'Macro internal validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
