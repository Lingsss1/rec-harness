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
HIDDEN_UNITS = int(os.environ.get('SASREC_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('SASREC_HEADS', '1'))
DROPOUT_RATE = float(os.environ.get('SASREC_DROPOUT', '0.2'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WEIGHT_DECAY = float(os.environ.get('SASREC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('SASREC_BATCH', '128'))
NUM_BATCHES = int(os.environ.get('SASREC_BATCHES', '200'))
NUM_EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '30'))
NUM_NEGATIVES = 128
TEMPERATURE = 0.1
MAX_LOSS_POSITIONS = 1024
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        if name in stats:
            itemnum = int(stats[name]['itemnum'])
    if not itemnum:
        itemnum = max((i for histories in (train, valid) for history in histories.values() for i in history), default=0)
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
        return inputs + outputs.transpose(-1, -2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.2, maxlen=128, normalize=True, temperature=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.normalize = normalize
        self.temperature = temperature
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=1.0)
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
        batch_size, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        values = self.item_emb(seq) * math.sqrt(self.hidden_units)
        values = self.emb_dropout(values + self.pos_emb(positions))
        mask = (seq != 0).unsqueeze(-1)
        values = values * mask
        for attention_norm, attention, forward_norm, forward_layer in zip(
                self.attention_layernorms, self.attention_layers,
                self.forward_layernorms, self.forward_layers):
            queries = attention_norm(values)
            attended, _ = attention(queries, values, values,
                                   attn_mask=self.causal_mask[:length, :length],
                                   need_weights=False)
            values = queries + attended
            values = forward_layer(forward_norm(values)) * mask
        return self.last_layernorm(values)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :].float()
        embeddings = self.item_emb(candidates).float()
        if self.normalize:
            features = F.normalize(features, dim=-1)
            embeddings = F.normalize(embeddings, dim=-1)
        return (embeddings * features.unsqueeze(1)).sum(-1) / self.temperature


class TrainingSampler:
    def __init__(self, train, itemnum, maxlen):
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.histories = [np.asarray(history, dtype=np.int64) for history in train.values()
                          if len(history) >= 2 and len(set(history)) < itemnum]
        if not self.histories:
            raise ValueError('No users with at least two training interactions and available negatives')
        self.rated = [np.unique(history) for history in self.histories]
        count = len(self.histories)
        self.sequences = np.zeros((count, maxlen), dtype=np.int64)
        self.positives = np.zeros((count, maxlen), dtype=np.int64)
        for row, history in enumerate(self.histories):
            length = min(maxlen, len(history) - 1)
            self.sequences[row, -length:] = history[-length-1:-1]
            self.positives[row, -length:] = history[-length:]

    def sample(self, batch_size):
        rows = np.random.randint(len(self.histories), size=batch_size)
        negatives = np.random.randint(1, self.itemnum + 1, size=(batch_size, NUM_NEGATIVES), dtype=np.int64)
        # A bank is shared across positions of one user. Each negative is uniform
        # conditional on not belonging to that user's complete training history.
        for row, user_index in enumerate(rows):
            excluded = self.rated[user_index]
            if len(excluded) > self.itemnum // 2:
                allowed = np.setdiff1d(np.arange(1, self.itemnum + 1), excluded, assume_unique=True)
                negatives[row] = np.random.choice(allowed, size=NUM_NEGATIVES, replace=True)
            else:
                bad = np.isin(negatives[row], excluded)
                while bad.any():
                    negatives[row, bad] = np.random.randint(1, self.itemnum + 1, size=int(bad.sum()))
                    bad = np.isin(negatives[row], excluded)
        return self.sequences[rows], self.positives[rows], negatives


def sampled_ce(model, features, positives, negatives):
    rows, columns = (positives != 0).nonzero(as_tuple=True)
    if rows.numel() > MAX_LOSS_POSITIONS:
        chosen = torch.randperm(rows.numel(), device=rows.device)[:MAX_LOSS_POSITIONS]
        rows, columns = rows[chosen], columns[chosen]
    h = F.normalize(features[rows, columns].float(), dim=-1)
    p = F.normalize(model.item_emb(positives[rows, columns]).float(), dim=-1)
    bank = F.normalize(model.item_emb(negatives).float(), dim=-1)
    # Chunk the candidate expansion; the loss is averaged over valid positions.
    total = h.new_zeros(())
    for start in range(0, rows.numel(), 256):
        end = min(start + 256, rows.numel())
        hc = h[start:end]
        positive = (hc * p[start:end]).sum(-1, keepdim=True)
        negative = torch.einsum('nd,nkd->nk', hc, bank[rows[start:end]])
        logits = torch.cat((positive, negative), dim=-1) / model.temperature
        labels = torch.zeros(end - start, device=h.device, dtype=torch.long)
        total = total + F.cross_entropy(logits, labels, reduction='sum')
    return total / rows.numel()


def make_validation(train, valid, itemnum, maxlen):
    # Local RNG prevents validation from changing subsequent training samples.
    rng = random.Random(42)
    users = [user for user in valid if user in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        available = itemnum - len({i for i in excluded if 1 <= i <= itemnum})
        if available < 99:
            raise ValueError('Validation requires 99 distinct available negative items')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        history = train[user][-maxlen:]
        sequences.append([0] * (maxlen - len(history)) + history)
        candidates.append([target] + list(negatives))
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, validation, device):
    sequences, candidates = validation
    model.eval()
    hits, ndcg, count = 0.0, 0.0, len(sequences)
    for start in range(0, count, 64):
        seq = torch.as_tensor(sequences[start:start+64], device=device, dtype=torch.long)
        cand = torch.as_tensor(candidates[start:start+64], device=device, dtype=torch.long)
        scores = model.predict(seq, cand)
        # Target is candidate zero, and wins equal-score ties as in stable sorting.
        ranks = (scores[:, 1:] > scores[:, :1]).sum(-1)
        hit = ranks < 10
        hits += hit.float().sum().item()
        ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
    model.train()
    return hits / max(count, 1), ndcg / max(count, 1)


def train_one_dataset(name, device):
    seed_everything(42)
    train, valid, itemnum = load_dataset(name)
    print(f'[{name}] users={len(train)} items={itemnum}', flush=True)
    sampler = TrainingSampler(train, itemnum, MAXLEN)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
                  normalize=True, temperature=TEMPERATURE)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    checkpoint = os.path.join(WORKING_DIR, name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            sequence, positive, negative = sampler.sample(BATCH_SIZE)
            seq = torch.as_tensor(sequence, device=device, dtype=torch.long)
            pos = torch.as_tensor(positive, device=device, dtype=torch.long)
            neg = torch.as_tensor(negative, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            features = model(seq)
            loss = sampled_ce(model, features, pos, neg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} val HR@10={hr:.5f} NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({'config': config, 'state_dict': state, 'epoch': epoch,
                            'validation_hr10': hr, 'validation_ndcg10': ndcg}, checkpoint)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        hr, ndcg = evaluate_val(model, validation, device)
        torch.save({'config': config, 'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    'epoch': NUM_EPOCHS, 'validation_hr10': hr, 'validation_ndcg10': ndcg}, checkpoint)
        best_hr, best_ndcg, best_epoch = hr, ndcg, NUM_EPOCHS
    with open(os.path.join(WORKING_DIR, name + '_meta.json'), 'w') as handle:
        json.dump(dict(config, best_epoch=best_epoch, validation_hr10=best_hr,
                       validation_ndcg10=best_ndcg, training_negatives=NUM_NEGATIVES), handle, indent=2)
    print(f'[{name}] best validation HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    del model, optimizer, sampler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script():
    # Embed the actual architecture, so prediction is independent of importing
    # the training entry point or having the training data environment present.
    header = '''from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    architecture = inspect.getsource(PointWiseFeedForward) + '\n\n' + inspect.getsource(SASRec)
    loader = '''

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_model(dataset_name):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _models:
        path = os.path.join(_HERE, name + '_model.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = dict(checkpoint['config'])
        config['dropout_rate'] = 0.0
        model = SASRec(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(_device).eval()
        _models[name] = model
    return _models[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    model = _get_model(dataset_name)
    history = [int(item) if 0 <= int(item) <= model.itemnum else 0
               for item in history[-model.maxlen:]]
    sequence = [0] * (model.maxlen - len(history)) + history
    ids = [int(item) for item in candidates]
    legal = [1 <= item <= model.itemnum for item in ids]
    safe_ids = [item if ok else 0 for item, ok in zip(ids, legal)]
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor([safe_ids], dtype=torch.long, device=_device)
    scores = model.predict(seq, cand)[0].cpu().tolist()
    return [float(score) if ok else -1e9 for score, ok in zip(scores, legal)]
'''
    with open(os.path.join(WORKING_DIR, 'predict.py'), 'w') as handle:
        handle.write(header + architecture + loader)


def main():
    seed_everything(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f'Device={device}; normalized sampled CE; K={NUM_NEGATIVES}; temperature={TEMPERATURE}', flush=True)
    write_predict_script()
    scores = []
    for dataset in DATASETS:
        scores.append(train_one_dataset(dataset, device))
    print(f'Macro-average best validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
