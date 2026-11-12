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
NUM_NEGATIVES = int(os.environ.get('SASREC_NEGATIVES', '128'))
TEMPERATURE = float(os.environ.get('SASREC_TEMPERATURE', '0.1'))
MAX_LOSS_POSITIONS = int(os.environ.get('SASREC_LOSS_POSITIONS', '1024'))
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_all(seed=42):
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
    itemnum = 0
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as handle:
            stats = json.load(handle)
        if name in stats:
            itemnum = int(stats[name]['itemnum'])
    if not itemnum:
        itemnum = max((item for split in (train, valid)
                       for history in split.values() for item in history), default=0)
    return train, valid, itemnum


class TrainingSampler:
    def __init__(self, train, itemnum, maxlen, seed=42):
        self.itemnum = itemnum
        self.rng = np.random.RandomState(seed)
        histories = [np.asarray(history, dtype=np.int64) for history in train.values()
                     if len(history) >= 2 and len(set(history)) < itemnum]
        if not histories:
            raise ValueError('No eligible training users')
        self.seen = [np.unique(history) for history in histories]
        self.sequences = np.zeros((len(histories), maxlen), dtype=np.int64)
        self.positives = np.zeros_like(self.sequences)
        self.allowed = {}
        for row, history in enumerate(histories):
            length = min(maxlen, len(history) - 1)
            self.sequences[row, -length:] = history[-length-1:-1]
            self.positives[row, -length:] = history[-length:]
            if len(self.seen[row]) > itemnum // 2:
                self.allowed[row] = np.setdiff1d(np.arange(1, itemnum + 1),
                                                self.seen[row], assume_unique=True)

    def sample(self, batch_size, k):
        rows = self.rng.randint(len(self.sequences), size=batch_size)
        negatives = self.rng.randint(1, self.itemnum + 1, size=(batch_size, k)).astype(np.int64)
        for row, index in enumerate(rows):
            if index in self.allowed:
                negatives[row] = self.rng.choice(self.allowed[index], size=k, replace=True)
            else:
                bad = np.isin(negatives[row], self.seen[index])
                while bad.any():
                    negatives[row, bad] = self.rng.randint(1, self.itemnum + 1, size=int(bad.sum()))
                    bad = np.isin(negatives[row], self.seen[index])
        return self.sequences[rows], self.positives[rows], negatives


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = F.relu(self.dropout1(outputs))
        outputs = self.dropout2(self.conv2(outputs)).transpose(-1, -2)
        return inputs + outputs


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
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(
                hidden_units, num_heads, dropout=dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer('causal_mask', torch.ones(maxlen, maxlen, dtype=torch.bool).triu(1), persistent=False)

    def log2feats(self, seq):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        features = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        mask = (seq != 0).unsqueeze(-1)
        features = self.emb_dropout(features) * mask
        for norm, attention, ff_norm, ff in zip(
                self.attention_layernorms, self.attention_layers,
                self.forward_layernorms, self.forward_layers):
            queries = norm(features)
            attended, _ = attention(queries, features, features,
                                   attn_mask=self.causal_mask[:length, :length], need_weights=False)
            features = ff(ff_norm(queries + attended)) * mask
        return self.last_layernorm(features)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1, :].float()
        embeddings = self.item_emb(candidates).float()
        if self.normalize:
            features = F.normalize(features, dim=-1)
            embeddings = F.normalize(embeddings, dim=-1)
        return torch.bmm(embeddings, features.unsqueeze(-1)).squeeze(-1) / self.temperature


def sampled_ce(model, features, positives, negatives):
    rows, columns = (positives != 0).nonzero(as_tuple=True)
    if MAX_LOSS_POSITIONS > 0 and rows.numel() > MAX_LOSS_POSITIONS:
        selected = torch.randperm(rows.numel(), device=rows.device)[:MAX_LOSS_POSITIONS]
        rows, columns = rows[selected], columns[selected]
    h = features[rows, columns].float()
    p = model.item_emb(positives[rows, columns]).float()
    bank = model.item_emb(negatives).float()
    if model.normalize:
        h = F.normalize(h, dim=-1)
        p = F.normalize(p, dim=-1)
        bank = F.normalize(bank, dim=-1)
    total = h.new_zeros(())
    for start in range(0, rows.numel(), 256):
        end = min(start + 256, rows.numel())
        hc = h[start:end]
        positive_logits = (hc * p[start:end]).sum(-1, keepdim=True)
        negative_logits = torch.bmm(bank[rows[start:end]], hc.unsqueeze(-1)).squeeze(-1)
        logits = torch.cat((positive_logits, negative_logits), dim=-1) / model.temperature
        labels = torch.zeros(end - start, dtype=torch.long, device=h.device)
        total = total + F.cross_entropy(logits, labels, reduction='sum')
    return total / rows.numel()


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [user for user in valid if user in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        excluded = set(train[user]) | set(valid[user])
        available = itemnum - len({item for item in excluded if 1 <= item <= itemnum})
        if available < 99:
            raise ValueError('Validation requires 99 unique available negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        history = train[user][-maxlen:]
        sequences.append([0] * (maxlen - len(history)) + history)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError('No validation examples')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate_val(model, validation, device):
    model.eval()
    sequences, candidates = validation
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start+64], device=device)
        cand = torch.as_tensor(candidates[start:start+64], device=device)
        scores = model.predict(seq, cand)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(-1)
        hit = (ranks < 10).float()
        hits += hit.sum().item()
        ndcg += (hit / torch.log2(ranks.float() + 2)).sum().item()
    return hits / len(sequences), ndcg / len(sequences)


def train_one_dataset(name, device):
    seed_all(42)
    train, valid, itemnum = load_dataset(name)
    sampler = TrainingSampler(train, itemnum, MAXLEN, seed=42)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
                  normalize=True, temperature=TEMPERATURE)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    checkpoint_path = os.path.join(WORKING_DIR, name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    print(f'[{name}] users={len(train)} items={itemnum} K={NUM_NEGATIVES}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for _ in range(NUM_BATCHES):
            sequence, positive, negative = sampler.sample(BATCH_SIZE, NUM_NEGATIVES)
            seq = torch.as_tensor(sequence, device=device)
            pos = torch.as_tensor(positive, device=device)
            neg = torch.as_tensor(negative, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = sampled_ce(model, model(seq), pos, neg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate_val(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={total_loss.item()/NUM_BATCHES:.5f} val HR@10={hr:.5f} NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({'config': config, 'state_dict': state, 'epoch': epoch,
                            'validation_hr10': hr, 'validation_ndcg10': ndcg}, checkpoint_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{name}] early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('No validation checkpoint was produced; SASREC_EPOCHS must be positive')
    with open(os.path.join(WORKING_DIR, name + '_meta.json'), 'w') as handle:
        json.dump(dict(config, best_epoch=best_epoch, validation_hr10=best_hr,
                       validation_ndcg10=best_ndcg, training_negatives=NUM_NEGATIVES,
                       max_loss_positions=MAX_LOSS_POSITIONS), handle, indent=2)
    print(f'[{name}] best validation HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    del model, optimizer, sampler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script():
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
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _get_model(dataset_name):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _models:
        checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'),
                                map_location='cpu', weights_only=True)
        config = dict(checkpoint['config'])
        config['dropout_rate'] = 0.0
        model = SASRec(**config)
        model.load_state_dict(checkpoint['state_dict'])
        _models[name] = model.to(_device).eval()
    return _models[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if len(candidates) == 0:
        return []
    model = _get_model(dataset_name)
    hist = [int(item) if 1 <= int(item) <= model.itemnum else 0
            for item in history[-model.maxlen:]]
    sequence = [0] * (model.maxlen - len(hist)) + hist
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
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script()
    print(f'Device={device}; merged normalized sampled-softmax SASRec', flush=True)
    scores = [train_one_dataset(name, device) for name in DATASETS]
    print(f'Macro-average best validation HR@10={sum(scores)/len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
