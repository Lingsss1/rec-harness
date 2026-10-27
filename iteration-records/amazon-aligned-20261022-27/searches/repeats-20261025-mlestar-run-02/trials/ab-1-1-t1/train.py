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
from torch import nn
from torch.nn import functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
HIDDEN = int(os.environ.get('SASREC_HIDDEN', '64'))
BLOCKS = int(os.environ.get('SASREC_BLOCKS', '2'))
HEADS = int(os.environ.get('SASREC_HEADS', '1'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
STEPS = int(os.environ.get('SASREC_BATCHES', '200'))
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
WD = float(os.environ.get('SASREC_WD', '0.0'))
TEMPERATURE = float(os.environ.get('DUOREC_TEMPERATURE', '0.2'))
DROP_WEIGHT = float(os.environ.get('DUOREC_DROP_WEIGHT', '0.1'))
SEM_WEIGHT = float(os.environ.get('DUOREC_SEM_WEIGHT', '0.1'))


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    if name in stats:
        itemnum = int(stats[name]['itemnum'])
    else:
        itemnum = max(i for split in (train, valid) for h in split.values() for i in h)
    return train, valid, itemnum


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = self.dropout1(self.conv1(x.transpose(1, 2)))
        y = self.dropout2(self.conv2(F.relu(y)))
        return x + y.transpose(1, 2)


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0.0, 1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
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
        self.register_buffer('causal_mask', torch.triu(
            torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1), persistent=False)

    def log2feats(self, seq):
        length = seq.shape[1]
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        mask = (seq != 0).unsqueeze(-1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_units)
        x = self.emb_dropout(x + self.pos_emb(positions)) * mask
        for norm, attention, ffn_norm, ffn in zip(
                self.attention_layernorms, self.attention_layers,
                self.forward_layernorms, self.forward_layers):
            q = norm(x)
            y, _ = attention(q, x, x,
                             attn_mask=self.causal_mask[:length, :length],
                             need_weights=False)
            x = ffn(ffn_norm(q + y)) * mask
        return self.last_layernorm(x)

    def predict(self, seq, candidates):
        z = self.log2feats(seq)[:, -1]
        return (self.item_emb(candidates) * z.unsqueeze(1)).sum(-1)


class Sampler:
    def __init__(self, train, itemnum, use_contrastive):
        self.histories = [h for h in train.values() if len(h) >= 2]
        if not self.histories:
            raise ValueError('No training histories with at least two interactions')
        self.seen = [set(h) for h in self.histories]
        self.itemnum = itemnum
        self.use_contrastive = use_contrastive
        self.by_target = defaultdict(list)
        if use_contrastive:
            for uid, h in enumerate(self.histories):
                for t in range(1, len(h)):
                    self.by_target[h[t]].append((uid, t))

    def pack(self, uid, t):
        h = self.histories[uid]
        begin = max(0, t - MAXLEN)
        n = t - begin
        seq = np.zeros(MAXLEN, dtype=np.int64)
        pos = np.zeros(MAXLEN, dtype=np.int64)
        seq[-n:] = h[begin:t]
        pos[-n:] = h[begin + 1:t + 1]
        return seq, pos

    def batch(self):
        sequences, positives, negatives = [], [], []
        targets, semantic, rows = [], [], []
        for row in range(BATCH):
            uid = random.randrange(len(self.histories))
            h = self.histories[uid]
            t = len(h) - 1
            if self.use_contrastive and random.random() >= 0.5:
                t = random.randrange(1, len(h))
            seq, pos = self.pack(uid, t)
            neg = np.zeros(MAXLEN, dtype=np.int64)
            seen = self.seen[uid]
            if len(seen) >= self.itemnum:
                raise ValueError('No available unobserved training items')
            for col in reversed(np.flatnonzero(pos)):
                item = random.randint(1, self.itemnum)
                while item in seen:
                    item = random.randint(1, self.itemnum)
                neg[col] = item
            sequences.append(seq)
            positives.append(pos)
            negatives.append(neg)
            target = h[t]
            targets.append(target)
            if self.use_contrastive:
                pool = self.by_target[target]
                if len(pool) > 1:
                    other = random.choice(pool)
                    while other == (uid, t):
                        other = random.choice(pool)
                    semantic.append(self.pack(*other)[0])
                    rows.append(row)
        sem = np.stack(semantic) if semantic else np.empty((0, MAXLEN), dtype=np.int64)
        return (np.stack(sequences), np.stack(positives), np.stack(negatives),
                np.asarray(targets, dtype=np.int64), sem, rows)


def contrastive_loss(a, b, targets):
    n = a.shape[0]
    if n < 2:
        return (a.sum() + b.sum()) * 0.0
    z = F.normalize(torch.cat([a, b], dim=0).float(), dim=-1)
    logits = z @ z.T / TEMPERATURE
    index = torch.arange(2 * n, device=z.device)
    labels = (index + n) % (2 * n)
    ids = torch.cat([targets, targets])
    excluded = ids[:, None].eq(ids[None, :])
    excluded[index, labels] = False
    return F.cross_entropy(logits.masked_fill(excluded, float('-inf')), labels)


def make_validation(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        hist = train[u][-MAXLEN:]
        sequences.append([0] * (MAXLEN - len(hist)) + hist)
        seen = set(train[u]) | set(valid[u])
        if itemnum - len(seen) < 99:
            raise ValueError('Insufficient unique validation negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in seen:
                negatives.add(item)
        candidates.append([valid[u][0]] + list(negatives))
    if not users:
        raise ValueError('No eligible validation users')
    return np.asarray(sequences, dtype=np.int64), np.asarray(candidates, dtype=np.int64)


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(sequences), 100):
        seq = torch.as_tensor(sequences[start:start + 100], device=device)
        cand = torch.as_tensor(candidates[start:start + 100], device=device)
        scores = model.predict(seq, cand)
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
        hits += (ranks < 10).sum().item()
        ndcg += torch.where(ranks < 10,
                            1.0 / torch.log2(ranks.float() + 2), 0.0).sum().item()
    return hits / len(sequences), ndcg / len(sequences)


def train_dataset(name, device):
    seed_all(42)
    train, valid, itemnum = load_data(name)
    use_contrastive = name == 'CDs_and_Vinyl'
    dropout = float(os.environ.get('SASREC_DROPOUT', '0.3' if use_contrastive else '0.5'))
    # Baseline effectively waits 20 validation checks, each five epochs apart.
    patience = int(os.environ.get('SASREC_PATIENCE', '30' if use_contrastive else '100'))
    config = dict(itemnum=itemnum, hidden_units=HIDDEN, num_blocks=BLOCKS,
                  num_heads=HEADS, dropout_rate=dropout, maxlen=MAXLEN)
    print(f'[{name}] users={len(train)} items={itemnum} contrastive={use_contrastive}', flush=True)
    sampler = Sampler(train, itemnum, use_contrastive)
    validation = make_validation(train, valid, itemnum)
    model = SASRec(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WD, betas=(0.9, 0.98))
    checkpoint = os.path.join(WORK, name + '_model.pt')
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        for _ in range(STEPS):
            seq, pos, neg, targets, sem, rows = sampler.batch()
            seq_t = torch.as_tensor(seq, device=device)
            pos_t = torch.as_tensor(pos, device=device)
            neg_t = torch.as_tensor(neg, device=device)
            optimizer.zero_grad(set_to_none=True)
            if use_contrastive:
                sem_t = torch.as_tensor(sem, device=device)
                features = model.log2feats(torch.cat([seq_t, seq_t, sem_t], dim=0))
                first = features[:BATCH]
            else:
                first = model.log2feats(seq_t)
            mask = pos_t != 0
            selected = first[mask]
            positive_logits = (selected * model.item_emb(pos_t[mask])).sum(-1)
            # Reuse only negatives belonging to the same user's row. Every
            # nonpadding entry was sampled outside that user's full history.
            lengths = mask.sum(dim=1)
            batch_size, length = neg_t.shape
            offsets = (torch.rand(batch_size, length, 3, device=device)
                       * lengths[:, None, None]).long()
            indices = length - lengths[:, None, None] + offsets
            extra_negatives = neg_t.gather(1, indices.reshape(batch_size, -1))
            extra_negatives = extra_negatives.reshape(batch_size, length, 3)
            negative_ids = torch.cat([neg_t.unsqueeze(-1), extra_negatives], dim=-1)
            negative_embeddings = model.item_emb(negative_ids[mask])
            negative_logits = (selected.unsqueeze(1) * negative_embeddings).sum(-1)
            bce = (F.softplus(-positive_logits).mean()
                   + F.softplus(negative_logits).mean())
            # Ablation: remove the auxiliary pairwise ranking loss only.
            recommendation = bce
            dropout_cl = recommendation.new_zeros(())
            semantic_cl = recommendation.new_zeros(())
            if use_contrastive:
                target_t = torch.as_tensor(targets, device=device)
                dropout_cl = contrastive_loss(first[:, -1], features[BATCH:2 * BATCH, -1], target_t)
                if rows:
                    row_t = torch.as_tensor(rows, dtype=torch.long, device=device)
                    semantic_cl = contrastive_loss(first[row_t, -1],
                                                   features[2 * BATCH:, -1], target_t[row_t])
            loss = recommendation + DROP_WEIGHT * dropout_cl + SEM_WEIGHT * semantic_cl
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: dataset={name}, epoch={epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [recommendation.detach().item(), dropout_cl.detach().item(), semantic_cl.detach().item()]
        if epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            averages = totals / max(STEPS, 1)
            print(f'[{name}] epoch={epoch} recommendation={averages[0]:.4f} dropout_CL={averages[1]:.4f} semantic_CL={averages[2]:.4f} val_HR@10={hr:.4f} val_NDCG@10={ndcg:.4f}', flush=True)
            improved = hr > best_hr
            if use_contrastive and hr == best_hr and ndcg > best_ndcg:
                improved = True
            if improved:
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= patience:
                print(f'[{name}] early stopping; best epoch={best_epoch}', flush=True)
                break
    if best_epoch == 0:
        best_hr, best_ndcg = evaluate(model, validation, device)
        torch.save(model.state_dict(), checkpoint)
    with open(os.path.join(WORK, name + '_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'[{name}] best validation HR@10={best_hr:.4f}', flush=True)
    del optimizer, model, sampler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script():
    header = 'from __future__ import annotations\nimport os\nimport math\nimport pickle\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'
    classes = inspect.getsource(PointWiseFeedForward) + '\n\n' + inspect.getsource(SASRec)
    runtime = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _load(dataset):
    global _device
    if dataset in _models:
        return _models[dataset], _metas[dataset]
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = SASRec(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta
    return model, meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model, meta = _load(dataset)
    maxlen = meta['maxlen']
    hist = [int(i) if 0 < int(i) <= meta['itemnum'] else 0 for i in history[-maxlen:]]
    seq = torch.tensor([[0] * (maxlen - len(hist)) + hist], dtype=torch.long, device=_device)
    valid = [0 < int(i) <= meta['itemnum'] for i in candidates]
    ids = [int(i) if ok else 0 for i, ok in zip(candidates, valid)]
    with torch.inference_mode():
        cand = torch.tensor([ids], dtype=torch.long, device=_device)
        scores = model.predict(seq, cand)[0].cpu().tolist()
    return [float(score) if ok else -1e30 for score, ok in zip(scores, valid)]
'''
    with open(os.path.join(WORK, 'predict.py'), 'w') as f:
        f.write(header + classes + '\n\n' + runtime)


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORK, exist_ok=True)
    write_predict_script()
    scores = [train_dataset(name, device) for name in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
