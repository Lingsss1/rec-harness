from __future__ import annotations

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
HIDDEN_UNITS = int(os.environ.get('HSTU_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('HSTU_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('HSTU_HEADS', '2'))
DROPOUT_RATE = float(os.environ.get('HSTU_DROPOUT', '0.3'))
LR = float(os.environ.get('HSTU_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('HSTU_WD', '1e-6'))
BATCH_SIZE = int(os.environ.get('HSTU_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('HSTU_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
MAXLEN = int(os.environ.get('HSTU_MAXLEN', '128'))
PATIENCE = int(os.environ.get('HSTU_PATIENCE', '20'))
CL_TEMPERATURE = 0.2
CL_WEIGHT = 0.005
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
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
    itemnum = 0
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as stream:
            stats = json.load(stream)
        if dataset_name in stats:
            itemnum = int(stats[dataset_name]['itemnum'])
    if not itemnum:
        itemnum = max((item for split in (train, valid) for history in split.values() for item in history), default=0)
    if itemnum <= 0:
        raise ValueError('Empty item vocabulary: ' + dataset_name)
    return train, valid, itemnum


_sampler_cache = None


def sample_batch(user_train, itemnum, batch_size, maxlen, return_metadata=False):
    global _sampler_cache
    if _sampler_cache is None or _sampler_cache[0] is not user_train:
        users = [u for u, history in user_train.items() if len(history) >= 2]
        if not users:
            raise ValueError('No eligible training users')
        seen = {u: set(user_train[u]) for u in users}
        if any(len(seen[u]) >= itemnum for u in users):
            raise ValueError('Training user has no negative items')
        _sampler_cache = (user_train, users, seen)
    _, users, seen = _sampler_cache
    seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros_like(seq)
    user_ids = np.empty(batch_size, dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)
    for row in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        user_ids[row] = user
        targets[row] = history[-1]
        nxt = history[-1]
        idx = maxlen - 1
        for item in reversed(history[:-1]):
            seq[row, idx] = item
            pos[row, idx] = nxt
            negative = random.randint(1, itemnum)
            while negative in seen[user]:
                negative = random.randint(1, itemnum)
            neg[row, idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
    if return_metadata:
        return seq, pos, neg, user_ids, targets
    return seq, pos, neg


class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(hidden_units, 2 * hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, 4 * hidden_units),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_units, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, attention_mask, padding_mask):
        normed = self.norm(x)
        attended, _ = self.attn(normed, normed, normed, attn_mask=attention_mask, need_weights=False)
        value, gate = self.gate(attended).chunk(2, dim=-1)
        x = x + self.dropout(value * F.silu(gate))
        x = x + self.ffn(self.ffn_norm(x))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2, dropout_rate=0.3, maxlen=128):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.num_heads = num_heads
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

    def log2feats(self, sequence):
        seq = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        batch, length = seq.shape
        padding = seq.eq(0)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions.masked_fill(padding, 0)
        x = self.emb_dropout(self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions))
        x = x.masked_fill(padding.unsqueeze(-1), 0.0)
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
        allowed = causal.unsqueeze(0) & (~padding).unsqueeze(1)
        # Give padded queries a safe self-edge; zero them after every block.
        diagonal = torch.eye(length, dtype=torch.bool, device=seq.device).unsqueeze(0)
        allowed = allowed | (padding.unsqueeze(-1) & diagonal)
        attention_mask = (~allowed).repeat_interleave(self.num_heads, dim=0)
        for block in self.blocks:
            x = block(x, attention_mask, padding)
        return self.last_norm(x).masked_fill(padding.unsqueeze(-1), 0.0)

    def forward(self, seq):
        return self.log2feats(seq)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return features @ embeddings.transpose(0, 1)
        if embeddings.ndim == features.ndim:
            return (features * embeddings).sum(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)


class ModelView:
    def __init__(self, model):
        self.model = model
        self.item_emb = model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def contrastive_loss(first, second, users, targets):
    batch = first.shape[0]
    representations = F.normalize(torch.cat((first, second), dim=0).float(), dim=-1)
    logits = representations @ representations.T / CL_TEMPERATURE
    user_ids = users.repeat(2)
    target_ids = targets.repeat(2)
    excluded = user_ids[:, None].eq(user_ids[None, :]) | target_ids[:, None].eq(target_ids[None, :])
    rows = torch.arange(2 * batch, device=logits.device)
    positives = (rows + batch) % (2 * batch)
    excluded[rows, positives] = False
    return F.cross_entropy(logits.masked_fill(excluded, float('-inf')), positives)


def make_validation_cases(train, valid, itemnum, maxlen, num_users=200):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    cases = []
    for user in users:
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        count = min(99, itemnum - len(excluded))
        negatives = []
        chosen = set()
        while len(negatives) < count:
            item = rng.randint(1, itemnum)
            if item not in excluded and item not in chosen:
                chosen.add(item)
                negatives.append(item)
        history = train[user][-maxlen:]
        cases.append(([0] * (maxlen - len(history)) + history, [target] + negatives))
    return cases


@torch.no_grad()
def evaluate_cases(model, cases, device):
    model.eval()
    hits = 0
    ndcg = 0.0
    for start in range(0, len(cases), 128):
        chunk = cases[start:start + 128]
        sequences = torch.tensor([case[0] for case in chunk], dtype=torch.long, device=device)
        features = model.log2feats(sequences)[:, -1, :]
        for feature, (_, candidates) in zip(features, chunk):
            scores = model.predict_score(feature, candidates)
            order = torch.argsort(scores, descending=True, stable=True)
            rank = int((order == 0).nonzero(as_tuple=False)[0, 0].item())
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    denominator = max(len(cases), 1)
    return {'HR@10': hits / denominator, 'NDCG@10': ndcg / denominator}


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen=128, num_users=200):
    cases = make_validation_cases(user_train, user_valid, itemnum, maxlen, num_users)
    return evaluate_cases(model, cases, device)


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                  num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = HSTURec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cases = make_validation_cases(train, valid, itemnum, MAXLEN)
    if not cases:
        raise ValueError('No validation cases: ' + dataset_name)
    checkpoint = os.path.join(WORKING_DIR, dataset_name + '_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    contrastive_weight = 0.001
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_rec = total_cl = 0.0
        cl_weight = contrastive_weight * min(1.0, max(0.0, (epoch - 5) / 15.0))
        for _ in range(NUM_BATCHES):
            seq, pos, neg, users, targets = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN, True)
            seq = torch.as_tensor(seq, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            optimizer.zero_grad(set_to_none=True)
            first = model.log2feats(seq)
            mask = pos.ne(0)
            features = first[mask]
            positive_scores = (features * model.item_emb(pos[mask])).sum(-1)
            negative_scores = (features * model.item_emb(neg[mask])).sum(-1)
            recommendation = 0.5 * (
                F.softplus(-positive_scores).mean()
                + F.softplus(negative_scores).mean()
            )
            regularizer = recommendation.new_zeros(())
            if cl_weight > 0:
                second = model.log2feats(seq)
                regularizer = contrastive_loss(
                    first[:, -1], second[:, -1],
                    torch.as_tensor(users, device=device),
                    torch.as_tensor(targets, device=device),
                )
            loss = recommendation + cl_weight * regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss: ' + dataset_name)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_rec += recommendation.detach().item()
            total_cl += regularizer.detach().item()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_cases(model, cases, device)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} logistic={total_rec / NUM_BATCHES:.5f} cl={total_cl / NUM_BATCHES:.5f} cl_weight={cl_weight:.5f} val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE * 5:
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint selected: ' + dataset_name)
    with open(os.path.join(WORKING_DIR, dataset_name + '_meta.pkl'), 'wb') as stream:
        pickle.dump(dict(config=config, best_epoch=best_epoch, validation_hr10=best_hr,
                         contrastive_temperature=CL_TEMPERATURE,
                         contrastive_weight=contrastive_weight,
                         recommendation_loss='balanced_logistic'), stream)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.5f} epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


PREDICT_SCRIPT = '''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_model_class = None


def _load(dataset_name):
    global _model_class
    if dataset_name in _models:
        return _models[dataset_name]
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('trained_hstu_definition', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.HSTURec
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as stream:
        meta = pickle.load(stream)
    model = _model_class(**meta['config']).to(_device)
    state = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    return model


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    name = _DATASETS[0] if dataset_name is None else dataset_name
    if name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if not candidates:
        return []
    model = _load(name)
    history = list(history[-model.maxlen:])
    sequence = [0] * (model.maxlen - len(history)) + history
    sequence = torch.tensor([sequence], dtype=torch.long, device=_device)
    features = model.log2feats(sequence)[0, -1]
    return model.predict_score(features, candidates).float().cpu().tolist()
'''


def write_predict_script(working_dir):
    with open(os.path.join(working_dir, 'predict.py'), 'w') as stream:
        stream.write(PREDICT_SCRIPT)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    print(f'HSTU BPR with weak dropout regularization: device={device}, epochs<={NUM_EPOCHS}, batches={NUM_BATCHES}', flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    print(f'Macro validation HR@10={sum(scores) / len(DATASETS):.6f}', flush=True)


if __name__ == '__main__':
    main()
