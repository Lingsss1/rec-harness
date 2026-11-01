"""Single-model-per-category BERT4Rec with dropout-view contrastive training."""
from __future__ import annotations

import importlib.util
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
MAXLEN = 128
HIDDEN_UNITS = 64
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT_RATE = 0.2
LR = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 128
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('BERT4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 20
CONTRASTIVE_WEIGHT = 0.025
TEMPERATURE = 0.2
_ELIGIBLE_CACHE = {}


def seed_everything():
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


def _load_itemnum(data_dir, dataset_name):
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    key = id(user_train)
    entry = _ELIGIBLE_CACHE.get(key)
    if entry is None or entry[0] is not user_train:
        users = [u for u, items in user_train.items() if len(items) >= 2]
        entry = (user_train, users)
        _ELIGIBLE_CACHE[key] = entry
    users = entry[1]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    batch_seq = np.zeros((batch_size, maxlen), dtype=np.int64)
    batch_pos = np.zeros_like(batch_seq)
    batch_neg = np.zeros_like(batch_seq)
    for b in range(batch_size):
        history = user_train[random.choice(users)]
        items = history[-maxlen:]
        batch_seq[b, -len(items):] = items
        eligible = list(range(maxlen - len(items), maxlen))
        selected = [i for i in eligible if random.random() < 0.2]
        if not selected:
            selected = [random.choice(eligible)]
        if random.random() < 0.5 and maxlen - 1 not in selected:
            selected.append(maxlen - 1)
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('No unobserved negative item available')
        for i in selected:
            batch_pos[b, i] = batch_seq[b, i]
            batch_seq[b, i] = itemnum + 1
            negative = random.randint(1, itemnum)
            while negative in interacted:
                negative = random.randint(1, itemnum)
            batch_neg[b, i] = negative
    return batch_seq, batch_pos, batch_neg


class BERT4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2,
                 num_heads=2, dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_token = itemnum + 1
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=num_heads,
            dim_feedforward=hidden_units * 4, dropout=dropout_rate,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def log2feats(self, sequence):
        device = self.item_emb.weight.device
        seq = torch.as_tensor(sequence, dtype=torch.long, device=device)
        if seq.ndim == 1:
            seq = seq.unsqueeze(0)
        batch, length = seq.shape
        positions = torch.arange(1, length + 1, device=device).unsqueeze(0).expand(batch, -1)
        positions = positions * seq.ne(0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        padding = seq.eq(0)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def encode_sequence(self, seq):
        return self.log2feats(seq)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if features.ndim == 1:
            return embeddings @ features
        if embeddings.ndim == features.ndim + 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        if embeddings.ndim == features.ndim and embeddings.shape == features.shape:
            return (features * embeddings).sum(dim=-1)
        return features @ embeddings.transpose(-1, -2)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return ((features * self.item_emb(pos)).sum(-1),
                (features * self.item_emb(neg)).sum(-1))

    def score_candidates(self, seq, candidates):
        # Left-padded history is shifted once to append a reserved mask.
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        features = self.log2feats(prediction_seq)
        # The appended mask is always the final nonpadding position.
        return self.predict_score(features[0, -1], candidates)


class ModelView(nn.Module):
    """Diagnostic adapter exposing the trained BERT4Rec features and embeddings."""
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
    first = model.log2feats(seq)
    second = model.log2feats(seq)
    selected = pos.ne(0)
    positives = model.item_emb(pos[selected])
    negatives = model.item_emb(neg[selected])
    recommendation_losses = []
    for features in (first, second):
        margin = (features[selected] * (positives - negatives)).sum(dim=-1)
        recommendation_losses.append(F.softplus(-margin).mean())
    recommendation_loss = 0.5 * (recommendation_losses[0] + recommendation_losses[1])
    mask = selected.unsqueeze(-1).to(first.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    first_view = F.normalize((first * mask).sum(dim=1) / denominator, dim=-1)
    second_view = F.normalize((second * mask).sum(dim=1) / denominator, dim=-1)
    logits = first_view @ second_view.T / TEMPERATURE
    identical = seq[:, None, :].eq(seq[None, :, :]).all(dim=-1)
    diagonal = torch.eye(seq.shape[0], device=seq.device, dtype=torch.bool)
    logits = logits.masked_fill(identical & ~diagonal, float('-inf'))
    labels = torch.arange(seq.shape[0], device=seq.device)
    contrastive_loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return recommendation_loss + CONTRASTIVE_WEIGHT * contrastive_loss


def _history_to_tensor(history, maxlen, device):
    array = np.zeros((1, maxlen), dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        array[0, -len(tail):] = tail
    return torch.as_tensor(array, dtype=torch.long, device=device)


def build_validation_cases(user_train, user_valid, itemnum):
    # Fixed validation-only candidates; this does not modify the host evaluator.
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    users_rng = random.Random(SEED)
    if len(users) > 200:
        users = users_rng.sample(users, 200)
    rng = random.Random(SEED)
    cases = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        available = itemnum - len([i for i in excluded if 1 <= i <= itemnum])
        count = min(99, available)
        candidates = [target]
        while len(candidates) < count + 1:
            negative = rng.randint(1, itemnum)
            if negative not in excluded:
                excluded.add(negative)
                candidates.append(negative)
        cases.append((user, candidates))
    return cases


@torch.no_grad()
def evaluate_val(model, user_train, cases, device, maxlen):
    was_training = model.training
    model.eval()
    hr10 = hr20 = ndcg10 = 0.0
    for user, candidates in cases:
        seq = _history_to_tensor(user_train[user], maxlen, device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        hr10 += float(rank < 10)
        hr20 += float(rank < 20)
        if rank < 10:
            ndcg10 += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    n = max(1, len(cases))
    return {'HR@10': hr10 / n, 'HR@20': hr20 / n, 'NDCG@10': ndcg10 / n}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    config = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                  num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                  dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = BERT4RecModel(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    checkpoint = os.path.join(working_dir, f'{dataset_name}_model.pt')
    cases = build_validation_cases(user_train, user_valid, itemnum)
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss in {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, cases, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} validation={metrics}', flush=True)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('Training completed without checkpoint selection')
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr10:.5f} epoch={best_epoch}', flush=True)
    _ELIGIBLE_CACHE.clear()
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""One BERT4Rec checkpoint per category; single-pass appended-mask prediction."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_ModelClass = None


def _ensure_loaded(dataset_name):
    global _device, _ModelClass
    if dataset_name in _models:
        return
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _ModelClass is None:
        spec = importlib.util.spec_from_file_location(
            '_bert4rec_training_definition', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ModelClass = module.BERT4RecModel
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = _ModelClass(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'),
                       map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError(f'Unknown dataset: {ds}')
    if not candidates:
        return []
    _ensure_loaded(ds)
    model = _models[ds]
    maxlen = int(_metas[ds]['maxlen'])
    array = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        array[-len(tail):] = tail
    seq = torch.tensor([array], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    if NUM_EPOCHS < 1:
        raise ValueError('Epoch count must be positive')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'BERT4Rec dropout contrastive training on {device}; weight={CONTRASTIVE_WEIGHT}', flush=True)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    write_predict_script(working_dir)
    print(f'Macro validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
