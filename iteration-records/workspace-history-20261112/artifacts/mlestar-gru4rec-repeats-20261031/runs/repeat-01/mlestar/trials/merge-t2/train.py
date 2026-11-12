"""Single-checkpoint, per-category GRU4Rec with locked dropout."""
from __future__ import annotations
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
MAXLEN = max(1, int(os.environ.get('GRU4REC_MAXLEN', '128')))
HIDDEN_UNITS = int(os.environ.get('GRU4REC_HIDDEN', '128'))
NUM_LAYERS = int(os.environ.get('GRU4REC_LAYERS', '1'))
DROPOUT_RATE = float(os.environ.get('GRU4REC_DROPOUT', '0.1'))
INPUT_DROPOUT = 0.2
OUTPUT_DROPOUT = 0.1
LR = float(os.environ.get('GRU4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('GRU4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('GRU4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get('GRU4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = max(1, int(os.environ.get('GRU4REC_EVAL_EVERY', '5')))
PATIENCE = max(1, int(os.environ.get('GRU4REC_PATIENCE', '20')))


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


seed_all = seed_everything


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
    path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(path):
        with open(path) as f:
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
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('no users with at least two training interactions')
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        if len(interacted) >= itemnum:
            raise RuntimeError('no unobserved training negative available')
        for item in reversed(history[:-1]):
            seq[idx], pos[idx] = item, nxt
            if nxt != 0:
                negative = random.randint(1, itemnum)
                while negative in interacted:
                    negative = random.randint(1, itemnum)
                neg[idx] = negative
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


class LockedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = float(p)
        if not 0 <= self.p < 1:
            raise ValueError('dropout probability must be in [0, 1)')

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(keep).div_(keep)
        return x * mask


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=128, num_layers=1,
                 dropout_rate=0.1, input_dropout=0.2, output_dropout=0.1):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.input_dropout = LockedDropout(input_dropout)
        self.output_dropout = LockedDropout(output_dropout)
        self.gru = nn.GRU(hidden_units, hidden_units, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0.0)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long,
                                   device=self.item_emb.weight.device)
        emb = self.input_dropout(self.item_emb(sequence))
        out, _ = self.gru(emb)
        return self.output_dropout(out)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if features.dim() == 1:
            return torch.matmul(embeddings, features)
        if items.dim() == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if items.dim() == features.dim() - 1:
            return (features * embeddings).sum(dim=-1)
        if items.dim() == features.dim():
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        raise ValueError('unsupported feature/item shapes')

    def encode(self, seq):
        seq = torch.as_tensor(seq, dtype=torch.long, device=self.item_emb.weight.device)
        out = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        return out[torch.arange(seq.size(0), device=seq.device), last]

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        return self.predict_score(self.encode(seq)[0], candidates)


class ModelView(nn.Module):
    """Family-specific diagnostic interface for the single recurrent model."""
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
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = pos != 0
    return F.softplus(neg_scores[mask] - pos_scores[mask]).mean()


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if len(tail):
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            if itemnum - sum(1 for x in excluded if 1 <= x <= itemnum) < 99:
                raise RuntimeError('validation requires 99 unique eligible negatives')
            candidates = [target]
            while len(candidates) < 100:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            cand = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, cand).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    return {key: value / max(1, len(users)) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'empty dataset or vocabulary: {dataset_name}')
    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE,
                         INPUT_DROPOUT, OUTPUT_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr10, best_epoch = -1.0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            arrays = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq, pos, neg = [torch.as_tensor(a, dtype=torch.long, device=device) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss in {dataset_name} epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} '
                  f'loss={total_loss / NUM_BATCHES:.4f} val_score: {hr10:.4f} '
                  f'validation={json.dumps(metrics)}', flush=True)
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] Early stopping at epoch {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('no checkpoint selected')
    meta = dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE,
                input_dropout=INPUT_DROPOUT, output_dropout=OUTPUT_DROPOUT,
                best_epoch=best_epoch)
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] Best val HR@10={best_hr10:.4f} epoch={best_epoch}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""One locked-dropout GRU4Rec checkpoint per category."""
from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if len(_models) == len(DATASETS):
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location(
        'merged_gru4rec_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ds in DATASETS:
        if ds in _models:
            continue
        with open(os.path.join(_HERE, f'{ds}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = module.GRU4RecModel(
            itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
            num_layers=meta['num_layers'], dropout_rate=meta['dropout_rate'],
            input_dropout=meta['input_dropout'], output_dropout=meta['output_dropout'],
        ).to(_device)
        state = torch.load(os.path.join(_HERE, f'{ds}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds], _metas[ds] = model, meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = DATASETS[0] if dataset_name is None else dataset_name
    if ds not in _models:
        raise ValueError(f'unknown dataset_name: {ds}')
    if len(candidates) == 0:
        return []
    model, meta = _models[ds], _metas[ds]
    maxlen = int(meta['maxlen'])
    seq_arr = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, cand).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    write_predict_script(working_dir)
    print(f'Merged locked-dropout GRU4Rec training on {device}; seed={SEED}', flush=True)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'All datasets done. Macro local validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
