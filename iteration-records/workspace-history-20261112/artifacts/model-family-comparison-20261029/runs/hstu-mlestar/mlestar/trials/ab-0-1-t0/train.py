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

HIDDEN_UNITS = int(os.environ.get('HSTU_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('HSTU_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('HSTU_HEADS', '2'))
DROPOUT_RATE = float(os.environ.get('HSTU_DROPOUT', '0.3'))
LR = float(os.environ.get('HSTU_LR', '1e-3'))
# Ablation: disable AdamW weight decay for every parameter.
WEIGHT_DECAY = 0.0
BATCH_SIZE = int(os.environ.get('HSTU_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(int(os.environ.get('HSTU_EPOCHS', '200')), 200)
MAXLEN = int(os.environ.get('HSTU_MAXLEN', '128'))
PATIENCE = int(os.environ.get('HSTU_PATIENCE', '20'))
EVAL_USERS = int(os.environ.get('HSTU_EVAL_USERS', '200'))
DATA_DIR = os.environ['GAGC_DATA_DIR']
DEFAULT_DATASETS = [
    'Movies_and_TV',
    'Industrial_and_Scientific',
    'Electronics',
    'CDs_and_Vinyl',
]
DATASETS = DEFAULT_DATASETS[:]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def _read_split(path: str) -> dict[int, list[int]]:
    data = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    itemnum = 0
    # Only visible training and validation splits may be read.
    for split in ('train', 'valid'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.txt')
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name: str, data_dir: str):
    user_train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    user_valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return user_train, user_valid, _load_itemnum(data_dir, dataset_name)


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [user for user, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError('No users with at least two training interactions')
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in interacted:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


class HSTUBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(
            hidden_units, num_heads, dropout=dropout_rate, batch_first=True,
        )
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(self, seq_emb, causal_mask, padding_mask):
        normed = self.norm(seq_emb)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        value, gate = self.gate(attn_out).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb


class HSTURec(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2,
                 num_heads=2, dropout_rate=0.3, maxlen=128):
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError('hidden_units must be divisible by num_heads')
        self.hidden_units = hidden_units
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            HSTUBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.last_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def forward(self, seq):
        batch_size, maxlen = seq.shape
        seq_emb = self.item_emb(seq) * math.sqrt(self.hidden_units)
        pos = torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1) + 1
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        causal_mask = torch.triu(
            torch.full((maxlen, maxlen), float('-inf'), device=seq.device),
            diagonal=1,
        )
        padding_mask = seq == 0
        for block in self.blocks:
            seq_emb = block(seq_emb, causal_mask, padding_mask)
        return self.last_norm(seq_emb)

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.forward(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        if features.ndim == 3:
            features = features[:, -1, :]
        embeddings = self.item_emb(items)
        if embeddings.ndim == 3:
            return (features.unsqueeze(1) * embeddings).sum(dim=-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))


def evaluate_val(model, user_train, user_valid, itemnum, device,
                 maxlen=128, num_users=200):
    val_users = [user for user in user_valid if user in user_train]
    if len(val_users) > num_users:
        random.seed(42)
        val_users = random.sample(val_users, num_users)
    if not val_users:
        return {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            pos_items = user_valid[user]
            if not pos_items:
                continue
            target = pos_items[0]
            hist = user_train[user][-maxlen:]
            seq_arr = np.zeros(maxlen, dtype=np.int64)
            for idx, item in enumerate(reversed(hist)):
                seq_arr[maxlen - 1 - idx] = item
            interacted = set(user_train[user]) | set(pos_items)
            neg_pool = [item for item in range(1, itemnum + 1) if item not in interacted]
            if len(neg_pool) > 100:
                neg_pool = random.sample(neg_pool, 100)
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            out = model(seq_t)
            user_emb = out[0, -1]
            scores = torch.matmul(model.item_emb(cand_t), user_emb)
            rank = int(torch.argsort(scores, descending=True).tolist().index(0))
            if rank < 10:
                hr10 += 1
                ndcg10 += 1.0 / math.log2(rank + 2)
            if rank < 20:
                hr20 += 1
                ndcg20 += 1.0 / math.log2(rank + 2)
    denom = len(val_users)
    return {'HR@10': hr10 / denom, 'HR@20': hr20 / denom,
            'NDCG@10': ndcg10 / denom, 'NDCG@20': ndcg20 / denom}


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise ValueError(f'Missing data for required category {dataset_name}')
    model = HSTURec(
        itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    working_dir = os.path.join(os.path.dirname(__file__), 'working')
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f'{dataset_name}_model.pt')
    best_hr10 = 0.0
    best_epoch = 0
    saved = False
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            out = model(seq_t)
            mask = (pos_t != 0).float()
            pos_scores = (out * model.item_emb(pos_t)).sum(dim=-1)
            neg_scores = (out * model.item_emb(neg_t)).sum(dim=-1)
            loss = F.softplus(-(pos_scores - neg_scores))
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % 5 == 0:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN, EVAL_USERS)
            hr10 = val_m['HR@10']
            print(
                f'  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}'
                f'  loss={avg_loss:.4f}  val_score: {hr10:.4f}'
                f'  HR@20={val_m["HR@20"]:.4f}  NDCG@10={val_m["NDCG@10"]:.4f}',
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                saved = True
            if epoch - best_epoch >= PATIENCE * 5:
                print(f'  [{dataset_name}] Early stopping at epoch {epoch}', flush=True)
                break
    if not saved:
        torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(working_dir, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump({
            'itemnum': itemnum, 'hidden_units': HIDDEN_UNITS,
            'num_blocks': NUM_BLOCKS, 'num_heads': NUM_HEADS,
            'dropout_rate': DROPOUT_RATE, 'maxlen': MAXLEN,
        }, f)
    print(f'  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = '''from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f'hstu_perdataset_train_{os.getpid()}', os.path.join(_HERE, '..', 'train.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HSTURec


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = _load_model_class()
    for dataset_name in DATASETS:
        with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = model_cls(
            itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
            num_blocks=meta['num_blocks'], num_heads=meta['num_heads'],
            dropout_rate=0.0, maxlen=meta['maxlen'],
        ).to(_device)
        model.load_state_dict(torch.load(
            os.path.join(_HERE, f'{dataset_name}_model.pt'),
            map_location=_device, weights_only=True,
        ))
        model.eval()
        _metas[dataset_name] = meta
        _models[dataset_name] = model


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name and dataset_name in _models:
        selected = dataset_name
    elif dataset_name:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    else:
        selected = next(iter(_models))
    model = _models[selected]
    meta = _metas[selected]
    maxlen = meta['maxlen']
    seq_arr = [0] * maxlen
    for idx, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - idx] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        out = model(seq_t)
        user_emb = out[0, -1]
        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()
    return scores
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as stream:
        stream.write(script)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)
    print(
        f'HSTU ID-only: hidden={HIDDEN_UNITS} blocks={NUM_BLOCKS} heads={NUM_HEADS} '
        f'dropout={DROPOUT_RATE} maxlen={MAXLEN} weight_decay={WEIGHT_DECAY}',
        flush=True,
    )
    working_dir = os.path.join(os.path.dirname(__file__), 'working')
    os.makedirs(working_dir, exist_ok=True)
    all_hr10 = []
    for dataset_name in DATASETS:
        print(f'Training {dataset_name}', flush=True)
        all_hr10.append(train_one_dataset(dataset_name, device, DATA_DIR))
    write_predict_script(working_dir)
    avg = sum(all_hr10) / len(all_hr10)
    print(f'All datasets done. Average val HR@10 = {avg:.4f}', flush=True)


if __name__ == '__main__':
    main()
