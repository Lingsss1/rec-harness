"""Independent per-category BERT4Rec with full-vocabulary masked-item CE."""
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
from torch.utils.checkpoint import checkpoint

DEFAULT_DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('BERT4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('BERT4REC_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('BERT4REC_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('BERT4REC_HEADS', '2'))
DROPOUT_RATE = float(os.environ.get('BERT4REC_DROPOUT', '0.2'))
LR = float(os.environ.get('BERT4REC_LR', '1e-3'))
WEIGHT_DECAY = float(os.environ.get('BERT4REC_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('BERT4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('BERT4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = max(1, int(os.environ.get('BERT4REC_EVAL_EVERY', '5')))
PATIENCE = int(os.environ.get('BERT4REC_PATIENCE', '20'))
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


seed_everything()


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
    # Every supervised item is replaced by MASK before bidirectional encoding.
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('No users with at least two training interactions')
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        history = user_train[random.choice(users)]
        items = history[-maxlen:]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        seq[-len(items):] = items
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
            pos[i] = seq[i]
            seq[i] = itemnum + 1
            # Retain the diagnostic sampler interface; CE ignores neg.
            item = random.randint(1, itemnum)
            while item in interacted:
                item = random.randint(1, itemnum)
            neg[i] = item
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


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
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq):
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * seq.ne(0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        padding = seq.eq(0)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    def log2feats(self, sequence):
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        return (features * self.item_emb(items)).sum(dim=-1)

    def forward(self, seq, pos, neg):
        out = self.log2feats(seq)
        return self.predict_score(out, pos), self.predict_score(out, neg)

    def score_candidates(self, seq, candidates):
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        # Left padding: the appended mask is always the final position.
        features = self.log2feats(prediction_seq)[0, -1]
        return self.predict_score(features, candidates)


def _ce_chunk(features, weights, labels):
    return F.cross_entropy(F.linear(features, weights), labels, reduction='sum')


def bpr_sequence_loss(model, seq, pos, neg):
    # Preserve the call interface while replacing BPR with exact vocabulary CE.
    selected = pos.ne(0)
    features = model.log2feats(seq)[selected]
    labels = pos[selected] - 1
    weights = model.item_emb.weight[1:model.itemnum + 1]
    if labels.numel() == 0:
        return features.sum() * 0.0
    loss_sum = features.sum() * 0.0
    for start in range(0, features.shape[0], 128):
        # Recompute each logits chunk during backward so chunks do not retain
        # a full batch-by-vocabulary activation graph. The objective is exact.
        loss_sum = loss_sum + checkpoint(
            _ce_chunk, features[start:start + 128], weights,
            labels[start:start + 128], use_reentrant=False,
        )
    return loss_sum / labels.numel()


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # Local, repeatable validation only; the host independently evaluates.
    rng = random.Random(SEED)
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(users) > num_users:
        users = rng.sample(users, num_users)
    totals = {'HR@10': 0.0, 'HR@20': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0}
    if not users:
        raise RuntimeError('No validation users')
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            wanted = min(99, itemnum - len({i for i in excluded if 1 <= i <= itemnum}))
            if wanted < 99:
                pool = [i for i in range(1, itemnum + 1) if i not in excluded]
                candidates.extend(rng.sample(pool, wanted))
            else:
                while len(candidates) < 100:
                    item = rng.randint(1, itemnum)
                    if item not in excluded:
                        excluded.add(item)
                        candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            for k in (10, 20):
                if rank < k:
                    totals[f'HR@{k}'] += 1.0
                    totals[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
    model.train()
    return {key: value / len(users) for key, value in totals.items()}


def train_one_dataset(dataset_name, device, data_dir):
    seed_everything()
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS,
                num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = BERT4RecModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    os.makedirs(WORKING_DIR, exist_ok=True)
    ckpt_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite CE loss for {dataset_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.5f} '
                  f'val_HR@10={hr10:.4f} val_NDCG@10={metrics["NDCG@10"]:.4f}', flush=True)
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE:
                break
    if best_epoch == 0:
        raise RuntimeError('No checkpoint selected; BERT4REC_EPOCHS must be positive')
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] selected_epoch={best_epoch} val_HR@10={best_hr10:.4f}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    # Embed the exact trained class, avoiding architecture drift at inference.
    header = '''from __future__ import annotations
import os
import pickle
import torch
import torch.nn as nn

'''
    footer = '''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["Movies_and_TV", "Industrial_and_Scientific", "Electronics", "CDs_and_Vinyl"]
_models = {}
_metas = {}
_device = None


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for ds in DATASETS:
        with open(os.path.join(_HERE, ds + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = BERT4RecModel(**meta).to(_device)
        state = torch.load(os.path.join(_HERE, ds + '_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ds = dataset_name if dataset_name is not None else DATASETS[0]
    if ds not in _models:
        raise ValueError(f'Unknown dataset: {ds}')
    if not candidates:
        return []
    model = _models[ds]
    maxlen = _metas[ds]['maxlen']
    seq_arr = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        seq_arr[-len(tail):] = tail
    seq = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + inspect.getsource(BERT4RecModel) + '\n' + footer)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'BERT4Rec full-vocabulary masked CE on {device}', flush=True)
    write_predict_script(WORKING_DIR)
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
