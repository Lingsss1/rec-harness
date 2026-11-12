"""Independent per-category BERT4Rec with AdamW warmup and linear decay."""
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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = DEFAULT_DATASETS[:]
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('BERT4REC_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('BERT4REC_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('BERT4REC_BLOCKS', '2'))
NUM_HEADS = int(os.environ.get('BERT4REC_HEADS', '2'))
DROPOUT_RATE = float(os.environ.get('BERT4REC_DROPOUT', '0.2'))
BATCH_SIZE = int(os.environ.get('BERT4REC_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, max(1, int(os.environ.get('BERT4REC_EPOCHS', os.environ.get('SASREC_EPOCHS', '200')))))
EVAL_EVERY = int(os.environ.get('BERT4REC_EVAL_EVERY', '5'))
PATIENCE = int(os.environ.get('BERT4REC_PATIENCE', '20'))
PEAK_LR = 5e-4
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


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
    return (_read_split(os.path.join(data_dir, f'{dataset_name}_train.txt')),
            _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt')),
            _load_itemnum(data_dir, dataset_name))


def sample_batch(user_train, itemnum, batch_size, maxlen):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError('no users with at least two training interactions')
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
            raise RuntimeError('no unobserved negative item available')
        for i in selected:
            pos[i] = seq[i]
            seq[i] = itemnum + 1
            item = random.randint(1, itemnum)
            while item in interacted:
                item = random.randint(1, itemnum)
            neg[i] = item
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


def _history_to_tensor(history, maxlen, device):
    seq = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        seq[-len(tail):] = tail
    return torch.as_tensor(seq[None, :], dtype=torch.long, device=device)


class BERT4RecModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=2, num_heads=2,
                 dropout_rate=0.2, maxlen=128):
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

    def encode_sequence(self, seq):
        batch, length = seq.shape
        positions = torch.arange(1, length + 1, device=seq.device)[None, :].expand(batch, -1)
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x)
        padding = seq == 0
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        return self.norm(self.encoder(x, src_key_padding_mask=safe_padding))

    def log2feats(self, sequence):
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=self.item_emb.weight.device)
        return self.encode_sequence(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=self.item_emb.weight.device)
        embeddings = self.item_emb(items)
        if features.ndim == embeddings.ndim:
            return (features * embeddings).sum(dim=-1)
        if features.ndim == 2 and embeddings.ndim == 3:
            return (features[:, None, :] * embeddings).sum(dim=-1)
        return torch.matmul(features, embeddings.transpose(-1, -2))

    def forward(self, seq, pos, neg):
        features = self.encode_sequence(seq)
        return ((features * self.item_emb(pos)).sum(dim=-1),
                (features * self.item_emb(neg)).sum(dim=-1))

    def score_candidates(self, seq, candidates):
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        features = self.encode_sequence(prediction_seq)[0, -1]
        return torch.matmul(self.item_emb(candidates), features)


class ModelView:
    """Expose the masked-sequence representation for family-specific diagnosis."""
    def __init__(self, model):
        self.model = model
        self.item_emb = model.item_emb

    def log2feats(self, sequence):
        return self.model.log2feats(sequence)

    def predict_score(self, features, items):
        return self.model.predict_score(features, items)


def bpr_sequence_loss(model, seq, pos, neg):
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def evaluate_val(model, user_train, user_valid, itemnum, device, maxlen, num_users=200):
    # Local fixed RNG keeps checkpoint comparisons reproducible and does not
    # consume training randomness. The host independently computes its metrics.
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    user_rng = random.Random(SEED)
    if len(users) > num_users:
        users = user_rng.sample(users, num_users)
    rng = random.Random(SEED)
    totals = np.zeros(4, dtype=np.float64)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for user in users:
            target = user_valid[user][0]
            history = user_train[user]
            excluded = set(history) | {target, 0}
            candidates = [target]
            available = itemnum - len({i for i in excluded if 1 <= i <= itemnum})
            count = min(99, available)
            while len(candidates) < count + 1:
                item = rng.randint(1, itemnum)
                if item not in excluded:
                    excluded.add(item)
                    candidates.append(item)
            seq = _history_to_tensor(history, maxlen, device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq, items).cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            totals[0] += rank < 10
            totals[1] += rank < 20
            totals[2] += 1 / math.log2(rank + 2) if rank < 10 else 0
            totals[3] += 1 / math.log2(rank + 2) if rank < 20 else 0
    model.train(was_training)
    values = totals / max(1, len(users))
    return dict(zip(('HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'), values.tolist()))


def train_one_dataset(dataset_name, device, data_dir):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f'[{dataset_name}] users={len(user_train)} items={itemnum}', flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f'empty dataset or item vocabulary for {dataset_name}')
    model = BERT4RecModel(itemnum, HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS,
                          DROPOUT_RATE, MAXLEN).to(device)
    decay_parameters, no_decay_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            destination = no_decay_parameters if parameter.ndim <= 1 else decay_parameters
            destination.append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': decay_parameters, 'weight_decay': 0.01},
        {'params': no_decay_parameters, 'weight_decay': 0.0}],
        lr=PEAK_LR, betas=(0.9, 0.999), eps=1e-6)
    total_updates = NUM_EPOCHS * NUM_BATCHES
    warmup_updates = min(1000, max(1, total_updates // 10))
    update_index = 0
    os.makedirs(WORKING_DIR, exist_ok=True)
    checkpoint = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr10, best_epoch = -1.0, 0
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            update_index += 1
            if update_index <= warmup_updates:
                factor = update_index / warmup_updates
            else:
                factor = max(0.0, (total_updates - update_index) /
                             max(1, total_updates - warmup_updates))
            for group in optimizer.param_groups:
                group['lr'] = PEAK_LR * factor
            optimizer.step()
            total_loss += float(loss.detach().item())
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = metrics['HR@10']
            print(f'[{dataset_name}] epoch={epoch}/{NUM_EPOCHS} '
                  f'loss={total_loss / NUM_BATCHES:.4f} val_score={hr10:.4f} '
                  f'HR@20={metrics["HR@20"]:.4f} NDCG@10={metrics["NDCG@10"]:.4f}',
                  flush=True)
            if hr10 > best_hr10:
                best_hr10, best_epoch = hr10, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] Early stopping at epoch {epoch}', flush=True)
                break
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(dict(itemnum=itemnum, maxlen=MAXLEN, hidden_units=HIDDEN_UNITS,
                         num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
                         dropout_rate=DROPOUT_RATE), f)
    print(f'[{dataset_name}] Best validation HR@10={best_hr10:.4f}', flush=True)
    return best_hr10


def write_predict_script(working_dir):
    script = r'''"""One independently trained BERT4Rec checkpoint per category."""
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


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        'bert4rec_trial_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BERT4RecModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Model = _load_model_class()
    for dataset in DATASETS:
        with open(os.path.join(_HERE, f'{dataset}_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = Model(itemnum=meta['itemnum'], hidden_units=meta['hidden_units'],
                      num_blocks=meta['num_blocks'], num_heads=meta['num_heads'],
                      dropout_rate=0.0, maxlen=meta['maxlen']).to(_device)
        state = torch.load(os.path.join(_HERE, f'{dataset}_model.pt'),
                           map_location=_device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        _models[dataset] = model
        _metas[dataset] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    dataset = dataset_name if dataset_name is not None else DATASETS[0]
    if dataset not in _models:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not candidates:
        return []
    model, meta = _models[dataset], _metas[dataset]
    maxlen = int(meta['maxlen'])
    sequence = [0] * maxlen
    tail = history[-maxlen:]
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(script)


def main():
    seed_everything()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'BERT4Rec AdamW warmup/decay training on {device}', flush=True)
    scores = []
    for dataset in DATASETS:
        scores.append(train_one_dataset(dataset, device, DATA_DIR))
    write_predict_script(WORKING_DIR)
    print(f'Macro-average validation HR@10={sum(scores) / len(scores):.4f}', flush=True)


if __name__ == '__main__':
    main()
