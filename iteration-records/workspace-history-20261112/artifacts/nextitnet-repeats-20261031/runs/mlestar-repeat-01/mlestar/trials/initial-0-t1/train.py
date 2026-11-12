"""Independent per-category NextItNet models trained with calibrated gBCE."""
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

DEFAULT_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
DATASETS = list(DEFAULT_DATASETS)
DATA_DIR = os.environ['GAGC_DATA_DIR']
SEED = 42
MAXLEN = int(os.environ.get('NEXTITNET_MAXLEN', '128'))
HIDDEN_UNITS = int(os.environ.get('NEXTITNET_HIDDEN', '64'))
NUM_BLOCKS = int(os.environ.get('NEXTITNET_BLOCKS', '8'))
KERNEL_SIZE = int(os.environ.get('NEXTITNET_KERNEL', '3'))
DROPOUT_RATE = float(os.environ.get('NEXTITNET_DROPOUT', '0.2'))
LR = float(os.environ.get('NEXTITNET_LR', '0.001'))
WEIGHT_DECAY = float(os.environ.get('NEXTITNET_WD', '0.0'))
BATCH_SIZE = int(os.environ.get('NEXTITNET_BATCH', '128'))
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get('NEXTITNET_EPOCHS', os.environ.get('SASREC_EPOCHS', '200'))))
EVAL_EVERY = 5
PATIENCE = 25
NUM_NEGATIVES = 64
CALIBRATION_T = 0.75
LOSS_CHUNK = 256
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def _load_itemnum(data_dir, dataset_name):
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]['itemnum'])
    maximum = 0
    for split in ('train', 'valid'):
        with open(os.path.join(data_dir, f'{dataset_name}_{split}.txt')) as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) == 2:
                    maximum = max(maximum, int(fields[1]))
    return maximum


def load_dataset(dataset_name, data_dir):
    train = _read_split(os.path.join(data_dir, f'{dataset_name}_train.txt'))
    valid = _read_split(os.path.join(data_dir, f'{dataset_name}_valid.txt'))
    return train, valid, _load_itemnum(data_dir, dataset_name)


class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.itemnum = itemnum
        self.histories = []
        self.exclusions = []
        for history in user_train.values():
            if len(history) < 2:
                continue
            exclusion = np.unique(np.asarray(history, dtype=np.int64))
            if len(exclusion) >= itemnum:
                continue
            self.histories.append(history)
            self.exclusions.append(exclusion)
        if not self.histories:
            raise RuntimeError('No training histories with eligible negatives')

    def sample(self, batch_size, maxlen, num_negatives):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        neg = np.zeros((batch_size, maxlen, num_negatives), dtype=np.int64)
        for row in range(batch_size):
            index = random.randrange(len(self.histories))
            history = self.histories[index]
            exclusion = self.exclusions[index]
            length = min(maxlen, len(history) - 1)
            seq[row, -length:] = history[-length-1:-1]
            pos[row, -length:] = history[-length:]
            draws = np.random.randint(1, self.itemnum + 1, size=(length, num_negatives))
            invalid = np.isin(draws, exclusion)
            # Each draw is uniform conditional on not being in the training history.
            while invalid.any():
                draws[invalid] = np.random.randint(1, self.itemnum + 1, size=int(invalid.sum()))
                invalid = np.isin(draws, exclusion)
            neg[row, -length:] = draws
        return seq, pos, neg


_sampler_cache = None


def sample_batch(user_train, itemnum, batch_size, maxlen, num_negatives=NUM_NEGATIVES):
    global _sampler_cache
    if (_sampler_cache is None or _sampler_cache[0] is not user_train
            or _sampler_cache[1] != itemnum):
        _sampler_cache = (user_train, itemnum, TrainingSampler(user_train, itemnum))
    return _sampler_cache[2].sample(batch_size, maxlen, num_negatives)


class NextItNetBlock(nn.Module):
    def __init__(self, hidden_units, dilation, kernel_size, dropout_rate):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv, x):
        return conv(F.pad(x, ((self.kernel_size - 1) * self.dilation, 0)))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm2(y).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2)
        return x + self.dropout(y)


class NextItNetModel(nn.Module):
    def __init__(self, itemnum, hidden_units=64, num_blocks=8, kernel_size=3,
                 dropout_rate=0.2, maxlen=128):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList([
            NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
            for i in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def log2feats(self, seq):
        batch, length = seq.shape
        mask = (seq != 0).unsqueeze(-1)
        positions = torch.arange(1, length + 1, device=seq.device).unsqueeze(0).expand(batch, -1)
        positions = positions * (seq != 0).long()
        x = self.item_emb(seq) * math.sqrt(self.hidden_units) + self.pos_emb(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x) * mask.to(x.dtype)
        return self.norm(x) * mask.to(x.dtype)

    def encode_sequence(self, seq):
        return self.log2feats(seq)

    def predict_score(self, features, items):
        embeddings = self.item_emb(items)
        if items.ndim == features.ndim - 1:
            return (features * embeddings).sum(-1)
        return (features.unsqueeze(-2) * embeddings).sum(-1)

    def forward(self, seq, pos, neg):
        features = self.log2feats(seq)
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def score_candidates(self, seq, candidates):
        features = self.log2feats(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).max(dim=1).values.clamp(min=0)
        final = features[torch.arange(seq.size(0), device=seq.device), last]
        if candidates.ndim == 1:
            scores = final @ self.item_emb(candidates).T
        else:
            scores = self.predict_score(final, candidates)
        return scores[0] if seq.size(0) == 1 else scores


class ModelView(nn.Module):
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


def gbce_loss(p, n, itemnum, t=CALIBRATION_T):
    k = n.size(-1)
    alpha = k / max(itemnum - 1, 1)
    beta = 1.0 - t + alpha * t
    return ((beta * F.softplus(-p.float()) + F.softplus(n.float()).sum(-1)) / (k + 1)).mean()


def gbce_sequence_loss(model, seq, pos, neg):
    features = model.log2feats(seq)
    active = pos != 0
    features = features[active]
    positives = pos[active]
    negatives = neg[active]
    count = positives.numel()
    if count == 0:
        return features.sum() * 0.0

    def chunk_loss(f, p, n):
        positive_logits = (f * model.item_emb(p)).sum(-1)
        negative_logits = torch.einsum('mh,mkh->mk', f, model.item_emb(n))
        return gbce_loss(positive_logits, negative_logits, model.itemnum) * f.size(0)

    total = features.new_zeros(())
    for start in range(0, count, LOSS_CHUNK):
        stop = min(count, start + LOSS_CHUNK)
        # Recompute each chunk's gathered embeddings during backward instead of
        # retaining a full supervised-positions x negatives x hidden allocation.
        total = total + checkpoint(
            chunk_loss, features[start:stop], positives[start:stop], negatives[start:stop],
            use_reentrant=False, preserve_rng_state=False,
        )
    return total / count


def _history_to_tensor(history, maxlen, device):
    array = np.zeros(maxlen, dtype=np.int64)
    tail = history[-maxlen:]
    if tail:
        array[-len(tail):] = tail
    return torch.as_tensor(array[None], dtype=torch.long, device=device)


def build_validation(user_train, user_valid, itemnum):
    users = [u for u in user_valid if u in user_train and user_valid[u]]
    # Isolated RNG keeps the validation protocol fixed across epochs and avoids
    # consuming or resetting any training RNG state.
    rng = random.Random(SEED)
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for user in users:
        target = user_valid[user][0]
        excluded = set(user_train[user]) | {target, 0}
        candidates = [target]
        needed = min(99, itemnum - len([x for x in excluded if 1 <= x <= itemnum]))
        while len(candidates) < needed + 1:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                excluded.add(candidate)
                candidates.append(candidate)
        examples.append((user_train[user], candidates))
    return examples


@torch.no_grad()
def evaluate_val(model, examples, device, maxlen):
    was_training = model.training
    model.eval()
    hr10 = ndcg10 = 0.0
    for history, candidates in examples:
        seq = _history_to_tensor(history, maxlen, device)
        items = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.score_candidates(seq, items).cpu().tolist()
        rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
        if rank < 10:
            hr10 += 1.0
            ndcg10 += 1.0 / math.log2(rank + 2)
    model.train(was_training)
    denominator = max(1, len(examples))
    return {'HR@10': hr10 / denominator, 'NDCG@10': ndcg10 / denominator}


def train_one_dataset(dataset_name, device, data_dir):
    global _sampler_cache
    seed_everything()
    _sampler_cache = None
    train, valid, itemnum = load_dataset(dataset_name, data_dir)
    if not train or itemnum <= 0:
        raise RuntimeError(f'Empty dataset: {dataset_name}')
    examples = build_validation(train, valid, itemnum)
    if not examples:
        raise RuntimeError(f'No validation users: {dataset_name}')
    meta = dict(itemnum=itemnum, hidden_units=HIDDEN_UNITS, num_blocks=NUM_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE, maxlen=MAXLEN)
    model = NextItNetModel(**meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint_path = os.path.join(WORKING_DIR, f'{dataset_name}_model.pt')
    best_hr = -1.0
    best_ndcg = -1.0
    best_epoch = 0
    print(f'[{dataset_name}] users={len(train)} items={itemnum} K={NUM_NEGATIVES} t={CALIBRATION_T}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(train, itemnum, BATCH_SIZE, MAXLEN)
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg = torch.as_tensor(neg, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = gbce_sequence_loss(model, seq, pos, neg)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {dataset_name}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.detach().item()
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate_val(model, examples, device, MAXLEN)
            hr, ndcg = metrics['HR@10'], metrics['NDCG@10']
            print(f'[{dataset_name}] epoch={epoch} loss={total_loss / NUM_BATCHES:.6f} validation HR@10={hr:.4f} NDCG@10={ndcg:.4f}', flush=True)
            if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), checkpoint_path)
            if epoch - best_epoch >= PATIENCE:
                print(f'[{dataset_name}] early stop at {epoch}', flush=True)
                break
    if best_epoch == 0:
        raise RuntimeError('Training completed without a validated checkpoint')
    with open(os.path.join(WORKING_DIR, f'{dataset_name}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'[{dataset_name}] best validation HR@10={best_hr:.4f} epoch={best_epoch}', flush=True)
    del optimizer, model
    _sampler_cache = None
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_hr


def write_predict_script(working_dir):
    # Embed the exact trained architecture so inference does not depend on
    # importing train.py or on training-time environment settings.
    header = 'from __future__ import annotations\nimport math\nimport os\nimport pickle\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n'
    architecture = inspect.getsource(NextItNetBlock) + '\n\n' + inspect.getsource(NextItNetModel)
    runtime = r'''
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None


def _ensure_loaded(dataset_name):
    global _device
    if dataset_name in _models:
        return
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(_HERE, f'{dataset_name}_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    model = NextItNetModel(**meta).to(_device)
    state = torch.load(os.path.join(_HERE, f'{dataset_name}_model.pt'), map_location=_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    _models[dataset_name] = model
    _metas[dataset_name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    if dataset_name is None:
        dataset_name = DATASETS[0]
    _ensure_loaded(dataset_name)
    if not candidates:
        return []
    model = _models[dataset_name]
    maxlen = int(_metas[dataset_name]['maxlen'])
    sequence = [0] * maxlen
    tail = list(history[-maxlen:])
    if tail:
        sequence[-len(tail):] = tail
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    items = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.inference_mode():
        scores = model.score_candidates(seq, items).cpu().tolist()
    return [float(score) for score in scores]
'''
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(header + architecture + '\n\n' + runtime)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything()
    os.makedirs(WORKING_DIR, exist_ok=True)
    write_predict_script(WORKING_DIR)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scores = [train_one_dataset(ds, device, DATA_DIR) for ds in DATASETS]
    print(f'Macro validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
