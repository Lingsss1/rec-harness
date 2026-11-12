from __future__ import annotations
import os
import json
import math
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR = os.environ['GAGC_DATA_DIR']
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
MAX_STEPS = int(os.environ.get('SASREC_BATCHES', '200'))
BATCH = int(os.environ.get('SASREC_BATCH', '128'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
LR = float(os.environ.get('SASREC_LR', os.environ.get('BERT4REC_LR', '0.001')))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '30'))
MASK_RATE = float(os.environ.get('BERT_MASK_RATE', '0.20'))
SOFTMAX_CHUNK = int(os.environ.get('BERT_SOFTMAX_CHUNK', '64'))
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')

MODEL_SOURCE = r'''
import torch
from torch import nn

class BertBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                nn.Linear(4 * hidden, hidden), nn.Dropout(dropout))
        self.ff_norm = nn.LayerNorm(hidden, eps=1e-6)

    def forward(self, x, padding):
        a = self.attn(x, x, x, key_padding_mask=padding, need_weights=False)[0]
        x = self.attn_norm(x + self.attn_dropout(a))
        x = self.ff_norm(x + self.ff(x))
        return x.masked_fill(padding.unsqueeze(-1), 0)

class BERT4Rec(nn.Module):
    def __init__(self, itemnum, hidden=64, blocks=2, heads=2,
                 dropout=0.2, maxlen=128, mask_id=None):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.mask_id = itemnum + 1 if mask_id is None else mask_id
        self.item_emb = nn.Embedding(itemnum + 2, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.input_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([BertBlock(hidden, heads, dropout) for _ in range(blocks)])
        self.output_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                         nn.LayerNorm(hidden, eps=1e-6))
        self.output_bias = nn.Parameter(torch.zeros(itemnum))
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=0.02)
            nn.init.zeros_(block.attn.in_proj_bias)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, seq):
        padding = seq.eq(0)
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.dropout(self.input_norm(self.item_emb(seq) + self.pos_emb(positions)[None]))
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, padding)
        return x

    def predict(self, seq, candidates):
        state = self.output_head(self.encode(seq)[:, -1])
        valid = (candidates >= 1) & (candidates <= self.itemnum)
        safe = candidates.clamp(1, self.itemnum)
        embeddings = self.item_emb(safe)
        scores = (embeddings * state[:, None]).sum(-1) + self.output_bias[safe - 1]
        return scores.masked_fill(~valid, -1e9)
'''
exec(MODEL_SOURCE, globals())


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


def load_dataset(name):
    train = read_split(os.path.join(DATA_DIR, name + '_train.txt'))
    valid = read_split(os.path.join(DATA_DIR, name + '_valid.txt'))
    observed = max((max(items) for split in (train, valid)
                    for items in split.values() if items), default=0)
    itemnum = 0
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats.get(name, {}).get('itemnum', 0))
    if itemnum == 0:
        itemnum = observed
    if itemnum < observed or itemnum < 1:
        raise ValueError('Invalid item vocabulary for ' + name)
    return train, valid, itemnum


def sample_batch(histories, itemnum, rng, terminal):
    seq = np.zeros((BATCH, MAXLEN), dtype=np.int64)
    labels = np.zeros_like(seq)
    mask_id = itemnum + 1
    for row in range(BATCH):
        history = histories[int(rng.integers(len(histories)))]
        if terminal:
            end = int(rng.integers(1, len(history)))
            context = history[max(0, end - MAXLEN + 1):end]
            start = MAXLEN - len(context) - 1
            seq[row, start:MAXLEN - 1] = context
            seq[row, -1] = mask_id
            labels[row, -1] = history[end]
        else:
            end = len(history)
            # Preserve short histories; vary the window for long histories.
            if end > MAXLEN and rng.random() < 0.5:
                end = int(rng.integers(MAXLEN, end + 1))
            tokens = np.asarray(history[max(0, end - MAXLEN):end], dtype=np.int64)
            selected = np.flatnonzero(rng.random(len(tokens)) < MASK_RATE)
            if len(selected) == 0:
                selected = np.array([int(rng.integers(len(tokens)))])
            start = MAXLEN - len(tokens)
            labels[row, start + selected] = tokens[selected]
            corrupted = tokens.copy()
            decisions = rng.random(len(selected))
            corrupted[selected[decisions < 0.8]] = mask_id
            random_positions = selected[(decisions >= 0.8) & (decisions < 0.9)]
            corrupted[random_positions] = rng.integers(1, itemnum + 1, size=len(random_positions))
            seq[row, start:] = corrupted
    return seq, labels


def make_validation(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train and train[u] and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        target = valid[user][0]
        context = train[user][-(MAXLEN - 1):]
        sequence = [0] * (MAXLEN - len(context) - 1) + context + [itemnum + 1]
        excluded = set(train[user]) | set(valid[user])
        available = itemnum - len({i for i in excluded if 1 <= i <= itemnum})
        if available < 99:
            raise ValueError('Fewer than 99 eligible validation negatives')
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(sequence)
        candidates.append([target] + list(negatives))
    return (np.asarray(sequences, dtype=np.int64),
            np.asarray(candidates, dtype=np.int64))


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    hits, ndcg = 0.0, 0.0
    for start in range(0, len(sequences), 64):
        seq = torch.as_tensor(sequences[start:start + 64], device=device)
        cand = torch.as_tensor(candidates[start:start + 64], device=device)
        scores = model.predict(seq, cand).float()
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
        hit = ranks < 10
        hits += hit.float().sum().item()
        ndcg += (hit.float() / torch.log2(ranks.float() + 2)).sum().item()
    n = max(1, len(sequences))
    return hits / n, ndcg / n


def exact_softmax_backward(model, seq, labels, device, use_bf16):
    selected = labels.ne(0)
    targets = labels[selected] - 1
    count = targets.numel()
    if count == 0:
        raise RuntimeError('No supervised positions')
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        features = model.output_head(model.encode(seq)[selected])
    feature_grad = torch.zeros_like(features)
    chunk = max(1, min(SOFTMAX_CHUNK, 4000000 // model.itemnum))
    total = 0.0
    # Output gradients accumulate directly into tied embeddings and bias.
    # Detached feature chunks let the encoder be backpropagated only once.
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        leaf = features[start:end].detach().requires_grad_(True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = F.linear(leaf, model.item_emb.weight[1:model.itemnum + 1], model.output_bias)
            loss = F.cross_entropy(logits.float(), targets[start:end], reduction='sum') / count
        loss.backward()
        feature_grad[start:end] = leaf.grad
        total += float(loss.detach())
        del logits, leaf, loss
    features.backward(feature_grad)
    return total


def train_one_dataset(name, device):
    seed_all(42)
    rng = np.random.default_rng(42)
    train, valid, itemnum = load_dataset(name)
    histories = [items for items in train.values() if len(items) >= 2]
    if not histories:
        raise RuntimeError('No usable training histories for ' + name)
    config = dict(itemnum=itemnum, hidden=64, blocks=2, heads=2,
                  dropout=0.2, maxlen=MAXLEN, mask_id=itemnum + 1)
    model = BERT4Rec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0,
                                  betas=(0.9, 0.999))
    validation = make_validation(train, valid, itemnum)
    if len(validation[0]) == 0:
        raise RuntimeError('No validation histories for ' + name)
    steps = min(MAX_STEPS, max(50, math.ceil(len(histories) / BATCH)))
    if steps < 1 or EPOCHS < 1:
        raise ValueError('Training steps and epochs must be positive')
    checkpoint = os.path.join(WORKING, name + '_model.pt')
    best, best_ndcg, best_epoch = -1.0, -1.0, 0
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    print(f'[{name}] users={len(train)} items={itemnum} steps_per_epoch={steps}', flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for step in range(steps):
            global_step = (epoch - 1) * steps + step
            sequence, labels = sample_batch(histories, itemnum, rng,
                                             terminal=(global_step % 2 == 1))
            seq = torch.as_tensor(sequence, dtype=torch.long, device=device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1.0, (global_step + 1) / max(1, 2 * steps))
            for group in optimizer.param_groups:
                group['lr'] = LR * warmup
            loss_sum += exact_softmax_backward(model, seq, labels, device, use_bf16)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.item_emb.weight[0].zero_()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={loss_sum / steps:.5f} '
                  f'val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if (hr, ndcg) > (best, best_ndcg):
                best, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save({'config': config, 'state_dict': model.state_dict(),
                            'epoch': epoch, 'validation_hr10': hr}, checkpoint)
            if epoch - best_epoch >= PATIENCE:
                break
    print(f'[{name}] best validation HR@10={best:.5f}, epoch={best_epoch}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best


PREDICT_SOURCE = r'''
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = None

def _load(dataset):
    global _DEVICE
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _DEVICE is None:
        torch.set_num_threads(2)
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                            map_location='cpu', weights_only=True)
    model = BERT4Rec(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(_DEVICE).eval()
    _MODELS[dataset] = model
    return model

def predict(user_id, history, candidates, *, dataset_name=None):
    candidates = list(candidates)
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _load(dataset)
    context = list(history)[-(model.maxlen - 1):]
    sequence = [0] * (model.maxlen - len(context) - 1) + context + [model.mask_id]
    seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    with torch.inference_mode():
        return model.predict(seq, cand)[0].float().cpu().tolist()
'''


def write_predict_script():
    with open(os.path.join(WORKING, 'predict.py'), 'w') as f:
        f.write(MODEL_SOURCE + '\n' + PREDICT_SOURCE)


def main():
    if MAXLEN < 2 or BATCH < 1 or SOFTMAX_CHUNK < 1:
        raise ValueError('Invalid sequence length, batch size, or softmax chunk')
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script()
    scores = [train_one_dataset(name, device) for name in DATASETS]
    print(f'Macro-average selected validation HR@10={sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
