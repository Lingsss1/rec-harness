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
NUM_EPOCHS = int(os.environ.get('SASREC_EPOCHS', '200'))
NUM_BATCHES = int(os.environ.get('SASREC_BATCHES', '50'))
BATCH_SIZE = int(os.environ.get('SASREC_BATCH', '128'))
MAXLEN = int(os.environ.get('SASREC_MAXLEN', '128'))
LR = float(os.environ.get('SASREC_LR', '0.001'))
PATIENCE = int(os.environ.get('SASREC_PATIENCE', '30'))
MASK_RATE = float(os.environ.get('BERT_MASK_RATE', '0.20'))
SOFTMAX_CHUNK = int(os.environ.get('BERT_SOFTMAX_CHUNK', '64'))

MODEL_SOURCE = r'''
import math
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
        # MultiheadAttention's packed projection is not an nn.Linear module.
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
        embeddings = self.item_emb(candidates)
        return (embeddings * state[:, None]).sum(-1) + self.output_bias[candidates - 1]
'''
exec(MODEL_SOURCE, globals())


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
    stats_path = os.path.join(DATA_DIR, 'dataset_stats.json')
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats.get(name, {}).get('itemnum', 0))
    if not itemnum:
        itemnum = max((max(items) for split in (train, valid)
                       for items in split.values() if items), default=0)
    return train, valid, itemnum


def sample_batch(histories, itemnum, rng, terminal):
    seq = np.zeros((BATCH_SIZE, MAXLEN), dtype=np.int64)
    labels = np.zeros_like(seq)
    mask_id = itemnum + 1
    for row in range(BATCH_SIZE):
        history = histories[int(rng.integers(len(histories)))]
        if terminal:
            # The target belongs to the training split and never appears as an
            # uncorrupted future token in this example.
            end = int(rng.integers(1, len(history)))
            context = history[max(0, end - MAXLEN + 1):end]
            start = MAXLEN - len(context) - 1
            seq[row, start:MAXLEN - 1] = context
            seq[row, -1] = mask_id
            labels[row, -1] = history[end]
        else:
            # Mix full trailing windows and randomly truncated training windows.
            end = len(history)
            if len(history) > 2 and rng.random() < 0.5:
                end = int(rng.integers(2, len(history) + 1))
            tokens = np.asarray(history[max(0, end - MAXLEN):end], dtype=np.int64)
            chosen = rng.random(len(tokens)) < MASK_RATE
            if not chosen.any():
                chosen[int(rng.integers(len(tokens)))] = True
            offsets = np.flatnonzero(chosen)
            start = MAXLEN - len(tokens)
            labels[row, start + offsets] = tokens[offsets]
            corrupted = tokens.copy()
            decisions = rng.random(len(offsets))
            corrupted[offsets[decisions < 0.8]] = mask_id
            random_offsets = offsets[(decisions >= 0.8) & (decisions < 0.9)]
            corrupted[random_offsets] = rng.integers(1, itemnum + 1, size=len(random_offsets))
            seq[row, start:] = corrupted
    return seq, labels


def make_validation(train, valid, itemnum):
    # Local RNG keeps candidates fixed and does not reset training randomness.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        target = valid[user][0]
        context = train[user][-(MAXLEN - 1):]
        seq = [0] * (MAXLEN - len(context) - 1) + context + [itemnum + 1]
        excluded = set(train[user]) | set(valid[user])
        negatives = set()
        needed = min(99, itemnum - len(excluded))
        while len(negatives) < needed:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([target] + list(negatives))
    return sequences, candidates


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    seqs, candidates = validation
    hits, ndcg = 0.0, 0.0
    # Normally every row has exactly 100 candidates. Handle small catalogs too.
    for start in range(0, len(seqs), 64):
        batch_candidates = candidates[start:start + 64]
        if len({len(c) for c in batch_candidates}) == 1:
            x = torch.tensor(seqs[start:start + 64], dtype=torch.long, device=device)
            c = torch.tensor(batch_candidates, dtype=torch.long, device=device)
            scores = model.predict(x, c).float()
            ranks = (scores[:, 1:] > scores[:, :1]).sum(1).cpu().tolist()
        else:
            ranks = []
            for seq, cand in zip(seqs[start:start + 64], batch_candidates):
                s = model.predict(torch.tensor([seq], device=device),
                                  torch.tensor([cand], device=device))[0]
                ranks.append(int((s[1:] > s[0]).sum()))
        for rank in ranks:
            if rank < 10:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)
    n = max(1, len(seqs))
    return hits / n, ndcg / n


def train_one_dataset(name, device, working):
    train, valid, itemnum = load_dataset(name)
    histories = [items for items in train.values() if len(items) >= 2]
    if not histories:
        raise RuntimeError('No usable training histories for ' + name)
    rng = np.random.default_rng(42)
    config = dict(itemnum=itemnum, hidden=64, blocks=2, heads=2,
                  dropout=0.2, maxlen=MAXLEN, mask_id=itemnum + 1)
    model = BERT4Rec(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                                  betas=(0.9, 0.999))
    validation = make_validation(train, valid, itemnum)
    checkpoint = os.path.join(working, name + '_model.pt')
    best, best_ndcg, best_epoch = -1.0, -1.0, 0
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    print(f'[{name}] users={len(train)} items={itemnum} batches/epoch={NUM_BATCHES}', flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step in range(NUM_BATCHES):
            global_step = (epoch - 1) * NUM_BATCHES + step
            # Exactly alternate random-mask and terminal-mask minibatches.
            seq, labels = sample_batch(histories, itemnum, rng, terminal=(global_step % 2 == 1))
            seq = torch.as_tensor(seq, dtype=torch.long, device=device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            selected = labels.ne(0)
            targets = labels[selected] - 1
            optimizer.zero_grad(set_to_none=True)
            # A short warmup stabilizes the small-initialization Transformer.
            warmup = min(1.0, (global_step + 1) / max(1, 2 * NUM_BATCHES))
            for group in optimizer.param_groups:
                group['lr'] = LR * warmup
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                features = model.output_head(model.encode(seq)[selected])
            # Exact softmax, chunked over selected positions. Accumulate the
            # feature gradient separately so the Transformer is backpropagated
            # only once, without retaining all vocabulary-sized logits.
            feature_grad = torch.zeros_like(features)
            count = targets.numel()
            batch_loss = 0.0
            for start in range(0, count, SOFTMAX_CHUNK):
                end = min(start + SOFTMAX_CHUNK, count)
                leaf = features[start:end].detach().requires_grad_(True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = F.linear(leaf, model.item_emb.weight[1:itemnum + 1], model.output_bias)
                    loss = F.cross_entropy(logits.float(), targets[start:end], reduction='sum') / count
                loss.backward()
                feature_grad[start:end] = leaf.grad
                batch_loss += float(loss.detach())
                del logits, leaf, loss
            features.backward(feature_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += batch_loss
        if epoch == 1 or epoch % 5 == 0 or epoch == NUM_EPOCHS:
            hr, ndcg = evaluate(model, validation, device)
            print(f'[{name}] epoch={epoch} loss={total_loss / max(1, NUM_BATCHES):.5f} '
                  f'val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}', flush=True)
            if hr > best or (hr == best and ndcg > best_ndcg):
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
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _load(dataset)
    context = list(history)[-(model.maxlen - 1):]
    sequence = [0] * (model.maxlen - len(context) - 1) + context + [model.mask_id]
    seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    with torch.inference_mode():
        return model.predict(seq, cand)[0].float().cpu().tolist()
'''


def write_predict_script(working_dir):
    # Standalone model definition and the complete trained output head are shared
    # with training; inference never imports or executes the training program.
    with open(os.path.join(working_dir, 'predict.py'), 'w') as f:
        f.write(MODEL_SOURCE + '\n' + PREDICT_SOURCE)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(int(os.environ.get('BERT_CPU_THREADS', '8')))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    working = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'working')
    os.makedirs(working, exist_ok=True)
    write_predict_script(working)
    scores = []
    for dataset in DATASETS:
        scores.append(train_one_dataset(dataset, device, working))
    print(f'Macro-average best validation HR@10={sum(scores) / len(scores):.5f}', flush=True)


if __name__ == '__main__':
    main()
