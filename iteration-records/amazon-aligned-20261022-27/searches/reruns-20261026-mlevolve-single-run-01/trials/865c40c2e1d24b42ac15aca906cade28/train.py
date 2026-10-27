from __future__ import annotations
import os
import math
import json
import random
import pickle
import importlib.util
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "working")
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "100"))
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "96"))
BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
HEADS = int(os.environ.get("SASREC_HEADS", "2"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.25"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
NEGATIVES = 192


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(max(s) for s in train.values() if s)
        itemnum = max(itemnum, max((max(s) for s in valid.values() if s), default=0))
    return train, valid, itemnum


class Histories(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(s, dtype=np.int64) for s in train.values() if len(s) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No trainable user histories")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        h = self.histories[index]
        # Random windows expose earlier transitions in histories exceeding maxlen.
        end = len(h)
        if len(h) > self.maxlen + 1 and random.random() < 0.5:
            end = random.randint(2, len(h))
        window = h[max(0, end - self.maxlen - 1) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        target = np.zeros(self.maxlen, dtype=np.int64)
        n = len(window) - 1
        seq[-n:] = window[:-1]
        target[-n:] = window[1:]
        return seq, target


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class Block(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x, mask):
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        a = a.transpose(1, 2).contiguous().reshape(b, t, d)
        x = (x + self.drop(self.out(a))) * mask
        return (x + self.drop(self.ff(self.norm2(x)))) * mask


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum,
        hidden_units=96,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.25,
        maxlen=100,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.input_norm = nn.LayerNorm(hidden_units)
        self.drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [Block(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)]
        )
        self.taste = nn.Sequential(
            nn.LayerNorm(hidden_units), nn.Linear(hidden_units, hidden_units), nn.GELU()
        )
        self.gate = nn.Linear(hidden_units * 2, hidden_units)
        self.output_norm = nn.LayerNorm(hidden_units)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.apply(self.initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.pos_emb.weight[0].zero_()
            self.item_bias.weight.zero_()
            self.gate.bias.fill_(-1.0)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def features(self, seq):
        valid = seq.ne(0)
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        # Positions count actual observed items rather than left padding.
        positions = valid.long().cumsum(1).clamp(max=self.maxlen)
        raw = self.item_emb(seq) * mask
        x = self.drop(self.input_norm(raw + self.pos_emb(positions))) * mask
        for block in self.blocks:
            x = block(x, mask)
        # Cumulative taste is causal at every supervised training position.
        count = valid.cumsum(1).clamp_min(1).unsqueeze(-1)
        taste = self.taste(raw.cumsum(1) / count)
        gate = torch.sigmoid(self.gate(torch.cat((x, taste), dim=-1)))
        return F.normalize(self.output_norm(x + gate * taste), dim=-1)

    def score_features(self, features, candidates):
        embeddings = F.normalize(self.item_emb(candidates), dim=-1)
        scale = self.log_scale.exp().clamp(5.0, 40.0)
        return scale * (features.unsqueeze(-2) * embeddings).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)

    def predict(self, seq, candidates):
        return self.score_features(self.features(seq)[:, -1], candidates)

    def training_loss(self, seq, target):
        keep = target.ne(0)
        features = self.features(seq)[keep]
        positive = target[keep]
        # Uniform sampled softmax matches the evaluator's uniform distractors.
        negatives = torch.randint(1, self.itemnum + 1, (NEGATIVES,), device=seq.device)
        scale = self.log_scale.exp().clamp(5.0, 40.0)
        positive_vectors = F.normalize(self.item_emb(positive), dim=-1)
        positive_logits = scale * (features * positive_vectors).sum(-1)
        positive_logits = positive_logits + self.item_bias(positive).squeeze(-1)
        negative_vectors = F.normalize(self.item_emb(negatives), dim=-1)
        negative_logits = scale * (features @ negative_vectors.T)
        negative_logits = negative_logits + self.item_bias(negatives).squeeze(-1)
        negative_logits = negative_logits.masked_fill(
            positive[:, None].eq(negatives[None]), -1e4
        )
        logits = torch.cat((positive_logits[:, None], negative_logits), dim=1)
        labels = torch.zeros(len(features), device=seq.device, dtype=torch.long)
        losses = F.cross_entropy(logits.float(), labels, reduction="none")
        # Equal user weight avoids domination by the longest histories.
        weights = (1.0 / keep.sum(1).clamp_min(1).float())[:, None].expand_as(target)[
            keep
        ]
        return (losses * weights).sum() / seq.shape[0]


def validation_cases(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        if not valid[u]:
            continue
        seen = set(train[u]) | set(valid[u])
        if itemnum - len(seen) < 99:
            raise ValueError("Not enough unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in seen:
                negatives.add(item)
        h = train[u][-maxlen:]
        sequences.append([0] * (maxlen - len(h)) + h)
        candidates.append([valid[u][0]] + list(negatives))
    if not sequences:
        raise ValueError("No validation users")
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(
        candidates, dtype=torch.long
    )


@torch.inference_mode()
def evaluate(model, cases, device):
    model.eval()
    hits, count = 0, 0
    sequences, candidates = cases
    for start in range(0, len(sequences), 128):
        scores = model.predict(
            sequences[start : start + 128].to(device),
            candidates[start : start + 128].to(device),
        ).float()
        # Stable tie handling places the target first, as in the baseline.
        rank = (scores[:, 1:] > scores[:, :1]).sum(1)
        hits += rank.lt(10).sum().item()
        count += len(scores)
    return hits / count


def train_category(name, device):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    config = dict(
        itemnum=itemnum,
        hidden_units=HIDDEN,
        num_blocks=BLOCKS,
        num_heads=HEADS,
        dropout_rate=DROPOUT,
        maxlen=MAXLEN,
    )
    model = SASRec(**config).to(device)
    dataset = Histories(train, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=STEPS * BATCH, generator=generator
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=1e-4
    )
    cases = validation_cases(train, valid, itemnum, MAXLEN)
    path = os.path.join(WORK, name + "_model.pt")
    best, best_epoch = -1.0, 0
    # One independent model and one selected checkpoint for this category.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for seq, target in loader:
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(seq, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.detach().item()
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            hr = evaluate(model, cases, device)
            if hr > best:
                best, best_epoch = hr, epoch
                torch.save(model.state_dict(), path)
            print(
                f"{name} epoch={epoch} loss={running / STEPS:.5f} val_HR@10={hr:.5f}",
                flush=True,
            )
            if epoch - best_epoch >= 40:
                break
        else:
            print(f"{name} epoch={epoch} loss={running / STEPS:.5f}", flush=True)
    with open(os.path.join(WORK, name + "_meta.pkl"), "wb") as f:
        pickle.dump(config, f)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    score = evaluate(model, cases, device)
    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def write_predict_script():
    script = r"""from __future__ import annotations
import os
import pickle
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = None
_class = None


def _load(dataset_name):
    global _device, _class
    if dataset_name in _models:
        return _models[dataset_name]
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset_name))
    if _class is None:
        spec = importlib.util.spec_from_file_location(
            'trained_single_network', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _class = module.SASRec
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    with open(os.path.join(_HERE, dataset_name + '_meta.pkl'), 'rb') as f:
        config = pickle.load(f)
    model = _class(**config).to(_device)
    model.load_state_dict(torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                                    map_location=_device, weights_only=True))
    model.eval()
    _models[dataset_name] = model
    return model


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    name = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(name)
    if len(candidates) == 0:
        return []
    h = list(history[-model.maxlen:])
    seq = torch.tensor([[0] * (model.maxlen - len(h)) + h],
                       dtype=torch.long, device=_device)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_device)
    return model.predict(seq, cand)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORK, "predict.py"), "w") as f:
        f.write(script)


def main():
    torch.set_num_threads(4)
    seed_everything(42)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WORK, exist_ok=True)
    scores = [train_category(name, device) for name in DATASETS]
    write_predict_script()
    with open(os.path.join(WORK, "validation_metrics.json"), "w") as f:
        json.dump(
            {
                "metric": "macro HR@10",
                "evaluation_seed": 42,
                "users_per_category": 200,
                "unique_negatives": 99,
                "per_category": dict(zip(DATASETS, scores)),
                "macro_HR@10": sum(scores) / len(scores),
            },
            f,
            indent=2,
        )
    score = sum(scores) / len(scores)
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
