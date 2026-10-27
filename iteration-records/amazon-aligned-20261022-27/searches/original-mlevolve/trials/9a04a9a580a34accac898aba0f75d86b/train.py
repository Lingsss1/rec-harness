from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
WORKING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "64"))
BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
HEADS = int(os.environ.get("SASREC_HEADS", "2"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.2"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
WD = float(os.environ.get("SASREC_WD", "0.00001"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))
NEGATIVES = 64


def seed_all(seed):
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
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(max(items) for items in train.values() if items)
    return train, valid, itemnum


class Windows(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.seen = [set(h.tolist()) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No eligible training histories")
        if any(len(s) >= itemnum for s in self.seen):
            raise ValueError("Cannot sample unseen training negatives")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        h = self.histories[index]
        end = len(h)
        if len(h) > self.maxlen + 1 and np.random.random() < 0.3:
            end = np.random.randint(2, len(h) + 1)
        window = h[max(0, end - self.maxlen - 1) : end]
        n = len(window) - 1
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        seq[-n:] = window[:-1]
        pos[-n:] = window[1:]
        seen = self.seen[index]
        negative = np.random.randint(
            1, self.itemnum + 1, size=NEGATIVES, dtype=np.int64
        )
        for j in range(NEGATIVES):
            while int(negative[j]) in seen:
                negative[j] = np.random.randint(1, self.itemnum + 1)
        return seq, pos, negative


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class Block(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep):
        b, length, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            b, length, 3, self.heads, hidden // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attention = attention.transpose(1, 2).reshape(b, length, hidden)
        x = (x + self.drop(self.proj(attention))) * keep
        x = (x + self.drop(self.ff(self.norm2(x)))) * keep
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum,
        hidden_units=64,
        num_blocks=2,
        num_heads=2,
        dropout_rate=0.2,
        maxlen=128,
    ):
        super().__init__()
        if hidden_units % num_heads:
            raise ValueError("hidden_units must be divisible by num_heads")
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [Block(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_units)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq):
        keep = seq.ne(0).unsqueeze(-1)
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = (
            self.drop(
                self.item_emb(seq) * math.sqrt(self.hidden_units)
                + self.pos_emb(positions)
            )
            * keep
        )
        for block in self.blocks:
            x = block(x, keep)
        return self.norm(x) * keep

    def ranking_loss(self, seq, pos, negative):
        features = self(seq)
        positive_logits = (features * self.item_emb(pos)).sum(-1)
        positive_logits = positive_logits + self.item_bias(pos).squeeze(-1)
        negative_logits = torch.bmm(features, self.item_emb(negative).transpose(1, 2))
        negative_logits = negative_logits + self.item_bias(negative).squeeze(
            -1
        ).unsqueeze(1)
        logits = torch.cat([positive_logits.unsqueeze(-1), negative_logits], -1)
        valid = pos.ne(0)
        logits = logits[valid].float()
        losses = F.cross_entropy(
            logits,
            torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device),
            reduction='none',
        )
        positions = torch.arange(
            pos.shape[1], dtype=torch.float32, device=pos.device
        ).unsqueeze(0).expand_as(pos)
        final_positions = positions.masked_fill(~valid, -1.0).amax(
            dim=1, keepdim=True
        )
        d = (final_positions - positions).clamp_min(0.0)
        w = 0.25 + 0.75 * 2 ** (-d / 8.0)
        weights = w[valid]
        return (losses * weights).sum() / weights.sum()

    def predict(self, seq, candidates):
        features = self(seq)[:, -1]
        return (self.item_emb(candidates) * features.unsqueeze(1)).sum(
            -1
        ) + self.item_bias(candidates).squeeze(-1)


def pack_history(history, maxlen):
    result = np.zeros(maxlen, dtype=np.int64)
    h = history[-maxlen:]
    if h:
        result[-len(h) :] = h
    return result


def make_validation(train, valid, itemnum, maxlen):
    # Matches the supplied benchmark protocol: fixed users and unique negatives.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for u in users:
        if not valid[u]:
            continue
        excluded = set(train[u]) | set(valid[u])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negative = set()
        while len(negative) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negative.add(item)
        sequences.append(pack_history(train[u], maxlen))
        candidates.append([valid[u][0]] + list(negative))
    if not sequences:
        raise ValueError("No validation users")
    return torch.as_tensor(np.stack(sequences)), torch.tensor(
        candidates, dtype=torch.long
    )


@torch.inference_mode()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    ranks = []
    for start in range(0, len(sequences), 100):
        scores = model.predict(
            sequences[start : start + 100].to(device),
            candidates[start : start + 100].to(device),
        ).float()
        # Positive occupies index zero; stable ordering matches the baseline on ties.
        rank = (scores[:, 1:] > scores[:, :1]).sum(1)
        ranks.append(rank.cpu())
    ranks = torch.cat(ranks)
    hr = (ranks < 10).float().mean().item()
    ndcg = torch.where(ranks < 10, 1 / torch.log2(ranks.float() + 2), 0).mean().item()
    return hr, ndcg


def train_one(name, device, deadline):
    seed_all(42)
    train, valid, itemnum = load_dataset(name)
    meta = dict(
        itemnum=itemnum,
        hidden_units=HIDDEN,
        num_blocks=BLOCKS,
        num_heads=HEADS,
        dropout_rate=DROPOUT,
        maxlen=MAXLEN,
    )
    model = SASRec(**meta).to(device)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    loader = DataLoader(
        Windows(train, itemnum, MAXLEN),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42),
        drop_last=False,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WD
    )
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best = (-1.0, -1.0)
    best_epoch = 0
    path = os.path.join(WORKING, name + "_model.pt")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        completed = 0
        for step in range(STEPS):
            try:
                seq, pos, neg = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                seq, pos, neg = next(iterator)
            seq, pos, neg = [x.to(device, non_blocking=True) for x in (seq, pos, neg)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                loss = model.ranking_loss(seq, pos, neg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            completed += 1
            if time.monotonic() >= deadline:
                break
        expired = time.monotonic() >= deadline
        check = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired
        if check:
            metric = evaluate(model, validation, device)
            if metric > best:
                best = metric
                best_epoch = epoch
                torch.save(model.state_dict(), path)
            print(
                f"{name} epoch={epoch} loss={loss_sum / max(completed, 1):.5f} "
                f"validation_HR@10={metric[0]:.5f} validation_NDCG@10={metric[1]:.5f}",
                flush=True,
            )
        else:
            print(
                f"{name} epoch={epoch} loss={loss_sum / max(completed, 1):.5f}",
                flush=True,
            )
        if expired or epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    score, _ = evaluate(model, validation, device)
    with open(os.path.join(WORKING, name + "_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    del iterator, loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


PREDICTOR = r"""from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_pack_history = None


def _ensure_loaded():
    global _device, _pack_history
    if _models:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = importlib.util.spec_from_file_location(
        'recommender_architecture', os.path.join(_HERE, '..', 'train.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _pack_history = module.pack_history
    for name in _NAMES:
        with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        model = module.SASRec(**meta).to(_device)
        model.load_state_dict(torch.load(os.path.join(_HERE, name + '_model.pt'),
                                        map_location=_device, weights_only=True))
        model.eval()
        _models[name] = model
        _metas[name] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    name = dataset_name if dataset_name is not None else _NAMES[0]
    if name not in _models:
        raise ValueError('Unknown dataset: ' + str(name))
    if not candidates:
        return []
    seq = torch.as_tensor(_pack_history(history, _metas[name]['maxlen']),
                          dtype=torch.long, device=_device).unsqueeze(0)
    cand = torch.tensor([candidates], dtype=torch.long, device=_device)
    with torch.inference_mode():
        return _models[name].predict(seq, cand)[0].float().cpu().tolist()
"""


def main():
    seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORKING, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reserve substantial time below the nine-hour hard limit and guarantee
    # that every category receives its own training allocation.
    start = time.monotonic()
    overall_deadline = start + 8.5 * 3600
    scores = []
    for i, name in enumerate(DATASETS):
        now = time.monotonic()
        category_deadline = now + max(1.0, overall_deadline - now) / (len(DATASETS) - i)
        scores.append(train_one(name, device, category_deadline))
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(PREDICTOR)
    score = float(np.mean(scores))
    with open(os.path.join(WORKING, "validation_metrics.json"), "w") as f:
        json.dump(
            {
                "protocol": {
                    "seed": 42,
                    "users_per_category": 200,
                    "unique_negatives": 99,
                    "metric": "HR@10",
                },
                "per_dataset": dict(zip(DATASETS, scores)),
                "macro_HR@10": score,
            },
            f,
            indent=2,
        )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
