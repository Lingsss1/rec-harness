from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import os
import random
import time
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
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "100"))
NEGATIVES = 2048
TOKEN_CAP = 2048


def seed_everything(seed):
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


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    if name in stats:
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(i for h in train.values() for i in h)
    return train, valid, itemnum


class PrefixDataset(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No usable training sequences")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        h = self.histories[index]
        end = len(h)
        if len(h) > 2 and np.random.random() < 0.20:
            end = np.random.randint(2, len(h) + 1)
        window = h[max(0, end - self.maxlen - 1) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        n = len(window) - 1
        seq[-n:] = window[:-1]
        pos[-n:] = window[1:]
        return torch.from_numpy(seq), torch.from_numpy(pos)


def init_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


class CausalBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + F.dropout(self.proj(a), self.dropout, self.training)
        x = x + self.ff(self.norm2(x))
        return x * mask


class SASRec(nn.Module):
    def __init__(self, itemnum, dim=96, blocks=2, heads=2, dropout=0.3, maxlen=100):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.dim = dim
        self.item_emb = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, dim)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [CausalBlock(dim, heads, dropout) for _ in range(blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.apply(self.initialize)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
            self.item_bias.weight.zero_()

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, seq):
        t = seq.shape[1]
        mask = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(t, device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.dim)
        x = self.drop(x + self.pos_emb(positions)) * mask
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) * mask

    def predict(self, seq, candidates):
        feat = self(seq)[:, -1]
        return (self.item_emb(candidates) * feat.unsqueeze(1)).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)

    def ranking_loss(self, seq, pos):
        all_features = self(seq)
        valid = pos != 0
        batch, length = pos.shape
        valid_count = valid.sum(dim=1)
        if batch == 0 or (valid_count == 0).any().item():
            raise ValueError("Every sequence must have at least one nonpadding target")
        positions = torch.arange(length, device=pos.device)
        endpoints = positions.expand_as(pos).masked_fill(~valid, -1).amax(dim=1)
        nonendpoint_count = valid_count - 1
        minimum_tokens = batch + (nonendpoint_count > 0).sum().item()
        if TOKEN_CAP < minimum_tokens:
            raise ValueError(
                "TOKEN_CAP must cover every endpoint and one nonendpoint "
                "per eligible sequence"
            )

        # Allocate balanced quotas, redistributing capacity from short sequences.
        budget = TOKEN_CAP - batch
        levels = torch.arange(length, device=pos.device)
        costs = torch.minimum(nonendpoint_count[:, None], levels[None, :]).sum(0)
        level = (costs <= budget).sum() - 1
        quotas = torch.minimum(nonendpoint_count, level)
        remaining = budget - quotas.sum()
        eligible = nonendpoint_count > quotas
        quotas = quotas + (eligible & (eligible.long().cumsum(0) <= remaining)).long()

        # Random ordering gives uniform sampling without replacement per sequence.
        nonendpoint = valid & (positions[None, :] != endpoints[:, None])
        order = torch.rand(pos.shape, device=pos.device).masked_fill(
            ~nonendpoint, float("inf")
        ).argsort(dim=1)
        selected = positions[None, :] < quotas[:, None]
        rows = torch.arange(batch, device=pos.device)
        sampled_indices = (rows[:, None] * length + order)[selected]
        indices = torch.cat([rows * length + endpoints, sampled_indices])
        losses = F.cross_entropy(
            logits.float(),
            torch.zeros(len(target), dtype=torch.long, device=seq.device),
            reduction="none",
        )
        endpoint_loss = losses[:batch]
        nonendpoint_sum = losses.new_zeros(batch).scatter_add(
            0, sampled_indices // length, losses[batch:]
        )
        sampled_nonendpoint_mean = nonendpoint_sum / quotas.clamp_min(1).float()
        # Correct for mandatory endpoint inclusion when estimating positional means.
        positional_mean = (
            endpoint_loss + nonendpoint_count.float() * sampled_nonendpoint_mean
        ) / valid_count.float()
        return (0.5 * endpoint_loss + 0.5 * positional_mean).mean()
        feat = all_features.reshape(-1, self.dim)[indices]
=======
        feat = all_features.reshape(-1, self.dim)[indices]
=======
        losses = F.cross_entropy(
            logits.float(),
            torch.zeros(len(target), dtype=torch.long, device=seq.device),
            reduction="none",
        )
        endpoint_loss = losses[:batch]
        nonendpoint_sum = losses.new_zeros(batch).scatter_add(
            0, sampled_indices // length, losses[batch:]
        )
        sampled_nonendpoint_mean = nonendpoint_sum / quotas.clamp_min(1).float()
        # Correct for mandatory endpoint inclusion when estimating positional means.
        positional_mean = (
            endpoint_loss + nonendpoint_count.float() * sampled_nonendpoint_mean
        ) / valid_count.float()
        return (0.5 * endpoint_loss + 0.5 * positional_mean).mean()
        feat = all_features.reshape(-1, self.dim)[indices]
        target = pos.reshape(-1)[indices]
        negative = torch.randint(1, self.itemnum + 1, (NEGATIVES,), device=seq.device)
        positive_score = (feat * self.item_emb(target)).sum(-1)
        positive_score = positive_score + self.item_bias(target).squeeze(-1)
        negative_score = feat @ self.item_emb(negative).T
        negative_score = negative_score + self.item_bias(negative).squeeze(-1)
        negative_score = negative_score.masked_fill(
            target[:, None] == negative[None, :], -1e4
        )
        logits = torch.cat([positive_score[:, None], negative_score], dim=1)
        losses = F.cross_entropy(
            logits.float(),
            torch.zeros(len(target), dtype=torch.long, device=seq.device),
            reduction="none",
        )
        endpoint_loss = losses[:batch]
        nonendpoint_sum = losses.new_zeros(batch).scatter_add(
            0, sampled_indices // length, losses[batch:]
        )
        sampled_nonendpoint_mean = nonendpoint_sum / quotas.clamp_min(1).float()
        # Correct for mandatory endpoint inclusion in the positional mean.
        positional_mean = (
            endpoint_loss + nonendpoint_count.float() * sampled_nonendpoint_mean
        ) / valid_count.float()
        return (0.5 * endpoint_loss + 0.5 * positional_mean).mean()


def validation_bank(train, valid, itemnum, maxlen):
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
            raise ValueError("Fewer than 99 unique eligible negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        h = train[u][-maxlen:]
        sequences.append([0] * (maxlen - len(h)) + h)
        candidates.append([valid[u][0]] + list(negatives))
    if not sequences:
        raise ValueError("Empty validation split")
    return torch.tensor(sequences), torch.tensor(candidates)


@torch.no_grad()
def evaluate(model, bank, device):
    model.eval()
    ranks = []
    for start in range(0, len(bank[0]), 64):
        seq = bank[0][start : start + 64].to(device)
        candidates = bank[1][start : start + 64].to(device)
        scores = model.predict(seq, candidates).float()
        # Stable candidate ordering places the positive first on exact ties.
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        rank = (order == 0).long().argmax(dim=1)
        ranks.append(rank.cpu())
    ranks = torch.cat(ranks)
    hr = (ranks < 10).float().mean().item()
    ndcg = (
        torch.where(
            ranks < 10,
            1.0 / torch.log2(ranks.float() + 2),
            torch.zeros_like(ranks, dtype=torch.float),
        )
        .mean()
        .item()
    )
    return hr, ndcg


def train_one(name, device):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    cfg = dict(itemnum=itemnum, dim=96, blocks=2, heads=2, dropout=0.3, maxlen=MAXLEN)
    model = SASRec(**cfg).to(device)
    bank = validation_bank(train, valid, itemnum, MAXLEN)
    data = PrefixDataset(train, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(
        data, replacement=True, num_samples=BATCH * STEPS, generator=generator
    )
    loader = DataLoader(
        data,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
        worker_init_fn=init_worker,
        generator=torch.Generator().manual_seed(42),
    )
    decay, no_decay = [], []
    for key, value in model.named_parameters():
        (no_decay if value.ndim < 2 or "item_bias" in key else decay).append(value)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.001,
        betas=(0.9, 0.98),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best = (-1.0, -1.0)
    best_epoch = 0
    path = os.path.join(WORK, name + "_model.pt")
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for step, (seq, pos) in enumerate(loader):
            progress = (epoch - 1) * STEPS + step + 1
            warmup = min(1.0, progress / max(STEPS * 3, 1))
            for group in opt.param_groups:
                group["lr"] = 0.001 * warmup
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_bf16
                else contextlib.nullcontext()
            )
            with amp:
                loss = model.ranking_loss(seq, pos)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach())
        expired = time.monotonic() - started > 7000
        check = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired
        suffix = ""
        if check:
            hr, ndcg = evaluate(model, bank, device)
            suffix = f" val_HR@10={hr:.6f} val_NDCG@10={ndcg:.6f}"
            if (hr, ndcg) > best:
                best = (hr, ndcg)
                best_epoch = epoch
                torch.save({"config": cfg, "state": model.state_dict()}, path)
        print(f"{name} epoch={epoch} loss={loss_sum / STEPS:.5f}" + suffix, flush=True)
        if expired or (check and epoch - best_epoch >= 35):
            break
    saved = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(saved["state"])
    score, _ = evaluate(model, bank, device)
    del loader, opt, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def write_predict_script():
    script = """from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_class = None


def _load(name):
    global _model_class
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _models:
        if _model_class is None:
            spec = importlib.util.spec_from_file_location(
                'trained_sasrec_architecture', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _model_class = module.SASRec
        checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'),
                                map_location='cpu', weights_only=True)
        model = _model_class(**checkpoint['config'])
        model.load_state_dict(checkpoint['state'])
        model.to(_device).eval()
        _models[name] = model
    return _models[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    name = dataset_name if dataset_name is not None else _NAMES[0]
    model = _load(name)
    h = list(history)[-model.maxlen:]
    sequence = [0] * (model.maxlen - len(h)) + h
    seq = torch.tensor([sequence], dtype=torch.long, device=_device)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_device)
    return model.predict(seq, cand)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORK, "predict.py"), "w") as f:
        f.write(script)


def main():
    torch.set_num_threads(4)
    os.makedirs(WORK, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = [train_one(name, device) for name in DATASETS]
    write_predict_script()
    score = float(np.mean(scores))
    with open(os.path.join(WORK, "validation.json"), "w") as f:
        json.dump(
            {
                "metric": "macro HR@10",
                "evaluation_seed": 42,
                "users_per_category": 200,
                "unique_negatives": 99,
                "per_category": dict(zip(DATASETS, scores)),
                "score": score,
            },
            f,
            indent=2,
        )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()