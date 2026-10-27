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
        self.recent_window = 5
        self.recent_projection = nn.Linear(dim, dim)
        self.recent_gate = nn.Linear(2 * dim, dim)
        self.recent_norm = nn.LayerNorm(dim)
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

    def recent_mean(self, seq):
        # Prefix counts locate the last five valid items across padding gaps.
        # Gathered prefix differences exclude all future items and return zero
        # for empty prefixes. Memory uses unscaled, tied item embeddings.
        valid = seq.ne(0)
        embeddings = self.item_emb(seq) * valid.unsqueeze(-1)
        sums = F.pad(embeddings.cumsum(dim=1), (0, 0, 1, 0))
        counts = F.pad(valid.long().cumsum(dim=1), (1, 0))
        targets = (counts[:, 1:] - self.recent_window).clamp_min(0)
        starts = torch.searchsorted(
            counts.contiguous(), targets.contiguous(), right=False
        )
        window_sums = sums[:, 1:] - sums.gather(
            1, starts.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        window_counts = counts[:, 1:] - counts.gather(1, starts)
        return window_sums / window_counts.clamp_min(1).unsqueeze(-1)

    def forward(self, seq):
        t = seq.shape[1]
        mask = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        positions = torch.arange(t, device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.dim)
        x = self.drop(x + self.pos_emb(positions)) * mask
        for layer in self.layers:
            x = layer(x, mask)
        h = self.norm(x) * mask
        m = self.recent_mean(seq)
        r = self.recent_projection(m)
        g = torch.sigmoid(self.recent_gate(torch.cat([h, m], dim=-1)))
        fused = self.recent_norm(h + g * r)
        return fused * mask

    def predict(self, seq, candidates):
        feat = self(seq)[:, -1]
        return (self.item_emb(candidates) * feat.unsqueeze(1)).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)

    def ranking_loss(self, seq, pos):
        all_features = self(seq)
        indices = torch.nonzero(pos.reshape(-1) != 0).flatten()
        if indices.numel() > TOKEN_CAP:
            indices = indices[
                torch.randperm(indices.numel(), device=indices.device)[:TOKEN_CAP]
            ]
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
        return F.cross_entropy(
            logits.float(),
            torch.zeros(len(target), dtype=torch.long, device=seq.device),
        )


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


def verify_recent_fusion(model):
    # Checks use the category model itself, without fitting another predictor
    # or advancing the training random streams.
    device = model.item_emb.weight.device
    was_training = model.training
    try:
        model.eval()
        rows = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
            [1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ]
        seq = torch.tensor(rows, device=device, dtype=torch.long)
        seq = seq[:, -min(model.maxlen, seq.shape[1]):]
        seq = torch.where(seq.ne(0), (seq - 1) % model.itemnum + 1, seq)
        with torch.no_grad():
            expected = torch.zeros(
                *seq.shape, model.dim, device=device,
                dtype=model.item_emb.weight.dtype,
            )
            for b in range(seq.shape[0]):
                for t in range(seq.shape[1]):
                    prefix = seq[b, :t + 1]
                    items = prefix[prefix.ne(0)][-model.recent_window:]
                    if items.numel():
                        expected[b, t] = model.item_emb(items).mean(dim=0)
            torch.testing.assert_close(
                model.recent_mean(seq), expected, rtol=1e-4, atol=1e-6
            )
            features = model(seq)
            assert torch.isfinite(features).all(), 'Nonfinite fused features'
            assert torch.count_nonzero(features[seq.eq(0)]) == 0
            if seq.shape[1] > 1:
                changed = seq.clone()
                changed[:, -1] = changed[:, -1] % model.itemnum + 1
                torch.testing.assert_close(
                    features[:, :-1], model(changed)[:, :-1],
                    rtol=1e-5, atol=1e-6,
                )
        parameters = tuple(
            p for layer in (
                model.recent_projection, model.recent_gate, model.recent_norm
            ) for p in layer.parameters()
        )
        gradients = torch.autograd.grad(model(seq).square().mean(), parameters)
        assert all(torch.isfinite(g).all() for g in gradients), (
            'Nonfinite fusion-layer gradients'
        )
    finally:
        model.train(was_training)


def train_one(name, device):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    cfg = dict(itemnum=itemnum, dim=96, blocks=2, heads=2, dropout=0.3, maxlen=MAXLEN)
    model = SASRec(**cfg).to(device)
    verify_recent_fusion(model)
    checkpoint_cfg = dict(
        cfg, architecture="sasrec_recent_gated_v1",
        recent_window=model.recent_window,
    )
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
                # Store deterministic candidate scores to verify the selected
                # checkpoint after serialization, without averaging checkpoints.
                with torch.no_grad():
                    verification_scores = model.predict(
                        bank[0][:2].to(device), bank[1][:2].to(device)
                    ).float().cpu()
                torch.save({
                    "config": checkpoint_cfg,
                    "state": model.state_dict(),
                    "verification_scores": verification_scores,
                }, path)
        print(f"{name} epoch={epoch} loss={loss_sum / STEPS:.5f}" + suffix, flush=True)
        if expired or (check and epoch - best_epoch >= 35):
            break
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["config"] != checkpoint_cfg:
        raise ValueError("Selected checkpoint architecture/config mismatch")
    model.load_state_dict(saved["state"], strict=True)
    model.eval()
    with torch.no_grad():
        reloaded_scores = model.predict(
            bank[0][:2].to(device), bank[1][:2].to(device)
        ).float().cpu()
    torch.testing.assert_close(
        reloaded_scores, saved["verification_scores"], rtol=1e-5, atol=1e-6
    )
    verify_recent_fusion(model)
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
        config = dict(checkpoint['config'])
        if config.pop('architecture') != 'sasrec_recent_gated_v1':
            raise ValueError('Unsupported checkpoint architecture')
        recent_window = config.pop('recent_window')
        model = _model_class(**config)
        if recent_window != model.recent_window:
            raise ValueError('Checkpoint recent-window mismatch')
        model.load_state_dict(checkpoint['state'], strict=True)
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
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_started = time.monotonic()
    scores = [train_one(name, device) for name in DATASETS]
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_elapsed = time.monotonic() - training_started
    write_predict_script()
    # Exercise the exported loader and public prediction interface against
    # the serialized score probes. Each route loads its one selected model.
    spec = importlib.util.spec_from_file_location(
        "exported_predictor_verification", os.path.join(WORK, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    for name in DATASETS:
        train, valid, itemnum = load_data(name)
        bank = validation_bank(train, valid, itemnum, MAXLEN)
        checkpoint = torch.load(
            os.path.join(WORK, name + "_model.pt"),
            map_location="cpu", weights_only=True,
        )
        for i in range(min(2, len(bank[0]))):
            history = bank[0][i]
            actual = predictor.predict(
                0, history[history.ne(0)].tolist(), bank[1][i].tolist(),
                dataset_name=name,
            )
            torch.testing.assert_close(
                torch.tensor(actual), checkpoint["verification_scores"][i],
                rtol=1e-5, atol=1e-6,
            )
        predictor._models.clear()
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.monotonic() - training_started
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
                "architecture": "sasrec_recent_gated_v1",
                "recent_window": 5,
                "training_elapsed_seconds": training_elapsed,
                "single_gpu_elapsed_seconds": (
                    total_elapsed if device.type == "cuda" else None
                ),
                "cost_note": (
                    "Measured single-GPU wall time includes verification; "
                    "host accounting is authoritative for allocated GPU cost."
                ),
            },
            f,
            indent=2,
        )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
