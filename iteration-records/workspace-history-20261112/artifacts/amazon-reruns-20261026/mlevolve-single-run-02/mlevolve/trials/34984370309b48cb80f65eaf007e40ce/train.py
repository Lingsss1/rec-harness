from __future__ import annotations
import os
import math
import json
import time
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "working")
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
DIM = int(os.environ.get("SASREC_HIDDEN", "96"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
NEGATIVES = 8


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


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    itemnum = (
        int(stats[name]["itemnum"])
        if name in stats
        else max(x for history in train.values() for x in history)
    )
    return train, valid, itemnum


class Histories(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No eligible training histories")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        # Retain full short histories; alternate long suffixes with earlier windows.
        end = len(history)
        if end > self.maxlen + 1 and np.random.random() < 0.5:
            end = np.random.randint(2, end + 1)
        window = history[max(0, end - self.maxlen - 1) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        targets = np.zeros(self.maxlen, dtype=np.int64)
        length = len(window) - 1
        seq[:length] = window[:-1]
        targets[:length] = window[1:]
        return seq, targets


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class GRURec(nn.Module):
    def __init__(self, itemnum, dim=96, maxlen=128, dropout=0.25):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.embedding = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(dim, dim, num_layers=2, batch_first=True, dropout=dropout)
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.log_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        nn.init.normal_(self.embedding.weight, std=0.05)
        nn.init.zeros_(self.bias.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        for name, parameter in self.gru.named_parameters():
            if "weight_hh" in name:
                for gate in parameter.chunk(3, 0):
                    nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def encode(self, seq):
        x = self.dropout(self.input_norm(self.embedding(seq)))
        x, _ = self.gru(x)
        x = self.projection(self.dropout(x))
        return nn.functional.normalize(x, dim=-1)

    def score(self, features, candidates):
        embeddings = nn.functional.normalize(self.embedding(candidates), dim=-1)
        similarity = (features * embeddings).sum(-1)
        return self.log_scale.exp().clamp(max=30.0) * similarity + self.bias(
            candidates
        ).squeeze(-1)

    def predict(self, seq, candidates):
        states = self.encode(seq)
        lengths = seq.ne(0).sum(-1).clamp(min=1)
        final = states[torch.arange(seq.shape[0], device=seq.device), lengths - 1]
        return self.score(final[:, None, :], candidates)


def make_validation(train, valid, itemnum):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        target = valid[user][0]
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Fewer than 99 eligible validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        history = train[user][-MAXLEN:]
        sequences.append(history + [0] * (MAXLEN - len(history)))
        candidates.append([target] + list(negatives))
    if not sequences:
        raise ValueError("No validation users")
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(
        candidates, dtype=torch.long
    )


@torch.inference_mode()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    hits, ndcg = 0, 0.0
    for start in range(0, len(sequences), 128):
        scores = model.predict(
            sequences[start : start + 128].to(device),
            candidates[start : start + 128].to(device),
        )
        # Candidate zero is the held-out positive. Stable ties follow candidate order.
        rank = (scores[:, 1:] > scores[:, :1]).sum(-1)
        selected = rank < 10
        hits += selected.sum().item()
        ndcg += (selected.float() / torch.log2(rank.float() + 2)).sum().item()
    return hits / len(sequences), ndcg / len(sequences)


def train_category(name, device, deadline):
    seed_all(42)
    train, valid, itemnum = load_data(name)
    validation = make_validation(train, valid, itemnum)
    dataset = Histories(train, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    iterator = iter(loader)
    meta = dict(itemnum=itemnum, dim=DIM, maxlen=MAXLEN, dropout=0.25)
    model = GRURec(**meta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    best_key = (-1.0, -1.0)
    best_epoch = 0
    path = os.path.join(WORK, name + "_model.pt")
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        steps_done = 0
        for step in range(STEPS):
            try:
                seq, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                seq, targets = next(iterator)
            seq = seq.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # Trim trailing padding, without changing sequence order or targets.
            length = int(targets.ne(0).sum(1).max().item())
            seq, targets = seq[:, :length], targets[:, :length]
            mask = targets.ne(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                states = model.encode(seq)[mask]
                positive = targets[mask]
                negative = torch.randint(
                    1, itemnum + 1, (len(positive), NEGATIVES), device=device
                )
                # Never label this position's target as a negative.
                collision = negative.eq(positive[:, None])
                while collision.any():
                    negative[collision] = torch.randint(
                        1, itemnum + 1, (int(collision.sum()),), device=device
                    )
                    collision = negative.eq(positive[:, None])
                positive_logits = model.score(states, positive).float()
                negative_logits = model.score(states[:, None, :], negative).float()
                loss = nn.functional.softplus(-positive_logits).mean()
                loss = loss + nn.functional.softplus(negative_logits).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            steps_done += 1
            if time.monotonic() >= deadline:
                break
        hr, ndcg = evaluate(model, validation, device)
        key = (hr, ndcg)
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(
                {
                    "meta": meta,
                    "state_dict": model.state_dict(),
                    "validation_hr10": hr,
                    "epoch": epoch,
                },
                path,
            )
        print(
            f"{name} epoch={epoch} loss={total/max(1,steps_done):.5f} HR@10={hr:.5f} NDCG@10={ndcg:.5f}",
            flush=True,
        )
        if epoch - best_epoch >= 25 or time.monotonic() >= deadline:
            break
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    score, _ = evaluate(model, validation, device)
    del iterator, loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(name):
    global _CLASS
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name not in _MODELS:
        if _CLASS is None:
            spec = importlib.util.spec_from_file_location('recurrent_trial_model', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _CLASS = module.GRURec
        checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location=_DEVICE, weights_only=True)
        model = _CLASS(**checkpoint['meta']).to(_DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        _MODELS[name] = model
    return _MODELS[name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    name = dataset_name if dataset_name is not None else _NAMES[0]
    model = _load(name)
    if len(candidates) == 0:
        return []
    hist = list(history)[-model.maxlen:]
    sequence = hist + [0] * (model.maxlen - len(hist))
    seq = torch.tensor([sequence], dtype=torch.long, device=_DEVICE)
    cand = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return model.predict(seq, cand)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORK, "predict.py"), "w") as f:
        f.write(script)


def main():
    os.makedirs(WORK, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.monotonic()
    # Eight hours of training maximum, with the remaining time reserved for I/O.
    deadline = start + 8 * 3600
    scores = []
    write_predict_script()
    for index, name in enumerate(DATASETS):
        remaining = len(DATASETS) - index
        category_deadline = (
            time.monotonic() + max(1.0, deadline - time.monotonic()) / remaining
        )
        scores.append(train_category(name, device, category_deadline))
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
