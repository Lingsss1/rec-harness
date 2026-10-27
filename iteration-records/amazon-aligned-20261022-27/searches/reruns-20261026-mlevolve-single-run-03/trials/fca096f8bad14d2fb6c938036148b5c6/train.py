from __future__ import annotations

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
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "working")
HIDDEN = int(os.environ.get("SASREC_HIDDEN", "96"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "64"))
DROPOUT = float(os.environ.get("SASREC_DROPOUT", "0.25"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
LR = float(os.environ.get("SASREC_LR", "0.001"))
NEGATIVES = 256
TOKEN_LIMIT = 2048
# Bound this run below the nine-hour host limit, with time for every category.
RUN_SECONDS = 8.0 * 3600


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
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
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
        itemnum = max(max(h) for h in train.values() if h)
        itemnum = max(itemnum, max((max(h) for h in valid.values() if h), default=0))
    return train, valid, itemnum


def prepare_sequence(history, maxlen):
    result = np.zeros(maxlen, dtype=np.int64)
    history = history[-maxlen:]
    if len(history):
        result[-len(history) :] = history
    return result


class PrefixDataset(Dataset):
    def __init__(self, train, maxlen, length):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        if not self.histories:
            raise ValueError("No trainable sequences")
        self.maxlen = maxlen
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        h = self.histories[np.random.randint(len(self.histories))]
        # Random endpoints expose earlier transitions in long histories;
        # terminal windows retain emphasis on the prediction-time regime.
        end = len(h) if np.random.random() < 0.5 else np.random.randint(2, len(h) + 1)
        start = max(0, end - self.maxlen - 1)
        window = h[start:end]
        x = prepare_sequence(window[:-1], self.maxlen)
        y = prepare_sequence(window[1:], self.maxlen)
        return torch.from_numpy(x), torch.from_numpy(y)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class CausalConv(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, 3, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x.transpose(1, 2), (self.left, 0))).transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.conv1 = CausalConv(hidden, dilation)
        self.conv2 = CausalConv(hidden, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(F.gelu(self.conv1(self.norm1(x) * mask))) * mask
        h = self.drop(self.conv2(F.gelu(self.norm2(h)) * mask)) * mask
        return (x + h) * mask


class ConvRec(nn.Module):
    def __init__(self, itemnum, hidden=96, maxlen=64, dropout=0.25):
        super().__init__()
        self.itemnum = itemnum
        self.hidden = hidden
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ConvBlock(hidden, d, dropout) for d in (1, 2, 4, 8)]
        )
        self.final_norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq):
        mask = (seq != 0).unsqueeze(-1).to(self.item_emb.weight.dtype)
        x = self.input_drop(self.item_emb(seq) * math.sqrt(self.hidden)) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x) * mask

    def predict(self, seq, candidates):
        h = self(seq)[:, -1]
        return (self.item_emb(candidates) * h.unsqueeze(1)).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)

    def ranking_loss(self, seq, targets):
        features = self(seq)
        b, t = torch.where(targets != 0)
        if b.numel() > TOKEN_LIMIT:
            take = torch.randperm(b.numel(), device=seq.device)[:TOKEN_LIMIT]
            b, t = b[take], t[take]
        h = features[b, t]
        positive = targets[b, t]
        negatives = torch.randint(1, self.itemnum + 1, (NEGATIVES,), device=seq.device)
        positive_scores = (h * self.item_emb(positive)).sum(-1)
        positive_scores = positive_scores + self.item_bias(positive).squeeze(-1)
        negative_scores = h @ self.item_emb(negatives).T
        negative_scores = negative_scores + self.item_bias(negatives).squeeze(-1)[None]
        allowed = positive[:, None] != negatives[None, :]
        logits = torch.cat(
            [
                positive_scores.float().unsqueeze(-1),
                negative_scores.float().masked_fill(~allowed, float('-inf')),
            ],
            dim=-1,
        )
        return (torch.logsumexp(logits, dim=-1) - positive_scores.float()).mean()


def make_validation(train, valid, itemnum, maxlen):
    # Match the supplied benchmark's user sampling and unique uniform negatives.
    # This RNG never alters training or worker RNG states.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidate_rows = [], []
    for u in users:
        if not valid[u]:
            continue
        seen = set(train[u]) | set(valid[u])
        if itemnum - len(seen) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            candidate = rng.randint(1, itemnum)
            if candidate not in seen:
                negatives.add(candidate)
        sequences.append(prepare_sequence(train[u], maxlen))
        candidate_rows.append([valid[u][0]] + list(negatives))
    if not sequences:
        raise ValueError("No eligible validation users")
    return torch.from_numpy(np.stack(sequences)), torch.tensor(
        candidate_rows, dtype=torch.long
    )


@torch.no_grad()
def evaluate(model, validation, device):
    model.eval()
    sequences, candidates = validation
    hits = 0
    for start in range(0, len(sequences), 64):
        scores = model.predict(
            sequences[start : start + 64].to(device),
            candidates[start : start + 64].to(device),
        )
        # Stable tie handling: target is first in the candidate list.
        rank = (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        hits += int((rank < 10).sum().item())
    return hits / len(sequences)


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
            'Electronics', 'CDs_and_Vinyl']
MODELS = {}
DEVICE = None
MODEL_CLASS = None


def load_model(dataset):
    global DEVICE, MODEL_CLASS
    if dataset in MODELS:
        return MODELS[dataset]
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if MODEL_CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'convolutional_rec_training', os.path.join(HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MODEL_CLASS = module.ConvRec
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    checkpoint = torch.load(os.path.join(HERE, dataset + '_model.pt'),
                            map_location='cpu', weights_only=True)
    model = MODEL_CLASS(**checkpoint['config']).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    MODELS[dataset] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = DATASETS[0] if dataset_name is None else dataset_name
    model = load_model(dataset)
    if not candidates:
        return []
    hist = list(history[-model.maxlen:])
    seq = [0] * (model.maxlen - len(hist)) + hist
    seq = torch.tensor([seq], device=DEVICE, dtype=torch.long)
    candidates = torch.tensor([candidates], device=DEVICE, dtype=torch.long)
    with torch.inference_mode():
        return model.predict(seq, candidates)[0].float().cpu().tolist()
"""
    with open(os.path.join(WORK, "predict.py"), "w") as f:
        f.write(script)


def train_category(name, device, deadline):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    validation = make_validation(train, valid, itemnum, MAXLEN)
    config = dict(itemnum=itemnum, hidden=HIDDEN, maxlen=MAXLEN, dropout=DROPOUT)
    model = ConvRec(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        weight_decay=float(os.environ.get("SASREC_WD", "0.0001")),
    )
    dataset = PrefixDataset(train, MAXLEN, BATCH * STEPS)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )
    path = os.path.join(WORK, name + "_model.pt")
    best_score, best_epoch = -1.0, 0
    last_validation = float("nan")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = 0.0
        completed = 0
        # Warmup and mild decay avoid destabilizing the sparse item embeddings.
        multiplier = min(1.0, epoch / 5.0) * (
            0.35 + 0.65 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(EPOCHS, 1)))
        )
        for group in optimizer.param_groups:
            group["lr"] = LR * multiplier
        for seq, targets in loader:
            seq = seq.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.ranking_loss(seq, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses += float(loss.detach())
            completed += 1
            if time.monotonic() >= deadline:
                break
        timed_out = time.monotonic() >= deadline
        check = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or timed_out
        if check:
            last_validation = evaluate(model, validation, device)
            if last_validation > best_score:
                best_score, best_epoch = last_validation, epoch
                torch.save({"config": config, "state_dict": model.state_dict()}, path)
        print(
            f"{name} epoch={epoch} loss={losses / max(1, completed):.5f} "
            f"val_HR@10={last_validation:.5f} best_HR@10={best_score:.5f}",
            flush=True,
        )
        if timed_out or epoch - best_epoch >= 35:
            break
    # Recompute the metric from the actual single selected checkpoint.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    score = evaluate(model, validation, device)
    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def main():
    os.makedirs(WORK, exist_ok=True)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_predict_script()
    started = time.monotonic()
    scores = []
    for index, name in enumerate(DATASETS):
        remaining = RUN_SECONDS - (time.monotonic() - started)
        deadline = time.monotonic() + max(1.0, remaining / (len(DATASETS) - index))
        scores.append(train_category(name, device, deadline))
    # Only train/validation files are accessed. The host independently performs
    # held-out inference through working/predict.py; test labels are unavailable.
    score = float(np.mean(scores))
    with open(os.path.join(WORK, "validation_metrics.json"), "w") as f:
        json.dump(
            {
                "metric": "HR@10",
                "evaluation_seed": 42,
                "users_per_category": 200,
                "unique_negatives": 99,
                "per_category": dict(zip(DATASETS, scores)),
                "macro_average": score,
            },
            f,
            indent=2,
        )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()