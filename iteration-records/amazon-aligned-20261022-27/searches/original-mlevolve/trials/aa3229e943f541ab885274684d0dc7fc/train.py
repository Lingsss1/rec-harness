from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
HERE = os.path.dirname(os.path.abspath(__file__))
WORKING = os.path.join(HERE, "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
BATCH = int(os.environ.get("VAE_BATCH", "256"))
PATIENCE = 25


def read_split(path):
    result = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                result[int(fields[0])].append(int(fields[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as handle:
            stats = json.load(handle)
        nitems = int(stats[name]["itemnum"])
    else:
        nitems = max(max(items) for items in train.values() if items)
        nitems = max(nitems, max((max(v) for v in valid.values() if v), default=0))
    return train, valid, nitems


def transform_history(items):
    # Compute occurrence ages before deduplicating or sorting item IDs.
    items = np.asarray(list(items), dtype=np.int64)
    if not items.size:
        return items, np.empty(0, dtype=np.float32)
    ids, inverse = np.unique(items, return_inverse=True)
    ages = items.size - 1 - np.arange(items.size, dtype=np.float64)
    occurrence_weights = np.exp2(-ages / 10.0)
    weights = np.bincount(inverse, weights=occurrence_weights, minlength=ids.size)
    # Aggregate repeated occurrences, then normalize once before input dropout.
    norm = np.linalg.norm(weights)
    if norm > 0:
        weights /= norm
    return ids, weights.astype(np.float32)


class Histories(Dataset):
    def __init__(self, train):
        self.rows = [
            transform_history(items)
            for items in train.values()
            if items
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def pack_rows(rows):
    lengths = np.asarray([len(row[0]) for row in rows], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    ids = np.concatenate([row[0] for row in rows]) if offsets[-1] else np.empty(0, dtype=np.int64)
    weights = np.concatenate([row[1] for row in rows]) if offsets[-1] else np.empty(0, dtype=np.float32)
    owners = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    # Unique IDs and owners retain binary reconstruction targets; weights only
    # affect the encoder and remain aligned with those IDs.
    return (
        torch.from_numpy(ids),
        torch.from_numpy(offsets),
        torch.from_numpy(owners),
        torch.from_numpy(weights),
    )


class MultVAE(nn.Module):
    def __init__(self, itemnum, hidden=256, latent=64, dropout=0.5):
        super().__init__()
        self.itemnum = itemnum
        self.dropout = dropout
        self.encoder_items = nn.EmbeddingBag(
            itemnum + 1, hidden, mode="sum", include_last_offset=True, padding_idx=0
        )
        self.encoder_bias = nn.Parameter(torch.zeros(hidden))
        self.posterior = nn.Linear(hidden, 2 * latent)
        self.decoder_hidden = nn.Linear(latent, hidden)
        self.decoder = nn.Linear(hidden, itemnum + 1)
        nn.init.normal_(self.encoder_items.weight, std=0.02)
        with torch.no_grad():
            self.encoder_items.weight[0].zero_()
        for layer in [self.posterior, self.decoder_hidden, self.decoder]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode(self, ids, offsets, input_weights=None):
        if input_weights is None:
            lengths = offsets[1:] - offsets[:-1]
            weights = torch.repeat_interleave(lengths.clamp_min(1).float().rsqrt(), lengths)
        else:
            weights = input_weights
        # History weights already have unit L2 norm; do not renormalize dropout.
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        h = torch.tanh(
            self.encoder_items(ids, offsets, per_sample_weights=weights)
            + self.encoder_bias
        )
        mu, logvar = self.posterior(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-12.0, 8.0)

    def representation(self, ids, offsets, input_weights=None):
        mu, logvar = self.encode(ids, offsets, input_weights)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return torch.tanh(self.decoder_hidden(z)), mu, logvar

    def forward(self, ids, offsets, owners, anneal, input_weights=None):
        h, mu, logvar = self.representation(ids, offsets, input_weights)
        logits = self.decoder(h).float()
        logits[:, 0] = -1e9
        logprobs = F.log_softmax(logits, dim=-1)
        # Standard multinomial reconstruction: every observed training item
        # contributes once. Validation items never appear in these targets.
        nll = -logprobs[owners, ids].sum() / h.shape[0]
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return nll + anneal * kl

    def predict(self, ids, offsets, candidates, input_weights=None):
        h, _, _ = self.representation(ids, offsets, input_weights)
        weights = F.embedding(candidates, self.decoder.weight)
        biases = self.decoder.bias[candidates]
        return (weights * h.unsqueeze(1)).sum(-1) + biases


def validation_batches(train, valid, nitems):
    # Fixed candidates across checkpoints; local RNG never changes training RNG.
    rng = random.Random(42)
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = rng.sample(users, 200)
    examples = []
    for u in users:
        target = valid[u][0]
        excluded = set(train[u]) | set(valid[u])
        if nitems - len(excluded) < 99:
            raise ValueError("Fewer than 99 eligible unique negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, nitems)
            if item not in excluded:
                negatives.add(item)
        examples.append(
            (
                transform_history(train[u]),
                [target] + list(negatives),
            )
        )
    if not examples:
        raise ValueError("No eligible validation users")
    batches = []
    for start in range(0, len(examples), 64):
        part = examples[start : start + 64]
        batches.append(
            (
                pack_rows([x[0] for x in part]),
                torch.tensor([x[1] for x in part], dtype=torch.long),
            )
        )
    return batches


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    hits = ndcg = count = 0
    for packed, candidates in batches:
        ids, offsets, _, input_weights = [x.to(device) for x in packed]
        scores = model.predict(ids, offsets, candidates.to(device), input_weights)
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        ranks = (order == 0).long().argmax(dim=1)
        hits += (ranks < 10).sum().item()
        ndcg += (
            torch.where(
                ranks < 10,
                1.0 / torch.log2(ranks.float() + 2),
                torch.zeros_like(ranks, dtype=torch.float),
            )
            .sum()
            .item()
        )
        count += len(ranks)
    return hits / count, ndcg / count


def train_category(name, device, deadline):
    train, valid, nitems = load_data(name)
    data = Histories(train)
    if not len(data):
        raise ValueError("Empty training dataset: " + name)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        data,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        collate_fn=pack_rows,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        generator=generator,
    )
    iterator = iter(loader)
    config = dict(itemnum=nitems, hidden=256, latent=64, dropout=0.5)
    model = MultVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    val_batches = validation_batches(train, valid, nitems)
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best = (-1.0, -1.0)
    best_epoch = 0
    update = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        completed = 0
        for _ in range(STEPS):
            try:
                packed = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed = next(iterator)
            ids, offsets, owners, input_weights = [x.to(device, non_blocking=True) for x in packed]
            anneal = min(0.2, update / 10000.0)
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, offsets, owners, anneal, input_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            update += 1
            completed += 1
            loss_sum += float(loss.detach())
            if time.monotonic() >= deadline:
                break
        expired = time.monotonic() >= deadline
        check = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired
        suffix = ""
        if check:
            metric = evaluate(model, val_batches, device)
            suffix = f" val_HR@10={metric[0]:.6f} val_NDCG@10={metric[1]:.6f}"
            if metric > best:
                best = metric
                best_epoch = epoch
                torch.save(
                    {"config": config, "state_dict": model.state_dict()}, checkpoint
                )
        print(
            f"{name} epoch={epoch} loss={loss_sum / max(1, completed):.6f}{suffix}",
            flush=True,
        )
        if expired or epoch - best_epoch >= PATIENCE:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    final_hr, _ = evaluate(model, val_batches, device)
    del iterator, loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_hr


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_module = None
_device = None


def _load(dataset_name):
    global _module, _device
    if _module is None:
        spec = importlib.util.spec_from_file_location(
            'trained_multvae_module', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    if dataset_name not in _models:
        saved = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                           map_location='cpu', weights_only=True)
        model = _module.MultVAE(**saved['config'])
        model.load_state_dict(saved['state_dict'])
        _models[dataset_name] = model.to(_device).eval()
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _load(ds)
    if not candidates:
        return []
    # Share chronological aggregation and normalization with training/validation.
    row = _module.transform_history([int(i) for i in history])
    if row[0].size and (row[0].min() < 1 or row[0].max() > model.itemnum):
        raise ValueError('History item outside the trained catalog')
    ids, offsets, _, input_weights = _module.pack_rows([row])
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_device)
    scores = model.predict(
        ids.to(_device), offsets.to(_device), candidate_tensor,
        input_weights.to(_device))
    return scores[0].cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as handle:
        handle.write(script)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(WORKING, exist_ok=True)
    write_predict_script()
    # Reserve a substantial margin under the nine-hour execution ceiling.
    end = time.monotonic() + 8 * 3600
    scores = []
    for index, name in enumerate(DATASETS):
        remaining_categories = len(DATASETS) - index
        category_deadline = time.monotonic() + max(
            1.0, (end - time.monotonic()) / remaining_categories
        )
        scores.append(train_category(name, device, category_deadline))
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
