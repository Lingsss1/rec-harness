from __future__ import annotations

import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "working"
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
DIM = int(os.environ.get("SASREC_HIDDEN", "64"))
BATCH = int(os.environ.get("SASREC_BATCH", "128"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
NEGATIVES = 32


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path):
    result = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2:
                u, i = map(int, fields)
                result[u].append(i)
    return dict(result)


def load_data(name):
    train = read_split(Path(DATA_DIR) / f"{name}_train.txt")
    valid = read_split(Path(DATA_DIR) / f"{name}_valid.txt")
    stats_path = Path(DATA_DIR) / "dataset_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as handle:
            stats = json.load(handle)
    if name in stats:
        itemnum = int(stats[name]["itemnum"])
    else:
        itemnum = max(max(items) for items in train.values() if items)
    return train, valid, itemnum


class AttentiveItemSimilarity(nn.Module):
    """One candidate-conditioned collaborative filtering network."""

    def __init__(self, itemnum, dim=64, maxlen=128, dropout=0.2):
        super().__init__()
        self.maxlen = maxlen
        self.dim = dim
        self.source = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.target = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.attention_source = nn.Linear(dim, dim, bias=False)
        self.attention_target = nn.Linear(dim, dim, bias=False)
        self.attention_vector = nn.Parameter(torch.zeros(dim))
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.beta_logit = nn.Parameter(torch.tensor(0.4))
        self.recency_raw = nn.Parameter(torch.tensor(-2.0))
        self.scale_raw = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.source.weight, std=0.08)
        nn.init.normal_(self.target.weight, std=0.08)
        nn.init.normal_(self.attention_vector, std=0.08)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.source.weight[0].zero_()
            self.target.weight[0].zero_()

    def predict(self, seq, candidates):
        # Candidate-specific nonlinear attention, computed in bounded chunks.
        source = self.dropout(self.source(seq))
        source_keys = self.attention_source(source)
        mask = seq.ne(0)
        length = mask.sum(-1).clamp_min(1).to(source.dtype)
        ages = torch.arange(
            seq.shape[1] - 1, -1, -1, device=seq.device, dtype=source.dtype
        )
        recency = torch.nn.functional.softplus(self.recency_raw)
        beta = torch.sigmoid(self.beta_logit)
        scale = torch.nn.functional.softplus(self.scale_raw)
        outputs = []
        for chunk in candidates.split(16, dim=1):
            targets = self.target(chunk)
            queries = self.attention_target(targets)
            attention = torch.tanh(source_keys[:, None, :, :] + queries[:, :, None, :])
            attention = (attention * self.attention_vector).sum(-1)
            attention = attention - recency * torch.log1p(ages)[None, None, :]
            attention = attention.masked_fill(~mask[:, None, :], -1e4)
            weights = torch.softmax(attention, dim=-1)
            weights = weights * mask[:, None, :]
            # Smoothed normalization retains useful evidence from longer histories.
            pooled = torch.einsum("bkl,bld->bkd", weights, source)
            pooled = pooled * length[:, None, None].pow(1.0 - beta)
            score = scale * (pooled * targets).sum(-1)
            score = score + self.item_bias(chunk).squeeze(-1)
            outputs.append(score)
        return torch.cat(outputs, dim=1)

    def forward(self, seq, candidates):
        return self.predict(seq, candidates)


class PrefixDataset(Dataset):
    def __init__(self, train, itemnum, maxlen):
        self.histories = [
            np.asarray(h, dtype=np.int64) for h in train.values() if len(h) >= 2
        ]
        self.seen = [set(h.tolist()) for h in self.histories]
        self.itemnum = itemnum
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No training histories with next-item supervision")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        # Include final training transitions frequently, alongside earlier prefixes.
        end = (
            len(history) - 1
            if np.random.random() < 0.5
            else np.random.randint(1, len(history))
        )
        prefix = history[max(0, end - self.maxlen) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-len(prefix) :] = prefix
        negatives = np.random.randint(1, self.itemnum + 1, size=NEGATIVES)
        seen = self.seen[index]
        if len(seen) >= self.itemnum:
            raise ValueError("Training user has no available negative items")
        for j in range(NEGATIVES):
            while int(negatives[j]) in seen:
                negatives[j] = np.random.randint(1, self.itemnum + 1)
        candidates = np.concatenate(([history[end]], negatives)).astype(np.int64)
        return seq, candidates


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def validation_tensors(train, valid, itemnum, maxlen):
    # Preserve input order and Python sampling semantics of the supplied baseline.
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
            raise ValueError("Cannot form the required 99 unique validation negatives")
        negative_pool = set()
        while len(negative_pool) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negative_pool.add(item)
        hist = train[user][-maxlen:]
        seq = [0] * maxlen
        if hist:
            seq[-len(hist) :] = hist
        sequences.append(seq)
        candidates.append([target] + list(negative_pool))
    if not sequences:
        raise ValueError("No validation users")
    return torch.tensor(sequences), torch.tensor(candidates)


@torch.inference_mode()
def evaluate(model, tensors, device):
    model.eval()
    hits = 0
    seqs, candidates = tensors
    for start in range(0, len(seqs), 64):
        scores = model.predict(
            seqs[start : start + 64].to(device),
            candidates[start : start + 64].to(device),
        )
        # Stable descending rank: target occupies candidate index zero.
        ranks = (scores[:, 1:] > scores[:, :1]).sum(dim=1)
        hits += int((ranks < 10).sum().item())
    return hits / len(seqs)


def train_category(name, device, seconds_available):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    dataset = PrefixDataset(train, itemnum, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=BATCH * STEPS, generator=generator
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )
    config = dict(itemnum=itemnum, dim=DIM, maxlen=MAXLEN, dropout=0.2)
    model = AttentiveItemSimilarity(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, betas=(0.9, 0.98), weight_decay=1e-4
    )
    val_tensors = validation_tensors(train, valid, itemnum, MAXLEN)
    checkpoint = WORK / f"{name}_model.pt"
    best, best_epoch = -1.0, 0
    started = time.monotonic()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seq, candidates in loader:
            seq = seq.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(seq, candidates)
                positive_loss = torch.nn.functional.softplus(-logits[:, 0]).mean()
                negative_loss = torch.nn.functional.softplus(logits[:, 1:]).mean()
                loss = positive_loss + negative_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        elapsed = time.monotonic() - started
        must_stop = elapsed >= seconds_available
        if epoch % 5 == 0 or epoch == EPOCHS or must_stop:
            hr = evaluate(model, val_tensors, device)
            if hr > best:
                best, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            print(
                f"{name} epoch={epoch} loss={total_loss / STEPS:.5f} "
                f"validation_HR@10={hr:.5f}",
                flush=True,
            )
            if epoch - best_epoch >= 40 or must_stop:
                break
        else:
            print(f"{name} epoch={epoch} loss={total_loss / STEPS:.5f}", flush=True)
    with open(WORK / f"{name}_meta.pkl", "wb") as handle:
        pickle.dump(config, handle)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    score = evaluate(model, val_tensors, device)
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def write_predict_script():
    script = r"""from __future__ import annotations
import importlib.util
import os
import pickle
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
            'Electronics', 'CDs_and_Vinyl']
_models = {}
_device = None
_model_class = None


def _load(dataset_name):
    global _device, _model_class
    if dataset_name in _models:
        return _models[dataset_name]
    if dataset_name not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _model_class is None:
        spec = importlib.util.spec_from_file_location(
            'attentive_item_training', str(HERE.parent / 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.AttentiveItemSimilarity
    with open(HERE / f'{dataset_name}_meta.pkl', 'rb') as handle:
        config = pickle.load(handle)
    model = _model_class(**config).to(_device)
    state = torch.load(HERE / f'{dataset_name}_model.pt',
                       map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[dataset_name] = model
    return model


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = DATASETS[0]
    model = _load(dataset_name)
    if not candidates:
        return []
    hist = list(history[-model.maxlen:])
    seq = [0] * model.maxlen
    if hist:
        seq[-len(hist):] = hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_device)
    return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
"""
    (WORK / "predict.py").write_text(script)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    scores = []
    # Reserve headroom below the nine-hour host limit and retain all categories.
    for index, name in enumerate(DATASETS):
        remaining = max(60.0, 8 * 3600 - (time.monotonic() - started))
        allocation = remaining / (len(DATASETS) - index)
        scores.append(train_category(name, device, allocation))
    write_predict_script()
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
