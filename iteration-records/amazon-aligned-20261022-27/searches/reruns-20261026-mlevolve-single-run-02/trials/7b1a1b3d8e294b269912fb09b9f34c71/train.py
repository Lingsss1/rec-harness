from __future__ import annotations

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
from torch.utils.data import Dataset, DataLoader, RandomSampler

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
BATCH = int(os.environ.get("SASREC_BATCH", "512"))
STEPS = int(os.environ.get("SASREC_BATCHES", "200"))
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "100"))
DIM = int(os.environ.get("SASREC_HIDDEN", "128"))
NEGATIVES = 64
PATIENCE = 30


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
    itemnum = 0
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        itemnum = int(stats.get(name, {}).get("itemnum", 0))
    # Metadata fallback uses visible training and validation item IDs only.
    for split in (train, valid):
        for items in split.values():
            if items:
                itemnum = max(itemnum, max(items))
    return train, valid, itemnum


class PrefixDataset(Dataset):
    def __init__(self, train, maxlen):
        self.histories = [
            np.asarray(v, dtype=np.int64) for v in train.values() if len(v) >= 2
        ]
        self.maxlen = maxlen
        if not self.histories:
            raise ValueError("No training histories with at least two items")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = self.histories[index]
        # Emphasize the current training boundary while retaining earlier transitions.
        end = (
            len(history) - 1
            if np.random.random() < 0.5
            else np.random.randint(1, len(history))
        )
        prefix = history[max(0, end - self.maxlen) : end]
        seq = np.zeros(self.maxlen, dtype=np.int64)
        seq[-len(prefix) :] = prefix
        return seq, int(history[end])


def worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class HistoryModel(nn.Module):
    def __init__(self, itemnum, dim=128, maxlen=100, dropout=0.25):
        super().__init__()
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.input_emb = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.output_emb = nn.Embedding(itemnum + 1, dim, padding_idx=0)
        self.item_bias = nn.Embedding(itemnum + 1, 1, padding_idx=0)
        self.key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim * 2, dim)
        self.attention = nn.Linear(dim, 2, bias=False)
        self.decay = nn.Parameter(torch.tensor([-2.0, 0.5]))
        self.encoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.empty_context = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.input_emb.weight, std=0.05)
        nn.init.normal_(self.output_emb.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.input_emb.weight[0].zero_()
            self.output_emb.weight[0].zero_()

    def encode(self, seq):
        mask = seq.ne(0)
        x = self.input_emb(seq)
        count = mask.sum(1, keepdim=True).clamp_min(1)
        mean = (x * mask.unsqueeze(-1)).sum(1) / count
        last = x[:, -1]
        query = self.query(torch.cat([last, mean], dim=-1))
        logits = self.attention(torch.tanh(self.key(x) + query[:, None]))
        age = torch.arange(seq.shape[1] - 1, -1, -1, device=seq.device, dtype=x.dtype)
        logits = logits - torch.log1p(age)[None, :, None] * F.softplus(self.decay)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        weights = torch.softmax(logits, dim=1) * mask.unsqueeze(-1)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("blh,bld->bhd", weights, x).flatten(1)
        features = torch.cat([last, mean, pooled], dim=-1)
        context = self.norm(self.encoder(self.dropout(features)) + mean)
        context = torch.where(
            mask.any(1, keepdim=True), context, self.empty_context[None]
        )
        return F.normalize(context, dim=-1)

    def score(self, context, candidates):
        items = F.normalize(self.output_emb(candidates), dim=-1)
        scale = self.log_scale.clamp(math.log(1.0), math.log(40.0)).exp()
        return scale * (items * context[:, None]).sum(-1) + self.item_bias(
            candidates
        ).squeeze(-1)

    def forward(self, seq, candidates):
        return self.score(self.encode(seq), candidates)

    def predict(self, seq, candidates):
        return self.forward(seq, candidates)


def make_validation(train, valid, itemnum, maxlen):
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    sequences, candidates = [], []
    for user in users:
        if not valid[user]:
            continue
        hist = train[user][-maxlen:]
        seq = [0] * (maxlen - len(hist)) + hist
        excluded = set(train[user]) | set(valid[user])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        sequences.append(seq)
        candidates.append([valid[user][0]] + list(negatives))
    if not sequences:
        raise ValueError("No validation users")
    return torch.tensor(sequences), torch.tensor(candidates)


@torch.inference_mode()
def evaluate(model, validation, device):
    model.eval()
    seqs, candidates = validation
    hits = 0
    for start in range(0, len(seqs), 128):
        scores = model.predict(
            seqs[start : start + 128].to(device),
            candidates[start : start + 128].to(device),
        )
        # Stable descending ordering preserves candidate order for exact ties.
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        hits += order[:, :10].eq(0).any(1).sum().item()
    return hits / len(seqs)


def train_category(name, device, deadline):
    seed_everything(42)
    train, valid, itemnum = load_data(name)
    config = dict(itemnum=itemnum, dim=DIM, maxlen=MAXLEN, dropout=0.25)
    model = HistoryModel(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.98), weight_decay=1e-5
    )
    dataset = PrefixDataset(train, MAXLEN)
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=STEPS * BATCH, generator=generator
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=4,
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )
    validation = make_validation(train, valid, itemnum, MAXLEN)
    checkpoint = os.path.join(WORKING, name + "_model.pt")
    best, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, batches = 0.0, 0
        for seq, target in loader:
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            negative = torch.randint(
                1, itemnum + 1, (len(seq), NEGATIVES), device=device
            )
            # Mask known prefix items and the current positive from the loss.
            allowed = negative.ne(target[:, None])
            allowed &= ~negative[:, :, None].eq(seq[:, None, :]).any(-1)
            candidate = torch.cat([target[:, None], negative], dim=1)
            scores = model(seq, candidate)
            # One candidate denominator per example; duplicate draws remain intact.
            logits = scores.float().masked_fill(
                torch.cat([
                    torch.zeros_like(target[:, None], dtype=torch.bool), ~allowed
                ], dim=1),
                float('-inf'),
            )
            loss = F.cross_entropy(logits, torch.zeros_like(target))
            if epoch == 1 and batches == 0:
                # Check exclusions without changing sampling or consuming RNG state.
                expected_allowed = negative.ne(target[:, None])
                expected_allowed &= ~negative[:, :, None].eq(seq[:, None, :]).any(-1)
                assert torch.equal(allowed, expected_allowed)
                assert torch.equal(candidate[:, 0], target)
                assert torch.isfinite(logits[:, 0]).all().item()
                assert torch.isneginf(logits[:, 1:][~allowed]).all().item()
                assert torch.isfinite(logits[:, 1:][allowed]).all().item()
                assert torch.isfinite(loss).item(), 'Nonfinite sampled cross-entropy'
                # Exercise a positive-only denominator even if this batch has none.
                probe = scores[:1].detach().float().clone().requires_grad_(True)
                probe_logits = probe.masked_fill(
                    torch.arange(probe.shape[1], device=device)[None].ne(0),
                    float('-inf'),
                )
                probe_loss = F.cross_entropy(probe_logits, torch.zeros_like(target[:1]))
                probe_grad, = torch.autograd.grad(probe_loss, probe)
                assert torch.isfinite(probe_loss).item() and probe_loss.item() == 0.0
                assert torch.isfinite(probe_grad).all().item()
                assert probe_grad.eq(0).all().item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1 and batches == 0:
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(
                    torch.isfinite(gradient).all().item() for gradient in gradients
                ), 'Nonfinite first-step gradients'
                print(f'{name}: objective masking, positive-only loss and gradients checked',
                      flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
            if time.monotonic() >= deadline:
                break
        check = (
            epoch == 1
            or epoch % 5 == 0
            or time.monotonic() >= deadline
            or epoch == EPOCHS
        )
        hr = None
        if check:
            hr = evaluate(model, validation, device)
            if hr > best:
                best, best_epoch = hr, epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= 15:
                for group in optimizer.param_groups:
                    group["lr"] = max(0.00015, group["lr"] * 0.8)
        suffix = "" if hr is None else f" HR@10={hr:.5f} best={best:.5f}"
        print(
            f"{name} epoch={epoch} loss={loss_sum / max(batches, 1):.5f}{suffix}",
            flush=True,
        )
        if (check and epoch - best_epoch >= PATIENCE) or time.monotonic() >= deadline:
            break
    with open(os.path.join(WORKING, name + "_meta.json"), "w") as f:
        json.dump(config, f)
    # Verify the reported score through the exact selected checkpoint.
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    result = evaluate(model, validation, device)
    del loader, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_predictor():
    source = r"""from __future__ import annotations
import importlib.util
import json
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
            spec = importlib.util.spec_from_file_location(
                'history_training_architecture', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _CLASS = module.HistoryModel
        with open(os.path.join(_HERE, name + '_meta.json')) as f:
            config = json.load(f)
        model = _CLASS(**config).to(_DEVICE)
        state = torch.load(os.path.join(_HERE, name + '_model.pt'),
                           map_location=_DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _MODELS[name] = model
    return _MODELS[name]

@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    model = _load(dataset_name if dataset_name is not None else _NAMES[0])
    history = list(history)[-model.maxlen:]
    seq = [0] * (model.maxlen - len(history)) + history
    seq = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    items = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return model.predict(seq, items)[0].cpu().tolist()
"""
    with open(os.path.join(WORKING, "predict.py"), "w") as f:
        f.write(source)


def main():
    os.makedirs(WORKING, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Leave time for artifact writing and independent final inference.
    total_deadline = time.monotonic() + 8 * 60 * 60
    results = []
    for index, name in enumerate(DATASETS):
        remaining = len(DATASETS) - index
        category_deadline = (
            time.monotonic() + max(1.0, total_deadline - time.monotonic()) / remaining
        )
        results.append(train_category(name, device, category_deadline))
    write_predictor()
    # Compare the generated inference path with each exact selected checkpoint.
    # These are identical checkpoint loads, not additional fitted predictors.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'selected_checkpoint_predictor', os.path.join(WORKING, 'predict.py')
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    for name in DATASETS:
        train, valid, itemnum = load_data(name)
        seqs, candidates = make_validation(train, valid, itemnum, MAXLEN)
        with open(os.path.join(WORKING, name + '_meta.json')) as f:
            config = json.load(f)
        reference = HistoryModel(**config).to(device)
        reference.load_state_dict(torch.load(
            os.path.join(WORKING, name + '_model.pt'),
            map_location=device, weights_only=True,
        ))
        reference.eval()
        with torch.inference_mode():
            expected = reference.predict(
                seqs[:1].to(device), candidates[:1].to(device)
            )[0].cpu()
        actual = torch.tensor(predictor.predict(
            0, seqs[0][seqs[0].ne(0)].tolist(), candidates[0].tolist(),
            dataset_name=name,
        ), dtype=expected.dtype)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        del reference
        predictor._MODELS.clear()
        print(f'{name}: selected-checkpoint predictor parity checked', flush=True)
    macro = sum(results) / len(results)
    with open(os.path.join(WORKING, "validation_metrics.json"), "w") as f:
        json.dump(
            {
                "metric": "HR@10",
                "objective": "sampled_listwise_cross_entropy",
                "training_negatives": NEGATIVES,
                "selected_checkpoint_prediction_parity": "passed for all categories",
                "evaluation_seed": 42,
=======
    for name, result in zip(DATASETS, results):
        print(f'{name} measured validation HR@10={result:.5f}', flush=True)
                "evaluation_seed": 42,
=======
                "first_step_checks": "passed for all categories",
                "selected_checkpoint_prediction_parity": "passed for all categories",
                "evaluation_seed": 42,
=======
                "selected_checkpoint_prediction_parity": "passed for all categories",
                "evaluation_seed": 42,
=======
    for name, result in zip(DATASETS, results):
        print(f'{name} measured validation HR@10={result:.5f}', flush=True)
                "evaluation_seed": 42,
                "users_per_category": 200,
                "unique_negatives": 99,
                "per_category": dict(zip(DATASETS, results)),
                "macro_average": sum(results) / len(results),
            },
            f,
            indent=2,
        )
    for name, result in zip(DATASETS, results):
        print(f'{name} measured validation HR@10={result:.5f}', flush=True)
    score = macro
    print(f"Final Validation Score: {score}", flush=True)


if __name__ == "__main__":
    main()