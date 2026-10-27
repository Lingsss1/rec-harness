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
    # Compute ages in chronological order, retaining only each item's latest use.
    items = np.asarray(list(items), dtype=np.int64)
    if not items.size:
        return items, np.empty(0, dtype=np.float32)
    ids, latest = np.unique(items[::-1], return_index=True)
    # Reverse indices are ages at the latest occurrence, aligned with sorted IDs.
    # Prefix inputs recompute these encoder weights relative to their own end.
    weights = np.exp2(-latest.astype(np.float64) / 10.0)
    weights /= np.linalg.norm(weights)
    return ids, weights.astype(np.float32)


class Histories(Dataset):
    def __init__(self, train):
        # Keep chronological training sequences; never include validation items.
        self.rows = [list(items) for items in train.values() if len(items) >= 2]

    def __len__(self):
        return len(self.rows)

    def example_at(self, index, t):
        items = self.rows[index]
        if not 1 <= t < len(items):
            raise ValueError("Prefix boundary outside the training sequence")
        # Transform only the prefix, so suffix occurrences cannot affect ages.
        # A next-item target may already occur in its prefix.
        return transform_history(items[:t]), int(items[t])

    def __getitem__(self, index):
        # Every complete shuffled traversal visits each eligible user once.
        # Persistent workers resample positions using their seeded RNG streams.
        t = random.randrange(1, len(self.rows[index]))
        return self.example_at(index, t)


def pack_rows(rows):
    lengths = np.asarray([len(row[0]) for row in rows], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    ids = np.concatenate([row[0] for row in rows]) if offsets[-1] else np.empty(0, dtype=np.int64)
    weights = np.concatenate([row[1] for row in rows]) if offsets[-1] else np.empty(0, dtype=np.float32)
    owners = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    # Latest-occurrence weights remain aligned with unique IDs and owners.
    return (
        torch.from_numpy(ids),
        torch.from_numpy(offsets),
        torch.from_numpy(owners),
        torch.from_numpy(weights),
    )


def pack_training_rows(rows):
    prefixes, targets = zip(*rows)
    return pack_rows(prefixes), torch.tensor(targets, dtype=torch.long)


def seed_training_worker(worker_id):
    # DataLoader derives each worker's Torch seed from its seed-42 generator.
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
        # Production paths supply unit-L2 latest-occurrence recency weights.
        # Retain the uniform fallback; apply dropout without renormalizing.
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

    def forward(self, ids, offsets, owners, anneal, input_weights=None, targets=None):
        if targets is None:
            raise ValueError("Training requires sampled next-item targets")
        if targets.ndim != 1 or targets.numel() != offsets.numel() - 1:
            raise ValueError("Targets must align with the packed prefix batch")
        if (targets <= 0).any():
            raise ValueError("Padding is not a next-item target")
        h, mu, logvar = self.representation(ids, offsets, input_weights)
        logits = self.decoder(h).float()
        # Exclude only padding. Previously observed items remain valid targets.
        logits[:, 0] = -torch.inf
        nll = F.cross_entropy(logits, targets)
        # Keep the existing KL and update-based annealing without rescaling CE.
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
        collate_fn=pack_training_rows,
        worker_init_fn=seed_training_worker,
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
                packed, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                packed, targets = next(iterator)
            ids, offsets, owners, input_weights = [x.to(device, non_blocking=True) for x in packed]
            targets = targets.to(device, non_blocking=True)
            anneal = min(0.2, update / 10000.0)
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, offsets, owners, anneal, input_weights, targets=targets)
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
    check_predictor_agreement(name, model, train, valid, val_batches, device)
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
        # Load the chronological next-item MultVAE with its saved configuration.
        # Inference retains the posterior mean and candidate decoder scoring.
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
    # Share the transform used for training prefixes and full validation histories.
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


def check_history_transform():
    # Deterministic fixtures use no Python, NumPy, or Torch random draws.
    ids, weights = transform_history([3, 1, 3])
    np.testing.assert_array_equal(ids, [1, 3])
    expected = np.exp2(-np.asarray([1.0, 0.0]) / 10.0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=0)
    assert weights.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(weights), 1.0, rtol=1e-6)

    # The oldest item is exactly ten interactions older than the newest item.
    # Repeated middle items receive their latest age, without frequency sums.
    history = [1] + [2] * 9 + [3]
    ids, weights = transform_history(history)
    raw = np.exp2(-np.asarray([10.0, 1.0, 0.0]) / 10.0)
    assert raw[0] / raw[2] == 0.5
    np.testing.assert_allclose(weights, raw / np.linalg.norm(raw), rtol=1e-6)
    np.testing.assert_allclose(weights[0] / weights[2], 0.5, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(weights), 1.0, rtol=1e-6)

    # Packing retains one entry per observed unique item.
    packed_ids, offsets, owners, input_weights = pack_rows([(ids, weights)])
    np.testing.assert_array_equal(packed_ids.numpy(), [1, 2, 3])
    np.testing.assert_array_equal(offsets.numpy(), [0, 3])
    np.testing.assert_array_equal(owners.numpy(), [0, 0, 0])
    assert len((packed_ids, offsets, owners, input_weights)) == 4
    np.testing.assert_array_equal(input_weights.numpy(), weights)

    ids, weights = transform_history([])
    assert ids.dtype == np.int64 and ids.size == 0
    assert weights.dtype == np.float32 and weights.size == 0
    packed_ids, offsets, owners, input_weights = pack_rows([(ids, weights)])
    np.testing.assert_array_equal(offsets.numpy(), [0, 0])
    assert packed_ids.numel() == owners.numel() == input_weights.numel() == 0

    # Explicit positions exercise both sampling boundaries without RNG draws.
    source = {0: [], 1: [4], 2: [3, 1, 3, 4], 3: [2, 2]}
    data = Histories(source)
    assert len(data) == 2
    assert data.rows == [[3, 1, 3, 4], [2, 2]]
    for index, items in enumerate(data.rows):
        for t in range(1, len(items)):
            (row_ids, row_weights), target = data.example_at(index, t)
            assert target == items[t]
            np.testing.assert_array_equal(row_ids, sorted(set(items[:t])))
            # Compute ages independently from the prefix, including repeats.
            latest = {item: position for position, item in enumerate(items[:t])}
            raw = np.asarray([
                2.0 ** (-(t - 1 - latest[int(item)]) / 10.0)
                for item in row_ids
            ])
            np.testing.assert_allclose(
                row_weights, raw / np.linalg.norm(raw), rtol=1e-6
            )
            assert 4 not in row_ids
    for t in (0, len(data.rows[0])):
        try:
            data.example_at(0, t)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid prefix boundary accepted")

    # Both aligned targets already occur in their respective prefixes.
    examples = [data.example_at(0, 2), data.example_at(1, 1)]
    packed, targets = pack_training_rows(examples)
    ids, offsets, owners, input_weights = packed
    assert len(packed) == 4 and targets.dtype == torch.long
    torch.testing.assert_close(ids, torch.tensor([1, 3, 2]))
    torch.testing.assert_close(offsets, torch.tensor([0, 2, 3]))
    torch.testing.assert_close(owners, torch.tensor([0, 0, 1]))
    torch.testing.assert_close(targets, torch.tensor([3, 2]))
    # The future occurrence of item 3 cannot make it newer than prefix item 1.
    raw = np.exp2(-np.asarray([0.0, 1.0]) / 10.0)
    np.testing.assert_allclose(
        input_weights[:2].numpy(), raw / np.linalg.norm(raw), rtol=1e-6
    )

    # Fixed logits and posterior exercise production CE and nonzero KL without
    # initialization, dropout, sampling, or changes to any training RNG state.
    class NextItemFixture(MultVAE):
        def __init__(self, logits):
            nn.Module.__init__(self)
            self.register_buffer("fixed_logits", logits)
            self.decoder = nn.Identity()
            self.seen_input_weights = None

        def representation(self, ids, offsets, input_weights=None):
            self.seen_input_weights = input_weights
            shape = (offsets.numel() - 1, 2)
            mu = self.fixed_logits.new_full(shape, 0.5)
            logvar = self.fixed_logits.new_full(shape, math.log(2.0))
            return self.fixed_logits.clone(), mu, logvar

    logits = torch.tensor([
        [10000.0, 0.1, -0.2, 0.7, 0.3],
        [10000.0, -0.4, 0.6, 0.2, -0.1],
    ], requires_grad=True)
    fixture = NextItemFixture(logits)
    original_weights = input_weights.clone()
    # Removing column zero independently checks its exclusion from softmax.
    expected_ce = F.cross_entropy(logits[:, 1:], targets - 1)
    expected_kl = logits.new_tensor(1.25 - math.log(2.0))
    for anneal in (0.0, 0.07, 0.2):
        actual = fixture(ids, offsets, owners, anneal, input_weights, targets)
        torch.testing.assert_close(actual, expected_ce + anneal * expected_kl)
        assert actual.ndim == 0 and torch.isfinite(actual)
        assert fixture.seen_input_weights is input_weights
    torch.testing.assert_close(input_weights, original_weights, rtol=0, atol=0)

    loss = fixture(ids, offsets, owners, 0.2, input_weights, targets)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient[:, 0], torch.zeros(2), rtol=0, atol=0)
    assert (gradient[torch.arange(2), targets] < 0).all()
    fallback = fixture(ids, offsets, owners, 0.2, targets=targets)
    torch.testing.assert_close(fallback, expected_ce + 0.2 * expected_kl)
    for bad_targets in (None, torch.tensor([0, 2]), torch.tensor([3])):
        try:
            fixture(ids, offsets, owners, 0.2, input_weights, bad_targets)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing, padding, or misaligned target accepted")
    # Full-history validation and generated inference remain covered by
    # check_predictor_agreement after each selected checkpoint is restored.


@torch.inference_mode()
def check_predictor_agreement(name, model, train, valid, batches, device):
    import sys

    # Reproduce validation's first selected user with a private RNG.
    users = [u for u in valid if u in train and valid[u]]
    if len(users) > 200:
        users = random.Random(42).sample(users, 200)
    user = users[0]
    _, candidates = batches[0]
    model.eval()
    packed = pack_rows([transform_history(train[user])])
    ids, offsets, _, input_weights = [x.to(device) for x in packed]
    expected = model.predict(
        ids, offsets, candidates[:1].to(device), input_weights
    )[0]

    spec = importlib.util.spec_from_file_location(
        "recency_predictor_check", os.path.join(WORKING, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    # Reuse the loaded checkpoint without constructing a model or consuming RNG.
    predictor._module = sys.modules[__name__]
    predictor._device = device
    predictor._models[name] = model
    actual = predictor.predict(
        user, train[user], candidates[0].tolist(), dataset_name=name
    )
    np.testing.assert_allclose(
        actual, expected.cpu().numpy(), rtol=1e-5, atol=1e-5
    )


def main():
    check_history_transform()
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
    for name, category_score in zip(DATASETS, scores):
        print(f"{name} selected_validation_HR@10={category_score:.6f}")
    print(f"Final Validation Score: {score}")
    # Report measured validation results; neither reference implies improvement.
    print(
        f"Macro validation HR@10={score:.6f} "
        f"delta_vs_latest_0.5125={score - 0.5125:+.6f} "
        f"delta_vs_recorded_best_0.5363={score - 0.5363:+.6f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        os.makedirs(WORKING, exist_ok=True)
        with open(os.path.join(WORKING, "training_error.txt"), "w") as handle:
            traceback.print_exc(file=handle)
        raise
