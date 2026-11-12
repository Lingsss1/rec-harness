from __future__ import annotations

import json
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
DIM = int(os.environ.get("GRAPH_DIM", "64"))
LAYERS = int(os.environ.get("GRAPH_LAYERS", "3"))
BATCH = int(os.environ.get("GRAPH_BATCH", "2048"))
STEPS = int(os.environ.get("GRAPH_STEPS", "20"))
NEGATIVES = int(os.environ.get("GRAPH_NEGATIVES", "8"))
LR = float(os.environ.get("GRAPH_LR", "0.003"))
REG = float(os.environ.get("GRAPH_REG", "0.0001"))
PATIENCE = int(os.environ.get("GRAPH_PATIENCE", "35"))
TIME_LIMIT = float(os.environ.get("GRAPH_SECONDS_PER_CATEGORY", "1800"))


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
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_data(name):
    train = read_split(os.path.join(DATA_DIR, name + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, name + "_valid.txt"))
    itemnum = max(i for history in train.values() for i in history)
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if name in stats:
            itemnum = int(stats[name]["itemnum"])
    return train, valid, itemnum


class GraphRecommender(nn.Module):
    """One jointly trained LightGCN, including its learned item bias."""

    def __init__(self, nusers, itemnum, dim, layers, adjacency):
        super().__init__()
        self.nusers = nusers
        self.itemnum = itemnum
        self.layers = layers
        self.embedding = nn.Embedding(nusers + itemnum + 1, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.item_bias = nn.Parameter(torch.zeros(itemnum + 1))
        self.register_buffer("adjacency", adjacency)
        self._cached = None

    def propagate(self):
        x = self.embedding.weight
        result = x
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adjacency, x)
            result = result + x
        result = result / (self.layers + 1)
        return result[: self.nusers], result[self.nusers :]

    def forward(self, users, positives, negatives):
        user_vectors, item_vectors = self.propagate()
        u = user_vectors[users]
        positive_scores = (u * item_vectors[positives]).sum(-1)
        positive_scores = positive_scores + self.item_bias[positives]
        negative_scores = (u[:, None] * item_vectors[negatives]).sum(-1)
        negative_scores = negative_scores + self.item_bias[negatives]
        logits = torch.cat([positive_scores[:, None], negative_scores], dim=1)
        ranking = F.cross_entropy(
            logits.float(),
            torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device),
        )
        # Regularize the directly optimized embeddings, rather than graph outputs.
        raw_u = self.embedding(users)
        raw_p = self.embedding(self.nusers + positives)
        raw_n = self.embedding(self.nusers + negatives)
        penalty = (
            raw_u.square().sum(-1).mean()
            + raw_p.square().sum(-1).mean()
            + raw_n.square().sum(-1).mean()
        )
        bias_penalty = self.item_bias.square().mean()
        return ranking + REG * penalty + 1e-5 * bias_penalty

    @torch.no_grad()
    def prepare_inference(self):
        self.eval()
        self._cached = self.propagate()

    @torch.no_grad()
    def predict_indices(self, users, candidates):
        if self._cached is None:
            self.prepare_inference()
        user_vectors, item_vectors = self._cached
        return (user_vectors[users, None] * item_vectors[candidates]).sum(
            -1
        ) + self.item_bias[candidates]

    @torch.no_grad()
    def predict_history(self, history, candidates):
        # Cold-user inference still uses learned graph embeddings.
        if self._cached is None:
            self.prepare_inference()
        _, items = self._cached
        if len(history) == 0:
            raise ValueError("An unseen user must supply a nonempty history")
        query = items[history].mean(0)
        return items[candidates].matmul(query) + self.item_bias[candidates]


def build_graph(train, user_ids, itemnum):
    mapping = {u: j for j, u in enumerate(user_ids)}
    # Repeated purchases do not create duplicate graph edges.
    pairs = np.asarray(
        [(mapping[u], i) for u in user_ids for i in sorted(set(train[u]))],
        dtype=np.int64,
    )
    nusers = len(user_ids)
    users = torch.from_numpy(pairs[:, 0].copy())
    items = torch.from_numpy(pairs[:, 1].copy()) + nusers
    rows = torch.cat((users, items))
    cols = torch.cat((items, users))
    nodes = nusers + itemnum + 1
    degree = torch.bincount(rows, minlength=nodes).float().clamp_min(1)
    values = degree[rows].rsqrt() * degree[cols].rsqrt()
    graph = torch.sparse_coo_tensor(
        torch.stack((rows, cols)), values, (nodes, nodes)
    ).coalesce()
    # CSR reduces sparse multiplication overhead on supported PyTorch builds.
    graph = graph.to_sparse_csr()
    return mapping, pairs, graph


def validation_cases(train, valid, mapping, itemnum):
    # Matches the supplied baseline protocol without perturbing training RNG.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    selected_users, candidates = [], []
    for u in users:
        if not valid[u]:
            continue
        excluded = set(train[u]) | set(valid[u])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            item = rng.randint(1, itemnum)
            if item not in excluded:
                negatives.add(item)
        selected_users.append(mapping[u])
        candidates.append([valid[u][0]] + list(negatives))
    if not candidates:
        raise ValueError("No eligible validation users")
    return selected_users, candidates


@torch.no_grad()
def evaluate(model, val_users, val_candidates):
    model.prepare_inference()
    scores = model.predict_indices(val_users, val_candidates)
    # Stable candidate ordering: the positive occupies position zero.
    ranks = (scores[:, 1:] > scores[:, :1]).sum(1)
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
    model._cached = None
    return hr, ndcg


def sample_negatives(users, positives, itemnum, known_keys):
    width = itemnum + 1
    negatives = torch.randint(1, width, (len(users), NEGATIVES), device=users.device)
    # Exclude every observed training item, using a sorted sparse membership table.
    for _ in range(20):
        keys = users[:, None] * width + negatives
        indices = torch.searchsorted(known_keys, keys)
        occupied = known_keys[indices.clamp_max(len(known_keys) - 1)] == keys
        if not occupied.any():
            return negatives
        negatives[occupied] = torch.randint(
            1, width, (int(occupied.sum()),), device=users.device
        )
    # Extremely dense users need an exact fallback rather than false negatives.
    for row in torch.where(occupied.any(1))[0].tolist():
        lower = int(users[row]) * width
        start = int(torch.searchsorted(known_keys, lower))
        end = int(torch.searchsorted(known_keys, lower + width))
        allowed = torch.ones(width, dtype=torch.bool, device=users.device)
        allowed[0] = False
        allowed[known_keys[start:end] - lower] = False
        pool = torch.where(allowed)[0]
        if not len(pool):
            raise ValueError("A user has no eligible training negatives")
        bad = occupied[row]
        negatives[row, bad] = pool[
            torch.randint(len(pool), (int(bad.sum()),), device=users.device)
        ]
    return negatives


def train_category(name, device):
    seed_all(42)
    train, valid, itemnum = load_data(name)
    user_ids = sorted(u for u in train if train[u])
    mapping, pairs, graph = build_graph(train, user_ids, itemnum)
    model = GraphRecommender(len(user_ids), itemnum, DIM, LAYERS, graph.to(device)).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    keys = pairs[:, 0] * (itemnum + 1) + pairs[:, 1]
    known_keys = torch.as_tensor(np.sort(keys), device=device)
    vu, vc = validation_cases(train, valid, mapping, itemnum)
    vu = torch.tensor(vu, dtype=torch.long, device=device)
    vc = torch.tensor(vc, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(pairs.copy())),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )
    iterator = iter(loader)
    path = os.path.join(WORK, name + "_model.pt")
    best_hr, best_ndcg, best_epoch = -1.0, -1.0, 0
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model._cached = None
        total_loss = 0.0
        for _ in range(STEPS):
            try:
                batch = next(iterator)[0]
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)[0]
            batch = batch.to(device, non_blocking=True)
            users, positives = batch[:, 0], batch[:, 1]
            negatives = sample_negatives(users, positives, itemnum, known_keys)
            optimizer.zero_grad(set_to_none=True)
            loss = model(users, positives, negatives)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite graph ranking loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        should_evaluate = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS
        timed_out = time.monotonic() - started >= TIME_LIMIT
        should_evaluate = should_evaluate or timed_out
        if should_evaluate:
            hr, ndcg = evaluate(model, vu, vc)
            if (hr, ndcg) > (best_hr, best_ndcg):
                best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
                torch.save(model.state_dict(), path)
            print(
                f"{name} epoch={epoch} loss={total_loss / STEPS:.5f} "
                f"validation_HR@10={hr:.5f} NDCG@10={ndcg:.5f}",
                flush=True,
            )
        else:
            print(f"{name} epoch={epoch} loss={total_loss / STEPS:.5f}", flush=True)
        if timed_out or epoch - best_epoch >= PATIENCE:
            break
    metadata = {
        "user_ids": user_ids,
        "itemnum": itemnum,
        "dim": DIM,
        "layers": LAYERS,
        "best_epoch": best_epoch,
    }
    with open(os.path.join(WORK, name + "_meta.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    # Report the actual selected checkpoint's metric.
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    selected_hr, _ = evaluate(model, vu, vc)
    del iterator, loader, model, optimizer, graph, known_keys
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return selected_hr


def write_predictor():
    script = r"""from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_MAPS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(dataset):
    global _CLASS
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'trained_graph_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.GraphRecommender
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                       map_location=_DEVICE, weights_only=True)
    model = _CLASS(len(meta['user_ids']), meta['itemnum'], meta['dim'],
                   meta['layers'], state['adjacency']).to(_DEVICE)
    model.load_state_dict(state)
    model.prepare_inference()
    _MAPS[dataset] = {int(u): j for j, u in enumerate(meta['user_ids'])}
    _MODELS[dataset] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    model = _load(dataset)
    if len(candidates) == 0:
        return []
    cand = torch.tensor(candidates, dtype=torch.long, device=_DEVICE)
    if bool(((cand < 1) | (cand > model.itemnum)).any()):
        raise ValueError('Candidate outside dataset item vocabulary')
    with torch.inference_mode():
        index = _MAPS[dataset].get(int(user_id))
        if index is not None:
            users = torch.tensor([index], dtype=torch.long, device=_DEVICE)
            scores = model.predict_indices(users, cand[None])[0]
        else:
            valid_history = [int(i) for i in history
                             if 1 <= int(i) <= model.itemnum]
            hist = torch.tensor(valid_history, dtype=torch.long, device=_DEVICE)
            scores = model.predict_history(hist, cand)
    return scores.cpu().tolist()
"""
    with open(os.path.join(WORK, "predict.py"), "w") as f:
        f.write(script)


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all(42)
    os.makedirs(WORK, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = [train_category(name, device) for name in DATASETS]
    write_predictor()
    # Test files are unavailable: the host invokes this predictor independently.
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()