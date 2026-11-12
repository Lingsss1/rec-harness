from __future__ import annotations

import os
import json
import math
import time
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DATA_DIR = os.environ["GAGC_DATA_DIR"]
DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
DIM = int(os.environ.get("GRAPH_DIM", "64"))
STEPS = int(os.environ.get("GRAPH_STEPS", "8"))
BATCH = int(os.environ.get("GRAPH_BATCH", "8192"))


def read_split(path):
    result = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                result[int(parts[0])].append(int(parts[1]))
    return dict(result)


def load_data(ds):
    train = read_split(os.path.join(DATA_DIR, ds + "_train.txt"))
    valid = read_split(os.path.join(DATA_DIR, ds + "_valid.txt"))
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    itemnum = (
        int(stats[ds]["itemnum"])
        if ds in stats
        else max(max(h) for h in train.values() if h)
    )
    return train, valid, itemnum


class PredictionModel(nn.Module):
    """Materialized learned graph embeddings, with history-aware scoring."""

    def __init__(self, user_ids, user_vectors, item_vectors):
        super().__init__()
        self.register_buffer("user_ids", user_ids)
        self.register_buffer("user_vectors", user_vectors)
        self.register_buffer("item_vectors", item_vectors)

    def forward(self, user_ids, histories, history_mask, candidates):
        locations = torch.searchsorted(self.user_ids, user_ids)
        safe_locations = locations.clamp(max=len(self.user_ids) - 1)
        known = (locations < len(self.user_ids)) & (
            self.user_ids[safe_locations] == user_ids
        )
        base = self.user_vectors[safe_locations] * known[:, None]
        mask = history_mask.to(self.item_vectors.dtype)
        # Count valid interactions after each position, independent of padding.
        age = mask.flip(1).cumsum(1).flip(1) - mask
        weights = torch.exp2(-age / 10.0) * mask
        weights = weights / weights.sum(1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        history_vector = (self.item_vectors[histories] * weights[..., None]).sum(1)
        representation = base + 0.5 * history_vector
        return (self.item_vectors[candidates] * representation[:, None]).sum(-1)


def sparse_matrix(rows, cols, values, shape, device):
    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)


class LightGCN(nn.Module):
    def __init__(self, users, itemnum, graph, history_matrix, dim):
        super().__init__()
        self.users = users
        self.itemnum = itemnum
        self.embedding = nn.Embedding(users + itemnum + 1, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[users].zero_()
        self.register_buffer("graph", graph)
        self.register_buffer("history_matrix", history_matrix)

    def encode(self):
        x = self.embedding.weight
        total = x
        for _ in range(2):
            x = torch.sparse.mm(self.graph, x)
            total = total + x
        total = total / 3.0
        return total[: self.users], total[self.users :]

    def training_vectors(self):
        base, items = self.encode()
        profiles = torch.sparse.mm(self.history_matrix, items)
        return base + 0.5 * profiles, items

    @torch.no_grad()
    def export(self, original_user_ids):
        base, items = self.encode()
        return PredictionModel(
            torch.tensor(original_user_ids, dtype=torch.long, device=base.device),
            base.detach().clone(),
            items.detach().clone(),
        )


def build_graph(train, ids, itemnum, device):
    unique_users, unique_items = [], []
    history_users, history_items, history_weights = [], [], []
    histories, seen = [], []
    for index, uid in enumerate(ids):
        hist = np.asarray(train[uid], dtype=np.int64)
        histories.append(hist)
        itemset = set(hist.tolist())
        seen.append(itemset)
        unique = np.asarray(sorted(itemset), dtype=np.int64)
        unique_users.extend([index] * len(unique))
        unique_items.extend(unique.tolist())
        history_users.extend([index] * len(hist))
        history_items.extend(hist.tolist())
        # Keep every occurrence; sparse coalescing sums repeated-item weights.
        weights = np.exp2(-np.arange(len(hist) - 1, -1, -1, dtype=np.float64) / 10.0)
        if len(hist):
            weights /= weights.sum()
        history_weights.extend(weights.tolist())
    n = len(ids)
    u = np.asarray(unique_users, dtype=np.int64)
    i = np.asarray(unique_items, dtype=np.int64)
    ud = np.bincount(u, minlength=n).astype(np.float32)
    it = np.bincount(i, minlength=itemnum + 1).astype(np.float32)
    weight = 1.0 / np.sqrt(ud[u] * it[i])
    graph = sparse_matrix(
        np.concatenate([u, n + i]),
        np.concatenate([n + i, u]),
        np.concatenate([weight, weight]),
        (n + itemnum + 1, n + itemnum + 1),
        device,
    )
    history_matrix = sparse_matrix(
        np.asarray(history_users),
        np.asarray(history_items),
        np.asarray(history_weights),
        (n, itemnum + 1),
        device,
    )
    return graph, history_matrix, histories, seen


def validation_cases(train, valid, itemnum):
    # Identical target and candidate semantics to the supplied benchmark.
    rng = random.Random(42)
    users = [u for u in valid if u in train]
    if len(users) > 200:
        users = rng.sample(users, 200)
    cases = []
    for u in users:
        if not valid[u]:
            continue
        excluded = set(train[u]) | set(valid[u])
        if itemnum - len(excluded) < 99:
            raise ValueError("Insufficient unique validation negatives")
        negatives = set()
        while len(negatives) < 99:
            candidate = rng.randint(1, itemnum)
            if candidate not in excluded:
                negatives.add(candidate)
        cases.append((u, train[u], [valid[u][0]] + list(negatives)))
    if not cases:
        raise ValueError("No validation users")
    return cases


@torch.no_grad()
def evaluate(model, cases, device):
    model.eval()
    hits, ndcg = 0.0, 0.0
    # Bucketing reduces padding while preserving complete histories.
    cases = sorted(cases, key=lambda case: len(case[1]))
    for start in range(0, len(cases), 32):
        batch = cases[start : start + 32]
        width = max(1, max(len(h) for _, h, _ in batch))
        histories = np.zeros((len(batch), width), dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.bool_)
        for row, (_, h, _) in enumerate(batch):
            histories[row, : len(h)] = h
            mask[row, : len(h)] = True
        scores = model(
            torch.tensor([u for u, _, _ in batch], device=device),
            torch.tensor(histories, device=device),
            torch.tensor(mask, device=device),
            torch.tensor([c for _, _, c in batch], device=device),
        )
        # Positive is first; ties follow the baseline's stable ordering.
        ranks = (scores[:, 1:] > scores[:, :1]).sum(1).cpu().numpy()
        hits += float((ranks < 10).sum())
        ndcg += float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0).sum())
    return hits / len(cases), ndcg / len(cases)


def train_dataset(ds, device, deadline):
    train, valid, itemnum = load_data(ds)
    ids = sorted(u for u in train if train[u])
    graph, history_matrix, histories, seen = build_graph(train, ids, itemnum, device)
    cases = validation_cases(train, valid, itemnum)
    rng = np.random.default_rng(42)
    model = LightGCN(len(ids), itemnum, graph, history_matrix, DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    path = os.path.join(WORK, ds + "_model.pt")
    eligible = np.asarray([i for i, h in enumerate(seen) if len(h) < itemnum])
    if not len(eligible):
        raise ValueError("No users with available training negatives")
    best_key = (-1.0, -1.0)
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step in range(STEPS):
            batch_users = rng.choice(eligible, size=BATCH)
            positives = np.fromiter(
                (histories[u][rng.integers(len(histories[u]))] for u in batch_users),
                dtype=np.int64,
                count=BATCH,
            )
            negatives = rng.integers(1, itemnum + 1, size=BATCH)
            for j, u in enumerate(batch_users):
                while int(negatives[j]) in seen[u]:
                    negatives[j] = rng.integers(1, itemnum + 1)
            u = torch.tensor(batch_users, dtype=torch.long, device=device)
            p = torch.tensor(positives, dtype=torch.long, device=device)
            n = torch.tensor(negatives, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_vectors, item_vectors = model.training_vectors()
            difference = (user_vectors[u] * (item_vectors[p] - item_vectors[n])).sum(-1)
            ranking = F.softplus(-difference).mean()
            raw_u = model.embedding(u)
            raw_p = model.embedding(p + len(ids))
            raw_n = model.embedding(n + len(ids))
            regularizer = (
                raw_u.square().sum() + raw_p.square().sum() + raw_n.square().sum()
            ) / (2.0 * BATCH)
            loss = ranking + 1e-4 * regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        should_evaluate = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS
        expired = time.monotonic() >= deadline
        if should_evaluate or expired:
            exported = model.export(ids)
            hr, ndcg = evaluate(exported, cases, device)
            key = (hr, ndcg)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                torch.save({k: v.cpu() for k, v in exported.state_dict().items()}, path)
            del exported
            print(
                f"{ds} epoch={epoch} loss={total_loss / STEPS:.5f} "
                f"val_HR@10={hr:.5f} val_NDCG@10={ndcg:.5f}",
                flush=True,
            )
        else:
            print(f"{ds} epoch={epoch} loss={total_loss / STEPS:.5f}", flush=True)
        if epoch - best_epoch >= 40 or expired:
            break
    state = torch.load(path, map_location=device, weights_only=True)
    best_model = PredictionModel(
        state["user_ids"], state["user_vectors"], state["item_vectors"]
    ).to(device)
    result, _ = evaluate(best_model, cases, device)
    del best_model, model, optimizer, graph, history_matrix
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_predict_script():
    script = r"""from __future__ import annotations
import os
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(ds):
    global _CLASS
    if ds in _MODELS:
        return _MODELS[ds]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'graph_training_definition', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.PredictionModel
    state = torch.load(os.path.join(_HERE, ds + '_model.pt'),
                       map_location=_DEVICE, weights_only=True)
    model = _CLASS(state['user_ids'], state['user_vectors'], state['item_vectors'])
    model.to(_DEVICE).eval()
    _MODELS[ds] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    model = _load(ds)
    # Preserve the complete chronological history and its validity mask.
    history = list(history)
    hist = history if history else [0]
    history_mask = [True] * len(history) if history else [False]
    with torch.inference_mode():
        scores = model(
            torch.tensor([user_id], dtype=torch.long, device=_DEVICE),
            torch.tensor([hist], dtype=torch.long, device=_DEVICE),
            torch.tensor([history_mask], dtype=torch.bool, device=_DEVICE),
            torch.tensor([candidates], dtype=torch.long, device=_DEVICE))
    return scores[0].cpu().tolist()
"""
    with open(os.path.join(WORK, "predict.py"), "w") as f:
        f.write(script)

    # Check sparse training profiles against padded scoring and the generated
    # predictor without loading data, changing random seeds, or training models.
    check_train = {1: [1, 2, 1], 2: [3], 3: []}
    check_ids = [1, 2, 3]
    check_device = torch.device("cpu")
    _, check_matrix, _, _ = build_graph(
        check_train, check_ids, 3, check_device
    )
    check_items = torch.eye(4, dtype=torch.float32)
    check_model = PredictionModel(
        torch.tensor(check_ids), torch.zeros(3, 4), check_items
    )
    check_profiles = torch.sparse.mm(check_matrix, check_items)
    expected_weights = torch.exp2(torch.tensor([-0.2, -0.1, 0.0]))
    expected_weights /= expected_weights.sum()
    expected_profiles = torch.zeros(3, 4)
    expected_profiles[0, 1] = expected_weights[0] + expected_weights[2]
    expected_profiles[0, 2] = expected_weights[1]
    expected_profiles[1, 3] = 1.0
    torch.testing.assert_close(check_profiles, expected_profiles)
    check_candidates = torch.arange(4).expand(3, -1)
    for width in (3, 7):
        check_histories = torch.zeros(3, width, dtype=torch.long)
        check_mask = torch.zeros(3, width, dtype=torch.bool)
        for row, uid in enumerate(check_ids):
            h = check_train[uid]
            # Include leading, internal, and trailing masked positions.
            positions = torch.arange(len(h)) * (2 if width == 7 else 1)
            check_histories[row, positions] = torch.tensor(h, dtype=torch.long)
            check_mask[row, positions] = True
        torch.testing.assert_close(
            check_model(torch.tensor(check_ids), check_histories,
                        check_mask, check_candidates),
            0.5 * check_profiles,
        )
    namespace = {"__file__": os.path.join(WORK, "predict.py")}
    exec(compile(script, namespace["__file__"], "exec"), namespace)
    namespace["_DEVICE"] = check_device
    namespace["_MODELS"][DATASETS[0]] = check_model
    for row, uid in enumerate(check_ids):
        scores = namespace["predict"](
            uid, check_train[uid], list(range(4)), dataset_name=DATASETS[0]
        )
        torch.testing.assert_close(torch.tensor(scores), 0.5 * check_profiles[row])


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORK, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scores = []
    # Fixed per-category allowance keeps all four models inside the run limit.
    allowance = float(os.environ.get("GRAPH_SECONDS_PER_DATASET", "6600"))
    for ds in DATASETS:
        scores.append(train_dataset(ds, device, time.monotonic() + allowance))
    write_predict_script()
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
