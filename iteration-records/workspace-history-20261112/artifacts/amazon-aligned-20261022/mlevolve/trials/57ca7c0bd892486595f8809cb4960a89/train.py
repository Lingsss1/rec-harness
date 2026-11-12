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
MAXLEN = 50
SASREC_CONFIG = dict(maxlen=MAXLEN, hidden_units=50, num_blocks=2,
                     num_heads=1, dropout_rate=0.2)
SASREC_BATCH = 128


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

    def __init__(self, user_ids, user_vectors, item_vectors, item_bias=None):
        super().__init__()
        self.register_buffer("user_ids", user_ids)
        self.register_buffer("user_vectors", user_vectors)
        self.register_buffer("item_vectors", item_vectors)
        if item_bias is None:
            item_bias = item_vectors.new_zeros(item_vectors.shape[0])
        self.register_buffer("item_bias", item_bias)

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
        return (
            (self.item_vectors[candidates] * representation[:, None]).sum(-1)
            + self.item_bias[candidates]
        )


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.conv1(inputs.transpose(1, 2))
        outputs = self.dropout1(outputs)
        outputs = self.conv2(self.relu(outputs))
        return inputs + self.dropout2(outputs).transpose(1, 2)


class SASRec(nn.Module):
    """Independent causal SASRec expert with the baseline optimization defaults."""

    def __init__(self, itemnum, maxlen=50, hidden_units=50, num_blocks=2,
                 num_heads=1, dropout_rate=0.2):
        super().__init__()
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.attention_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.forward_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)
        ])
        self.forward_layers = nn.ModuleList([
            PointWiseFeedForward(hidden_units, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)

    def log2feats(self, seq):
        valid = seq.ne(0).unsqueeze(-1)
        features = self.item_emb(seq) * math.sqrt(self.hidden_units)
        positions = torch.arange(seq.shape[1], device=seq.device)
        features = self.emb_dropout(features + self.pos_emb(positions)[None])
        features = features * valid
        causal = torch.ones(seq.shape[1], seq.shape[1], dtype=torch.bool,
                            device=seq.device).triu(1)
        for attention_norm, attention, forward_norm, feedforward in zip(
            self.attention_layernorms, self.attention_layers,
            self.forward_layernorms, self.forward_layers,
        ):
            values = features.transpose(0, 1)
            queries = attention_norm(values)
            attended, _ = attention(queries, values, values,
                                    attn_mask=causal, need_weights=False)
            features = (queries + attended).transpose(0, 1)
            features = feedforward(forward_norm(features)) * valid
        return self.last_layernorm(features) * valid

    def forward(self, seq, pos=None, neg=None):
        features = self.log2feats(seq)
        if pos is None:
            return features
        # Gather active transitions before embedding the 16 negative candidates.
        active = pos.ne(0)
        features = features[active]
        positive_items = pos[active]
        negative_items = neg[active]
        return ((features * self.item_emb(positive_items)).sum(-1),
                (features[:, None, :] * self.item_emb(negative_items)).sum(-1))

    def predict(self, seq, candidates):
        features = self.log2feats(seq)[:, -1]
        return (features[:, None] * self.item_emb(candidates)).sum(-1)


def sasrec_history(histories, history_mask, maxlen):
    # Compact valid interactions in chronological order, then right-align the
    # last maxlen. This is shared by validation and the generated predictor.
    valid = history_mask.bool() & histories.ne(0)
    positions = valid.long().cumsum(1) - 1
    lengths = valid.sum(1, keepdim=True)
    destinations = positions - (lengths - maxlen)
    keep = valid & destinations.ge(0)
    result = histories.new_zeros((histories.shape[0], maxlen))
    rows = torch.arange(histories.shape[0], device=histories.device)
    rows = rows[:, None].expand_as(histories)
    result[rows[keep], destinations[keep]] = histories[keep]
    return result


def candidate_ranks(scores, candidates):
    # Stable sorts implement descending score, ascending item ID. Scatter
    # restores the supplied candidate order; the target position is never used.
    by_item = torch.argsort(candidates, dim=1, stable=True)
    by_score = torch.argsort(scores.gather(1, by_item), dim=1,
                             descending=True, stable=True)
    order = by_item.gather(1, by_score)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, order.shape[1] + 1,
                   device=order.device).expand_as(order))
    return ranks


class SASRecPredictionModel(nn.Module):
    def __init__(self, sasrec):
        super().__init__()
        self.sasrec = sasrec

    def forward(self, user_ids, histories, history_mask, candidates):
        seq = sasrec_history(histories, history_mask, self.sasrec.maxlen)
        return self.sasrec.predict(seq, candidates)


def reciprocal_rank_scores(graph_rank, sasrec_rank, graph_weight):
    # Share float64 fusion between inference and the validation-only grid.
    return (graph_weight / (60.0 + graph_rank.to(torch.float64))
            + (1.0 - graph_weight) / (60.0 + sasrec_rank.to(torch.float64)))


class EnsemblePredictionModel(nn.Module):
    def __init__(self, graph, sasrec, graph_weight=0.5):
        super().__init__()
        self.graph = graph
        self.sasrec = sasrec
        if graph_weight not in (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Unsupported graph fusion weight")
        self.graph_weight = float(graph_weight)

    def forward(self, user_ids, histories, history_mask, candidates):
        graph_scores = self.graph(user_ids, histories, history_mask, candidates)
        seq = sasrec_history(histories, history_mask, self.sasrec.maxlen)
        sasrec_scores = self.sasrec.predict(seq, candidates)
        graph_rank = candidate_ranks(graph_scores, candidates).to(torch.float64)
        sasrec_rank = candidate_ranks(sasrec_scores, candidates).to(torch.float64)
        return reciprocal_rank_scores(graph_rank, sasrec_rank, self.graph_weight)


def load_graph(directory, ds, device):
    state = torch.load(os.path.join(directory, ds + "_model.pt"),
                       map_location=device, weights_only=True)
    model = PredictionModel(state["user_ids"], state["user_vectors"],
                            state["item_vectors"], state["item_bias"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_sasrec(directory, ds, device):
    checkpoint = torch.load(os.path.join(directory, ds + "_sasrec.pt"),
                            map_location=device, weights_only=True)
    # Reconstruct the actual saved architecture rather than current defaults.
    with torch.random.fork_rng(devices=[]):
        model = SASRec(checkpoint["itemnum"], **checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def load_ensemble(directory, ds, device):
    with open(os.path.join(directory, ds + "_ensemble.json")) as f:
        metadata = json.load(f)
    # Only legacy metadata without a weight uses the historical equal blend.
    graph_weight = metadata.get("graph_weight", 0.5)
    return EnsemblePredictionModel(load_graph(directory, ds, device),
                                   load_sasrec(directory, ds, device),
                                   graph_weight=graph_weight).eval()


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
        # Explicit item intercepts; unseen-negative sampling excludes padding.
        self.item_bias = nn.Parameter(torch.zeros(itemnum + 1))
        # Each category learns its own mixture of the existing three depths.
        self.layer_logits = nn.Parameter(torch.zeros(3))
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[users].zero_()
        self.register_buffer("graph", graph)
        self.register_buffer("history_matrix", history_matrix)
        # Coalesced entries aggregate every occurrence of a repeated item.
        history_indices = history_matrix.indices()
        self.register_buffer(
            "history_keys",
            history_indices[0] * (itemnum + 1) + history_indices[1],
        )
        self.register_buffer(
            "history_counts",
            torch.bincount(history_indices[0], minlength=users),
        )

    def encode(self):
        x = self.embedding.weight
        weights = torch.softmax(self.layer_logits, dim=0)
        total = weights[0] * x
        for layer in range(2):
            x = torch.sparse.mm(self.graph, x)
            total = total + weights[layer + 1] * x
        return total[: self.users], total[self.users :]

    def training_vectors(self):
        base, items = self.encode()
        profiles = torch.sparse.mm(self.history_matrix, items)
        # Retain the differentiable profiles for correction within this batch.
        self._training_profiles = profiles
        return base + 0.5 * profiles, items

    def target_excluded_vectors(self, u, p, user_vectors, item_vectors):
        """Remove all occurrences of each positive from its training profile."""
        if self.history_keys.numel() == 0:
            return user_vectors[u]
        keys = u * (self.itemnum + 1) + p
        locations = torch.searchsorted(self.history_keys, keys)
        safe_locations = locations.clamp(max=self.history_keys.numel() - 1)
        found = (locations < self.history_keys.numel()) & (
            self.history_keys[safe_locations] == keys
        )
        weights = torch.where(
            found,
            self.history_matrix.values()[safe_locations],
            torch.zeros_like(keys, dtype=item_vectors.dtype),
        ).unsqueeze(1)
        profiles = self._training_profiles[u]
        denominator = (1.0 - weights).clamp_min(
            torch.finfo(item_vectors.dtype).eps
        )
        excluded_profiles = (profiles - weights * item_vectors[p]) / denominator
        # Count distinct entries rather than relying on rounded weight sums.
        has_other = (self.history_counts[u] - found.long()) > 0
        excluded_profiles = torch.where(
            has_other[:, None], excluded_profiles, torch.zeros_like(profiles)
        )
        return user_vectors[u] + 0.5 * (excluded_profiles - profiles)

    @torch.no_grad()
    def export(self, original_user_ids):
        base, items = self.encode()
        return PredictionModel(
            torch.tensor(original_user_ids, dtype=torch.long, device=base.device),
            base.detach().clone(),
            items.detach().clone(),
            self.item_bias.detach().clone(),
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
        # Match the supplied baseline's stable candidate-order tie handling;
        # the validation target occupies candidate position zero.
        ranks = (scores > scores[:, :1]).sum(dim=1).cpu().numpy()
        hits += float((ranks < 10).sum())
        ndcg += float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0).sum())
    return hits / len(cases), ndcg / len(cases)


def metrics_from_scores(scores):
    # Exactly evaluate()'s target-at-zero, strict-greater ranking semantics.
    ranks = (scores > scores[:, :1]).sum(dim=1).cpu().numpy()
    return (float((ranks < 10).sum()) / len(ranks),
            float(np.where(ranks < 10, 1.0 / np.log2(ranks + 2), 0).sum())
            / len(ranks))


@torch.no_grad()
def cache_expert_ranks(graph, sasrec, cases, device):
    # Use the same bucketing, complete histories and batches as evaluate().
    # Each selected expert scores each case once; the grid uses CPU caches.
    graph.eval()
    sasrec.eval()
    graph_batches, sasrec_batches, candidate_batches = [], [], []
    cases = sorted(cases, key=lambda case: len(case[1]))
    for start in range(0, len(cases), 32):
        batch = cases[start : start + 32]
        width = max(1, max(len(h) for _, h, _ in batch))
        histories = np.zeros((len(batch), width), dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.bool_)
        for row, (_, h, _) in enumerate(batch):
            histories[row, : len(h)] = h
            mask[row, : len(h)] = True
        users = torch.tensor([u for u, _, _ in batch], device=device)
        histories = torch.tensor(histories, device=device)
        mask = torch.tensor(mask, device=device)
        candidates = torch.tensor([c for _, _, c in batch], device=device)
        graph_batches.append(graph(users, histories, mask, candidates).cpu())
        seq = sasrec_history(histories, mask, sasrec.maxlen)
        sasrec_batches.append(sasrec.predict(seq, candidates).cpu())
        candidate_batches.append(candidates.cpu())
    graph_scores = torch.cat(graph_batches)
    sasrec_scores = torch.cat(sasrec_batches)
    candidates = torch.cat(candidate_batches)
    return (candidate_ranks(graph_scores, candidates),
            candidate_ranks(sasrec_scores, candidates),
            metrics_from_scores(graph_scores), metrics_from_scores(sasrec_scores))


def select_fusion_result(grid_results):
    # HR only; prefer the equal blend, then the smaller graph weight on ties.
    return max(grid_results, key=lambda result: (
        result["HR@10"], -abs(result["graph_weight"] - 0.5),
        -result["graph_weight"],
    ))


def fusion_grid_results(graph_rank, sasrec_rank):
    results = []
    for graph_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        hr, ndcg = metrics_from_scores(
            reciprocal_rank_scores(graph_rank, sasrec_rank, graph_weight)
        )
        results.append({"graph_weight": graph_weight,
                        "sasrec_weight": 1.0 - graph_weight,
                        "HR@10": hr, "NDCG@10": ndcg})
    return results


def check_layer_mixture():
    # Deterministic CPU fixtures preserve training RNG state and add no graph
    # passes to training. Use an asymmetric graph with repeated positives.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        check_train = {1: [1, 2, 1], 2: [2, 3], 3: [3]}
        check_ids = sorted(check_train)
        graph, history_matrix, _, _ = build_graph(
            check_train, check_ids, 5, torch.device("cpu")
        )
        model = LightGCN(len(check_ids), 5, graph, history_matrix, 4)
        initial = model.embedding.weight
        first = torch.sparse.mm(graph, initial)
        second = torch.sparse.mm(graph, first)
        base, items = model.encode()
        torch.testing.assert_close(
            torch.cat([base, items]), (initial + first + second) / 3.0
        )
        with torch.no_grad():
            model.layer_logits.copy_(torch.tensor([-0.7, 0.2, 1.1]))
        weights = torch.softmax(model.layer_logits, dim=0)
        expected = weights[0] * initial + weights[1] * first + weights[2] * second
        base, items = model.encode()
        torch.testing.assert_close(torch.cat([base, items]), expected)
        users, items = model.training_vectors()
        torch.testing.assert_close(
            users, expected[:len(check_ids)]
            + 0.5 * torch.sparse.mm(history_matrix, expected[len(check_ids):])
        )
        u = torch.tensor([0, 0, 1, 2])
        p = torch.tensor([1, 2, 3, 3])
        n = torch.tensor([[4, 5], [5, 4], [1, 4], [1, 5]])
        corrected = model.target_excluded_vectors(u, p, users, items)
        positive = (corrected * items[p]).sum(-1) + model.item_bias[p]
        negative = (corrected[:, None] * items[n]).sum(-1) + model.item_bias[n]
        ranking = F.cross_entropy(
            torch.cat([positive[:, None], negative], dim=1),
            torch.zeros(len(u), dtype=torch.long),
        )
        gradient = torch.autograd.grad(ranking, model.layer_logits)[0]
        assert torch.isfinite(ranking)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0
        exported = model.export(check_ids)
        torch.testing.assert_close(exported.user_vectors, base)
        torch.testing.assert_close(exported.item_vectors, items)


def check_target_excluded_profiles():
    # CPU-only fixtures cover repeated positives, singleton histories, and
    # histories consisting entirely of the positive. Preserve training RNG state.
    check_train = {1: [1, 2, 1], 2: [3], 3: [2, 2, 2], 4: [1, 2, 3, 2]}
    check_ids = sorted(check_train)
    graph, history_matrix, _, _ = build_graph(
        check_train, check_ids, 3, torch.device("cpu")
    )
    with torch.random.fork_rng(devices=[]):
        model = LightGCN(len(check_ids), 3, graph, history_matrix, 4)
    with torch.no_grad():
        model.embedding.weight.copy_(
            torch.arange(model.embedding.weight.numel(), dtype=torch.float32)
            .reshape_as(model.embedding.weight) / 31.0
        )
    user_vectors, item_vectors = model.training_vectors()
    u = torch.tensor([0, 0, 1, 2, 3, 3])
    p = torch.tensor([1, 2, 3, 2, 2, 1])
    corrected = model.target_excluded_vectors(u, p, user_vectors, item_vectors)
    base, _ = model.encode()
    expected_profiles = []
    for row, positive in zip(u.tolist(), p.tolist()):
        history = torch.tensor(check_train[check_ids[row]])
        # Filter occurrences while preserving their original chronological ages.
        weights = torch.exp2(
            -torch.arange(len(history) - 1, -1, -1, dtype=torch.float32) / 10.0
        )
        keep = history != positive
        if keep.any():
            kept_weights = weights[keep] / weights[keep].sum()
            profile = (item_vectors[history[keep]] * kept_weights[:, None]).sum(0)
        else:
            profile = torch.zeros_like(item_vectors[0])
        expected_profiles.append(profile)
    expected = base[u] + 0.5 * torch.stack(expected_profiles)
    torch.testing.assert_close(corrected, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(corrected[2:4], base[u[2:4]])
    # Compare gradients as well as values to catch detached correction paths.
    actual_gradient = torch.autograd.grad(
        corrected.square().sum(), model.embedding.weight, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(
        expected.square().sum(), model.embedding.weight
    )[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6
    )


def sample_negatives(rng, batch_users, seen, itemnum, negative_count=32):
    # Independent uniform draws with replacement, rejecting only seen items.
    negatives = rng.integers(
        1, itemnum + 1, size=(len(batch_users), negative_count)
    )
    for j, u in enumerate(batch_users):
        if len(seen[u]) >= itemnum:
            raise ValueError("No available training negatives")
        invalid = np.fromiter(
            (int(candidate) in seen[u] for candidate in negatives[j]),
            dtype=np.bool_,
            count=negative_count,
        )
        while invalid.any():
            negatives[j, invalid] = rng.integers(
                1, itemnum + 1, size=int(invalid.sum())
            )
            invalid = np.fromiter(
                (int(candidate) in seen[u] for candidate in negatives[j]),
                dtype=np.bool_,
                count=negative_count,
            )
    return negatives


def check_sampled_softmax():
    # Local NumPy RNG and forked Torch RNG preserve the training RNG state.
    check_train = {1: [1, 2, 1], 2: [3], 3: [2, 2], 4: [1, 2, 3, 4]}
    check_ids = sorted(check_train)
    graph, history_matrix, _, seen = build_graph(
        check_train, check_ids, 5, torch.device("cpu")
    )
    rng = np.random.default_rng(123)
    batch_users = np.asarray([0, 0, 1, 2, 3, 3])
    negatives = sample_negatives(rng, batch_users, seen, 5)
    assert negatives.shape == (len(batch_users), 32)
    assert ((negatives >= 1) & (negatives <= 5)).all()
    for row, uid in enumerate(batch_users):
        assert all(int(candidate) not in seen[uid] for candidate in negatives[row])
    # A singleton eligible pool must still support 32 draws with replacement.
    assert (negatives[batch_users == 3] == 5).all()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = LightGCN(len(check_ids), 5, graph, history_matrix, 4)
    u = torch.tensor(batch_users, dtype=torch.long)
    p = torch.tensor([1, 2, 3, 2, 4, 1], dtype=torch.long)
    n = torch.tensor(negatives, dtype=torch.long)
    user_vectors, item_vectors = model.training_vectors()
    batch_user_vectors = model.target_excluded_vectors(
        u, p, user_vectors, item_vectors
    )
    positive_scores = (
        (batch_user_vectors * item_vectors[p]).sum(-1) + model.item_bias[p]
    )
    negative_scores = (
        batch_user_vectors[:, None, :] * item_vectors[n]
    ).sum(-1) + model.item_bias[n]
    # Zero initialization preserves the original sampled-softmax logits.
    torch.testing.assert_close(
        positive_scores, (batch_user_vectors * item_vectors[p]).sum(-1)
    )
    torch.testing.assert_close(
        negative_scores,
        (batch_user_vectors[:, None, :] * item_vectors[n]).sum(-1),
    )
    logits = torch.cat([positive_scores[:, None], negative_scores], dim=1)
    assert positive_scores.shape == (len(batch_users),)
    assert negative_scores.shape == (len(batch_users), 32)
    assert logits.shape == (len(batch_users), 33)
    zero_class_targets = torch.zeros(len(batch_users), dtype=torch.long)
    ranking = F.cross_entropy(logits.float(), zero_class_targets)
    assert torch.isfinite(ranking)
    gradients = torch.autograd.grad(
        ranking,
        (batch_user_vectors, model.embedding.weight, model.item_bias),
        retain_graph=True,
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[-1][0].item() == 0.0
    assert gradients[-1][1:].abs().sum().item() > 0.0
    # Temperature-1 cross-entropy with one negative equals softplus BPR.
    one_negative = F.cross_entropy(logits[:, :2].float(), zero_class_targets)
    bpr = F.softplus(-(positive_scores - negative_scores[:, 0])).mean()
    torch.testing.assert_close(one_negative, bpr)


def recency_position_cdf(length):
    # Match inference's ten-interaction half-life without changing the loss.
    weights = np.exp2(-np.arange(length - 1, -1, -1, dtype=np.float64) / 10.0)
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return cdf


def check_recency_sampling():
    # Separate local RNG leaves all training RNG state untouched.
    rng = np.random.default_rng(123)
    for length in (1, 3, 11, 37):
        cdf = recency_position_cdf(length)
        probabilities = np.diff(np.concatenate(([0.0], cdf)))
        assert cdf.dtype == np.float64
        assert cdf[-1] == 1.0
        assert np.isfinite(probabilities).all()
        assert (probabilities > 0.0).all()
        np.testing.assert_allclose(probabilities.sum(), 1.0)
        expected = np.exp2(
            -np.arange(length - 1, -1, -1, dtype=np.float64) / 10.0
        )
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities, expected)
        if length >= 11:
            np.testing.assert_allclose(probabilities[-1] / probabilities[-11], 2.0)
        positions = np.searchsorted(cdf, rng.random(256), side="right")
        assert ((positions >= 0) & (positions < length)).all()
        assert np.searchsorted(cdf, 0.0, side="right") == 0
        assert np.searchsorted(cdf, np.nextafter(1.0, 0.0), side="right") == length - 1
        if length == 1:
            np.testing.assert_array_equal(cdf, [1.0])
            assert (positions == 0).all()
    # Repeated items retain the sum of their distinct chronological position masses.
    history = np.asarray([1, 2, 1], dtype=np.int64)
    cdf = recency_position_cdf(len(history))
    probabilities = np.diff(np.concatenate(([0.0], cdf)))
    expected = np.exp2(np.asarray([-2.0, -1.0, 0.0]) / 10.0)
    expected /= expected.sum()
    np.testing.assert_allclose(
        probabilities[history == 1].sum(), expected[0] + expected[2]
    )
    midpoints = (np.concatenate(([0.0], cdf[:-1])) + cdf) / 2.0
    positions = np.searchsorted(cdf, midpoints, side="right")
    np.testing.assert_array_equal(positions, np.arange(len(history)))
    np.testing.assert_array_equal(history[positions], history)


def check_sasrec_sampled_softmax():
    # CPU fixtures use isolated NumPy/Torch RNG state. Preserve both experts'
    # training streams while checking sampling, active gathering, and the loss.
    rng = np.random.default_rng(123)
    check_train = {1: [4, 1, 2, 3], 2: [2, 3]}
    users = [1, 2]
    seen = [set(check_train[u]) for u in users]
    chosen = np.asarray([0, 1])
    seq = np.zeros((2, 4), dtype=np.int64)
    pos = np.zeros_like(seq)
    neg = np.zeros((2, 4, 16), dtype=np.int64)
    for row, index in enumerate(chosen):
        h = check_train[users[index]]
        length = min(len(h) - 1, seq.shape[1])
        seq[row, -length:] = h[-length - 1:-1]
        pos[row, -length:] = h[-length:]
        neg[row, -length:, :] = sample_negatives(
            rng, [index], seen, 5, negative_count=length * 16
        )[0].reshape(length, 16)
    active = pos != 0
    assert neg.shape == (2, 4, 16)
    assert (neg[~active] == 0).all()
    assert ((neg[active] >= 1) & (neg[active] <= 5)).all()
    for row, index in enumerate(chosen):
        assert all(int(candidate) not in seen[index]
                   for candidate in neg[row, active[row]].ravel())
    # Only item 5 is eligible for the first user: duplicates are required.
    assert (neg[0, active[0]] == 5).all()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = SASRec(5, maxlen=4, hidden_units=4, num_blocks=1,
                       num_heads=1, dropout_rate=0.0).eval()
        seq = torch.from_numpy(seq)
        pos = torch.from_numpy(pos)
        neg = torch.from_numpy(neg)
        active = pos.ne(0)
        features = model(seq)
        assert features.shape == (2, 4, 4)
        positive, negative = model(seq, pos, neg)
        assert positive.shape == (4,)
        assert negative.shape == (4, 16)
        torch.testing.assert_close(
            positive, (features[active] * model.item_emb(pos[active])).sum(-1)
        )
        torch.testing.assert_close(
            negative,
            (features[active][:, None, :] * model.item_emb(neg[active])).sum(-1),
        )
        # Out-of-range IDs in padded positions must never reach the embedding.
        padded_neg = neg.clone()
        padded_neg[~active] = 6
        padded_positive, padded_negative = model(seq, pos, padded_neg)
        torch.testing.assert_close(padded_positive, positive)
        torch.testing.assert_close(padded_negative, negative)
        logits = torch.cat([positive[:, None], negative], dim=1).float()
        assert logits.shape == (4, 17)
        loss = F.cross_entropy(
            logits, torch.zeros(positive.shape[0], dtype=torch.long)
        )
        expected = (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()
        torch.testing.assert_close(loss, expected)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert model.item_emb.weight.grad[1:].abs().sum().item() > 0.0
        assert model.pos_emb.weight.grad.abs().sum().item() > 0.0
        assert model.item_emb.weight.grad[0].abs().sum().item() == 0.0
    print("SASRec 16-negative sampled-softmax fixtures passed", flush=True)


def train_sasrec(ds, train, itemnum, cases, device, deadline):
    # Independent streams leave the graph expert's Torch/NumPy RNG untouched.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        if devices:
            torch.cuda.manual_seed_all(42)
        rng = np.random.default_rng(42)
        config = dict(SASREC_CONFIG)
        model = SASRec(itemnum, **config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                     betas=(0.9, 0.98))
        users = [u for u, h in train.items()
                 if len(h) > 1 and len(set(h)) < itemnum]
        if not users:
            raise ValueError("No eligible SASRec training users")
        seen = [set(train[u]) for u in users]
        path = os.path.join(WORK, ds + "_sasrec.pt")
        best_key = (-1.0, -1.0)
        best_epoch = 0

        def validate_and_save(epoch):
            nonlocal best_key, best_epoch
            key = evaluate(SASRecPredictionModel(model), cases, device)
            if key > best_key:
                best_key, best_epoch = key, epoch
                torch.save({
                    "itemnum": itemnum,
                    "architecture": config,
                    "state_dict": {k: v.detach().cpu()
                                   for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "validation_hr": key[0],
                    "validation_ndcg": key[1],
                }, path)
            print(f"{ds} SASRec epoch={epoch} val_HR@10={key[0]:.5f} "
                  f"val_NDCG@10={key[1]:.5f}", flush=True)

        # Always establish a validated checkpoint, even if the deadline is tight.
        validate_and_save(0)
        steps = max(1, len(users) // SASREC_BATCH)
        for epoch in range(1, EPOCHS + 1):
            if time.monotonic() >= deadline:
                break
            model.train()
            total_loss, completed = 0.0, 0
            for _ in range(steps):
                if time.monotonic() >= deadline:
                    break
                chosen = rng.integers(len(users), size=SASREC_BATCH)
                seq = np.zeros((SASREC_BATCH, config["maxlen"]), dtype=np.int64)
                pos = np.zeros_like(seq)
                neg = np.zeros((SASREC_BATCH, config["maxlen"], 16),
                               dtype=np.int64)
                for row, index in enumerate(chosen):
                    h = train[users[index]]
                    length = min(len(h) - 1, config["maxlen"])
                    seq[row, -length:] = h[-length - 1:-1]
                    pos[row, -length:] = h[-length:]
                    neg[row, -length:, :] = sample_negatives(
                        rng, [index], seen, itemnum, negative_count=length * 16
                    )[0].reshape(length, 16)
                seq = torch.as_tensor(seq, device=device)
                pos = torch.as_tensor(pos, device=device)
                neg = torch.as_tensor(neg, device=device)
                positive, negative = model(seq, pos, neg)
                # Temperature-1 competition over one positive and 16 negatives;
                # every active chronological transition receives equal weight.
                loss = F.cross_entropy(
                    torch.cat([positive[:, None], negative], dim=1).float(),
                    torch.zeros(positive.shape[0], dtype=torch.long, device=device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()
                completed += 1
            expired = time.monotonic() >= deadline
            if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS or expired:
                validate_and_save(epoch)
            print(f"{ds} SASRec epoch={epoch} "
                  f"loss={total_loss / max(1, completed):.5f}", flush=True)
            if expired:
                break
        del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_dataset(ds, device, deadline):
    # Freeze the graph configuration. Reserve explicit sequential expert shares
    # and final reload/validation time within this category's allocation.
    started = time.monotonic()
    remaining = max(0.0, deadline - started)
    export_reserve = min(60.0, remaining * 0.05)
    training_seconds = max(0.0, remaining - export_reserve)
    graph_deadline = started + 0.45 * training_seconds
    sasrec_deadline = deadline - export_reserve
    train, valid, itemnum = load_data(ds)
    ids = sorted(u for u in train if train[u])
    graph, history_matrix, histories, seen = build_graph(train, ids, itemnum, device)
    # Precompute per-position distributions; histories retain their original order.
    positive_cdfs = [recency_position_cdf(len(h)) for h in histories]
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
                (
                    histories[u][np.searchsorted(
                        positive_cdfs[u], rng.random(), side="right"
                    )]
                    for u in batch_users
                ),
                dtype=np.int64,
                count=BATCH,
            )
            negatives = sample_negatives(rng, batch_users, seen, itemnum)
            u = torch.tensor(batch_users, dtype=torch.long, device=device)
            p = torch.tensor(positives, dtype=torch.long, device=device)
            n = torch.tensor(negatives, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_vectors, item_vectors = model.training_vectors()
            batch_user_vectors = model.target_excluded_vectors(
                u, p, user_vectors, item_vectors
            )
            positive_scores = (
                (batch_user_vectors * item_vectors[p]).sum(-1) + model.item_bias[p]
            )
            negative_scores = (
                batch_user_vectors[:, None, :] * item_vectors[n]
            ).sum(-1) + model.item_bias[n]
            logits = torch.cat([positive_scores[:, None], negative_scores], dim=1)
            zero_class_targets = torch.zeros(BATCH, dtype=torch.long, device=device)
            ranking = F.cross_entropy(logits.float(), zero_class_targets)
            raw_u = model.embedding(u)
            raw_p = model.embedding(p + len(ids))
            raw_n = model.embedding(n + len(ids))
            regularizer = (
                raw_u.square().sum()
                + raw_p.square().sum()
                + raw_n.square().sum(-1).mean(dim=1).sum()
                + model.item_bias[p].square().sum()
                + model.item_bias[n].square().mean(dim=1).sum()
            ) / (2.0 * BATCH)
            loss = ranking + 1e-4 * regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        should_evaluate = epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS
        expired = time.monotonic() >= graph_deadline
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
    del model, optimizer, graph, history_matrix
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_sasrec(ds, train, itemnum, cases, device, sasrec_deadline)
    # Reload both independently selected checkpoints, without consulting old
    # ensemble metadata. No training or checkpoint search is added for fusion.
    graph = load_graph(WORK, ds, device)
    sasrec = load_sasrec(WORK, ds, device)
    graph_rank, sasrec_rank, graph_metrics, sasrec_metrics = cache_expert_ranks(
        graph, sasrec, cases, device
    )
    grid_results = fusion_grid_results(graph_rank, sasrec_rank)
    selected = select_fusion_result(grid_results)
    equal_result = next(result for result in grid_results
                        if result["graph_weight"] == 0.5)
    assert selected["HR@10"] >= equal_result["HR@10"]
    graph_weight = selected["graph_weight"]
    fused_metrics = (selected["HR@10"], selected["NDCG@10"])
    del graph, sasrec, graph_rank, sasrec_rank
    print(f"{ds} selected graph HR@10={graph_metrics[0]:.5f} "
          f"graph NDCG@10={graph_metrics[1]:.5f} "
          f"SASRec HR@10={sasrec_metrics[0]:.5f} "
          f"SASRec NDCG@10={sasrec_metrics[1]:.5f} "
          f"ensemble HR@10={fused_metrics[0]:.5f} "
          f"ensemble NDCG@10={fused_metrics[1]:.5f}", flush=True)
    with open(os.path.join(WORK, ds + "_ensemble.json"), "w") as f:
        json.dump({"fusion": "weighted_reciprocal_rank", "constant": 60,
                   "graph_weight": graph_weight,
                   "sasrec_weight": 1.0 - graph_weight,
                   "grid_validation": grid_results,
                   "graph_checkpoint": ds + "_model.pt",
                   "sasrec_checkpoint": ds + "_sasrec.pt",
                   "sasrec_architecture": dict(SASREC_CONFIG),
                   "graph_validation": graph_metrics,
                   "sasrec_validation": sasrec_metrics,
                   "ensemble_validation": fused_metrics,
                   "equal_weight_validation": [equal_result["HR@10"],
                                               equal_result["NDCG@10"]]},
                  f, indent=2)
    # Persist the selected weight before exercising the production load path.
    ensemble = load_ensemble(WORK, ds, device)
    reloaded_metrics = evaluate(ensemble, cases, device)
    np.testing.assert_allclose(reloaded_metrics, fused_metrics, rtol=0, atol=1e-12)
    print(f"{ds} selected graph_weight={graph_weight:.2f} "
          f"local HR@10={fused_metrics[0]:.5f} "
          f"same-checkpoint equal-weight HR@10={equal_result['HR@10']:.5f}; "
          "local nondecrease verified; independent host improvement unproven",
          flush=True)
    del ensemble
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fused_metrics[0]


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
_LOADER = None


def _load(ds):
    global _CLASS, _LOADER
    if ds in _MODELS:
        return _MODELS[ds]
    if _CLASS is None:
        if _LOADER is None:
            spec = importlib.util.spec_from_file_location(
                'ensemble_training_definition', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _LOADER = module.load_ensemble
        model = _LOADER(_HERE, ds, _DEVICE)
        _MODELS[ds] = model
        return model
    # Retain the materialized graph loader for the existing graph-only fixtures.
    state = torch.load(os.path.join(_HERE, ds + '_model.pt'),
                       map_location=_DEVICE, weights_only=True)
    model = _CLASS(state['user_ids'], state['user_vectors'], state['item_vectors'],
                   state['item_bias'])
    model.load_state_dict(state, strict=True)
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
    check_graph, check_matrix, _, _ = build_graph(
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
    # Verify explicit zero-bias equivalence and candidate-wise intercepts.
    # These fixtures use no random draws and leave training RNG state intact.
    check_zero_model = PredictionModel(
        torch.tensor(check_ids), torch.zeros(3, 4), check_items,
        torch.zeros(4),
    )
    torch.testing.assert_close(
        check_zero_model(torch.tensor(check_ids), check_histories,
                         check_mask, check_candidates),
        check_model(torch.tensor(check_ids), check_histories,
                    check_mask, check_candidates),
    )
    check_bias = torch.tensor([0.0, 0.3, -0.2, 0.7])
    with torch.random.fork_rng(devices=[]):
        check_graph_model = LightGCN(
            len(check_ids), 3,
            check_graph,
            check_matrix, 4,
        )
    with torch.no_grad():
        check_graph_model.item_bias.copy_(check_bias)
        check_graph_model.layer_logits.copy_(torch.tensor([-0.7, 0.2, 1.1]))
    check_export = check_graph_model.export(check_ids)
    torch.testing.assert_close(check_export.item_bias, check_bias)
    assert not check_export.item_bias.requires_grad
    assert check_export.item_bias.data_ptr() != check_graph_model.item_bias.data_ptr()
    check_model = PredictionModel(
        torch.tensor(check_ids), torch.zeros(3, 4), check_items, check_bias
    )
    torch.testing.assert_close(
        check_model(torch.tensor(check_ids), check_histories,
                    check_mask, check_candidates),
        0.5 * check_profiles + check_bias[check_candidates],
    )
    namespace = {"__file__": os.path.join(WORK, "predict.py")}
    exec(compile(script, namespace["__file__"], "exec"), namespace)
    namespace["_DEVICE"] = check_device
    namespace["_MODELS"][DATASETS[0]] = check_model
    for row, uid in enumerate(check_ids):
        scores = namespace["predict"](
            uid, check_train[uid], list(range(4)), dataset_name=DATASETS[0]
        )
        torch.testing.assert_close(
            torch.tensor(scores), 0.5 * check_profiles[row] + check_bias
        )

    # Materialize the learned nonuniform mixture into the unchanged checkpoint
    # buffers and compare validation-style scores through the actual loader.
    # PredictionModel.forward(), routing, and inference format stay unchanged.
    check_model = check_export
    with torch.no_grad():
        check_direct_scores = check_model(
            torch.tensor(check_ids), check_histories,
            check_mask, check_candidates,
        )
    import tempfile
    with tempfile.TemporaryDirectory(dir=WORK) as check_directory:
        torch.save(
            check_model.state_dict(),
            os.path.join(check_directory, DATASETS[0] + "_model.pt"),
        )
        state = torch.load(
            os.path.join(check_directory, DATASETS[0] + "_model.pt"),
            map_location=check_device, weights_only=True,
        )
        check_restored = PredictionModel(
            state["user_ids"], state["user_vectors"], state["item_vectors"],
            state["item_bias"],
        )
        torch.testing.assert_close(
            check_restored(torch.tensor(check_ids), check_histories,
                           check_mask, check_candidates),
            check_direct_scores,
        )
        namespace["_HERE"] = check_directory
        namespace["_CLASS"] = PredictionModel
        namespace["_MODELS"].clear()
        for row, uid in enumerate(check_ids):
            scores = namespace["predict"](
                uid, check_train[uid], list(range(4)), dataset_name=DATASETS[0]
            )
            torch.testing.assert_close(
                torch.tensor(scores), check_direct_scores[row]
            )
        # Also exercise validation ranking and dataset routing for every category.
        check_cases = [
            (uid, check_train[uid], [1, 2, 3]) for uid in check_ids
        ]
        direct_metrics = evaluate(check_model, check_cases, check_device)
        for ds in DATASETS:
            torch.save(
                check_model.state_dict(),
                os.path.join(check_directory, ds + "_model.pt"),
            )
            namespace["_MODELS"].clear()
            loaded = namespace["_load"](ds)
            torch.testing.assert_close(
                loaded(torch.tensor(check_ids), check_histories,
                       check_mask, check_candidates),
                check_direct_scores,
            )
            assert evaluate(loaded, check_cases, check_device) == direct_metrics
            for row, uid in enumerate(check_ids):
                scores = namespace["predict"](
                    uid, check_train[uid], list(range(4)), dataset_name=ds
                )
                torch.testing.assert_close(
                    torch.tensor(scores), check_direct_scores[row]
                )


    check_ensemble_export(script)


def check_ensemble_export(script):
    # CPU fixtures use local/forked RNG only. Cover real checkpoint loading,
    # all dataset routes, empty histories, padding, ties and candidate order.
    import tempfile
    device = torch.device("cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        config = dict(maxlen=3, hidden_units=4, num_blocks=1,
                      num_heads=1, dropout_rate=0.0)
        sasrec = SASRec(5, **config).eval()
        graph = PredictionModel(torch.tensor([1, 2]), torch.zeros(2, 4),
                                torch.randn(6, 4), torch.zeros(6)).eval()
        model = EnsemblePredictionModel(graph, sasrec).eval()
        users = torch.tensor([1, 2, 99])
        histories = torch.tensor([[0, 1, 0, 2, 3, 4, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 2, 0, 0, 0, 0]])
        mask = histories.ne(0)
        expected_seq = torch.tensor([[2, 3, 4], [0, 0, 0], [0, 1, 2]])
        torch.testing.assert_close(sasrec_history(histories, mask, 3), expected_seq)
        candidates = torch.tensor([[5, 1, 4, 2, 3]]).expand(3, -1)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        with torch.inference_mode():
            direct = model(users, histories, mask, candidates)
            assert torch.isfinite(direct).all()
            torch.testing.assert_close(
                model(users, histories, mask, candidates[:, permutation]),
                direct[:, permutation], rtol=0, atol=0,
            )
            graph_scores = graph(users, histories, mask, candidates)
            sas_scores = sasrec.predict(expected_seq, candidates)
            expected = (0.5 / (60 + candidate_ranks(graph_scores, candidates).double())
                        + 0.5 / (60 + candidate_ranks(sas_scores, candidates).double()))
            torch.testing.assert_close(direct, expected, rtol=0, atol=0)
            graph_rank = candidate_ranks(graph_scores, candidates)
            sasrec_rank = candidate_ranks(sas_scores, candidates)
            for graph_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                weighted = EnsemblePredictionModel(
                    graph, sasrec, graph_weight=graph_weight
                ).eval()
                actual = weighted(users, histories, mask, candidates)
                expected_weighted = (
                    graph_weight / (60.0 + graph_rank.double())
                    + (1.0 - graph_weight) / (60.0 + sasrec_rank.double())
                )
                assert actual.dtype == torch.float64
                torch.testing.assert_close(actual, expected_weighted, rtol=0, atol=0)
                torch.testing.assert_close(
                    weighted(users, histories, mask, candidates[:, permutation]),
                    actual[:, permutation], rtol=0, atol=0,
                )
                # Internal/leading/trailing padding and empty histories agree
                # with compact chronological histories for both experts.
                for row in range(len(users)):
                    compact = histories[row, mask[row]].tolist()
                    compact_history = torch.tensor([compact if compact else [0]])
                    compact_mask = torch.tensor(
                        [[True] * len(compact) if compact else [False]]
                    )
                    torch.testing.assert_close(
                        weighted(users[row:row + 1], compact_history,
                                 compact_mask, candidates[row:row + 1]),
                        actual[row:row + 1], rtol=0, atol=0,
                    )
                if graph_weight in (0.0, 1.0):
                    expert_rank = graph_rank if graph_weight == 1.0 else sasrec_rank
                    torch.testing.assert_close(
                        candidate_ranks(actual, candidates), expert_rank,
                    )

        # Compare cached grid metrics with evaluate(), including target ties
        # and ranks on both sides of the HR@10 cutoff.
        fixture_graph_rank = torch.tensor([
            [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
            [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            [6, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        ])
        fixture_sasrec_rank = torch.tensor([
            [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
            [7, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        ])

        class FixtureScores(nn.Module):
            def __init__(self, scores):
                super().__init__()
                self.register_buffer("scores", scores)

            def forward(self, user_ids, histories, history_mask, candidates):
                return self.scores[user_ids]

        fixture_cases = [(row, [], list(range(1, 13))) for row in range(3)]
        grid_results = fusion_grid_results(fixture_graph_rank, fixture_sasrec_rank)
        for result in grid_results:
            scores = reciprocal_rank_scores(
                fixture_graph_rank, fixture_sasrec_rank, result["graph_weight"]
            )
            measured = evaluate(FixtureScores(scores), fixture_cases, device)
            assert measured == (result["HR@10"], result["NDCG@10"])
        equal_tie = [{"graph_weight": weight, "HR@10": 0.5,
                      "NDCG@10": float(weight != 0.5)}
                     for weight in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert select_fusion_result(equal_tie)["graph_weight"] == 0.5
        distance_tie = [result for result in equal_tie
                        if result["graph_weight"] in (0.25, 0.75)]
        assert select_fusion_result(distance_tie)["graph_weight"] == 0.25
        hr_priority = equal_tie + [{"graph_weight": 1.0, "HR@10": 0.6,
                                   "NDCG@10": 0.0}]
        assert select_fusion_result(hr_priority)["graph_weight"] == 1.0
        selected = select_fusion_result(grid_results)
        assert selected["HR@10"] >= grid_results[2]["HR@10"]
        tied_candidates = torch.tensor([[5, 2, 4, 1, 3]])
        tied_ranks = candidate_ranks(torch.zeros(1, 5), tied_candidates)
        torch.testing.assert_close(tied_ranks, tied_candidates)
        # Force exact ties in both experts and verify one-based RRF numerically.
        with torch.no_grad():
            tied_graph = PredictionModel(torch.tensor([1]), torch.zeros(1, 4),
                                         torch.zeros(6, 4))
            tied_sas = SASRec(5, **config).eval()
            for parameter in tied_sas.parameters():
                parameter.zero_()
            tied = EnsemblePredictionModel(tied_graph, tied_sas).eval()
            actual = tied(torch.tensor([1]), torch.zeros(1, 1, dtype=torch.long),
                          torch.zeros(1, 1, dtype=torch.bool), tied_candidates)
            torch.testing.assert_close(actual, 1.0 / (60 + tied_candidates.double()))
            for graph_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                tied.graph_weight = graph_weight
                actual = tied(
                    torch.tensor([1]), torch.zeros(1, 1, dtype=torch.long),
                    torch.zeros(1, 1, dtype=torch.bool), tied_candidates,
                )
                torch.testing.assert_close(
                    candidate_ranks(actual, tied_candidates), tied_candidates,
                )
                # evaluate() retains target-position tie semantics even when
                # endpoint fusion resolves the expert's raw-score ties by ID.
                assert metrics_from_scores(torch.zeros(1, 12)) == (1.0, 1.0)

        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            for ds in DATASETS:
                torch.save(graph.state_dict(), os.path.join(directory, ds + "_model.pt"))
                torch.save({"itemnum": 5, "architecture": config,
                            "state_dict": sasrec.state_dict()},
                           os.path.join(directory, ds + "_sasrec.pt"))
            namespace = {"__file__": os.path.join(directory, "predict.py")}
            exec(compile(script, namespace["__file__"], "exec"), namespace)
            namespace["_HERE"] = directory
            namespace["_DEVICE"] = device
            # Exercise the generated lazy-loading path using the same loader
            # that its import of train.py supplies in production.
            namespace["_LOADER"] = load_ensemble
            for ds in DATASETS:
                # Every route reloads each predefined weight from metadata;
                # an omitted weight exercises the legacy equal-weight default.
                for graph_weight in (0.0, 0.25, 0.5, 0.75, 1.0, None):
                    metadata = {"constant": 60}
                    if graph_weight is not None:
                        metadata.update(graph_weight=graph_weight,
                                        sasrec_weight=1.0 - graph_weight)
                    with open(os.path.join(directory, ds + "_ensemble.json"), "w") as f:
                        json.dump(metadata, f)
                    expected_weight = 0.5 if graph_weight is None else graph_weight
                    namespace["_MODELS"].clear()
                    restored = namespace["_load"](ds)
                    assert restored.graph_weight == expected_weight
                    expected_model = EnsemblePredictionModel(
                        graph, sasrec, graph_weight=expected_weight
                    ).eval()
                    with torch.inference_mode():
                        expected_direct = expected_model(
                            users, histories, mask, candidates
                        )
                        torch.testing.assert_close(
                            restored(users, histories, mask, candidates),
                            expected_direct, rtol=0, atol=0,
                        )
                    check_cases = [
                        (uid, histories[row, mask[row]].tolist(),
                         candidates[row].tolist())
                        for row, uid in enumerate(users.tolist())
                    ]
                    cached_graph, cached_sasrec, graph_metrics, sasrec_metrics = (
                        cache_expert_ranks(graph, sasrec, check_cases, device)
                    )
                    np.testing.assert_allclose(
                        graph_metrics, evaluate(graph, check_cases, device),
                        rtol=0, atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        sasrec_metrics,
                        evaluate(SASRecPredictionModel(sasrec), check_cases, device),
                        rtol=0, atol=1e-12,
                    )
                    cached_metrics = metrics_from_scores(reciprocal_rank_scores(
                        cached_graph, cached_sasrec, expected_weight
                    ))
                    np.testing.assert_allclose(
                        cached_metrics, evaluate(restored, check_cases, device),
                        rtol=0, atol=1e-12,
                    )
                    for row, uid in enumerate(users.tolist()):
                        history = histories[row, mask[row]].tolist()
                        for order in (torch.arange(5), permutation):
                            scores = namespace["predict"](
                                uid, history, candidates[row, order].tolist(),
                                dataset_name=ds,
                            )
                            torch.testing.assert_close(
                                torch.tensor(scores, dtype=torch.float64),
                                expected_direct[row, order], rtol=0, atol=0,
                            )
    print("Ensemble export fixtures passed", flush=True)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    os.makedirs(WORK, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    check_layer_mixture()
    check_target_excluded_profiles()
    check_sampled_softmax()
    check_recency_sampling()
    check_sasrec_sampled_softmax()
    scores = []
    # Sequential single-GPU training; cap requested allocations below the shared
    # budget and reserve time for export fixtures. Existing split files are used
    # directly, so preprocessing and its seed-0 splits remain unchanged.
    allowance = float(os.environ.get("GRAPH_SECONDS_PER_DATASET", "6600"))
    total_seconds = min(43080.0, max(0.0, allowance) * len(DATASETS))
    run_deadline = time.monotonic() + total_seconds
    for index, ds in enumerate(DATASETS):
        now = time.monotonic()
        category_seconds = max(0.0, run_deadline - now - 30.0) / (len(DATASETS) - index)
        scores.append(train_dataset(ds, device, now + category_seconds))
    write_predict_script()
    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}")
    for ds, category_score in zip(DATASETS, scores):
        print(f"Measured selected local {ds} HR@10={category_score:.6f}", flush=True)
    print(f"Measured selected local ensemble macro HR@10={score:.6f}; "
          f"delta versus current host 0.5763={score - 0.5763:+.6f}; "
          "grid selection guarantees local nondecrease relative to weight 0.5 "
          "on the same checkpoints and validation cases; "
          "independent host improvement remains unproven", flush=True)


if __name__ == "__main__":
    main()