from __future__ import annotations

import io
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

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
NEGATIVES = int(os.environ.get("GRAPH_NEGATIVES", "64"))
LR = float(os.environ.get("GRAPH_LR", "0.003"))
REG = float(os.environ.get("GRAPH_REG", "0.0001"))
PATIENCE = int(os.environ.get("GRAPH_PATIENCE", "35"))
TIME_LIMIT = float(os.environ.get("GRAPH_SECONDS_PER_CATEGORY", "1800"))
HISTORY_LEN = 128
HISTORY_GATE_BIAS = 2.0
ARCHITECTURE_VERSION = "candidate_history_v1"
CANDIDATE_CHUNK = int(os.environ.get("GRAPH_CANDIDATE_CHUNK", "8"))


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
    """One jointly trained LightGCN with gated, recency-aware history attention.

    History keys combine propagated item vectors and relative-position embeddings.
    The propagated user supplies the query; values remain the tied item vectors.
    Only propagated embeddings are cached, so supplied histories always affect
    attention at prediction time. No independently fitted scorer is introduced.
    """

    def __init__(self, nusers, itemnum, dim, layers, adjacency,
                 history_len=128, gate_bias=2.0):
        super().__init__()
        self.nusers = nusers
        self.itemnum = itemnum
        self.layers = layers
        self.embedding = nn.Embedding(nusers + itemnum + 1, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.item_bias = nn.Parameter(torch.zeros(itemnum + 1))
        self.history_len = int(history_len)
        self.gate_bias = float(gate_bias)
        if self.history_len < 1:
            raise ValueError("history_len must be positive")
        self.history_positions = nn.Embedding(self.history_len, dim)
        nn.init.normal_(self.history_positions.weight, std=0.02)
        self.history_key = nn.Linear(dim, dim, bias=False)
        self.history_query = nn.Linear(dim, dim, bias=False)
        self.history_attention = nn.Linear(dim, 1, bias=False)
        self.history_gate = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.history_gate.weight)
        nn.init.constant_(self.history_gate.bias, self.gate_bias)
        # Preserve every existing initialization and the subsequent training RNG.
        with torch.random.fork_rng(devices=[]):
            self.history_candidate = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.history_candidate.weight)
        self.architecture_version = ARCHITECTURE_VERSION
        self.candidate_chunk = CANDIDATE_CHUNK
        if self.candidate_chunk < 1:
            raise ValueError("candidate_chunk must be positive")
        self.register_buffer("adjacency", adjacency)
        # Only unweighted topology metadata is cached, never minibatch graphs or
        # embeddings. This buffer follows device transfers but is not serialized.
        self.register_buffer("_topology_keys", None, persistent=False)
        self._cached = None

    def _get_topology_keys(self):
        if self._topology_keys is None:
            indices = self.adjacency.to_sparse_coo().coalesce().indices()
            rows, cols = indices
            forward = (rows < self.nusers) & (cols > self.nusers)
            reverse = (cols < self.nusers) & (rows > self.nusers)
            width = self.itemnum + 1
            keys = torch.cat((
                rows[forward] * width + cols[forward] - self.nusers,
                cols[reverse] * width + rows[reverse] - self.nusers,
            ))
            self._topology_keys = torch.unique(keys, sorted=True)
        return self._topology_keys

    def _masked_adjacency(self, users, positives):
        # Remove each supervised pair once, including repeated target events.
        # Histories remain unchanged; masking is defined by the supervised pairs.
        width = self.itemnum + 1
        keys = self._get_topology_keys()
        supervised = torch.unique(users * width + positives, sorted=True)
        if supervised.numel():
            positions = torch.searchsorted(supervised, keys)
            matched = supervised[positions.clamp_max(supervised.numel() - 1)] == keys
            keys = keys[~matched]
        users = torch.div(keys, width, rounding_mode="floor")
        items = keys.remainder(width) + self.nusers
        rows = torch.cat((users, items))
        cols = torch.cat((items, users))
        nodes = self.nusers + self.itemnum + 1
        # Recompute normalization from retained, unique, unweighted edges.
        # Clamping keeps isolated nodes finite, including an empty graph.
        degree = torch.bincount(rows, minlength=nodes).to(
            dtype=self.adjacency.dtype
        ).clamp_min(1)
        values = degree[rows].rsqrt() * degree[cols].rsqrt()
        return torch.sparse_coo_tensor(
            torch.stack((rows, cols)), values, self.adjacency.shape,
            device=self.adjacency.device,
        ).coalesce().to_sparse_csr()

    def propagate(self):
        # Validation and inference always use the persistent full training graph.
        return self._propagate(self.adjacency)

    def _propagate(self, adjacency):
        x = self.embedding.weight
        result = x
        for _ in range(self.layers):
            x = torch.sparse.mm(adjacency, x)
            result = result + x
        result = result / (self.layers + 1)
        return result[: self.nusers], result[self.nusers :]

    def pack_histories(self, histories, device=None):
        """Keep the latest valid items, preserving order and repeated events.

        All training, validation and predictor inputs use right padding with zero.
        Relative positions are subsequently measured backwards from the last item.
        """
        if device is None:
            device = self.embedding.weight.device
        if isinstance(histories, torch.Tensor):
            histories = histories.detach().cpu().tolist()
        packed = torch.zeros(
            (len(histories), self.history_len), dtype=torch.long
        )
        for row, history in enumerate(histories):
            items = [int(i) for i in history if 1 <= int(i) <= self.itemnum]
            items = items[-self.history_len :]
            if items:
                packed[row, : len(items)] = torch.tensor(items, dtype=torch.long)
        return packed.to(device)

    def encode_history(self, u, item_vectors, histories=None):
        if histories is None:
            return u
        if not isinstance(histories, torch.Tensor):
            histories = self.pack_histories(histories, u.device)
        else:
            histories = histories.to(device=u.device, dtype=torch.long)
        if histories.ndim == 1:
            histories = histories.unsqueeze(0)
        if histories.ndim != 2 or histories.shape[0] != u.shape[0]:
            raise ValueError("History batch must match the user batch")
        histories = histories[:, -self.history_len :]
        if histories.shape[1] == 0:
            return u
        mask = (histories > 0) & (histories <= self.itemnum)
        indices = histories.masked_fill(~mask, 0)
        values = item_vectors[indices]
        # Position zero denotes the latest nonpadding event, independent of length.
        positions = mask.long().flip(1).cumsum(1).flip(1) - 1
        positions = positions.clamp(0, self.history_len - 1)
        keys = self.history_key(values + self.history_positions(positions))
        query = self.history_query(u).unsqueeze(1)
        logits = self.history_attention(torch.tanh(keys + query)).squeeze(-1)
        # Finite masked logits avoid NaNs for a fully padded history. Multiplying
        # and renormalizing after softmax makes its summary exactly zero.
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        h = (weights.unsqueeze(-1) * values).sum(1)
        g = torch.sigmoid(self.history_gate(torch.cat((u, h), dim=-1)))
        fused = g * u + (1.0 - g) * h
        # An empty history preserves the graph user; an unknown empty user is zero.
        return torch.where(mask.any(1, keepdim=True), fused, u)

    def _score_candidates(self, u, item_vectors, candidates, histories=None):
        """Shared differentiable scoring for training and both prediction routes.

        Stream additive attention over history positions, retaining only B*C*D
        temporaries and B*C*H logits. Checkpoint each candidate chunk during
        fitting so saved attention activations do not accumulate across chunks.
        """
        if candidates.ndim != 2 or candidates.shape[0] != u.shape[0]:
            raise ValueError("Candidate batch must match the user batch")
        if candidates.shape[1] == 0:
            return u.new_empty((u.shape[0], 0))
        if histories is None:
            return torch.bmm(
                item_vectors[candidates], u.unsqueeze(-1)
            ).squeeze(-1) + self.item_bias[candidates]
        if not isinstance(histories, torch.Tensor):
            histories = self.pack_histories(histories, u.device)
        else:
            histories = histories.to(device=u.device, dtype=torch.long)
        if histories.ndim == 1:
            histories = histories.unsqueeze(0)
        if histories.ndim != 2 or histories.shape[0] != u.shape[0]:
            raise ValueError("History batch must match the user batch")
        histories = histories[:, -self.history_len :]
        if histories.shape[1] == 0:
            return torch.bmm(
                item_vectors[candidates], u.unsqueeze(-1)
            ).squeeze(-1) + self.item_bias[candidates]
        mask = (histories > 0) & (histories <= self.itemnum)
        values = item_vectors[histories.masked_fill(~mask, 0)]
        positions = mask.long().flip(1).cumsum(1).flip(1) - 1
        positions = positions.clamp(0, self.history_len - 1)
        keys = self.history_key(values + self.history_positions(positions))
        query = self.history_query(u)

        def score_chunk(graph_u, graph_query, history_keys, history_values,
                        candidate_values, candidate_bias):
            candidate_query = (
                graph_query[:, None] + self.history_candidate(candidate_values)
            )
            logits = torch.stack([
                self.history_attention(torch.tanh(
                    history_keys[:, position, None] + candidate_query
                )).squeeze(-1)
                for position in range(history_keys.shape[1])
            ], dim=-1)
            logits = logits.masked_fill(
                ~mask[:, None], torch.finfo(logits.dtype).min
            )
            weights = torch.softmax(logits, dim=-1) * mask[:, None].to(logits.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            h = torch.bmm(weights, history_values)
            expanded_u = graph_u[:, None].expand_as(h)
            g = torch.sigmoid(self.history_gate(torch.cat((expanded_u, h), dim=-1)))
            fused = g * expanded_u + (1.0 - g) * h
            fused = torch.where(mask.any(1)[:, None, None], fused, expanded_u)
            return (fused * candidate_values).sum(-1) + candidate_bias

        scores = []
        for start in range(0, candidates.shape[1], self.candidate_chunk):
            chunk = candidates[:, start : start + self.candidate_chunk]
            args = (u, query, keys, values, item_vectors[chunk], self.item_bias[chunk])
            if self.training and torch.is_grad_enabled():
                scores.append(checkpoint(score_chunk, *args, use_reentrant=False))
            else:
                scores.append(score_chunk(*args))
        return torch.cat(scores, dim=1)

    def forward(self, users, positives, negatives, histories=None):
        if self.training:
            self._cached = None
            # All layers share this minibatch's freshly normalized graph.
            adjacency = self._masked_adjacency(users, positives)
            user_vectors, item_vectors = self._propagate(adjacency)
        else:
            user_vectors, item_vectors = self.propagate()
        candidates = torch.cat((positives[:, None], negatives), dim=1)
        logits = self._score_candidates(
            user_vectors[users], item_vectors, candidates, histories
        )
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
    def predict_indices(self, users, candidates, histories=None):
        if self._cached is None:
            self.prepare_inference()
        user_vectors, item_vectors = self._cached
        return self._score_candidates(
            user_vectors[users], item_vectors, candidates, histories
        )

    @torch.no_grad()
    def predict_history(self, history, candidates):
        # Unknown users use the same trained attention and gate with a zero query.
        if self._cached is None:
            self.prepare_inference()
        _, items = self._cached
        histories = self.pack_histories([history], items.device)
        u = items.new_zeros((1, items.shape[1]))
        return self._score_candidates(u, items, candidates[None], histories)[0]


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
def evaluate(model, val_users, val_candidates, val_histories=None):
    model.prepare_inference()
    scores = model.predict_indices(val_users, val_candidates, val_histories)
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


class SequenceCuts(Dataset):
    """Supervise every nonempty training prefix, retaining repeated occurrences.

    The target occurrence at t is never included in the encoder input. Earlier
    occurrences of the same item remain valid history. Training propagation masks
    all supervised minibatch pairs from the full training topology. Other training
    edges remain available; this is not strictly prefix-only graph causality.
    """

    def __init__(self, train, user_ids, history_len):
        self.history_len = history_len
        self.sequences = [torch.tensor(train[u], dtype=torch.long) for u in user_ids]
        self.cuts = np.asarray(
            [(j, t) for j, sequence in enumerate(self.sequences)
             for t in range(1, len(sequence))],
            dtype=np.int64,
        ).reshape(-1, 2)
        if not len(self.cuts):
            raise ValueError("No nonempty training prefixes")
        # Boundary checks verify target exclusion by position, including repeats.
        for row in (0, len(self.cuts) - 1):
            j, t = self.cuts[row]
            user, target, history = self[row]
            expected = self.sequences[j][max(0, t - history_len) : t]
            assert user == j and target == int(self.sequences[j][t])
            assert len(expected) == min(t, history_len)
            assert torch.equal(history[: len(expected)], expected)
            assert not history[len(expected) :].any()

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, index):
        user, t = self.cuts[index]
        sequence = self.sequences[user]
        prefix = sequence[max(0, t - self.history_len) : t]
        history = torch.zeros(self.history_len, dtype=torch.long)
        history[: len(prefix)] = prefix
        return int(user), int(sequence[t]), history


def check_history_fusion(model, dataset, itemnum, known_keys, device):
    """Run charged, finite-output/gradient and serialization checks before fitting.

    This operates on the single category network, then clears its gradients and
    cache. Forking the RNG prevents diagnostics from changing training randomness.
    """
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        user, target, prefix = dataset[0]
        seen = prefix[prefix > 0].tolist()
        histories = model.pack_histories([[], seen[-1:], seen], device)
        users = torch.full((3,), user, dtype=torch.long, device=device)
        positives = torch.full((3,), target, dtype=torch.long, device=device)
        negatives = sample_negatives(users, positives, itemnum, known_keys)
        model.train()
        model._cached = None
        with torch.no_grad():
            # Deterministic graph checks share the existing RNG-forked diagnostic
            # scope and category timer. They do not fit another network.
            full = model.adjacency.to_sparse_coo().coalesce()
            full_indices = full.indices().clone()
            full_values = full.values().clone()
            keys = model._get_topology_keys()
            width = itemnum + 1
            target_key = user * width + target
            if not bool((keys == target_key).any()):
                raise RuntimeError("Diagnostic target missing from full graph")
            single = model._masked_adjacency(users[:1], positives[:1])
            masked = model._masked_adjacency(users, positives)
            single = single.to_sparse_coo().coalesce()
            masked = masked.to_sparse_coo().coalesce()
            if masked._nnz() != full._nnz() - 2:
                raise RuntimeError("Duplicate pairs did not remove exactly one edge")
            if not torch.equal(single.indices(), masked.indices()):
                raise RuntimeError("Duplicate pairs changed retained topology")
            torch.testing.assert_close(
                single.values(), masked.values(), rtol=0, atol=0
            )
            rows, cols = masked.indices()
            item_node = model.nusers + target
            removed = ((rows == user) & (cols == item_node)) | (
                (rows == item_node) & (cols == user)
            )
            if bool(removed.any()):
                raise RuntimeError("Supervised edge survived in either direction")
            retained = keys[keys != target_key]
            expected_users = torch.div(retained, width, rounding_mode="floor")
            expected_items = retained.remainder(width) + model.nusers
            expected_indices = torch.cat((
                torch.stack((expected_users, expected_items)),
                torch.stack((expected_items, expected_users)),
            ), dim=1)
            nodes = model.nusers + itemnum + 1
            expected = torch.sparse_coo_tensor(
                expected_indices,
                torch.ones(expected_indices.shape[1], device=device),
                (nodes, nodes), device=device,
            ).coalesce()
            if not torch.equal(masked.indices(), expected.indices()):
                raise RuntimeError("Mask changed an unrelated training edge")
            degrees = torch.bincount(
                expected.indices()[0], minlength=nodes
            ).to(full_values.dtype).clamp_min(1)
            expected_weights = degrees[rows].rsqrt() * degrees[cols].rsqrt()
            torch.testing.assert_close(
                masked.values(), expected_weights, rtol=1e-6, atol=1e-7,
                msg="Masked graph weights do not match recomputed degrees",
            )
            # Remove every edge of this user to exercise isolated-node handling.
            incident = keys[
                torch.div(keys, width, rounding_mode="floor") == user
            ]
            isolated = model._masked_adjacency(
                torch.full_like(incident, user), incident.remainder(width)
            )
            isolated_coo = isolated.to_sparse_coo().coalesce()
            if bool((isolated_coo.indices() == user).any()):
                raise RuntimeError("Diagnostic user was not isolated")
            if not torch.isfinite(isolated_coo.values()).all():
                raise RuntimeError("Nonfinite isolated-node graph weights")
            isolated_users, isolated_items = model._propagate(isolated)
            if not (torch.isfinite(isolated_users).all()
                    and torch.isfinite(isolated_items).all()):
                raise RuntimeError("Nonfinite isolated-node propagation")
        loss = model(users, positives, negatives, histories)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite history diagnostic loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Missing/nonfinite diagnostic gradient: " + name)
        model.zero_grad(set_to_none=True)
        unchanged = model.adjacency.to_sparse_coo().coalesce()
        if not (torch.equal(unchanged.indices(), full_indices)
                and torch.equal(unchanged.values(), full_values)):
            raise RuntimeError("Training masking mutated persistent full adjacency")
        candidates = torch.cat((positives[:, None], negatives), dim=1)
        with torch.no_grad():
            model.prepare_inference()
            users_graph, items = model._cached
            encoded = model.encode_history(users_graph[users], items, histories)
            if not torch.isfinite(encoded).all():
                raise RuntimeError("Nonfinite padded/single/empty history encoding")
            if not torch.equal(encoded[0], users_graph[user]):
                raise RuntimeError("Empty history changed the graph user")
            empty_query = items.new_zeros((1, items.shape[1]))
            empty_summary = model.encode_history(empty_query, items, histories[:1])
            if torch.count_nonzero(empty_summary):
                raise RuntimeError("Fully masked history did not return zero")
            scores = model.predict_indices(users, candidates, histories)
            if not torch.isfinite(scores).all():
                raise RuntimeError("Nonfinite history diagnostic scores")
            baseline = (encoded[:, None] * items[candidates]).sum(-1)
            baseline = baseline + model.item_bias[candidates]
            torch.testing.assert_close(
                scores, baseline, rtol=1e-5, atol=1e-6,
                msg="Zero candidate projection changed baseline scores",
            )
            saved_projection = model.history_candidate.weight.detach().clone()
            saved_chunk = model.candidate_chunk
            try:
                # Use multiple distinct events: one-event attention cannot change
                # its summary when only the candidate query changes.
                probe_item = 1 if seen[-1] != 1 else 2
                if probe_item <= itemnum:
                    repeated = [seen[-1], probe_item, seen[-1]]
                    rich = model.pack_histories([repeated] * len(users), device)
                    if rich[0, :3].tolist() != repeated:
                        raise RuntimeError("History packing lost repeated events")
                    before = model.predict_indices(users, candidates, rich)
                    nn.init.normal_(model.history_candidate.weight, std=1.0)
                    after = model.predict_indices(users, candidates, rich)
                    if torch.equal(before, after):
                        raise RuntimeError("Candidate projection has no score effect")
                    # Explicit event-wise reference checks repeated-event weighting.
                    graph_u = users_graph[users]
                    vals = items[rich[:, :3]]
                    pos = torch.tensor([2, 1, 0], device=device)[None]
                    key = model.history_key(vals + model.history_positions(pos))
                    reference = []
                    for column in range(candidates.shape[1]):
                        cv = items[candidates[:, column]]
                        query = model.history_query(graph_u) + model.history_candidate(cv)
                        logits = model.history_attention(torch.tanh(
                            key + query[:, None]
                        )).squeeze(-1)
                        h = torch.bmm(logits.softmax(-1)[:, None], vals).squeeze(1)
                        gate = torch.sigmoid(model.history_gate(torch.cat((graph_u, h), -1)))
                        fused = gate * graph_u + (1.0 - gate) * h
                        reference.append((fused * cv).sum(-1)
                                         + model.item_bias[candidates[:, column]])
                    torch.testing.assert_close(
                        after, torch.stack(reference, dim=1), rtol=1e-5, atol=1e-6
                    )
                    model.candidate_chunk = 1
                    torch.testing.assert_close(
                        model.predict_indices(users, candidates, rich), after,
                        rtol=1e-5, atol=1e-6,
                    )
                    model.candidate_chunk = saved_chunk
                    permutation = torch.randperm(candidates.shape[1], device=device)
                    torch.testing.assert_close(
                        model.predict_indices(users, candidates[:, permutation], rich),
                        after[:, permutation], rtol=1e-5, atol=1e-6,
                    )
                    shifted = torch.zeros_like(rich)
                    shifted[:, -3:] = rich[:, :3]
                    torch.testing.assert_close(
                        model.predict_indices(users, candidates, shifted), after,
                        rtol=1e-5, atol=1e-6,
                    )
                    torch.testing.assert_close(
                        model.predict_indices(users, candidates, rich[:, :3]), after,
                        rtol=1e-5, atol=1e-6,
                    )
                    # Reload the same network with a nonzero projection as well.
                    probe_buffer = io.BytesIO()
                    torch.save(model.state_dict(), probe_buffer)
                    probe_buffer.seek(0)
                    model.load_state_dict(torch.load(
                        probe_buffer, map_location=device, weights_only=True
                    ), strict=True)
                    model.prepare_inference()
                    torch.testing.assert_close(
                        model.predict_indices(users, candidates, rich), after,
                        rtol=1e-5, atol=1e-6,
                    )
                    probe_buffer.close()
            finally:
                model.history_candidate.weight.copy_(saved_projection)
                model.candidate_chunk = saved_chunk
            # The same known user and candidates must respond to supplied history.
            probe_item = 1 if seen[-1] != 1 else 2
            if probe_item <= itemnum:
                probes = model.pack_histories([[seen[-1]], [probe_item]], device)
                probe_users = users[:2]
                probe_candidates = candidates[:1].expand(2, -1)
                probe_scores = model.predict_indices(
                    probe_users, probe_candidates, probes
                )
                if torch.equal(probe_scores[0], probe_scores[1]):
                    raise RuntimeError("Known-user scores ignore supplied histories")
            cold_scores = model.predict_history([], candidates[0])
            if not torch.isfinite(cold_scores).all():
                raise RuntimeError("Nonfinite unknown-user empty-history scores")
            # Serialize and strictly reload the same network, without fitting or
            # retaining another model or checkpoint.
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            model.load_state_dict(
                torch.load(buffer, map_location=device, weights_only=True), strict=True
            )
            model._topology_keys = None
            model._cached = None
            restored = model.adjacency.to_sparse_coo().coalesce()
            if not (torch.equal(restored.indices(), full_indices)
                    and torch.equal(restored.values(), full_values)):
                raise RuntimeError("Checkpoint did not retain full training adjacency")
            model.prepare_inference()
            reloaded = model.predict_indices(users, candidates, histories)
            if not torch.isfinite(reloaded).all():
                raise RuntimeError("Nonfinite checkpoint reload predictions")
            torch.testing.assert_close(
                reloaded, scores, rtol=1e-5, atol=1e-6,
                msg="Checkpoint reload prediction mismatch",
            )
            buffer.close()
        model._cached = None
        model.train()


def train_category(name, device):
    started = time.monotonic()
    seed_all(42)
    train, valid, itemnum = load_data(name)
    user_ids = sorted(u for u in train if train[u])
    mapping, pairs, graph = build_graph(train, user_ids, itemnum)
    model = GraphRecommender(
        len(user_ids), itemnum, DIM, LAYERS, graph.to(device),
        history_len=HISTORY_LEN, gate_bias=HISTORY_GATE_BIAS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    keys = pairs[:, 0] * (itemnum + 1) + pairs[:, 1]
    known_keys = torch.as_tensor(np.sort(keys), device=device)
    vu, vc = validation_cases(train, valid, mapping, itemnum)
    vh = model.pack_histories([train[user_ids[j]] for j in vu], device)
    vu = torch.tensor(vu, dtype=torch.long, device=device)
    vc = torch.tensor(vc, dtype=torch.long, device=device)
    sequence_dataset = SequenceCuts(train, user_ids, model.history_len)
    check_history_fusion(model, sequence_dataset, itemnum, known_keys, device)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        sequence_dataset,
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
    # The category timer includes data preparation and diagnostics above.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model._cached = None
        total_loss = 0.0
        for _ in range(STEPS):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            users, positives, histories = (
                value.to(device, non_blocking=True) for value in batch
            )
            negatives = sample_negatives(users, positives, itemnum, known_keys)
            optimizer.zero_grad(set_to_none=True)
            model._cached = None
            loss = model(users, positives, negatives, histories)
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
            hr, ndcg = evaluate(model, vu, vc, vh)
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
        "history_len": model.history_len,
        "gate_bias": model.gate_bias,
        "best_epoch": best_epoch,
        "architecture_version": model.architecture_version,
        "candidate_chunk": model.candidate_chunk,
    }
    with open(os.path.join(WORK, name + "_meta.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    # Report the actual selected checkpoint's metric.
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True), strict=True
    )
    selected_hr, selected_ndcg = evaluate(model, vu, vc, vh)
    print(
        f"{name} selected_epoch={best_epoch} "
        f"selected_validation_HR@10={selected_hr:.5f} "
        f"selected_validation_NDCG@10={selected_ndcg:.5f}",
        flush=True,
    )
    del iterator, loader, sequence_dataset, model, optimizer, graph, known_keys
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
    if meta['architecture_version'] != 'candidate_history_v1':
        raise ValueError('Unsupported checkpoint architecture')
    if int(meta['candidate_chunk']) < 1:
        raise ValueError('Invalid checkpoint candidate chunk')
    state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                       map_location=_DEVICE, weights_only=True)
    model = _CLASS(len(meta['user_ids']), meta['itemnum'], meta['dim'],
                   meta['layers'], state['adjacency'],
                   history_len=meta['history_len'],
                   gate_bias=meta['gate_bias']).to(_DEVICE)
    # Reconstruct exactly one category network and strictly restore its full
    # training adjacency. Minibatch masks are training-only, nonpersistent data.
    model.candidate_chunk = int(meta['candidate_chunk'])
    model.load_state_dict(state, strict=True)
    model._topology_keys = None
    model._cached = None
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
        # Both routes apply the architecture's identical filtering, truncation
        # and padding. Attention is recomputed for every supplied history.
        valid_history = [int(i) for i in history
                         if 1 <= int(i) <= model.itemnum]
        valid_history = valid_history[-model.history_len:]
        index = _MAPS[dataset].get(int(user_id))
        if index is not None:
            users = torch.tensor([index], dtype=torch.long, device=_DEVICE)
            hist = model.pack_histories([valid_history], _DEVICE)
            scores = model.predict_indices(users, cand[None], hist)[0]
        else:
            scores = model.predict_history(valid_history, cand)
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
    print(f"Validation HR@10 by category: {dict(zip(DATASETS, scores))}")
    print(f"Validation macro HR@10 change versus supplied 0.6225: {score - 0.6225:+.5f}")
    print("Historical 0.6275 is a reference only; independent host validation determines improvement.")
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        details = traceback.format_exc()
        print(details, flush=True)
        os.makedirs(WORK, exist_ok=True)
        with open(os.path.join(WORK, "training_error.txt"), "w") as f:
            f.write(details)
        raise