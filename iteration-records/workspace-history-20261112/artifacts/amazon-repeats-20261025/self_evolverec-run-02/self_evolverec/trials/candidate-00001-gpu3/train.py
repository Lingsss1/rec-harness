"""Per-dataset SASRec — trains one model per Amazon Reviews dataset.

Architecture (aligned with LLM-SRec SeqRec/sasrec, adapted for the RecHarness raw .txt data format):
  - One SASRec model per dataset: Movies_and_TV, Industrial_and_Scientific,
    Electronics, CDs_and_Vinyl
  - Checkpoints saved to ./working/{dataset}_model.pt
  - Official SASRec positional encoding: positions 0..maxlen-1 with padding masked after dropout
  - Official PointWiseFeedForward: Conv1d → Dropout → ReLU → Conv1d → Dropout + residual
  - Official BCEWithLogits loss on positive and negative sequence logits
  - Early stopping with patience=20 (checked every 5 epochs)
  - Writes predict.py routing to per-dataset models via dataset_name kwarg

Key hyper-parameters (all tunable by RecHarness):
  HIDDEN_UNITS : embedding / attention dimension
  NUM_BLOCKS   : number of self-attention blocks
  NUM_HEADS    : number of attention heads
  DROPOUT_RATE : dropout on attention + FFN
  LR           : Adam learning rate
  WEIGHT_DECAY : Adam weight decay
  BATCH_SIZE   : training batch size
  NUM_BATCHES  : gradient steps per epoch
  NUM_EPOCHS   : max training epochs
  MAXLEN       : max history length
  PATIENCE     : early stopping patience (epochs without val improvement)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# ── Hyper-parameters ──────────────────────────────────────────────────
HIDDEN_UNITS = int(os.environ.get("SASREC_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("SASREC_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("SASREC_HEADS", "1"))
DROPOUT_RATE = float(os.environ.get("SASREC_DROPOUT", "0.5"))
LR = float(os.environ.get("SASREC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("SASREC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("SASREC_BATCH", "128"))
NUM_BATCHES = int(os.environ.get("SASREC_BATCHES", "200"))
### >>> Self_EvolveRec-BLOCK-START: Preserve the matched training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Require the host-provided dataset directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Data loading ──────────────────────────────────────────────────────


def _read_split(path: str) -> dict[int, list[int]]:
    d: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                d[int(parts[0])].append(int(parts[1]))
    return dict(d)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return stats[dataset_name]["itemnum"]
    # fallback: scan train file
    itemnum = 0
    path = os.path.join(data_dir, f"{dataset_name}_train.txt")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def load_dataset(dataset_name: str, data_dir: str):
    user_train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    user_valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    itemnum = _load_itemnum(data_dir, dataset_name)
    return user_train, user_valid, itemnum


# ── Batch sampler ─────────────────────────────────────────────────────


def sample_batch(user_train: dict, itemnum: int, batch_size: int, maxlen: int):
    users = [u for u, items in user_train.items() if len(items) >= 2]
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        u = random.choice(users)
        history = user_train[u]
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in ts:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.array(batch_seq), np.array(batch_pos), np.array(batch_neg)


# ── Model ─────────────────────────────────────────────────────────────


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.dropout1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum

        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.item_emb.weight.data.normal_(0.0, 1.0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)

        pos = torch.arange(M, device=seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_emb(pos)
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = seq == 0
        seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)
        attention_mask = ~torch.tril(
            torch.ones((M, M), dtype=torch.bool, device=seq.device)
        )

        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            queries = self.attention_layernorms[i](seq_emb)
            attn_out, _ = self.attention_layers[i](
                queries, seq_emb, seq_emb, attn_mask=attention_mask
            )
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        return self.last_layernorm(seq_emb)

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        log_feats = self.log2feats(seq)
        if pos is None or neg is None:
            return log_feats
        pos_embs = self.item_emb(pos)
        neg_embs = self.item_emb(neg)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(candidates)
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)


### >>> Self_EvolveRec-BLOCK-START: Frozen directed transitions with vectorized causal context and CSR inference
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def build_transition_graph(user_train, itemnum):
    from collections import Counter

    if itemnum < 1 or itemnum > math.isqrt(np.iinfo(np.int64).max) - 1:
        raise ValueError("Item count cannot be encoded safely in int64 edge keys")
    rows = defaultdict(Counter)
    frequency = np.zeros(itemnum + 1, dtype=np.int64)
    for history in user_train.values():
        for item in history:
            if item != 0:
                if not 1 <= item <= itemnum:
                    raise ValueError("Training item outside catalog")
                frequency[item] += 1
        for source, destination in zip(history, history[1:]):
            if source != 0 and destination != 0:
                rows[source][destination] += 1

    keys, values, destinations = [], [], []
    row_ptr = [0]
    outgoing = np.zeros(itemnum + 1, dtype=np.int64)
    for source in range(itemnum + 1):
        counts = rows.get(source, {})
        outgoing[source] = sum(counts.values())
        retained = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:64]
        # Destination ordering makes both encoded keys and CSR rows sorted.
        for destination, count in sorted(retained):
            keys.append(source * (itemnum + 1) + destination)
            destinations.append(destination)
            values.append(
                math.log1p(100.0 * count / (outgoing[source] + 10.0)) / math.log(101.0)
            )
        row_ptr.append(len(keys))

    popularity = np.zeros(itemnum + 1, dtype=np.float32)
    probabilities = (frequency[1:].astype(np.float64) + 1.0) / (
        frequency.sum() + itemnum
    )
    popularity[1:] = np.log1p(100.0 * probabilities) / math.log(101.0)
    return {
        "keys": torch.tensor(keys, dtype=torch.long),
        "values": torch.tensor(values, dtype=torch.float32),
        "row_ptr": torch.tensor(row_ptr, dtype=torch.long),
        "destinations": torch.tensor(destinations, dtype=torch.long),
        "frequency": torch.from_numpy(frequency),
        "outgoing": torch.from_numpy(outgoing),
        "popularity": torch.from_numpy(popularity),
    }


class FrozenTransition(nn.Module):
    """Frozen backbone and empirical transition residual; no fitted graph parameters."""

    def __init__(self, backbone, graph, alpha=0.0):
        super().__init__()
        self.backbone = backbone
        self.itemnum = backbone.itemnum
        self.maxlen = backbone.maxlen
        self.alpha = float(alpha)
        for name, value in graph.items():
            self.register_buffer(name, value)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        # Freezing parameters alone does not disable backbone dropout.
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def item_emb(self):
        return self.backbone.item_emb

    def _lookup(self, sources, items):
        sources, items = torch.broadcast_tensors(sources, items)
        result = torch.zeros(
            sources.shape, device=sources.device, dtype=self.values.dtype
        )
        if self.keys.numel() == 0 or sources.numel() == 0:
            return result
        query = sources * (self.itemnum + 1) + items
        indices = torch.searchsorted(self.keys, query.contiguous())
        safe = indices.clamp(max=self.keys.numel() - 1)
        valid = (
            (indices < self.keys.numel())
            & (self.keys[safe] == query)
            & (sources > 0)
            & (sources <= self.itemnum)
            & (items > 0)
            & (items <= self.itemnum)
        )
        return torch.where(valid, self.values[safe], result)

    def _recent_sources(self, sequence):
        # Compact nonpadding events, then gather five causal event ranks.
        # Padding contributes zero, including padding between real events.
        batch, length = sequence.shape
        ranks = (sequence != 0).long().cumsum(dim=1)
        compact = sequence.new_zeros((batch, length + 1))
        compact.scatter_add_(1, ranks, sequence)
        requested = ranks.unsqueeze(-1) - torch.arange(5, device=sequence.device)
        return compact.gather(
            1, requested.clamp_min(0).reshape(batch, length * 5)
        ).reshape(batch, length, 5)

    def _from_sources(self, sources, items):
        # sources: (..., 5), items: (..., candidates)
        weights = torch.exp(
            -torch.arange(5, device=sources.device, dtype=self.values.dtype)
        )
        weights = weights * (sources != 0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        edge_values = self._lookup(sources.unsqueeze(-1), items.unsqueeze(-2))
        return (weights.unsqueeze(-1) * edge_values).sum(dim=-2)

    def transition_score(self, sequence, candidates):
        if sequence.shape[1] == 0:
            return torch.zeros_like(candidates, dtype=self.values.dtype)
        return self._from_sources(self._recent_sources(sequence)[:, -1], candidates)

    def log2feats(self, sequence):
        features = self.backbone.log2feats(sequence)
        features._transition_sources = self._recent_sources(sequence)
        return features

    def predict_score(self, features, items, sequence=None):
        # Slicing/copying tensors drops Python attributes. Callers can explicitly
        # supply the corresponding sequence when using sliced feature batches.
        if sequence is not None:
            sources = self._recent_sources(sequence)
            if features.ndim == 2:
                if sequence.shape[1] == 0:
                    raise ValueError(
                        "Final-position features require a nonempty sequence"
                    )
                sources = sources[:, -1]
        else:
            sources = getattr(features, "_transition_sources", None)
        if sources is None:
            raise ValueError(
                "Sequence context missing: pass sequence explicitly for sliced features"
            )
        if sources.shape[:-1] != features.shape[:-1]:
            raise ValueError(
                "Feature positions and causal sequence context do not match"
            )
        items = torch.as_tensor(items, device=features.device, dtype=torch.long)
        if items.shape == features.shape[:-1]:
            base = (features * self.item_emb(items)).sum(dim=-1)
            residual = self._from_sources(sources, items.unsqueeze(-1)).squeeze(-1)
        elif items.ndim == features.ndim and items.shape[:-1] == features.shape[:-1]:
            base = (features.unsqueeze(-2) * self.item_emb(items)).sum(dim=-1)
            residual = self._from_sources(sources, items)
        else:
            raise ValueError("Expected items shaped B x L or B x L x candidates")
        return base + self.alpha * residual

    def forward(self, sequence, pos=None, neg=None):
        features = self.log2feats(sequence)
        if pos is None or neg is None:
            return features
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def predict(self, sequence, candidates):
        base = self.backbone.predict(sequence, candidates)
        if self.alpha == 0.0:
            return base
        return base + self.alpha * self.transition_score(sequence, candidates)

    def predict_full(self, sequence, chunk_size=4096):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        features = self.backbone.log2feats(sequence)[:, -1]
        chunks = []
        for start in range(1, self.itemnum + 1, chunk_size):
            items = torch.arange(
                start, min(start + chunk_size, self.itemnum + 1), device=sequence.device
            )
            candidates = items.unsqueeze(0).expand(sequence.shape[0], -1)
            chunks.append(
                self.item_emb(candidates).matmul(features.unsqueeze(-1)).squeeze(-1)
            )
        scores = torch.cat(chunks, dim=1)
        if self.alpha == 0.0 or self.values.numel() == 0:
            return scores

        # Each of five sources has at most 64 retained edges. Scatter only
        # those edges instead of searching five times for every catalog item.
        sources = self._recent_sources(sequence)[:, -1]
        weights = torch.exp(
            -torch.arange(5, device=sequence.device, dtype=self.values.dtype)
        ) * (sources != 0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        starts = self.row_ptr[sources]
        stops = self.row_ptr[sources + 1]
        indices = starts.unsqueeze(-1) + torch.arange(64, device=sequence.device)
        valid = (indices < stops.unsqueeze(-1)) & (sources.unsqueeze(-1) != 0)
        safe = indices.clamp(max=self.values.numel() - 1)
        destinations = self.destinations[safe] - 1
        contributions = torch.where(valid, self.values[safe], 0.0)
        contributions = contributions * weights.unsqueeze(-1)
        residual = torch.zeros_like(scores)
        residual.scatter_add_(
            1,
            destinations.reshape(sequence.shape[0], -1),
            contributions.reshape(sequence.shape[0], -1),
        )
        return scores + self.alpha * residual


def verify_transition_scoring(model, device):
    length = model.maxlen
    sequence = torch.zeros((4, length), dtype=torch.long, device=device)
    nonempty = torch.nonzero(model.row_ptr[1:] > model.row_ptr[:-1]).flatten()
    source = int(nonempty[0]) if nonempty.numel() else 1
    sequence[1, -1] = source
    sequence[2, -min(3, length) :] = source
    sequence[3, -1] = model.itemnum
    candidates = torch.tensor([[0, 1, model.itemnum]] * 4, device=device)
    if nonempty.numel():
        destination = int(model.destinations[int(model.row_ptr[source])])
        candidates[:, 1] = destination
    old_alpha = model.alpha
    model.alpha = 1.0
    model.eval()
    with torch.no_grad():
        candidate_scores = model.predict(sequence, candidates)
        # Compare against the same items indexed out of a complete score vector.
        full = model.predict_full(sequence)
        real = candidates.clamp_min(1)
        indexed = full.gather(1, real - 1)
        torch.testing.assert_close(
            candidate_scores[candidates > 0],
            indexed[candidates > 0],
            rtol=1e-5,
            atol=1e-5,
        )
        features = model.log2feats(sequence)
        position_items = candidates[:, 1:2].expand(-1, length)
        position_scores = model.predict_score(features, position_items)
        torch.testing.assert_close(
            position_scores[:, -1], candidate_scores[:, 1], rtol=1e-5, atol=1e-5
        )
        assert (
            torch.count_nonzero(model.transition_score(sequence[:1], candidates[:1]))
            == 0
        )
        rebound_scores = model.predict_score(
            features[:, -1], candidates, sequence=sequence
        )
        torch.testing.assert_close(
            rebound_scores, candidate_scores, rtol=1e-5, atol=1e-5
        )
        # Compare vectorized context against the literal causal recurrence,
        # including repeated events and padding between events.
        probe = torch.tensor(
            [[0, source, 0, source, 1, 0, source], [0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
            device=device,
        )
        expected = torch.zeros((2, 5), dtype=torch.long, device=device)
        contexts = []
        for position in range(probe.shape[1]):
            current = probe[:, position]
            updated = torch.cat((current[:, None], expected[:, :4]), dim=1)
            expected = torch.where(current[:, None] != 0, updated, expected)
            contexts.append(expected)
        torch.testing.assert_close(
            model._recent_sources(probe), torch.stack(contexts, dim=1)
        )
        model.train()
        assert not model.backbone.training
        model.eval()
    model.alpha = old_alpha


def select_shared_transition_alpha(working_dir, device):
    import time

    started = time.monotonic()
    modes = [("baseline", 0.0), ("popularity_only", 1.0), ("transition_only", 1.0)]
    modes += [("sasrec_popularity", alpha) for alpha in ALPHAS]
    modes += [("sasrec_transition", alpha) for alpha in ALPHAS]
    results = {mode: {} for mode in modes}
    paired_results = {mode: {} for mode in modes}
    coverage = {}

    for dataset_name in DATASETS:
        user_train, user_valid, itemnum = load_dataset(dataset_name, DATA_DIR)
        with open(
            os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "rb"
        ) as handle:
            meta = pickle.load(handle)
        backbone = SASRec(**meta).to(device)
        checkpoint = os.path.join(working_dir, f"{dataset_name}_model.pt")
        backbone.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        graph = build_transition_graph(user_train, itemnum)
        torch.save(graph, os.path.join(working_dir, f"{dataset_name}_transition.pt"))
        model = FrozenTransition(backbone, graph).to(device).eval()
        verify_transition_scoring(model, device)
        records = []

        class ValidationRecorder:
            def eval(self):
                model.eval()

            def train(self):
                # The baseline checkpoint remains frozen throughout all controls.
                model.eval()

            def predict(self, sequence, candidates):
                base = model.backbone.predict(sequence, candidates)
                transition = model.transition_score(sequence, candidates)
                popularity = model.popularity[candidates]
                records.append(
                    (
                        base[0].cpu().numpy(),
                        transition[0].cpu().numpy(),
                        popularity[0].cpu().numpy(),
                    )
                )
                return base

        # Use the existing local validation implementation, not a reimplementation
        # of or modification to the independent host evaluator. Cache its scores
        # once so every control receives identical users, candidates and tie order.
        rng_state = random.getstate()
        try:
            random.seed(42)
            evaluate_val(
                ValidationRecorder(), user_train, user_valid, itemnum, device, MAXLEN
            )
        finally:
            random.setstate(rng_state)
        if not records:
            raise RuntimeError(f"No validation records for {dataset_name}")

        coverage[dataset_name] = {
            "users": len(records),
            "target_transition_coverage": float(
                np.mean([transition[0] > 0 for _, transition, _ in records])
            ),
            "candidate_transition_coverage": float(
                np.mean([np.mean(transition > 0) for _, transition, _ in records])
            ),
            "mean_candidate_baseline_logit_std": float(
                np.mean([np.std(base) for base, _, _ in records])
            ),
            "mean_candidate_transition_std": float(
                np.mean([np.std(transition) for _, transition, _ in records])
            ),
        }
        baseline_hits = None
        for mode, alpha in modes:
            hits = []
            for base, transition, popularity in records:
                if mode == "baseline":
                    scores = base
                elif mode == "popularity_only":
                    scores = popularity
                elif mode == "transition_only":
                    scores = transition
                elif mode == "sasrec_popularity":
                    scores = base + alpha * popularity
                else:
                    scores = base + alpha * transition
                # Same stable descending ranking as evaluate_val, including ties.
                rank = sorted(
                    range(len(scores)), key=lambda index: -float(scores[index])
                ).index(0)
                hits.append(rank < 10)
            hits = np.asarray(hits, dtype=np.bool_)
            if mode == "baseline":
                baseline_hits = hits.copy()
            results[(mode, alpha)][dataset_name] = float(np.mean(hits))
            paired_results[(mode, alpha)][dataset_name] = {
                "wins_vs_baseline": int(np.sum(hits & ~baseline_hits)),
                "losses_vs_baseline": int(np.sum(~hits & baseline_hits)),
                "unchanged_hits": int(np.sum(hits & baseline_hits)),
                "unchanged_misses": int(np.sum(~hits & ~baseline_hits)),
            }
        del model, backbone

    attempts = []
    for mode, alpha in modes:
        per_category = results[(mode, alpha)]
        attempts.append(
            {
                "mode": mode,
                "alpha": alpha,
                "per_category_HR@10": per_category,
                "paired_local_validation": paired_results[(mode, alpha)],
                "macro_HR@10": sum(per_category.values()) / len(DATASETS),
            }
        )
    eligible = [
        attempt for attempt in attempts if attempt["mode"] == "sasrec_transition"
    ]
    selected = max(
        eligible, key=lambda attempt: (attempt["macro_HR@10"], -attempt["alpha"])
    )
    report = {
        "alpha": selected["alpha"],
        "constants": {
            "recent_items": 5,
            "top_edges": 64,
            "shrinkage": 10,
            "feature_scale": 100,
        },
        "validation_source": "existing local evaluate_val; independent host metrics are not available here",
        "selection_metric": "equal-weight category macro validation HR@10",
        "attempts": attempts,
        "coverage": coverage,
        "fusion_wall_seconds": time.monotonic() - started,
        "diagnostic_limitations": {
            "full_graph_training_margins": "Target-reusing descriptive measurements only",
            "leave_user_out_margins": "Unavailable here: no user-attributing diagnosis implementation was supplied",
            "metadata_categories_and_simulator_traits": "Unobserved",
            "uncertainty": "No host per-user validation outputs were supplied",
        },
    }
    with open(os.path.join(working_dir, "transition_fusion.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)
    return report


### <<< Self_EvolveRec-BLOCK-END

# ── Validation ────────────────────────────────────────────────────────


def evaluate_val(
    model: SASRec,
    user_train: dict,
    user_valid: dict,
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    num_users: int = 200,
) -> dict[str, float]:
    val_users = [u for u in user_valid if u in user_train]
    if len(val_users) > num_users:
        random.seed(42)
        val_users = random.sample(val_users, num_users)

    HR10 = HR20 = NDCG10 = NDCG20 = 0.0
    model.eval()
    with torch.no_grad():
        for u in val_users:
            pos_items = user_valid[u]
            if not pos_items:
                continue
            target = pos_items[0]

            hist = user_train[u][-maxlen:]
            seq_arr = np.zeros(maxlen, dtype=np.int32)
            for i, item in enumerate(reversed(hist)):
                seq_arr[maxlen - 1 - i] = item

            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            neg_pool: set[int] = set()
            interacted = set(user_train[u]) | set(pos_items)
            while len(neg_pool) < 99:
                ni = random.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            scores = model.predict(seq_t, cand_t)[0].cpu().numpy().tolist()
            rank = sorted(range(len(candidates)), key=lambda i: -scores[i]).index(0)

            HR10 += 1.0 if rank < 10 else 0.0
            HR20 += 1.0 if rank < 20 else 0.0
            NDCG10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            NDCG20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0

    n = max(len(val_users), 1)
    model.train()
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always save the first evaluated checkpoint
    best_hr10 = float("-inf")
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
            indices = pos_t != 0
            pos_labels = torch.ones(pos_logits[indices].shape, device=device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=device)
            loss = bce_criterion(pos_logits[indices], pos_labels)
            loss = loss + bce_criterion(neg_logits[indices], neg_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        if epoch % 5 == 0:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}"
                f"  loss={avg_loss:.4f}  val_score: {hr10:.4f}"
                f"  HR@20={val_m['HR@20']:.4f}  NDCG@10={val_m['NDCG@10']:.4f}",
                flush=True,
            )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            if epoch - best_epoch >= PATIENCE * 5:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        else:
            if epoch % 10 == 0:
                print(
                    f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                    flush=True,
                )

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
            },
            f,
        )

    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10


# ── Write unified predict.py ──────────────────────────────────────────


def write_predict_script(working_dir: str) -> None:
    script = '''"""Per-dataset SASRec predict — routes by dataset_name kwarg.

Supports both legacy signature:
    predict(user_id, history, candidates) -> list[float]
and per-dataset signature (used by the RecHarness harness when available):
    predict(user_id, history, candidates, *, dataset_name) -> list[float]
"""
from __future__ import annotations

import importlib.util
import os
import pickle

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

_models: dict = {}
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ### >>> Self_EvolveRec-BLOCK-START: Load the trained transition wrapper implementation
    return mod
    ### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### >>> Self_EvolveRec-BLOCK-START: Restore the shared coefficient and graph architecture
    import json
    implementation = _load_model_class()
    SASRec = implementation.SASRec
    with open(os.path.join(_HERE, "transition_fusion.json")) as handle:
        fusion = json.load(handle)
    for ds in DATASETS:
    ### <<< Self_EvolveRec-BLOCK-END
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on missing category artifacts
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing required model artifacts for {ds}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Deploy identical frozen SASRec and transition scoring
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        graph_path = os.path.join(_HERE, f"{ds}_transition.pt")
        graph = torch.load(graph_path, map_location=_device, weights_only=True)
        m = implementation.FrozenTransition(m, graph, alpha=fusion["alpha"]).to(_device)
        m.eval()
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject explicit unknown category routes
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"Unknown or unavailable dataset route: {dataset_name}")
    ### <<< Self_EvolveRec-BLOCK-END
    # Preserve the legacy fallback only when dataset_name is omitted.
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
    elif _models:
        ds    = next(iter(_models))
        model = _models[ds]
        meta  = _metas[ds]
    else:
        return [0.0] * len(candidates)

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    with torch.no_grad():
        cand_t   = torch.tensor([candidates], dtype=torch.long, device=_device)
        scores   = model.predict(seq_t, cand_t)[0].cpu().tolist()

    return scores
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written → {predict_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)

    all_hr10: list[float] = []
    for ds in DATASETS:
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        hr10 = train_one_dataset(ds, device, DATA_DIR)
        all_hr10.append(hr10)

    ### >>> Self_EvolveRec-BLOCK-START: Report the selected shared fusion validation result
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError("This experiment requires all four benchmark categories")
    fusion = select_shared_transition_alpha(working_dir, device)
    write_predict_script(working_dir)
    selected = next(
        attempt
        for attempt in fusion["attempts"]
        if attempt["mode"] == "sasrec_transition"
        and attempt["alpha"] == fusion["alpha"]
    )
    avg = selected["macro_HR@10"]
    # Review checks: traced graph lookup, causal padding/repetition handling,
    # frozen dropout, CSR/candidate equivalence, routing, and predictor loading.
    # Runtime assertions above execute on the host; no training was run here.
    # Host validation equivalence and DiagnosisProbe integration remain
    # unverified because their implementations were not supplied. Local paired
    # counts describe local validation only; they are not host uncertainty data.
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


