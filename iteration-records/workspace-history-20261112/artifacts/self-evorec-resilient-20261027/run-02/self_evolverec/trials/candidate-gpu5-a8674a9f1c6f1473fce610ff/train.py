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
### >>> Self_EvolveRec-BLOCK-START: Preserve host epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Preserve required input boundary
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all category models and declare fusion controls
_requested_datasets = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]
if set(_requested_datasets) != set(DEFAULT_DATASETS):
    raise ValueError("All four matched Amazon datasets are required")
DATASETS = DEFAULT_DATASETS.copy()
RECENT_WINDOW = int(os.environ.get("SASREC_RECENT_WINDOW", "10"))
FUSION_MODE = os.environ.get("SASREC_FUSION_MODE", "context")
FUSION_CHUNK = int(os.environ.get("SASREC_FUSION_CHUNK", "512"))
if FUSION_CHUNK < 1:
    raise ValueError("SASREC_FUSION_CHUNK must be positive")

random.seed(42)
### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Reject impossible negative sampling
    users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    if not users:
        raise ValueError("No training users support next-item negative sampling")
    batch_seq, batch_pos, batch_neg = [], [], []
    ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Persist recent-branch configuration
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_window: int = 10,
        fusion_mode: str = "context",
        fusion_chunk: int = 512,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
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

        ### >>> Self_EvolveRec-BLOCK-START: Shared encoder with zero-initialized context gate
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if not 1 <= recent_window <= maxlen:
            raise ValueError("recent_window must lie between one and maxlen")
        if fusion_mode not in {
            "baseline",
            "normalized_full",
            "recent",
            "fixed25",
            "fixed50",
            "fixed75",
            "global",
            "context",
        }:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        if fusion_chunk < 1:
            raise ValueError("fusion_chunk must be positive")
        self.recent_window = recent_window
        self.fusion_mode = fusion_mode
        self.fusion_chunk = fusion_chunk
        self.context_gate = nn.Sequential(
            nn.Linear(2 * hidden_units + 1, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, 1),
        )
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)
        self.global_gate = nn.Parameter(torch.zeros(()))

    def _encode(self, seq: torch.Tensor) -> torch.Tensor:
        ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unused attention-weight materialization
            attn_out, _ = self.attention_layers[i](
                queries,
                seq_emb,
                seq_emb,
                attn_mask=attention_mask,
                need_weights=False,
            )
            ### <<< Self_EvolveRec-BLOCK-END
            seq_emb = queries + attn_out
            seq_emb = torch.transpose(seq_emb, 0, 1)

            seq_emb = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)

        ### >>> Self_EvolveRec-BLOCK-START: Causal recent-window fusion and checkpointed scalar losses
        return self.last_layernorm(seq_emb).masked_fill(timeline_mask.unsqueeze(-1), 0)

    @staticmethod
    def _normalize(features):
        return features * torch.rsqrt(features.square().mean(-1, keepdim=True) + 1e-6)

    def _window_state(self, sequence):
        # Compute once per sequence batch, rather than once per endpoint chunk.
        batch, length = sequence.shape
        ranks = sequence.ne(0).long().cumsum(1)
        compressed = sequence.new_zeros(batch, length + 1)
        compressed.scatter_add_(1, ranks, sequence)
        return ranks, compressed

    def recent_windows(self, sequence, endpoints, window_state=None):
        if window_state is None:
            window_state = self._window_state(sequence)
        ranks, compressed = window_state
        b, t = endpoints.unbind(1)
        lengths = ranks[b, t]
        offsets = torch.arange(self.recent_window - 1, -1, -1, device=sequence.device)
        indices = lengths[:, None] - offsets
        windows = compressed[b[:, None], indices.clamp_min(0)]
        windows = windows.masked_fill(indices <= 0, 0)
        return windows, lengths

    def fuse(self, full, recent, lengths):
        z_full = self._normalize(full)
        z_recent = self._normalize(recent)
        mode = self.fusion_mode
        if mode == "baseline":
            gate = full.new_zeros(full.shape[0], 1)
            return full, gate
        if mode == "normalized_full":
            gate = full.new_zeros(full.shape[0], 1)
        elif mode == "recent":
            gate = full.new_ones(full.shape[0], 1)
        elif mode.startswith("fixed"):
            gate = full.new_full((full.shape[0], 1), int(mode[5:]) / 100.0)
        elif mode == "global":
            gate = self.global_gate.sigmoid().expand(full.shape[0], 1)
        else:
            inputs = torch.cat(
                (z_full, z_recent, lengths.to(full.dtype).log1p().unsqueeze(-1)),
                dim=-1,
            )
            gate = self.context_gate(inputs).sigmoid()
        return (1.0 - gate) * z_full + gate * z_recent, gate

    def _endpoint_features(self, full, windows, lengths):
        if self.fusion_mode in {"baseline", "normalized_full"}:
            recent = torch.zeros_like(full)
        else:
            recent = self._encode(windows)[:, -1]
        return self.fuse(full, recent, lengths)[0]

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        full = self._encode(seq)
        if self.fusion_mode == "baseline":
            return full
        if self.fusion_mode == "normalized_full":
            return self._normalize(full)
        locations = seq.ne(0).nonzero(as_tuple=False)
        result = torch.zeros_like(full)
        window_state = self._window_state(seq)
        for endpoints in locations.split(self.fusion_chunk):
            if not endpoints.numel():
                continue
            b, t = endpoints.unbind(1)
            windows, lengths = self.recent_windows(seq, endpoints, window_state)
            result[b, t] = self._endpoint_features(full[b, t], windows, lengths)
        return result

    def final_features(self, seq):
        # Rightmost observed endpoint also supports empty or internally padded input.
        positions = torch.arange(seq.shape[1], device=seq.device)
        last = torch.where(seq.ne(0), positions, -1).amax(1)
        endpoints = torch.stack(
            (torch.arange(seq.shape[0], device=seq.device), last.clamp_min(0)), dim=1
        )
        full = self._encode(seq)[endpoints[:, 0], endpoints[:, 1]]
        windows, lengths = self.recent_windows(seq, endpoints)
        fused = self._endpoint_features(full, windows, lengths)
        return fused.masked_fill((last < 0).unsqueeze(-1), 0)

    def predict_score(self, features, items):
        # Matching IDs: features (..., D), items (...).
        # Candidate IDs: features (..., D), items (..., K), or shared IDs (K,).
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        leading = features.shape[:-1]
        if tuple(items.shape) == tuple(leading):
            return (features * self.item_emb(items)).sum(-1)
        if items.ndim == 1:
            return features.matmul(self.item_emb(items).transpose(0, 1))
        if tuple(items.shape[:-1]) != tuple(leading):
            raise ValueError("Candidate dimensions must match feature positions")
        return (features.unsqueeze(-2) * self.item_emb(items)).sum(-1)

    def _chunk_loss(self, windows, full, lengths, positive, negative):
        features = self._endpoint_features(full, windows, lengths)
        positive_scores = self.predict_score(features, positive)
        negative_scores = self.predict_score(features, negative)
        # Same sum of positive and negative BCE means as the original baseline.
        return (
            torch.nn.functional.softplus(-positive_scores)
            + torch.nn.functional.softplus(negative_scores)
        ).sum()

    def training_loss(self, sequence, positive, negative):
        from torch.utils.checkpoint import checkpoint

        full = self._encode(sequence)
        locations = (sequence.ne(0) & positive.ne(0)).nonzero(as_tuple=False)
        if not locations.numel():
            raise ValueError("Training batch has no supervised positions")
        if self.fusion_mode in {"baseline", "normalized_full"}:
            # Preserve the original BCE weighting without checkpoint overhead
            # for controls that have no recent encoder computation.
            features = full[locations[:, 0], locations[:, 1]]
            if self.fusion_mode == "normalized_full":
                features = self._normalize(features)
            positive_scores = self.predict_score(
                features, positive[locations[:, 0], locations[:, 1]]
            )
            negative_scores = self.predict_score(
                features, negative[locations[:, 0], locations[:, 1]]
            )
            return (
                torch.nn.functional.softplus(-positive_scores)
                + torch.nn.functional.softplus(negative_scores)
            ).mean()
        loss = full.sum() * 0.0
        window_state = self._window_state(sequence)
        for endpoints in locations.split(self.fusion_chunk):
            b, t = endpoints.unbind(1)
            windows, lengths = self.recent_windows(sequence, endpoints, window_state)
            args = (windows, full[b, t], lengths, positive[b, t], negative[b, t])
            if self.training and torch.is_grad_enabled():
                chunk_loss = checkpoint(
                    self._chunk_loss,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                chunk_loss = self._chunk_loss(*args)
            loss = loss + chunk_loss / locations.shape[0]
        return loss

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        features = self.log2feats(seq)
        if pos is None or neg is None:
            return features
        return self.predict_score(features, pos), self.predict_score(features, neg)

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        features = self.final_features(seq)
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0).expand(seq.shape[0], -1)
        if candidates.ndim != 2 or candidates.shape[0] != seq.shape[0]:
            raise ValueError("Expected one candidate row per sequence")
        if candidates.shape[1] == 0:
            return features.new_empty(seq.shape[0], 0)
        return torch.cat(
            [
                self.predict_score(features, chunk)
                for chunk in candidates.split(1024, dim=1)
            ],
            dim=1,
        )
        ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Measured fusion diagnostics and portable adapter export
class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)

    def _full_catalog_recommendations(self):
        # Five actual training users per category; never substitute pseudo-users.
        histories = self.view.user_train
        histories = histories.get("History", histories)
        users = sorted(
            (user for user, history in histories.items() if len(history)),
            key=lambda user: str(user),
        )[:5]
        model = self.model
        recommendations = []
        for user in users:
            history = [int(item) for item in histories[user]]
            if any(item < 1 or item > model.itemnum for item in history):
                raise ValueError("Recommendation history contains an invalid item ID")
            seen = set(history)
            sequence = torch.zeros(
                (1, model.maxlen), dtype=torch.long, device=self.device
            )
            suffix = history[-model.maxlen :]
            sequence[0, -len(suffix) :] = torch.tensor(suffix, device=self.device)
            features = model.final_features(sequence)
            best = []
            for start in range(1, model.itemnum + 1, 1024):
                ids = [
                    item
                    for item in range(start, min(start + 1024, model.itemnum + 1))
                    if item not in seen
                ]
                if not ids:
                    continue
                candidates = torch.tensor([ids], dtype=torch.long, device=self.device)
                scores = model.predict_score(features, candidates)[0]
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("Nonfinite full-catalog recommendation scores")
                best.extend(zip(scores.float().cpu().tolist(), ids))
                # Deterministic tie handling, also at chunk boundaries.
                best = sorted(best, key=lambda pair: (-pair[0], pair[1]))[:10]
            recommendations.append(
                {
                    "user": str(user),
                    "items": [item for _, item in best],
                }
            )
        return recommendations

    def run_full_diagnosis(self, train_loader):
        import itertools

        model = self.model
        flags = [(module, module.training) for module in model.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        samples = defaultdict(list)
        definitions = {
            "margin_mean": "Mean complete positive-minus-negative score on sampled training positions.",
            "margin_p10": "Linear-interpolated tenth percentile of complete training margins.",
            "negative_beats_positive_rate": "Fraction of sampled margins strictly below zero.",
            "small_margin_rate": "Fraction of sampled margins below 0.5; score-scale dependent.",
            "gate": "Mean recent-branch scalar mixture weight.",
            "full_raw_norm": "Mean L2 norm of unnormalized full branch outputs.",
            "recent_raw_norm": "Mean L2 norm of unnormalized recent branch outputs.",
            "full_contribution_norm": "Mean L2 norm of (1-g) times normalized full features; raw features for baseline mode.",
            "recent_contribution_norm": "Mean L2 norm of g times normalized recent features.",
            "branch_cosine": "Mean full/recent cosine similarity; does not establish intent or compatibility.",
            "embedding_cosine_mean": "Mean cosine of seed-42 sampled distinct nonpadding item pairs; alone not a collapse measure.",
            "sampled_positions": "Measured positions from at most four batches and 256 endpoints per batch.",
        }
        try:
            model.eval()
            with torch.no_grad():
                for _, sequence, positive, negative in itertools.islice(
                    train_loader, 4
                ):
                    sequence, positive, negative = [
                        torch.as_tensor(x, dtype=torch.long, device=self.device)
                        for x in (sequence, positive, negative)
                    ]
                    endpoints = (
                        sequence.ne(0) & positive.ne(0) & negative.ne(0)
                    ).nonzero()
                    if endpoints.shape[0] > 256:
                        indices = torch.linspace(
                            0, endpoints.shape[0] - 1, 256, device=self.device
                        ).long()
                        endpoints = endpoints[indices]
                    if not endpoints.numel():
                        continue
                    b, t = endpoints.unbind(1)
                    full = model._encode(sequence)[b, t]
                    windows, lengths = model.recent_windows(sequence, endpoints)
                    recent = model._encode(windows)[:, -1]
                    fused, gate = model.fuse(full, recent, lengths)
                    margin = model.predict_score(
                        fused, positive[b, t]
                    ) - model.predict_score(fused, negative[b, t])
                    z_full = (
                        full
                        if model.fusion_mode == "baseline"
                        else model._normalize(full)
                    )
                    values = {
                        "margins": margin,
                        "gate": gate.flatten(),
                        "full_raw_norm": full.norm(dim=-1),
                        "recent_raw_norm": recent.norm(dim=-1),
                        "full_contribution_norm": ((1 - gate) * z_full).norm(dim=-1),
                        "recent_contribution_norm": (
                            gate * model._normalize(recent)
                        ).norm(dim=-1),
                        "branch_cosine": torch.nn.functional.cosine_similarity(
                            full, recent, dim=-1
                        ),
                    }
                    for name, value in values.items():
                        samples[name].append(value.detach().cpu())
                    for name, mask in {
                        "prefix_1_to_10": lengths <= 10,
                        "prefix_11_to_30": (lengths > 10) & (lengths <= 30),
                        "prefix_over_30": lengths > 30,
                    }.items():
                        samples[name].append(margin[mask].detach().cpu())
                metrics = {}
                unavailable = {
                    "item_categories": "Category metadata is unavailable; no item-category breakdown is inferred.",
                    "review_preferences": "Review ratings/text are unavailable.",
                    "simulator_traits": "Diversity and conformity traits are unobserved.",
                }
                for name, parts in samples.items():
                    values = torch.cat(parts).float()
                    if name == "margins":
                        metrics["sampled_positions"] = values.numel()
                        if values.numel():
                            metrics.update(
                                {
                                    "margin_mean": float(values.mean()),
                                    "margin_p10": float(torch.quantile(values, 0.1)),
                                    "negative_beats_positive_rate": float(
                                        (values < 0).float().mean()
                                    ),
                                    "small_margin_rate": float(
                                        (values < 0.5).float().mean()
                                    ),
                                }
                            )
                        continue
                    definitions[name] = definitions.get(
                        name,
                        "Mean complete training margin in the named observed-prefix-length stratum.",
                    )
                    definitions[name + "_count"] = (
                        "Number of measured observations for " + name
                    )
                    metrics[name + "_count"] = values.numel()
                    if values.numel():
                        metrics[name] = float(values.mean())
                    else:
                        unavailable[name] = "No sampled positions in this stratum."
                metrics.setdefault("sampled_positions", 0)
                if not metrics["sampled_positions"]:
                    unavailable["margins"] = (
                        "No eligible training positions were sampled."
                    )
                embedding = model.item_emb.weight[1:]
                if len(embedding) >= 2:
                    generator = torch.Generator().manual_seed(42)
                    pairs = torch.randint(
                        len(embedding), (2048, 2), generator=generator
                    )
                    pairs = pairs[pairs[:, 0] != pairs[:, 1]].to(embedding.device)
                    metrics["embedding_cosine_mean"] = float(
                        torch.nn.functional.cosine_similarity(
                            embedding[pairs[:, 0]], embedding[pairs[:, 1]], dim=-1
                        ).mean()
                    )
                else:
                    unavailable["embedding_cosine_mean"] = "Fewer than two items."
                recommendations = self._full_catalog_recommendations()
                metrics["full_catalog_user_count"] = len(recommendations)
                definitions["full_catalog_user_count"] = (
                    "Number of real training users receiving full-catalog unseen-item "
                    "top-ten lists in this category view; five per view are required "
                    "for twenty users across the four categories."
                )
                if len(recommendations) < 5:
                    unavailable["twenty_user_report"] = (
                        "This category has fewer than five nonempty training histories."
                    )
                return {
                    "metrics": metrics,
                    "metric_definitions": definitions,
                    "full_catalog_recommendations": recommendations,
                    "dataset_name": getattr(self.view, "dataset_name", None),
                    "unavailable": unavailable,
                    "limitations": [
                        "Training diagnostics do not establish validation improvement.",
                        "Branch normalization changes score scale; compare matched normalized controls.",
                        "Gate variation does not establish intent; branch cancellation remains possible.",
                        "No test labels or simulator traits are observed.",
                    ],
                }
        finally:
            for module, training in flags:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def diagnosis_interpreter_prompt(raw_diagnosis):
    return (
        "Interpret only these measured metrics, definitions, sample counts and missing-data "
        "limitations. Training margins do not establish validation gains. A mapped cosine "
        "near 0.5 does not establish collapse. Discuss branch scale, cancellation and gate "
        "variation without inferring intent, unavailable metadata, or simulator traits.\n\n"
        + json.dumps(raw_diagnosis, indent=2, ensure_ascii=False, allow_nan=False)
    )


def write_diagnosis_script(working_dir):
    import inspect

    source = (
        "import json\nimport random\nfrom collections import defaultdict\n"
        "import numpy as np\nimport torch\n\n"
        + inspect.getsource(DiagnosisProbe)
        + "\n\n"
        + inspect.getsource(diagnosis_interpreter_prompt)
    )
    compile(source, "diagnosis_tools.py", "exec")
    for directory in (os.path.dirname(os.path.abspath(__file__)), working_dir):
        with open(os.path.join(directory, "diagnosis_tools.py"), "w") as handle:
            handle.write(source)


# ── Validation ────────────────────────────────────────────────────────
### <<< Self_EvolveRec-BLOCK-END


def evaluate_val(
    model: SASRec,
    user_train: dict,
    user_valid: dict,
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Fixed validation candidates without perturbing training RNG
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    was_training = model.training
    ### <<< Self_EvolveRec-BLOCK-END

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
            ### >>> Self_EvolveRec-BLOCK-START: Unique validation negatives with fixed local seed
            if itemnum - len(interacted) < 99:
                raise ValueError("Validation requires 99 unique unseen negatives")
            while len(neg_pool) < 99:
                ni = rng.randint(1, itemnum)
                if ni not in interacted:
                    neg_pool.add(ni)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + list(neg_pool)

            cand_t = torch.tensor([candidates], dtype=torch.long, device=device)
            scores = model.predict(seq_t, cand_t)[0].cpu().numpy().tolist()
            rank = sorted(range(len(candidates)), key=lambda i: -scores[i]).index(0)

            HR10 += 1.0 if rank < 10 else 0.0
            HR20 += 1.0 if rank < 20 else 0.0
            NDCG10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            NDCG20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0

    n = max(len(val_users), 1)
    ### >>> Self_EvolveRec-BLOCK-START: Restore validation caller mode
    model.train(was_training)
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }
    ### <<< Self_EvolveRec-BLOCK-END


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
        ### >>> Self_EvolveRec-BLOCK-START: Activate configured shared recent branch
        maxlen=MAXLEN,
        recent_window=RECENT_WINDOW,
        fusion_mode=FUSION_MODE,
        fusion_chunk=FUSION_CHUNK,
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Reject stale checkpoints and always save the first evaluated model
    if NUM_EPOCHS < 1 or NUM_BATCHES < 1:
        raise ValueError("Training requires positive epochs and batches")
    if os.path.exists(ckpt_path):
        raise FileExistsError(
            f"Refusing to reuse an existing trial checkpoint: {ckpt_path}"
        )
    best_hr10 = -1.0
    best_epoch = 0
    validation_history = []
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)

            ### >>> Self_EvolveRec-BLOCK-START: Bound saved recent activations and preserve baseline BCE weighting
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(seq_t, pos_t, neg_t)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Nonfinite training loss for {dataset_name} at epoch {epoch}"
                )
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Record every evaluated checkpoint and include final epoch
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            validation_history.append({"epoch": epoch, **val_m})
            with open(
                os.path.join(working_dir, f"{dataset_name}_validation.json"), "w"
            ) as handle:
                json.dump(
                    {
                        "selection_metric": "HR@10",
                        "seed": 42,
                        "fusion_mode": FUSION_MODE,
                        "recent_window": RECENT_WINDOW,
                        "tie_rule": "earliest checkpoint on equal HR@10",
                        "evaluations": validation_history,
                    },
                    handle,
                    indent=2,
                )
            ### <<< Self_EvolveRec-BLOCK-END
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
                ### >>> Self_EvolveRec-BLOCK-START: Serialize architecture and recent positional convention
                "maxlen": MAXLEN,
                "architecture": "sasrec_recent_fusion_v1",
                "recent_window": RECENT_WINDOW,
                "fusion_mode": FUSION_MODE,
                "fusion_chunk": FUSION_CHUNK,
                "recent_positions": "left padded W slots; positions 0 through W-1",
                "item_id_mapping": "original IDs 1 through itemnum; padding 0",
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    ### >>> Self_EvolveRec-BLOCK-START: Reload selected weights and check causal inference parity
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    user = next(u for u, history in user_train.items() if history)
    history = user_train[user][-MAXLEN:]
    sequence = torch.zeros((1, MAXLEN), dtype=torch.long, device=device)
    sequence[0, -len(history) :] = torch.tensor(history, device=device)
    with torch.no_grad():
        all_features = model.log2feats(sequence)
        final = model.final_features(sequence)
        torch.testing.assert_close(final, all_features[:, -1], atol=2e-5, rtol=2e-5)
        ids = torch.tensor([[1, max(1, itemnum // 2), itemnum]], device=device)
        scores = model.predict(sequence, ids)
        torch.testing.assert_close(
            scores, model.predict_score(final, ids), atol=2e-5, rtol=2e-5
        )
        # Score every catalog item in bounded chunks with one encoder call.
        selected_catalog_scores = {}
        for start in range(1, itemnum + 1, 1024):
            catalog_ids = torch.arange(
                start, min(start + 1024, itemnum + 1), device=device
            ).unsqueeze(0)
            catalog_scores = model.predict_score(final, catalog_ids)
            for column, item in enumerate(ids[0].tolist()):
                if start <= item < start + catalog_ids.shape[1]:
                    selected_catalog_scores[column] = catalog_scores[0, item - start]
        torch.testing.assert_close(
            scores[0],
            torch.stack([selected_catalog_scores[i] for i in range(ids.shape[1])]),
            atol=2e-5,
            rtol=2e-5,
        )
        restored = SASRec(
            itemnum=itemnum,
            hidden_units=HIDDEN_UNITS,
            num_blocks=NUM_BLOCKS,
            num_heads=NUM_HEADS,
            dropout_rate=0.0,
            maxlen=MAXLEN,
            recent_window=RECENT_WINDOW,
            fusion_mode=FUSION_MODE,
            fusion_chunk=FUSION_CHUNK,
        ).to(device)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        torch.testing.assert_close(
            scores, restored.predict(sequence, ids), atol=2e-5, rtol=2e-5
        )
        del restored
        cut = MAXLEN // 2
        changed = sequence.clone()
        changed[:, cut + 1 :] = 1
        torch.testing.assert_close(
            all_features[:, : cut + 1],
            model.log2feats(changed)[:, : cut + 1],
            atol=2e-5,
            rtol=2e-5,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10
    ### <<< Self_EvolveRec-BLOCK-END


# ── Write unified predict.py ──────────────────────────────────────────


def write_predict_script(working_dir: str) -> None:
    ### >>> Self_EvolveRec-BLOCK-START: Export the trained implementation independently of training imports
    import inspect

    model_source = (
        "from __future__ import annotations\nimport torch\nimport torch.nn as nn\n\n"
        + inspect.getsource(PointWiseFeedForward)
        + "\n\n"
        + inspect.getsource(SASRec)
    )
    compile(model_source, "fusion_model.py", "exec")
    with open(os.path.join(working_dir, "fusion_model.py"), "w") as handle:
        handle.write(model_source)
    ### <<< Self_EvolveRec-BLOCK-END
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
        ### >>> Self_EvolveRec-BLOCK-START: Load frozen exported fusion architecture
        f"perdataset_fusion_{os.getpid()}", os.path.join(_HERE, "fusion_model.py")
        ### <<< Self_EvolveRec-BLOCK-END
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SASRec


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Reject incompatible checkpoints
        if meta.get("architecture") != "sasrec_recent_fusion_v1":
            raise ValueError(f"Unsupported checkpoint architecture for {ds}")
        _metas[ds] = meta
        m = SASRec(
        ### <<< Self_EvolveRec-BLOCK-END
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            ### >>> Self_EvolveRec-BLOCK-START: Restore exact trained fusion behavior
            maxlen=meta["maxlen"],
            recent_window=meta["recent_window"],
            fusion_mode=meta["fusion_mode"],
            fusion_chunk=meta["fusion_chunk"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Prevent silent cross-category fallback
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"No trained model for dataset_name={dataset_name!r}")
    if dataset_name and dataset_name in _models:
    ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Compile and verify exported predictor against selected checkpoints
    compile(script, predict_path, "exec")
    with open(predict_path, "w") as f:
        f.write(script)

    import importlib.util

    # Predictor loading initializes fresh modules; preserve the caller's RNG.
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        spec = importlib.util.spec_from_file_location(
            "verified_trial_predictor", predict_path
        )
        exported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exported)
        for dataset in DEFAULT_DATASETS:
            with open(os.path.join(working_dir, f"{dataset}_meta.pkl"), "rb") as handle:
                meta = pickle.load(handle)
            histories, _, _ = load_dataset(dataset, DATA_DIR)
            user = next(user for user, history in histories.items() if history)
            history = histories[user][-meta["maxlen"] :]
            ids = [1, max(1, meta["itemnum"] // 2), meta["itemnum"]]
            actual = exported.predict(user, history, ids, dataset_name=dataset)
            device = exported._device
            reference = SASRec(
                itemnum=meta["itemnum"],
                hidden_units=meta["hidden_units"],
                num_blocks=meta["num_blocks"],
                num_heads=meta["num_heads"],
                dropout_rate=0.0,
                maxlen=meta["maxlen"],
                recent_window=meta["recent_window"],
                fusion_mode=meta["fusion_mode"],
                fusion_chunk=meta["fusion_chunk"],
            ).to(device)
            reference.load_state_dict(
                torch.load(
                    os.path.join(working_dir, f"{dataset}_model.pt"),
                    map_location=device,
                    weights_only=True,
                ),
                strict=True,
            )
            reference.eval()
            sequence = torch.zeros((1, meta["maxlen"]), dtype=torch.long, device=device)
            sequence[0, -len(history) :] = torch.tensor(history, device=device)
            with torch.no_grad():
                expected = reference.predict(
                    sequence, torch.tensor([ids], device=device)
                )[0].cpu()
            torch.testing.assert_close(
                torch.tensor(actual, dtype=expected.dtype),
                expected,
                atol=2e-5,
                rtol=2e-5,
            )
            del reference
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    print(f"predict.py written and reload parity verified → {predict_path}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END


# ── Main ──────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)

    ### >>> Self_EvolveRec-BLOCK-START: Export diagnostics and record the trial protocol before training
    write_diagnosis_script(working_dir)
    with open(os.path.join(working_dir, "trial_protocol.json"), "w") as handle:
        json.dump(
            {
                "training_seed": 42,
                "preprocessing_seed": 0,
                "splits": "unchanged host-provided train and validation files",
                "datasets": DATASETS,
                "fusion_mode": FUSION_MODE,
                "recent_window": RECENT_WINDOW,
                "fusion_chunk": FUSION_CHUNK,
                "max_epochs": NUM_EPOCHS,
                "validation_cadence_epochs": 5,
                "stopping_rule": "baseline: PATIENCE * 5 epochs since strict HR@10 improvement",
                "checkpoint_tie_rule": "retain earliest checkpoint",
                "selection": "independent host macro validation HR@10",
                "shared_search_budget_gpu_seconds": 43200,
                "budget_enforcement": "host; includes training, diagnostics and failures",
                "trial_scope": "one configured architecture across all four categories",
                "control_modes_available": [
                    "baseline",
                    "normalized_full",
                    "recent",
                    "fixed25",
                    "fixed50",
                    "fixed75",
                    "global",
                    "context",
                ],
                "control_status": "available via SASREC_FUSION_MODE; no control results claimed",
                "minimum_practical_macro_gain": 0.005,
                "cross_trial_tie_rule": "prefer lower measured GPU time within 0.005 macro HR@10",
                "paired_validation_intervals": "unavailable unless supplied by the unchanged host evaluator",
            },
            handle,
            indent=2,
        )
    all_hr10: list[float] = []
    ### <<< Self_EvolveRec-BLOCK-END
    for ds in DATASETS:
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        hr10 = train_one_dataset(ds, device, DATA_DIR)
        all_hr10.append(hr10)

    write_predict_script(working_dir)

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


