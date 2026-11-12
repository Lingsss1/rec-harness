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
### >>> Self_EvolveRec-BLOCK-START: Match the supplied training epoch default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Freeze baseline architecture and objective for sparse transition fusion
DATA_DIR = os.environ["GAGC_DATA_DIR"]
EXPERIMENT = "baseline"
FUSION_MODE = "none"
USE_CANDIDATE_CE = False
RECENT_WINDOW = 5
NORM_EPS = 1e-8
NUM_NEGATIVES = 64
POSITION_CHUNK = 256
# Each invocation emits one configuration for independent host evaluation.
# The zero configuration is deduplicated to window one. No local surrogate
# evaluation selects a transition coefficient or changes the frozen checkpoint.
TRANSITION_ALPHA = float(os.environ.get("SASREC_TRANSITION_ALPHA", "0.1"))
TRANSITION_WINDOW = int(os.environ.get("SASREC_TRANSITION_WINDOW", "5"))
if TRANSITION_ALPHA not in {0.0, 0.1, 0.3}:
    raise ValueError("SASREC_TRANSITION_ALPHA must be 0, 0.1, or 0.3")
if TRANSITION_WINDOW not in {1, 5}:
    raise ValueError("SASREC_TRANSITION_WINDOW must be 1 or 5")
if TRANSITION_ALPHA == 0.0:
    TRANSITION_WINDOW = 1
TRANSITION_TAU = 10.0
TRANSITION_DECAY = 0.5
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four dataset routing paths
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

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


### >>> Self_EvolveRec-BLOCK-START: Preserve sampler identities and handle empty negative supports
def sample_batch(
    user_train: dict,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users: bool = False,
    eligible_users=None,
    rng=None,
    make_negatives: bool = True,
):
    rng = random if rng is None else rng
    users = eligible_users
    if users is None:
        users = [
            u
            for u, items in user_train.items()
            if len(items) >= 2 and len(set(items)) < itemnum
        ]
    if not users:
        raise ValueError("No training users with nonempty unseen-item support")
    batch_seq, batch_pos, batch_neg, batch_users = [], [], [], []
    ### <<< Self_EvolveRec-BLOCK-END
    for _ in range(batch_size):
        ### >>> Self_EvolveRec-BLOCK-START: Keep internally sampled user IDs
        u = rng.choice(users)
        batch_users.append(u)
        history = user_train[u]
        ### <<< Self_EvolveRec-BLOCK-END
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)
        nxt = history[-1]
        idx = maxlen - 1
        ts = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            ### >>> Self_EvolveRec-BLOCK-START: Dedicated sampler RNG and optional BCE negatives
            if nxt != 0 and make_negatives:
                neg_item = rng.randint(1, itemnum)
                while neg_item in ts:
                    neg_item = rng.randint(1, itemnum)
                neg[idx] = neg_item
            ### <<< Self_EvolveRec-BLOCK-END
            nxt = item
            idx -= 1
            if idx == -1:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    ### >>> Self_EvolveRec-BLOCK-START: Preserve diagnostic tuple compatibility while exposing internal identities
    result = (np.array(batch_seq), np.array(batch_pos), np.array(batch_neg))
    return (np.asarray(batch_users), *result) if return_users else result
    ### <<< Self_EvolveRec-BLOCK-END


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


### >>> Self_EvolveRec-BLOCK-START: Sparse uniformly smoothed adjacent-transition log residual
class SparseTransitionResidual:
    """Training-only counts; neural features and diagnostic scores stay neural-only."""

    def __init__(self, histories, itemnum):
        self.itemnum = int(itemnum)
        self.tau = TRANSITION_TAU
        self.decay = TRANSITION_DECAY
        counts = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        for history in histories.values():
            for item in history:
                if not 1 <= item <= self.itemnum:
                    raise ValueError("Training item outside the nonpadding catalog")
            for source, target in zip(history, history[1:]):
                counts[source][target] += 1
                totals[source] += 1
        self.rows = {}
        for source, row in counts.items():
            keys = np.asarray(sorted(row), dtype=np.int64)
            values = np.asarray([row[int(k)] for k in keys], dtype=np.int64)
            self.rows[int(source)] = (keys, values)
        self.row_totals = dict(totals)
        self.scales = {}
        self.calibration = []
        self.alpha = TRANSITION_ALPHA
        self.window = TRANSITION_WINDOW

    def correction(self, history, candidates, window=None, subtract=None):
        """Candidate-independent log pooling, retaining repeated source occurrences.

        subtract=(source,target) removes exactly one identified adjacent occurrence
        globally. Callers must provide provenance; loader item matching is unsafe.
        The row-total term cancels from this ranking-only correction.
        """
        ids = np.asarray(candidates, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("Candidates must be one-dimensional")
        if np.any((ids < 1) | (ids > self.itemnum)):
            raise ValueError("Candidate outside the nonpadding catalog")
        width = self.window if window is None else int(window)
        if width < 1:
            raise ValueError("Transition window must be positive")
        sources = [int(x) for x in history if int(x) != 0][-width:][::-1]
        if any(x < 1 or x > self.itemnum for x in sources):
            raise ValueError("Source outside the nonpadding catalog")
        if subtract is not None:
            subtract = tuple(map(int, subtract))
            if len(subtract) != 2:
                raise ValueError("Subtraction requires one identified adjacent pair")
            source, target = subtract
            keys, values = self.rows.get(
                source, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            )
            index = int(np.searchsorted(keys, target))
            if index >= len(keys) or keys[index] != target or values[index] < 1:
                raise ValueError("Cannot subtract an absent transition occurrence")
        result = np.zeros(ids.shape, dtype=np.float64)
        if not sources or not ids.size:
            return result
        weights = self.decay ** np.arange(len(sources), dtype=np.float64)
        weights /= weights.sum()
        for source, weight in zip(sources, weights):
            row = self.rows.get(source)
            if row is None:
                continue
            keys, values = row
            indices = np.searchsorted(keys, ids)
            inside = indices < len(keys)
            matched = np.zeros(ids.shape, dtype=bool)
            matched[inside] = keys[indices[inside]] == ids[inside]
            queried = np.zeros(ids.shape, dtype=np.float64)
            queried[matched] = values[indices[matched]]
            if subtract is not None and source == subtract[0]:
                queried[ids == subtract[1]] -= 1.0
            result += weight * np.log1p(self.itemnum * queried / self.tau)
        return result

    def fuse(self, neural, history, candidates):
        # Validate every inference path, including zero-alpha baseline scoring.
        values = np.asarray(neural, dtype=np.float64)
        ids = np.asarray(candidates, dtype=np.int64)
        if values.ndim != 1 or ids.ndim != 1 or values.shape != ids.shape:
            raise ValueError("Neural scores and candidates must be aligned vectors")
        if np.any((ids < 1) | (ids > self.itemnum)):
            raise ValueError("Candidate outside the nonpadding catalog")
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite neural inference scores")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("Invalid serialized transition coefficient")
        # Preserve original values exactly in the baseline configuration.
        if self.alpha == 0.0:
            return neural
        scale = self.scales[self.window]
        if not scale["enabled"]:
            return neural
        ratio = float(scale["ratio"])
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise FloatingPointError("Invalid frozen transition scale ratio")
        transition = self.correction(history, ids)
        fused = values + self.alpha * ratio * transition
        if not np.isfinite(fused).all():
            raise FloatingPointError("Nonfinite fused inference scores")
        return fused.tolist()


def calibrate_transition(model, histories, residual, device, maxlen):
    """Uniform identified training prefixes; no validation targets or filtering."""
    rng = np.random.default_rng(42)
    users = [u for u, h in histories.items() if len(h) >= 2]
    sizes = np.asarray([len(histories[u]) - 1 for u in users], dtype=np.int64)
    ends = sizes.cumsum()
    if not len(ends) or ends[-1] == 0:
        raise ValueError("No eligible training prefixes for calibration")
    sample_count = min(2048, int(ends[-1]))
    selected = rng.choice(int(ends[-1]), size=sample_count, replace=False)
    records = []
    for flat_index in selected:
        user_index = int(np.searchsorted(ends, flat_index, side="right"))
        start = 0 if user_index == 0 else int(ends[user_index - 1])
        position = int(flat_index) - start
        user = users[user_index]
        history = histories[user]
        prefix = history[max(0, position - maxlen + 1) : position + 1]
        candidates = (
            rng.choice(residual.itemnum, size=min(256, residual.itemnum), replace=False)
            + 1
        ).astype(np.int64)
        records.append(
            {
                "user": int(user),
                "source_position": position,
                "target": int(history[position + 1]),
                "prefix": list(prefix),
                "candidates": candidates,
            }
        )
    sums = {1: [0.0, 0.0], 5: [0.0, 0.0]}
    denominator = 0
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(records), 32):
                batch = records[start : start + 32]
                sequences = np.zeros((len(batch), maxlen), dtype=np.int64)
                for row, record in enumerate(batch):
                    prefix = record["prefix"]
                    sequences[row, -len(prefix) :] = prefix
                candidates = np.stack([record["candidates"] for record in batch])
                neural = (
                    model.predict(
                        torch.as_tensor(sequences, device=device),
                        torch.as_tensor(candidates, device=device),
                    )
                    .double()
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(neural).all():
                    raise FloatingPointError("Nonfinite neural calibration scores")
                centered = neural - neural.mean(axis=1, keepdims=True)
                neural_ss = float(np.square(centered).sum())
                denominator += int(neural.size)
                for window in (1, 5):
                    transition = np.stack(
                        [
                            residual.correction(r["prefix"], r["candidates"], window)
                            for r in batch
                        ]
                    )
                    if not np.isfinite(transition).all():
                        raise FloatingPointError(
                            "Nonfinite transition calibration scores"
                        )
                    transition -= transition.mean(axis=1, keepdims=True)
                    sums[window][0] += neural_ss
                    sums[window][1] += float(np.square(transition).sum())
    finally:
        for module, training in modes:
            module.training = training
    for window, (neural_ss, transition_ss) in sums.items():
        sigma_n = math.sqrt(neural_ss / denominator)
        sigma_t = math.sqrt(transition_ss / denominator)
        if not math.isfinite(sigma_n) or not math.isfinite(sigma_t):
            raise FloatingPointError("Nonfinite frozen calibration scale")
        enabled = sigma_n > 1e-8 and sigma_t > 1e-8
        epsilon_n = max(1e-6, 0.01 * sigma_n)
        epsilon_t = max(1e-6, 0.01 * sigma_t)
        residual.scales[window] = {
            "sigma_neural": sigma_n,
            "sigma_transition": sigma_t,
            "epsilon_neural": epsilon_n,
            "epsilon_transition": epsilon_t,
            "ratio": sigma_n / max(sigma_t, epsilon_t) if enabled else 0.0,
            "enabled": enabled,
            "disabled_reason": None if enabled else "Measured component RMS <= 1e-8",
            "queries": len(records),
            "candidate_scores": denominator,
        }
    residual.calibration = records


def check_transition_formula():
    # Small deterministic checks of smoothing, repeated sources, subtraction,
    # candidate permutations, chunks, unsupported rows and zero-weight parity.
    residual = SparseTransitionResidual({1: [1, 2, 1, 2], 2: [1, 3]}, 4)
    candidates = np.arange(1, 5)
    history = [1, 2, 1]
    counts = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 2, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    for subtract in (None, (1, 2)):
        adjusted = counts.copy()
        if subtract is not None:
            adjusted[subtract] -= 1
        weights = 0.5 ** np.arange(len(history))
        weights /= weights.sum()
        dense_log = np.zeros(4)
        constant = 0.0
        for source, weight in zip(history[::-1], weights):
            total = adjusted[source].sum()
            dense_log += weight * np.log(
                (adjusted[source, candidates] + 10.0 / 4) / (total + 10.0)
            )
            constant += weight * (math.log(10.0 / 4) - math.log(total + 10.0))
        sparse = residual.correction(history, candidates, 5, subtract)
        np.testing.assert_allclose(sparse, dense_log - constant, atol=1e-12)
        chunked = np.concatenate(
            [
                residual.correction(history, candidates[:2], 5, subtract),
                residual.correction(history, candidates[2:], 5, subtract),
            ]
        )
        np.testing.assert_array_equal(sparse, chunked)
        np.testing.assert_array_equal(
            sparse[::-1], residual.correction(history, candidates[::-1], 5, subtract)
        )
    np.testing.assert_array_equal(residual.correction([], candidates), np.zeros(4))
    np.testing.assert_array_equal(residual.correction([4], candidates), np.zeros(4))
    residual.alpha = 0.0
    original = [0.25, -0.5, 1.0, 2.0]
    assert residual.fuse(original, history, candidates) is original


### <<< Self_EvolveRec-BLOCK-END


class SASRec(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Serialize architecture-specific fusion settings
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        fusion_mode: str = "none",
        recent_window: int = 5,
        norm_eps: float = 1e-8,
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

        ### >>> Self_EvolveRec-BLOCK-START: Equal-scale branches with a baseline-favoring convex gate
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if fusion_mode not in {"none", "normalized", "learned"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        self.fusion_mode = fusion_mode
        self.recent_window = recent_window
        self.norm_eps = norm_eps
        if fusion_mode != "none":
            self.history_norm = nn.LayerNorm(
                hidden_units, eps=norm_eps, elementwise_affine=False
            )
        if fusion_mode == "learned":
            self.recent_projection = nn.Linear(hidden_units, hidden_units, bias=False)
            nn.init.eye_(self.recent_projection.weight)
            self.recent_norm = nn.LayerNorm(
                hidden_units, eps=norm_eps, elementwise_affine=False
            )
            self.gate = nn.Linear(2 * hidden_units, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)
        ### <<< Self_EvolveRec-BLOCK-END

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)

        pos = torch.arange(M, device=seq.device).unsqueeze(0).expand(B, -1)
        seq_emb = seq_emb + self.pos_emb(pos)
        seq_emb = self.emb_dropout(seq_emb)

        timeline_mask = seq == 0
        seq_emb = seq_emb * (~timeline_mask).unsqueeze(-1)
        ### >>> Self_EvolveRec-BLOCK-START: Safe causal padding mask for fusion with unchanged backbone controls
        attention_mask = ~torch.tril(
            torch.ones((M, M), dtype=torch.bool, device=seq.device)
        )
        if self.fusion_mode != "none":
            blocked = attention_mask.unsqueeze(0) | timeline_mask.unsqueeze(1)
            # Only padding queries get a diagonal escape. Valid queries retain
            # the exact causal, nonpadding-key mask, including on empty histories.
            diagonal = torch.eye(M, dtype=torch.bool, device=seq.device).unsqueeze(0)
            blocked = blocked & ~(timeline_mask.unsqueeze(-1) & diagonal)
            heads = self.attention_layers[0].num_heads if self.attention_layers else 1
            attention_mask = blocked.repeat_interleave(heads, dim=0)
        ### <<< Self_EvolveRec-BLOCK-END

        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            queries = self.attention_layernorms[i](seq_emb)
            ### >>> Self_EvolveRec-BLOCK-START: Avoid computing unused attention weights
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

        ### >>> Self_EvolveRec-BLOCK-START: Linear-time causal recent-context pooling and convex fusion
        history = self.last_layernorm(seq_emb)
        if self.fusion_mode == "none":
            return history
        history = self.history_norm(history)
        valid = ~timeline_mask
        if self.fusion_mode == "normalized":
            return history.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        # Compact valid embeddings by their causal rank. Padding contributes
        # exactly zero, including when padding occurs inside a sequence.
        ranks = valid.long().cumsum(dim=1)
        embeddings = self.item_emb(seq).masked_fill(timeline_mask.unsqueeze(-1), 0.0)
        rank_index = ranks.unsqueeze(-1).expand(-1, -1, self.hidden_units)
        compact = embeddings.new_zeros(B, M + 1, self.hidden_units)
        compact = compact.scatter_add(1, rank_index, embeddings)
        prefix = compact.cumsum(dim=1)
        start_ranks = (ranks - self.recent_window).clamp_min(0)
        start_index = start_ranks.unsqueeze(-1).expand_as(rank_index)
        recent = prefix.gather(1, rank_index) - prefix.gather(1, start_index)
        count = ranks.clamp(max=self.recent_window).clamp_min(1)
        recent = recent / count.unsqueeze(-1).to(recent.dtype)
        recent = self.recent_norm(self.recent_projection(recent))
        gate = torch.sigmoid(self.gate(torch.cat((history, recent), dim=-1)))
        fused = (1.0 - gate) * history + gate * recent
        return fused.masked_fill(timeline_mask.unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END

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
    ### >>> Self_EvolveRec-BLOCK-START: Deterministic validation without perturbing training randomness
    rng = random.Random(42)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Batch local validation while preserving sampled candidates and stable ties
    if not val_users:
        raise ValueError("No eligible validation users for checkpoint selection")
    sequences, candidate_rows = [], []
    for u in val_users:
        pos_items = user_valid[u]
        target = pos_items[0]
        hist = user_train[u][-maxlen:]
        if not 1 <= target <= itemnum:
            raise ValueError("Validation target outside the nonpadding catalog")
        if any(item < 1 or item > itemnum for item in hist):
            raise ValueError("Validation history outside the nonpadding catalog")
        seq_arr = np.zeros(maxlen, dtype=np.int64)
        if hist:
            seq_arr[-len(hist) :] = hist
        interacted = set(user_train[u]) | set(pos_items)
        if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
            raise ValueError("Validation requires 99 unique unseen negatives")
        neg_pool = set()
        while len(neg_pool) < 99:
            ni = rng.randint(1, itemnum)
            if ni not in interacted:
                neg_pool.add(ni)
        # Retain the original RNG consumption and set iteration convention.
        sequences.append(seq_arr)
        candidate_rows.append([target] + list(neg_pool))

    modes = [(module, module.training) for module in model.modules()]
    ranks = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(sequences), 32):
                seq_t = torch.as_tensor(
                    np.stack(sequences[start : start + 32]),
                    dtype=torch.long,
                    device=device,
                )
                cand_t = torch.as_tensor(
                    candidate_rows[start : start + 32],
                    dtype=torch.long,
                    device=device,
                )
                scores = model.predict(seq_t, cand_t)
                if scores.shape != cand_t.shape:
                    raise ValueError("Validation scores must match candidate shape")
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Nonfinite local validation scores")
                # Target is candidate zero, so original stable sorting places it
                # ahead of ties. Only strictly larger scores contribute to rank.
                ranks.append((scores > scores[:, :1]).sum(-1).cpu().numpy())
    finally:
        for module, training in modes:
            module.training = training
    ranks = np.concatenate(ranks)
    discounts = 1.0 / np.log2(ranks + 2.0)
    return {
        "HR@10": float(np.mean(ranks < 10)),
        "HR@20": float(np.mean(ranks < 20)),
        "NDCG@10": float(np.mean(discounts * (ranks < 10))),
        "NDCG@20": float(np.mean(discounts * (ranks < 20))),
    }
    ### <<< Self_EvolveRec-BLOCK-END


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)

    ### >>> Self_EvolveRec-BLOCK-START: Reproducible independent dataset models and training-only negative supports
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    batch_rng = random.Random(42)
    negative_rng = np.random.default_rng(42)
    eligible_users = []
    excluded_thresholds = {}
    empty_support_positions = 0
    for u, history in user_train.items():
        if len(history) < 2:
            continue
        excluded = np.asarray(sorted(set(history)), dtype=np.int64)
        if len(excluded) >= itemnum:
            empty_support_positions += min(len(history) - 1, MAXLEN)
            continue
        eligible_users.append(u)
        excluded_thresholds[u] = excluded - np.arange(len(excluded), dtype=np.int64)
    if not eligible_users:
        raise ValueError(f"{dataset_name}: no eligible training users")
    print(
        f"  [{dataset_name}] empty-support training positions excluded="
        f"{empty_support_positions}",
        flush=True,
    )
    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        fusion_mode=FUSION_MODE,
        recent_window=RECENT_WINDOW,
        norm_eps=NORM_EPS,
    ).to(device)
    ### <<< Self_EvolveRec-BLOCK-END

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always save the first evaluated checkpoint, including zero-hit runs
    best_hr10 = -float("inf")
    best_epoch = 0
    optimizer_steps = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Exact chunked candidate-set CE with one backbone backward pass
            users, seq, pos, neg = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                return_users=True,
                eligible_users=eligible_users,
                rng=batch_rng,
                make_negatives=not USE_CANDIDATE_CE,
            )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            valid_np = np.argwhere(pos != 0)
            n_valid = len(valid_np)
            if n_valid == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            if USE_CANDIDATE_CE:
                features = model.log2feats(seq_t)
                # Chunk losses accumulate dL/dfeatures in this leaf. After all
                # chunks, explicitly propagate that gradient into the backbone.
                # Output embedding gradients still accumulate on model weights.
                scoring_features = features.detach().requires_grad_(True)
                negative_ids = np.empty((n_valid, NUM_NEGATIVES), dtype=np.int64)
                for row in np.unique(valid_np[:, 0]):
                    positions = np.flatnonzero(valid_np[:, 0] == row)
                    thresholds = excluded_thresholds[int(users[row])]
                    support_size = itemnum - len(thresholds)
                    # Map uniform ranks into the sorted complement. Duplicates
                    # are retained. Every target is in the excluded full history.
                    ranks = negative_rng.integers(
                        1, support_size + 1, size=(len(positions), NUM_NEGATIVES)
                    )
                    draws = ranks + np.searchsorted(thresholds, ranks, side="right")
                    negative_ids[positions] = draws
                batch_loss = 0.0
                for start in range(0, n_valid, POSITION_CHUNK):
                    stop = min(start + POSITION_CHUNK, n_valid)
                    coordinates = valid_np[start:stop]
                    rows = torch.as_tensor(coordinates[:, 0], device=device)
                    cols = torch.as_tensor(coordinates[:, 1], device=device)
                    targets = pos_t[rows, cols].unsqueeze(1)
                    negatives = torch.as_tensor(negative_ids[start:stop], device=device)
                    candidates = torch.cat((targets, negatives), dim=1)
                    logits = torch.bmm(
                        model.item_emb(candidates),
                        scoring_features[rows, cols].unsqueeze(-1),
                    ).squeeze(-1)
                    loss = (
                        nn.functional.cross_entropy(
                            logits,
                            torch.zeros(stop - start, dtype=torch.long, device=device),
                            reduction="sum",
                        )
                        / n_valid
                    )
                    loss.backward()
                    batch_loss += loss.detach().item()
                if scoring_features.grad is None:
                    raise RuntimeError("Candidate CE produced no feature gradients")
                features.backward(scoring_features.grad)
            else:
                neg_t = torch.tensor(neg, dtype=torch.long, device=device)
                pos_logits, neg_logits = model(seq_t, pos_t, neg_t)
                indices = pos_t != 0
                loss = bce_criterion(
                    pos_logits[indices], torch.ones_like(pos_logits[indices])
                )
                loss = loss + bce_criterion(
                    neg_logits[indices], torch.zeros_like(neg_logits[indices])
                )
                loss.backward()
                batch_loss = loss.detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            total_loss += batch_loss
            ### <<< Self_EvolveRec-BLOCK-END

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Evaluate the final epoch even for short training runs
        should_evaluate = epoch % 5 == 0 or epoch == NUM_EPOCHS
        ### <<< Self_EvolveRec-BLOCK-END
        if should_evaluate:
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist resolved architecture, objective and training exposure
                "maxlen": MAXLEN,
                "fusion_mode": FUSION_MODE,
                "recent_window": RECENT_WINDOW,
                "norm_eps": NORM_EPS,
                "experiment": EXPERIMENT,
                "objective": (
                    "candidate_set_ce" if USE_CANDIDATE_CE else "single_negative_bce"
                ),
                "num_negatives": NUM_NEGATIVES if USE_CANDIDATE_CE else 1,
                "temperature": 1.0,
                "training_seed": 42,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "configured_epochs": NUM_EPOCHS,
                "completed_epochs": epoch,
                "optimizer_steps": optimizer_steps,
                "best_epoch": best_epoch,
                "patience": PATIENCE,
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )

    ### >>> Self_EvolveRec-BLOCK-START: Freeze neural checkpoint before training-only transition calibration
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    residual = SparseTransitionResidual(user_train, itemnum)
    calibrate_transition(model, user_train, residual, device, MAXLEN)
    # Plain arrays/dicts allow the generated predictor to load artifacts without
    # depending on the module name used to execute train.py.
    transition_path = os.path.join(working_dir, f"{dataset_name}_transition.pkl")
    with open(transition_path, "wb") as f:
        pickle.dump(vars(residual), f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "dataset": dataset_name,
        "neural_checkpoint_epoch": best_epoch,
        "neural_local_validation_hr10": best_hr10,
        "selection_metric": "host pinned four-category macro validation HR@10",
        "residual_selected_by_local_validation": False,
        "configuration": {
            "alpha": residual.alpha,
            "window": residual.window,
            "decay": residual.decay,
            "tau": residual.tau,
            "count_window": 1,
            "count_distance_weight": 1.0,
        },
        "host_search_grid": {
            "alpha": [0.0, 0.1, 0.3],
            "window": [1, 5],
            "zero_coefficient_window": 1,
            "tie_break": "zero alpha, smaller alpha, smaller window",
        },
        "calibration_seed": 42,
        "calibration_convention": (
            "Uniform distinct identified training source positions; right-aligned "
            "maxlen-truncated prefixes; unique uniform unfiltered nonpadding "
            "candidates; pooled within-query-centered RMS; frozen before inference"
        ),
        "scales": residual.scales,
        "diagnostics": {
            "compatibility_scores": "neural-only; actual SASRec module unchanged",
            "fused_margin": None,
            "fused_margin_reason": (
                "Not measured: diagnostic loader tuples do not establish unique "
                "training-occurrence provenance. Calibration records retain explicit "
                "user and source positions for a separate identified-prefix probe."
            ),
        },
        "controls_completed": [],
        "host_validation_results": None,
    }
    with open(
        os.path.join(working_dir, f"{dataset_name}_transition_manifest.json"), "w"
    ) as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    print(
        f"  [{dataset_name}] Frozen neural local val HR@10={best_hr10:.4f}"
        f"  checkpoint={ckpt_path}  transition={transition_path}"
        f"  alpha={residual.alpha} window={residual.window}",
        flush=True,
    )
    return best_hr10
    ### <<< Self_EvolveRec-BLOCK-END


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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four models in the generated predictor
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

### >>> Self_EvolveRec-BLOCK-START: Keep sparse artifacts separate from trained neural modules
_models: dict = {}
_metas: dict = {}
_transitions: dict = {}
_device = None
### <<< Self_EvolveRec-BLOCK-END


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ### >>> Self_EvolveRec-BLOCK-START: Load matching neural and sparse scoring implementations
    return mod
    ### <<< Self_EvolveRec-BLOCK-END


def _ensure_loaded():
    ### >>> Self_EvolveRec-BLOCK-START: Reject incomplete predictor initialization on retry
    global _device
    expected = set(DATASETS)
    if (
        set(_models) == expected
        and set(_metas) == expected
        and set(_transitions) == expected
    ):
        return
    # A previous loading exception must never make a partial dictionary appear
    # successfully initialized. The original exception propagates to the caller.
    _models.clear()
    _metas.clear()
    _transitions.clear()
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### >>> Self_EvolveRec-BLOCK-START: Reconstruct serialized sparse residuals without rebuilding counts
    runtime = _load_model_class()
    SASRec = runtime.SASRec
    ### <<< Self_EvolveRec-BLOCK-END
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Report missing dataset artifacts explicitly
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model artifacts for {ds} in {_HERE}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        m = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Load the identical trained fusion architecture at inference
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            fusion_mode=meta.get("fusion_mode", "none"),
            recent_window=meta.get("recent_window", 5),
            norm_eps=meta.get("norm_eps", 1e-8),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        ### >>> Self_EvolveRec-BLOCK-START: Require category-matched frozen transition artifacts
        m.eval()
        transition_path = os.path.join(_HERE, f"{ds}_transition.pkl")
        with open(transition_path, "rb") as f:
            state = pickle.load(f)
        residual = runtime.SparseTransitionResidual.__new__(runtime.SparseTransitionResidual)
        residual.__dict__.update(state)
        if residual.itemnum != meta["itemnum"]:
            raise ValueError(f"Transition catalog mismatch for {ds}")
        _transitions[ds] = residual
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Validate explicit dataset routing while retaining the legacy signature
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"Unknown or unavailable dataset: {dataset_name}")
        ds = dataset_name
    else:
        if not _models:
            raise RuntimeError("No dataset models loaded")
        ds = next(iter(_models))
    model = _models[ds]
    meta = _metas[ds]
    if not candidates:
        return []
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = meta["maxlen"]
    hist   = history[-maxlen:]
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(hist)):
        seq_arr[maxlen - 1 - i] = item

    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)

    ### >>> Self_EvolveRec-BLOCK-START: Fixed-scale sparse fusion with one neural encoding per query
    with torch.no_grad():
        features = model.log2feats(seq_t)[:, -1, :]
        scores = []
        # Bound full-catalog output-embedding memory. Each candidate receives
        # the same dot product and frozen residual regardless of chunk size.
        for start in range(0, len(candidates), 8192):
            chunk = candidates[start:start + 8192]
            cand_t = torch.tensor([chunk], dtype=torch.long, device=_device)
            chunk_scores = model.item_emb(cand_t).matmul(
                features.unsqueeze(-1)
            ).squeeze(-1)[0].cpu().tolist()
            scores.extend(_transitions[ds].fuse(chunk_scores, hist, chunk))
    return scores
    ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Check sparse formula before training and record the bounded host search
    check_transition_formula()
    protocol = {
        "datasets": DATASETS,
        "preprocessing_seed": 0,
        "training_seed": 42,
        "neural_architecture": "unchanged per-dataset SASRec",
        "neural_objective": "single-negative BCE",
        "epochs": NUM_EPOCHS,
        "checkpoint_cadence_epochs": 5,
        "checkpoint_rule": "strict local validation HR@10 improvement; earlier on ties",
        "early_stopping_checks": PATIENCE,
        "configuration_emitted": {
            "alpha": TRANSITION_ALPHA,
            "window": TRANSITION_WINDOW,
            "tau": TRANSITION_TAU,
            "decay": TRANSITION_DECAY,
        },
        "selection_owner": "independent pinned host evaluator",
        "shared_gpu_second_budget": 43200,
        "cost_accounting_owner": "host; all calibration and checks are charged",
        "controls_status": "Not executed by this training invocation",
        "reuse_rule": (
            "Reuse these frozen checkpoints and calibration artifacts for other "
            "host-grid configurations; do not repeat identical neural training. "
            "Change alpha/window together for every category in copied trial artifacts."
        ),
    }
    with open(os.path.join(working_dir, "transition_protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2, allow_nan=False)
    all_hr10: list[float] = []
    for ds in DATASETS:
        ### <<< Self_EvolveRec-BLOCK-END
        print(f"\n{'='*55}", flush=True)
        print(f"Training {ds}", flush=True)
        print(f"{'='*55}", flush=True)
        hr10 = train_one_dataset(ds, device, DATA_DIR)
        all_hr10.append(hr10)

    write_predict_script(working_dir)

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    ### >>> Self_EvolveRec-BLOCK-START: Distinguish neural checkpoint diagnostics from host fusion selection
    print(
        f"\nAll datasets done. Neural checkpoint local average HR@10 = {avg:.4f}. "
        "Fused pinned macro validation HR@10 is computed independently by the host.",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


if __name__ == "__main__":
    main()


