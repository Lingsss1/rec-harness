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
### >>> Self_EvolveRec-BLOCK-START: Match the host training default
NUM_EPOCHS = int(os.environ.get("SASREC_EPOCHS", "200"))
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("SASREC_MAXLEN", "128"))
PATIENCE = int(os.environ.get("SASREC_PATIENCE", "20"))

### >>> Self_EvolveRec-BLOCK-START: Directional null-slot residual with unique candidate negatives
DATA_DIR = os.environ["GAGC_DATA_DIR"]
VARIANT = os.environ.get("SASREC_VARIANT", "combined")
if VARIANT not in {"baseline", "loss_only", "residual_only", "combined"}:
    raise ValueError(f"Unknown SASREC_VARIANT: {VARIANT}")
USE_SOFTMAX = VARIANT in {"loss_only", "combined"}
USE_RESIDUAL = VARIANT in {"residual_only", "combined"}
NEGATIVE_COUNT = 64 if USE_SOFTMAX else 1
RECENT_WINDOW = 5  # Retained for checkpoint constructor compatibility.
TEMPERATURE = 1.0
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four dataset routes
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

# DEBUG: Do not reset process-wide RNGs when the generated predictor imports
# train.py. Lazy predictor loading can otherwise disturb evaluator sampling.
# train_one_dataset already seeds Python, NumPy and PyTorch with seed 42.

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


### >>> Self_EvolveRec-BLOCK-START: Batch-vectorized seeded uniform sampling without replacement from training-history complements
class TrainingSampler:
    def __init__(self, user_train, itemnum):
        self.user_train = user_train
        self.itemnum = itemnum
        self.users = [u for u, history in user_train.items() if len(history) >= 2]
        if not self.users:
            raise ValueError("Training requires a user with at least two interactions")
        self.user_rng = random.Random(42)
        self.negative_rng = np.random.default_rng(42)
        self.exclusions = {}
        self.eligible_counts = {}
        for u in self.users:
            forbidden = np.unique(np.asarray(user_train[u], dtype=np.int64))
            if np.any((forbidden < 1) | (forbidden > itemnum)):
                raise ValueError(
                    f"Training history contains an invalid item ID: user={u}"
                )
            eligible = itemnum - len(forbidden)
            if eligible < 1:
                raise ValueError(f"No eligible training negative for user={u}")
            self.eligible_counts[u] = eligible
            self.exclusions[u] = forbidden - np.arange(len(forbidden))
        # A common feasible K keeps every candidate-set denominator rectangular.
        self.negative_count = min(NEGATIVE_COUNT, min(self.eligible_counts.values()))

    def sample(self, batch_size, maxlen):
        seq = np.zeros((batch_size, maxlen), dtype=np.int64)
        pos = np.zeros_like(seq)
        k = self.negative_count
        neg = np.zeros((batch_size, maxlen, k), dtype=np.int64)
        sampled_users = self.user_rng.choices(self.users, k=batch_size)
        lengths = np.empty(batch_size, dtype=np.int64)
        for row, u in enumerate(sampled_users):
            history = self.user_train[u]
            length = min(len(history) - 1, maxlen)
            lengths[row] = length
            seq[row, -length:] = history[-length - 1 : -1]
            pos[row, -length:] = history[-length:]

        # Sample all valid positions together, reducing Python/RNG call overhead.
        eligible = np.repeat(
            np.asarray([self.eligible_counts[u] for u in sampled_users]), lengths
        )
        ranks = np.empty((len(eligible), k), dtype=np.int64)
        for column in range(k):
            upper = eligible - k + column + 1
            draw = self.negative_rng.integers(1, upper + 1)
            if column:
                duplicate = (ranks[:, :column] == draw[:, None]).any(axis=1)
                draw[duplicate] = upper[duplicate]
            ranks[:, column] = draw
        offset = 0
        for row, u in enumerate(sampled_users):
            length = int(lengths[row])
            selected = ranks[offset : offset + length]
            neg[row, -length:] = selected + np.searchsorted(
                self.exclusions[u], selected, side="right"
            )
            offset += length
        return seq, pos, neg if USE_SOFTMAX else neg[..., 0]


def sample_batch(user_train, itemnum, batch_size, maxlen, sampler=None):
    if sampler is None:
        sampler = TrainingSampler(user_train, itemnum)
    return sampler.sample(batch_size, maxlen)


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


class SASRec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        ### >>> Self_EvolveRec-BLOCK-START: Explicit checkpointed residual configuration
        maxlen: int = 128,
        use_residual: bool = False,
        recent_window: int = 5,
    ):
        super().__init__()
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.itemnum = itemnum
        self.use_residual = use_residual
        self.recent_window = recent_window
        ### <<< Self_EvolveRec-BLOCK-END

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

        ### >>> Self_EvolveRec-BLOCK-START: Zero-gated directional projections and nonnegative recency penalty
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        if self.use_residual:
            # Optional parameters do not consume the backbone/dropout RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.history_query = nn.Linear(hidden_units, hidden_units, bias=False)
                self.history_key = nn.Linear(hidden_units, hidden_units, bias=False)
                self.history_value = nn.Linear(hidden_units, hidden_units, bias=False)
                nn.init.xavier_uniform_(self.history_query.weight)
                nn.init.xavier_uniform_(self.history_key.weight)
                nn.init.xavier_uniform_(self.history_value.weight)
            self.recency_beta = nn.Parameter(torch.zeros(()))
            self.residual_alpha = nn.Parameter(torch.zeros(()))
        ### <<< Self_EvolveRec-BLOCK-END

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        # DEBUG: Reject zero-length sequences before attention and final-state
        # indexing. Represent an empty history using at least one padding slot.
        if M < 1 or M > self.maxlen:
            raise ValueError(
                f"Sequence length must be between 1 and {self.maxlen}; got {M}"
            )
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
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unused attention weight materialization
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

        ### >>> Self_EvolveRec-BLOCK-START: Preserve tensor features and explicitly zero padded backbone states
        features = self.last_layernorm(seq_emb)
        features = features * (~timeline_mask).unsqueeze(-1)
        # DEBUG: Preserve tensor compatibility while carrying the complete
        # request-local scorer context. No mutable model history cache is used.
        mask = ~timeline_mask
        if self.use_residual:
            raw = self.item_emb(seq)
            features._sasrec_history_context = (
                self.history_key(raw),
                self.history_value(raw),
                mask,
            )
        else:
            features._sasrec_history_context = (None, None, mask)
        return features
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Request-local context and memory-efficient directional null-slot scoring
    def encode_context(self, seq):
        # DEBUG: Reuse the projections attached by log2feats instead of encoding
        # twice; both public interfaces now use the identical trained scorer.
        features = self.log2feats(seq)
        keys, values, mask = features._sasrec_history_context
        return features, keys, values, mask

    def score_positions(self, features, keys, values, mask, rows, steps, items):
        """Score N explicit prefixes against N x C candidate IDs."""
        selected = features[rows, steps]
        outputs = []
        scale = self.hidden_units**-0.5
        if self.use_residual:
            positions = torch.arange(mask.shape[1], device=mask.device)
            ages = steps[:, None] - positions[None, :]
            allowed = mask[rows] & ages.ge(0)
            penalty = nn.functional.softplus(self.recency_beta) * ages.to(
                features.dtype
            )
            bias = (-penalty).masked_fill(~allowed, float("-inf"))
            # Zero keys yield a fixed zero null logit for every candidate.
            # Zero values preserve exact zero residuals for empty histories.
            history_keys = keys[rows].masked_fill(~mask[rows, :, None], 0.0)
            history_values = values[rows].masked_fill(~mask[rows, :, None], 0.0)
            history_keys = torch.cat(
                (history_keys, torch.zeros_like(history_keys[:, :1])), dim=1
            )
            history_values = torch.cat(
                (history_values, torch.zeros_like(history_values[:, :1])), dim=1
            )
            bias = torch.cat((bias, bias.new_zeros((len(rows), 1))), dim=1)
        for start in range(0, items.shape[1], 32):
            embeddings = self.item_emb(items[:, start : start + 32])
            scores = torch.einsum("nh,nch->nc", selected, embeddings)
            if self.use_residual:
                queries = self.history_query(embeddings)
                # Explicit prefix mask: rectangular is_causal alignment is unused.
                # SDPA can avoid materializing attention weights on supported GPUs.
                attended = nn.functional.scaled_dot_product_attention(
                    queries.unsqueeze(1),
                    history_keys.unsqueeze(1),
                    history_values.unsqueeze(1),
                    attn_mask=bias[:, None, None, :],
                    dropout_p=0.0,
                    is_causal=False,
                ).squeeze(1)
                residual = (embeddings * attended).sum(-1) * scale
                scores = scores + self.residual_alpha * residual
            outputs.append(scores)
        if not outputs:
            return features.new_empty((rows.numel(), 0))
        return torch.cat(outputs, dim=1)

    def predict_score(self, context, items):
        """Score final-prefix candidates from context or intact log2feats output."""
        # DEBUG: The compatibility contract supplies a tensor from log2feats;
        # unpacking that tensor as a four-element context raises an exception
        # or silently interprets batch rows as context when batch size is four.
        if isinstance(context, torch.Tensor):
            features = context
            history_context = getattr(features, "_sasrec_history_context", None)
            # DEBUG: Plain or sliced feature tensors remain sufficient for
            # baseline/loss-only scoring. The residual requires actual history;
            # never silently substitute missing context for that architecture.
            if history_context is None:
                if self.use_residual:
                    raise ValueError(
                        "Residual scoring requires the intact tensor returned "
                        "by log2feats(sequence), or the tuple returned by "
                        "encode_context(sequence). Slicing or copying features "
                        "discards their history context."
                    )
                if features.ndim != 3:
                    raise ValueError("Scoring features must have shape B x L x H")
                keys, values = None, None
                mask = torch.ones(
                    features.shape[:2], dtype=torch.bool, device=features.device
                )
            else:
                keys, values, mask = history_context
        elif isinstance(context, tuple) and len(context) == 4:
            features, keys, values, mask = context
        else:
            raise TypeError("Expected log2feats features or an encoded context tuple")
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError("Scoring features must have shape B x L x H with L >= 1")
        # DEBUG: The compatibility scorer must also accept B x L x C
        # diagnostic candidates. Score each position through the complete
        # causal residual head rather than rejecting them or using dot products.
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        batch_size, length = features.shape[:2]
        if items.ndim not in (1, 2, 3):
            raise ValueError("Candidates must have shape C, B x C, or B x S x C")
        if items.ndim >= 2 and items.shape[0] != batch_size:
            raise ValueError("Candidate and feature batch sizes must match")
        if items.ndim == 1:
            items = items.unsqueeze(0).expand(batch_size, -1)

        if items.ndim == 3:
            count = items.shape[1]
            if count not in (1, length):
                raise ValueError(
                    "Candidate positions must cover the final or every state"
                )
            candidate_count = items.shape[2]
            if batch_size == 0 or candidate_count == 0:
                return features.new_empty(items.shape)
            rows = torch.arange(batch_size, device=features.device).repeat_interleave(
                count
            )
            steps = (
                torch.full_like(rows, length - 1)
                if count == 1
                else torch.arange(length, device=features.device).repeat(batch_size)
            )
            flat_items = items.reshape(batch_size * count, candidate_count)
            pieces = []
            for start in range(0, rows.numel(), 128):
                stop = start + 128
                candidate_chunks = []
                for candidate_start in range(0, candidate_count, 512):
                    candidate_chunks.append(
                        self.score_positions(
                            features,
                            keys,
                            values,
                            mask,
                            rows[start:stop],
                            steps[start:stop],
                            flat_items[
                                start:stop, candidate_start : candidate_start + 512
                            ],
                        )
                    )
                pieces.append(torch.cat(candidate_chunks, dim=1))
            return torch.cat(pieces, dim=0).reshape(items.shape)

        if batch_size == 0 or items.shape[1] == 0:
            return features.new_empty(items.shape)
        rows = torch.arange(batch_size, device=features.device)
        steps = torch.full_like(rows, length - 1)
        chunks = []
        for start in range(0, items.shape[1], 512):
            chunks.append(
                self.score_positions(
                    features,
                    keys,
                    values,
                    mask,
                    rows,
                    steps,
                    items[:, start : start + 512],
                )
            )
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        context = self.encode_context(seq)
        if pos is None or neg is None:
            return context
        features, keys, values, mask = context
        rows, steps = torch.where(seq.ne(0) & pos.ne(0))
        negative_items = neg[rows, steps]
        if neg.ndim == seq.ndim:
            negative_items = negative_items.unsqueeze(-1)
        candidates = torch.cat((pos[rows, steps, None], negative_items), -1)
        pieces = []
        for start in range(0, len(rows), 128):
            stop = start + 128
            pieces.append(
                self.score_positions(
                    features,
                    keys,
                    values,
                    mask,
                    rows[start:stop],
                    steps[start:stop],
                    candidates[start:stop],
                )
            )
        scores = (
            torch.cat(pieces, 0)
            if pieces
            else features.new_empty((0, candidates.shape[1]))
        )
        pos_logits = features.new_zeros(pos.shape)
        pos_logits[rows, steps] = scores[:, 0]
        neg_logits = features.new_zeros((*seq.shape, negative_items.shape[-1]))
        neg_logits[rows, steps] = scores[:, 1:]
        if neg.ndim == seq.ndim:
            neg_logits = neg_logits.squeeze(-1)
        return pos_logits, neg_logits

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # DEBUG: Accept B x 0 empty histories through the public inference API.
        # Use a padding-only prefix so both backbone and residual scores are
        # exactly zero and the existing checkpoint architecture remains valid.
        if seq.ndim != 2:
            raise ValueError("Prediction sequences must have shape B x L")
        if seq.shape[1] == 0:
            seq = seq.new_zeros((seq.shape[0], 1))
        return self.predict_score(self.encode_context(seq), candidates)

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
    ### >>> Self_EvolveRec-BLOCK-START: Stable validation candidates without perturbing training RNG
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
            ### >>> Self_EvolveRec-BLOCK-START: Bound validation sampling and retain unique negatives
            if itemnum - len({i for i in interacted if 1 <= i <= itemnum}) < 99:
                raise ValueError(
                    f"User {u} has fewer than 99 eligible validation negatives"
                )
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

    ### >>> Self_EvolveRec-BLOCK-START: Fail clearly on missing validation and restore model mode
    if not val_users:
        raise ValueError("No eligible validation users")
    n = len(val_users)
    model.train(was_training)
    ### <<< Self_EvolveRec-BLOCK-END
    return {
        "HR@10": HR10 / n,
        "HR@20": HR20 / n,
        "NDCG@10": NDCG10 / n,
        "NDCG@20": NDCG20 / n,
    }


# ── Per-dataset training ──────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str):
    ### >>> Self_EvolveRec-BLOCK-START: Reproducible per-category training and cached sampler
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    sampler = TrainingSampler(user_train, itemnum)
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        ### >>> Self_EvolveRec-BLOCK-START: Activate the selected fusion ablation
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        use_residual=USE_RESIDUAL,
        recent_window=RECENT_WINDOW,
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always save the first evaluated checkpoint and record selection
    best_hr10 = float("-inf")
    best_epoch = 0
    trial_log = []
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Checkpointed prefix groups with one globally normalized batch update
            from torch.utils.checkpoint import checkpoint

            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, sampler=sampler
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
            rows, steps = torch.where(seq_t.ne(0) & pos_t.ne(0))
            if rows.numel() == 0:
                continue
            features, keys, values, mask = model.encode_context(seq_t)
            negative_items = neg_t[rows, steps]
            if negative_items.ndim == 1:
                negative_items = negative_items.unsqueeze(-1)
            candidates = torch.cat((pos_t[rows, steps, None], negative_items), -1)
            losses = []
            for start in range(0, rows.numel(), 128):
                stop = start + 128
                logits = checkpoint(
                    model.score_positions,
                    features,
                    keys,
                    values,
                    mask,
                    rows[start:stop],
                    steps[start:stop],
                    candidates[start:stop],
                    use_reentrant=False,
                )
                if USE_SOFTMAX:
                    group_loss = nn.functional.cross_entropy(
                        logits / TEMPERATURE,
                        torch.zeros(logits.shape[0], dtype=torch.long, device=device),
                        reduction="sum",
                    )
                else:
                    group_loss = nn.functional.binary_cross_entropy_with_logits(
                        logits[:, 0], torch.ones_like(logits[:, 0]), reduction="sum"
                    )
                    group_loss = (
                        group_loss
                        + nn.functional.binary_cross_entropy_with_logits(
                            logits[:, 1],
                            torch.zeros_like(logits[:, 1]),
                            reduction="sum",
                        )
                    )
                losses.append(group_loss)
            loss = torch.stack(losses).sum() / rows.numel()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite loss for {dataset_name}, epoch {epoch}"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / NUM_BATCHES

        ### >>> Self_EvolveRec-BLOCK-START: Validate the final epoch and persist observed validation history
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            hr10 = val_m["HR@10"]
            trial_log.append({"epoch": epoch, "loss": avg_loss, **val_m})
            with open(
                os.path.join(working_dir, f"{dataset_name}_training.json"), "w"
            ) as f:
                json.dump(trial_log, f, indent=2)
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist directional scorer and resolved training configuration
                "maxlen": MAXLEN,
                "use_residual": USE_RESIDUAL,
                "recent_window": RECENT_WINDOW,
                "architecture": "sasrec_directional_null_v1",
                "variant": VARIANT,
                "negative_count": sampler.negative_count,
                "requested_negative_count": NEGATIVE_COUNT,
                "temperature": TEMPERATURE,
                "sampler": "uniform_without_replacement_excluding_full_training_history",
                "negative_seed": 42,
                "training_seed": 42,
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "betas": (0.9, 0.98),
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "num_epochs": NUM_EPOCHS,
                "patience_checks": PATIENCE,
                "checkpoint_rule": "maximum validation HR@10; earliest epoch on ties",
                "best_epoch": best_epoch,
                "best_hr10": best_hr10,
                "prefix_chunk": 128,
                "candidate_chunk": 32,
                "recency": "softplus(beta) times original position age",
                "null_logit": 0.0,
                "null_value": 0.0,
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
                ### <<< Self_EvolveRec-BLOCK-END
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
### >>> Self_EvolveRec-BLOCK-START: Preserve all four inference routes
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

_models: dict = {}
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
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
        ### >>> Self_EvolveRec-BLOCK-START: Report missing artifacts explicitly
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing model artifacts for {ds}: {meta_path}, {ckpt_path}")
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[ds] = meta
        # DEBUG: Checkpoint reconstruction initializes temporary CPU parameters.
        # Preserve the caller's RNG state while rebuilding the trained model.
        with torch.random.fork_rng(devices=[]):
            m = SASRec(
                itemnum=meta["itemnum"],
                hidden_units=meta["hidden_units"],
                num_blocks=meta["num_blocks"],
                num_heads=meta["num_heads"],
                ### >>> Self_EvolveRec-BLOCK-START: Load the identical directional null-slot architecture
                dropout_rate=0.0,
                maxlen=meta["maxlen"],
                use_residual=meta["use_residual"],
                recent_window=meta["recent_window"],
                ### <<< Self_EvolveRec-BLOCK-END
            ).to(_device)
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject invalid explicit routes
    if dataset_name is not None and dataset_name not in _models:
        raise ValueError(f"Unknown dataset route: {dataset_name}")
    ### <<< Self_EvolveRec-BLOCK-END
    # Select model: use dataset_name kwarg if provided, else fall back to first available
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta  = _metas[dataset_name]
    elif _models:
        ds    = next(iter(_models))
        model = _models[ds]
        meta  = _metas[ds]
    ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when inference artifacts are unavailable
    else:
        raise RuntimeError("No dataset models loaded")
    ### <<< Self_EvolveRec-BLOCK-END

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

    ### >>> Self_EvolveRec-BLOCK-START: Verify saved inference against the complete candidate-conditioned scorer
    write_predict_script(working_dir)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "saved_predictor_check", os.path.join(working_dir, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor._ensure_loaded()
    for ds in DATASETS:
        meta = predictor._metas[ds]
        reference = SASRec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
            use_residual=meta["use_residual"],
            recent_window=meta["recent_window"],
        ).to(device)
        reference.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{ds}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        reference.eval()
        histories = _read_split(os.path.join(DATA_DIR, f"{ds}_train.txt"))
        for u in sorted(histories)[:5]:
            history = histories[u][-meta["maxlen"] :]
            candidates = list(range(1, min(meta["itemnum"], 100) + 1))
            seq = torch.tensor(
                [[0] * (meta["maxlen"] - len(history)) + history],
                dtype=torch.long,
                device=device,
            )
            items = torch.tensor([candidates], dtype=torch.long, device=device)
            with torch.no_grad():
                context = reference.encode_context(seq)
                expected = reference.predict_score(context, items)[0].cpu()
                actual = torch.tensor(
                    predictor.predict(u, history, candidates, dataset_name=ds),
                    dtype=expected.dtype,
                )
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                pieces = [
                    reference.predict_score(context, c) for c in items.split(17, dim=1)
                ]
                torch.testing.assert_close(
                    torch.cat(pieces, 1)[0].cpu(), expected, rtol=1e-4, atol=1e-4
                )
                permutation = torch.arange(items.shape[1] - 1, -1, -1, device=device)
                reordered = reference.predict_score(context, items[:, permutation])
                torch.testing.assert_close(
                    reordered[0].cpu(), expected.flip(0), rtol=1e-4, atol=1e-4
                )
                empty = reference.predict(torch.zeros_like(seq), items)
                torch.testing.assert_close(
                    empty, torch.zeros_like(empty), rtol=0, atol=0
                )
                # Intervening requests must not mutate the saved context.
                torch.testing.assert_close(
                    reference.predict_score(context, items)[0].cpu(),
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                )
        del reference
    print("Saved predictor agrees with the complete directional scorer", flush=True)
    avg = sum(all_hr10) / max(len(all_hr10), 1)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


