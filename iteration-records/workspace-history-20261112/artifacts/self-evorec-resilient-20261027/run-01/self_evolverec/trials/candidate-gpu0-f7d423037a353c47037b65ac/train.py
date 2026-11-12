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

### >>> Self_EvolveRec-BLOCK-START: Track feature contexts by weak tensor identity
import random
import weakref
from types import MappingProxyType
from collections import defaultdict

### <<< Self_EvolveRec-BLOCK-END

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
### >>> Self_EvolveRec-BLOCK-START: Honor the matched benchmark epoch default
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
### >>> Self_EvolveRec-BLOCK-START: Retain all four required dataset routes
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


### >>> Self_EvolveRec-BLOCK-START: Cache eligible training users and prevent impossible negative rejection loops
def sample_batch(
    user_train: dict, itemnum: int, batch_size: int, maxlen: int, users=None
):
    if users is None:
        users = [
            u
            for u, items in user_train.items()
            if len(items) >= 2 and len(set(items)) < itemnum
        ]
    if not users:
        raise ValueError(
            "No training users have both next-item targets and eligible negatives"
        )
    ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Configure an isolated recent-context residual under the baseline BCE objective
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        maxlen: int = 128,
        recent_residual: bool = True,
        dataset_name: str | None = None,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
        super().__init__()
        # DEBUG: diagnostics require routing identity, which was never assigned.
        if dataset_name is not None and dataset_name not in DEFAULT_DATASETS:
            raise ValueError(f"Unsupported dataset route: {dataset_name!r}")
        self.dataset_name = dataset_name
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

        ### >>> Self_EvolveRec-BLOCK-START: Initialize isolated candidate-conditioned scoring with an exactly zero gate
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.recent_residual = bool(recent_residual)
        self.recent_window = 5
        self._feature_contexts = {}
        if self.recent_residual:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(1042)
                self.recent_norm_h = nn.LayerNorm(hidden_units, eps=1e-5)
                self.recent_norm_i = nn.LayerNorm(hidden_units, eps=1e-5)
                self.recent_norm_k = nn.LayerNorm(hidden_units, eps=1e-5)
                self.recent_norm_v = nn.LayerNorm(hidden_units, eps=1e-5)
                self.recent_wh = nn.Linear(hidden_units, hidden_units, bias=False)
                self.recent_wi = nn.Linear(hidden_units, hidden_units, bias=False)
                self.recent_wk = nn.Linear(hidden_units, hidden_units, bias=False)
                self.recent_wv = nn.Linear(hidden_units, hidden_units, bias=False)
                for layer in (
                    self.recent_wh,
                    self.recent_wi,
                    self.recent_wk,
                    self.recent_wv,
                ):
                    nn.init.xavier_uniform_(layer.weight)
            self.recent_alpha = nn.Parameter(torch.zeros(()))
        ### <<< Self_EvolveRec-BLOCK-END

    def log2feats(self, seq: torch.Tensor) -> torch.Tensor:
        B, M = seq.shape
        seq_emb = self.item_emb(seq) * (self.hidden_units**0.5)
        ### >>> Self_EvolveRec-BLOCK-START: Preserve the original SASRec feature path
        # Recent evidence is used only by the candidate-conditioned scorer.
        ### <<< Self_EvolveRec-BLOCK-END

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

        ### >>> Self_EvolveRec-BLOCK-START: Bind immutable causal evidence to unchanged backbone features
        features = self.last_layernorm(seq_emb)
        offsets = torch.arange(self.recent_window, device=seq.device)
        positions = torch.arange(M, device=seq.device)
        source_positions = positions[:, None] - offsets[None, :]
        window_ids = seq[:, source_positions.clamp_min(0)]
        valid = (
            source_positions.ge(0).unsqueeze(0)
            & window_ids.ne(0)
            & seq.ne(0).unsqueeze(-1)
        )
        self._bind_context(
            features,
            window_ids=window_ids.clone(),
            valid=valid,
            gaps=offsets.to(features.dtype),
            positions=positions,
        )
        return features
        ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Validate candidate contexts under no-grad and inference-mode execution
    @staticmethod
    def _tensor_version(tensor):
        # Inference tensors deliberately have no version counter.
        return None if torch.is_inference(tensor) else tensor._version

    def _weight_versions(self):
        return tuple(
            (id(p), self._tensor_version(p), p.device, p.dtype)
            for p in self.parameters()
        )

    def _bind_context(self, features, **fields):
        key = id(features)
        owner = weakref.ref(self)

        def release(reference):
            model = owner()
            if model is not None:
                registered = model._feature_contexts.get(key)
                if registered is not None and registered["source"] is reference:
                    model._feature_contexts.pop(key, None)

        context = MappingProxyType(
            dict(
                fields,
                source=weakref.ref(features, release),
                owner=id(self),
                shape=tuple(features.shape),
                device=features.device,
                dtype=features.dtype,
                feature_version=self._tensor_version(features),
                field_versions=tuple(
                    (name, id(value), self._tensor_version(value))
                    for name, value in fields.items()
                    if isinstance(value, torch.Tensor)
                ),
                weight_versions=self._weight_versions(),
            )
        )
        self._feature_contexts[key] = context
        return context

    def _require_context(self, features, context=None):
        if context is None:
            context = self._feature_contexts.get(id(features))
        if context is None:
            raise ValueError("Unknown feature tensor; use encode/select_positions")
        if (
            context["source"]() is not features
            or context["owner"] != id(self)
            or context["shape"] != tuple(features.shape)
            or context["device"] != features.device
            or context["dtype"] != features.dtype
            or context["feature_version"] != self._tensor_version(features)
            or any(
                id(context[name]) != identity
                or self._tensor_version(context[name]) != version
                for name, identity, version in context["field_versions"]
            )
            or context["weight_versions"] != self._weight_versions()
        ):
            raise ValueError("Feature/context identity, ownership or weight mismatch")
        return context

    def encode(self, sequence):
        features = self.log2feats(sequence)
        return features, self._require_context(features)

    def select_positions(self, features, context, positions):
        context = self._require_context(features, context)
        if isinstance(positions, int):
            length = features.shape[1]
            if not -length <= positions < length:
                raise IndexError("Selected sequence position is out of range")
            position = positions % length
            positions = slice(position, position + 1)
        selected = features[:, positions, :]
        if selected.ndim != 3:
            raise ValueError("Position selection must retain B x L x D axes")
        selected_context = self._bind_context(
            selected,
            window_ids=context["window_ids"][:, positions, :],
            valid=context["valid"][:, positions, :],
            gaps=context["gaps"],
            positions=context["positions"][positions],
        )
        return selected, selected_context

    def score_candidates(self, features, items, context=None):
        context = self._require_context(features, context)
        single = items.ndim == 2
        if single:
            items = items.unsqueeze(-1)
        if items.ndim != 3 or items.shape[:2] != features.shape[:2]:
            raise ValueError("Candidates must have shape B x L or B x L x K")
        if items.device != features.device:
            raise ValueError("Candidates and features must share a device")
        embeddings = self.item_emb(items)
        baseline = (features.unsqueeze(-2) * embeddings).sum(-1)
        if not self.recent_residual or items.shape[-1] == 0:
            return baseline.squeeze(-1) if single else baseline

        values = self.item_emb(context["window_ids"])
        keys = self.recent_wk(self.recent_norm_k(values))
        candidate_features = self.recent_norm_i(embeddings)
        query = self.recent_wh(self.recent_norm_h(features)).unsqueeze(
            -2
        ) + self.recent_wi(candidate_features)
        scale = self.hidden_units**-0.5
        logits = torch.einsum("blkd,blwd->blkw", query, keys) * scale
        logits = logits - context["gaps"]
        valid = context["valid"].unsqueeze(-2)
        has_context = valid.any(-1, keepdim=True)
        logits = logits.masked_fill(~valid, -torch.inf)
        # Empty rows enter softmax as finite zeros, then are explicitly masked.
        logits = torch.where(has_context, logits, torch.zeros_like(logits))
        attention = torch.softmax(logits, dim=-1).masked_fill(~valid, 0.0)
        pooled = torch.einsum("blkw,blwd->blkd", attention, values)
        pooled = self.recent_norm_v(self.recent_wv(pooled))
        branch = (candidate_features * pooled).sum(-1) * scale
        branch = branch.masked_fill(~has_context.squeeze(-1) | items.eq(0), 0.0)
        total = baseline + self.recent_alpha.tanh() * branch
        return total.squeeze(-1) if single else total

    def predict_score(self, features, items, context=None):
        return self.score_candidates(features, items, context)

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor | None = None,
        neg: torch.Tensor | None = None,
    ):
        features, context = self.encode(seq)
        if pos is None or neg is None:
            return features
        scores = self.score_candidates(
            features, torch.stack((pos, neg), dim=-1), context
        )
        return scores[..., 0], scores[..., 1]

    def predict(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 2 or candidates.shape[0] != seq.shape[0]:
            raise ValueError("Prediction candidates must have shape B x K")
        features, context = self.encode(seq)
        final, final_context = self.select_positions(features, context, -1)
        # Every candidate is scored; attention normalizes over history slots.
        pieces = [
            self.score_candidates(
                final, candidates[:, None, start : start + 256], final_context
            )[:, 0, :]
            for start in range(0, candidates.shape[1], 256)
        ]
        if not pieces:
            return features.new_empty((seq.shape[0], 0))
        return torch.cat(pieces, dim=1)

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
    ### >>> Self_EvolveRec-BLOCK-START: Fix validation randomness independently of training and count only evaluable users
    rng = random.Random(42)
    val_users = [
        u for u in user_valid if u in user_train and user_train[u] and user_valid[u]
    ]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    if not val_users:
        raise ValueError("Validation contains no users with histories and targets")

    HR10 = HR20 = NDCG10 = NDCG20 = 0.0
    was_training = model.training
    model.eval()
    ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Preserve unique validation negatives and fail clearly for insufficient catalogs
            interacted = set(user_train[u]) | set(pos_items)
            excluded_count = sum(1 for item in interacted if 1 <= item <= itemnum)
            if itemnum - excluded_count < 99:
                raise ValueError(
                    f"Validation user {u} has fewer than 99 eligible negatives"
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

    ### >>> Self_EvolveRec-BLOCK-START: Restore the caller's model mode after validation
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
    ### >>> Self_EvolveRec-BLOCK-START: Initialize each category reproducibly and prepare training-only sampler eligibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    eligible_users = [
        u
        for u, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    if not eligible_users:
        raise ValueError(
            f"{dataset_name}: no users with targets and eligible negatives"
        )
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    ### <<< Self_EvolveRec-BLOCK-END

    model = SASRec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        ### >>> Self_EvolveRec-BLOCK-START: Invoke the residual experiment with an explicit baseline switch
        # DEBUG: expose the actual training category to architecture-aware probes.
        recent_residual=os.environ.get("SASREC_RECENT_RESIDUAL", "1") == "1",
        dataset_name=dataset_name,
        ### <<< Self_EvolveRec-BLOCK-END
    ).to(device)

    ### >>> Self_EvolveRec-BLOCK-START: Verify baseline initialization and candidate-scoring contracts before fitting
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        canonical = SASRec(
            itemnum=itemnum,
            hidden_units=HIDDEN_UNITS,
            num_blocks=NUM_BLOCKS,
            num_heads=NUM_HEADS,
            dropout_rate=DROPOUT_RATE,
            maxlen=MAXLEN,
            recent_residual=False,
        )
    shared = canonical.state_dict()
    current = model.state_dict()
    for name, value in shared.items():
        if not torch.equal(current[name].detach().cpu(), value):
            raise AssertionError(f"Backbone initialization mismatch: {name}")
        current[name].copy_(value.to(current[name].device))
    del canonical

    model.eval()
    with torch.no_grad():
        length = min(MAXLEN, 8)
        sequence = torch.zeros((2, length), dtype=torch.long, device=device)
        if length > 1:
            sequence[0, 1:] = torch.arange(length - 1, device=device) % itemnum + 1
        candidates = (torch.arange(9, device=device).remainder(itemnum) + 1).expand(
            2, -1
        )
        features, context = model.encode(sequence)
        final, final_context = model.select_positions(features, context, -1)
        original = model.score_candidates(final, candidates[:, None], final_context)
        base = (final.unsqueeze(-2) * model.item_emb(candidates[:, None])).sum(-1)
        torch.testing.assert_close(original, base, rtol=0, atol=0)

        if model.recent_residual:
            model.recent_alpha.fill_(0.3)
            # Re-encode after every parameter mutation: stale contexts must fail.
            try:
                model.score_candidates(final, candidates[:, None], final_context)
            except ValueError:
                pass
            else:
                raise AssertionError("Stale context was accepted")
            features, context = model.encode(sequence)
            final, final_context = model.select_positions(features, context, -1)
            complete = model.score_candidates(final, candidates[:, None], final_context)
            for chunk_size in (1, 7, 9):
                chunked = torch.cat(
                    [
                        model.score_candidates(
                            final,
                            candidates[:, None, start : start + chunk_size],
                            final_context,
                        )
                        for start in range(0, 9, chunk_size)
                    ],
                    dim=-1,
                )
                torch.testing.assert_close(complete, chunked, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(
                complete[1], torch.zeros_like(complete[1]), rtol=0, atol=0
            )
            # Interleaved encodes must not replace the first sequence's context.
            changed = sequence.clone()
            prefix = max(1, length // 2)
            changed[:, prefix:] = itemnum
            changed_features, changed_context = model.encode(changed)
            left, left_context = model.select_positions(
                features, context, slice(0, prefix)
            )
            right, right_context = model.select_positions(
                changed_features, changed_context, slice(0, prefix)
            )
            probe_items = candidates[:, :1, None].expand(-1, prefix, -1)
            torch.testing.assert_close(
                model.score_candidates(left, probe_items, left_context),
                model.score_candidates(right, probe_items, right_context),
                rtol=1e-4,
                atol=1e-5,
            )
            model.recent_alpha.zero_()
    model._feature_contexts.clear()
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    ### <<< Self_EvolveRec-BLOCK-END
    bce_criterion = nn.BCEWithLogitsLoss()

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always save the first evaluated checkpoint and retain earliest ties
    best_hr10 = float("-inf")
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Reuse the eligible-user list for every batch
            seq, pos, neg = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, users=eligible_users
            )
            ### <<< Self_EvolveRec-BLOCK-END
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

        ### >>> Self_EvolveRec-BLOCK-START: Use the declared twenty-epoch local checkpoint cadence
        if epoch % 20 == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
            ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Keep checkpoint opportunities independent of metric-dependent stopping
            # Host accounting remains authoritative for the shared search budget.
            # This local monitor does not invoke or replace the host evaluator.
            ### <<< Self_EvolveRec-BLOCK-END
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
                # DEBUG: persist category identity alongside architecture metadata.
                "dataset_name": dataset_name,
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                "maxlen": MAXLEN,
                ### >>> Self_EvolveRec-BLOCK-START: Persist candidate-attention architecture and unchanged training objective
                "recent_residual": model.recent_residual,
                "recent_window": model.recent_window,
                "scorer": "candidate_recent_attention_v1",
                "objective": "baseline_bce",
                "training_seed": 42,
                "preprocessing_seed": 0,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "batches_per_epoch": NUM_BATCHES,
                "requested_epochs": NUM_EPOCHS,
                "checkpoint_epoch": best_epoch,
                "local_validation_HR@10": best_hr10,
                "checkpoint_monitor": "local validation; host evaluates independently",
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
### >>> Self_EvolveRec-BLOCK-START: Load every required category in the generated predictor
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
    ### >>> Self_EvolveRec-BLOCK-START: Retry incomplete predictor initialization explicitly
    if len(_models) == len(DATASETS):
        return
    _models.clear()
    _metas.clear()
    ### <<< Self_EvolveRec-BLOCK-END
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SASRec = _load_model_class()
    for ds in DATASETS:
        meta_path = os.path.join(_HERE, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(_HERE, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required category artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing required artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        # DEBUG: reject artifacts explicitly labeled for a different category.
        if meta.get("dataset_name", ds) != ds:
            raise ValueError(f"Checkpoint metadata route mismatch for {ds}")
        _metas[ds] = meta
        m = SASRec(
            dataset_name=ds,
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved backbone configuration
            dropout_rate=meta["dropout_rate"],
            ### <<< Self_EvolveRec-BLOCK-END
            maxlen=meta["maxlen"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the checkpoint architecture in the generated predictor
            recent_residual=meta.get("recent_residual", False),
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        ### >>> Self_EvolveRec-BLOCK-START: Require the matching candidate scorer and verify serialized reconstruction
        if meta.get("scorer") != "candidate_recent_attention_v1":
            raise ValueError(f"Unsupported scorer for {ds}")
        if meta.get("recent_window") != m.recent_window:
            raise ValueError(f"Recent-window mismatch for {ds}")
        m.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        m.eval()
        _models[ds] = m
        ### <<< Self_EvolveRec-BLOCK-END


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()

    ### >>> Self_EvolveRec-BLOCK-START: Reject missing explicit routes instead of scoring with another category
    if dataset_name is not None:
        if dataset_name not in _models:
            raise ValueError(f"No trained model for dataset {dataset_name!r}")
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
    ### <<< Self_EvolveRec-BLOCK-END
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

    write_predict_script(working_dir)

    avg = sum(all_hr10) / max(len(all_hr10), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


