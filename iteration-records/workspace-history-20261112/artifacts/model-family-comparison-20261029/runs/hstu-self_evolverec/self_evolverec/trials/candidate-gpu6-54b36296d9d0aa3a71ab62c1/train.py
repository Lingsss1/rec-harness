"""ID-only HSTU-style per-dataset template for Amazon Reviews.

Trains one sequential recommender per dataset and writes checkpoints under
``./working``.  The public benchmark contract remains unchanged:
``predict(user_id, history, candidates, *, dataset_name=None) -> list[float]``.

This is intentionally ID-only: no text/image/category metadata is loaded, so it
is directly comparable to SASRec/GRU4Rec/BERT4Rec per-dataset templates.
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
import torch.nn.functional as F

# ── Hyper-parameters ──────────────────────────────────────────────────
HIDDEN_UNITS = int(os.environ.get("HSTU_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("HSTU_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("HSTU_HEADS", "2"))
DROPOUT_RATE = float(os.environ.get("HSTU_DROPOUT", "0.3"))
LR = float(os.environ.get("HSTU_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("HSTU_WD", "1e-6"))
BATCH_SIZE = int(os.environ.get("HSTU_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Fixed training budget and inherited epoch default
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200, int(os.environ.get("HSTU_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
)
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("HSTU_MAXLEN", "128"))
PATIENCE = int(os.environ.get("HSTU_PATIENCE", "20"))
EVAL_USERS = int(os.environ.get("HSTU_EVAL_USERS", "200"))

### >>> Self_EvolveRec-BLOCK-START: Preserve host data boundary
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Preserve all four category routes
DATASETS = list(DEFAULT_DATASETS)
### <<< Self_EvolveRec-BLOCK-END

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Data loading ──────────────────────────────────────────────────────


def _read_split(path: str) -> dict[int, list[int]]:
    data: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data[int(parts[0])].append(int(parts[1]))
    return dict(data)


def _load_itemnum(data_dir: str, dataset_name: str) -> int:
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            return int(stats[dataset_name]["itemnum"])
    itemnum = 0
    ### >>> Self_EvolveRec-BLOCK-START: Catalog fallback uses permitted splits only
    for split in ("train", "valid"):
        ### <<< Self_EvolveRec-BLOCK-END
        path = os.path.join(data_dir, f"{dataset_name}_{split}.txt")
        if not os.path.isfile(path):
            continue
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


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    users = [user for user, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No users with at least two training interactions")
    batch_seq, batch_pos, batch_neg = [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        history = user_train[user]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg_item = random.randint(1, itemnum)
                while neg_item in interacted:
                    neg_item = random.randint(1, itemnum)
                neg[idx] = neg_item
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


### >>> Self_EvolveRec-BLOCK-START: Deterministic public-title assets and jointly trained positional CNN
import re
import time
import unicodedata
from collections import Counter

TITLE_PATTERN = r"[^\W_]+(?:[./-][^\W_]+)*|[^\w\s]"
TITLE_MAXLEN = 32


def build_title_assets(data_dir, dataset_name, itemnum):
    """Accept only explicitly internally keyed public-title metadata.

    HSTU_TITLE_METADATA may contain a JSON path with a {dataset} placeholder.
    Supported JSON is {internal_item_id: title_or_metadata}. A top-level
    category dictionary is also accepted. No ASIN-to-internal-ID mapping is
    inferred, and no reviews or non-title metadata fields are consumed.
    """
    configured = os.environ.get("HSTU_TITLE_METADATA")
    if configured:
        paths = [configured.format(dataset=dataset_name)]
    else:
        paths = [
            os.path.join(data_dir, f"{dataset_name}_{suffix}.json")
            for suffix in ("titles", "metadata", "meta")
        ]
    source = next((path for path in paths if os.path.isfile(path)), None)
    records = {}
    if source is not None:
        with open(source, encoding="utf-8") as stream:
            records = json.load(stream)
        if isinstance(records, dict) and dataset_name in records:
            records = records[dataset_name]
        if not isinstance(records, dict):
            raise ValueError("Title metadata must map internal item IDs to records")
    elif configured:
        raise FileNotFoundError(paths[0])

    documents, frequency = {}, Counter()
    for key, record in records.items():
        # A numeric key must explicitly denote an internal catalog ID.
        # Reject opaque identifiers rather than guessing a remapping.
        try:
            item = int(key)
        except (TypeError, ValueError):
            continue
        if not 1 <= item <= itemnum:
            continue
        title = record.get("title", "") if isinstance(record, dict) else record
        if not isinstance(title, str):
            continue
        tokens = re.findall(TITLE_PATTERN, unicodedata.normalize("NFKC", title).lower())
        if tokens:
            documents[item] = tokens
            frequency.update(set(tokens))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    retained = sorted(
        (token for token, count in frequency.items() if count >= 2),
        key=lambda token: (-frequency[token], token),
    )[:49998]
    vocabulary.update({token: index + 2 for index, token in enumerate(retained)})
    tokens = torch.zeros(itemnum + 1, TITLE_MAXLEN, dtype=torch.long)
    original_lengths = torch.zeros(itemnum + 1, dtype=torch.long)
    for item, document in sorted(documents.items()):
        original_lengths[item] = len(document)
        ids = [vocabulary.get(token, 1) for token in document[:TITLE_MAXLEN]]
        tokens[item, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    specification = {
        "source": source,
        "scope": "Supplied public titles keyed by internal catalog ID; no interaction labels",
        "normalization": "Unicode NFKC then lowercase",
        "pattern": TITLE_PATTERN,
        "max_tokens": TITLE_MAXLEN,
        "minimum_document_frequency": 2,
        "vocabulary": vocabulary,
        "title_vocab_size": len(vocabulary),
        "items_with_titles": len(documents),
        "catalog_items": itemnum,
        "missing_title_fraction": 1.0 - len(documents) / max(1, itemnum),
        "truncated_titles": int(original_lengths.gt(TITLE_MAXLEN).sum()),
        "intervention_active": bool(documents),
    }
    if not documents:
        print(
            f"[{dataset_name}] No usable internally keyed titles; title residual is zero. "
            "This run cannot establish the effect of the proposed title encoder.",
            flush=True,
        )
    return tokens, original_lengths, specification


class PositionalTitleEncoder(nn.Module):
    def __init__(self, vocabulary_size, hidden_units):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(32, 32, width) for width in (1, 2, 3)]
        )
        self.projection = nn.Linear(384, hidden_units, bias=False)
        self.normalization = nn.LayerNorm(hidden_units, eps=1e-5)
        self.gate = nn.Parameter(torch.tensor(-3.0))
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, tokens):
        length = tokens.ne(0).sum(-1)
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for width, convolution in zip((1, 2, 3), self.convolutions):
            activation = F.relu(convolution(embedded))
            starts = torch.arange(activation.shape[-1], device=tokens.device)
            valid = tokens.unfold(1, width, 1).ne(0).all(-1)
            # Integer form of floor(4 * (s + (h-1)/2) / n).
            bins = (
                (4 * (2 * starts[None, :] + width - 1))
                // (2 * length.clamp_min(1)[:, None])
            ).clamp_max(3)
            for position_bin in range(4):
                selected = valid & bins.eq(position_bin)
                values = activation.masked_fill(~selected[:, None, :], -float("inf"))
                values = values.amax(-1)
                values = torch.where(
                    selected.any(-1)[:, None], values, torch.zeros_like(values)
                )
                pooled.append(values)
        residual = self.normalization(self.projection(torch.cat(pooled, dim=-1)))
        # LayerNorm affine parameters must never create a missing-title signal.
        return residual.masked_fill(length.eq(0)[:, None], 0.0)


### <<< Self_EvolveRec-BLOCK-END

# ── HSTU-style model ──────────────────────────────────────────────────


class HSTUBlock(nn.Module):
    """A compact HSTU-style block: causal attention + SiLU gates + residual FFN."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_units)
        self.attn = nn.MultiheadAttention(
            hidden_units,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.gate = nn.Linear(hidden_units, hidden_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        seq_emb: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.norm(seq_emb)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        value, gate = self.gate(attn_out).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb


class HSTURec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        ### >>> Self_EvolveRec-BLOCK-START: Persist title architecture configuration
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
        title_vocab_size: int = 2,
        title_tokens=None,
        title_original_lengths=None,
    ):
        ### <<< Self_EvolveRec-BLOCK-END
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError(
                f"hidden_units={hidden_units} must be divisible by num_heads={num_heads}"
            )
        self.hidden_units = hidden_units
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(hidden_units, num_heads, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.last_norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

        ### >>> Self_EvolveRec-BLOCK-START: Near-zero tied title residual with isolated initialization
        # The backbone is initialized first. Only the new branch consumes this
        # separate CPU RNG stream; CUDA and subsequent training RNGs are unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.title_encoder = PositionalTitleEncoder(title_vocab_size, hidden_units)
        if title_tokens is None:
            title_tokens = torch.zeros(itemnum + 1, TITLE_MAXLEN, dtype=torch.long)
        if title_original_lengths is None:
            title_original_lengths = torch.zeros(itemnum + 1, dtype=torch.long)
        self.register_buffer("title_tokens", title_tokens.clone())
        self.register_buffer("title_original_lengths", title_original_lengths.clone())
        self.register_buffer("title_missing", title_tokens.ne(0).sum(-1).eq(0))
        self._joint_cache = None

    def train(self, mode=True):
        self._joint_cache = None
        return super().train(mode)

    def _joint_uncached(self, items):
        unique, inverse = torch.unique(
            items.reshape(-1), sorted=True, return_inverse=True
        )
        vectors = self.item_emb(unique)
        present = unique.ne(0) & ~self.title_missing[unique]
        selected = present.nonzero(as_tuple=False).flatten()
        if selected.numel():
            # Recompute CNN activations during backward to bound retained memory.
            # The encoder contains no dropout or other stochastic operations.
            from torch.utils.checkpoint import checkpoint

            residuals = []
            for indices in selected.split(1024):
                tokens = self.title_tokens[unique[indices]]
                if self.training and torch.is_grad_enabled():
                    residuals.append(
                        checkpoint(
                            self.title_encoder,
                            tokens,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    )
                else:
                    residuals.append(self.title_encoder(tokens))
            residual = torch.zeros_like(vectors).index_copy(
                0, selected, torch.cat(residuals)
            )
            vectors = vectors + self.title_encoder.gate.sigmoid() * residual
        vectors = vectors.masked_fill(unique.eq(0)[:, None], 0.0)
        return vectors[inverse].reshape(*items.shape, self.hidden_units)

    def joint_item(self, items):
        # This cache is derived exclusively from this model's current parameters.
        # It is never used during gradient-enabled execution and is invalidated
        # on every train/eval transition and state-dict load.
        # CPU parity checks need only their requested items; avoid encoding the
        # entire catalog there. GPU evaluation retains its reusable catalog cache.
        if (
            not self.training
            and not torch.is_grad_enabled()
            and self.item_emb.weight.device.type == "cuda"
        ):
            if self._joint_cache is None:
                pieces = []
                for start in range(0, self.itemnum + 1, 1024):
                    ids = torch.arange(
                        start,
                        min(start + 1024, self.itemnum + 1),
                        device=self.item_emb.weight.device,
                    )
                    pieces.append(self._joint_uncached(ids))
                self._joint_cache = torch.cat(pieces).detach()
            return F.embedding(items, self._joint_cache)
        return self._joint_uncached(items)

    def _apply(self, fn, recurse=True):
        # Derived tensors are not registered buffers. Invalidate them before
        # device/dtype conversion so a later lookup cannot reuse a stale cache.
        self._joint_cache = None
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        self._joint_cache = None
        return super().load_state_dict(*args, **kwargs)

    def log2feats(self, sequence):
        return self(sequence)

    def predict_score(self, features, items):
        vectors = self.joint_item(items)
        if items.ndim == 1:
            return torch.matmul(features, vectors.transpose(-1, -2))
        if items.shape == features.shape[:-1]:
            return (features * vectors).sum(-1)
        if items.shape[:-1] != features.shape[:-1]:
            raise ValueError(
                f"Incompatible feature/item shapes: {features.shape}, {items.shape}"
            )
        return (features.unsqueeze(-2) * vectors).sum(-1)

    @staticmethod
    def final_features(features, sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        last = positions.expand_as(sequence).masked_fill(sequence.eq(0), -1).amax(-1)
        rows = torch.arange(sequence.shape[0], device=sequence.device)
        selected = features[rows, last.clamp_min(0)]
        return selected.masked_fill(last.lt(0)[:, None], 0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        seq_emb = self.joint_item(seq) * math.sqrt(self.hidden_units)
        ### <<< Self_EvolveRec-BLOCK-END
        pos = (
            torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            + 1
        )
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        ### >>> Self_EvolveRec-BLOCK-START: Safe causal HSTU attention for left-padded histories
        padding_mask = seq.eq(0)
        blocked = (
            torch.triu(
                torch.ones(maxlen, maxlen, dtype=torch.bool, device=seq.device),
                diagonal=1,
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        blocked |= padding_mask.unsqueeze(1)
        diagonal = torch.arange(maxlen, device=seq.device)
        # Padding queries get one legal key; their outputs are erased below.
        blocked[:, diagonal, diagonal] &= ~padding_mask
        heads = self.blocks[0].attn.num_heads if len(self.blocks) else 1
        causal_mask = blocked.repeat_interleave(heads, dim=0)
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            seq_emb = block(seq_emb, causal_mask, None)
            seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.last_norm(seq_emb).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END


# ── Validation ────────────────────────────────────────────────────────


def evaluate_val(
    model: HSTURec,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int = 128,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Validation RNG isolation
    rng = random.Random(42)
    val_users = [user for user in user_valid if user in user_train and user_valid[user]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            pos_items = user_valid[user]
            if not pos_items:
                continue
            target = pos_items[0]
            hist = user_train[user][-maxlen:]
            seq_arr = np.zeros(maxlen, dtype=np.int64)
            for idx, item in enumerate(reversed(hist)):
                seq_arr[maxlen - 1 - idx] = item
            interacted = set(user_train[user]) | set(pos_items)
            neg_pool = [
                item for item in range(1, itemnum + 1) if item not in interacted
            ]
            ### >>> Self_EvolveRec-BLOCK-START: Match validation candidate count
            if len(neg_pool) > 99:
                neg_pool = rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            out = model(seq_t)
            ### >>> Self_EvolveRec-BLOCK-START: Tied joint scoring and final nonpadding readout
            user_emb = model.final_features(out, seq_t)[0]
            scores = model.predict_score(user_emb, cand_t)
            ### <<< Self_EvolveRec-BLOCK-END
            rank = int(torch.argsort(scores, descending=True).tolist().index(0))
            if rank < 10:
                hr10 += 1
                ndcg10 += 1.0 / math.log2(rank + 2)
            if rank < 20:
                hr20 += 1
                ndcg20 += 1.0 / math.log2(rank + 2)
    denom = len(val_users)
    return {
        "HR@10": hr10 / denom,
        "HR@20": hr20 / denom,
        "NDCG@10": ndcg10 / denom,
        "NDCG@20": ndcg20 / denom,
    }


# ── Training ──────────────────────────────────────────────────────────


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Per-category seeded initialization and recorded title assets
    started = time.monotonic()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise ValueError(f"[{dataset_name}] Missing required training data")
    if NUM_EPOCHS < 5 or not any(user_valid.get(user) for user in user_train):
        raise ValueError(
            f"[{dataset_name}] Missing validation targets or fewer than five epochs"
        )
    if any(
        len(set(history)) >= itemnum
        for history in user_train.values()
        if len(history) >= 2
    ):
        raise ValueError(
            "Baseline negative sampler requires an unseen catalog item per training user"
        )
    title_tokens, title_lengths, title_spec = build_title_assets(
        data_dir, dataset_name, itemnum
    )
    model = HSTURec(
        ### <<< Self_EvolveRec-BLOCK-END
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Activate title branch in the single training model
        maxlen=MAXLEN,
        title_vocab_size=title_spec["title_vocab_size"],
        title_tokens=title_tokens,
        title_original_lengths=title_lengths,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Record actual experiment settings and save the first eligible checkpoint
    manifest = {
        "dataset": dataset_name,
        "family": "HSTURec/HSTUBlock",
        "variant": (
            "positional_title_residual"
            if title_spec["intervention_active"]
            else "id_only_missing_titles"
        ),
        "training_seed": 42,
        "preprocessing": "Host supplied seed-0 splits; no split modifications",
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "optimizer_betas": list(optimizer.param_groups[0]["betas"]),
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "maximum_epochs": NUM_EPOCHS,
        "loss": "Original unnormalized pairwise softplus; one sampled negative",
        "sampler": "Original causal HSTU sampler; training-history exclusions",
        "validation_every": 5,
        "checkpoint_rule": "Strictly larger validation HR@10; earliest tie",
        "selection_limit": "Local category checkpoint selection; pinned host macro HR@10 is authoritative",
        "title": title_spec,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "controls": "No additional controls or independent predictors trained in this invocation",
    }
    manifest_path = os.path.join(working_dir, f"{dataset_name}_experiment.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    best_hr10 = -float("inf")
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
            out = model(seq_t)
            mask = (pos_t != 0).float()
            ### >>> Self_EvolveRec-BLOCK-START: Unchanged pairwise objective with tied title-aware candidates
            paired_scores = model.predict_score(
                out, torch.stack((pos_t, neg_t), dim=-1)
            )
            pos_scores, neg_scores = paired_scores.unbind(-1)
            ### <<< Self_EvolveRec-BLOCK-END
            loss = F.softplus(-(pos_scores - neg_scores))
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on invalid training loss or gradients
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"[{dataset_name}] Nonfinite loss at epoch {epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            ### <<< Self_EvolveRec-BLOCK-END
            total_loss += float(loss.item())
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        if epoch % 5 == 0:
            val_m = evaluate_val(
                model, user_train, user_valid, itemnum, device, MAXLEN, EVAL_USERS
            )
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
        elif epoch % 10 == 0:
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist encoder configuration and measured runtime
                "maxlen": MAXLEN,
                "title_vocab_size": title_spec["title_vocab_size"],
                "title_specification": title_spec,
            },
            f,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    manifest.update(
        {
            "best_validation_hr10": best_hr10,
            "best_epoch": best_epoch,
            "executed_epochs": epoch,
            "elapsed_seconds_including_assets_and_validation": time.monotonic()
            - started,
        }
    )
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END
    return best_hr10


# ── Main ──────────────────────────────────────────────────────────────


def write_predict_script(working_dir):
    script = '"""Per-dataset HSTU predict — routes by dataset_name kwarg."""\nfrom __future__ import annotations\n\nimport importlib.util\nimport os\nimport pickle\n\nimport torch\n\n_HERE = os.path.dirname(os.path.abspath(__file__))\n\nDEFAULT_DATASETS = [\n    "Movies_and_TV",\n    "Industrial_and_Scientific",\n    "Electronics",\n    "CDs_and_Vinyl",\n]\nDATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]\n\n_models: dict = {}\n_metas: dict = {}\n_device = None\n\n\ndef _load_model_class():\n    spec = importlib.util.spec_from_file_location(\n        f"hstu_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")\n    )\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n    return module.HSTURec\n\n\ndef _ensure_loaded() -> None:\n    global _device\n    if _models:\n        return\n    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    model_cls = _load_model_class()\n    working_dir = _HERE\n    for dataset_name in DATASETS:\n        meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")\n        ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")\n        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n            continue\n        with open(meta_path, "rb") as f:\n            meta = pickle.load(f)\n        _metas[dataset_name] = meta\n        model = model_cls(\n            itemnum=meta["itemnum"],\n            hidden_units=meta["hidden_units"],\n            num_blocks=meta["num_blocks"],\n            num_heads=meta["num_heads"],\n            dropout_rate=0.0,\n            maxlen=meta["maxlen"],\n        ).to(_device)\n        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))\n        model.eval()\n        _models[dataset_name] = model\n\n\ndef predict(user_id: int, history: list[int], candidates: list[int],\n            *, dataset_name: str | None = None) -> list[float]:\n    _ensure_loaded()\n    if dataset_name and dataset_name in _models:\n        model = _models[dataset_name]\n        meta = _metas[dataset_name]\n    elif _models:\n        first_dataset = next(iter(_models))\n        model = _models[first_dataset]\n        meta = _metas[first_dataset]\n    else:\n        return [0.0] * len(candidates)\n\n    maxlen = meta["maxlen"]\n    seq_arr = [0] * maxlen\n    for idx, item in enumerate(reversed(history[-maxlen:])):\n        seq_arr[maxlen - 1 - idx] = item\n    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)\n    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)\n    with torch.no_grad():\n        out = model(seq_t)\n        user_emb = out[0, -1]\n        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()\n    return scores\n'
    ### >>> Self_EvolveRec-BLOCK-START: Architecture-consistent predictor and selected-checkpoint score parity
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    script = script.replace(
        '            maxlen=meta["maxlen"],\n',
        '            maxlen=meta["maxlen"],\n'
        '            title_vocab_size=meta["title_vocab_size"],\n',
    )
    script = script.replace(
        "            continue\n",
        '            raise FileNotFoundError(f"Missing category checkpoint: {dataset_name}")\n',
    )
    script = script.replace(
        "    elif _models:\n",
        "    elif dataset_name is not None:\n"
        '        raise ValueError(f"Unknown dataset: {dataset_name}")\n'
        "    elif _models:\n",
    )
    script = script.replace(
        "user_emb = out[0, -1]", "user_emb = model.final_features(out, seq_t)[0]"
    )
    script = script.replace(
        "torch.matmul(model.item_emb(cand_t), user_emb)",
        "model.predict_score(user_emb, cand_t)",
    )
    predictor_path = os.path.join(working_dir, "predict.py")
    with open(predictor_path, "w") as stream:
        stream.write(script)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hstu_title_predict_parity", predictor_path
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor._ensure_loaded()
    parity = {}
    for dataset_name in DATASETS:
        meta = predictor._metas[dataset_name]
        state = torch.load(
            os.path.join(working_dir, f"{dataset_name}_model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        # CPU reference avoids retaining an additional category model on the GPU.
        reference = HSTURec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            title_vocab_size=meta["title_vocab_size"],
        )
        reference.load_state_dict(state)
        reference.eval()
        histories = _read_split(os.path.join(DATA_DIR, f"{dataset_name}_train.txt"))
        user = next(iter(histories))
        candidates = set(range(1, min(16, meta["itemnum"]) + 1))
        lengths = reference.title_tokens.ne(0).sum(-1)
        for mask in (reference.title_missing, lengths.eq(1), lengths.ge(2)):
            ids = (mask & torch.arange(meta["itemnum"] + 1).ne(0)).nonzero().flatten()
            if ids.numel():
                candidates.add(int(ids[0]))
        candidates = sorted(candidates)
        short = [candidates[-1]]
        maximum_difference = 0.0
        for history in (histories[user][-meta["maxlen"] :], short, []):
            sequence = torch.tensor([[0] * (meta["maxlen"] - len(history)) + history])
            with torch.no_grad():
                expected = reference.predict_score(
                    reference.final_features(reference.log2feats(sequence), sequence)[
                        0
                    ],
                    torch.tensor(candidates),
                )
            actual = torch.tensor(
                predictor.predict(user, history, candidates, dataset_name=dataset_name)
            )
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            maximum_difference = max(
                maximum_difference, float((actual - expected).abs().max())
            )
        parity[dataset_name] = {
            "maximum_absolute_difference": maximum_difference,
            "precision": "float32",
            "rtol": 1e-4,
            "atol": 1e-5,
        }
        del reference, state
    with open(os.path.join(working_dir, "predictor_parity.json"), "w") as stream:
        json.dump(parity, stream, indent=2)
    print(
        "Selected-checkpoint predictor parity passed for all four categories.",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(
        f"HSTU ID-only: hidden={HIDDEN_UNITS} blocks={NUM_BLOCKS} heads={NUM_HEADS} "
        f"dropout={DROPOUT_RATE} maxlen={MAXLEN}",
        flush=True,
    )
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    all_hr10: list[float] = []
    for dataset_name in DATASETS:
        print(f"\n{'=' * 55}", flush=True)
        print(f"Training {dataset_name}", flush=True)
        print(f"{'=' * 55}", flush=True)
        all_hr10.append(train_one_dataset(dataset_name, device, DATA_DIR))
    write_predict_script(working_dir)
    avg = sum(all_hr10) / max(len(all_hr10), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


