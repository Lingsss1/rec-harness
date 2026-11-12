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
### >>> Self_EvolveRec-BLOCK-START: Preserve the fixed training budget and epoch ceiling
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200, int(os.environ.get("HSTU_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
)
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("HSTU_MAXLEN", "128"))
PATIENCE = int(os.environ.get("HSTU_PATIENCE", "20"))
EVAL_USERS = int(os.environ.get("HSTU_EVAL_USERS", "200"))

### >>> Self_EvolveRec-BLOCK-START: Require the host data directory
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four category routes
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
    ### >>> Self_EvolveRec-BLOCK-START: Infer catalog size without opening test files
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


### >>> Self_EvolveRec-BLOCK-START: Deterministic permitted text assets for an ordered residual
TEXT_PATTERN = r"\d+(?:\.\d+)?|[^\W\d_]+|[^\w\s]"
TITLE_LENGTH = 32
DESCRIPTION_LENGTH = 96


def load_text_assets(dataset_name, data_dir, itemnum):
    """Read only explicitly supplied, mapped title/description metadata.

    GAGC_TEXT_METADATA may name a JSON file, optionally containing {dataset}.
    Accepted JSON is either {dataset: {mapped_id: record}} or {mapped_id: record}.
    No source metadata filename or raw-ASIN-to-ID correspondence is assumed.
    Without this optional input, the run is explicitly an ID-only control.
    """
    import re
    import time
    from collections import Counter

    started = time.perf_counter()
    source = os.environ.get("GAGC_TEXT_METADATA")
    records = {}
    scope = "No explicitly supplied mapped title/description metadata"
    if source:
        source = source.format(dataset=dataset_name)
        if not os.path.isabs(source):
            source = os.path.join(data_dir, source)
        with open(source, encoding="utf-8") as stream:
            records = json.load(stream)
        if not isinstance(records, dict):
            raise ValueError("Text metadata must be a JSON mapping")
        if dataset_name in records:
            records = records[dataset_name]
        if not isinstance(records, dict):
            raise ValueError("Category text metadata must map item IDs to records")
        scope = "Explicit host-supplied mapped catalog titles/descriptions only"
        mapped_records = sum(
            isinstance(records.get(str(item), records.get(item)), dict)
            for item in range(1, itemnum + 1)
        )
        if not mapped_records:
            raise ValueError(
                f"[{dataset_name}] Supplied text metadata has no mapped catalog "
                "item records; verify category nesting and integer item-ID mapping"
            )

    def field(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(part for part in value if isinstance(part, str))
        return ""

    tokenize = re.compile(TEXT_PATTERN, flags=re.UNICODE).findall
    documents = {}
    frequencies = Counter()
    nonempty = [0, 0]
    truncated = [0, 0]
    for item in range(1, itemnum + 1):
        record = records.get(str(item), records.get(item, {}))
        if not isinstance(record, dict):
            continue
        title = tokenize(field(record.get("title")))
        description = tokenize(field(record.get("description")))
        frequencies.update(set(title) | set(description))
        for index, (tokens, limit) in enumerate(
            ((title, TITLE_LENGTH), (description, DESCRIPTION_LENGTH))
        ):
            nonempty[index] += bool(tokens)
            truncated[index] += len(tokens) > limit
        if title or description:
            documents[item] = (title[:TITLE_LENGTH], description[:DESCRIPTION_LENGTH])
    if source and not documents:
        raise ValueError(
            f"[{dataset_name}] Supplied metadata contains no usable titles or "
            "descriptions; refusing to silently run a text experiment without text"
        )
    retained = sorted(
        (token for token, count in frequencies.items() if count >= 2),
        key=lambda token: (-frequencies[token], token),
    )[:50000]
    vocabulary = ["<PAD>", "<UNK>"] + retained
    lookup = {token: index + 2 for index, token in enumerate(retained)}
    title_tokens = torch.zeros((itemnum + 1, TITLE_LENGTH), dtype=torch.long)
    description_tokens = torch.zeros(
        (itemnum + 1, DESCRIPTION_LENGTH), dtype=torch.long
    )
    token_counts = [0, 0]
    unknown_counts = [0, 0]
    for item, fields in documents.items():
        for index, (tokens, destination) in enumerate(
            zip(fields, (title_tokens, description_tokens))
        ):
            encoded = [lookup.get(token, 1) for token in tokens]
            if encoded:
                destination[item, : len(encoded)] = torch.tensor(encoded)
            token_counts[index] += len(encoded)
            unknown_counts[index] += sum(value == 1 for value in encoded)
    report = {
        "metadata_scope": scope,
        "text_enabled": bool(documents),
        "tokenizer": TEXT_PATTERN,
        "case_sensitive": True,
        "minimum_document_frequency": 2,
        "maximum_retained_tokens": 50000,
        "vocabulary_size": len(vocabulary),
        "title_length": TITLE_LENGTH,
        "description_length": DESCRIPTION_LENGTH,
        "items_with_any_text": len(documents),
        "catalog_items": itemnum,
        "nonempty_fields": nonempty,
        "truncated_fields": truncated,
        "encoded_token_counts": token_counts,
        "unknown_token_counts": unknown_counts,
        "unknown_rates": [
            unknown / count if count else None
            for unknown, count in zip(unknown_counts, token_counts)
        ],
        "preprocessing_wall_seconds": time.perf_counter() - started,
    }
    print(f"  [{dataset_name}] text_assets={json.dumps(report)}", flush=True)
    return {
        "title_tokens": title_tokens,
        "description_tokens": description_tokens,
        "vocabulary": vocabulary,
        "report": report,
    }


### <<< Self_EvolveRec-BLOCK-END

# ── Batch sampler ─────────────────────────────────────────────────────


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    ### >>> Self_EvolveRec-BLOCK-START: Exclude users with no eligible training negative
    users = [
        user
        for user, items in user_train.items()
        if len(items) >= 2 and len(set(items)) < itemnum
    ]
    ### <<< Self_EvolveRec-BLOCK-END
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
                ### >>> Self_EvolveRec-BLOCK-START: Bound rejection sampling while retaining uniform negatives
                for _attempt in range(32):
                    neg_item = random.randint(1, itemnum)
                    if neg_item not in interacted:
                        break
                else:
                    eligible = np.setdiff1d(
                        np.arange(1, itemnum + 1), np.asarray(list(interacted))
                    )
                    neg_item = int(random.choice(eligible))
                neg[idx] = neg_item
                ### <<< Self_EvolveRec-BLOCK-END
            nxt = item
            idx -= 1
            if idx < 0:
                break
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    return np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)


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
    ### >>> Self_EvolveRec-BLOCK-START: Configure the jointly trained text encoder at every construction site
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
        text_vocabulary_size: int = 2,
        text_assets=None,
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

        ### >>> Self_EvolveRec-BLOCK-START: Tied ordered text residual with invalidated inference caching and safe causal left-padding
        # These tensors and all text parameters are part of the single checkpoint.
        title_tokens = torch.zeros((itemnum + 1, TITLE_LENGTH), dtype=torch.long)
        description_tokens = torch.zeros(
            (itemnum + 1, DESCRIPTION_LENGTH), dtype=torch.long
        )
        if text_assets is not None:
            title_tokens.copy_(text_assets["title_tokens"])
            description_tokens.copy_(text_assets["description_tokens"])
        self.register_buffer("title_tokens", title_tokens)
        self.register_buffer("description_tokens", description_tokens)
        self.register_buffer(
            "has_text", title_tokens.ne(0).any(-1) | description_tokens.ne(0).any(-1)
        )
        self.token_emb = nn.Embedding(text_vocabulary_size, 32, padding_idx=0)
        self.title_convs = nn.ModuleList(
            [nn.Conv1d(32, 32, width, padding=width // 2) for width in (3, 5)]
        )
        self.description_convs = nn.ModuleList(
            [nn.Conv1d(32, 32, width, padding=width // 2) for width in (3, 5)]
        )
        self.text_projection = nn.Linear(128, hidden_units, bias=False)
        self.text_gate = nn.Parameter(torch.tensor(-3.0))
        with torch.no_grad():
            self.token_emb.weight[0].zero_()

    def _field_text(self, tokens, convolutions):
        mask = tokens.ne(0)
        x = self.token_emb(tokens) * mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        outputs = []
        for convolution in convolutions:
            activation = F.gelu(convolution(x))
            # Boundary-padded windows are eligible exactly at real token centers.
            pooled = activation.masked_fill(~mask[:, None, :], -float("inf")).amax(-1)
            pooled = torch.where(
                mask.any(-1, keepdim=True), pooled, torch.zeros_like(pooled)
            )
            outputs.append(pooled)
        return torch.cat(outputs, dim=-1)

    def _encode_text(self, items):
        title = self._field_text(self.title_tokens[items], self.title_convs)
        description = self._field_text(
            self.description_tokens[items], self.description_convs
        )
        residual = self.text_projection(torch.cat((title, description), dim=-1))
        return residual * self.has_text[items, None]

    def _joint_unique(self, unique):
        vectors = self.item_emb(unique)
        selected = self.has_text[unique].nonzero(as_tuple=False).flatten()
        if selected.numel():
            from torch.utils.checkpoint import checkpoint

            residuals = []
            for start in range(0, selected.numel(), 512):
                subset = unique[selected[start : start + 512]]
                if self.training and torch.is_grad_enabled():
                    encoded = checkpoint(
                        self._encode_text,
                        subset,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    encoded = self._encode_text(subset)
                residuals.append(encoded)
            residual = torch.zeros_like(vectors).index_copy(
                0, selected, torch.cat(residuals)
            )
            vectors = vectors + torch.sigmoid(self.text_gate) * residual
        return vectors.masked_fill(unique.eq(0)[:, None], 0.0)

    def joint_item(self, items):
        flat = items.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if self.training or torch.is_grad_enabled():
            # Never reuse detached representations in training or gradient probes.
            self._item_cache = None
            self._item_cache_signature = None
            vectors = self._joint_unique(unique)
        else:
            # Cache only deterministic outputs of this same fitted network.
            # Parameter/buffer versions invalidate after optimizer updates or
            # checkpoint loading; identity/device/dtype invalidate replacement
            # or movement. These transient tensors never enter state_dict().
            signature = tuple(
                (id(tensor), tensor._version, tensor.device, tensor.dtype)
                for tensor in (*self.parameters(), *self.buffers())
            )
            if (
                getattr(self, "_item_cache", None) is None
                or getattr(self, "_item_cache_signature", None) != signature
            ):
                # Disable inference tensors here so ordinary no_grad callers can
                # safely extend a cache first created under inference_mode().
                with torch.inference_mode(False):
                    self._item_cache = self.item_emb.weight.new_empty(
                        (self.itemnum + 1, self.hidden_units)
                    )
                    self._item_cache_ready = torch.zeros(
                        self.itemnum + 1,
                        dtype=torch.bool,
                        device=self.item_emb.weight.device,
                    )
                self._item_cache_signature = signature
            missing = unique[~self._item_cache_ready[unique]]
            for start in range(0, missing.numel(), 512):
                subset = missing[start : start + 512]
                self._item_cache[subset] = self._joint_unique(subset)
                self._item_cache_ready[subset] = True
            vectors = self._item_cache[unique]
        return vectors[inverse].reshape(*items.shape, self.hidden_units)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        seq_emb = self.joint_item(seq) * math.sqrt(self.hidden_units)
        pos = (
            torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            + 1
        )
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
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
        # Give otherwise fully masked padding queries a self-key, then erase them.
        diagonal = torch.arange(maxlen, device=seq.device)
        blocked[:, diagonal, diagonal] &= ~padding_mask
        heads = self.blocks[0].attn.num_heads if len(self.blocks) else 1
        blocked = blocked.repeat_interleave(heads, dim=0)
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            seq_emb = block(seq_emb, blocked, None)
            seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.last_norm(seq_emb).masked_fill(padding_mask.unsqueeze(-1), 0.0)

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
        return selected.masked_fill(last.lt(0).unsqueeze(-1), 0.0)

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
    ### >>> Self_EvolveRec-BLOCK-START: Local validation RNG without perturbing training sampling
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
            ### >>> Self_EvolveRec-BLOCK-START: Use 99 unique local validation negatives
            if len(neg_pool) > 99:
                neg_pool = rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            out = model(seq_t)
            ### >>> Self_EvolveRec-BLOCK-START: Validate with the trained tied representation and physical readout
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
    ### >>> Self_EvolveRec-BLOCK-START: Seed each category and invoke the ordered text path explicitly
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise ValueError(f"[{dataset_name}] Missing training data or catalog size")
    if not any(user in user_train and targets for user, targets in user_valid.items()):
        raise ValueError(f"[{dataset_name}] Missing usable validation targets")
    if NUM_EPOCHS < 5:
        raise ValueError("At least five epochs are required by the validation schedule")
    text_assets = load_text_assets(dataset_name, data_dir, itemnum)
    model = HSTURec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        text_vocabulary_size=len(text_assets["vocabulary"]),
        text_assets=text_assets,
    ).to(device)
    ### <<< Self_EvolveRec-BLOCK-END
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Record the actual experiment and save the earliest best checkpoint
    manifest = {
        "dataset": dataset_name,
        "family": "HSTURec/HSTUBlock",
        "condition": (
            "ordered_text_residual"
            if text_assets["report"]["text_enabled"]
            else "id_only_missing_text_control"
        ),
        "preprocessing_seed": 0,
        "preprocessing": "Host-supplied splits; no split reconstruction",
        "training_seed": 42,
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "maxlen": MAXLEN,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "maximum_epochs": NUM_EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "Original single-uniform-negative pairwise softplus; tied joint item vectors",
        "selection": "Local validation HR@10 every 5 epochs; strict improvement; earliest tie",
        "early_stopping_epochs_without_improvement": PATIENCE * 5,
        "host_evaluator": "Unchanged and authoritative for four-category macro HR@10",
        "text": text_assets["report"],
        "pooling": "Max over real token centers; symmetric boundary padding",
        "omitted_controls": [
            "mean_pool",
            "token_mean_mlp",
            "permuted_tokens",
            "shuffled_alignment",
        ],
        "deployment": "Exactly one independently trained checkpoint per category",
    }
    manifest_path = os.path.join(working_dir, f"{dataset_name}_experiment.json")
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2)
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
            ### >>> Self_EvolveRec-BLOCK-START: Score only supervised positions with the unchanged tied pairwise objective
            out = model.log2feats(seq_t)
            supervised = seq_t.ne(0) & pos_t.ne(0) & neg_t.ne(0)
            if not bool(supervised.any()):
                raise RuntimeError(
                    f"[{dataset_name}] Batch has no valid supervised positions"
                )
            paired_scores = model.predict_score(
                out[supervised],
                torch.stack((pos_t[supervised], neg_t[supervised]), dim=-1),
            )
            pos_scores, neg_scores = paired_scores.unbind(-1)
            loss = F.softplus(neg_scores - pos_scores).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"[{dataset_name}] Nonfinite training loss at epoch {epoch}"
                )
            ### <<< Self_EvolveRec-BLOCK-END
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist text architecture and tokenizer alongside checkpoint buffers
                "maxlen": MAXLEN,
                "text_vocabulary": text_assets["vocabulary"],
                "text_config": text_assets["report"],
                "text_encoder": {
                    "embedding_dimension": 32,
                    "channels_per_branch": 32,
                    "widths": [3, 5],
                    "separate_field_convolutions": True,
                    "pooling": "max_at_nonpadding_centers",
                    "projection_bias": False,
                    "initial_gate": -3.0,
                },
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10


# ── Main ──────────────────────────────────────────────────────────────


def write_predict_script(working_dir):
    script = '"""Per-dataset HSTU predict — routes by dataset_name kwarg."""\nfrom __future__ import annotations\n\nimport importlib.util\nimport os\nimport pickle\n\nimport torch\n\n_HERE = os.path.dirname(os.path.abspath(__file__))\n\nDEFAULT_DATASETS = [\n    "Movies_and_TV",\n    "Industrial_and_Scientific",\n    "Electronics",\n    "CDs_and_Vinyl",\n]\nDATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]\n\n_models: dict = {}\n_metas: dict = {}\n_device = None\n\n\ndef _load_model_class():\n    spec = importlib.util.spec_from_file_location(\n        f"hstu_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")\n    )\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n    return module.HSTURec\n\n\ndef _ensure_loaded() -> None:\n    global _device\n    if _models:\n        return\n    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    model_cls = _load_model_class()\n    working_dir = _HERE\n    for dataset_name in DATASETS:\n        meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")\n        ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")\n        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n            continue\n        with open(meta_path, "rb") as f:\n            meta = pickle.load(f)\n        _metas[dataset_name] = meta\n        model = model_cls(\n            itemnum=meta["itemnum"],\n            hidden_units=meta["hidden_units"],\n            num_blocks=meta["num_blocks"],\n            num_heads=meta["num_heads"],\n            dropout_rate=0.0,\n            maxlen=meta["maxlen"],\n        ).to(_device)\n        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))\n        model.eval()\n        _models[dataset_name] = model\n\n\ndef predict(user_id: int, history: list[int], candidates: list[int],\n            *, dataset_name: str | None = None) -> list[float]:\n    _ensure_loaded()\n    if dataset_name and dataset_name in _models:\n        model = _models[dataset_name]\n        meta = _metas[dataset_name]\n    elif _models:\n        first_dataset = next(iter(_models))\n        model = _models[first_dataset]\n        meta = _metas[first_dataset]\n    else:\n        return [0.0] * len(candidates)\n\n    maxlen = meta["maxlen"]\n    seq_arr = [0] * maxlen\n    for idx, item in enumerate(reversed(history[-maxlen:])):\n        seq_arr[maxlen - 1 - idx] = item\n    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)\n    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)\n    with torch.no_grad():\n        out = model(seq_t)\n        user_emb = out[0, -1]\n        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()\n    return scores\n'
    ### >>> Self_EvolveRec-BLOCK-START: Reload the identical single-model text architecture and verify predictor parity
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    script = script.replace(
        '            maxlen=meta["maxlen"],\n',
        '            maxlen=meta["maxlen"],\n'
        '            text_vocabulary_size=len(meta["text_vocabulary"]),\n',
    )
    script = script.replace(
        "        user_emb = out[0, -1]\n",
        "        user_emb = model.final_features(out, seq_t)[0]\n",
    ).replace(
        "torch.matmul(model.item_emb(cand_t), user_emb)",
        "model.predict_score(user_emb, cand_t)",
    )
    script = script.replace(
        '            continue\n        with open(meta_path, "rb") as f:',
        '            raise RuntimeError(f"Missing trained category: {dataset_name}")\n'
        '        with open(meta_path, "rb") as f:',
    )
    script = script.replace(
        "    elif _models:\n",
        "    elif dataset_name is not None:\n"
        '        raise ValueError(f"Unknown dataset route: {dataset_name}")\n'
        "    elif _models:\n",
    )
    predictor_path = os.path.join(working_dir, "predict.py")
    with open(predictor_path, "w") as stream:
        stream.write(script)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hstu_text_predict_parity", predictor_path
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor._ensure_loaded()
    for dataset_name in DATASETS:
        meta = predictor._metas[dataset_name]
        device = predictor._device
        reference = HSTURec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            text_vocabulary_size=len(meta["text_vocabulary"]),
        ).to(device)
        reference.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{dataset_name}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        reference.eval()
        histories = _read_split(os.path.join(DATA_DIR, f"{dataset_name}_train.txt"))
        user = next(iter(histories))
        history = histories[user][-meta["maxlen"] :]
        candidates = list(range(1, min(meta["itemnum"], 16) + 1))
        # Include available short-text and missing-text items without inventing metadata.
        short = reference.has_text & (
            (reference.title_tokens.ne(0).sum(-1) <= 2)
            | (reference.description_tokens.ne(0).sum(-1) <= 2)
        )
        for eligible in (short, ~reference.has_text):
            ids = eligible.nonzero(as_tuple=False).flatten()
            ids = ids[ids.ne(0)]
            if ids.numel():
                candidates.append(int(ids[0]))
        candidates = list(dict.fromkeys(candidates))
        examples = [history, history[-1:], []]
        for permitted_history in examples:
            sequence = [0] * (
                meta["maxlen"] - len(permitted_history)
            ) + permitted_history
            sequence_t = torch.tensor([sequence], dtype=torch.long, device=device)
            candidates_t = torch.tensor(candidates, dtype=torch.long, device=device)
            with torch.no_grad():
                expected = reference.predict_score(
                    reference.final_features(
                        reference.log2feats(sequence_t), sequence_t
                    )[0],
                    candidates_t,
                ).cpu()
            actual = torch.tensor(
                predictor.predict(
                    user,
                    permitted_history,
                    candidates,
                    dataset_name=dataset_name,
                )
            )
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        del reference
    print(
        "Checkpoint/predictor parity passed for all four category routes.", flush=True
    )
    # Review checks: traced sampling -> supervised loss -> checkpoint -> predictor.
    # Preserved HSTU, tied input/output vectors, baseline score scale, seed 42,
    # four routes, physical-position readout, and one checkpoint per category.
    # Inference caches derive solely from that checkpoint and are not predictors.
    # Existing diagnosis interfaces remain unchanged; item_emb remains ID-only.
    # No execution or new validation results are claimed by this code review.
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


