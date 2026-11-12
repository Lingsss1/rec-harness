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

### >>> Self_EvolveRec-BLOCK-START: Deterministic training-metadata text preprocessing
from collections import defaultdict, Counter
import re
import time
import unicodedata

### <<< Self_EvolveRec-BLOCK-END

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
### >>> Self_EvolveRec-BLOCK-START: Validate the training budget before expensive initialization
NUM_BATCHES = 200
NUM_EPOCHS = min(200, int(os.environ.get("HSTU_EPOCHS", "200")))
if NUM_EPOCHS < 50:
    raise ValueError(
        "HSTU_EPOCHS must be at least 50 for the prespecified validation schedule"
    )
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("HSTU_MAXLEN", "128"))
PATIENCE = int(os.environ.get("HSTU_PATIENCE", "20"))
### >>> Self_EvolveRec-BLOCK-START: Preserve the required validation user limit
EVAL_USERS = 200
### <<< Self_EvolveRec-BLOCK-END

### >>> Self_EvolveRec-BLOCK-START: Required host data boundary
DATA_DIR = os.environ["GAGC_DATA_DIR"]
### <<< Self_EvolveRec-BLOCK-END

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
### >>> Self_EvolveRec-BLOCK-START: Retain all four category models
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
    ### >>> Self_EvolveRec-BLOCK-START: Never inspect held-out interaction files
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


### >>> Self_EvolveRec-BLOCK-START: Mean-text vocabulary fitted only on training items
TEXT_VERSION = "nfkc-lower-mean-title32-description96-v1"
TEXT_TOKEN_RE = re.compile(r"[^\W_]+(?:[./+−-][^\W_]+)*|[^\s\w]", re.UNICODE)


def _text_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_text_value(part) for part in value)
    return ""


def _text_tokens(value):
    return TEXT_TOKEN_RE.findall(
        unicodedata.normalize("NFKC", _text_value(value)).lower()
    )


def build_text_tables(dataset_name, data_dir, user_train, itemnum):
    # Only these public metadata files are considered. No interaction file,
    # review file, external source, or other trial is used for text features.
    explicit = os.environ.get(f"HSTU_METADATA_{dataset_name}")
    paths = (
        [explicit]
        if explicit
        else [
            os.path.join(data_dir, f"{dataset_name}_{suffix}.{extension}")
            for suffix in ("meta", "metadata", "meta_data")
            for extension in ("json", "jsonl", "jsonl.gz", "pkl")
        ]
    )
    path = next((p for p in paths if p and os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"Public title/description metadata missing for {dataset_name}; "
            f"set HSTU_METADATA_{dataset_name} to the supplied metadata file. "
            "Metadata item IDs must use the supplied interaction-ID mapping."
        )

    if path.endswith(".pkl"):
        with open(path, "rb") as stream:
            raw = pickle.load(stream)
    elif path.endswith((".jsonl", ".jsonl.gz")):
        import gzip

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as stream:
            raw = [json.loads(line) for line in stream if line.strip()]
    else:
        with open(path, encoding="utf-8") as stream:
            raw = json.load(stream)
    if isinstance(raw, dict) and dataset_name in raw:
        raw = raw[dataset_name]
    if isinstance(raw, dict) and isinstance(raw.get("items"), (dict, list)):
        raw = raw["items"]
    if isinstance(raw, dict):
        records = raw.items()
    elif isinstance(raw, list):
        records = (
            (record.get("item_id", record.get("id")), record)
            for record in raw
            if isinstance(record, dict)
        )
    else:
        raise ValueError(f"Unsupported public metadata structure: {path}")

    fields = {}
    for key, record in records:
        if not isinstance(record, dict):
            continue
        try:
            item = int(key)
        except (ValueError, TypeError):
            continue
        if 1 <= item <= itemnum:
            # Deliberately ignore all metadata fields other than public text.
            title = _text_tokens(record.get("title", ""))[:32]
            description = _text_tokens(record.get("description", ""))[:96]
            if title or description:
                fields[item] = (title, description)

    training_items = {item for history in user_train.values() for item in history}
    frequencies = Counter()
    for item in sorted(training_items):
        title, description = fields.get(item, ([], []))
        frequencies.update(title + description)
    if not frequencies:
        raise ValueError(
            f"No training-item text matched {dataset_name}; check the supplied metadata ID mapping"
        )
    retained = sorted(
        (token for token, count in frequencies.items() if count >= 2),
        key=lambda token: (-frequencies[token], token),
    )[:49998]
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    vocabulary.update({token: index + 2 for index, token in enumerate(retained)})
    token_ids = torch.zeros((itemnum + 1, 128), dtype=torch.long)
    field_lengths = torch.zeros((itemnum + 1, 2), dtype=torch.int16)
    for item, (title, description) in fields.items():
        token_ids[item, : len(title)] = torch.tensor(
            [vocabulary.get(token, 1) for token in title], dtype=torch.long
        )
        token_ids[item, 32 : 32 + len(description)] = torch.tensor(
            [vocabulary.get(token, 1) for token in description], dtype=torch.long
        )
        field_lengths[item] = torch.tensor(
            [len(title), len(description)], dtype=torch.int16
        )
    text_meta = {
        "text_version": TEXT_VERSION,
        "text_vocab_size": len(vocabulary),
        "text_dim": 64,
        "text_max_tokens": 128,
        "vocabulary": vocabulary,
        "field_boundaries": {"title": [0, 32], "description": [32, 128]},
        "minimum_token_frequency": 2,
        "vocabulary_fit": "training items only",
    }
    coverage = sum(item in fields for item in training_items) / max(
        len(training_items), 1
    )
    print(
        f"  [{dataset_name}] training text coverage={coverage:.4f} "
        f"vocabulary={len(vocabulary)}; rare/OOV tokens map to UNK; "
        "text truncation can remove compatibility details",
        flush=True,
    )
    return token_ids, field_lengths, text_meta


# ── Batch sampler ─────────────────────────────────────────────────────
### <<< Self_EvolveRec-BLOCK-END


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
        ### >>> Self_EvolveRec-BLOCK-START: Finite attention for left-padded and empty histories
        batch_size, length = padding_mask.shape
        blocked = torch.isneginf(causal_mask).unsqueeze(0) | padding_mask.unsqueeze(1)
        # Padded queries may attend to themselves; their outputs are discarded.
        diagonal = torch.eye(length, dtype=torch.bool, device=normed.device).unsqueeze(
            0
        )
        blocked = blocked & ~(diagonal & padding_mask.unsqueeze(-1))
        blocked = blocked.repeat_interleave(self.attn.num_heads, dim=0)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=blocked,
            need_weights=False,
        )
        ### <<< Self_EvolveRec-BLOCK-END
        value, gate = self.gate(attn_out).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        ### >>> Self_EvolveRec-BLOCK-START: Keep padded HSTU states zero
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END


class HSTURec(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Joint text encoder architecture configuration
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
        text_vocab_size: int = 2,
        text_dim: int = 64,
        text_max_tokens: int = 128,
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
        ### >>> Self_EvolveRec-BLOCK-START: Checkpointed live mean-text encoder
        self.token_emb = nn.Embedding(text_vocab_size, text_dim, padding_idx=0)
        self.text_projection = nn.Linear(text_dim, hidden_units, bias=False)
        self.text_gate = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer(
            "text_token_ids",
            torch.zeros((itemnum + 1, text_max_tokens), dtype=torch.long),
        )
        self.register_buffer(
            "text_field_lengths", torch.zeros((itemnum + 1, 2), dtype=torch.int16)
        )
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[0].zero_()
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        ### <<< Self_EvolveRec-BLOCK-END
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

    ### >>> Self_EvolveRec-BLOCK-START: Identical fused representations for histories and candidates
    def text_repr(self, items: torch.Tensor) -> torch.Tensor:
        flat = items.reshape(-1)
        if flat.numel() == 0:
            return self.item_emb.weight.new_zeros((*items.shape, self.hidden_units))
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        tokens = self.text_token_ids[unique]
        width = tokens.shape[1]
        offsets = torch.arange(0, tokens.numel(), width, device=tokens.device)
        # EmbeddingBag avoids materializing B x L x tokens x text_dim.
        pooled = F.embedding_bag(
            tokens.reshape(-1),
            self.token_emb.weight,
            offsets,
            mode="sum",
            padding_idx=0,
        )
        counts = tokens.ne(0).sum(dim=-1, keepdim=True)
        pooled = pooled / counts.clamp_min(1).to(pooled.dtype)
        text = F.normalize(self.text_projection(pooled), p=2, dim=-1, eps=1e-8)
        text = text * counts.gt(0).to(text.dtype)
        return text[inverse].reshape(*items.shape, self.hidden_units)

    def item_repr(self, items: torch.Tensor) -> torch.Tensor:
        fused = self.item_emb(items) + self.text_gate.sigmoid() * self.text_repr(items)
        return fused * items.ne(0).unsqueeze(-1).to(fused.dtype)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.item_repr(items)
        if items.ndim == 1:
            return torch.matmul(features, embedded.transpose(-1, -2))
        while features.ndim < embedded.ndim:
            features = features.unsqueeze(-2)
        return (features * embedded).sum(dim=-1)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.forward(sequence)

    def readout(self, sequence: torch.Tensor, features=None) -> torch.Tensor:
        if features is None:
            features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        last = (
            positions.unsqueeze(0)
            .expand_as(sequence)
            .masked_fill(sequence.eq(0), -1)
            .amax(dim=1)
        )
        selected = features[
            torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)
        ]
        return selected * last.ge(0).unsqueeze(-1).to(selected.dtype)

    def forward(
        self, seq: torch.Tensor, item_vectors: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        if item_vectors is None:
            item_vectors = self.item_repr(seq)
        elif item_vectors.shape != (*seq.shape, self.hidden_units):
            raise ValueError(
                "Precomputed item representations must have shape B x L x hidden_units"
            )
        seq_emb = item_vectors * math.sqrt(self.hidden_units)
        ### <<< Self_EvolveRec-BLOCK-END
        pos = (
            torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            + 1
        )
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        causal_mask = torch.triu(
            torch.full((maxlen, maxlen), float("-inf"), device=seq.device), diagonal=1
        )
        padding_mask = seq == 0
        for block in self.blocks:
            seq_emb = block(seq_emb, causal_mask, padding_mask)
        ### >>> Self_EvolveRec-BLOCK-START: Deterministic empty-history representation
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
    ### >>> Self_EvolveRec-BLOCK-START: Fixed validation sampling without changing training RNG
    rng = random.Random(42)
    val_users = [user for user in user_valid if user in user_train]
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
            ### >>> Self_EvolveRec-BLOCK-START: Ninety-nine unique validation negatives
            if len(neg_pool) > 99:
                neg_pool = rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Validate the trained fused scorer
            user_emb = model.readout(seq_t)[0]
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
    ### >>> Self_EvolveRec-BLOCK-START: Independent seed-42 category models with training-only vocabularies
    started = time.perf_counter()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    if itemnum <= 0 or not user_train:
        raise ValueError(f"Required category has missing data: {dataset_name}")
    token_ids, field_lengths, text_meta = build_text_tables(
        dataset_name, data_dir, user_train, itemnum
    )
    model = HSTURec(
        ### <<< Self_EvolveRec-BLOCK-END
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        ### >>> Self_EvolveRec-BLOCK-START: Activate and initialize the text branch before optimization
        maxlen=MAXLEN,
        text_vocab_size=text_meta["text_vocab_size"],
        text_dim=text_meta["text_dim"],
        text_max_tokens=text_meta["text_max_tokens"],
    ).to(device)
    model.text_token_ids.copy_(token_ids.to(device))
    model.text_field_lengths.copy_(field_lengths.to(device))
    del token_ids, field_lengths
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Prespecified validation opportunities and earliest-tie selection
    best_hr10 = -1.0
    best_epoch = 0
    validation_epochs = {50, 100, 150, 200}
    validation_log = []
    ### <<< Self_EvolveRec-BLOCK-END
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            seq, pos, neg = sample_batch(user_train, itemnum, BATCH_SIZE, MAXLEN)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Encode each unique batch item once with live gradients
            # The text encoder is deterministic: jointly deduplicating histories
            # and targets preserves the fused scoring rule and all gradients.
            # No representation survives this optimizer step.
            batch_items = torch.stack((seq_t, pos_t, neg_t), dim=0)
            batch_vectors = model.item_repr(batch_items)
            out = model(seq_t, item_vectors=batch_vectors[0])
            mask = (pos_t != 0).float()
            pos_scores = (out * batch_vectors[1]).sum(dim=-1)
            neg_scores = (out * batch_vectors[2]).sum(dim=-1)
            ### <<< Self_EvolveRec-BLOCK-END
            loss = F.softplus(-(pos_scores - neg_scores))
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on nonfinite training loss
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Nonfinite HSTU loss for {dataset_name} at epoch {epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(NUM_BATCHES, 1)
        ### >>> Self_EvolveRec-BLOCK-START: Four prespecified validation looks
        if epoch in validation_epochs:
            ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Measured text-branch utilization and single-checkpoint selection
            with torch.no_grad():
                probe_ids = torch.unique(pos_t[pos_t.ne(0)])[:1024]
                raw_norm = model.item_emb(probe_ids).norm(dim=-1).mean().item()
                text_norm = model.text_repr(probe_ids).norm(dim=-1).mean().item()
                gate = model.text_gate.sigmoid().item()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            entry = {
                "epoch": epoch,
                "HR@10": hr10,
                "elapsed_seconds": elapsed,
                "raw_ID_mean_L2_norm": raw_norm,
                "normalized_text_mean_L2_norm": text_norm,
                "text_gate": gate,
            }
            validation_log.append(entry)
            print(
                f"  [{dataset_name}] measured_branch_diagnostics={json.dumps(entry)}",
                flush=True,
            )
            with open(
                os.path.join(working_dir, f"{dataset_name}_validation.json"), "w"
            ) as stream:
                json.dump(
                    {
                        "looks": validation_log,
                        "minimum_worthwhile_macro_gain": 0.005,
                        "definitions": {
                            "elapsed_seconds": "Synchronized elapsed wall seconds since category setup; not a GPU utilization counter",
                            "raw_ID_mean_L2_norm": "Raw ID embedding geometry on up to 1024 unique positives in the last training batch",
                            "normalized_text_mean_L2_norm": "Mean projected normalized text norm on the same items, before scalar gating",
                            "text_gate": "Jointly learned sigmoid scalar multiplying text before fusion",
                        },
                        "selection": "Largest validation HR@10; ties retain the earlier checkpoint",
                        "interpretation": "Exploratory; the independently pinned host evaluation remains authoritative",
                    },
                    stream,
                    indent=2,
                )
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
            ### <<< Self_EvolveRec-BLOCK-END
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )
    ### >>> Self_EvolveRec-BLOCK-START: Require a prespecified checkpoint before serialization
    if best_epoch == 0:
        raise RuntimeError(
            "No prespecified validation epoch completed; require HSTU_EPOCHS >= 50"
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Serialize vocabulary and text architecture alongside checkpoint buffers
                "maxlen": MAXLEN,
                **text_meta,
                "best_epoch": best_epoch,
                "training_seed": 42,
                "preprocessing_seed": 0,
                "token_masks": "text_token_ids.ne(0), reconstructed exactly from checkpoint buffer",
                "text_limitations": "Mean pooling ignores order; frequency filtering, UNK mapping and truncation can erase rare details",
            },
            f,
        )
        ### <<< Self_EvolveRec-BLOCK-END
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10


# ── Main ──────────────────────────────────────────────────────────────


def write_predict_script(working_dir):
    script = '"""Per-dataset HSTU predict — routes by dataset_name kwarg."""\nfrom __future__ import annotations\n\nimport importlib.util\nimport os\nimport pickle\n\nimport torch\n\n_HERE = os.path.dirname(os.path.abspath(__file__))\n\nDEFAULT_DATASETS = [\n    "Movies_and_TV",\n    "Industrial_and_Scientific",\n    "Electronics",\n    "CDs_and_Vinyl",\n]\nDATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]\n\n_models: dict = {}\n_metas: dict = {}\n_device = None\n\n\ndef _load_model_class():\n    spec = importlib.util.spec_from_file_location(\n        f"hstu_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")\n    )\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n    return module.HSTURec\n\n\ndef _ensure_loaded() -> None:\n    global _device\n    if _models:\n        return\n    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    model_cls = _load_model_class()\n    working_dir = _HERE\n    for dataset_name in DATASETS:\n        meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")\n        ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")\n        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n            continue\n        with open(meta_path, "rb") as f:\n            meta = pickle.load(f)\n        _metas[dataset_name] = meta\n        model = model_cls(\n            itemnum=meta["itemnum"],\n            hidden_units=meta["hidden_units"],\n            num_blocks=meta["num_blocks"],\n            num_heads=meta["num_heads"],\n            dropout_rate=0.0,\n            maxlen=meta["maxlen"],\n        ).to(_device)\n        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))\n        model.eval()\n        _models[dataset_name] = model\n\n\ndef predict(user_id: int, history: list[int], candidates: list[int],\n            *, dataset_name: str | None = None) -> list[float]:\n    _ensure_loaded()\n    if dataset_name and dataset_name in _models:\n        model = _models[dataset_name]\n        meta = _metas[dataset_name]\n    elif _models:\n        first_dataset = next(iter(_models))\n        model = _models[first_dataset]\n        meta = _metas[first_dataset]\n    else:\n        return [0.0] * len(candidates)\n\n    maxlen = meta["maxlen"]\n    seq_arr = [0] * maxlen\n    for idx, item in enumerate(reversed(history[-maxlen:])):\n        seq_arr[maxlen - 1 - idx] = item\n    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)\n    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)\n    with torch.no_grad():\n        out = model(seq_t)\n        user_emb = out[0, -1]\n        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()\n    return scores\n'
    ### >>> Self_EvolveRec-BLOCK-START: Emit the identical fused architecture and scorer for host inference
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    script = script.replace(
        '            maxlen=meta["maxlen"],\n',
        '            maxlen=meta["maxlen"],\n'
        '            text_vocab_size=meta["text_vocab_size"],\n'
        '            text_dim=meta["text_dim"],\n'
        '            text_max_tokens=meta["text_max_tokens"],\n',
    )
    script = script.replace(
        "    elif _models:\n",
        "    elif dataset_name is not None:\n"
        '        raise ValueError(f"No trained category model: {dataset_name}")\n'
        "    elif _models:\n",
    )
    script = script.replace(
        "        out = model(seq_t)\n"
        "        user_emb = out[0, -1]\n"
        "        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()\n",
        "        user_emb = model.readout(seq_t)[0]\n"
        "        scores = model.predict_score(user_emb, cand_t).cpu().tolist()\n",
    )
    required = (
        'text_vocab_size=meta["text_vocab_size"]',
        "model.readout(seq_t)",
        "model.predict_score(user_emb, cand_t)",
    )
    if not all(fragment in script for fragment in required):
        raise RuntimeError("Predictor writer did not install the fused architecture")
    with open(os.path.join(working_dir, "predict.py"), "w") as stream:
        stream.write(script)
    ### <<< Self_EvolveRec-BLOCK-END


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(
        ### >>> Self_EvolveRec-BLOCK-START: Identify the primary text-fusion experiment
        f"HSTU gated mean-text: hidden={HIDDEN_UNITS} blocks={NUM_BLOCKS} heads={NUM_HEADS} "
        ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Verify emitted predictor scores using permitted training prefixes
    write_predict_script(working_dir)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hstu_generated_predictor_verification", os.path.join(working_dir, "predict.py")
    )
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor._ensure_loaded()
    for dataset_name in DATASETS:
        with open(
            os.path.join(working_dir, f"{dataset_name}_meta.pkl"), "rb"
        ) as stream:
            meta = pickle.load(stream)
        kwargs = {
            key: meta[key]
            for key in (
                "itemnum",
                "hidden_units",
                "num_blocks",
                "num_heads",
                "maxlen",
                "text_vocab_size",
                "text_dim",
                "text_max_tokens",
            )
        }
        reference = HSTURec(**kwargs, dropout_rate=0.0).to(device)
        reference.load_state_dict(
            torch.load(
                os.path.join(working_dir, f"{dataset_name}_model.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        reference.eval()
        histories = _read_split(os.path.join(DATA_DIR, f"{dataset_name}_train.txt"))
        user = next(user for user in sorted(histories) if histories[user])
        prefix = histories[user][:-1][-meta["maxlen"] :]
        candidates = list(range(1, min(meta["itemnum"], 32) + 1))
        for history in (prefix, []):
            seq = torch.zeros((1, meta["maxlen"]), dtype=torch.long, device=device)
            if history:
                seq[0, -len(history) :] = torch.tensor(history, device=device)
            items = torch.tensor(candidates, dtype=torch.long, device=device)
            with torch.no_grad():
                expected = reference.predict_score(
                    reference.readout(seq)[0], items
                ).cpu()
            actual = torch.tensor(
                predictor.predict(user, history, candidates, dataset_name=dataset_name),
                dtype=expected.dtype,
            )
            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                raise RuntimeError(f"Nonfinite predictor scores: {dataset_name}")
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        del reference
        print(
            f"  [{dataset_name}] emitted predictor reload agreement passed", flush=True
        )
    avg = sum(all_hr10) / max(len(all_hr10), 1)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


