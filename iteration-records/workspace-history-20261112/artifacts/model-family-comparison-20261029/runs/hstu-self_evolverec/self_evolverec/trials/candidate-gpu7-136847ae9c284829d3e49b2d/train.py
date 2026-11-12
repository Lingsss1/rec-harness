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

### >>> Self_EvolveRec-BLOCK-START: Deterministic training-catalog lexical assets
import random
import re
import unicodedata
from collections import Counter, defaultdict

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
### >>> Self_EvolveRec-BLOCK-START: Fix epoch parsing and enforce the matched training budget
NUM_BATCHES = 200
NUM_EPOCHS = max(
    1,
    min(
        200, int(os.environ.get("HSTU_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
    ),
)
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("HSTU_MAXLEN", "128"))
PATIENCE = int(os.environ.get("HSTU_PATIENCE", "20"))
EVAL_USERS = int(os.environ.get("HSTU_EVAL_USERS", "200"))

### >>> Self_EvolveRec-BLOCK-START: Preserve the host data boundary
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
    ### >>> Self_EvolveRec-BLOCK-START: Never inspect hidden test files
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


### >>> Self_EvolveRec-BLOCK-START: Build lexical assets only from training-catalog public text
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def _text_tokens(value):
    if isinstance(value, (list, tuple)):
        value = " ".join(str(part) for part in value if part is not None)
    if not isinstance(value, str):
        return []
    return _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", value).lower())


def build_lexical_assets(data_dir, dataset_name, user_train, itemnum):
    # Metadata must use the same integer item IDs as the supplied splits.
    # An explicit path may contain a {dataset} placeholder. Only title and
    # description are consumed; unavailable fields are never synthesized.
    explicit_path = os.environ.get("GAGC_METADATA_PATH")
    metadata_dir = os.environ.get("GAGC_METADATA_DIR", data_dir)
    # DEBUG: Recognize public metadata in JSON, JSONL and pickle formats.
    # The wrapper traceback does not expose the child's underlying exception;
    # missing metadata must still fail explicitly rather than disable the idea.
    paths = (
        [explicit_path.format(dataset=dataset_name)]
        if explicit_path
        else [
            os.path.join(metadata_dir, f"{dataset_name}_{suffix}.{extension}")
            for extension in ("json", "jsonl", "pkl")
            for suffix in ("metadata", "meta")
        ]
    )
    metadata = {}
    source = next((path for path in paths if os.path.isfile(path)), None)
    if explicit_path and source is None:
        raise FileNotFoundError(paths[0])
    if source is not None:
        # DEBUG: Match the deserializer to the public metadata file format.
        if source.endswith((".pkl", ".pickle")):
            with open(source, "rb") as stream:
                metadata = pickle.load(stream)
        else:
            with open(source, encoding="utf-8") as stream:
                if source.endswith(".jsonl"):
                    metadata = [json.loads(line) for line in stream if line.strip()]
                else:
                    metadata = json.load(stream)
        if isinstance(metadata, dict) and dataset_name in metadata:
            metadata = metadata[dataset_name]
        if isinstance(metadata, list):
            metadata = {
                str(record["item_id"]): record
                for record in metadata
                if isinstance(record, dict) and "item_id" in record
            }
        if not isinstance(metadata, dict):
            raise ValueError("Public metadata must map split item IDs to records")
        # DEBUG: Pickled mappings can have integer keys; normalize them without
        # changing item identities or guessing an ASIN-to-split-ID mapping.
        metadata = {str(key): record for key, record in metadata.items()}
    else:
        # DEBUG: Explain the actual missing input before allocating a model or
        # consuming training budget. Public text remains required for this idea.
        raise FileNotFoundError(
            f"[{dataset_name}] No public title/description metadata found. "
            f"Checked: {paths}. Set GAGC_METADATA_PATH to the supplied public "
            "metadata file keyed by split item IDs. No ID-only fallback is used."
        )

    training_items = sorted(
        {
            int(item)
            for history in user_train.values()
            for item in history
            if 0 < int(item) <= itemnum
        }
    )
    documents = {}
    document_frequency = Counter()
    for item in training_items:
        record = metadata.get(str(item), {})
        if not isinstance(record, dict):
            record = {}
        title = _text_tokens(record.get("title", ""))
        description = _text_tokens(record.get("description", ""))
        documents[item] = (title, description)
        document_frequency.update(set(title) | set(description))
    words = sorted(
        (word for word, count in document_frequency.items() if count >= 2),
        key=lambda word: (-document_frequency[word], word),
    )[:19998]
    vocabulary = {"<pad>": 0, "<unk>": 1}
    vocabulary.update({word: index + 2 for index, word in enumerate(words)})
    title_ids = torch.zeros(itemnum + 1, 32, dtype=torch.long)
    description_ids = torch.zeros_like(title_ids)
    for item, fields in documents.items():
        for destination, tokens in zip((title_ids, description_ids), fields):
            encoded = [vocabulary.get(token, 1) for token in tokens[:32]]
            if encoded:
                destination[item, : len(encoded)] = torch.tensor(encoded)
    available = int(((title_ids != 0).any(1) | (description_ids != 0).any(1)).sum())
    print(
        f"  [{dataset_name}] public text: {available}/{len(training_items)} training items; "
        f"vocabulary={len(vocabulary)}; source={source}",
        flush=True,
    )
    return vocabulary, title_ids, description_ids


### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Finite causal attention for left-padded histories
        batch_size, length = padding_mask.shape
        blocked = causal_mask.bool().unsqueeze(0) | padding_mask.unsqueeze(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=seq_emb.device).unsqueeze(
            0
        )
        # A padded query may attend only to its own position, preventing an
        # entirely masked softmax. Valid queries still cannot attend to padding.
        blocked = blocked & ~(padding_mask.unsqueeze(2) & diagonal)
        blocked = blocked.repeat_interleave(self.attn.num_heads, dim=0)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=blocked,
            need_weights=False,
        )
        attn_out = attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END
        value, gate = self.gate(attn_out).chunk(2, dim=-1)
        seq_emb = seq_emb + self.dropout(value * F.silu(gate))
        ### >>> Self_EvolveRec-BLOCK-START: Remask padded HSTU outputs
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END


class HSTURec(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Configure one jointly trained lexical HSTU
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
        lexical_vocab_size: int = 2,
        title_ids=None,
        description_ids=None,
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

        ### >>> Self_EvolveRec-BLOCK-START: Shared item-side gated lexical residual
        # Initialize after the baseline layers to preserve their seeded initialization.
        self.register_buffer(
            "title_ids",
            (
                torch.zeros(itemnum + 1, 32, dtype=torch.long)
                if title_ids is None
                else title_ids.clone().long()
            ),
        )
        self.register_buffer(
            "description_ids",
            (
                torch.zeros(itemnum + 1, 32, dtype=torch.long)
                if description_ids is None
                else description_ids.clone().long()
            ),
        )
        self.token_emb = nn.Embedding(lexical_vocab_size, 32, padding_idx=0)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[0].zero_()
        self.text_projection = nn.Linear(64, hidden_units, bias=False)
        self.text_gate = nn.Linear(hidden_units, 1)
        nn.init.constant_(self.text_gate.bias, -2.0)
        self.text_alpha = nn.Parameter(torch.zeros(()))
        self._batch_text_keep = None

    def prepare_shared_batch_text_dropout(self, sequence, positive, negative):
        self._batch_text_keep = None
        if self.training:
            distinct = torch.unique(
                torch.cat(
                    [sequence.reshape(-1), positive.reshape(-1), negative.reshape(-1)]
                )
            )
            keep = torch.ones(self.itemnum + 1, device=sequence.device)
            keep[distinct] = (
                torch.rand(distinct.numel(), device=sequence.device) >= 0.1
            ).float() / 0.9
            keep[0] = 0.0
            self._batch_text_keep = keep

    def _pool_text(self, tokens):
        # DEBUG: Preserve masked-mean pooling while avoiding a large
        # distinct-items x field-length x token-dimension activation tensor.
        width = tokens.shape[-1]
        flat = tokens.reshape(-1)
        offsets = torch.arange(
            0, flat.numel(), width, device=tokens.device, dtype=torch.long
        )
        pooled = F.embedding_bag(
            flat,
            self.token_emb.weight,
            offsets,
            mode="sum",
            padding_idx=0,
        )
        counts = tokens.ne(0).sum(dim=-1, keepdim=True).clamp_min(1)
        return pooled / counts

    def item_representation(self, items):
        # Pool each distinct item once per lookup, sharing the whole-text
        # dropout mask across sequence, positive and negative occurrences.
        distinct, inverse = torch.unique(items.reshape(-1), return_inverse=True)
        identity = self.item_emb(distinct)
        title = self.title_ids[distinct]
        description = self.description_ids[distinct]
        text = self.text_projection(
            torch.cat([self._pool_text(title), self._pool_text(description)], dim=-1)
        )
        available = (title.ne(0).any(-1) | description.ne(0).any(-1)) & distinct.ne(0)
        residual = self.text_alpha * torch.sigmoid(self.text_gate(identity)) * text
        residual = residual * available.unsqueeze(-1)
        if self.training and self._batch_text_keep is not None:
            residual = residual * self._batch_text_keep[distinct].unsqueeze(-1)
        effective = (identity + residual) * distinct.ne(0).unsqueeze(-1)
        return effective[inverse].reshape(*items.shape, self.hidden_units)

    def predict_score(self, features, items):
        candidates = self.item_representation(items)
        if items.ndim == 1:
            return torch.matmul(features, candidates.transpose(-1, -2))
        if candidates.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * candidates).sum(dim=-1)

    def log2feats(self, sequence):
        return self.forward(sequence)

    @staticmethod
    def final_features(features, sequence):
        indices = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        final = indices.masked_fill(sequence.eq(0), -1).amax(dim=1)
        selected = features[
            torch.arange(sequence.shape[0], device=sequence.device), final.clamp_min(0)
        ]
        return selected * final.ge(0).unsqueeze(-1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        seq_emb = self.item_representation(seq) * math.sqrt(self.hidden_units)
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
        ### >>> Self_EvolveRec-BLOCK-START: Keep padding features exactly zero
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
    ### >>> Self_EvolveRec-BLOCK-START: Keep validation sampling deterministic without resetting training randomness
    validation_rng = random.Random(42)
    val_users = [user for user in user_valid if user in user_train]
    if len(val_users) > num_users:
        val_users = validation_rng.sample(val_users, num_users)
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
            ### >>> Self_EvolveRec-BLOCK-START: Use 99 unique validation negatives with an isolated RNG
            if len(neg_pool) < 99:
                raise ValueError(
                    f"User {user} has only {len(neg_pool)} eligible validation negatives; expected 99"
                )
            neg_pool = validation_rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Validate with the canonical lexical scorer
            out = model.log2feats(seq_t)
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
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### >>> Self_EvolveRec-BLOCK-START: Require every category instead of reporting missing data as a measured score
    if itemnum <= 0 or not user_train or not user_valid:
        raise ValueError(
            f"[{dataset_name}] missing training data, validation data, or catalog size"
        )
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Require usable public text before training one lexical HSTU per category
    vocabulary, title_ids, description_ids = build_lexical_assets(
        data_dir, dataset_name, user_train, itemnum
    )
    if not bool(title_ids.ne(0).any() | description_ids.ne(0).any()):
        raise ValueError(
            f"[{dataset_name}] no training items have usable public titles/descriptions. "
            "Supply metadata keyed by the integer item IDs used in the splits via "
            "GAGC_METADATA_PATH or GAGC_METADATA_DIR. Refusing to run an inactive lexical experiment."
        )
    model = HSTURec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        lexical_vocab_size=len(vocabulary),
        title_ids=title_ids,
        description_ids=description_ids,
    ).to(device)
    ### <<< Self_EvolveRec-BLOCK-END
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Save the first evaluated model even when validation HR is zero
    best_hr10 = -1.0
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
            ### >>> Self_EvolveRec-BLOCK-START: Share lexical dropout and scoring throughout the batch
            model.prepare_shared_batch_text_dropout(seq_t, pos_t, neg_t)
            out = model.log2feats(seq_t)
            mask = (pos_t != 0).float()
            pos_scores = model.predict_score(out, pos_t)
            neg_scores = model.predict_score(out, neg_t)
            ### <<< Self_EvolveRec-BLOCK-END
            loss = F.softplus(-(pos_scores - neg_scores))
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
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
                ### >>> Self_EvolveRec-BLOCK-START: Save lexical architecture and vocabulary with the single checkpoint
                "maxlen": MAXLEN,
                "lexical_vocab_size": len(vocabulary),
                "lexical_vocabulary": vocabulary,
                "lexical_token_dim": 32,
                "lexical_field_length": 32,
                "lexical_dropout": 0.1,
                # Per-item token buffers and alpha are included in model.state_dict().
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
    ### >>> Self_EvolveRec-BLOCK-START: Emit the identical lexical architecture and canonical inference path
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    old_constructor = '            maxlen=meta["maxlen"],\n        ).to(_device)'
    new_constructor = (
        '            maxlen=meta["maxlen"],\n'
        '            lexical_vocab_size=meta["lexical_vocab_size"],\n'
        "        ).to(_device)"
    )
    old_scoring = (
        "        out = model(seq_t)\n"
        "        user_emb = out[0, -1]\n"
        "        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()"
    )
    new_scoring = (
        "        out = model.log2feats(seq_t)\n"
        "        user_emb = model.final_features(out, seq_t)[0]\n"
        "        scores = model.predict_score(user_emb, cand_t).cpu().tolist()"
    )
    if script.count(old_constructor) != 1 or script.count(old_scoring) != 1:
        raise RuntimeError("Predictor template no longer matches the lexical writer")
    script = script.replace(old_constructor, new_constructor).replace(
        old_scoring, new_scoring
    )
    with open(os.path.join(working_dir, "predict.py"), "w") as stream:
        stream.write(script)
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
    # DEBUG: CalledProcessError only reports the child's nonzero exit status;
    # it does not identify the training exception. The supplied traceback is
    # insufficient to establish a root cause or justify changing the model.
    # Emit the underlying traceback to stdout so stdout-capturing hosts retain
    # it, while preserving failure status and the required lexical experiment.
    import sys
    import traceback

    try:
        main()
    except Exception:
        # DEBUG: Preserve the actual training exception; CalledProcessError
        # alone does not identify a metadata, model, or runtime failure.
        failure = traceback.format_exc()
        sys.stdout.write(failure)
        sys.stdout.flush()
        error_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
        try:
            os.makedirs(error_dir, exist_ok=True)
            with open(
                os.path.join(error_dir, "training_error.txt"),
                "w",
                encoding="utf-8",
            ) as stream:
                stream.write(failure)
        except OSError as artifact_error:
            # DEBUG: An artifact-write failure must not replace the original
            # exception or allow an unsuccessful experiment to continue.
            print(
                f"Could not save training traceback: {artifact_error}",
                file=sys.stderr,
                flush=True,
            )
        raise


