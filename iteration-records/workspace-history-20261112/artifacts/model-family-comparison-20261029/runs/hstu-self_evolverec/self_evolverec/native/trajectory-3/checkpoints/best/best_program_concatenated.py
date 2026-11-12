# === main_code.py ===
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
### >>> Self_EvolveRec-BLOCK-START: Correct epoch parsing and enforce the fixed training budget
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200, int(os.environ.get("HSTU_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
)
if NUM_EPOCHS < 1:
    raise ValueError("HSTU_EPOCHS must be positive")
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
    ### >>> Self_EvolveRec-BLOCK-START: Infer catalog size without reading test files
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


### >>> Self_EvolveRec-BLOCK-START: Deterministic permitted-text preprocessing and jointly trained lexical embeddings
import re
import unicodedata
from collections import Counter

_TOKEN_PATTERN = r"[a-z0-9]+(?:[._/+:-][a-z0-9]+)*"
_TOKEN_RE = re.compile(_TOKEN_PATTERN)


def _text_tokens(value, limit):
    if isinstance(value, (list, tuple)):
        value = " ".join(str(part) for part in value if isinstance(part, str))
    if not isinstance(value, str):
        return []
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", value).lower())[:limit]


def build_lexical_state(dataset_name, data_dir, itemnum):
    # Only explicit catalog-metadata files are considered. Keys must already be
    # host item IDs; raw ASINs are never guessed to be remapped interaction IDs.
    paths = [
        os.path.join(data_dir, f"{dataset_name}_metadata.json"),
        os.path.join(data_dir, f"{dataset_name}_meta.json"),
        os.path.join(data_dir, f"{dataset_name}_items.json"),
        os.path.join(data_dir, f"{dataset_name}_metadata.jsonl"),
        os.path.join(data_dir, f"{dataset_name}_meta.jsonl"),
    ]
    source = next((path for path in paths if os.path.isfile(path)), None)
    if source is None:
        print(
            f"  [{dataset_name}] No supported public catalog metadata; using ID embeddings",
            flush=True,
        )
        return None
    with open(source, encoding="utf-8") as stream:
        if source.endswith(".jsonl"):
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if isinstance(records, dict):
        records = records.items()
    elif isinstance(records, list):
        records = (
            (record.get("item_id"), record)
            for record in records
            if isinstance(record, dict)
        )
    else:
        raise ValueError(f"Unsupported catalog metadata structure: {source}")

    item_words = {}
    for key, record in records:
        if not isinstance(record, dict):
            continue
        try:
            item_id = int(key)
        except (TypeError, ValueError):
            continue
        if not 1 <= item_id <= itemnum:
            continue
        words = _text_tokens(record.get("title"), 24) + _text_tokens(
            record.get("description"), 40
        )
        if words:
            item_words[item_id] = words
    if not item_words:
        print(
            f"  [{dataset_name}] No title/description records with host item IDs; using ID embeddings",
            flush=True,
        )
        return None

    counts = Counter(word for words in item_words.values() for word in words)
    ordered = sorted(
        (word for word, count in counts.items() if count >= 2),
        key=lambda word: (-counts[word], word),
    )[:20000]
    vocabulary = {word: index + 2 for index, word in enumerate(ordered)}
    flat_tokens, offsets = [], [0]
    for item_id in range(itemnum + 1):
        flat_tokens.extend(
            vocabulary.get(word, 1) for word in item_words.get(item_id, ())
        )
        offsets.append(len(flat_tokens))
    print(
        f"  [{dataset_name}] Text coverage={len(item_words)}/{itemnum}; vocabulary={len(vocabulary)}",
        flush=True,
    )
    return {
        "tokenizer_version": 1,
        "normalization": "NFKC+lower",
        "pattern": _TOKEN_PATTERN,
        "title_limit": 24,
        "description_limit": 40,
        "padding_token_id": 0,
        "unknown_token_id": 1,
        "vocabulary": vocabulary,
        "flat_tokens": flat_tokens,
        "offsets": offsets,
        "token_dim": 64,
        "gate_init": -3.0,
    }


class JointLexicalEmbedding(nn.Module):
    def __init__(self, original_id, lexical_state):
        super().__init__()
        self.id_embedding = original_id
        self.num_embeddings = original_id.num_embeddings
        self.embedding_dim = original_id.embedding_dim
        self.padding_idx = 0
        self.register_buffer(
            "flat_tokens", torch.tensor(lexical_state["flat_tokens"], dtype=torch.long)
        )
        self.register_buffer(
            "offsets", torch.tensor(lexical_state["offsets"], dtype=torch.long)
        )
        self.tokens = nn.EmbeddingBag(
            len(lexical_state["vocabulary"]) + 2,
            lexical_state["token_dim"],
            mode="mean",
            padding_idx=0,
            include_last_offset=True,
        )
        nn.init.normal_(self.tokens.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tokens.weight[0].zero_()
        self.projection = nn.Linear(
            lexical_state["token_dim"], self.embedding_dim, bias=False
        )
        self.lexical_norm = nn.LayerNorm(
            self.embedding_dim, elementwise_affine=False, eps=1e-5
        )
        self.gate = nn.Parameter(torch.tensor(float(lexical_state["gate_init"])))
        # Xavier has no single item-count-independent standard deviation. Record
        # the actual initialized nonpadding ID RMS before any optimization.
        self.register_buffer(
            "sigma_id", original_id.weight[1:].detach().float().square().mean().sqrt()
        )

    def forward(self, item_ids):
        shape = item_ids.shape
        unique, inverse = torch.unique(item_ids.reshape(-1), return_inverse=True)
        if unique.numel() == 0:
            return self.id_embedding(item_ids)
        starts = self.offsets[unique]
        lengths = self.offsets[unique + 1] - starts
        bag_offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        bag_indices = torch.repeat_interleave(
            torch.arange(unique.numel(), device=item_ids.device),
            lengths,
        )
        positions = (
            starts[bag_indices]
            + torch.arange(bag_indices.numel(), device=item_ids.device)
            - bag_offsets[bag_indices]
        )
        pooled = self.tokens(self.flat_tokens[positions], bag_offsets)
        residual = self.lexical_norm(self.projection(pooled))
        residual = residual * (lengths > 0).unsqueeze(-1)
        effective = (
            self.id_embedding(unique) + self.gate.sigmoid() * self.sigma_id * residual
        )
        effective = effective * (unique != 0).unsqueeze(-1)
        return effective[inverse].reshape(*shape, self.embedding_dim)

    @property
    def weight(self):
        # Diagnostics see the same effective vectors used by history and target
        # lookups. No detached cache or optimizer-spanning graph is retained.
        return torch.cat(
            [
                self(
                    torch.arange(
                        start,
                        min(start + 4096, self.num_embeddings),
                        device=self.id_embedding.weight.device,
                    )
                )
                for start in range(0, self.num_embeddings, 4096)
            ],
            dim=0,
        )


# ── HSTU-style model ──────────────────────────────────────────────────
### <<< Self_EvolveRec-BLOCK-END


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
        ### >>> Self_EvolveRec-BLOCK-START: Make left-padding attention finite including empty histories
        length = seq_emb.size(1)
        blocked = causal_mask.bool().unsqueeze(0) | padding_mask.unsqueeze(1)
        # A padded query may attend only its own diagonal. Real queries still
        # cannot attend padding or future items.
        diagonal = torch.eye(length, dtype=torch.bool, device=seq_emb.device).unsqueeze(
            0
        )
        blocked = blocked & ~(padding_mask.unsqueeze(-1) & diagonal)
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
        ### >>> Self_EvolveRec-BLOCK-START: Keep padded HSTU states exactly zero
        seq_emb = seq_emb + self.ffn(self.ffn_norm(seq_emb))
        return seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END


class HSTURec(nn.Module):
    ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the lexical architecture from checkpoint metadata
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
        lexical_state=None,
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

        ### >>> Self_EvolveRec-BLOCK-START: Share jointly trained lexical embeddings across all scoring paths
        if lexical_state is not None:
            self.item_emb = JointLexicalEmbedding(self.item_emb, lexical_state)

    def log2feats(self, sequence):
        device = self.pos_emb.weight.device
        sequence = torch.as_tensor(sequence, dtype=torch.long, device=device)
        return self(sequence)

    def predict_score(self, features, items):
        items = torch.as_tensor(items, dtype=torch.long, device=features.device)
        embeddings = self.item_emb(items)
        if items.ndim == 1:
            return torch.matmul(features, embeddings.transpose(0, 1))
        if features.ndim == embeddings.ndim - 1:
            return (features.unsqueeze(-2) * embeddings).sum(dim=-1)
        return (features * embeddings).sum(dim=-1)

    def readout(self, features, sequence):
        indices = torch.arange(sequence.size(1), device=sequence.device).expand_as(
            sequence
        )
        last = indices.masked_fill(sequence == 0, -1).amax(dim=1)
        selected = features[
            torch.arange(sequence.size(0), device=sequence.device), last.clamp_min(0)
        ]
        return selected.masked_fill((last < 0).unsqueeze(-1), 0.0)
        ### <<< Self_EvolveRec-BLOCK-END

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        seq_emb = self.item_emb(seq) * math.sqrt(self.hidden_units)
        pos = (
            torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            + 1
        )
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        ### >>> Self_EvolveRec-BLOCK-START: Preserve causal HSTU features with explicit padding semantics
        causal_mask = torch.triu(
            torch.ones((maxlen, maxlen), dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        padding_mask = seq == 0
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            seq_emb = block(seq_emb, causal_mask, padding_mask)
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
    ### >>> Self_EvolveRec-BLOCK-START: Isolate validation RNG from training and use fixed validation candidates
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
            ### >>> Self_EvolveRec-BLOCK-START: Match the 99 unique negative validation-candidate count
            if len(neg_pool) > 99:
                neg_pool = rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            out = model(seq_t)
            ### >>> Self_EvolveRec-BLOCK-START: Validate with the actual model score and final nonpadding readout
            user_emb = model.readout(out, seq_t)[0]
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
    ### >>> Self_EvolveRec-BLOCK-START: Require all four category training and validation datasets
    if itemnum <= 0 or not user_train or not user_valid:
        raise ValueError(
            f"[{dataset_name}] Missing or empty training/validation data in {data_dir}"
        )
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Train one jointly parameterized lexical HSTU per category from seed 42
    lexical_state = build_lexical_state(dataset_name, data_dir, itemnum)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = HSTURec(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
        lexical_state=lexical_state,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ### <<< Self_EvolveRec-BLOCK-END

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always select a checkpoint from the current training run
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
            out = model(seq_t)
            mask = (pos_t != 0).float()
            ### >>> Self_EvolveRec-BLOCK-START: Preserve the pairwise objective using shared effective embeddings
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
    ### >>> Self_EvolveRec-BLOCK-START: Save current weights when no scheduled checkpoint was selected
    if best_epoch == 0:
        torch.save(model.state_dict(), ckpt_path)
    ### <<< Self_EvolveRec-BLOCK-END
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
                ### >>> Self_EvolveRec-BLOCK-START: Serialize exact lexical preprocessing and reconstruction state
                "maxlen": MAXLEN,
                "lexical_state": lexical_state,
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
    ### >>> Self_EvolveRec-BLOCK-START: Preserve lexical checkpoint reconstruction and enforce category routing
    replacements = [
        (
            'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
            "DATASETS = list(DEFAULT_DATASETS)",
        ),
        (
            '            maxlen=meta["maxlen"],\n',
            '            maxlen=meta["maxlen"],\n'
            '            lexical_state=meta.get("lexical_state"),\n',
        ),
        (
            "        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n"
            "            continue\n",
            "        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n"
            '            raise FileNotFoundError(f"Missing HSTU artifacts for {dataset_name}")\n',
        ),
        (
            "    if _models:\n" "        return\n",
            "    if len(_models) == len(DATASETS):\n" "        return\n",
        ),
        (
            "    if dataset_name and dataset_name in _models:\n"
            "        model = _models[dataset_name]\n"
            "        meta = _metas[dataset_name]\n"
            "    elif _models:\n",
            "    if dataset_name is not None:\n"
            "        if dataset_name not in _models:\n"
            '            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")\n'
            "        model = _models[dataset_name]\n"
            "        meta = _metas[dataset_name]\n"
            "    elif _models:\n",
        ),
        (
            "        user_emb = out[0, -1]\n"
            "        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()",
            "        user_emb = model.readout(out, seq_t)[0]\n"
            "        scores = model.predict_score(user_emb, cand_t).cpu().tolist()",
        ),
    ]
    for old, new in replacements:
        if script.count(old) != 1:
            raise RuntimeError(
                f"Predictor template replacement must match exactly once: {old!r}"
            )
        script = script.replace(old, new, 1)
    compile(script, os.path.join(working_dir, "predict.py"), "exec")
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
    main()


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
import torch
import numpy as np
import torch.nn.functional as F
import json
import random
from collections import Counter


def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
    You are a Senior AI Researcher analyzing the mathematical health of a Recommendation System.
    
    Here are the raw mathematical metrics measured from the current model:
    {json.dumps(raw_diagnosis['metrics'], indent=2)}
    
    Here are the definitions of what these metrics mean:
    {json.dumps(raw_diagnosis['metric_definitions'], indent=2)}
    
    Your task: produce a concise diagnosis summary that captures ONLY the core findings and implications.
    Do NOT propose web searches or action plans. Do NOT list "what to look up".

    Strictly follow the JSON format below.

    Output Format (JSON):
    {{
    "status": <CRITICAL | NEEDS_IMPROVEMENT | STABLE>,
    "core_findings": [
        "<1-2 sentences: the most important interpretation of the metrics>",
        "<1-2 sentences: the second most important interpretation (only if truly necessary)>"
    ],
    "key_implications": [
        "<1 sentence: what this implies about model behavior>",
        "<1 sentence: what this implies about training dynamics or representation>",
        "<optional 1 sentence: risk/trade-off if relevant>"
    ],
    "evidence": {{
        "headline_metrics": {{
        "<metric_name>": <value>,
        "<metric_name>": <value>
        }},
        "brief_metric_read": {{
        "<metric_name>": "<very short interpretation tied to its definition>",
        "<metric_name>": "<very short interpretation tied to its definition>"
        }}
    }}
    }}

    Rules:
    - Keep it short: core_findings <= 3 bullets, key_implications <= 3 bullets.
    - Use technical terms only when they are essential (e.g., 'gradient', 'loss landscape', 'inductive bias'), and keep them brief.
    ### >>> Self_EvolveRec-BLOCK-START: Interpret measured training sensitivity conservatively
    - Every statement must be grounded in the provided metric values/definitions.
    - History interventions measure sensitivity of a frozen fitted model on training prefixes, not validation performance, user intent, shortcut learning, or generalization.
    - Null statistics are unavailable, not zero. Do not infer elapsed-time effects: this supplied HSTU consumes item IDs and learned absolute positions only.
    - Confidence intervals condition on the fitted checkpoint and fixed diagnostic protocol. They do not include training uncertainty.
    - Do not infer diversity, conformity, exposure-adjusted category effects, or unavailable review/metadata attributes.
    # Review checks: traced HSTU feature/scoring shapes, validated IDs before
    # lookups, checked final-nonpadding readout and matched candidates, removed
    # unnecessary category loops, and checked undefined interval reporting.
    # Four-category macro statistics remain explicitly unavailable in this
    # single-category invocation. No training, checkpoint, or predictor changes.
    # Static review only; execution and benchmark performance were not measured.
    ### <<< Self_EvolveRec-BLOCK-END
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Preserve the HSTU ModelView contract and unavailable metadata boundaries
    def __init__(self, view, device):
        self.model = view
        self.network = view.model
        self.device = torch.device(device)
        self.meta_data = getattr(view, "meta_data", {})
        self.review_data = getattr(view, "review_data", {})
        self.user_train = getattr(view, "user_train", {})
        self.itemnum = int(self.network.itemnum)
        self.maxlen = int(self.network.maxlen)
        self.datasets = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        names = [
            getattr(owner, key, None)
            for owner in (view, self.network)
            for key in ("dataset_name", "dataset", "category_name")
        ]
        self.category = next(
            (name for name in names if isinstance(name, str) and name in self.datasets),
            None,
        )
        self.category_index = (
            self.datasets.index(self.category) if self.category is not None else 4
        )

    def _get_categories(self, item_id: int):
        # Item-category attributes are unavailable in this matched task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Sample actual effective HSTU embeddings without materializing the catalog
        with torch.inference_mode():
            n = self.itemnum
            if n < 2:
                return None
            rng = np.random.default_rng(seed)
            k = min(int(sample_size), n)
            if k < 2:
                return None
            idx = torch.as_tensor(
                rng.choice(n, size=k, replace=False) + 1,
                dtype=torch.long,
                device=self.device,
            )
            x = self.model.item_emb(idx).detach().float()
            if not torch.isfinite(x).all() or (x.norm(dim=-1) == 0).any():
                return None
            ### <<< Self_EvolveRec-BLOCK-END
            x = F.normalize(x, p=2, dim=-1)

            gram = x @ x.t()
            offdiag_sum = gram.sum() - gram.diag().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()

            collapse_score = (mean_cos + 1.0) * 0.5
            return round(float(collapse_score), 4)

    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        ### >>> Self_EvolveRec-BLOCK-START: Manage the actual module mode at the diagnostic boundary
        # run_full_diagnosis owns evaluation mode and restores it in finally.
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        for step, batch in enumerate(dataloader):
            if step >= num_batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                continue

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Validate loader tensors before HSTU embedding lookups
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError("Diagnostic loader must supply matching B x L tensors")
            if not 1 <= seq_t.size(1) <= self.maxlen:
                raise ValueError(
                    "Diagnostic sequence length exceeds the HSTU position table"
                )
            for name, tensor in (
                ("sequence", seq_t),
                ("positive", pos_t),
                ("negative", neg_t),
            ):
                if ((tensor < 0) | (tensor > self.itemnum)).any():
                    raise ValueError(
                        f"Diagnostic {name} contains IDs outside 0..itemnum"
                    )
            log_feats = self.model.log2feats(seq_t.detach().cpu().numpy())
            # Two aligned calls use the same scoring path as HSTU training.
            pos_logits = self.model.predict_score(log_feats, pos_t)
            neg_logits = self.model.predict_score(log_feats, neg_t)
            if pos_logits.shape != seq_t.shape or neg_logits.shape != seq_t.shape:
                raise ValueError("Aligned HSTU scores must have shape B x L")
            margin = pos_logits - neg_logits
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Restrict pair diagnostics to finite ordinary-item training positions
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            valid = (
                (seq_t != 0)
                & (pos_t > 0)
                & (pos_t <= self.itemnum)
                & (neg_t > 0)
                & (neg_t <= self.itemnum)
                & torch.isfinite(margin)
            )
            ### <<< Self_EvolveRec-BLOCK-END
            if valid.any():
                margins_all.append(margin[valid].detach().float().cpu())

            margin_cpu = margin.detach().float().cpu().numpy()
            pos_cpu = pos_t.detach().cpu().numpy()
            neg_cpu = neg_t.detach().cpu().numpy()
            ### >>> Self_EvolveRec-BLOCK-START: Use the same validity mask for pair summaries
            valid_cpu = valid.detach().cpu().numpy()
            ### <<< Self_EvolveRec-BLOCK-END

            low_mask = valid_cpu & (margin_cpu < float(eps))
            wrong_mask = valid_cpu & (margin_cpu < 0.0)

            ### >>> Self_EvolveRec-BLOCK-START: Vectorize position counts when item categories are unavailable
            low_total += int(np.count_nonzero(low_mask))
            wrong_total += int(np.count_nonzero(wrong_mask))
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Represent undefined sampled-pair statistics explicitly
            margin_metrics = {
                "mean": None,
                "p10": None,
                "neg_beats_pos_rate": None,
                "small_margin_rate": None,
            }
            ### <<< Self_EvolveRec-BLOCK-END
        else:
            m = torch.cat(margins_all, dim=0)
            N = m.numel()

            k = max(1, int(np.ceil(0.10 * N)))
            p10 = float(torch.kthvalue(m, k).values.item())

            margin_metrics = {
                "mean": round(float(m.mean().item()), 4),
                "p10": round(p10, 4),
                "neg_beats_pos_rate": round(float((m < 0.0).float().mean().item()), 4),
                "small_margin_rate": round(
                    float((m < float(eps)).float().mean().item()), 4
                ),
            }

        def _topk(counter: Counter, k: int):
            return [{"category": c, "count": int(n)} for c, n in counter.most_common(k)]

        breakdown = {
            "eps": float(eps),
            "low_margin_total_positions": int(low_total),
            "wrong_rank_total_positions": int(wrong_total),
            "low_margin_top_categories": {
                "pos_item": _topk(low_pos_cat, topk_categories),
                "neg_item": _topk(low_neg_cat, topk_categories),
            },
            "wrong_rank_top_categories": {
                "pos_item": _topk(wrong_pos_cat, topk_categories),
                "neg_item": _topk(wrong_neg_cat, topk_categories),
            },
        }

        return margin_metrics, breakdown

    ### >>> Self_EvolveRec-BLOCK-START: Matched frozen-HSTU history interventions with clustered rank diagnostics
    def _audit_prefixes(self, batches):
        histories = (
            self.user_train.get("History", {})
            if isinstance(self.user_train, dict)
            else {}
        )
        if not isinstance(histories, dict):
            raise ValueError(
                "ModelView.user_train['History'] must map user IDs to histories"
            )
        exclusions = Counter()
        records = []
        coverage = {
            "source": "identified_training_users" if histories else "training_rows",
            "eligible_users": 0,
            "selected_users": 0,
            "unique_users": 0 if histories else None,
            "selected_rows": 0,
            "prefixes_requested": 0,
            "user_cap": 32,
            "targets_per_user_or_row": 2,
        }
        if histories:
            ordered = sorted(histories, key=lambda key: (str(key), type(key).__name__))
            eligible = []
            for index, uid in enumerate(ordered):
                try:
                    h = [int(item) for item in histories[uid]]
                except (TypeError, ValueError):
                    exclusions["malformed_history_user"] += 1
                    continue
                if any(item < 1 or item > self.itemnum for item in h):
                    exclusions["noncatalog_history_user"] += 1
                    continue
                if len(h) < 5 or self.maxlen < 4:
                    exclusions["history_too_short_user"] += 1
                    continue
                eligible.append((index, uid, h))
            coverage["eligible_users"] = len(eligible)
            rng = np.random.default_rng(
                np.random.SeedSequence([42, self.category_index])
            )
            chosen = sorted(
                rng.choice(
                    len(eligible), min(32, len(eligible)), replace=False
                ).tolist()
            )
            coverage["selected_users"] = len(chosen)
            for picked in chosen:
                index, uid, h = eligible[picked]
                rng = np.random.default_rng(
                    np.random.SeedSequence([42, self.category_index, index])
                )
                positions = sorted(
                    rng.choice(
                        np.arange(4, len(h)), min(2, len(h) - 4), replace=False
                    ).tolist()
                )
                for j in positions:
                    records.append((index, j, h[max(0, j - self.maxlen) : j], h[j]))
        else:
            # Rows have no reliable user identity. Never infer one from sequences.
            row_index = 0
            for batch in batches:
                if not isinstance(batch, (tuple, list)) or len(batch) < 4:
                    continue
                seq = torch.as_tensor(batch[1]).detach().cpu().numpy()
                pos = torch.as_tensor(batch[2]).detach().cpu().numpy()
                for sequence, positive in zip(seq, pos):
                    if coverage["selected_rows"] >= 32:
                        break
                    candidates = []
                    for t in range(len(sequence)):
                        h = [int(x) for x in sequence[: t + 1] if int(x) != 0][
                            -self.maxlen :
                        ]
                        target = int(positive[t])
                        if (
                            len(h) >= 4
                            and 1 <= target <= self.itemnum
                            and all(1 <= x <= self.itemnum for x in h)
                        ):
                            candidates.append((t, h, target))
                    if not candidates:
                        exclusions["no_eligible_target_row"] += 1
                    else:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([42, self.category_index, row_index])
                        )
                        selected = sorted(
                            rng.choice(
                                len(candidates), min(2, len(candidates)), replace=False
                            ).tolist()
                        )
                        for selected_index in selected:
                            t, h, target = candidates[selected_index]
                            records.append((row_index, t, h, target))
                        coverage["selected_rows"] += 1
                    row_index += 1
                if coverage["selected_rows"] >= 32:
                    break
            coverage["rows_examined"] = row_index
        coverage["prefixes_requested"] = len(records)
        return records, coverage, exclusions

    def _score_prefix(self, index, position, history, target, counters):
        rng = np.random.default_rng(
            np.random.SeedSequence([42, self.category_index, index, position])
        )
        # This implementation has only padding=0 and ordinary IDs 1..itemnum.
        # Exclude this prefix, never future events or held-out interactions.
        pool = np.setdiff1d(
            np.arange(1, self.itemnum + 1, dtype=np.int64),
            np.asarray(list(set(history) | {target}), dtype=np.int64),
            assume_unique=False,
        )
        if not len(pool):
            return None, "no_eligible_negative"
        negatives = rng.choice(pool, min(999, len(pool)), replace=False)
        candidates = np.concatenate(([target], negatives))
        n = len(history)
        k = min(3, (n - 2) // 2)
        names = [
            "original",
            "recent_removed",
            "oldest_removed",
            "final_removed",
            "recent_preserving_final",
            "shuffle_0",
            "shuffle_1",
            "shuffle_2",
        ]
        variants = [
            history,
            history[:-k],
            history[k:],
            history[:-1],
            history[: -(k + 1)] + history[-1:],
        ]
        variants.extend(
            rng.permutation(history[:-1]).tolist() + history[-1:] for _ in range(3)
        )
        sequences = np.zeros((len(variants), self.maxlen), dtype=np.int64)
        for row, items in enumerate(variants):
            sequences[row, -len(items) :] = items
        seq_t = torch.as_tensor(sequences, dtype=torch.long, device=self.device)
        features = self.model.log2feats(sequences)
        counters["sequence_evaluations"] += len(variants)
        if features.ndim != 3 or features.shape[:2] != seq_t.shape:
            raise ValueError("HSTU log2feats must return [variants, maxlen, hidden]")
        indices = torch.arange(self.maxlen, device=features.device).expand_as(seq_t)
        last = indices.masked_fill(seq_t == 0, -1).amax(dim=1)
        readout = features[torch.arange(len(variants), device=features.device), last]
        scores = []
        for start in range(0, len(candidates), 256):
            items = torch.as_tensor(
                candidates[start : start + 256], dtype=torch.long, device=self.device
            )
            scores.append(
                self.model.predict_score(readout, items).detach().float().cpu().numpy()
            )
            counters["candidate_score_calls"] += 1
        scores = np.concatenate(scores, axis=-1)
        counters["candidate_scores"] += len(variants) * len(candidates)
        if scores.shape != (len(variants), len(candidates)):
            raise ValueError(
                "HSTU diagnostic score contract must return [variants, candidates]"
            )
        if not np.isfinite(scores).all():
            return None, "nonfinite_scores"
        target_scores = scores[:, :1]
        ranks = (
            1
            + (scores > target_scores).sum(axis=1)
            + ((scores == target_scores) & (candidates[None, :] < target)).sum(axis=1)
        )
        values = np.stack([ranks, 1.0 / ranks, (ranks <= 10).astype(float)], axis=1)
        outcomes = {name: value for name, value in zip(names, values)}
        outcomes["shuffle"] = np.mean(values[5:8], axis=0)
        effects = {
            name: outcomes[name] - outcomes["original"]
            for name in (
                "recent_removed",
                "oldest_removed",
                "final_removed",
                "recent_preserving_final",
                "shuffle",
            )
        }
        # Direct equal-count comparison, in addition to effects versus original.
        effects["recent_minus_oldest_removal"] = (
            outcomes["recent_removed"] - outcomes["oldest_removed"]
        )
        effects["final_minus_prefinal_removal"] = (
            outcomes["final_removed"] - outcomes["recent_preserving_final"]
            if k == 1
            else None
        )
        effects = {key: value for key, value in effects.items() if value is not None}
        tied = ((scores == target_scores) & (candidates[None, :] != target)).any(axis=1)
        return {
            "original": outcomes["original"],
            "effects": effects,
            "candidate_count": len(candidates),
            "tie_frequency": float(tied.mean()),
        }, None

    def _history_audit(self, batches):
        records, coverage, exclusions = self._audit_prefixes(batches)
        counters = {
            "sequence_evaluations": 0,
            "candidate_score_calls": 0,
            "candidate_scores": 0,
        }
        grouped = {}
        counts, ties = [], []
        for index, position, history, target in records:
            result, reason = self._score_prefix(
                index, position, history, target, counters
            )
            if reason:
                exclusions[reason] += 1
                continue
            grouped.setdefault(index, []).append(result)
            counts.append(result["candidate_count"])
            ties.append(result["tie_frequency"])
        reliable = coverage["source"] == "identified_training_users"
        coverage["prefixes_scored"] = len(counts)
        coverage["unique_users"] = len(grouped) if reliable else None
        coverage["rows_scored"] = len(grouped) if not reliable else None
        coverage["exclusions"] = dict(exclusions)
        labels = ("rank", "reciprocal_rank", "hit10")

        def as_metrics(values):
            return {label: float(value) for label, value in zip(labels, values)}

        baseline = None
        effects = {}
        if grouped:
            baseline = as_metrics(
                np.mean(
                    [
                        np.mean([record["original"] for record in rows], axis=0)
                        for rows in grouped.values()
                    ],
                    axis=0,
                )
            )
            names = sorted(
                set.intersection(
                    *[
                        set(record["effects"])
                        for rows in grouped.values()
                        for record in rows
                    ]
                )
            )
            rng = np.random.default_rng(42)
            bootstrap_indices = (
                rng.integers(0, len(grouped), size=(2000, len(grouped)))
                if reliable
                else None
            )
            for name in names:
                clusters = np.asarray(
                    [
                        np.mean([record["effects"][name] for record in rows], axis=0)
                        for rows in grouped.values()
                    ]
                )
                interval = None
                if reliable and len(grouped) >= 2:
                    boot = clusters[bootstrap_indices].mean(axis=1)
                    lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
                    interval = {
                        label: [float(lo), float(hi)]
                        for label, lo, hi in zip(labels, lower, upper)
                    }
                effects[name] = {
                    "mean": as_metrics(clusters.mean(axis=0)),
                    "user_cluster_ci95": interval,
                    "ci_null_reason": (
                        "Loader rows have no verified user IDs"
                        if not reliable
                        else (
                            "At least two identified users are required for a user-cluster interval"
                            if len(grouped) < 2
                            else None
                        )
                    ),
                    "cluster_effect_p10": as_metrics(
                        np.quantile(clusters, 0.1, axis=0)
                    ),
                    "cluster_effect_p90": as_metrics(
                        np.quantile(clusters, 0.9, axis=0)
                    ),
                }
        result = {
            "category": self.category,
            "category_null_reason": (
                None
                if self.category
                else "ModelView does not identify a matched category"
            ),
            "coverage": coverage,
            "cost_counts": counters,
            "original": baseline,
            "effects": effects,
            "null_reason": None if grouped else "No complete eligible matched prefixes",
            "candidate_counts": {
                "min": int(min(counts)) if counts else None,
                "max": int(max(counts)) if counts else None,
                "mean": float(np.mean(counts)) if counts else None,
                "fraction_at_most_10": (
                    float(np.mean(np.asarray(counts) <= 10)) if counts else None
                ),
            },
            "target_tie_frequency": (
                float(
                    np.mean(
                        [
                            np.mean([record["tie_frequency"] for record in rows])
                            for rows in grouped.values()
                        ]
                    )
                )
                if grouped
                else None
            ),
            "undefined_effects": {
                "final_minus_prefinal_removal": (
                    "Equal-count comparison requires k=1 for every scored prefix"
                    if grouped and "final_minus_prefinal_removal" not in effects
                    else (
                        "No complete eligible matched prefixes" if not grouped else None
                    )
                ),
            },
            "protocol": {
                "seed": 42,
                "maxlen": self.maxlen,
                "negative_cap": 999,
                "shuffle_repeats": 3,
                "bootstrap_replicates": 2000,
                "input_channels": "item IDs plus learned absolute positions; no elapsed-time channel",
                "position_policy": "compact deletions, restore left padding; shuffle nonfinal items with final item fixed",
                "identity_scope": "category-local user IDs; cross-category identity is unverified",
                "selection_metric": "Host macro validation HR@10 remains unchanged",
            },
        }
        # A probe receives one actual trained category model. Do not manufacture
        # a four-category estimate from one category or pool local user IDs.
        return {
            "per_category": {
                name: result if name == self.category else None
                for name in self.datasets
            },
            "unidentified_category_result": result if self.category is None else None,
            "macro_four_category_mean": None,
            "macro_four_category_ci95": None,
            "macro_null_reason": "This invocation contains one category checkpoint; all four category user-cluster records are required to recompute a macro estimate and bootstrap",
        }

    def run_full_diagnosis(self, train_loader):
        import time
        from itertools import islice

        print(">>> Running frozen HSTU training-prefix diagnosis...")
        modes = [(module, module.training) for module in self.network.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        generators = []
        for owner in (train_loader, getattr(train_loader, "sampler", None)):
            generator = getattr(owner, "generator", None)
            if isinstance(generator, torch.Generator) and all(
                generator is not old for old, _ in generators
            ):
                generators.append((generator, generator.get_state()))
        devices = []
        if self.device.type == "cuda":
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                self.network.eval()
                if devices:
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(torch.cuda.current_stream(self.device))
                start = time.perf_counter()
                batches = list(islice(iter(train_loader), 10))
                collapse_score = self.analyze_embeddings_collapse_cosine(
                    sample_size=2048
                )
                eps = 0.5
                margin_metrics, breakdown = (
                    self.analyze_pos_neg_margin_with_category_breakdown(
                        batches, num_batches=10, eps=eps, topk_categories=20
                    )
                )
                audit = self._history_audit(batches)
                gpu_seconds, peak = None, None
                if devices:
                    end_event.record(torch.cuda.current_stream(self.device))
                    torch.cuda.synchronize(self.device)
                    gpu_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
                    peak = int(torch.cuda.max_memory_allocated(self.device))
                wall_seconds = time.perf_counter() - start
                breakdown["null_reason"] = (
                    "Item-category attributes are unavailable; raw counts would not adjust for exposure"
                )
                return {
                    "metrics": {
                        "embedding_collapse_score": collapse_score,
                        "embedding_null_reason": (
                            None
                            if collapse_score is not None
                            else "Fewer than two items, zero-norm embeddings, or nonfinite embeddings"
                        ),
                        "pos_neg_margin_mean": margin_metrics["mean"],
                        "pos_neg_margin_p10": margin_metrics["p10"],
                        "pos_neg_neg_beats_pos_rate": margin_metrics[
                            "neg_beats_pos_rate"
                        ],
                        "pos_neg_small_margin_rate": margin_metrics[
                            "small_margin_rate"
                        ],
                        "pos_neg_null_reason": (
                            None
                            if margin_metrics["mean"] is not None
                            else "No finite valid sampled training pairs"
                        ),
                        "pos_neg_margin_category_breakdown": breakdown,
                        "history_interventions": audit,
                        "diagnostic_cost": {
                            "wall_seconds": float(wall_seconds),
                            "cuda_event_seconds": gpu_seconds,
                            "peak_allocated_bytes": peak,
                            "gpu_null_reason": None if devices else "CPU execution",
                        },
                    },
                    "metric_definitions": {
                        "embedding_collapse_score": "(Mean off-diagonal cosine + 1)/2 for up to 2048 effective ordinary-item embeddings sampled with local seed 42. Approximately 0.5 means mean cosine zero. This measures angular alignment, not recommendation diversity or representation rank. Null for undefined inputs.",
                        "pos_neg_margin_mean": "Mean positive-minus-loader-negative score over finite, nonpadding, ordinary-item training positions in at most ten batches. Supplied HSTU features are causal: position t sees sequence[:t+1], not later events.",
                        "pos_neg_margin_p10": "Nearest-order-statistic 10th percentile of sampled pair margins.",
                        "pos_neg_neg_beats_pos_rate": "Fraction of valid sampled pairs with negative score strictly greater than positive score.",
                        "pos_neg_small_margin_rate": "Fraction of valid sampled pairs with positive-minus-negative score below eps=0.5; depends on score scale.",
                        "pos_neg_margin_category_breakdown": "Empty category lists because item categories are unavailable. Position counts remain descriptive; no exposure-adjusted conclusion is supported.",
                        "history_interventions": "Frozen-model training-prefix sampled diagnostics, distinct from the host validation benchmark. Up to 32 eligible users/category and two distinct training targets/user, with >=4 context events truncated to model.maxlen. Fallback uses sequence[:t+1] and positive[t], never later events, and reports rows without user confidence intervals. Earlier occurrences of the target remain valid context. One target plus up to 999 unique negatives excludes only prefix-seen items and target, not later events. All variants share candidates. Rank is one plus scores strictly above target plus tied candidates with smaller item ID; target tie frequency counts any equal-scoring other candidate. Greatest absolute nonpadding index supplies readout. Recent/oldest removal delete k=min(3,floor((n-2)/2)) items; final removal deletes one; recent-preserving-final deletes k immediately preceding the final item. Three nonfinal shuffles are averaged within prefix. Rank, reciprocal rank and hit10 effects are perturbed-minus-original: positive rank is worse; positive reciprocal rank/hit10 is better. Recent-minus-oldest compares equal-count deletions; final-minus-prefinal is available only when k=1 for every included prefix. Repeats average within prefixes, prefixes within user (or unidentified row), then users within category. Bootstrap resamples complete category-local users 2000 times with seed 42, retaining all paired outcomes. Intervals condition on fitted weights and fixed candidates. Fallback empirical quantiles describe row effects, not user uncertainty. Candidate counts <=10 make hit10 trivially one. A four-category macro summary is null until all four category user-cluster records are available; do not average category interval endpoints or assume matching local IDs denote the same person. Sensitivity does not identify intent, shortcuts, elapsed-time effects, or generalization; invariance does not establish that history is ignored.",
                        "diagnostic_cost": "Wall time includes loader access and all probes. CUDA-event elapsed time measures the diagnostic stream interval, including host-induced idle gaps, not kernel-active GPU time. Peak allocated memory is the absolute PyTorch allocation high-water mark after resetting it at audit entry. History cost_counts record actual sequence evaluations, candidate-score calls and scalar candidate scores. CPU GPU statistics are null. No claim of inexpensive auditing is made.",
                    },
                }
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in generators:
                generator.set_state(state)
            for module, training in modes:
                module.training = training

    ### <<< Self_EvolveRec-BLOCK-END
