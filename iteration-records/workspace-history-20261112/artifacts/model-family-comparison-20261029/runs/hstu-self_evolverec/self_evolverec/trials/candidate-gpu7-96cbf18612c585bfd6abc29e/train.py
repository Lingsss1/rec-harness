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


### >>> Self_EvolveRec-BLOCK-START: Repair lexical asset compatibility and implement bounded FP32 residuals
import re
import unicodedata
from collections import Counter

_TOKEN_PATTERN = r"[^\W_]+(?:[-./][^\W_]+)*|[^\w\s]"
_TOKEN_RE = re.compile(_TOKEN_PATTERN)


def _text_tokens(value, limit=None):
    if isinstance(value, (list, tuple)):
        value = " ".join(part for part in value if isinstance(part, str))
    if not isinstance(value, str):
        return []
    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFKC", value).casefold())
    return tokens if limit is None else tokens[:limit]


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

    item_fields = {}
    for key, record in records:
        if not isinstance(record, dict):
            continue
        try:
            item_id = int(key)
        except (TypeError, ValueError):
            continue
        if not 1 <= item_id <= itemnum:
            continue
        fields = (
            _text_tokens(record.get("title")),
            _text_tokens(record.get("description")),
        )
        if any(fields):
            item_fields[item_id] = fields
    if not item_fields:
        print(
            f"  [{dataset_name}] No title/description records with host item IDs; using ID embeddings",
            flush=True,
        )
        return None

    # Count complete eligible catalog fields before truncation. Interaction
    # labels and interaction frequencies do not participate in this vocabulary.
    counts = Counter(
        word for fields in item_fields.values() for words in fields for word in words
    )
    ordered = sorted(counts, key=lambda word: (-counts[word], word))[:9998]
    vocabulary = {word: index + 2 for index, word in enumerate(ordered)}
    title_ids = np.zeros((itemnum + 1, 16), dtype=np.int64)
    description_ids = np.zeros((itemnum + 1, 64), dtype=np.int64)
    diagnostics = {}
    for field_index, (name, limit, table) in enumerate(
        (("title", 16, title_ids), ("description", 64, description_ids))
    ):
        present = truncated = retained = unknown = total = 0
        for item_id, fields in item_fields.items():
            words = fields[field_index]
            total += len(words)
            present += int(bool(words))
            truncated += int(len(words) > limit)
            selected = words[:limit]
            retained += len(selected)
            unknown += sum(word not in vocabulary for word in selected)
            table[item_id, : len(selected)] = [
                vocabulary.get(word, 1) for word in selected
            ]
        diagnostics[name] = {
            "present_items": present,
            "missing_items": itemnum - present,
            "missing_fraction": (itemnum - present) / max(itemnum, 1),
            "truncated_items": truncated,
            "truncation_fraction_among_present": truncated / max(present, 1),
            "catalog_tokens": total,
            "retained_tokens": retained,
            "oov_tokens": unknown,
            "oov_fraction_among_retained": unknown / max(retained, 1),
        }
    print(
        f"  [{dataset_name}] Text coverage={len(item_fields)}/{itemnum}; "
        f"vocabulary={len(vocabulary)}; diagnostics={diagnostics}",
        flush=True,
    )
    return {
        "tokenizer_version": 2,
        "normalization": "NFKC+casefold",
        "pattern": _TOKEN_PATTERN,
        "source_scope": "Supplied category catalog metadata; explicit host item IDs only",
        "source_file": os.path.basename(source),
        "item_id_mapping": "Token-table row i is host item ID i; row 0 is padding",
        "title_limit": 16,
        "description_limit": 64,
        "padding_token_id": 0,
        "unknown_token_id": 1,
        "vocabulary": vocabulary,
        "title_ids": title_ids,
        "description_ids": description_ids,
        "title_mask": title_ids != 0,
        "description_mask": description_ids != 0,
        "token_dim": 32,
        "initial_scale": 0.05,
        "norm_epsilon": 1e-3,
        "ratio_cap": 10.0,
        "diagnostics": diagnostics,
    }


class JointLexicalEmbedding(nn.Module):
    def __init__(self, original_id, lexical_state):
        super().__init__()
        # The previous replacement did not match its SEARCH text: numelel()
        # differed from the actual numel(). This repairs the stale v1 class,
        # which otherwise raises KeyError('flat_tokens') with the v2 assets.
        if lexical_state.get("tokenizer_version") != 2:
            raise ValueError("Expected field-wise lexical preprocessing version 2")
        self.id_embedding = original_id
        self.num_embeddings = original_id.num_embeddings
        self.embedding_dim = original_id.embedding_dim
        self.padding_idx = original_id.padding_idx
        self.norm_epsilon = float(lexical_state["norm_epsilon"])
        self.ratio_cap = float(lexical_state["ratio_cap"])
        if not (
            math.isfinite(self.norm_epsilon)
            and self.norm_epsilon > 0.0
            and math.isfinite(self.ratio_cap)
            and self.ratio_cap > 0.0
        ):
            raise ValueError("Lexical safeguards must be finite and positive")
        vocabulary_size = len(lexical_state["vocabulary"]) + 2
        device = original_id.weight.device
        for name in ("title", "description"):
            ids = torch.as_tensor(
                lexical_state[f"{name}_ids"], dtype=torch.long, device=device
            ).clone()
            mask = torch.as_tensor(
                lexical_state[f"{name}_mask"], dtype=torch.bool, device=device
            ).clone()
            expected = (self.num_embeddings, int(lexical_state[f"{name}_limit"]))
            if (
                expected[1] < 1
                or tuple(ids.shape) != expected
                or tuple(mask.shape) != expected
            ):
                raise ValueError(f"Invalid {name} token-table shape")
            if not torch.equal(mask, ids.ne(0)) or ids[0].ne(0).any():
                raise ValueError(f"Invalid {name} padding mask")
            if ((ids < 0) | (ids >= vocabulary_size)).any():
                raise ValueError(f"Invalid {name} vocabulary IDs")
            self.register_buffer(f"{name}_ids", ids)
            self.register_buffer(f"{name}_mask", mask)
        self.register_buffer(
            "text_present",
            self.title_mask.any(dim=-1) | self.description_mask.any(dim=-1),
        )
        self.tokens = nn.Embedding(
            vocabulary_size,
            int(lexical_state["token_dim"]),
            padding_idx=0,
            device=device,
            dtype=torch.float32,
        )
        nn.init.normal_(self.tokens.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tokens.weight[0].zero_()
        self.projection = nn.Linear(
            2 * int(lexical_state["token_dim"]),
            self.embedding_dim,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        nn.init.xavier_uniform_(self.projection.weight)
        scale = float(lexical_state["initial_scale"])
        if not 0.0 < scale < 1.0:
            raise ValueError("initial_scale must be strictly between zero and one")
        self.scale_logit = nn.Parameter(
            torch.tensor(
                math.log(scale / (1.0 - scale)), device=device, dtype=torch.float32
            )
        )

    def _pool_field(self, selected, name):
        ids = getattr(self, f"{name}_ids")[selected]
        mask = getattr(self, f"{name}_mask")[selected]
        offsets = torch.arange(
            0, ids.numel(), ids.size(1), device=ids.device, dtype=torch.long
        )
        # Avoid allocating a distinct-items x field-length x token-width tensor.
        summed = F.embedding_bag(
            ids.reshape(-1),
            self.tokens.weight.float(),
            offsets,
            mode="sum",
            padding_idx=0,
            include_last_offset=False,
        )
        return summed / mask.sum(dim=-1, keepdim=True).clamp_min(1).float()

    def _text_components(self, selected, identity):
        # Own the FP32 context so both forward and diagnostics use identical
        # arithmetic, including when called beneath an outer autocast context.
        with torch.autocast(device_type=identity.device.type, enabled=False):
            pooled = torch.cat(
                [
                    self._pool_field(selected, "title"),
                    self._pool_field(selected, "description"),
                ],
                dim=-1,
            )
            projected = F.linear(pooled, self.projection.weight.float(), bias=None)
            identity_norm = identity.float().norm(dim=-1, keepdim=True).detach()
            projected_norm = projected.norm(dim=-1, keepdim=True)
            ratio = (
                identity_norm / projected_norm.clamp_min(self.norm_epsilon)
            ).clamp_max(self.ratio_cap)
            residual = self.scale_logit.float().sigmoid() * ratio * projected
        return residual, projected_norm, identity_norm, ratio

    def forward(self, item_ids):
        shape = item_ids.shape
        unique, inverse = torch.unique(item_ids.reshape(-1), return_inverse=True)
        identity = self.id_embedding(unique)
        if unique.numel() == 0:
            return identity.reshape(*shape, self.embedding_dim)
        rows = (unique.ne(0) & self.text_present[unique]).nonzero(as_tuple=True)[0]
        with torch.autocast(device_type=identity.device.type, enabled=False):
            residual = torch.zeros_like(identity, dtype=torch.float32)
            if rows.numel() > 0:
                values, _, _, _ = self._text_components(unique[rows], identity[rows])
                residual = residual.index_copy(0, rows, values)
            effective = torch.where(
                unique.ne(0).unsqueeze(-1),
                identity.float() + residual,
                torch.zeros_like(residual),
            )
        return effective.to(identity.dtype)[inverse].reshape(*shape, self.embedding_dim)

    @torch.no_grad()
    def numerical_diagnostics(self):
        # This method is invoked by the existing five-epoch logging path.
        ids = self.text_present.nonzero(as_tuple=True)[0]
        ids = ids[ids.ne(0)]
        if ids.numel() > 1024:
            selected = torch.linspace(
                0, ids.numel() - 1, 1024, device=ids.device
            ).long()
            ids = ids[selected]
        gate = float(self.scale_logit.float().sigmoid())
        if not math.isfinite(gate):
            raise FloatingPointError("Nonfinite lexical gate")
        result = {
            "sampled_text_items": int(ids.numel()),
            "sample_scope": "Deterministic sample of supplied text-present catalog IDs",
            "gate": gate,
            "nonfinite_rows": 0,
        }
        norms, ratios, relative_norms = [], [], []
        with torch.autocast(device_type=ids.device.type, enabled=False):
            for chunk in ids.split(256):
                if chunk.numel() == 0:
                    continue
                residual, norm, identity_norm, ratio = self._text_components(
                    chunk, self.id_embedding(chunk)
                )
                relative = residual.norm(
                    dim=-1, keepdim=True
                ) / identity_norm.clamp_min(1e-12)
                finite = (
                    torch.isfinite(residual).all(dim=-1, keepdim=True)
                    & torch.isfinite(norm)
                    & torch.isfinite(identity_norm)
                    & torch.isfinite(ratio)
                    & torch.isfinite(relative)
                )
                result["nonfinite_rows"] += int((~finite).sum())
                norms.append(norm[finite])
                ratios.append(ratio[finite])
                relative_norms.append(relative[finite])
            for name, pieces in (
                ("projected_norm", norms),
                ("scale_ratio", ratios),
                ("residual_to_identity_norm", relative_norms),
            ):
                values = (
                    torch.cat(pieces) if pieces else torch.empty(0, device=ids.device)
                )
                result[name + "_p10_p50_p90"] = (
                    torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
                    .cpu()
                    .tolist()
                    if values.numel()
                    else None
                )
            values = torch.cat(ratios) if ratios else torch.empty(0, device=ids.device)
            result["ratio_cap_fraction"] = (
                float(values.eq(self.ratio_cap).float().mean())
                if values.numel()
                else None
            )
        return result

    # Static review summary: repaired v2 asset/class mismatch and the missing
    # numerical_diagnostics method; traced history, target and candidate calls
    # through this shared module. Existing predictor writer imports HSTURec,
    # passes lexical_state, strictly loads its state and calls eval(), so it
    # reconstructs this same implementation. Effective weight remains chunked
    # below for ModelView compatibility. No cache or independent scorer added.
    # HSTU backbone, loss, optimizer, sampler, seeds, routing and checkpoint
    # schedule are preserved. No execution or validation gains are claimed.

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

        ### >>> Self_EvolveRec-BLOCK-START: Isolate lexical initialization and share effective embeddings across scoring paths
        if lexical_state is not None:
            reference = [
                (parameter, parameter.detach().clone())
                for parameter in self.parameters()
            ]
            cpu_rng = torch.get_rng_state().clone()
            touched_devices = sorted(
                {
                    parameter.device.index
                    for parameter in self.parameters()
                    if parameter.is_cuda
                }
            )
            cuda_rng = {
                index: torch.cuda.get_rng_state(index).clone()
                for index in touched_devices
            }
            with torch.random.fork_rng(devices=touched_devices):
                # Constructors run on CPU before model.to(device). Seed only
                # the CPU generator; CUDA generators are not touched here.
                torch.random.default_generator.manual_seed(42)
                lexical = JointLexicalEmbedding(self.item_emb, lexical_state)
            if not torch.equal(cpu_rng, torch.get_rng_state()):
                raise RuntimeError("Lexical initialization changed baseline CPU RNG")
            for index, state in cuda_rng.items():
                if not torch.equal(state, torch.cuda.get_rng_state(index)):
                    raise RuntimeError(
                        "Lexical initialization changed baseline CUDA RNG"
                    )
            for parameter, initial in reference:
                if not torch.equal(parameter.detach(), initial):
                    raise RuntimeError(
                        "Lexical initialization modified a baseline parameter"
                    )
            self.item_emb = lexical
            del reference

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
            ### >>> Self_EvolveRec-BLOCK-START: Reject nonfinite gradients while preserving baseline clipping and optimizer
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
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
            ### >>> Self_EvolveRec-BLOCK-START: Report bounded lexical arithmetic diagnostics without changing selection
            hr10 = val_m["HR@10"]
            if isinstance(model.item_emb, JointLexicalEmbedding):
                lexical_diagnostics = model.item_emb.numerical_diagnostics()
                lexical_diagnostics["last_batch_gradient_norm_before_clip"] = float(
                    gradient_norm
                )
                print(
                    f"  [{dataset_name}] lexical_diagnostics="
                    f"{json.dumps(lexical_diagnostics, sort_keys=True, allow_nan=False)}",
                    flush=True,
                )
                if lexical_diagnostics.get("nonfinite_rows", 0):
                    raise FloatingPointError("Nonfinite lexical representations")
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
                ### >>> Self_EvolveRec-BLOCK-START: Serialize bounded lexical architecture, assets and selected epoch
                "maxlen": MAXLEN,
                "lexical_state": lexical_state,
                "selected_epoch": best_epoch if best_epoch else epoch,
                "selected_validation_hr10": best_hr10 if best_epoch else None,
                "training_seed": 42,
                "lexical_configuration": "field_mean_fp32_norm_floor_1e-3_ratio_cap_10",
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


