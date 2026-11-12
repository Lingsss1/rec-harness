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
### >>> Self_EvolveRec-BLOCK-START: Matched uniform control and lexical curriculum
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get("HSTU_EPOCHS", "200"))))
NUM_NEGATIVES = 32
NEGATIVE_MODE = os.environ.get("HSTU_NEGATIVE_MODE", "curriculum")
if NEGATIVE_MODE not in {"uniform", "curriculum"}:
    raise ValueError("HSTU_NEGATIVE_MODE must be uniform or curriculum")
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("HSTU_MAXLEN", "128"))
PATIENCE = int(os.environ.get("HSTU_PATIENCE", "20"))
EVAL_USERS = int(os.environ.get("HSTU_EVAL_USERS", "200"))

### >>> Self_EvolveRec-BLOCK-START: Required host dataset boundary
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
    ### >>> Self_EvolveRec-BLOCK-START: Require public catalog statistics without reading test labels
    raise FileNotFoundError(
        f"Public itemnum for {dataset_name} is missing from {stats_path}"
    )
    ### <<< Self_EvolveRec-BLOCK-END


def load_dataset(dataset_name: str, data_dir: str):
    user_train = _read_split(os.path.join(data_dir, f"{dataset_name}_train.txt"))
    user_valid = _read_split(os.path.join(data_dir, f"{dataset_name}_valid.txt"))
    itemnum = _load_itemnum(data_dir, dataset_name)
    return user_train, user_valid, itemnum


# ── Batch sampler ─────────────────────────────────────────────────────


### >>> Self_EvolveRec-BLOCK-START: Expose training user identities without changing diagnostic batches
def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    return_users: bool = False,
):
    ### <<< Self_EvolveRec-BLOCK-END
    users = [user for user, items in user_train.items() if len(items) >= 2]
    if not users:
        raise ValueError("No users with at least two training interactions")
    ### >>> Self_EvolveRec-BLOCK-START: Track sampled users for full-history exclusion
    batch_seq, batch_pos, batch_neg = [], [], []
    batch_users = []
    ### <<< Self_EvolveRec-BLOCK-END
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
            ### >>> Self_EvolveRec-BLOCK-START: Bounded diagnostic sampling and separate training negatives
            if nxt != 0 and not return_users and len(interacted) < itemnum:
                for attempt in range(32):
                    neg_item = random.randint(1, itemnum)
                    if neg_item not in interacted:
                        neg[idx] = neg_item
                        break
                else:
                    eligible = [i for i in range(1, itemnum + 1) if i not in interacted]
                    if eligible:
                        neg[idx] = random.choice(eligible)
            ### <<< Self_EvolveRec-BLOCK-END
            nxt = item
            idx -= 1
            if idx < 0:
                break
        ### >>> Self_EvolveRec-BLOCK-START: Preserve default sampler compatibility
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
        batch_users.append(user)
    result = np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg)
    if return_users:
        return (*result, batch_users)
    return result
    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Uniform-only 32-negative CE control with exact exclusions
class UniformNegativeSampler:
    """Distinct uniform negatives excluding each user's entire training history."""

    def __init__(self, user_train, itemnum, seed=42):
        self.itemnum = itemnum
        self.rng = np.random.default_rng(seed)
        self.histories = {
            user: np.asarray(
                sorted({i for i in history if 1 <= i <= itemnum}), dtype=np.int64
            )
            for user, history in user_train.items()
        }

    def sample(self, users, positives, count=32):
        result = np.zeros((*positives.shape, count), dtype=np.int64)
        for row, user in enumerate(users):
            positions = np.flatnonzero(positives[row] != 0)
            observed = self.histories[user]
            width = min(count, self.itemnum - len(observed))
            if not len(positions) or width <= 0:
                continue
            draws = self.rng.integers(
                1, self.itemnum + 1, size=(len(positions), width), dtype=np.int64
            )
            for attempt in range(8):
                invalid = np.isin(draws, observed)
                order = np.argsort(draws, axis=1, kind="stable")
                ordered = np.take_along_axis(draws, order, axis=1)
                repeated = np.zeros_like(invalid)
                repeated[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
                duplicate = np.zeros_like(invalid)
                np.put_along_axis(duplicate, order, repeated, axis=1)
                invalid |= duplicate
                if not invalid.any():
                    break
                if attempt < 7:
                    draws[invalid] = self.rng.integers(
                        1, self.itemnum + 1, size=int(invalid.sum()), dtype=np.int64
                    )
            if invalid.any():
                # Exact fallback only for unresolved rows. Accepted entries remain
                # unique, and all remaining entries are sampled without replacement.
                eligible = np.setdiff1d(
                    np.arange(1, self.itemnum + 1, dtype=np.int64),
                    observed,
                    assume_unique=True,
                )
                for local_row in np.flatnonzero(invalid.any(axis=1)):
                    keep = draws[local_row, ~invalid[local_row]]
                    pool = np.setdiff1d(eligible, keep, assume_unique=False)
                    draws[local_row, invalid[local_row]] = self.rng.choice(
                        pool, size=int(invalid[local_row].sum()), replace=False
                    )
            result[row, positions, :width] = draws
        return result


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Deterministic lexical curriculum with cached exclusions and separate RNG
class CurriculumNegativeSampler(UniformNegativeSampler):
    """One model's training sampler; lexical pools never participate in inference."""

    def __init__(self, user_train, itemnum, dataset_name, data_dir, artifact_dir):
        super().__init__(user_train, itemnum, seed=42)
        import time

        from collections import OrderedDict

        self.pools = {}
        self.last_sources = None
        self.stats = defaultdict(int)
        self.preprocessing_seconds = 0.0
        # Lexical choices must not advance the uniform sampler's random stream.
        self.lexical_rng = np.random.default_rng(42)
        # Training histories are fixed, so filtered pools can be reused exactly.
        self.filtered_pool_cache = OrderedDict()
        self.filtered_pool_cache_limit = 8192
        if NEGATIVE_MODE == "uniform":
            return
        started = time.perf_counter()
        # An explicit path may contain {dataset}. No raw ASIN-to-integer mapping
        # is guessed: metadata keys must already use the benchmark's item IDs.
        template = os.environ.get("HSTU_TEXT_FILE")
        text_path = (
            template.format(dataset=dataset_name)
            if template
            else os.path.join(data_dir, f"{dataset_name}_meta.json")
        )
        if not os.path.isfile(text_path):
            if template:
                raise FileNotFoundError(
                    f"[{dataset_name}] Explicit HSTU_TEXT_FILE does not exist: {text_path}"
                )
            self.preprocessing_seconds = time.perf_counter() - started
            print(
                f"[{dataset_name}] No aligned public text at {text_path}; "
                "lexical replacements disabled (uniform control behavior).",
                flush=True,
            )
            return

        import re
        import unicodedata
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer

        with open(text_path, encoding="utf-8") as stream:
            metadata = json.load(stream)
        if isinstance(metadata, dict) and dataset_name in metadata:
            metadata = metadata[dataset_name]
        if not isinstance(metadata, dict):
            raise ValueError("Public text must be an item-ID keyed JSON object")

        def text_value(value):
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return " ".join(v for v in value if isinstance(v, str))
            return ""

        raw_text = []
        for item in range(1, itemnum + 1):
            entry = metadata.get(str(item), {})
            if not isinstance(entry, dict):
                entry = {}
            raw_text.append(
                " ".join(
                    text_value(entry.get(key, "")) for key in ("title", "description")
                )
            )
        normalized = [unicodedata.normalize("NFKC", s).lower() for s in raw_text]
        pattern = r"[^\W_]+(?:[-./+][^\W_]+)*"
        vectorizer = TfidfVectorizer(
            tokenizer=re.compile(pattern).findall,
            token_pattern=None,
            lowercase=False,
            min_df=1,
            max_df=1.0,
            stop_words=None,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            max_features=None,
        )
        os.makedirs(artifact_dir, exist_ok=True)
        matrix = None
        if any(re.search(pattern, text) for text in normalized):
            matrix = vectorizer.fit_transform(normalized).tocsr()
            # Bound query-block storage; never materialize the catalog-square matrix.
            transpose = matrix.transpose().tocsc()
            for start in range(0, itemnum, 128):
                similarities = (matrix[start : start + 128] @ transpose).tocsr()
                for local in range(similarities.shape[0]):
                    item = start + local + 1
                    lo, hi = similarities.indptr[local : local + 2]
                    ids = similarities.indices[lo:hi] + 1
                    values = similarities.data[lo:hi]
                    keep = (ids != item) & (values > 0)
                    ids, values = ids[keep], values[keep]
                    if not len(ids):
                        continue
                    # Deterministic top-64 boundary, including all boundary ties
                    # before resolving them by ascending benchmark item ID.
                    if len(ids) > 64:
                        threshold = np.partition(values, len(values) - 64)[-64]
                        keep = values >= threshold
                        ids, values = ids[keep], values[keep]
                    order = np.lexsort((ids, -values))[:64]
                    ids, values = ids[order][8:], values[order][8:]
                    if len(ids):
                        self.pools[item] = (ids.astype(np.int64), values.copy())
                del similarities
        else:
            print(
                f"[{dataset_name}] Empty vocabulary; lexical replacements disabled.",
                flush=True,
            )
        self.preprocessing_seconds = time.perf_counter() - started
        with open(
            os.path.join(artifact_dir, f"{dataset_name}_lexical.pkl"), "wb"
        ) as stream:
            pickle.dump(
                {
                    "vectorizer": vectorizer if matrix is not None else None,
                    "item_rows": np.arange(1, itemnum + 1, dtype=np.int64),
                    "raw_text": raw_text,
                    "regex": pattern,
                    "normalization": "Unicode NFKC, then lowercase",
                    "versions": {
                        "numpy": np.__version__,
                        "sklearn": sklearn.__version__,
                    },
                    "pools": self.pools,
                    "preprocessing_seed": 0,
                    "preprocessing_seconds": self.preprocessing_seconds,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(
            f"[{dataset_name}] lexical preprocessing={self.preprocessing_seconds:.2f}s "
            f"items_with_moderate_neighbors={len(self.pools)}",
            flush=True,
        )

    def sample(self, users, positives, count=32, epoch=1):
        # The base draws already exclude full training histories and are unique.
        result = super().sample(users, positives, count)
        self.last_sources = np.zeros_like(result, dtype=np.bool_)
        quota = (
            min(8, (8 * max(0, epoch - 5)) // 15)
            if NEGATIVE_MODE == "curriculum"
            else 0
        )
        for row, user in enumerate(users):
            positions = np.flatnonzero(positives[row] != 0)
            observed = self.histories[user]
            width = min(count, self.itemnum - len(observed))
            self.stats["positions"] += len(positions)
            self.stats["shortage_positions"] += len(positions) * int(width < count)
            if not quota or width <= 0:
                continue
            wanted = min(quota, width)
            self.stats["requested_lexical"] += wanted * len(positions)
            # Missing text or empty vocabulary needs no per-target filtering.
            if not self.pools:
                self.stats["fallback_positions"] += len(positions)
                continue
            for position in positions:
                target = int(positives[row, position])
                key = (user, target)
                cached = self.filtered_pool_cache.get(key)
                if cached is None:
                    pool = self.pools.get(target)
                    if pool is None:
                        ids = np.empty(0, dtype=np.int64)
                        similarities = np.empty(0, dtype=np.float32)
                        examined = excluded = 0
                    else:
                        ids, similarities = pool
                        eligible = (
                            (ids >= 1)
                            & (ids <= self.itemnum)
                            & (ids != target)
                            & ~np.isin(ids, observed)
                        )
                        examined = len(ids)
                        excluded = int((~eligible).sum())
                        ids, similarities = ids[eligible], similarities[eligible]
                    cached = (ids, similarities, examined, excluded)
                    self.filtered_pool_cache[key] = cached
                    if len(self.filtered_pool_cache) > self.filtered_pool_cache_limit:
                        self.filtered_pool_cache.popitem(last=False)
                else:
                    self.filtered_pool_cache.move_to_end(key)
                ids, similarities, examined, excluded = cached
                # Count every pool use, including cache hits, for comparable rates.
                self.stats["pool_candidates"] += examined
                self.stats["excluded_pool_candidates"] += excluded
                take = min(wanted, len(ids))
                self.stats["fallback_positions"] += int(take < wanted)
                if not take:
                    continue
                chosen_indices = self.lexical_rng.choice(
                    len(ids), size=take, replace=False
                )
                chosen = ids[chosen_indices]
                # Lexical choices use a separate RNG. Filtering the independent
                # uniform sample leaves enough entries: at most `take` overlap.
                # Its retained prefix is uniform over the remaining eligible IDs.
                uniform = result[row, position, :width]
                uniform = uniform[~np.isin(uniform, chosen)]
                need = width - take
                result[row, position, :width] = np.concatenate((chosen, uniform[:need]))
                self.last_sources[row, position, :take] = True
                self.stats["achieved_lexical"] += take
                self.stats["lexical_similarity_sum"] += float(
                    similarities[chosen_indices].sum()
                )
        return result


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
        ### >>> Self_EvolveRec-BLOCK-START: Prevent fully masked left-padding attention rows
        normed = self.norm(seq_emb)
        blocked = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)
        # A padded query can attend to itself; its output is subsequently zeroed.
        # Valid queries still cannot attend to padding or future positions.
        blocked.diagonal(dim1=-2, dim2=-1).fill_(False)
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
        return seq_emb


class HSTURec(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        maxlen: int = 128,
    ):
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

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, maxlen = seq.shape
        seq_emb = self.item_emb(seq) * math.sqrt(self.hidden_units)
        pos = (
            torch.arange(maxlen, device=seq.device).unsqueeze(0).expand(batch_size, -1)
            + 1
        )
        pos = pos * (seq != 0).long()
        seq_emb = self.emb_dropout(seq_emb + self.pos_emb(pos))
        ### >>> Self_EvolveRec-BLOCK-START: Finite padding states and shared compatibility scorer
        causal_mask = torch.triu(
            torch.ones((maxlen, maxlen), dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        padding_mask = seq == 0
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            seq_emb = block(seq_emb, causal_mask, padding_mask)
            seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.last_norm(seq_emb).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.forward(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    def final_features(self, sequence: torch.Tensor) -> torch.Tensor:
        features = self.log2feats(sequence)
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        final = (
            positions.unsqueeze(0).expand_as(sequence).masked_fill(sequence == 0, -1)
        )
        final = final.amax(dim=1)
        result = features[
            torch.arange(sequence.shape[0], device=sequence.device), final.clamp_min(0)
        ]
        return result.masked_fill((final < 0).unsqueeze(-1), 0.0)
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
    ### >>> Self_EvolveRec-BLOCK-START: Deterministic validation without resetting training RNG
    val_rng = random.Random(42)
    val_users = [user for user in user_valid if user in user_train]
    if len(val_users) > num_users:
        val_users = val_rng.sample(val_users, num_users)
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
            ### >>> Self_EvolveRec-BLOCK-START: Match the validation candidate count
            if len(neg_pool) > 99:
                neg_pool = val_rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            ### >>> Self_EvolveRec-BLOCK-START: Shared greatest-nonpadding readout and scorer
            user_emb = model.final_features(seq_t)[0]
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
    ### >>> Self_EvolveRec-BLOCK-START: Require training data for every routed category
    if itemnum <= 0 or not user_train:
        raise ValueError(f"[{dataset_name}] Missing training data or invalid itemnum")
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Independent seed-42 model with training-only lexical curriculum
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)
    import time

    training_started = time.perf_counter()
    artifact_dir = os.path.join(os.path.dirname(__file__), "working")
    negative_sampler = CurriculumNegativeSampler(
        user_train, itemnum, dataset_name, data_dir, artifact_dir
    )
    model = HSTURec(
        itemnum=itemnum,
        ### <<< Self_EvolveRec-BLOCK-END
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Always select a checkpoint from this training run
    best_hr10 = -1.0
    ### <<< Self_EvolveRec-BLOCK-END
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Memory-bounded curriculum CE and measured source difficulty
            seq, pos, _, users = sample_batch(
                user_train, itemnum, BATCH_SIZE, MAXLEN, return_users=True
            )
            negatives = negative_sampler.sample(users, pos, NUM_NEGATIVES, epoch=epoch)
            valid = (pos != 0) & (negatives != 0).any(axis=-1)
            if not valid.any():
                continue
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
            positives = torch.as_tensor(pos[valid], dtype=torch.long, device=device)
            negatives_t = torch.as_tensor(
                negatives[valid], dtype=torch.long, device=device
            )
            sources_t = torch.as_tensor(
                negative_sampler.last_sources[valid], dtype=torch.bool, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            features = model.log2feats(seq_t)[valid_t]
            feature_grads = torch.empty_like(features)
            loss_value = features.new_zeros(())
            # Source diagnostics use existing logits and draw no random samples.
            # Logging therefore cannot advance either training RNG.
            gap_totals = features.new_zeros(4)
            for start in range(0, len(positives), 1024):
                stop = min(start + 1024, len(positives))
                feats = features[start:stop].detach().requires_grad_(True)
                items = torch.cat(
                    (positives[start:stop, None], negatives_t[start:stop]), dim=1
                )
                logits = model.predict_score(feats, items)
                with torch.no_grad():
                    gaps = logits[:, :1] - logits[:, 1:]
                    available = items[:, 1:] != 0
                    lexical = sources_t[start:stop]
                    for source, source_mask in enumerate((~lexical, lexical)):
                        mask = available & source_mask
                        gap_totals[2 * source] += gaps.masked_fill(~mask, 0).sum()
                        gap_totals[2 * source + 1] += mask.sum()
                logits = logits.masked_fill(items == 0, float("-inf"))
                labels = torch.zeros(len(feats), dtype=torch.long, device=device)
                chunk_loss = F.cross_entropy(logits, labels, reduction="sum") / len(
                    positives
                )
                if not torch.isfinite(chunk_loss):
                    raise FloatingPointError(
                        f"Nonfinite CE in {dataset_name}, epoch {epoch}"
                    )
                loss_value += chunk_loss.detach()
                chunk_loss.backward()
                feature_grads[start:stop].copy_(feats.grad)
                del feats, items, logits, labels, chunk_loss
            features.backward(feature_grads)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss_value)
            measured = gap_totals.cpu().tolist()
            for source, name in enumerate(("uniform", "lexical")):
                negative_sampler.stats[f"{name}_gap_sum"] += measured[2 * source]
                negative_sampler.stats[f"{name}_gap_count"] += measured[2 * source + 1]
            ### <<< Self_EvolveRec-BLOCK-END
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
    ### >>> Self_EvolveRec-BLOCK-START: Prevent stale checkpoints when no scheduled validation ran
    if best_epoch == 0:
        # Short configured runs may finish before the first five-epoch check.
        # Save this run explicitly, regardless of any existing checkpoint file.
        val_m = evaluate_val(
            model, user_train, user_valid, itemnum, device, MAXLEN, EVAL_USERS
        )
        best_hr10 = val_m["HR@10"]
        best_epoch = epoch
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
                ### >>> Self_EvolveRec-BLOCK-START: Persist actual single-model curriculum configuration
                "maxlen": MAXLEN,
                "scorer": "raw_dot_product",
                "readout": "greatest_nonpadding_index",
                "training_loss": "sampled_candidate_ce",
                "negative_mode": NEGATIVE_MODE,
                "lexical_enabled": bool(negative_sampler.pools),
                "lexical_schedule": "min(8, floor(8 * max(0, epoch - 5) / 15))",
                "num_negatives": NUM_NEGATIVES,
                "training_seed": 42,
                "best_epoch": best_epoch,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    ### >>> Self_EvolveRec-BLOCK-START: Persist measured sampling diagnostics and experiment cost
    stats = dict(negative_sampler.stats)
    metrics = {
        "elapsed_seconds": time.perf_counter() - training_started,
        "preprocessing_seconds": negative_sampler.preprocessing_seconds,
        "executed_epochs": epoch,
        "executed_batches": epoch * NUM_BATCHES,
        "peak_gpu_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "lexical_quota_achieved": stats.get("achieved_lexical", 0)
        / max(1, stats.get("requested_lexical", 0)),
        "pool_fallback_rate": stats.get("fallback_positions", 0)
        / max(1, stats.get("positions", 0)),
        "known_positive_exclusion_rate": stats.get("excluded_pool_candidates", 0)
        / max(1, stats.get("pool_candidates", 0)),
        "eligible_catalog_shortage_rate": stats.get("shortage_positions", 0)
        / max(1, stats.get("positions", 0)),
        "lexical_similarity": (
            stats.get("lexical_similarity_sum", 0) / stats["achieved_lexical"]
            if stats.get("achieved_lexical", 0)
            else None
        ),
    }
    for source in ("uniform", "lexical"):
        count = stats.get(f"{source}_gap_count", 0)
        metrics[f"{source}_positive_minus_negative_gap"] = (
            stats.get(f"{source}_gap_sum", 0) / count if count else None
        )
    definitions = {
        "elapsed_seconds": "Wall time from category initialization through training, preprocessing, validation and checkpoint writes; not measured GPU utilization time.",
        "preprocessing_seconds": "Wall time building and persisting this category's lexical artifacts.",
        "executed_epochs": "Number of executed training epochs.",
        "executed_batches": "Number of sampled batches, including batches without eligible negatives.",
        "peak_gpu_bytes": "Peak PyTorch allocated GPU memory since this category began.",
        "lexical_quota_achieved": "Selected lexical negatives divided by requested lexical negatives; zero if none requested.",
        "pool_fallback_rate": "Valid target positions with an unmet positive lexical quota divided by all valid target positions, including warm-up.",
        "known_positive_exclusion_rate": "Pool entries rejected for being the target or a known training positive divided by examined pool entries.",
        "eligible_catalog_shortage_rate": "Valid target positions with fewer than 32 eligible catalog items divided by all valid target positions.",
        "lexical_similarity": "Mean TF-IDF cosine similarity of selected lexical negatives to their targets; null if none selected.",
        "uniform_positive_minus_negative_gap": "Mean training-time history-conditioned positive logit minus uniform-negative logit before each optimizer step; null if unobserved.",
        "lexical_positive_minus_negative_gap": "Mean training-time history-conditioned positive logit minus lexical-negative logit before each optimizer step; null if unobserved.",
    }
    with open(
        os.path.join(working_dir, f"{dataset_name}_sampling_diagnosis.json"), "w"
    ) as stream:
        json.dump(
            {"metrics": metrics, "metric_definitions": definitions, "counts": stats},
            stream,
            indent=2,
        )
    print(f"[{dataset_name}] sampling diagnostics: {json.dumps(metrics)}", flush=True)
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10
    ### <<< Self_EvolveRec-BLOCK-END


# ── Main ──────────────────────────────────────────────────────────────


def write_predict_script(working_dir):
    script = '"""Per-dataset HSTU predict — routes by dataset_name kwarg."""\nfrom __future__ import annotations\n\nimport importlib.util\nimport os\nimport pickle\n\nimport torch\n\n_HERE = os.path.dirname(os.path.abspath(__file__))\n\nDEFAULT_DATASETS = [\n    "Movies_and_TV",\n    "Industrial_and_Scientific",\n    "Electronics",\n    "CDs_and_Vinyl",\n]\nDATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]\n\n_models: dict = {}\n_metas: dict = {}\n_device = None\n\n\ndef _load_model_class():\n    spec = importlib.util.spec_from_file_location(\n        f"hstu_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")\n    )\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n    return module.HSTURec\n\n\ndef _ensure_loaded() -> None:\n    global _device\n    if _models:\n        return\n    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    model_cls = _load_model_class()\n    working_dir = _HERE\n    for dataset_name in DATASETS:\n        meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")\n        ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")\n        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):\n            continue\n        with open(meta_path, "rb") as f:\n            meta = pickle.load(f)\n        _metas[dataset_name] = meta\n        model = model_cls(\n            itemnum=meta["itemnum"],\n            hidden_units=meta["hidden_units"],\n            num_blocks=meta["num_blocks"],\n            num_heads=meta["num_heads"],\n            dropout_rate=0.0,\n            maxlen=meta["maxlen"],\n        ).to(_device)\n        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))\n        model.eval()\n        _models[dataset_name] = model\n\n\ndef predict(user_id: int, history: list[int], candidates: list[int],\n            *, dataset_name: str | None = None) -> list[float]:\n    _ensure_loaded()\n    if dataset_name and dataset_name in _models:\n        model = _models[dataset_name]\n        meta = _metas[dataset_name]\n    elif _models:\n        first_dataset = next(iter(_models))\n        model = _models[first_dataset]\n        meta = _metas[first_dataset]\n    else:\n        return [0.0] * len(candidates)\n\n    maxlen = meta["maxlen"]\n    seq_arr = [0] * maxlen\n    for idx, item in enumerate(reversed(history[-maxlen:])):\n        seq_arr[maxlen - 1 - idx] = item\n    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)\n    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)\n    with torch.no_grad():\n        out = model(seq_t)\n        user_emb = out[0, -1]\n        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()\n    return scores\n'
    ### >>> Self_EvolveRec-BLOCK-START: Keep generated prediction aligned with the trained HSTU
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    script = script.replace("dropout_rate=0.0,", 'dropout_rate=meta["dropout_rate"],')
    script = script.replace(
        "    if dataset_name and dataset_name in _models:",
        "    if dataset_name is not None and dataset_name not in _models:\n"
        '        raise ValueError(f"Missing category model: {dataset_name}")\n'
        "    if dataset_name and dataset_name in _models:",
    )
    old_scoring = (
        "        out = model(seq_t)\n"
        "        user_emb = out[0, -1]\n"
        "        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()"
    )
    new_scoring = (
        "        user_emb = model.final_features(seq_t)[0]\n"
        "        scores = model.predict_score(user_emb, cand_t).cpu().tolist()"
    )
    if old_scoring not in script:
        raise RuntimeError("Generated predictor scoring template changed")
    script = script.replace(old_scoring, new_scoring)
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


