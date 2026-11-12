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
### >>> Self_EvolveRec-BLOCK-START: Fixed-budget normalized uniform-negative control
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200, int(os.environ.get("HSTU_EPOCHS", os.environ.get("SASREC_EPOCHS", "200")))
)
NUM_NEGATIVES = 128
TEMPERATURE = 0.05
# Review checks: traced sampler -> loss -> checkpoint -> generated predictor;
# retained seed 42, four routes, 200 batches/epoch and the 200-epoch ceiling.
# Candidate scoring now supports paired diagnostic tensors; loss recomputation
# bounds retained candidate activations without changing the objective.
# This remains the ID-only uniform control: text/curriculum are unimplemented.
# No execution or new validation measurements were performed in this review.
### <<< Self_EvolveRec-BLOCK-END
MAXLEN = int(os.environ.get("HSTU_MAXLEN", "128"))
PATIENCE = int(os.environ.get("HSTU_PATIENCE", "20"))
EVAL_USERS = int(os.environ.get("HSTU_EVAL_USERS", "200"))

### >>> Self_EvolveRec-BLOCK-START: Require the host-supplied data directory
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
    ### >>> Self_EvolveRec-BLOCK-START: Never inspect held-out test interactions
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


### >>> Self_EvolveRec-BLOCK-START: Preserve diagnostic batches while exposing training-only user identities
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
    ### >>> Self_EvolveRec-BLOCK-START: Carry sampled users without changing the default sampler contract
    batch_seq, batch_pos, batch_neg, batch_users = [], [], [], []
    for _ in range(batch_size):
        user = random.choice(users)
        batch_users.append(user)
        history = user_train[user]
        ### <<< Self_EvolveRec-BLOCK-END
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = history[-1]
        idx = maxlen - 1
        interacted = set(history)
        for item in reversed(history[:-1]):
            seq[idx] = item
            pos[idx] = nxt
            ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic sampling and avoid unused training negatives
            if nxt != 0 and not return_users and len(interacted) < itemnum:
                for _attempt in range(32):
                    neg_item = random.randint(1, itemnum)
                    if neg_item not in interacted:
                        break
                else:
                    eligible = np.setdiff1d(
                        np.arange(1, itemnum + 1),
                        np.asarray(list(interacted)),
                        assume_unique=False,
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
    ### >>> Self_EvolveRec-BLOCK-START: Filtered independent uniform-negative control
    result = (np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg))
    return (*result, batch_users) if return_users else result


class UniformNegativeSampler:
    """Training-history exclusions and unique negatives at each supervised position.

    This implements the uniform control only. No text assets, neighbor curriculum,
    independently fitted scores, or additional predictors are introduced.
    """

    def __init__(self, user_train, itemnum, device):
        self.histories = {
            user: np.asarray(
                sorted({int(i) for i in items if 1 <= int(i) <= itemnum}),
                dtype=np.int64,
            )
            for user, items in user_train.items()
        }
        self.itemnum = itemnum
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(42)
        self.fallback_rng = np.random.default_rng(42)

    @torch.no_grad()
    def sample(self, users, positive):
        batch_size, length = positive.shape
        histories = [self.histories[user] for user in users]
        width = max(1, max(map(len, histories)))
        observed_np = np.full((batch_size, width), self.itemnum + 1, dtype=np.int64)
        for row, history in enumerate(histories):
            observed_np[row, : len(history)] = history
        observed = torch.as_tensor(observed_np, device=self.device)
        available = torch.tensor(
            [min(NUM_NEGATIVES, self.itemnum - len(h)) for h in histories],
            device=self.device,
        )
        slots = torch.arange(NUM_NEGATIVES, device=self.device)
        valid = (slots[None, None, :] < available[:, None, None]) & positive.ne(
            0
        ).unsqueeze(-1)
        shape = (batch_size, length, NUM_NEGATIVES)
        candidates = torch.zeros(shape, dtype=torch.long, device=self.device)
        bad = valid.clone()
        for _ in range(32):
            proposals = torch.randint(
                1,
                self.itemnum + 1,
                shape,
                device=self.device,
                generator=self.generator,
            )
            candidates = torch.where(bad, proposals, candidates)
            flat = candidates.reshape(batch_size, -1).contiguous()
            locations = torch.searchsorted(observed, flat).clamp_max(width - 1)
            seen = observed.gather(1, locations).eq(flat).reshape(shape)
            values, order = candidates.sort(dim=-1)
            repeated = torch.zeros_like(valid)
            repeated[..., 1:] = values[..., 1:].eq(values[..., :-1])
            duplicate = torch.zeros_like(valid).scatter(-1, order, repeated)
            bad = valid & (seen | duplicate | candidates.eq(positive.unsqueeze(-1)))
            if not bool(bad.any()):
                break
        if bool(bad.any()):
            # Rare bounded-retry fallback; replace the entire affected position
            # with an exact uniform sample without replacement from its complement.
            catalog = np.arange(1, self.itemnum + 1, dtype=np.int64)
            pools = {}
            for row, column in bad.any(-1).nonzero(as_tuple=False).cpu().tolist():
                if row not in pools:
                    pools[row] = np.setdiff1d(
                        catalog, histories[row], assume_unique=True
                    )
                target = int(positive[row, column].item())
                pool = pools[row]
                if target in pool:
                    pool = pool[pool != target]
                count = min(NUM_NEGATIVES, len(pool))
                selected = self.fallback_rng.choice(pool, size=count, replace=False)
                candidates[row, column].zero_()
                valid[row, column].zero_()
                if count:
                    candidates[row, column, :count] = torch.as_tensor(
                        selected,
                        device=self.device,
                    )
                    valid[row, column, :count] = True
        return candidates, valid

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
        causal_mask = torch.triu(
            torch.full((maxlen, maxlen), float("-inf"), device=seq.device), diagonal=1
        )
        ### >>> Self_EvolveRec-BLOCK-START: Safe causal attention for left padding and consistent cosine scoring
        padding_mask = seq == 0
        # Padding queries otherwise have no legal attention key. Give each one
        # its own zero-valued key, then erase its output after every block.
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
        blocked[:, diagonal, diagonal] &= ~padding_mask
        heads = self.blocks[0].attn.num_heads if len(self.blocks) else 1
        attention_mask = blocked.repeat_interleave(heads, dim=0)
        seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            seq_emb = block(seq_emb, attention_mask, None)
            seq_emb = seq_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.last_norm(seq_emb).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        vectors = F.normalize(self.item_emb(items), dim=-1)
        if items.ndim == 1:
            # A one-dimensional item list is a shared candidate catalog.
            return torch.matmul(features, vectors.transpose(-1, -2)) / TEMPERATURE
        if items.shape == features.shape[:-1]:
            # Diagnosis supplies one positive/negative item per sequence position.
            return (features * vectors).sum(-1) / TEMPERATURE
        # Otherwise the final item axis contains candidates for each feature.
        if items.shape[:-1] != features.shape[:-1]:
            raise ValueError(
                f"Incompatible feature/item shapes: {tuple(features.shape)}, "
                f"{tuple(items.shape)}"
            )
        return (features.unsqueeze(-2) * vectors).sum(-1) / TEMPERATURE

    @staticmethod
    def final_features(features: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        last = (
            positions.unsqueeze(0)
            .expand_as(sequence)
            .masked_fill(sequence.eq(0), -1)
            .amax(-1)
        )
        selected = features[
            torch.arange(sequence.shape[0], device=sequence.device), last.clamp_min(0)
        ]
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
    ### >>> Self_EvolveRec-BLOCK-START: Stable validation candidates without perturbing training randomness
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
            ### >>> Self_EvolveRec-BLOCK-START: Use 99 unique validation negatives
            if len(neg_pool) > 99:
                neg_pool = rng.sample(neg_pool, 99)
            ### <<< Self_EvolveRec-BLOCK-END
            candidates = [target] + neg_pool
            seq_t = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            out = model(seq_t)
            ### >>> Self_EvolveRec-BLOCK-START: Validate the trained cosine scorer at the final nonpadding position
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


### >>> Self_EvolveRec-BLOCK-START: Independently initialize each category with training seed 42
def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### <<< Self_EvolveRec-BLOCK-END
    ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required category cannot train
    if itemnum <= 0 or not user_train:
        raise ValueError(
            f"[{dataset_name}] Missing training data or invalid catalog size"
        )
    if not user_valid or not any(
        user in user_train and items for user, items in user_valid.items()
    ):
        raise ValueError(f"[{dataset_name}] No usable validation targets")
    if NUM_EPOCHS < 5:
        raise ValueError(
            "At least five epochs are required for the validation schedule"
        )
    ### <<< Self_EvolveRec-BLOCK-END
    model = HSTURec(
        itemnum=itemnum,
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

    ### >>> Self_EvolveRec-BLOCK-START: Prespecified uniform control and earliest-best checkpoint selection
    negative_sampler = UniformNegativeSampler(user_train, itemnum, device)
    best_hr10 = -float("inf")
    best_epoch = 0
    print(
        f"  [{dataset_name}] variant=normalized_uniform128_id_only "
        f"temperature={TEMPERATURE} validation_every=5 earliest_tie=True "
        f"epochs<={NUM_EPOCHS} batches_per_epoch={NUM_BATCHES}; "
        "text/curriculum ablations not run; host validation is authoritative",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Positive-once normalized cross-entropy with 128 filtered unique negatives
            seq, pos, _, users = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                return_users=True,
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            neg_t, neg_valid = negative_sampler.sample(users, pos_t)
            supervised = pos_t.ne(0)
            features = F.normalize(model.log2feats(seq_t)[supervised], dim=-1)
            positive = pos_t[supervised]
            negatives = neg_t[supervised]
            negative_mask = neg_valid[supervised]
            # Recompute candidate embeddings during backward instead of retaining
            # P * 128 * hidden_units activations across all chunks. Every position
            # still has its complete positive-once cross-entropy denominator.
            from torch.utils.checkpoint import checkpoint

            def chunk_loss(query, targets, sampled, sampled_valid):
                positive_vectors = F.normalize(model.item_emb(targets), dim=-1)
                negative_vectors = F.normalize(model.item_emb(sampled), dim=-1)
                positive_logits = (query * positive_vectors).sum(-1, keepdim=True)
                negative_logits = torch.einsum(
                    "pd,pkd->pk",
                    query,
                    negative_vectors,
                ).masked_fill(~sampled_valid, -float("inf"))
                logits = torch.cat((positive_logits, negative_logits), dim=-1)
                return F.cross_entropy(
                    logits / TEMPERATURE,
                    torch.zeros(query.shape[0], dtype=torch.long, device=query.device),
                    reduction="sum",
                )

            loss = features.sum() * 0.0
            for start in range(0, features.shape[0], 256):
                stop = start + 256
                loss = loss + checkpoint(
                    chunk_loss,
                    features[start:stop],
                    positive[start:stop],
                    negatives[start:stop],
                    negative_mask[start:stop],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            loss = loss / max(1, features.shape[0])
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
                "maxlen": MAXLEN,
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
    ### >>> Self_EvolveRec-BLOCK-START: Embedded predictor uses the identical trained scorer and validates reload parity
    script = script.replace(
        'DATASETS = [\n    part.strip()\n    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")\n    if part.strip()\n]',
        "DATASETS = list(DEFAULT_DATASETS)",
    )
    script = script.replace(
        "user_emb = out[0, -1]",
        "user_emb = model.final_features(out, seq_t)[0]",
    ).replace(
        "torch.matmul(model.item_emb(cand_t), user_emb)",
        "model.predict_score(user_emb, cand_t)",
    )
    script = script.replace(
        "    elif _models:\n",
        "    elif dataset_name is not None:\n"
        '        raise ValueError(f"Unknown or unavailable dataset: {dataset_name}")\n'
        "    elif _models:\n",
    )
    predictor_path = os.path.join(working_dir, "predict.py")
    with open(predictor_path, "w") as stream:
        stream.write(script)

    import importlib.util

    spec = importlib.util.spec_from_file_location("hstu_predict_parity", predictor_path)
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor._ensure_loaded()
    for dataset_name in DATASETS:
        if dataset_name not in predictor._models:
            raise RuntimeError(f"Missing trained category model: {dataset_name}")
        meta = predictor._metas[dataset_name]
        device = predictor._device
        reference = HSTURec(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
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
        for permitted_history in (history, []):
            sequence = [0] * (
                meta["maxlen"] - len(permitted_history)
            ) + permitted_history
            sequence_t = torch.tensor([sequence], device=device, dtype=torch.long)
            candidates_t = torch.tensor(candidates, device=device, dtype=torch.long)
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
        "Checkpoint/predictor score parity passed for all four categories.", flush=True
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


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
### >>> Self_EvolveRec-BLOCK-START: Bounded paired HSTU sensitivity diagnostics
import json
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F

### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Ground interpretation in sensitivity controls and missingness
def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
Analyze these measurements from one fitted HSTU model. Return concise JSON with
status (CRITICAL, NEEDS_IMPROVEMENT, STABLE, or INSUFFICIENT_EVIDENCE),
core_findings, key_implications, evidence, limitations, and hypotheses.

Measured values, counts, eligibility, and missingness:
{json.dumps(raw_diagnosis['metrics'], indent=2, allow_nan=False)}

Metric definitions:
{json.dumps(raw_diagnosis['metric_definitions'], indent=2, allow_nan=False)}

Rules:
- Ground every finding in measured values and their definitions.
- Report identity failures and missingness first. Null is unavailable, never zero.
  No checked identity rows means insufficient evidence of identity parity.
- These are training-input sensitivity measurements conditional on one fitted
  model and fixed target pairs. Do not infer user intent, causal preferences,
  time sensitivity, model compatibility, or expected validation improvement.
- Ordinary deletion combines information removal and coordinate repacking.
  Keep final-event removal, penultimate-event removal, and prefix deletion separate.
- Embedding cosine geometry does not measure recommendation diversity or accuracy.
  Margin < 0.5 is a score-scale-dependent threshold.
- Rows and replicates are dependent and are not verified unique users. Omit
  user-level confidence intervals and significance claims without verified IDs.
- Loader diagnostics cannot measure validation HR@10 or target-rank changes.
  Do not forecast numerical uplift or invoke unavailable metadata or simulator traits.
- Include at most two explicitly untested hypotheses. Any suggested follow-up must
  follow the predeclared sequence; avoid repeated adaptive validation mining.
- Preserve the HSTU family, one trained checkpoint per category, and unchanged host
  four-category macro validation HR@10 for selecting any future training change.
- Keep core_findings and key_implications to at most three entries each.
"""


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Identity-gated fixed-pair HSTU diagnostics with bounded geometry computation
class DiagnosisProbe:
    CONDITIONS = (
        "prefix_delete_oldest",
        "prefix_delete_recent",
        "prefix_delete_random",
        "delete_final",
        "delete_penultimate",
        "shuffle_prefix",
        "swap_adjacent_prefix",
        "shuffle_before_suffix",
    )

    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = torch.device(device)
        # Preserve the public adapter references without consuming unavailable fields.
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    @staticmethod
    def _unavailable(reason):
        return {"value": None, "reason": reason}

    @staticmethod
    def _last(sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        return positions.expand_as(sequence).masked_fill(sequence.eq(0), -1).amax(-1)

    @classmethod
    def _final(cls, features, sequence):
        last = cls._last(sequence)
        rows = torch.arange(sequence.shape[0], device=sequence.device)
        return features[rows, last.clamp_min(0)].masked_fill(last.lt(0)[:, None], 0)

    def _scores(self, features, candidates):
        # ModelView's paired-candidate contract is B x 1 x D / B x 1 x 2.
        scores = self.view.predict_score(features[:, None, :], candidates[:, None, :])
        expected = (features.shape[0], 1, 2)
        if tuple(scores.shape) != expected:
            raise ValueError(
                f"ModelView returned {tuple(scores.shape)}; expected {expected}"
            )
        return scores[:, 0, :].float()

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        return {
            "count": int(finite.size),
            "nonfinite_count": int(values.size - finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "p10": float(np.quantile(finite, 0.1)) if finite.size else None,
            "reason": None if finite.size else "No finite eligible observations",
        }

    @classmethod
    def _margin_summary(cls, values):
        result = cls._summary(values)
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        result.update(
            {
                "negative_wins_rate": (
                    float((values < 0).mean()) if values.size else None
                ),
                "small_margin_rate": (
                    float((values < 0.5).mean()) if values.size else None
                ),
                "small_margin_threshold": 0.5,
            }
        )
        return result

    def _audit(self):
        supplied = (
            type(self.model).__name__ == "HSTURec"
            and hasattr(self.model, "pos_emb")
            and hasattr(self.model, "blocks")
            and all(type(b).__name__ == "HSTUBlock" for b in self.model.blocks)
        )
        return {
            "actual_module": type(self.model).__name__,
            "documentation_scope": (
                "Supplied HSTURec/HSTUBlock source; structural name checks only, "
                "not an instrumented proof of an arbitrary adapter's implementation"
            ),
            "supplied_structure_matches": supplied,
            "timestamp_inputs": (
                (
                    "Supplied forward(seq) consumes item IDs only; no timestamp inputs "
                    "or next-position temporal query anchors"
                )
                if supplied
                else None
            ),
            "absolute_positions": (
                ("Learned pos_emb at physical slots 1..L, zero on padding")
                if supplied
                else None
            ),
            "relative_positions": (
                "No explicit relative-position bias" if supplied else None
            ),
            "attention": (
                (
                    "Causal MultiheadAttention; padding keys excluded, padding queries "
                    "have a self-key and their outputs are erased after each block"
                )
                if supplied
                else None
            ),
            "aggregation_scaling": (
                (
                    "Input item embeddings scaled by sqrt(hidden_units); standard "
                    "scaled dot-product softmax attention, then value * SiLU(gate)"
                )
                if supplied
                else None
            ),
            "normalization": (
                (
                    "Pre-attention LayerNorm, pre-FFN LayerNorm, final LayerNorm; "
                    "candidate scores use normalized features/items divided by temperature"
                )
                if supplied
                else None
            ),
            "coordinate_control": self._unavailable(
                "Sequence-only ModelView has no verified API for preserving original "
                "absolute/relative coordinates and temporal query anchors while fully "
                "excluding deleted keys, queries, and normalization contributions; "
                "instrumented identity parity is unverified"
            ),
            "time_specific_probe": self._unavailable(
                "No verified timestamp-aware diagnostic path; do not fabricate timestamps"
            ),
        }

    @torch.no_grad()
    def analyze_embeddings_collapse_cosine(self, sample_size=2048, seed=42):
        emb = self.view.item_emb.weight.detach()[1 : self.model.itemnum + 1]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(emb), size=min(sample_size, len(emb)), replace=False)
        x = emb[torch.as_tensor(idx, device=emb.device, dtype=torch.long)].float()
        norms = x.norm(dim=-1)
        valid = torch.isfinite(x).all(-1) & torch.isfinite(norms) & norms.gt(1e-12)
        x = F.normalize(x[valid], dim=-1).double()
        k = len(x)
        score = None
        if k >= 2:
            # Sum all off-diagonal dot products without constructing a k x k Gram matrix.
            off_diagonal_sum = x.sum(dim=0).square().sum() - x.square().sum()
            mean_cos = (off_diagonal_sum / (k * (k - 1))).clamp(-1, 1)
            score = float((mean_cos + 1) / 2)
        return {
            "value": score,
            "sampled_items": len(idx),
            "usable_items": k,
            "excluded_items": len(idx) - k,
            "ordered_pair_denominator": k * (k - 1),
            "reason": (
                None if score is not None else "Fewer than two finite nonzero vectors"
            ),
        }

    @staticmethod
    def _alter(sequence, condition, replicate, batch_id):
        original = sequence.detach().cpu().numpy()
        altered = np.zeros_like(original)
        eligible = np.zeros(len(original), dtype=bool)
        changed = np.zeros(len(original), dtype=bool)
        for row, padded in enumerate(original):
            items = padded[padded != 0]
            n = len(items)
            # Independent local streams; global training RNG state is untouched.
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        42,
                        batch_id,
                        row,
                        replicate,
                        DiagnosisProbe.CONDITIONS.index(condition),
                    ]
                )
            )
            result = items.copy()
            if condition.startswith("prefix_delete_") and n >= 3:
                k = max(1, (n - 1) // 5)
                if condition == "prefix_delete_oldest":
                    remove = np.arange(k)
                elif condition == "prefix_delete_recent":
                    remove = np.arange(n - 1 - k, n - 1)
                else:
                    remove = rng.choice(n - 1, size=k, replace=False)
                result = np.delete(items, remove)
                eligible[row] = True
            elif condition in ("delete_final", "delete_penultimate") and n >= 2:
                result = np.delete(
                    items, n - 1 if condition == "delete_final" else n - 2
                )
                eligible[row] = True
            elif condition == "shuffle_prefix" and n >= 3:
                result[:-1] = rng.permutation(result[:-1])
                eligible[row] = True
            elif condition == "swap_adjacent_prefix" and n >= 3:
                # Nonoverlapping adjacent swaps, with a locally randomized offset.
                offset = 0 if n == 3 else int(rng.integers(0, 2))
                for j in range(offset, n - 2, 2):
                    result[j], result[j + 1] = result[j + 1], result[j]
                eligible[row] = True
            elif condition == "shuffle_before_suffix":
                suffix = min(3, n - 1) if n else 0
                width = n - suffix
                if width >= 2:
                    result[:width] = rng.permutation(result[:width])
                    eligible[row] = True
            if len(result):
                altered[row, -len(result) :] = result
            changed[row] = not np.array_equal(altered[row], padded)
        return (
            torch.as_tensor(altered, device=sequence.device, dtype=sequence.dtype),
            torch.as_tensor(eligible, device=sequence.device),
            torch.as_tensor(changed, device=sequence.device),
        )

    @torch.no_grad()
    def run_full_diagnosis(self, train_loader):
        was_training = self.model.training
        # Preserve any deliberately mixed submodule modes as well.
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            cache, batches = [], []
            # Clone immediately in case the loader reuses its backing buffers.
            for batch_id, batch in enumerate(islice(train_loader, 10)):
                if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                    batches.append(
                        {"batch": batch_id, "reason": "Expected four-element batch"}
                    )
                    continue
                tensors = tuple(
                    torch.as_tensor(x).detach().cpu().clone() for x in batch[1:]
                )
                if (
                    any(x.ndim != 2 for x in tensors)
                    or any(x.shape != tensors[0].shape for x in tensors)
                    or tensors[0].shape[1] == 0
                ):
                    batches.append(
                        {
                            "batch": batch_id,
                            "reason": "Expected matching B x L tensors, L > 0",
                        }
                    )
                    continue
                cache.append((batch_id, tensors))
            records = {condition: [] for condition in self.CONDITIONS}
            identities, training_margins = [], []
            for batch_id, tensors in cache:
                seq, pos, neg = (x.to(self.device, dtype=torch.long) for x in tensors)
                b, length = seq.shape
                if b == 0:
                    batches.append(
                        {"batch": batch_id, "rows": 0, "reason": "Empty batch"}
                    )
                    continue
                rows = torch.arange(b, device=self.device)
                last = self._last(seq)
                candidates = torch.stack(
                    (pos[rows, last.clamp_min(0)], neg[rows, last.clamp_min(0)]), -1
                )
                base_eligible = last.ge(0) & candidates.ne(0).all(-1)
                features = self.view.log2feats(seq)
                original_features = self._final(features, seq)
                scores = self._scores(original_features, candidates)
                margin = scores[:, 0] - scores[:, 1]
                finite_base = torch.isfinite(scores).all(-1) & torch.isfinite(margin)
                eligible = base_eligible & finite_base
                valid_positions = seq.ne(0) & pos.ne(0) & neg.ne(0)
                logits = self.view.predict_score(features, torch.stack((pos, neg), -1))
                if tuple(logits.shape) != (b, length, 2):
                    raise ValueError(
                        "ModelView position scores must have shape B x L x 2"
                    )
                position_margins = (logits[..., 0] - logits[..., 1]).float()
                training_margins.extend(
                    position_margins[valid_positions].cpu().tolist()
                )
                repeat_features = self._final(self.view.log2feats(seq.clone()), seq)
                repeated = self._scores(repeat_features, candidates)
                checked = base_eligible
                close = torch.isclose(scores, repeated, atol=1e-5, rtol=1e-4).all(-1)
                identity_finite = torch.isfinite(scores).all(-1) & torch.isfinite(
                    repeated
                ).all(-1)
                differences = (scores - repeated).abs()[checked & identity_finite]
                identity_failed = int((checked & ~(close & identity_finite)).sum())
                identities.append(
                    {
                        "batch": batch_id,
                        "checked_rows": int(checked.sum()),
                        "failed_rows": identity_failed,
                        "nonfinite_rows": int((checked & ~identity_finite).sum()),
                        "max_absolute_score_difference": (
                            float(differences.max()) if differences.numel() else None
                        ),
                    }
                )
                batches.append(
                    {
                        "batch": batch_id,
                        "rows": b,
                        "tensor_positions": b * length,
                        "nonpadding_positions": int(seq.ne(0).sum()),
                        "valid_target_positions": int(valid_positions.sum()),
                        "nonfinite_position_margins": int(
                            (valid_positions & ~torch.isfinite(position_margins)).sum()
                        ),
                        "empty_rows": int(last.lt(0).sum()),
                        "nonempty_zero_target_rows": int(
                            (last.ge(0) & ~candidates.ne(0).all(-1)).sum()
                        ),
                        "nonfinite_final_score_or_margin_rows": int(
                            (base_eligible & ~finite_base).sum()
                        ),
                        "eligible_final_rows": int(eligible.sum()),
                    }
                )
                for condition in self.CONDITIONS:
                    for replicate in range(5):
                        altered, condition_eligible, changed = self._alter(
                            seq, condition, replicate, batch_id
                        )
                        eligible_before_identity = eligible & condition_eligible
                        # A failed batch-level control invalidates its intervention contrasts.
                        selected = eligible_before_identity & (identity_failed == 0)
                        record = {
                            "identity_excluded_rows": (
                                int(eligible_before_identity.sum())
                                if identity_failed
                                else 0
                            ),
                            "batch": batch_id,
                            "replicate": replicate,
                            "eligible_rows": int(selected.sum()),
                            "length_ineligible_rows": int(
                                (eligible & ~condition_eligible).sum()
                            ),
                            "identity_control_passed": identity_failed == 0,
                        }
                        if not bool(selected.any()):
                            record.update(
                                {
                                    "pairs": [],
                                    "reason": (
                                        "Batch identity control failed; intervention not evaluated"
                                        if identity_failed
                                        else "No eligible rows"
                                    ),
                                }
                            )
                            records[condition].append(record)
                            continue
                        altered_features = self._final(
                            self.view.log2feats(altered), altered
                        )
                        altered_scores = self._scores(altered_features, candidates)
                        altered_margin = altered_scores[:, 0] - altered_scores[:, 1]
                        finite = torch.isfinite(altered_scores).all(
                            -1
                        ) & torch.isfinite(altered_margin)
                        usable = selected & finite
                        record["nonfinite_score_or_margin_rows"] = int(
                            (selected & ~finite).sum()
                        )
                        cosine_valid = (
                            torch.isfinite(original_features).all(-1)
                            & torch.isfinite(altered_features).all(-1)
                            & original_features.float().norm(dim=-1).gt(1e-12)
                            & altered_features.float().norm(dim=-1).gt(1e-12)
                        )
                        cosine = F.cosine_similarity(
                            original_features.float(), altered_features.float(), dim=-1
                        )
                        pairs = []
                        for row in (
                            usable.nonzero(as_tuple=False).flatten().cpu().tolist()
                        ):
                            before, after = float(margin[row]), float(
                                altered_margin[row]
                            )
                            cos_ok = bool(cosine_valid[row]) and bool(
                                torch.isfinite(cosine[row])
                            )
                            pairs.append(
                                {
                                    "row": row,
                                    "original_margin": before,
                                    "altered_margin": after,
                                    "margin_change": after - before,
                                    "negative_wins": after < 0,
                                    "pairwise_order_flip": int(np.sign(before))
                                    != int(np.sign(after)),
                                    "feature_cosine": (
                                        float(cosine[row]) if cos_ok else None
                                    ),
                                    "cosine_reason": (
                                        None
                                        if cos_ok
                                        else "Nonfinite or zero-norm feature"
                                    ),
                                    "input_changed": bool(changed[row]),
                                }
                            )
                        record["pairs"] = pairs
                        records[condition].append(record)
            interventions = {}
            for condition, condition_records in records.items():
                pairs = [
                    p for record in condition_records for p in record.get("pairs", [])
                ]
                cosines = [
                    p["feature_cosine"]
                    for p in pairs
                    if p["feature_cosine"] is not None
                ]
                interventions[condition] = {
                    "replicates_requested": 5,
                    "replicates_with_eligible_rows": len(
                        {
                            r["replicate"]
                            for r in condition_records
                            if r["eligible_rows"]
                        }
                    ),
                    "eligible_row_replicates": sum(
                        r["eligible_rows"] for r in condition_records
                    ),
                    "identity_excluded_row_replicates": sum(
                        r["identity_excluded_rows"] for r in condition_records
                    ),
                    "reason": (
                        None
                        if pairs
                        else (
                            "No finite eligible pairs after length and identity controls; "
                            "see raw_paired_records for exclusions"
                        )
                    ),
                    "finite_pair_denominator": len(pairs),
                    "original_margins": self._margin_summary(
                        [p["original_margin"] for p in pairs]
                    ),
                    "altered_margins": self._margin_summary(
                        [p["altered_margin"] for p in pairs]
                    ),
                    "margin_changes": self._summary(
                        [p["margin_change"] for p in pairs]
                    ),
                    "pairwise_order_flip_rate": (
                        sum(p["pairwise_order_flip"] for p in pairs) / len(pairs)
                        if pairs
                        else None
                    ),
                    "unchanged_input_rate": (
                        sum(not p["input_changed"] for p in pairs) / len(pairs)
                        if pairs
                        else None
                    ),
                    "feature_cosines": self._summary(cosines),
                    "undefined_cosine_count": len(pairs) - len(cosines),
                    "raw_paired_records": condition_records,
                }
            metrics = {
                "backbone_audit": self._audit(),
                "seed": 42,
                "maximum_cached_batches": 10,
                "cached_batches": len(cache),
                "batch_counts": batches,
                "embedding_collapse_score": self.analyze_embeddings_collapse_cosine(),
                "training_pair_separation": self._margin_summary(training_margins),
                "identity": {
                    "atol": 1e-5,
                    "rtol": 1e-4,
                    "batches": identities,
                    "checked_rows": sum(x["checked_rows"] for x in identities),
                    "failed_rows": sum(x["failed_rows"] for x in identities),
                    "passed": (
                        all(x["failed_rows"] == 0 for x in identities)
                        if any(x["checked_rows"] for x in identities)
                        else None
                    ),
                    "reason": (
                        None
                        if any(x["checked_rows"] for x in identities)
                        else "No eligible rows checked; identity parity is unmeasured"
                    ),
                },
                "interventions": interventions,
                "item_category_breakdown": self._unavailable(
                    "Item categories are unavailable in this task"
                ),
                "user_confidence_intervals": self._unavailable(
                    "Loader identities are None; row-to-user mapping is unverified. "
                    "Repeated rows and replicates are dependent; summaries are descriptive only"
                ),
                "paired_validation": self._unavailable(
                    "Training loader cannot calculate validation ranks, HR@10, or top-10 overlap; "
                    "requires separately authorized fixed-candidate host evaluation"
                ),
                "predeclared_follow_up_sequence": [
                    "First resolve identity failures or missingness without changing training.",
                    "If subsequently authorized, evaluate only prefix deletion contrasts under the pinned host protocol.",
                    "Then evaluate only prefix order contrasts under the same fixed users and candidates; no automatic training changes.",
                ],
            }
            return {
                "metrics": metrics,
                "metric_definitions": {
                    "scope": "Conditional on one fitted model and sampled training histories; no retraining or predictor changes.",
                    "counts": "Batch-local row IDs are not user IDs. Exclusions report empty rows, zero targets, length eligibility, and nonfinite values separately.",
                    "embedding_collapse_score": "(Mean sampled off-diagonal embedding cosine + 1)/2; finite nonzero catalog vectors only. A directional geometry statistic, not accuracy or recommendation diversity.",
                    "training_pair_separation": "Untransformed positive-minus-negative scores at nonpadding positions with both targets nonzero. Summary count excludes nonfinite margins; small means margin < 0.5, a scale-dependent threshold.",
                    "fixed_pairs": "Each intervention uses the original final nonpadding position's positive/negative pair. Readout uses masked physical position indices for both original and altered histories, never count-1.",
                    "identity": "Repeated ordinary inference on an unchanged cloned sequence in eval/no_grad mode. Any nonfinite eligible score or tolerance failure invalidates all intervention contrasts from that batch; excluded row-replicates are counted separately. With no checked rows, passed is null with a reason.",
                    "prefix_deletion": "For n >= 3, remove k=max(1,floor(0.2*(n-1))) oldest, most recent, or uniformly sampled prefix events, preserving the final event. Same k within each row.",
                    "endpoint_deletion": "For n >= 2, final-event and penultimate-event removal are separately labeled. Both retain the original candidate pair; final-event removal also changes the readout event.",
                    "order_probes": "Preserve length, padding, final item, and multiset. Shuffle the full prefix, swap disjoint adjacent prefix pairs, or shuffle before a suffix of min(3,n-1) events. At least two permutable events required. Identity draws and repeated item IDs may leave inputs unchanged; these remain in the denominator and are reported.",
                    "coordinate_confounding": "Deletion repacks retained events with left padding at fixed tensor length. It combines information removal with changes in coordinate distances and attention normalization. The supplied absolute embeddings use physical slots. Without instrumented controls, differences do not isolate temporal, positional, or causal preference effects.",
                    "replicates": "Five local seed-42 plans per condition and batch. Deterministic deletions repeat identical subsets; random plans may also repeat. Replicates are not independent fitted models, users, or observations for inference.",
                    "raw_paired_records": "Batch/row/replicate-indexed original and altered raw margins, altered-minus-original change, strict negative-wins indicator, sign-change indicator including ties, and feature cosine. Only finite eligible score pairs are included; missing scores and eligibility counts remain explicit.",
                    "denominators": "Margin and flip summaries use finite_pair_denominator. Cosine summaries use their own finite nonzero-feature count, with null/reason for undefined vectors. Empty measurements have null values and reasons. P10 uses NumPy's interpolated 10th percentile.",
                    "dependence": "Summaries weight sampled row-replicates, not unique users. No user-level bootstrap is justified without verified identity mapping. If later supplied, first aggregate histories/replicates within users, then bootstrap users within categories using 1000 seed-42 resamples; intervals are conditional on this fitted model and fixed candidates, not training-seed uncertainty.",
                    "validation": "Only unchanged host evaluation selects models using four-category macro HR@10, with its 200 users/category, target, 99 unique negatives, eligibility, and seed 42 fixed across any authorized paired comparisons. Training-pair flips are not validation rank changes.",
                    "unavailable_attributes": "No ratings, review text, item categories, prices, stores, or simulator traits are consumed. Public metadata/timestamps are not synthesized.",
                },
            }
        finally:
            self.model.train(was_training)
            for module, training in modes:
                module.training = training


### <<< Self_EvolveRec-BLOCK-END
