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


