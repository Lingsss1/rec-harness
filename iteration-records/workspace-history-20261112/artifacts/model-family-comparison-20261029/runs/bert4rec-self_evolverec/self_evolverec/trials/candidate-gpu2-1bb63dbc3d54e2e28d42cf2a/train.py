"""Per-dataset BERT4Rec baseline for Amazon Reviews.

Trains one independent BERT4Rec checkpoint per dataset while sharing one default
hyper-parameter profile across datasets. The generated predict.py routes by the
optional dataset_name keyword used by the RecHarness Amazon Reviews harness.
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

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

### >>> Self_EvolveRec-BLOCK-START: Preserve required data source and training seed
DATA_DIR = os.environ["GAGC_DATA_DIR"]

SEED = 42
### <<< Self_EvolveRec-BLOCK-END
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
    for split in ("train", "valid"):
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


### >>> Self_EvolveRec-BLOCK-START: Independent reproducible masked diagnostic sampling with bounded negatives
_DIAGNOSTIC_USER_RNG = random.Random(SEED)
_DIAGNOSTIC_MASK_RNG = random.Random(SEED + 101)
_DIAGNOSTIC_NEGATIVE_RNG = random.Random(SEED + 102)


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    if batch_size < 1 or maxlen < 1 or itemnum < 1:
        raise ValueError("batch_size, maxlen and itemnum must be positive")
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)
    for row in range(batch_size):
        history = user_train[_DIAGNOSTIC_USER_RNG.choice(users)]
        items = history[-maxlen:]
        sequences[row, -len(items) :] = items
        eligible = list(range(maxlen - len(items), maxlen))
        selected = [i for i in eligible if _DIAGNOSTIC_MASK_RNG.random() < 0.2]
        if not selected:
            selected = [_DIAGNOSTIC_MASK_RNG.choice(eligible)]
        if _DIAGNOSTIC_MASK_RNG.random() < 0.5 and maxlen - 1 not in selected:
            selected.append(maxlen - 1)
        interacted = set(history)
        for column in selected:
            target = int(sequences[row, column])
            sequences[row, column] = itemnum + 1
            sampled = _uniform_pool(interacted, itemnum, 1, _DIAGNOSTIC_NEGATIVE_RNG)
            # Zero labels explicitly exclude positions without eligible negatives.
            if sampled:
                positives[row, column] = target
                negatives[row, column] = sampled[0]
    return sequences, positives, negatives


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Fixed mixed Cloze supervision with bounded current-model moderate-negative replacement
from torch.utils.checkpoint import checkpoint

# Each invocation trains only this condition, with one network per category.
# The host may compare separate uniform64 and moderate_replacement trials.
NEGATIVE_MODE = os.environ.get("BERT4REC_NEGATIVES", "moderate_replacement")
if NEGATIVE_MODE not in {"uniform64", "moderate_replacement"}:
    raise ValueError(f"Unknown negative condition: {NEGATIVE_MODE}")
INPUT_MICROBATCH = 32


def _uniform_pool(interacted, itemnum, count, rng):
    count = min(count, itemnum - len(interacted))
    if count <= 0:
        return []
    selected, seen = [], set()
    for _ in range(max(128, count * 16)):
        item = rng.randint(1, itemnum)
        if item not in interacted and item not in seen:
            selected.append(item)
            seen.add(item)
            if len(selected) == count:
                return selected
    complement = [
        item
        for item in range(1, itemnum + 1)
        if item not in interacted and item not in seen
    ]
    selected.extend(rng.sample(complement, count - len(selected)))
    return selected


def sample_objective_batch(histories, itemnum, batch_size, maxlen, rngs):
    if batch_size < 2 or maxlen < 2:
        raise ValueError("mixed supervision requires batch_size >= 2 and maxlen >= 2")
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    rows, columns, targets, pools, groups, mine = [], [], [], [], [], []
    excluded = fallback = 0
    terminal_count = batch_size // 2
    for row in range(batch_size):
        history, interacted = rngs["user"].choice(histories)
        terminal = row < terminal_count
        seq = sequences[row]
        if terminal:
            n = len(history)
            t = rngs["mask"].randint(n - min(10, n - 1), n - 1)
            prefix = history[max(0, t - (maxlen - 1)) : t]
            seq[-len(prefix) - 1 : -1] = prefix
            seq[-1] = itemnum + 1
            supervised = [(maxlen - 1, history[t])]
        else:
            items = history[-maxlen:]
            seq[-len(items) :] = items
            eligible = list(range(maxlen - len(items), maxlen))
            selected = [i for i in eligible if rngs["mask"].random() < 0.2]
            if not selected:
                selected = [rngs["mask"].choice(eligible)]
            if rngs["mask"].random() < 0.5 and maxlen - 1 not in selected:
                selected.append(maxlen - 1)
            supervised = [(i, int(seq[i])) for i in selected]
            for column, _ in supervised:
                seq[column] = itemnum + 1
        for column, positive in supervised:
            pool = _uniform_pool(interacted, itemnum, 128, rngs["pool"])
            # Independent stream; both conditions consume identical draws.
            rngs["reservation"].shuffle(pool)
            if not pool:
                excluded += 1
                continue
            full_pool = len(pool) == 128
            fallback += int(not full_pool)
            rows.append(row)
            columns.append(column)
            targets.append(positive)
            pools.append(pool + [0] * (128 - len(pool)))
            groups.append(int(terminal))
            mine.append(full_pool)
    return (
        sequences,
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(pools, dtype=np.int64).reshape(-1, 128),
        np.asarray(groups, dtype=np.int64),
        np.asarray(mine, dtype=bool),
        excluded,
        fallback,
    )


@torch.no_grad()
def _select_negatives(model, sequence, rows, columns, pools, mine):
    selected = pools[:, :64].clone()
    if NEGATIVE_MODE != "moderate_replacement" or not bool(mine.any()):
        return selected
    was_training = model.training
    try:
        model.eval()
        # Recompute dropout-free features from the identical masked inputs.
        hidden = model.log2feats(sequence)[rows[mine], columns[mine]]
        remainder = pools[mine, 48:]
        replacements = []
        for start in range(0, len(hidden), 128):
            ids = remainder[start : start + 128]
            # Sort IDs first; stable score sorting then breaks ties by ID.
            ids = ids.sort(dim=1).values
            logits = model.predict_score(hidden[start : start + 128], ids)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite mining scores")
            order = torch.argsort(logits, dim=1, descending=True, stable=True)
            replacements.append(ids.gather(1, order[:, 8:24]))
        selected[mine, 48:64] = torch.cat(replacements, dim=0)
    finally:
        model.train(was_training)
    return selected


def backward_objective_batch(model, batch, device):
    sequence, rows, columns, targets, pools, groups, mine, excluded, fallback = batch
    counts = (int((groups == 0).sum()), int((groups == 1).sum()))
    if not len(rows):
        return None, counts, excluded, fallback
    active_components = sum(count > 0 for count in counts)
    total = 0.0

    def candidate_losses(hidden, ids):
        logits = model.predict_score(hidden, ids)
        logits = logits.masked_fill(ids == 0, float("-inf"))
        return torch.nn.functional.cross_entropy(
            logits,
            torch.zeros(len(hidden), dtype=torch.long, device=hidden.device),
            reduction="none",
        )

    # Backward each input microbatch immediately, using full-batch denominators.
    # Parameters remain fixed until all microbatches have contributed gradients.
    for start in range(0, len(sequence), INPUT_MICROBATCH):
        stop = min(start + INPUT_MICROBATCH, len(sequence))
        keep = (rows >= start) & (rows < stop)
        if not keep.any():
            continue
        seq_t = torch.as_tensor(sequence[start:stop], dtype=torch.long, device=device)
        row_t = torch.as_tensor(rows[keep] - start, dtype=torch.long, device=device)
        col_t = torch.as_tensor(columns[keep], dtype=torch.long, device=device)
        pool_t = torch.as_tensor(pools[keep], dtype=torch.long, device=device)
        mine_t = torch.as_tensor(mine[keep], dtype=torch.bool, device=device)
        negatives = _select_negatives(model, seq_t, row_t, col_t, pool_t, mine_t)
        positive = torch.as_tensor(targets[keep], dtype=torch.long, device=device)
        candidates = torch.cat((positive[:, None], negatives), dim=1)
        weights = np.asarray(
            [1.0 / (active_components * counts[int(group)]) for group in groups[keep]],
            dtype=np.float32,
        )
        weight_t = torch.as_tensor(weights, device=device)
        model.train()
        hidden = model.log2feats(seq_t)[row_t, col_t]
        terms = []
        for offset in range(0, len(hidden), 256):
            # Recompute tied candidate embeddings on backward to bound memory.
            losses = checkpoint(
                candidate_losses,
                hidden[offset : offset + 256],
                candidates[offset : offset + 256],
                use_reentrant=False,
                preserve_rng_state=False,
            )
            terms.append((losses * weight_t[offset : offset + 256]).sum())
        loss = torch.stack(terms).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite mixed candidate loss")
        total += float(loss.detach().item())
        loss.backward()
    return total, counts, excluded, fallback


### <<< Self_EvolveRec-BLOCK-END


def _history_to_tensor(
    history: list[int], maxlen: int, device: torch.device
) -> torch.Tensor:
    seq_arr = np.zeros(maxlen, dtype=np.int64)
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    return torch.tensor([seq_arr], dtype=torch.long, device=device)


def evaluate_val(
    model: nn.Module,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    itemnum: int,
    device: torch.device,
    maxlen: int,
    num_users: int = 200,
) -> dict[str, float]:
    ### >>> Self_EvolveRec-BLOCK-START: Freeze local validation sampling without consuming training randomness
    # Local checkpoint validation only; the host benchmark remains unchanged.
    rng = random.Random(SEED)
    val_users = [u for u in user_valid if u in user_train and user_valid[u]]
    if len(val_users) > num_users:
        val_users = rng.sample(val_users, num_users)
    ### <<< Self_EvolveRec-BLOCK-END
    if not val_users:
        return {"HR@10": 0.0, "HR@20": 0.0, "NDCG@10": 0.0, "NDCG@20": 0.0}

    hr10 = hr20 = ndcg10 = ndcg20 = 0.0
    model.eval()
    with torch.no_grad():
        for user in val_users:
            target = user_valid[user][0]
            history = user_train[user]
            interacted = set(history) | {target, 0}
            ### >>> Self_EvolveRec-BLOCK-START: Reproducible unique local validation negatives
            candidates = [target]
            available = itemnum - sum(1 for i in interacted if 1 <= i <= itemnum)
            candidate_count = 1 + min(99, available)
            while len(candidates) < candidate_count:
                neg_item = rng.randint(1, itemnum)
                if neg_item not in interacted:
                    interacted.add(neg_item)
                    candidates.append(neg_item)
            ### <<< Self_EvolveRec-BLOCK-END
            seq_t = _history_to_tensor(history, maxlen, device)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
            rank = sorted(range(len(scores)), key=lambda i: -scores[i]).index(0)
            hr10 += 1.0 if rank < 10 else 0.0
            hr20 += 1.0 if rank < 20 else 0.0
            ndcg10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
            ndcg20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0
    n = float(len(val_users))
    model.train()
    return {
        "HR@10": hr10 / n,
        "HR@20": hr20 / n,
        "NDCG@10": ndcg10 / n,
        "NDCG@20": ndcg20 / n,
    }


def bpr_sequence_loss(
    model: nn.Module, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
) -> torch.Tensor:
    ### >>> Self_EvolveRec-BLOCK-START: Numerically stable masked-position BPR
    pos_scores, neg_scores = model(seq, pos, neg)
    selected = pos != 0
    if not selected.any():
        raise RuntimeError("Cloze batch contains no supervised masked positions")
    return torch.nn.functional.softplus(
        neg_scores[selected] - pos_scores[selected]
    ).mean()
    ### <<< Self_EvolveRec-BLOCK-END


# Hyper-parameters
MAXLEN = int(os.environ.get("BERT4REC_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("BERT4REC_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("BERT4REC_BLOCKS", "2"))
NUM_HEADS = int(os.environ.get("BERT4REC_HEADS", "2"))
DROPOUT_RATE = float(os.environ.get("BERT4REC_DROPOUT", "0.2"))
LR = float(os.environ.get("BERT4REC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("BERT4REC_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("BERT4REC_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Fixed training steps and predeclared validation cadence
NUM_BATCHES = 200
NUM_EPOCHS = max(1, min(200, int(os.environ.get("BERT4REC_EPOCHS", "200"))))
EVAL_EVERY = 10
### <<< Self_EvolveRec-BLOCK-END
PATIENCE = int(os.environ.get("BERT4REC_PATIENCE", "20"))


class BERT4RecModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.mask_token = itemnum + 1
        self.item_emb = nn.Embedding(itemnum + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Trim common left padding while preserving absolute positions and feature shape
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        if length == 0 or length > self.maxlen:
            raise ValueError("sequence length must be between 1 and model.maxlen")
        occupied_columns = (seq != 0).any(dim=0)
        indices = occupied_columns.nonzero(as_tuple=False).flatten()
        start = int(indices[0].item()) if indices.numel() else length - 1
        compact = seq[:, start:]
        # Keep the original positional IDs; trimming must not reposition items.
        pos = (
            torch.arange(start + 1, length + 1, device=seq.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        pos = pos * (compact != 0).long()
        x = self.item_emb(compact) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        padding = compact == 0
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        x = self.norm(x).masked_fill(padding.unsqueeze(-1), 0.0)
        # Preserve B x L x D for forward(), ModelView, and diagnostic probes.
        return torch.nn.functional.pad(x, (0, 0, start, 0))

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Expose masked BERT features and tied item scores
    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.item_emb(items)
        if embeddings.ndim == features.ndim + 1:
            features = features.unsqueeze(-2)
        return (features * embeddings).sum(dim=-1)

    ### <<< Self_EvolveRec-BLOCK-END

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    ### >>> Self_EvolveRec-BLOCK-START: Explicit final nonpadding MASK readout and catalog-only candidate scores
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.shape[0] != 1:
            raise ValueError("score_candidates expects one left-padded history")
        if not 1 <= seq.shape[1] <= self.maxlen:
            raise ValueError("invalid sequence length")
        if candidates.ndim != 1:
            raise ValueError("candidates must be one-dimensional")
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        out = self.log2feats(prediction_seq)
        positions = torch.arange(seq.shape[1], device=seq.device)
        final_position = positions.masked_fill(prediction_seq[0] == 0, -1).max()
        features = out[0, final_position]
        valid = (candidates >= 1) & (candidates <= self.itemnum)
        safe_candidates = candidates.masked_fill(~valid, 0)
        return self.predict_score(features, safe_candidates).masked_fill(
            ~valid, float("-inf")
        )

    ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Matched initialization and independent example and candidate RNG streams
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    rng_seeds = {
        "user": SEED,
        "mask": SEED + 1,
        "pool": SEED + 2,
        "reservation": SEED + 3,
    }
    rngs = {name: random.Random(seed) for name, seed in rng_seeds.items()}
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    terminal_histories = [
        (history, set(history)) for history in user_train.values() if len(history) >= 2
    ]
    if not terminal_histories:
        raise RuntimeError("no eligible training histories")
    if not any(len(seen) < itemnum for _, seen in terminal_histories):
        raise RuntimeError("no eligible unseen training negatives")
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    print(
        f"  condition={NEGATIVE_MODE} objective=candidate_ce temperature=1.0 "
        "component_weights=0.5,0.5 recent_targets=10 negatives=64 "
        "pool=128 reserved=48 mined_descending_ranks=9:24 "
        "cloze_mask_rate=0.2 force_last_probability=0.5",
        flush=True,
    )
    ### <<< Self_EvolveRec-BLOCK-END
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = BERT4RecModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    best_hr10 = -1.0
    best_epoch = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ### >>> Self_EvolveRec-BLOCK-START: Microbatch mixed supervision with one optimizer step per valid batch
        total_loss = 0.0
        updates = excluded_targets = fallback_targets = single_component_batches = 0
        cloze_targets = terminal_targets = 0
        for _ in range(NUM_BATCHES):
            batch = sample_objective_batch(
                terminal_histories, itemnum, BATCH_SIZE, MAXLEN, rngs
            )
            optimizer.zero_grad(set_to_none=True)
            loss_value, counts, excluded, fallback = backward_objective_batch(
                model, batch, device
            )
            cloze_targets += counts[0]
            terminal_targets += counts[1]
            excluded_targets += excluded
            fallback_targets += fallback
            if loss_value is None:
                continue
            single_component_batches += int(min(counts) == 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
            total_loss += loss_value

        avg_loss = total_loss / max(updates, 1)
        print(
            f"  [{dataset_name}] epoch={epoch} updates={updates}"
            f" cloze_targets={cloze_targets} terminal_targets={terminal_targets}"
            f" excluded_targets={excluded_targets} fallback_targets={fallback_targets}"
            f" skipped_batches={NUM_BATCHES - updates}"
            f" single_component_batches={single_component_batches}",
            flush=True,
        )
        ### <<< Self_EvolveRec-BLOCK-END
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            val_m = evaluate_val(model, user_train, user_valid, itemnum, device, MAXLEN)
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
            ### >>> Self_EvolveRec-BLOCK-START: Preserve the configured training horizon and earlier checkpoint ties
            # Complete the configured epochs; strict HR improvement above keeps
            # the earlier checkpoint on ties. Host macro scoring remains separate.
            ### <<< Self_EvolveRec-BLOCK-END
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    ### >>> Self_EvolveRec-BLOCK-START: Record the actual single-condition objective and RNG states
    manifest = {
        "dataset": dataset_name,
        "seed": SEED,
        "condition": NEGATIVE_MODE,
        "loss": "uncorrected sampled candidate cross entropy",
        "temperature": 1.0,
        "component_reduction": "full-batch masked-position mean per component",
        "component_weights": [0.5, 0.5],
        "terminal_recent_count": 10,
        "cloze_mask_rate": 0.2,
        "cloze_force_last_probability": 0.5,
        "cloze_ensure_one_mask": True,
        "corruption": "every supervised target replaced by reserved MASK",
        "negative_exclusions": "complete training history; PAD; MASK",
        "pool_size": 128,
        "negative_count": 64,
        "uniform_reservation": 48,
        "mining_remainder": 80,
        "mining_descending_ranks_inclusive": [9, 24],
        "mining_tie_rule": "ascending item ID",
        "small_pool_fallback": "uniform min(64, eligible_count)",
        "input_microbatch": INPUT_MICROBATCH,
        "epochs_completed": epoch,
        "batches_per_epoch": NUM_BATCHES,
        "validation_every": EVAL_EVERY,
        "checkpoint_epoch": best_epoch,
        "checkpoint_rule": "existing local category validation HR@10; earlier ties",
        "selection_scope": "host independently evaluates the four-category bundle",
        "rng_implementation": "python.Random independent streams; torch dropout",
        "rng_seeds": rng_seeds,
        "torch_seed": SEED,
        "gpu_bitwise_reproducibility_verified": False,
    }
    with open(os.path.join(working_dir, f"{dataset_name}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(working_dir, f"{dataset_name}_rng.pkl"), "wb") as f:
        pickle.dump(
            {
                "streams": {name: rng.getstate() for name, rng in rngs.items()},
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            },
            f,
        )
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    ### <<< Self_EvolveRec-BLOCK-END
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "num_heads": NUM_HEADS,
                "dropout_rate": DROPOUT_RATE,
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""Per-dataset BERT4Rec predict function."""
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
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]
_models = {}
_metas = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"bert4rec_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BERT4RecModel


def _ensure_loaded():
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = _load_model_class()
    for ds in DATASETS:
        working_dir = _HERE if os.path.basename(_HERE) == "working" else os.path.join(_HERE, "working")
        meta_path = os.path.join(working_dir, f"{ds}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{ds}_model.pt")
        ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly when a required category artifact is missing
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing required BERT4Rec artifacts for {ds!r}: "
                f"{meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved single BERT4Rec configuration
            dropout_rate=meta["dropout_rate"],
            maxlen=meta["maxlen"],
            ### <<< Self_EvolveRec-BLOCK-END
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject missing category checkpoints instead of silently misrouting predictions
    if dataset_name is not None:
        if dataset_name not in _models:
            raise RuntimeError(f"No trained BERT4Rec checkpoint for {dataset_name!r}")
        ds = dataset_name
    elif len(_models) == 1:
        ds = next(iter(_models))
    else:
        raise ValueError("dataset_name is required when multiple category models are loaded")
    model = _models[ds]
    meta = _metas[ds]
    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Match the trained appended-MASK inference contract
    maxlen = int(meta["maxlen"])
    if maxlen < 2:
        raise ValueError("saved maxlen must be at least two")
    seq_arr = [0] * maxlen
    # The model shifts once and appends MASK, keeping maxlen-1 history items.
    for i, item in enumerate(reversed(history[-(maxlen - 1):])):
        item = int(item)
        if not 1 <= item <= model.itemnum:
            raise ValueError(f"history contains a noncatalog item: {item}")
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    ### <<< Self_EvolveRec-BLOCK-END
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        scores = model.score_candidates(seq_t, cand_t).detach().cpu().tolist()
    return [float(s) for s in scores]
'''
    predict_path = os.path.join(working_dir, "predict.py")
    with open(predict_path, "w") as f:
        f.write(script)
    print(f"predict.py written -> {predict_path}", flush=True)


def main() -> None:
    ### >>> Self_EvolveRec-BLOCK-START: Require exactly the four benchmark categories before training
    if len(DATASETS) != len(DEFAULT_DATASETS) or set(DATASETS) != set(DEFAULT_DATASETS):
        raise ValueError(
            "GAGC_DATASETS must contain each required benchmark category exactly once: "
            + ", ".join(DEFAULT_DATASETS)
        )
    ### <<< Self_EvolveRec-BLOCK-END
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Per-dataset BERT4Rec training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


