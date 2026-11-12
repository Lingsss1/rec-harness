# === main_code.py ===
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


### >>> Self_EvolveRec-BLOCK-START: Isolate diagnostic randomness and bound diagnostic negative sampling
_DIAGNOSTIC_USER_RNG = random.Random(SEED)
_DIAGNOSTIC_MASK_RNG = random.Random(SEED + 101)
_DIAGNOSTIC_NEGATIVE_RNG = random.Random(SEED + 102)


def sample_batch(
    user_train: dict[int, list[int]], itemnum: int, batch_size: int, maxlen: int
):
    # Preserve the host's B x L diagnostic batch contract. These streams do
    # not consume optimization user, masking, negative, or dropout randomness.
    if batch_size < 1 or maxlen < 1 or itemnum < 1:
        raise ValueError("batch_size, maxlen and itemnum must be positive")
    users = [u for u, items in user_train.items() if len(items) >= 2]
    if not users:
        raise RuntimeError("no users with at least two training interactions")
    seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(seqs)
    negatives = np.zeros_like(seqs)
    for row in range(batch_size):
        history = user_train[_DIAGNOSTIC_USER_RNG.choice(users)]
        items = history[-maxlen:]
        seqs[row, -len(items) :] = items
        eligible = list(range(maxlen - len(items), maxlen))
        selected = [i for i in eligible if _DIAGNOSTIC_MASK_RNG.random() < 0.2]
        if not selected:
            selected = [_DIAGNOSTIC_MASK_RNG.choice(eligible)]
        if _DIAGNOSTIC_MASK_RNG.random() < 0.5 and maxlen - 1 not in selected:
            selected.append(maxlen - 1)
        interacted = set(history)
        for column in selected:
            target = int(seqs[row, column])
            seqs[row, column] = itemnum + 1
            sampled = _uniform_unseen(interacted, itemnum, 1, _DIAGNOSTIC_NEGATIVE_RNG)
            # A zero positive explicitly excludes a position with no valid
            # negative from probes that select positive != 0.
            if sampled:
                positives[row, column] = target
                negatives[row, column] = sampled[0]
    return seqs, positives, negatives


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Factorial masking and loss controls with separately normalized terminal and Cloze supervision
from torch.utils.checkpoint import checkpoint

# One invocation trains one configuration, with one checkpoint per category.
# The host can schedule the four cells independently; their scores are never fused.
# Cells: baseline/bpr, mixed_terminal_cloze/bpr,
#        baseline/candidate_ce, mixed_terminal_cloze/candidate_ce.
MASKING_MODE = os.environ.get("BERT4REC_MASKING", "mixed_terminal_cloze")
LOSS_MODE = os.environ.get("BERT4REC_LOSS", "candidate_ce")
if MASKING_MODE not in {"baseline", "mixed_terminal_cloze"}:
    raise ValueError(f"Unknown masking mode: {MASKING_MODE}")
if LOSS_MODE not in {"bpr", "candidate_ce"}:
    raise ValueError(f"Unknown loss mode: {LOSS_MODE}")


def _uniform_unseen(interacted, itemnum, count, rng):
    count = min(count, itemnum - len(interacted))
    if count <= 0:
        return []
    chosen, chosen_set = [], set()
    # Bounded rejection followed by an exact complement fallback. Conditional
    # draws are uniform and without replacement, including across the fallback.
    for _ in range(max(128, count * 16)):
        item = rng.randint(1, itemnum)
        if item not in interacted and item not in chosen_set:
            chosen.append(item)
            chosen_set.add(item)
            if len(chosen) == count:
                return chosen
    remaining = [
        item
        for item in range(1, itemnum + 1)
        if item not in interacted and item not in chosen_set
    ]
    chosen.extend(rng.sample(remaining, count - len(chosen)))
    return chosen


def sample_objective_batch(
    histories, itemnum, batch_size, maxlen, user_rng, mask_rng, negative_rng
):
    mixed = MASKING_MODE == "mixed_terminal_cloze"
    if mixed and (batch_size < 2 or maxlen < 2):
        raise ValueError("mixed training requires batch_size >= 2 and maxlen >= 2")
    terminal_count = max(1, batch_size // 2) if mixed else 0
    negative_count = 64 if LOSS_MODE == "candidate_ce" else 1
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    rows, columns, candidates, groups = [], [], [], []
    excluded_targets = 0
    for row in range(batch_size):
        history, interacted = user_rng.choice(histories)
        seq = sequences[row]
        terminal = row < terminal_count
        if terminal:
            n = len(history)
            # Target indices are zero-based; no future suffix enters the input.
            t = mask_rng.randint(n - min(10, n - 1), n - 1)
            prefix = history[max(0, t - (maxlen - 1)) : t]
            seq[-len(prefix) - 1 : -1] = prefix
            seq[-1] = itemnum + 1
            targets = [(maxlen - 1, history[t])]
        else:
            # Exact supplied baseline Cloze window and target-selection rules.
            items = history[-maxlen:]
            seq[-len(items) :] = items
            eligible = list(range(maxlen - len(items), maxlen))
            selected = [i for i in eligible if mask_rng.random() < 0.2]
            if not selected:
                selected = [mask_rng.choice(eligible)]
            if mask_rng.random() < 0.5 and maxlen - 1 not in selected:
                selected.append(maxlen - 1)
            targets = [(i, int(seq[i])) for i in selected]
            for i, _ in targets:
                seq[i] = itemnum + 1
        for column, positive in targets:
            negatives = _uniform_unseen(
                interacted, itemnum, negative_count, negative_rng
            )
            if not negatives:
                excluded_targets += 1
                continue
            candidate = [positive] + negatives
            candidate.extend([0] * (negative_count + 1 - len(candidate)))
            rows.append(row)
            columns.append(column)
            candidates.append(candidate)
            groups.append(int(terminal))
    return (
        sequences,
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(candidates, dtype=np.int64).reshape(-1, negative_count + 1),
        np.asarray(groups, dtype=np.int64),
        excluded_targets,
    )


def objective_batch_loss(model, batch, device):
    sequences, rows, columns, candidates, groups, excluded = batch
    counts = (int((groups == 0).sum()), int((groups == 1).sum()))
    if not len(rows):
        return None, counts, excluded
    seq_t = torch.as_tensor(sequences, dtype=torch.long, device=device)
    row_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    col_t = torch.as_tensor(columns, dtype=torch.long, device=device)
    features = model.log2feats(seq_t)[row_t, col_t]
    candidate_t = torch.as_tensor(candidates, dtype=torch.long, device=device)
    group_t = torch.as_tensor(groups, dtype=torch.long, device=device)

    def target_losses(hidden, item_ids):
        logits = model.predict_score(hidden, item_ids)
        if LOSS_MODE == "bpr":
            return torch.nn.functional.softplus(logits[:, 1] - logits[:, 0])
        logits = logits.masked_fill(item_ids == 0, float("-inf"))
        return torch.nn.functional.cross_entropy(
            logits,
            torch.zeros(len(hidden), dtype=torch.long, device=hidden.device),
            reduction="none",
        )

    parts = []
    for start in range(0, len(rows), 512):
        # Recompute candidate embedding lookups on backward so chunking also
        # bounds saved M*K*D activations, rather than only forward temporaries.
        parts.append(
            checkpoint(
                target_losses,
                features[start : start + 512],
                candidate_t[start : start + 512],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        )
    losses = torch.cat(parts)
    component_means = [
        losses[group_t == component].mean()
        for component in (0, 1)
        if counts[component] > 0
    ]
    # Exactly 0.5 per component when both exist, irrespective of mask counts.
    return torch.stack(component_means).mean(), counts, excluded


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

    ### >>> Self_EvolveRec-BLOCK-START: Explicit appended-mask readout and real-catalog candidate filtering
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.shape[0] != 1:
            raise ValueError("score_candidates expects one left-padded history")
        if not 1 <= seq.shape[1] <= self.maxlen:
            raise ValueError("history tensor length exceeds model capacity")
        if candidates.ndim != 1:
            raise ValueError("candidates must be a one-dimensional tensor")
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        out = self.log2feats(prediction_seq)
        positions = torch.arange(seq.shape[1], device=seq.device)
        final_position = positions.masked_fill(prediction_seq[0] == 0, -1).max()
        features = out[0, final_position]
        valid = (candidates >= 1) & (candidates <= self.itemnum)
        safe_candidates = candidates.masked_fill(~valid, 0)
        scores = self.predict_score(features, safe_candidates)
        return scores.masked_fill(~valid, float("-inf"))

    ### <<< Self_EvolveRec-BLOCK-END


def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    ### >>> Self_EvolveRec-BLOCK-START: Seeded independent sampling streams and recorded factorial cell
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_rng = random.Random(SEED)
    mask_rng = random.Random(SEED + 1)
    negative_rng = random.Random(SEED + 2)
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
        f"  masking={MASKING_MODE} loss={LOSS_MODE} "
        "cloze_mask_rate=0.2 force_last_probability=0.5 "
        "terminal_recent_count=10 component_reduction=masked_position_mean "
        "mixed_component_weights=0.5,0.5 candidate_negatives=64 temperature=1.0 "
        "rng=python.Random user_seed=42 mask_seed=43 negative_seed=44",
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
        ### >>> Self_EvolveRec-BLOCK-START: Fixed within-batch mixture with independently controlled candidate objective
        total_loss = 0.0
        cloze_targets = terminal_targets = excluded_targets = 0
        updates = skipped_batches = single_component_batches = 0
        for _ in range(NUM_BATCHES):
            batch = sample_objective_batch(
                terminal_histories,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                user_rng,
                mask_rng,
                negative_rng,
            )
            loss, counts, excluded = objective_batch_loss(model, batch, device)
            cloze_targets += counts[0]
            terminal_targets += counts[1]
            excluded_targets += excluded
            if loss is None:
                skipped_batches += 1
                continue
            if MASKING_MODE == "mixed_terminal_cloze" and min(counts) == 0:
                single_component_batches += 1
            updates += 1
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Avoid unnecessary gradient-buffer clearing
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ### <<< Self_EvolveRec-BLOCK-END
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        ### >>> Self_EvolveRec-BLOCK-START: Report realized component exposure and skipped supervision
        avg_loss = total_loss / max(updates, 1)
        print(
            f"  [{dataset_name}] epoch={epoch} updates={updates}"
            f" cloze_targets={cloze_targets} terminal_targets={terminal_targets}"
            f" excluded_targets={excluded_targets} skipped_batches={skipped_batches}"
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
            ### >>> Self_EvolveRec-BLOCK-START: Keep the declared epoch exposure identical across factorial cells
            # Validation selects a single checkpoint; it does not change the
            # configured training horizon. Strict improvement retains earlier ties.
            ### <<< Self_EvolveRec-BLOCK-END
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    ### >>> Self_EvolveRec-BLOCK-START: Persist actual sampling configuration and reproducible stream states
    manifest = {
        "dataset": dataset_name,
        "training_seed": SEED,
        "masking": MASKING_MODE,
        "loss": LOSS_MODE,
        "baseline_loss": "masked-position mean softplus(negative-positive)",
        "cloze_mask_rate": 0.2,
        "cloze_force_last_probability": 0.5,
        "cloze_ensure_one_mask": True,
        "cloze_corruption": "all selected targets replaced with MASK",
        "terminal_recent_count": 10,
        "component_weights": [0.5, 0.5] if MASKING_MODE != "baseline" else [1.0, 0.0],
        "negative_count": 64 if LOSS_MODE == "candidate_ce" else 1,
        "negative_exclusions": "complete user training history; PAD; MASK",
        "negative_sampling": "uniform without replacement per target",
        "temperature": 1.0,
        "rng": "python.Random independent streams",
        "rng_seeds": {"user": SEED, "mask": SEED + 1, "negative": SEED + 2},
        "epochs_completed": epoch,
        "batches_per_epoch": NUM_BATCHES,
        "validation_every": EVAL_EVERY,
        "checkpoint_epoch": best_epoch,
        "checkpoint_rule": "category local validation HR@10; earlier ties",
        "selection_scope": "host compares complete four-category configuration by macro HR@10",
    }
    with open(os.path.join(working_dir, f"{dataset_name}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(working_dir, f"{dataset_name}_rng.pkl"), "wb") as f:
        pickle.dump(
            {
                "user": user_rng.getstate(),
                "mask": mask_rng.getstate(),
                "negative": negative_rng.getstate(),
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
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
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

    maxlen = int(meta.get("maxlen", 128))
    ### >>> Self_EvolveRec-BLOCK-START: Emit history validation consistent with appended-mask model inference
    if maxlen < 1:
        raise ValueError("saved maxlen must be positive")
    seq_arr = [0] * maxlen
    # score_candidates shifts once to reserve the final position for MASK,
    # retaining exactly the most recent maxlen-1 history items.
    for i, item in enumerate(reversed(history[-maxlen:])):
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


# === self_evolverec_interface.py ===
def self_evolverec_interface():
    raise RuntimeError("Training and grading are executed by the common host boundary.")


# === diagnosis_tools.py ===
import torch
import numpy as np
import torch.nn.functional as F
import json
import random

### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic batch consumption exactly
from collections import Counter
from itertools import islice

### <<< Self_EvolveRec-BLOCK-END


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
    - Every statement must be grounded in the provided metric values/definitions.
    - If metrics look healthy, status should remain STABLE and implications should focus on "what is already working" plus one potential risk to monitor.
    """


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Bind the actual BERT4Rec module and respect unavailable metadata
    def __init__(self, view, device):
        self.view = view
        self.model = view.model
        self.device = device
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train

    def _get_categories(self, item_id: int):
        # Item categories are unavailable in this shared task.
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            ### >>> Self_EvolveRec-BLOCK-START: Measure catalog embeddings only and report empty geometry as unavailable
            emb = self.view.item_emb.weight.detach()[1 : self.model.itemnum + 1]
            n = emb.size(0)
            if min(sample_size, n) < 2:
                return None
            ### <<< Self_EvolveRec-BLOCK-END

            g = torch.Generator(device=emb.device)
            g.manual_seed(seed)

            k = min(sample_size, n)
            idx = torch.randperm(n, generator=g, device=emb.device)[:k]
            x = emb.index_select(0, idx)
            x = F.normalize(x, p=2, dim=-1)

            ### >>> Self_EvolveRec-BLOCK-START: Compute the same mean cosine without a quadratic Gram matrix
            x = x.float()
            if not torch.isfinite(x).all():
                raise RuntimeError(
                    "Nonfinite catalog embeddings in diagnostic geometry"
                )
            # Sum of all pairwise dot products equals the squared norm of
            # the vector sum. Subtract the actual diagonal, including zeros.
            summed = x.sum(dim=0)
            offdiag_sum = summed.square().sum() - x.square().sum()
            offdiag_cnt = k * (k - 1)
            mean_cos = (offdiag_sum / offdiag_cnt).clamp(-1, 1).item()
            ### <<< Self_EvolveRec-BLOCK-END

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
        ### >>> Self_EvolveRec-BLOCK-START: Preserve the trained module's mode during diagnostic inference
        was_training = self.model.training
        self.model.eval()
        try:
            return self._measure_masked_margins(
                dataloader, num_batches, eps, topk_categories
            )
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def _measure_masked_margins(self, dataloader, num_batches, eps, topk_categories):
        margins_all = []
        ### <<< Self_EvolveRec-BLOCK-END
        low_pos_cat, low_neg_cat = Counter(), Counter()
        wrong_pos_cat, wrong_neg_cat = Counter(), Counter()
        low_total, wrong_total = 0, 0

        ### >>> Self_EvolveRec-BLOCK-START: Bound diagnostic consumption and reject malformed batches explicitly
        if not isinstance(num_batches, int) or num_batches < 0:
            raise ValueError("num_batches must be a nonnegative integer")
        for batch in islice(dataloader, num_batches):
            if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                raise ValueError(
                    "Diagnosis requires (None, sequence, positive, negative) batches"
                )
            ### <<< Self_EvolveRec-BLOCK-END

            _, seq, pos, neg = batch[0], batch[1], batch[2], batch[3]

            ### >>> Self_EvolveRec-BLOCK-START: Score only valid supervised BERT4Rec mask positions without exposing labels
            seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.long)
            pos_t = torch.as_tensor(pos, device=self.device, dtype=torch.long)
            neg_t = torch.as_tensor(neg, device=self.device, dtype=torch.long)
            if (
                seq_t.ndim != 2
                or pos_t.shape != seq_t.shape
                or neg_t.shape != seq_t.shape
            ):
                raise ValueError(
                    "Diagnosis requires matching B x L sequence/positive/negative tensors"
                )
            itemnum = self.model.itemnum
            valid = (
                (seq_t == self.model.mask_token)
                & (pos_t >= 1)
                & (pos_t <= itemnum)
                & (neg_t >= 1)
                & (neg_t <= itemnum)
                & (neg_t != pos_t)
            )
            if not valid.any():
                continue

            # Labels are used exclusively for scoring, never as encoder inputs.
            features = self.model.log2feats(seq_t)[valid]
            all_items = torch.stack([pos_t[valid], neg_t[valid]], dim=-1)
            all_logits = self.model.predict_score(features, all_items)
            measured = all_logits[..., 0] - all_logits[..., 1]
            if not torch.isfinite(measured).all():
                raise RuntimeError("Nonfinite BERT4Rec diagnostic margins")
            margins_all.append(measured.detach().float().cpu())

            # The measured margins are already on CPU in margins_all[-1].
            ### <<< Self_EvolveRec-BLOCK-END

            ### >>> Self_EvolveRec-BLOCK-START: Count valid margins directly without dense tensors or extra device transfers
            measured_cpu = margins_all[-1]
            low_total += int((measured_cpu < float(eps)).sum().item())
            wrong_total += int((measured_cpu < 0.0).sum().item())
            # Category counters remain empty because annotations are unavailable.
            ### <<< Self_EvolveRec-BLOCK-END

        if len(margins_all) == 0:
            ### >>> Self_EvolveRec-BLOCK-START: Represent absent supervised measurements as unavailable
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

        ### >>> Self_EvolveRec-BLOCK-START: Report measured sample counts and unavailable category annotations
        breakdown = {
            "category_status": "unavailable: item categories are not supplied in this task",
            "supervised_masked_positions": sum(int(m.numel()) for m in margins_all),
            "eps": float(eps),
            ### <<< Self_EvolveRec-BLOCK-END
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

    def run_full_diagnosis(self, train_loader):
        print(">>> Running Simplified Diagnosis (no SVD, no shuffle)...")

        collapse_score = self.analyze_embeddings_collapse_cosine(sample_size=2048)
        eps = 0.5
        margin_metrics, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
            train_loader, num_batches=10, eps=eps, topk_categories=20
        )
        return {
            "metrics": {
                "embedding_collapse_score": collapse_score,
                "pos_neg_margin_mean": margin_metrics["mean"],
                "pos_neg_margin_p10": margin_metrics["p10"],
                "pos_neg_neg_beats_pos_rate": margin_metrics["neg_beats_pos_rate"],
                "pos_neg_small_margin_rate": margin_metrics["small_margin_rate"],
                "pos_neg_margin_category_breakdown": breakdown,
            },
            "metric_definitions": {
                "embedding_collapse_score": (
                    ### >>> Self_EvolveRec-BLOCK-START: Interpret catalog geometry without inferring recommendation diversity
                    "Mean off-diagonal cosine of sampled catalog embeddings, mapped by (cosine+1)/2. "
                    "PAD and MASK are excluded. Higher values indicate directional similarity; "
                    "0.5 corresponds to mean cosine zero. This does not measure recommendation diversity. "
                    "Null means fewer than two sampled catalog embeddings."
                    ### <<< Self_EvolveRec-BLOCK-END
                ),
                ### >>> Self_EvolveRec-BLOCK-START: Define masked-position diagnostics separately from host ranking metrics
                "pos_neg_margin_mean": (
                    "Mean positive-minus-negative logit at supervised MASK positions with valid, distinct "
                    "catalog positive/negative IDs. Uses one training diagnostic negative per position; "
                    "does not estimate host 100-candidate HR@10. Null means no measured positions."
                ),
                "pos_neg_margin_p10": (
                    "Nearest-rank 10th percentile of measured masked-position margins. "
                    "Null means no measured positions."
                ),
                "pos_neg_neg_beats_pos_rate": (
                    "Fraction of measured masked positions with margin < 0. "
                    "Null means no measured positions."
                ),
                "pos_neg_small_margin_rate": (
                    "Fraction of measured masked positions with margin < eps=0.5. "
                    "This threshold depends on score scale. Null means no measured positions."
                ),
                "pos_neg_margin_category_breakdown": (
                    "Reports measured supervised-mask count, margin threshold and low/wrong-margin counts. "
                    "Category annotations are unavailable; empty category lists are not measured zero counts. "
                    "Reviews, ownership and simulator traits are unobserved."
                ),
                ### <<< Self_EvolveRec-BLOCK-END
            },
        }
