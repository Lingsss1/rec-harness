"""Per-dataset NextItNet baseline for Amazon Reviews.

Trains one independent NextItNet checkpoint per dataset while sharing one default
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


### >>> Self_EvolveRec-BLOCK-START: Restore legacy sampler and optionally retain private row provenance
# DEBUG: sample_batch was missing, so the first training batch raised NameError.
# Keep the diagnostic return contract and expose provenance only to the trainer.
def sample_batch(
    user_train, itemnum, batch_size, maxlen, users=None, return_users=False
):
    if itemnum < 1 or batch_size < 1 or maxlen < 1:
        raise ValueError("itemnum, batch_size and maxlen must be positive")
    if users is None:
        users = [
            u
            for u, h in user_train.items()
            if len(h) >= 2
            and len({int(i) for i in h if 1 <= int(i) <= itemnum}) < itemnum
        ]
    if not users:
        raise ValueError("No eligible users for sequential sampling")
    sequences = np.zeros((batch_size, maxlen), dtype=np.int64)
    positives = np.zeros_like(sequences)
    negatives = np.zeros_like(sequences)
    row_users = []
    for row in range(batch_size):
        uid = users[int(np.random.randint(len(users)))]
        history = user_train[uid]
        forbidden = set(map(int, history))
        if len(history) < 2 or any(i < 1 or i > itemnum for i in forbidden):
            raise ValueError("Sampler requires valid mapped training histories")
        if len(forbidden) >= itemnum:
            raise ValueError("Sampled user has no eligible negative items")
        row_users.append(uid)
        target = int(history[-1])
        position = maxlen - 1
        complement = None
        for item in reversed(history[:-1]):
            sequences[row, position] = int(item)
            positives[row, position] = target
            # Bounded rejection avoids hanging on dense user histories.
            for _ in range(128):
                candidate = int(np.random.randint(1, itemnum + 1))
                if candidate not in forbidden:
                    break
            else:
                if complement is None:
                    complement = np.fromiter(
                        (i for i in range(1, itemnum + 1) if i not in forbidden),
                        dtype=np.int64,
                    )
                candidate = int(complement[int(np.random.randint(len(complement)))])
            negatives[row, position] = candidate
            target = int(item)
            position -= 1
            if position < 0:
                break
    result = (sequences, positives, negatives)
    return result + (row_users,) if return_users else result


# DEBUG: Candidate sampling uses a separate identifier-derived generator;
# it never consumes the sequence sampler's global NumPy RNG stream.
def sample_training_candidates(
    seq,
    pos,
    row_users,
    catalog,
    training_sets,
    dataset_name,
    epoch,
    batch_index,
    count=16,
):
    category_index = DEFAULT_DATASETS.index(dataset_name)
    candidates = np.zeros((*pos.shape, count), dtype=np.int64)
    for row, uid in enumerate(row_users):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [SEED, category_index, int(epoch), int(batch_index), row]
            )
        )
        forbidden = training_sets[uid]
        complement = None
        for position in np.flatnonzero((seq[row] != 0) & (pos[row] != 0)):
            target = int(pos[row, position])
            if target not in forbidden:
                raise ValueError(
                    "Positive target is absent from complete training history"
                )
            accepted, seen = [], set()
            # Acceptance order preserves uniform ordered sampling without replacement.
            proposals = rng.integers(0, len(catalog), size=max(64, 8 * count))
            for index in proposals:
                item = int(catalog[index])
                if item in forbidden or item in seen:
                    continue
                accepted.append(item)
                seen.add(item)
                if len(accepted) == count:
                    break
            if len(accepted) < count:
                # Transient per-row complement; never cache user-by-catalog pools.
                if complement is None:
                    complement = np.fromiter(
                        (int(i) for i in catalog if int(i) not in forbidden),
                        dtype=np.int64,
                    )
                remaining = complement[~np.isin(complement, accepted)]
                take = min(count - len(accepted), len(remaining))
                if take:
                    accepted.extend(
                        map(int, rng.choice(remaining, take, replace=False))
                    )
            candidates[row, position, : len(accepted)] = accepted
    return candidates


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
    ### >>> Self_EvolveRec-BLOCK-START: Reuse fixed validation candidates without consuming training randomness
    rng = random.Random(42)
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
            ### >>> Self_EvolveRec-BLOCK-START: Deterministic unique validation negatives with explicit feasibility checks
            candidates = [target]
            available = itemnum - sum(1 for item in interacted if 1 <= item <= itemnum)
            if available < 99:
                raise RuntimeError(
                    f"Validation user {user} has only {available} unseen negatives; 99 required"
                )
            while len(candidates) < 100:
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
    pos_scores, neg_scores = model(seq, pos, neg)
    mask = (pos != 0).float()
    ### >>> Self_EvolveRec-BLOCK-START: Numerically stable baseline BPR objective
    loss = nn.functional.softplus(neg_scores - pos_scores)
    ### <<< Self_EvolveRec-BLOCK-END
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# DEBUG: Integrate the proposed CE16 objective through the deployed scoring API.
# The existing BPR helper remains available as the native one-negative reference.
def candidate_sequence_loss(model, seq, pos, candidates):
    features = model.log2feats(seq)
    valid = seq.ne(0) & pos.ne(0)
    eligible = valid & candidates.ne(0).any(dim=-1)
    eligible_count = int(eligible.sum().item())
    negative_counts = candidates.ne(0).sum(dim=-1)
    stats = {
        "valid_positions": int(valid.sum().item()),
        "eligible_positions": eligible_count,
        "skipped_positions": int((valid & ~eligible).sum().item()),
        "shortage_positions": int((eligible & negative_counts.lt(16)).sum().item()),
        "negative_slots": int(negative_counts[eligible].sum().item()),
    }
    if eligible_count == 0:
        return None, stats
    selected = features[eligible]
    targets = pos[eligible]
    negatives = candidates[eligible]
    loss_sum = selected.sum() * 0.0
    # Sum chunk losses before dividing by the global position count; retain one
    # encoder/dropout realization and one optimizer step for the original batch.
    for start in range(0, eligible_count, 1024):
        stop = start + 1024
        f = selected[start:stop]
        ids = negatives[start:stop]
        positive_scores = model.predict_score(f, targets[start:stop]).float()
        negative_scores = model.predict_score(f.unsqueeze(1), ids).float()
        negative_scores = negative_scores.masked_fill(ids.eq(0), float("-inf"))
        logits = torch.cat((positive_scores.unsqueeze(1), negative_scores), dim=1)
        labels = torch.zeros(len(f), dtype=torch.long, device=logits.device)
        loss_sum = loss_sum + nn.functional.cross_entropy(
            logits, labels, reduction="sum"
        )
    return loss_sum / eligible_count, stats


# Hyper-parameters
MAXLEN = int(os.environ.get("NEXTITNET_MAXLEN", "128"))
HIDDEN_UNITS = int(os.environ.get("NEXTITNET_HIDDEN", "64"))
NUM_BLOCKS = int(os.environ.get("NEXTITNET_BLOCKS", "8"))
KERNEL_SIZE = int(os.environ.get("NEXTITNET_KERNEL", "3"))
DROPOUT_RATE = float(os.environ.get("NEXTITNET_DROPOUT", "0.2"))
LR = float(os.environ.get("NEXTITNET_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("NEXTITNET_WD", "0.0"))
BATCH_SIZE = int(os.environ.get("NEXTITNET_BATCH", "128"))
### >>> Self_EvolveRec-BLOCK-START: Honor the host epoch default within the fixed training ceiling
NUM_BATCHES = 200
NUM_EPOCHS = min(
    200,
    max(
        1,
        int(os.environ.get("NEXTITNET_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
### <<< Self_EvolveRec-BLOCK-END
EVAL_EVERY = int(os.environ.get("NEXTITNET_EVAL_EVERY", "5"))
PATIENCE = int(os.environ.get("NEXTITNET_PATIENCE", "20"))


class NextItNetBlock(nn.Module):
    def __init__(
        self, hidden_units: int, dilation: int, kernel_size: int, dropout_rate: float
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.conv2 = nn.Conv1d(
            hidden_units, hidden_units, kernel_size, dilation=dilation, padding=0
        )
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))
        return conv(x)

    ### >>> Self_EvolveRec-BLOCK-START: Prevent padded activations from entering causal convolutions
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x * mask
        y = (self.norm1(residual) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv1, y).transpose(1, 2) * mask
        y = self.dropout(self.activation(y)) * mask
        y = (self.norm2(y) * mask).transpose(1, 2)
        y = self._causal_conv(self.conv2, y).transpose(1, 2) * mask
        y = self.dropout(y) * mask
        return (residual + y) * mask

    ### <<< Self_EvolveRec-BLOCK-END


class NextItNetModel(nn.Module):
    def __init__(
        self,
        itemnum: int,
        hidden_units: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        maxlen: int = 128,
    ):
        super().__init__()
        self.itemnum = itemnum
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(itemnum + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=0)
        dilations = [1, 2, 4, 8] * max(1, math.ceil(num_blocks / 4))
        self.blocks = nn.ModuleList(
            [
                NextItNetBlock(hidden_units, dilations[i], kernel_size, dropout_rate)
                for i in range(num_blocks)
            ]
        )
        ### >>> Self_EvolveRec-BLOCK-START: Small gated depth residual with causal prefix conditioning
        if num_blocks < 1:
            raise ValueError("NextItNet requires at least one residual block")
        self.depth_norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_units, eps=1e-5, elementwise_affine=False)
                for _ in range(num_blocks)
            ]
        )
        self.prefix_norm = nn.LayerNorm(
            hidden_units, eps=1e-5, elementwise_affine=False
        )
        self.depth_gate = nn.Sequential(
            nn.Linear(2 * hidden_units + 1, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, num_blocks),
        )
        nn.init.zeros_(self.depth_gate[-1].weight)
        nn.init.zeros_(self.depth_gate[-1].bias)
        self.depth_scale = nn.Parameter(torch.tensor(math.atanh(0.1)))
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        ### <<< Self_EvolveRec-BLOCK-END
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    ### >>> Self_EvolveRec-BLOCK-START: Padding-invariant causal depth gate preserving the final-block path
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        valid = seq != 0
        mask = valid.unsqueeze(-1).to(self.item_emb.weight.dtype)
        # Observed-prefix positions preserve causality and do not depend on
        # the number of padding tokens preceding the retained history.
        pos = valid.long().cumsum(dim=1).clamp_max(self.maxlen)
        pos = pos * valid.long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x * mask) * mask
        states = []
        for block, depth_norm in zip(self.blocks, self.depth_norms):
            x = block(x, mask) * mask
            states.append(depth_norm(x) * mask)
        values = torch.stack(states, dim=2)
        current = states[-1]
        count = mask.cumsum(dim=1)
        prefix = current.cumsum(dim=1) / count.clamp_min(1.0)
        prefix = self.prefix_norm(prefix) * mask
        gate_input = torch.cat((current, prefix, torch.log1p(count)), dim=-1)
        weights = self.depth_gate(gate_input).softmax(dim=-1)
        aggregate = (weights.unsqueeze(-1) * values).sum(dim=2)
        # Keep the original final residual state as the main scoring path;
        # depth aggregation begins as a small, uniformly weighted correction.
        features = x + self.depth_scale.tanh() * aggregate
        return self.norm(features) * mask

    def log2feats(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(sequence)

    def predict_score(
        self, features: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        return (features * self.item_emb(items)).sum(dim=-1)

    ### <<< Self_EvolveRec-BLOCK-END

    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        out = self.encode_sequence(seq)
        ### >>> Self_EvolveRec-BLOCK-START: Share trained item scoring with diagnostic compatibility methods
        pos_scores = self.predict_score(out, pos)
        neg_scores = self.predict_score(out, neg)
        ### <<< Self_EvolveRec-BLOCK-END
        return pos_scores, neg_scores

    ### >>> Self_EvolveRec-BLOCK-START: Score every history at its final nonpadding position
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.size(1) == 0:
            raise ValueError("seq must have shape B x L with L > 0")
        if candidates.ndim not in (1, 2):
            raise ValueError("candidates must have shape N or B x N")
        if candidates.ndim == 2 and candidates.size(0) != seq.size(0):
            raise ValueError("Candidate and history batch sizes must match")
        out = self.encode_sequence(seq)
        positions = torch.arange(seq.size(1), device=seq.device).expand_as(seq)
        last = positions.masked_fill(seq == 0, -1).amax(dim=1)
        rows = torch.arange(seq.size(0), device=seq.device)
        user_emb = out[rows, last.clamp_min(0)]
        user_emb = user_emb * (last >= 0).unsqueeze(-1)
        scores = self.predict_score(user_emb.unsqueeze(1), candidates)
        # Preserve the deployed single-history list-of-scores contract.
        return scores[0] if seq.size(0) == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Reset seed 42 independently for each category
def train_one_dataset(dataset_name: str, device: torch.device, data_dir: str) -> float:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    user_train, user_valid, itemnum = load_dataset(dataset_name, data_dir)
    ### <<< Self_EvolveRec-BLOCK-END
    print(f"  [{dataset_name}] users={len(user_train)}  items={itemnum}", flush=True)
    if itemnum <= 0 or not user_train:
        raise RuntimeError(f"empty dataset or item vocabulary for {dataset_name}")

    model = NextItNetModel(
        itemnum=itemnum,
        hidden_units=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
        maxlen=MAXLEN,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Cache eligible sampler users without changing sampling probabilities
    eligible_users = [
        user
        for user, history in user_train.items()
        if len(history) >= 2
        and len({item for item in history if 1 <= item <= itemnum}) < itemnum
    ]
    if not eligible_users:
        raise RuntimeError(f"No eligible training users for {dataset_name}")
    # DEBUG: Obtain complete-user exclusions inside sampling rather than trying
    # to recover user identity from the diagnostic loader's None field.
    training_sets = {u: set(map(int, h)) for u, h in user_train.items()}
    catalog = np.asarray(sorted(set().union(*training_sets.values())), dtype=np.int64)
    if catalog.size == 0 or np.any((catalog < 1) | (catalog > itemnum)):
        raise ValueError("Training catalog contains invalid mapped item IDs")
    training_settings = {
        "objective": "uncorrected_candidate_cross_entropy",
        "negative_count": 16,
        "temperature": 1.0,
        "reduction": "mean over eligible nonpadding causal positions",
        "negative_catalog": "training-observed mapped items only",
        "exclusions": "complete permitted user training history; padding",
        "replacement": False,
        "shortages": "zero-pad and mask; skip empty pools",
        "negative_rng_key": [
            42,
            DEFAULT_DATASETS.index(dataset_name),
            "epoch",
            "batch_index",
            "row_occurrence",
        ],
        "seed": SEED,
        "datasets": DATASETS,
        "max_epochs": NUM_EPOCHS,
        "batches_per_epoch": NUM_BATCHES,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": 5.0,
        "eval_every": EVAL_EVERY,
        "patience": PATIENCE,
        "selection": "strict improvement in validation HR@10",
        "validation_ties": "candidate order; target first",
        "architecture": "nextitnet_causal_prefix_depth_residual_v2",
        "hidden_units": HIDDEN_UNITS,
        "num_blocks": NUM_BLOCKS,
        "kernel_size": KERNEL_SIZE,
        "maxlen": MAXLEN,
        "dropout_rate": DROPOUT_RATE,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "comparison_scope": "CE16 configuration only; no isolated objective-effect claim",
    }
    with open(os.path.join(working_dir, f"{dataset_name}_training.json"), "w") as f:
        json.dump(training_settings, f, indent=2)
    sampling_totals = {
        key: 0
        for key in (
            "valid_positions",
            "eligible_positions",
            "skipped_positions",
            "shortage_positions",
            "negative_slots",
            "optimizer_steps",
            "skipped_optimizer_steps",
        )
    }
    best_hr10 = -1.0
    best_epoch = 0
    ### <<< Self_EvolveRec-BLOCK-END
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_index in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Execute exclusion-aware CE16 training
            seq, pos, neg, row_users = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                users=eligible_users,
                return_users=True,
            )
            candidate_ids = sample_training_candidates(
                seq,
                pos,
                row_users,
                catalog,
                training_sets,
                dataset_name,
                epoch,
                batch_index,
                count=16,
            )
            seq_t = torch.as_tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=device)
            candidate_t = torch.as_tensor(
                candidate_ids, dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss, batch_stats = candidate_sequence_loss(
                model, seq_t, pos_t, candidate_t
            )
            for key, value in batch_stats.items():
                sampling_totals[key] += value
            if batch_stats["eligible_positions"] == 0:
                sampling_totals["skipped_optimizer_steps"] += 1
                continue
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite CE16 loss for {dataset_name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sampling_totals["optimizer_steps"] += 1
            total_loss += float(loss.item())
            ### <<< Self_EvolveRec-BLOCK-END

        avg_loss = total_loss / max(NUM_BATCHES, 1)
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
            if epoch - best_epoch >= PATIENCE:
                print(f"  [{dataset_name}] Early stopping at epoch {epoch}", flush=True)
                break
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)

    # DEBUG: Persist actual sampling shortages and update counts, including skips.
    training_settings["sampling_totals"] = sampling_totals
    training_settings["best_epoch"] = best_epoch
    training_settings["epochs_executed"] = epoch
    with open(os.path.join(working_dir, f"{dataset_name}_training.json"), "w") as f:
        json.dump(training_settings, f, indent=2)
    meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "itemnum": itemnum,
                "maxlen": MAXLEN,
                "hidden_units": HIDDEN_UNITS,
                "num_blocks": NUM_BLOCKS,
                "kernel_size": KERNEL_SIZE,
                ### >>> Self_EvolveRec-BLOCK-START: Persist gated residual architecture and prefix configuration
                "dropout_rate": DROPOUT_RATE,
                "architecture": "nextitnet_causal_prefix_depth_residual_v2",
                "prefix_conditioning": True,
                "position_encoding": "observed_prefix_count",
                "depth_value_norm_affine": False,
                ### <<< Self_EvolveRec-BLOCK-END
            },
            f,
        )
    print(
        f"  [{dataset_name}] Best val HR@10={max(best_hr10, 0.0):.4f}  ckpt={ckpt_path}",
        flush=True,
    )
    return max(best_hr10, 0.0)


def write_predict_script(working_dir: str) -> None:
    script = r'''"""Per-dataset NextItNet predict function."""
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
        f"nextitnet_perdataset_train_{os.getpid()}", os.path.join(_HERE, "..", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NextItNetModel


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
        ### >>> Self_EvolveRec-BLOCK-START: Require the trained checkpoint for every routed category
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing NextItNet metadata or checkpoint for {ds}: "
                f"{meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        ### >>> Self_EvolveRec-BLOCK-START: Match the single checkpoint to the causal prefix depth residual
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if (
            meta.get("architecture") != "nextitnet_causal_prefix_depth_residual_v2"
            or meta.get("prefix_conditioning") is not True
            or meta.get("position_encoding") != "observed_prefix_count"
            or meta.get("depth_value_norm_affine") is not False
        ):
            raise RuntimeError(f"Incompatible NextItNet architecture for {ds}")
        # Import the trained class; strict loading restores the backbone,
        # prefix gate, residual coefficient and shared item-scoring head.
        ### <<< Self_EvolveRec-BLOCK-END
        model = ModelClass(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            kernel_size=meta["kernel_size"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Reject ambiguous routing instead of scoring with another category
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple categories are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"No trained NextItNet model for {dataset_name!r}")
    model = _models[dataset_name]
    meta = _metas[dataset_name]
    ### <<< Self_EvolveRec-BLOCK-END

    maxlen = int(meta.get("maxlen", 128))
    seq_arr = [0] * maxlen
    for i, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - i] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
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
    print(f"Per-dataset NextItNet training on {device}", flush=True)
    scores = []
    for dataset_name in DATASETS:
        scores.append(train_one_dataset(dataset_name, device, DATA_DIR))
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    write_predict_script(working_dir)
    avg = sum(scores) / max(len(scores), 1)
    print(f"\nAll datasets done. Average val HR@10 = {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()


