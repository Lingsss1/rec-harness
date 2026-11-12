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


### >>> Self_EvolveRec-BLOCK-START: Preserve paired branch sampling with an isolated deterministic RNG
def sample_batch(
    user_train: dict[int, list[int]],
    itemnum: int,
    batch_size: int,
    maxlen: int,
    *,
    return_branch_types: bool = False,
    rng=None,
    negative_count: int = 1,
    negative_rng=None,
    return_eligible_counts: bool = False,
    eligible_users=None,
):
    import random as random_module

    random = random_module if rng is None else rng
    if negative_count < 1:
        raise ValueError("negative_count must be positive")
    if negative_count > 1 and negative_rng is None:
        raise ValueError("Multi-negative training requires an isolated negative RNG")
    extra_random = random if negative_rng is None else negative_rng
    batch_eligible = []
    if batch_size < 2 or batch_size % 2:
        raise ValueError("balanced branches require a positive even batch size")
    if maxlen < 2:
        raise ValueError("terminal training requires space for context and mask")
    # Training supplies a cached pool; public diagnostic callers retain the
    # original default and B x L single-negative return contract.
    users = (
        [u for u, items in user_train.items() if len(items) >= 2]
        if eligible_users is None
        else eligible_users
    )
    if len(users) == 0:
        raise RuntimeError("no users with at least two training interactions")

    terminal_rows = [True] * (batch_size // 2) + [False] * (batch_size // 2)
    random.shuffle(terminal_rows)
    batch_seq, batch_pos, batch_neg = [], [], []
    for terminal in terminal_rows:
        history = user_train[random.choice(users)]
        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(
            (maxlen,) if negative_count == 1 else (maxlen, negative_count),
            dtype=np.int64,
        )
        if terminal:
            # Uniform user, then uniform endpoint 1..n-1, including the last item.
            endpoint = random.randrange(1, len(history))
            context = history[max(0, endpoint - maxlen + 1) : endpoint]
            seq[-len(context) - 1 : -1] = context
            seq[-1] = history[endpoint]
            selected = [maxlen - 1]
        else:
            # Preserve final-L windows, Bernoulli masking, fallback and last-item boost.
            items = history[-maxlen:]
            seq[-len(items) :] = items
            eligible = list(range(maxlen - len(items), maxlen))
            selected = [i for i in eligible if random.random() < 0.2]
            if not selected:
                selected = [random.choice(eligible)]
            if random.random() < 0.5 and maxlen - 1 not in selected:
                selected.append(maxlen - 1)
        interacted = set(history)
        if any(item < 1 or item > itemnum for item in interacted):
            raise ValueError("Training history contains a reserved or unknown item")
        eligible_count = itemnum - len(interacted)
        if eligible_count <= 0:
            raise RuntimeError("no unobserved negative item available")
        batch_eligible.append(eligible_count)
        for i in selected:
            pos[i] = seq[i]
            seq[i] = itemnum + 1
            # Preserve the original sampler stream for the first draw. Additional
            # draws use an independent stream and never change target sampling.
            item = random.randint(1, itemnum)
            while item in interacted:
                item = random.randint(1, itemnum)
            if negative_count == 1:
                neg[i] = item
            else:
                neg[i, 0] = item
                for j in range(1, negative_count):
                    item = extra_random.randint(1, itemnum)
                    while item in interacted:
                        item = extra_random.randint(1, itemnum)
                    # Sampling WITH replacement: duplicates remain loss terms.
                    neg[i, j] = item
        if not np.any(pos != 0):
            raise RuntimeError("every example must have a supervised target")
        batch_seq.append(seq)
        batch_pos.append(pos)
        batch_neg.append(neg)
    result = (np.asarray(batch_seq), np.asarray(batch_pos), np.asarray(batch_neg))
    if return_branch_types:
        result = (*result, np.asarray(terminal_rows, dtype=np.bool_))
    if return_eligible_counts:
        result = (*result, np.asarray(batch_eligible, dtype=np.int64))
    return result


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
            ### >>> Self_EvolveRec-BLOCK-START: Fixed unique validation negatives with explicit catalog eligibility checks
            candidates = [target]
            excluded_real = sum(1 for item in interacted if 1 <= item <= itemnum)
            if itemnum - excluded_real < 99:
                raise RuntimeError(
                    f"validation user {user} has fewer than 99 eligible negatives"
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


### >>> Self_EvolveRec-BLOCK-START: Matched sixteen-negative BCE and gBCE with inherited branch reductions
NEGATIVE_COUNT = 16
LOSS_ARM = os.environ.get("BERT4REC_LOSS_ARM", "gbce")
if LOSS_ARM not in {"bce", "gbce"}:
    raise ValueError("BERT4REC_LOSS_ARM must be bce or gbce")
CALIBRATION = 0.75 if LOSS_ARM == "gbce" else 0.0


def bpr_sequence_loss(
    model: nn.Module,
    seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    terminal_rows=None,
    terminal_weight: float = 0.5,
    *,
    eligible_counts=None,
    calibration=None,
) -> torch.Tensor:
    """Candidate-averaged BCE/gBCE; legacy name preserves host compatibility.

    Public diagnostic calls may still supply B x L negatives. Calibration
    defaults to ordinary BCE unless explicitly supplied by training.
    """
    if not 0.0 <= terminal_weight <= 1.0:
        raise ValueError("terminal_weight must lie in [0, 1]")
    if seq.ndim != 2 or seq.shape != pos.shape:
        raise ValueError("Expected matching B x L sequences and positives")
    if neg.ndim == 2:
        neg = neg.unsqueeze(-1)
    if neg.ndim != 3 or neg.shape[:2] != pos.shape or neg.shape[-1] < 1:
        raise ValueError("Expected B x L or B x L x K negatives")
    valid = pos.ne(0)
    count = valid.sum(dim=1)
    if torch.any(count == 0):
        raise ValueError("every example must have a supervised target")
    if torch.any(seq[valid] != model.mask_token):
        raise ValueError("all supervised input positions must contain MASK")
    positive_ids, negative_ids = pos[valid], neg[valid]
    if torch.any((positive_ids < 1) | (positive_ids > model.itemnum)):
        raise ValueError("Positive IDs must be real catalog items")
    if torch.any((negative_ids < 1) | (negative_ids > model.itemnum)):
        raise ValueError("Negative IDs must be real catalog items")
    if torch.any(negative_ids == positive_ids[:, None]):
        raise ValueError("A positive cannot also be a sampled negative")
    t = 0.0 if calibration is None else float(calibration)
    if not 0.0 <= t <= 1.0:
        raise ValueError("calibration must lie in [0, 1]")
    k = neg.shape[-1]
    if eligible_counts is None:
        if t != 0.0:
            raise ValueError("gBCE requires measured eligible catalog counts")
        beta = 1.0
    else:
        n = torch.as_tensor(eligible_counts, device=pos.device, dtype=torch.float32)
        if n.shape != (pos.shape[0],):
            raise ValueError("eligible_counts must contain one count per example")
        if not bool((torch.isfinite(n) & (n >= 1) & (n == n.floor())).all()):
            raise ValueError("Eligible counts must be positive integers")
        n = n[:, None].expand_as(pos)[valid]
        # K/N is expected multiplicity under sampling with replacement.
        # With history exclusions this is a weighting heuristic, not a
        # demonstrated full-catalog probability calibration guarantee.
        beta = 1.0 - t + t * (k / n)
    # Encode once and embed negatives only at supervised positions. This avoids
    # allocating a B x L x K x D tensor for mostly unsupervised PAD positions.
    features = model.encode_sequence(seq)[valid].float()
    pos_scores = (features * model.item_emb(positive_ids).float()).sum(-1)
    neg_scores = torch.einsum(
        "td,tkd->tk", features, model.item_emb(negative_ids).float()
    )
    losses = (
        beta * torch.nn.functional.softplus(-pos_scores)
        + torch.nn.functional.softplus(neg_scores).sum(-1)
    ) / (k + 1)
    # Every repeated negative contributes separately; no deduplication.
    # Unique masked destinations avoid CUDA scatter_add atomic accumulation,
    # whose summation order can vary between otherwise paired loss arms.
    target_losses = losses.new_zeros(pos.shape).masked_scatter(valid, losses)
    per_example = target_losses.sum(dim=1) / count
    if terminal_rows is None:
        return per_example.mean()
    terminal_rows = torch.as_tensor(terminal_rows, device=pos.device, dtype=torch.bool)
    if terminal_rows.shape != (pos.shape[0],):
        raise ValueError("branch metadata must have shape B")
    n_terminal = int(terminal_rows.sum().item())
    if n_terminal == 0 or 2 * n_terminal != pos.shape[0]:
        raise ValueError("require equal nonempty terminal and interior branches")
    return (
        terminal_weight * per_example[terminal_rows].mean()
        + (1.0 - terminal_weight) * per_example[~terminal_rows].mean()
    )


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
### >>> Self_EvolveRec-BLOCK-START: Fixed 100-epoch curriculum with independently generated exposure controls
NUM_BATCHES = 200
NUM_EPOCHS = max(
    1,
    min(
        200,
        int(os.environ.get("BERT4REC_EPOCHS", os.environ.get("SASREC_EPOCHS", "200"))),
    ),
)
ALPHA_SCHEDULE = tuple(0.5 + 0.3 * min(e / 100.0, 1.0) for e in range(NUM_EPOCHS))
ALPHA_BAR = math.fsum(ALPHA_SCHEDULE) / NUM_EPOCHS
# This local generator consumes neither sampler nor model/dropout randomness.
_ALPHA_PERMUTATION = np.random.default_rng(SEED).permutation(NUM_EPOCHS)
OBJECTIVE_ARM = os.environ.get("BERT4REC_OBJECTIVE_ARM", "schedule")
CONSTANT_WEIGHTS = {
    "equal": 0.5,
    "constant_0.5": 0.5,
    "terminal_heavy": 0.8,
    "constant_0.8": 0.8,
    "matched_mean": ALPHA_BAR,
    "exposure_matched_constant": ALPHA_BAR,
    "terminal_only": 1.0,
}
SCHEDULE_WEIGHTS = {
    "schedule": ALPHA_SCHEDULE,
    "forward": ALPHA_SCHEDULE,
    "reversed": ALPHA_SCHEDULE[::-1],
    "shuffled_epochs": tuple(ALPHA_SCHEDULE[int(e)] for e in _ALPHA_PERMUTATION),
}
if OBJECTIVE_ARM not in {*SCHEDULE_WEIGHTS, *CONSTANT_WEIGHTS}:
    raise ValueError(f"Unknown BERT4REC_OBJECTIVE_ARM: {OBJECTIVE_ARM!r}")


def terminal_weight_for_epoch(epoch_index: int) -> float:
    if not 0 <= epoch_index < NUM_EPOCHS:
        raise ValueError("epoch_index outside configured training horizon")
    if OBJECTIVE_ARM in SCHEDULE_WEIGHTS:
        return SCHEDULE_WEIGHTS[OBJECTIVE_ARM][epoch_index]
    return CONSTANT_WEIGHTS[OBJECTIVE_ARM]


### <<< Self_EvolveRec-BLOCK-END
### >>> Self_EvolveRec-BLOCK-START: Validate checkpoint cadence under fixed-horizon training
EVAL_EVERY = int(os.environ.get("BERT4REC_EVAL_EVERY", "5"))
if EVAL_EVERY < 1:
    raise ValueError("BERT4REC_EVAL_EVERY must be positive")
# Fixed-horizon experiments do not use patience-based stopping.
### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Vectorize matched-count prefix dropout while preserving paired random draws
PREFIX_DROPOUT_ARM = os.environ.get("BERT4REC_PREFIX_DROPOUT_ARM", "protected")
if PREFIX_DROPOUT_ARM not in {"none", "uniform", "protected"}:
    raise ValueError("BERT4REC_PREFIX_DROPOUT_ARM must be none, uniform, or protected")
PREFIX_DROPOUT_RATE = 0.15
if OBJECTIVE_ARM != "schedule":
    raise ValueError(
        "Prefix-dropout comparisons require the unchanged schedule objective"
    )


def protected_recent_dropout(sequence, mask_id, generator, mode):
    """CPU augmentation with fixed slots and an isolated random stream."""
    if (
        sequence.device.type != "cpu"
        or sequence.ndim != 2
        or sequence.shape[0] == 0
        or sequence.shape[1] < 2
        or sequence.dtype != torch.long
    ):
        raise ValueError("Expected nonempty CPU B x L int64 terminal sequences, L >= 2")
    if generator is None or torch.device(generator.device).type != "cpu":
        raise ValueError("A dedicated CPU augmentation generator is required")
    if mode not in {"none", "uniform", "protected"}:
        raise ValueError("Invalid prefix dropout arm")
    if mask_id <= 1 or not 0.0 <= PREFIX_DROPOUT_RATE <= 1.0:
        raise ValueError("Invalid reserved MASK or deletion probability")
    if not bool(((sequence >= 0) & (sequence <= mask_id)).all()):
        raise ValueError("Invalid token ID")
    if not bool(sequence[:, -1].eq(mask_id).all()):
        raise ValueError("Terminal inputs must end in MASK")
    if not bool(sequence.eq(mask_id).sum(1).eq(1).all()):
        raise ValueError("Expected exactly one MASK per terminal example")
    visible = sequence.gt(0) & sequence.lt(mask_id)
    if not bool(visible.any(1).all()):
        raise ValueError("Terminal examples require visible history")
    from_right = visible.flip(1).long().cumsum(1).flip(1)
    older = visible & from_right.gt(2)
    # Preserve both draw shapes and their order, including in the none arm.
    count_draws = torch.rand(sequence.shape, generator=generator, device="cpu")
    priorities = torch.rand(sequence.shape, generator=generator, device="cpu")
    counts = (older & count_draws.lt(PREFIX_DROPOUT_RATE)).sum(1)
    deleted = torch.zeros_like(visible)
    if mode != "none":
        eligible = older if mode == "protected" else visible
        if bool((counts > eligible.sum(1)).any()):
            raise RuntimeError("Deletion count exceeds eligible history")
        # Finite eligible priorities precede all ineligible slots. Stable sorting
        # retains the original ascending-position tie break used by each row.
        order = torch.argsort(
            priorities.masked_fill(~eligible, float("inf")), dim=1, stable=True
        )
        take = torch.arange(sequence.shape[1], device="cpu")[None, :] < counts[:, None]
        deleted.scatter_(1, order, take)
        if not torch.equal(deleted.sum(1), counts):
            raise RuntimeError("Matched deletion counts violated")
    if bool((deleted & ~visible).any()):
        raise RuntimeError("Deletion touched a PAD or MASK position")
    if mode == "protected" and bool((deleted & from_right.le(2)).any()):
        raise RuntimeError("Recent history protection violated")
    augmented = sequence.masked_fill(deleted, 0)
    if not torch.equal(augmented[:, -1], sequence[:, -1]):
        raise RuntimeError("Augmentation changed the terminal MASK")
    return augmented, deleted, visible.sum(1), older.sum(1), counts


### <<< Self_EvolveRec-BLOCK-END


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
        ### >>> Self_EvolveRec-BLOCK-START: Preserve dense absolute slots and attention masking across internal PAD holes
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks, enable_nested_tensor=False
        )
        ### <<< Self_EvolveRec-BLOCK-END
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_units)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_emb.weight)
            self.item_emb.weight[0].zero_()
            nn.init.xavier_uniform_(self.pos_emb.weight)
            self.pos_emb.weight[0].zero_()

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        batch, length = seq.shape
        pos = torch.arange(length, device=seq.device).unsqueeze(0).expand(batch, -1) + 1
        pos = pos * (seq != 0).long()
        x = self.item_emb(seq) * (self.hidden_units**0.5) + self.pos_emb(pos)
        x = self.dropout(x)
        padding = seq == 0
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), -1] = False
        x = self.encoder(x, src_key_padding_mask=safe_padding)
        return self.norm(x)

    ### >>> Self_EvolveRec-BLOCK-START: Support both public single-negative and internal multi-negative forward calls
    def forward(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        if seq.ndim != 2 or pos.shape != seq.shape:
            raise ValueError("Expected matching B x L sequences and positives")
        if neg.ndim == 2 and neg.shape == pos.shape:
            multiple_negatives = False
        elif neg.ndim == 3 and neg.shape[:2] == pos.shape and neg.shape[-1] > 0:
            multiple_negatives = True
        else:
            raise ValueError("Expected B x L or B x L x K negatives")
        out = self.encode_sequence(seq)
        pos_scores = (out * self.item_emb(pos)).sum(dim=-1)
        if multiple_negatives:
            neg_scores = torch.einsum("bld,blkd->blk", out, self.item_emb(neg))
        else:
            neg_scores = (out * self.item_emb(neg)).sum(dim=-1)
        return pos_scores, neg_scores

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Preserve appended-mask inference while supporting batched candidate scoring
    def score_candidates(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if seq.ndim != 2 or seq.shape[1] != self.maxlen:
            raise ValueError("seq must have shape B x configured maxlen")
        if candidates.ndim not in (1, 2):
            raise ValueError("candidates must have shape C or B x C")
        if candidates.ndim == 2 and candidates.shape[0] != seq.shape[0]:
            raise ValueError("candidate and sequence batch sizes must match")
        prediction_seq = torch.zeros_like(seq)
        prediction_seq[:, :-1] = seq[:, 1:]
        prediction_seq[:, -1] = self.mask_token
        # The appended MASK is the final nonpadding position under left padding.
        user_emb = self.encode_sequence(prediction_seq)[:, -1, :]
        cand_emb = self.item_emb(candidates)
        if candidates.ndim == 1:
            scores = user_emb @ cand_emb.transpose(0, 1)
        else:
            scores = torch.einsum("bd,bcd->bc", user_emb, cand_emb)
        # Preserve the existing singleton evaluator and predictor contract.
        return scores[0] if seq.shape[0] == 1 else scores

    ### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Share one canonical architecture factory across training and serving
def build_model(config: dict) -> BERT4RecModel:
    return BERT4RecModel(
        itemnum=int(config["itemnum"]),
        hidden_units=int(config["hidden_units"]),
        num_blocks=int(config["num_blocks"]),
        num_heads=int(config["num_heads"]),
        dropout_rate=float(config["dropout_rate"]),
        maxlen=int(config["maxlen"]),
    )


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Initialize every independent category model with training seed 42
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

    ### >>> Self_EvolveRec-BLOCK-START: Construct the trained model through the canonical factory
    model = build_model(
        {
            "itemnum": itemnum,
            "hidden_units": HIDDEN_UNITS,
            "num_blocks": NUM_BLOCKS,
            "num_heads": NUM_HEADS,
            "dropout_rate": DROPOUT_RATE,
            "maxlen": MAXLEN,
        }
    ).to(device)
    ### <<< Self_EvolveRec-BLOCK-END
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    working_dir = os.path.join(os.path.dirname(__file__), "working")
    os.makedirs(working_dir, exist_ok=True)
    ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")

    ### >>> Self_EvolveRec-BLOCK-START: Record paired sampling and isolated terminal augmentation randomness
    import time

    training_started = time.perf_counter()
    sampler_rng = random.Random(SEED)
    # Preserve dictionary order and sampling probabilities while avoiding a
    # complete scan of every training user on each of 40,000 updates.
    eligible_users = tuple(
        user for user, history in user_train.items() if len(history) >= 2
    )
    if not eligible_users:
        raise RuntimeError(f"[{dataset_name}] no eligible training users")
    negative_rng = random.Random(SEED + 1009)
    augmentation_rng = torch.Generator(device="cpu").manual_seed(SEED)
    loss_stats = {
        "arm": LOSS_ARM,
        "calibration_t": CALIBRATION,
        "negative_count": NEGATIVE_COUNT,
        "replacement": True,
        "exclusions": "All real items in the sampled user's training history only",
        "normalization": "(beta*softplus(-positive)+sum softplus(negative))/(K+1), then mean targets/example and examples/branch",
        "eligible_count_target_histogram": {},
        "supervised_targets": 0,
        "negative_draws": 0,
        "duplicate_draws_beyond_first_occurrence": 0,
        "beta_sum": 0.0,
        "calibration_claim": "History-dependent K/N weighting heuristic; full-catalog calibration is unestablished",
    }
    augmentation_stats = {
        "arm": PREFIX_DROPOUT_ARM,
        "probability": PREFIX_DROPOUT_RATE,
        "protected_recent_items": 2,
        "terminal_examples": 0,
        "visible_items": 0,
        "older_eligible_items": 0,
        "proposed_deletions": 0,
        "actual_deletions": 0,
        "unmodified_examples": 0,
        "n_q_k_removed_histogram": {},
        "removed_absolute_position_counts": [0] * MAXLEN,
        "trace_format": "Per update: row-major B/2 x ceil(MAXLEN/8) uint8 bytes, np.packbits along positions with bitorder=little; trailing bits are zero; terminal rows retain sampled batch order.",
    }
    deletion_trace_path = os.path.join(
        working_dir, f"{dataset_name}_prefix_deletions.bin"
    )
    # Truncate trial-local diagnostic data; never reuse another run's artifacts.
    with open(deletion_trace_path, "wb"):
        pass
    print(
        f"  [{dataset_name}] prefix_dropout={PREFIX_DROPOUT_ARM}"
        f" p={PREFIX_DROPOUT_RATE} protected_recent=2",
        flush=True,
    )
    eligible_lengths = [len(items) for items in user_train.values() if len(items) >= 2]
    print(
        f"  [{dataset_name}] arm={OBJECTIVE_ARM} alpha_bar={ALPHA_BAR:.8f}"
        f" eligible_users={len(eligible_lengths)}"
        f" eligible_endpoints={sum(n - 1 for n in eligible_lengths)}"
        f" sampler=uniform_user_then_uniform_endpoint"
        f" interior=final_L_bernoulli_0.2_fallback_final_boost_0.5",
        flush=True,
    )
    best_hr10 = -1.0
    best_epoch = 0
    completed_updates = 0
    cumulative_targets = [0, 0]
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        alpha = terminal_weight_for_epoch(epoch - 1)
        print(
            f"  [{dataset_name}] epoch={epoch} terminal_weight={alpha:.8f}", flush=True
        )
        ### <<< Self_EvolveRec-BLOCK-END
        for _ in range(NUM_BATCHES):
            ### >>> Self_EvolveRec-BLOCK-START: Apply matched-count prefix dropout only after terminal target masking
            seq, pos, neg, terminal_rows, eligible_counts = sample_batch(
                user_train,
                itemnum,
                BATCH_SIZE,
                MAXLEN,
                return_branch_types=True,
                rng=sampler_rng,
                negative_count=NEGATIVE_COUNT,
                negative_rng=negative_rng,
                return_eligible_counts=True,
                eligible_users=eligible_users,
            )
            target_counts = np.count_nonzero(pos, axis=1)
            target_total = int(target_counts.sum())
            loss_stats["supervised_targets"] += target_total
            loss_stats["negative_draws"] += target_total * NEGATIVE_COUNT
            sorted_negatives = np.sort(neg[pos != 0], axis=-1)
            loss_stats["duplicate_draws_beyond_first_occurrence"] += int(
                np.count_nonzero(sorted_negatives[:, 1:] == sorted_negatives[:, :-1])
            )
            beta_per_row = (
                1.0 - CALIBRATION + CALIBRATION * NEGATIVE_COUNT / eligible_counts
            )
            loss_stats["beta_sum"] += float(np.dot(beta_per_row, target_counts))
            eligible_histogram = loss_stats["eligible_count_target_histogram"]
            for eligible_n, targets_n in zip(eligible_counts, target_counts):
                key = str(int(eligible_n))
                eligible_histogram[key] = eligible_histogram.get(key, 0) + int(
                    targets_n
                )
            cumulative_targets[0] += int(np.count_nonzero(pos[terminal_rows]))
            cumulative_targets[1] += int(np.count_nonzero(pos[~terminal_rows]))
            terminal_cpu = torch.from_numpy(seq[terminal_rows].copy())
            augmented, deleted, n, q, k = protected_recent_dropout(
                terminal_cpu, model.mask_token, augmentation_rng, PREFIX_DROPOUT_ARM
            )
            # No compaction: encode_sequence uses fixed arange positions for all
            # surviving tokens, and excludes every PAD hole from attention keys.
            seq[terminal_rows] = augmented.numpy()
            removed = deleted.sum(1)
            augmentation_stats["terminal_examples"] += int(n.numel())
            augmentation_stats["visible_items"] += int(n.sum())
            augmentation_stats["older_eligible_items"] += int(q.sum())
            augmentation_stats["proposed_deletions"] += int(k.sum())
            augmentation_stats["actual_deletions"] += int(removed.sum())
            augmentation_stats["unmodified_examples"] += int(removed.eq(0).sum())
            histogram = augmentation_stats["n_q_k_removed_histogram"]
            for values in zip(n.tolist(), q.tolist(), k.tolist(), removed.tolist()):
                key = ",".join(map(str, values))
                histogram[key] = histogram.get(key, 0) + 1
            position_counts = augmentation_stats["removed_absolute_position_counts"]
            for index, count in enumerate(deleted.sum(0).tolist()):
                position_counts[index] += count
            with open(deletion_trace_path, "ab") as trace:
                trace.write(
                    np.packbits(deleted.numpy(), axis=1, bitorder="little").tobytes()
                )
            seq_t = torch.tensor(seq, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg, dtype=torch.long, device=device)
            loss = bpr_sequence_loss(
                model,
                seq_t,
                pos_t,
                neg_t,
                terminal_rows,
                terminal_weight=alpha,
                eligible_counts=eligible_counts,
                calibration=CALIBRATION,
            )
            completed_updates += 1
            ### <<< Self_EvolveRec-BLOCK-END
            ### >>> Self_EvolveRec-BLOCK-START: Fail explicitly on numerical divergence and reuse the measured scalar loss
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"[{dataset_name}] nonfinite training loss at epoch {epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += loss_value
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
            ### >>> Self_EvolveRec-BLOCK-START: Complete the prespecified horizon for comparable curriculum exposure
            # Keep strict-improvement checkpoint selection and validation cadence.
            # Every arm executes the same configured number of optimizer updates.
            ### <<< Self_EvolveRec-BLOCK-END
        elif epoch % 10 == 0:
            print(
                f"  [{dataset_name}] epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}",
                flush=True,
            )

    ### >>> Self_EvolveRec-BLOCK-START: Record completed objective exposure and measured prefix deletion statistics
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)
    completed_weights = [terminal_weight_for_epoch(e) for e in range(epoch)]
    terminal_exposure = NUM_BATCHES * math.fsum(completed_weights)
    interior_exposure = NUM_BATCHES * math.fsum(1.0 - a for a in completed_weights)
    # Adam uses the unchanged constant LR throughout this training loop.
    augmentation_stats["realized_visible_deletion_rate"] = augmentation_stats[
        "actual_deletions"
    ] / max(1, augmentation_stats["visible_items"])
    augmentation_stats["unmodified_example_fraction"] = augmentation_stats[
        "unmodified_examples"
    ] / max(1, augmentation_stats["terminal_examples"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loss_stats["training_and_local_validation_wall_seconds"] = (
        time.perf_counter() - training_started
    )
    loss_stats["mean_beta_per_target"] = loss_stats["beta_sum"] / max(
        1, loss_stats["supervised_targets"]
    )
    loss_stats["duplicate_fraction_of_draws"] = loss_stats[
        "duplicate_draws_beyond_first_occurrence"
    ] / max(1, loss_stats["negative_draws"])
    experiment = {
        "sampled_loss": loss_stats,
        "prefix_dropout": augmentation_stats,
        "objective_arm": OBJECTIVE_ARM,
        "configured_epochs": NUM_EPOCHS,
        "completed_epochs": epoch,
        "completed_updates": completed_updates,
        "early_terminated": epoch < NUM_EPOCHS,
        "alpha_bar_configured": ALPHA_BAR,
        "actual_arm_alpha_bar": math.fsum(completed_weights) / len(completed_weights),
        "terminal_weights_completed": completed_weights,
        "schedule_resolution": "epoch; 200 identical coefficients per epoch",
        "ramp_epochs": 100,
        "coefficient_sums_terminal_interior": [terminal_exposure, interior_exposure],
        "lr_weighted_coefficient_sums_terminal_interior": [
            LR * terminal_exposure,
            LR * interior_exposure,
        ],
        "coefficient_exposure_definition": (
            "Sum of branch loss coefficients over completed optimizer updates; "
            "LR-weighted sums use the unchanged constant learning rate. "
            "These quantities do not imply equal gradients or Adam trajectories."
        ),
        "terminal_supervised_targets": cumulative_targets[0],
        "interior_supervised_targets": cumulative_targets[1],
        "branch_examples_per_update": BATCH_SIZE // 2,
        "selected_epoch": best_epoch,
        "selected_local_validation_hr10": best_hr10,
        "last_completed_local_validation_hr10": val_m["HR@10"],
        "final_epoch_local_validation_hr10": (
            val_m["HR@10"] if epoch == NUM_EPOCHS else None
        ),
        "checkpoint_selection": "strict HR@10 improvement; earliest epoch wins ties",
        "checkpoint_eligibility": "configured validation cadence and final epoch",
        "seed": SEED,
        "comparison_status": (
            "Single sampled-loss arm with inherited schedule and prefix dropout. "
            "Run BERT4REC_LOSS_ARM=bce separately for matched K=16 ordinary BCE. "
            "No gBCE advantage is established without independent host evaluation "
            "of both arms; historical BPR scores are not a matched control. "
            "Architecture, candidate scoring and canonical predictor writer are unchanged."
        ),
    }
    with open(os.path.join(working_dir, f"{dataset_name}_experiment.json"), "w") as f:
        json.dump(experiment, f, indent=2)

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
    ### >>> Self_EvolveRec-BLOCK-START: Restore the designated checkpoint so post-training inspection matches serving
    if best_epoch == 0:
        raise RuntimeError(f"[{dataset_name}] no validation checkpoint was selected")
    selected_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(selected_state, strict=True)
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, selected_state[name]):
            raise RuntimeError(
                f"[{dataset_name}] selected checkpoint mismatch for {name}"
            )
    del selected_state
    model.eval()
    print(
        f"  [{dataset_name}] Best val HR@10={best_hr10:.4f}"
        f"  selected_epoch={best_epoch}  ckpt={ckpt_path}",
        flush=True,
    )
    return best_hr10
    ### <<< Self_EvolveRec-BLOCK-END


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
    ### >>> Self_EvolveRec-BLOCK-START: Load the same canonical model factory used during training
    import sys

    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the canonical BERT4Rec training module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build_model
    ### <<< Self_EvolveRec-BLOCK-END


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
        ### >>> Self_EvolveRec-BLOCK-START: Reject missing category artifacts explicitly
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing BERT4Rec artifacts for {ds}: {meta_path}, {ckpt_path}"
            )
        ### <<< Self_EvolveRec-BLOCK-END
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        ### >>> Self_EvolveRec-BLOCK-START: Reconstruct the saved architecture and strictly load its single checkpoint
        model = ModelClass(meta).to(_device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=_device, weights_only=True),
            strict=True,
        )
        ### <<< Self_EvolveRec-BLOCK-END
        model.eval()
        _models[ds] = model
        _metas[ds] = meta


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    ### >>> Self_EvolveRec-BLOCK-START: Route explicitly to exactly one category checkpoint
    if dataset_name is None:
        if len(_models) != 1:
            raise ValueError("dataset_name is required when multiple categories are loaded")
        dataset_name = next(iter(_models))
    if dataset_name not in _models:
        raise ValueError(f"No trained BERT4Rec model for dataset {dataset_name!r}")
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
from collections import Counter

### >>> Self_EvolveRec-BLOCK-START: Transactional conditional optimizer-transfer audit with exact continuation checks
"""Temporary steps of the existing model; no alternative predictor is retained."""
import copy
import math
import random
import time

import numpy as np
import torch


def _equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


class _Transaction:
    def __init__(self, model, optimizer, generators):
        self.model, self.optimizer = model, optimizer
        self.generators = generators
        self.parameters = tuple(model.parameters())
        self.modules = tuple(model.modules())
        self.identities = tuple(id(p) for p in self.parameters)
        self.saved = self.capture()

    def capture(self):
        return copy.deepcopy(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "gradients": [
                    None if p.grad is None else p.grad.detach().clone()
                    for p in self.parameters
                ],
                "modes": [m.training for m in self.modules],
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "explicit": [
                    g.get_state() if isinstance(g, torch.Generator) else g.getstate()
                    for g in self.generators
                ],
            }
        )

    def restore(self):
        s = self.saved
        self.model.load_state_dict(s["model"], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(s["optimizer"]))
        for p, gradient in zip(self.parameters, s["gradients"]):
            p.grad = None if gradient is None else gradient.detach().clone()
        for module, mode in zip(self.modules, s["modes"]):
            module.training = mode
        # Restore randomness last. Neither loading nor verification follows a
        # training iterator, and all cached inputs have requires_grad=False.
        for generator, state in zip(self.generators, s["explicit"]):
            if isinstance(generator, torch.Generator):
                generator.set_state(state)
            else:
                generator.setstate(state)
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.random.set_rng_state(s["cpu"])
        if s["cuda"]:
            torch.cuda.set_rng_state_all(s["cuda"])
        if self.identities != tuple(id(p) for p in self.model.parameters()):
            raise RuntimeError("Audit changed parameter identity; stop training")
        if not _equal(self.capture(), s):
            raise RuntimeError(
                "Audit rollback failed exact verification; stop training"
            )


@torch.inference_mode(False)
def optimizer_transfer_audit(
    model,
    optimizer,
    histories,
    sampler,
    loss_function,
    *,
    negative_count,
    calibration,
    explicit_generators,
    update_index,
):
    device = next(model.parameters()).device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter()
    result = {
        "available": False,
        "reason": None,
        "records": [],
        "summary": {},
        "protocol": {
            "update_index": update_index,
            "seed": 424242,
            "paired_batches": 2,
            "update_batch_size": 16,
            "assessment_users_requested": 8,
            "seconds_cap": 30.0,
            "clip_norm": 5.0,
            "precision": "float32 eval-mode autograd; training replay uses train mode",
            "decay_settings": ["configured", "disabled_current_step"],
            "conditions": ["NO_STEP", "ZERO", "HALF_T", "HALF_R", "MIXTURE", "ONE_R"],
            "active_support": "All requires_grad parameters; natural missing gradients recorded",
            "loss": "Supplied candidate-averaged BCE/gBCE, sequence means, fixed half mixture",
            "window_policy": "Unchanged sampler: uniform-user/uniform-endpoint terminal prefixes; final-L reconstruction windows. Branch windows and right context are not matched.",
            "assessment": "Eight user-disjoint training prefixes; fixed one target plus 99 unique prefix-excluded negatives; no later-event exclusions",
            "relative_denominator_minimum": 1e-12,
            "bootstrap": "500 paired update-batch resamples, seed 42; descriptive conditional intervals, only two batches",
            "scheduler_scaler": "Absent in the supplied float32 Adam training loop",
            "mutable_state": "Supplied BERT4Rec has no caches or mutable non-state tensor attributes; registered hooks are rejected",
        },
        "metric_definitions": {
            "affinity": "Assessment loss after matched ZERO minus loss after condition; positive is beneficial conditional current-gradient transfer",
            "total_loss_change": "Assessment loss after condition minus NO_STEP; includes optimizer history and current decay",
            "relative_transfer": "1 - after/NO_STEP, when NO_STEP > 1e-12; total update effect, not zero-controlled affinity",
            "margin": "Target logit minus tenth-largest fixed negative logit; training-prefix diagnostic, not validation HR@10",
            "ONE_R": "Half-scaled one uniformly selected original target per reconstruction sequence; all original input MASKs retained",
            "decay": "Disabling current coupled Adam decay does not remove its historical moment contributions",
        },
    }
    transaction = _Transaction(model, optimizer, explicit_generators)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    replay_failed = False
    completed_steps = 0
    result["continuation_replay"] = {
        "passed": False,
        "available": False,
        "reason": "Replay has not completed",
    }

    def boundary():
        synchronize()
        if time.perf_counter() - started >= 30.0:
            raise TimeoutError("Audit synchronized wall-time cap reached")

    def norm(gradients):
        return math.sqrt(
            sum(
                float(g.detach().double().square().sum())
                for g in gradients
                if g is not None
            )
        )

    def loss(batch, rows=None, one=False):
        sequence, positive, negative, terminal, eligible, chosen = batch
        if rows is None:
            return loss_function(
                model,
                sequence,
                positive,
                negative,
                terminal,
                terminal_weight=0.5,
                eligible_counts=eligible,
                calibration=calibration,
            )
        s, p, n, e = sequence[rows], positive[rows], negative[rows], eligible[rows]
        if one:
            selected = chosen[rows]
            p = p.masked_fill(~selected, 0)
            n = n.masked_fill(~selected[..., None], 0)
        return loss_function(model, s, p, n, eligible_counts=e, calibration=calibration)

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        value = loss(batch)
        value.backward()
        gradients = [
            None if p.grad is None else p.grad.detach().clone() for p in parameters
        ]
        torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        return float(value.detach()), gradients

    try:
        if update_index != 200 or type(optimizer) is not torch.optim.Adam:
            raise ValueError("Audit requires update 200 of the supplied Adam loop")
        if any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("Audit requires float32 parameters")
        if any(
            m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
            for m in model.modules()
        ):
            raise ValueError("Untracked hooks make transactional auditing unavailable")
        # Verify the equality detector on copied fixtures, never deployed tensors.
        fixtures = []
        for field in ("model", "optimizer", "gradients", "modes", "cpu"):
            changed = copy.deepcopy(transaction.saved)
            if field == "model":
                next(iter(changed[field].values())).view(-1)[0] += 1
            elif field == "optimizer":
                changed[field]["param_groups"][0]["lr"] += 1.0
            elif field == "gradients":
                changed[field][0] = (
                    None
                    if changed[field][0] is not None
                    else torch.zeros_like(parameters[0])
                )
            elif field == "modes":
                changed[field][0] = not changed[field][0]
            else:
                changed[field][0] ^= 1
            fixtures.append(not _equal(changed, transaction.saved))
        result["verification_fixtures"] = {
            "faults_detected": sum(fixtures),
            "eligible_faults": len(fixtures),
            "clean_false_alarms": int(
                not _equal(copy.deepcopy(transaction.saved), transaction.saved)
            ),
            "clean_controls": 1,
            "scope": "Snapshot comparator only; not exhaustive fault coverage",
        }
        if not all(fixtures) or result["verification_fixtures"]["clean_false_alarms"]:
            raise ValueError("Snapshot comparator fixture failed")
        rng, negative_rng = random.Random(424242), random.Random(425251)
        target_rng = random.Random(426263)
        result["protocol"]["one_target_seed"] = 426263
        users = [u for u, h in histories.items() if len(h) >= 2]
        rng.shuffle(users)
        if len(users) < 40:
            raise ValueError("Insufficient users for disjoint audit panels")
        update_histories = {u: histories[u] for u in users[:32]}
        batches = []
        for _ in range(2):
            s, p, n, t, e = sampler(
                update_histories,
                model.itemnum,
                16,
                model.maxlen,
                return_branch_types=True,
                rng=rng,
                negative_count=negative_count,
                negative_rng=negative_rng,
                return_eligible_counts=True,
            )
            if n.ndim == 2:
                n = n[..., None]
            chosen = np.zeros_like(p, dtype=np.bool_)
            for row in range(len(p)):
                chosen[row, target_rng.choice(np.flatnonzero(p[row]).tolist())] = True
            batches.append(
                tuple(
                    torch.as_tensor(x, device=device).clone()
                    for x in (s, p, n, t, e, chosen)
                )
            )
        # Assessment eligibility uses only the complete prefix, never its suffix.
        assessment = []
        candidate_rng = np.random.default_rng(424242)
        for user in users[32:]:
            history = histories[user]
            seen, endpoints = set(), []
            for index, target in enumerate(history):
                if index >= 1 and target not in seen:
                    endpoints.append(index)
                seen.add(target)
            if not endpoints:
                continue
            endpoint = rng.choice(endpoints)
            prefix, target = history[:endpoint], history[endpoint]
            pool = np.setdiff1d(
                np.arange(1, model.itemnum + 1), list(set(prefix) | {target})
            )
            if len(pool) < 99:
                continue
            ids = [target] + candidate_rng.choice(pool, 99, replace=False).tolist()
            s = np.zeros(model.maxlen, dtype=np.int64)
            context = prefix[-(model.maxlen - 1) :]
            s[-len(context) - 1 : -1], s[-1] = context, model.mask_token
            assessment.append((str(user), s, ids, model.itemnum - len(set(prefix))))
            if len(assessment) == 8:
                break
        if len(assessment) != 8:
            raise ValueError("Fewer than eight eligible disjoint assessment users")
        sequence = torch.tensor(np.stack([r[1] for r in assessment]), device=device)
        candidates = torch.tensor([r[2] for r in assessment], device=device)
        eligible = torch.tensor([r[3] for r in assessment], device=device)
        positive = torch.zeros_like(sequence)
        positive[:, -1] = candidates[:, 0]
        negatives = torch.zeros(
            (*sequence.shape, negative_count), dtype=torch.long, device=device
        )
        # Reuse fixed eligible candidates; support K > 99 by deterministic cycling.
        negatives[:, -1] = candidates[:, 1:][
            :, torch.arange(negative_count, device=device) % 99
        ]

        def assess():
            with torch.no_grad():
                value = loss_function(
                    model,
                    sequence,
                    positive,
                    negatives,
                    eligible_counts=eligible,
                    calibration=calibration,
                )
                features = model.encode_sequence(sequence)[:, -1]
                scores = torch.einsum(
                    "bd,bcd->bc", features, model.item_emb(candidates)
                )
                margins = (
                    scores[:, 0] - scores[:, 1:].sort(descending=True).values[:, 9]
                )
                if not torch.isfinite(value) or not torch.isfinite(margins).all():
                    raise FloatingPointError("Nonfinite audit assessment")
                return {
                    "loss": float(value),
                    "margin": float(margins.mean()),
                    "per_user_margins": margins.cpu().tolist(),
                }

        result["assessment_user_ids"] = [r[0] for r in assessment]
        result["assessment_candidates"] = candidates.cpu().tolist()
        result["branch_target_counts"] = [
            [int(b[1][b[3]].ne(0).sum()), int(b[1][~b[3]].ne(0).sum())] for b in batches
        ]
        # One actual stochastic training update is replayed before and after
        # the audit, including exact gradients, optimizer state and RNG state.
        with torch.inference_mode(False), torch.enable_grad():
            transaction.restore()
            model.train()
            reference_loss, reference_gradients = step(batches[0])
            reference_state = transaction.capture()
            transaction.restore()
            for batch_index, batch in enumerate(batches):
                boundary()
                transaction.restore()
                model.eval()
                branch_gradients = {}
                for name, rows in (("T", batch[3]), ("R", ~batch[3])):
                    value = loss(batch, rows)
                    branch_gradients[name] = torch.autograd.grad(
                        value, parameters, allow_unused=True
                    )
                    del value
                branch_gradients["M"] = torch.autograd.grad(
                    loss(batch), parameters, allow_unused=True
                )
                branch_gradients["ONE"] = torch.autograd.grad(
                    loss(batch, ~batch[3], one=True), parameters, allow_unused=True
                )
                supports = {
                    name: [g is not None for g in values]
                    for name, values in branch_gradients.items()
                }
                common = supports["M"]
                support_exact = all(
                    support == common for support in supports.values()
                ) and all(common)
                for decay in ("configured", "disabled_current_step"):
                    outcomes = {}
                    for condition in result["protocol"]["conditions"]:
                        boundary()
                        transaction.restore()
                        model.eval()
                        if decay == "disabled_current_step":
                            for group in optimizer.param_groups:
                                group["weight_decay"] = 0.0
                        raw_norm = scaled_norm = clipped_norm = 0.0
                        if condition != "NO_STEP":
                            source, coefficient = {
                                "ZERO": (None, 0.0),
                                "HALF_T": ("T", 0.5),
                                "HALF_R": ("R", 0.5),
                                "MIXTURE": ("M", 1.0),
                                "ONE_R": ("ONE", 0.5),
                            }[condition]
                            optimizer.zero_grad(set_to_none=True)
                            gradients = (
                                branch_gradients[source]
                                if source
                                else [None] * len(parameters)
                            )
                            raw_norm = norm(gradients)
                            for parameter, gradient in zip(parameters, gradients):
                                parameter.grad = (
                                    torch.zeros_like(parameter)
                                    if gradient is None
                                    else gradient.detach().clone().mul_(coefficient)
                                )
                            scaled_norm = norm([p.grad for p in parameters])
                            torch.nn.utils.clip_grad_norm_(
                                parameters, 5.0, error_if_nonfinite=True
                            )
                            clipped_norm = norm([p.grad for p in parameters])
                            optimizer.step()
                            completed_steps += 1
                        displacement = {}
                        for name, parameter in named:
                            block = (
                                ".".join(name.split(".")[:3])
                                if name.startswith("encoder.layers.")
                                else name.split(".")[0]
                            )
                            delta = (
                                parameter.detach() - transaction.saved["model"][name]
                            )
                            displacement[block] = displacement.get(block, 0.0) + float(
                                delta.double().square().sum()
                            )
                        measured = assess()
                        measured.update(
                            raw_gradient_norm=raw_norm,
                            scaled_gradient_norm=scaled_norm,
                            clipped_gradient_norm=clipped_norm,
                            clipping_factor=min(1.0, 5.0 / (scaled_norm + 1e-6)),
                            displacement={
                                k: math.sqrt(v) for k, v in displacement.items()
                            },
                        )
                        outcomes[condition] = measured
                    baseline, zero = (
                        outcomes["NO_STEP"]["loss"],
                        outcomes["ZERO"]["loss"],
                    )
                    for condition, measured in outcomes.items():
                        measured.update(
                            affinity=zero - measured["loss"],
                            total_loss_change=measured["loss"] - baseline,
                            relative_transfer=(
                                1.0 - measured["loss"] / baseline
                                if baseline > 1e-12
                                else None
                            ),
                        )
                    result["records"].append(
                        {
                            "batch": batch_index,
                            "decay": decay,
                            "outcomes": outcomes,
                            "natural_gradient_support": supports,
                            "faithful_training_support": support_exact,
                            "mixture_minus_half_terminal_loss": outcomes["MIXTURE"][
                                "loss"
                            ]
                            - outcomes["HALF_T"]["loss"],
                            "mixture_minus_half_reconstruction_loss": outcomes[
                                "MIXTURE"
                            ]["loss"]
                            - outcomes["HALF_R"]["loss"],
                        }
                    )
                del branch_gradients
            transaction.restore()
            model.train()
            replay_loss, replay_gradients = step(batches[0])
            replay_state = transaction.capture()
            passed = (
                _equal(reference_loss, replay_loss)
                and _equal(reference_gradients, replay_gradients)
                and _equal(reference_state, replay_state)
            )
            result["continuation_replay"] = {
                "passed": passed,
                "available": True,
                "updates": 1,
                "comparison": "Exact train-mode loss, gradients, full state and subsequent RNGs",
            }
            if not passed:
                replay_failed = True
                raise RuntimeError(
                    "Exact stochastic continuation replay failed; training must stop"
                )
            result["available"] = all(
                r["faithful_training_support"] for r in result["records"]
            )
            if not result["available"]:
                result["reason"] = (
                    "Common explicit-zero support differs from the faithful training pipeline"
                )
        for decay in result["protocol"]["decay_settings"]:
            rows = [r for r in result["records"] if r["decay"] == decay]
            indices = np.random.default_rng(42).integers(0, len(rows), (500, len(rows)))
            result["summary"][decay] = {}
            for condition in result["protocol"]["conditions"]:
                values = np.array([r["outcomes"][condition]["affinity"] for r in rows])
                result["summary"][decay][condition] = {
                    "mean_affinity": float(values.mean()),
                    "paired_batches": len(rows),
                    "descriptive_ci95": np.percentile(
                        values[indices].mean(1), [2.5, 97.5]
                    ).tolist(),
                }
    except Exception as exc:
        result["available"] = False
        result["reason"] = f"{type(exc).__name__}: {exc}"
        print(f"BERT4Rec optimizer audit unavailable: {result['reason']}", flush=True)
    finally:
        # A restoration failure escapes this function and stops training.
        transaction.restore()
        result["exact_restoration_passed"] = True
        synchronize()
        result["synchronized_wall_seconds"] = time.perf_counter() - started
        # Count actual steps, including work from an interrupted condition group.
        result["completed_condition_steps"] = completed_steps
        result["limitations"] = (
            "Conditional finite-update diagnostic at one early checkpoint; two paired batches. "
            "Eval-mode transfer differs from stochastic training. No future validation, "
            "convergence, semantic mechanism or additive Adam decomposition is established. "
            "The cap is checked at synchronized boundaries; in-flight operations can overrun it. "
            "No audit result changes subsequent training or serving."
        )
    if replay_failed:
        raise RuntimeError(
            "Optimizer audit restored its snapshot, but exact continuation failed; "
            f"training stopped. Audit seconds={result['synchronized_wall_seconds']:.6f}, "
            f"completed condition steps={completed_steps}"
        )
    return result


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Evidence-linked interpretation of paired BERT4Rec diagnostics
def diagnosis_interpreter_prompt(raw_diagnosis):
    return f"""
Analyze these measured BERT4Rec diagnostics and their definitions:
{json.dumps(raw_diagnosis, indent=2, allow_nan=False)}
Return JSON with status, core_findings, key_implications, and evidence.
Use at most three findings and three implications. Cite metric names, values,
counts, availability flags, and uncertainty intervals for each substantive claim.
Distinguish serving defects, measured history sensitivity, missing information,
and hypotheses requiring independent validation. If serving parity or sampler
checks fail or are unavailable, withhold substantive sensitivity conclusions.
Null values mean unavailable, never zero or successful checks. Missing category
coverage prevents a four-category macro conclusion. These training-prefix
candidate results are not host validation HR@10. Stability, sensitivity, and
embedding alignment do not establish recommendation quality or validation gains.
A mapped embedding cosine of 0.5 means approximately zero mean cosine, not proven
healthy representation capacity. Do not infer intent, simulator traits, ratings,
review content, item categories, prices, stores, or ownership. Unknown identity
and abstention are not negative labels. Intervals cover sampled-user uncertainty
only, not training seeds or candidate sampling. Repeated validation selection is
optimistic. No recommendation-score fusion or additional predictors are allowed.
Use INSUFFICIENT_EVIDENCE when the available measurements do not support a status.
For residual_patching, report its cap, elapsed time, completed interventions,
partition denominators, all attempted sites, missing donors, and control results.
Separate exploration from held-out-user confirmation. Interpret effects only
conditional on the specified corruption and patch; natural activation donors can
produce out-of-distribution hybrid computations. Frequency and recency matching
provide no semantic compatibility evidence. Never describe a local raw p-value
or the conservative Bonferroni bound as a four-category Holm-adjusted result.
No Holm discovery is established without the complete cross-category family.
Identity, structural-zero, and MASK-restoration controls test implementation;
nonzero unrelated or random control effects require discussion, not concealment.
For optimizer-transfer audits, require available=true, exact_restoration_passed=true,
and continuation_replay.passed=true before interpreting effects. Distinguish
ZERO-controlled affinity from total change relative to NO_STEP. Report completed
condition steps, wall seconds, natural gradient support, branch target counts,
decay settings, and the two-batch uncertainty limitation. Adam branch updates
are not additive. Comparator fixtures test only their reported fault coverage.
Missing trajectories leave optimization/convergence unassessed. These diagnostics
neither demonstrate a repair nor establish validation improvements.
"""


### <<< Self_EvolveRec-BLOCK-END


### >>> Self_EvolveRec-BLOCK-START: Auditable budgeted residual patching with verified hooks and controls
@torch.no_grad()
def _residual_patching_pilot(probe, examples, seconds=30.0):
    import time
    from math import comb

    net = probe.net
    started = time.perf_counter()
    rng = np.random.default_rng(1842)
    result = {
        "available": False,
        "reason": None,
        "records": [],
        "summary": {},
        "protocol": {
            "seed": 1842,
            "seconds_cap": float(seconds),
            "maximum_users": 64,
            "minimum_users_per_partition": 8,
            "partition": "Seeded permutation; first half exploration, remainder confirmation.",
            "rules": ["matched_donor_A", "matched_donor_B"],
            "replacement_rules": {
                "matched_donor_A": "Uniform distinct item from the matched donor pool.",
                "matched_donor_B": "Closest training frequency to the original among remaining matched items; ascending item ID breaks ties.",
            },
            "frequency_edges": [1, 4, 16, 64, 256],
            "recency_bands": 4,
            "primary": "target minus strongest clean negative, identity fixed before intervention",
            "directions": ["restore", "disrupt"],
            "sites": ["input_residual", "enter_second_attention"],
            "selection": "Largest absolute exploration mean per rule and direction; lock sign and site.",
            "test": "One-sided exact sign test of locked-direction confirmation effects; zeros excluded and reported.",
            "family_size": 16,
            "multiplicity": "Four categories x two rules x two directions. Global Holm requires all category p-values; no Holm rejection is declared here.",
            "magnitude_edges": None,
            "magnitude_matching": "Match intervention norm directly; descriptive quartile boundaries fitted on exploration only and frozen before confirmation.",
            "random_control": "One seeded random history site in the same recency band and layer, with an isotropic residual change of exactly the intervention norm.",
            "tolerances": {"atol": 1e-5, "rtol": 1e-4},
        },
        "training_trajectories": {
            "available": False,
            "reason": "The supplied ModelView exposes no loss or validation trajectory; convergence is unassessed.",
        },
        "confirmation": [],
        "counts": {},
        "timed_out": False,
    }
    hooks = []
    locked = {}

    def checkpoint(synchronize=False):
        if synchronize and probe.device.type == "cuda":
            torch.cuda.synchronize(probe.device)
        if time.perf_counter() - started >= seconds:
            raise TimeoutError("Diagnostic cap reached at a checked boundary.")

    try:
        layers = net.encoder.layers
        if len(layers) != 2 or not all(layer.norm_first for layer in layers):
            raise ValueError(
                "Pilot supports the supplied two-layer pre-norm BERT4Rec only."
            )
        modules = {
            "input_residual": net.dropout,
            "enter_second_attention": layers[0],
            "final_history_output": layers[-1],
            "final_mask_feature": net.norm,
        }
        result["hook_locations"] = {
            "input_residual": "net.dropout output; before first attention",
            "enter_second_attention": "net.encoder.layers[0] output; before second attention",
            "final_history_output": "net.encoder.layers[1] output; no later cross-position operation",
            "final_mask_feature": "net.norm output at appended MASK; positive control",
        }
        frequencies, occurrences = Counter(), {}
        for user, history in probe._histories().items():
            checkpoint()
            items = [int(item) for item in history]
            for item in items:
                if 1 <= item <= net.itemnum:
                    frequencies[item] += 1
            occurrences[str(user)] = items
        bins = lambda item: int(
            np.searchsorted([1, 4, 16, 64, 256], frequencies[item], side="right")
        )
        donors = {}
        for user, history in occurrences.items():
            checkpoint()
            visible = history[-(net.maxlen - 1) :]
            for index, item in enumerate(visible):
                if 1 <= item <= net.itemnum:
                    band = min(3, 4 * index // max(1, len(visible)))
                    donors.setdefault((band, bins(item)), []).append((user, item))
        permutation = rng.permutation(len(examples))
        ordered = [examples[int(index)] for index in permutation]
        midpoint = len(ordered) // 2
        if midpoint < 8 or len(ordered) - midpoint < 8:
            raise ValueError("Fewer than eight eligible users in each partition.")
        result["counts"].update(
            eligible_users=len(ordered),
            exploration_users=midpoint,
            confirmation_users=len(ordered) - midpoint,
        )
        result["locked_sites"] = []
        donor_histories = sorted(occurrences.items())

        def forward(sequence, positions, candidates, patch=None, capture=False):
            checkpoint(synchronize=True)
            cache, visits = {}, Counter()
            try:
                for name, module in modules.items():

                    def hook(_module, _args, output, name=name):
                        visits[name] += 1
                        if not isinstance(output, torch.Tensor) or output.ndim != 3:
                            raise ValueError(f"Unsupported residual output at {name}")
                        if not bool(torch.isfinite(output).all()):
                            raise ValueError(f"Nonfinite residual output at {name}")
                        if capture:
                            cache[name] = output.detach().clone()
                        if patch is not None and name == patch[0]:
                            if output[0, patch[1]].shape != patch[2].shape:
                                raise ValueError(f"Patch shape mismatch at {name}")
                            output = output.clone()
                            output[0, patch[1]] = patch[2]
                        return output

                    hooks.append(module.register_forward_hook(hook))
                scores = probe._score(sequence, positions, candidates)
                if any(visits[name] != 1 for name in modules):
                    raise ValueError(
                        f"Unsupported hook execution counts: {dict(visits)}"
                    )
                result["counts"]["completed_forwards"] = (
                    result["counts"].get("completed_forwards", 0) + 1
                )
                checkpoint(synchronize=True)
                return scores, cache
            finally:
                while hooks:
                    hooks.pop().remove()

        def measures(scores, candidates, competitor, baseline_order):
            order = probe._ranking(scores, candidates)
            rank = order.index(int(candidates[0])) + 1
            negatives = scores[1:]
            return {
                "fixed_contrast": float(scores[0] - scores[competitor]),
                "switching_margin": float(scores[0] - negatives.max()),
                "tenth_negative_margin": float(
                    scores[0] - negatives.sort(descending=True).values[9]
                ),
                "target_rank": float(rank),
                "reciprocal_rank": 1.0 / rank,
                "hit10": float(rank <= 10),
                "top10_overlap": len(set(order[:10]) & set(baseline_order[:10])) / 10.0,
                "ordered_agreement": float(order == baseline_order),
            }

        for partition, subset in (
            ("exploration", ordered[:midpoint]),
            ("confirmation", ordered[midpoint:]),
        ):
            if partition == "confirmation":
                norms = [row["change_norm"] for row in result["records"]]
                result["protocol"]["magnitude_edges"] = (
                    np.unique(np.quantile(norms, [0.25, 0.5, 0.75])).tolist()
                    if norms
                    else []
                )
                for rule in result["protocol"]["rules"]:
                    for direction in ("restore", "disrupt"):
                        choices = []
                        for site in result["protocol"]["sites"]:
                            rows = [
                                r
                                for r in result["records"]
                                if r["partition"] == "exploration"
                                and r["rule"] == rule
                                and r["direction"] == direction
                                and r["site"] == site
                            ]
                            if len(rows) >= 8 and all(
                                r["controls_passed"] for r in rows
                            ):
                                effect = float(
                                    np.mean(
                                        [r["delta"]["fixed_contrast"] for r in rows]
                                    )
                                )
                                if effect != 0.0:
                                    choices.append(
                                        (abs(effect), site, 1 if effect > 0 else -1)
                                    )
                        if choices:
                            _, site, sign = max(choices)
                            locked[(rule, direction)] = (site, sign)
                            result["locked_sites"].append(
                                {
                                    "rule": rule,
                                    "direction": direction,
                                    "site": site,
                                    "sign": sign,
                                }
                            )
                if not locked:
                    raise ValueError(
                        "No nonzero exploration effect with sufficient users and passing controls."
                    )
            for user, prefix, target, ids in subset:
                checkpoint()
                _, clean, positions = probe._serving_tensors(prefix)
                visible = torch.nonzero(clean[0, :-1].ne(0)).flatten().tolist()
                position = visible[-1]
                band = min(3, 4 * visible.index(position) // len(visible))
                original = int(clean[0, position])
                eligible = sorted(
                    set(
                        item
                        for owner, item in donors.get((band, bins(original)), [])
                        if owner != user and item != original and item not in ids
                    )
                )
                if len(eligible) < 2:
                    result["counts"]["unavailable_matched_donors"] = (
                        result["counts"].get("unavailable_matched_donors", 0) + 1
                    )
                    continue
                clean_scores, clean_cache = forward(
                    clean,
                    positions,
                    torch.tensor(ids, device=probe.device),
                    capture=True,
                )
                candidates = torch.tensor(ids, dtype=torch.long, device=probe.device)
                clean_order = probe._ranking(clean_scores, candidates)
                competitor_id = next(item for item in clean_order if item != target)
                competitor = ids.index(competitor_id)
                first = int(rng.choice(eligible))
                second = min(
                    (item for item in eligible if item != first),
                    key=lambda item: (
                        abs(frequencies[item] - frequencies[original]),
                        item,
                    ),
                )
                chosen = (first, second)
                unrelated = next(
                    (
                        history
                        for owner, history in donor_histories
                        if owner != user
                        and len(history) >= len(visible)
                        and not (set(history[-len(visible) :]) & set(ids))
                        and all(
                            1 <= item <= net.itemnum
                            for item in history[-len(visible) :]
                        )
                    ),
                    None,
                )
                unrelated_cache = None
                if unrelated is not None:
                    _, donor_sequence, donor_positions = probe._serving_tensors(
                        unrelated[-len(visible) :]
                    )
                    _, unrelated_cache = forward(
                        donor_sequence, donor_positions, candidates, capture=True
                    )
                for rule, donor in zip(result["protocol"]["rules"], chosen):
                    corrupt = clean.clone()
                    corrupt[0, position] = int(donor)
                    corrupt_scores, corrupt_cache = forward(
                        corrupt, positions, candidates, capture=True
                    )
                    mask_scores, _ = forward(
                        corrupt,
                        positions,
                        candidates,
                        (
                            "final_mask_feature",
                            net.maxlen - 1,
                            clean_cache["final_mask_feature"][0, -1],
                        ),
                    )
                    zero_scores, _ = forward(
                        corrupt,
                        positions,
                        candidates,
                        (
                            "final_history_output",
                            position,
                            clean_cache["final_history_output"][0, position],
                        ),
                    )
                    positive_ok = bool(
                        torch.allclose(mask_scores, clean_scores, atol=1e-5, rtol=1e-4)
                    )
                    structural_ok = bool(
                        torch.allclose(
                            zero_scores, corrupt_scores, atol=1e-5, rtol=1e-4
                        )
                    )
                    for direction in ("restore", "disrupt"):
                        base_sequence, base_scores, source, destination = (
                            (corrupt, corrupt_scores, clean_cache, corrupt_cache)
                            if direction == "restore"
                            else (clean, clean_scores, corrupt_cache, clean_cache)
                        )
                        sites = (
                            result["protocol"]["sites"]
                            if partition == "exploration"
                            else (
                                [locked[(rule, direction)][0]]
                                if (rule, direction) in locked
                                else []
                            )
                        )
                        for site in sites:
                            base = measures(
                                base_scores, candidates, competitor, clean_order
                            )
                            vector = source[site][0, position]
                            scores, _ = forward(
                                base_sequence,
                                positions,
                                candidates,
                                (site, position, vector),
                            )
                            identity, _ = forward(
                                base_sequence,
                                positions,
                                candidates,
                                (site, position, destination[site][0, position]),
                            )
                            norm = float(
                                (vector - destination[site][0, position]).norm()
                            )
                            same_band = [
                                p
                                for i, p in enumerate(visible)
                                if min(3, 4 * i // len(visible)) == band
                            ]
                            random_position = int(rng.choice(same_band))
                            noise = torch.tensor(
                                rng.normal(size=vector.numel()),
                                dtype=vector.dtype,
                                device=vector.device,
                            )
                            noise = F.normalize(noise, dim=0) * norm
                            random_vector = (
                                destination[site][0, random_position] + noise
                            )
                            achieved_norm = float(
                                (
                                    random_vector
                                    - destination[site][0, random_position]
                                ).norm()
                            )
                            magnitude_ok = bool(
                                np.isclose(achieved_norm, norm, atol=1e-5, rtol=1e-4)
                            )
                            random_scores, _ = forward(
                                base_sequence,
                                positions,
                                candidates,
                                (site, random_position, random_vector),
                            )
                            unrelated_scores = None
                            if unrelated_cache is not None:
                                unrelated_scores, _ = forward(
                                    base_sequence,
                                    positions,
                                    candidates,
                                    (
                                        site,
                                        position,
                                        unrelated_cache[site][0, position],
                                    ),
                                )
                            values = measures(
                                scores, candidates, competitor, clean_order
                            )
                            identity_ok = bool(
                                torch.allclose(
                                    identity, base_scores, atol=1e-5, rtol=1e-4
                                )
                            )
                            result["records"].append(
                                {
                                    "user": user,
                                    "partition": partition,
                                    "rule": rule,
                                    "direction": direction,
                                    "site": site,
                                    "position": position,
                                    "donor": int(donor),
                                    "competitor": competitor_id,
                                    "baseline": base,
                                    "clean": measures(
                                        clean_scores,
                                        candidates,
                                        competitor,
                                        clean_order,
                                    ),
                                    "corrupt": measures(
                                        corrupt_scores,
                                        candidates,
                                        competitor,
                                        clean_order,
                                    ),
                                    "delta": {
                                        key: values[key] - base[key] for key in base
                                    },
                                    "random_control": measures(
                                        random_scores,
                                        candidates,
                                        competitor,
                                        clean_order,
                                    ),
                                    "unrelated_control": (
                                        measures(
                                            unrelated_scores,
                                            candidates,
                                            competitor,
                                            clean_order,
                                        )
                                        if unrelated_scores is not None
                                        else None
                                    ),
                                    "unrelated_control_reason": (
                                        None
                                        if unrelated_scores is not None
                                        else "No candidate-excluded unrelated training history of matching length."
                                    ),
                                    "change_norm": norm,
                                    "random_change_norm": achieved_norm,
                                    "random_position": random_position,
                                    "random_magnitude_passed": magnitude_ok,
                                    "magnitude_bin": None,
                                    "identity_passed": identity_ok,
                                    "mask_restoration_passed": positive_ok,
                                    "structural_zero_passed": structural_ok,
                                    "controls_passed": identity_ok
                                    and positive_ok
                                    and structural_ok
                                    and magnitude_ok
                                    and unrelated_scores is not None,
                                }
                            )
        result["available"] = bool(result["records"])
        if not result["available"]:
            result["reason"] = (
                "No completed interventions with available matched donors."
            )
    except TimeoutError as exc:
        result["timed_out"] = True
        result["reason"] = str(exc)
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        while hooks:
            hooks.pop().remove()
        if probe.device.type == "cuda":
            torch.cuda.synchronize(probe.device)
        result["forward_phase_seconds"] = time.perf_counter() - started
    # Partial runs remain descriptive; confirmation requires a completed pilot.
    groups = {}
    edges = result["protocol"]["magnitude_edges"]
    for row in result["records"]:
        if edges is not None:
            row["magnitude_bin"] = int(
                np.searchsorted(edges, row["change_norm"], side="right")
            )
        key = (row["partition"], row["rule"], row["direction"], row["site"])
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        values = np.asarray(
            [[row["delta"][name] for name in row["delta"]] for row in rows]
        )
        indices = np.random.default_rng(42).integers(0, len(rows), (500, len(rows)))
        intervals = np.percentile(values[indices].mean(1), [2.5, 97.5], axis=0)
        result["summary"]["/".join(key)] = {
            "users": len(rows),
            "controls_passed": all(row["controls_passed"] for row in rows),
            "effects": {
                name: {
                    "mean": float(values[:, i].mean()),
                    "ci95": intervals[:, i].tolist(),
                    "zero_effect_users": int(np.count_nonzero(values[:, i] == 0)),
                }
                for i, name in enumerate(rows[0]["delta"])
            },
        }
        if key[0] == "confirmation":
            sign = locked[(key[1], key[2])][1]
            effects = values[:, 0] * sign
            nonzero = effects[effects != 0]
            valid = (
                result["available"]
                and not result["timed_out"]
                and result["reason"] is None
                and len(rows) >= 8
                and all(row["controls_passed"] for row in rows)
            )
            # Exact one-sided Binomial(n, 0.5) tail; n <= 64 here.
            p = (
                sum(
                    comb(len(nonzero), k)
                    for k in range(int((nonzero > 0).sum()), len(nonzero) + 1)
                )
                / (2 ** len(nonzero))
                if valid and len(nonzero)
                else None
            )
            result["confirmation"].append(
                {
                    "rule": key[1],
                    "direction": key[2],
                    "site": key[3],
                    "locked_sign": sign,
                    "users": len(rows),
                    "nonzero_users": len(nonzero),
                    "raw_p": p,
                    "holm_adjusted_p": None,
                    "holm_status": "Requires all four category results; no confirmed discovery declared.",
                    "conservative_family_bound_p": (
                        min(1.0, 16 * p) if p is not None else None
                    ),
                }
            )
    result["counts"]["completed_interventions"] = len(result["records"])
    result["counts"]["completed_users_by_partition"] = {
        partition: len(
            {r["user"] for r in result["records"] if r["partition"] == partition}
        )
        for partition in ("exploration", "confirmation")
    }
    result["limitations"] = (
        "Frozen-checkpoint intervention-conditional pilot. Natural donor hybrids may be out of distribution. Frequency/recency matching does not establish semantic compatibility. Random controls directly match the measured residual-change norm; descriptive bins are learned on exploration only. No unique mechanism, intent, validation gain, training-seed uncertainty, or repair comparison is established. The wall-time cap is checked around synchronized forwards and during preparation; an in-flight operation and CPU summaries can overrun it."
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


### <<< Self_EvolveRec-BLOCK-END


class DiagnosisProbe:
    ### >>> Self_EvolveRec-BLOCK-START: Propagate explicit category identity from the trained network
    def __init__(self, view, device):
        self.model = view
        self.view = view
        self.net = view.model
        self.device = torch.device(device)
        self.meta_data = view.meta_data
        self.review_data = view.review_data
        self.user_train = view.user_train
        self.category = getattr(view, "dataset_name", None)
        if self.category is None:
            self.category = getattr(self.net, "dataset_name", None)
        self._contract = {}
        self._embedding_count = 0

    def _get_categories(self, item_id: int):
        return []

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Seeded real-item embedding alignment with explicit missingness
    def analyze_embeddings_collapse_cosine(
        self, sample_size: int = 2048, seed: int = 42
    ):
        with torch.no_grad():
            emb = self.view.item_emb.weight.detach()[1 : self.net.itemnum + 1]
            k = min(2048, sample_size, emb.size(0))
            self._embedding_count = k
            if k < 2:
                return None
            generator = torch.Generator(device="cpu").manual_seed(seed)
            indices = torch.randperm(emb.size(0), generator=generator)[:k].to(
                emb.device
            )
            x = F.normalize(emb.index_select(0, indices).float(), dim=-1)
            if not torch.isfinite(x).all():
                return None
            mean_cos = (x.sum(0).square().sum() - x.square().sum()) / (k * (k - 1))
            return float((mean_cos.clamp(-1, 1).item() + 1.0) / 2.0)

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Check supplied all-MASK sampler before in-sample margin measurement
    @torch.no_grad()
    def analyze_pos_neg_margin_with_category_breakdown(
        self,
        dataloader,
        num_batches: int = 10,
        eps: float = 0.5,
        topk_categories: int = 20,
    ):
        from itertools import islice

        margins = []
        checks = {
            "batches": 0,
            "valid_batches": 0,
            "invalid_batches": 0,
            "supervised_positions": 0,
            "errors": [],
        }
        reserved_ok = (
            self.net.mask_token == self.net.itemnum + 1
            and self.view.item_emb.num_embeddings == self.net.itemnum + 2
            and self.view.item_emb.padding_idx == 0
        )
        checks["reserved_tokens_valid"] = bool(reserved_ok)
        checks["policy"] = (
            "Supplied terminal/interior sampler: all selected targets replaced by MASK."
        )
        previous_state = self.net.training
        try:
            self.net.eval()
            for batch in islice(dataloader, min(10, num_batches)):
                checks["batches"] += 1
                try:
                    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                        raise ValueError(
                            "expected (None, sequence, positive, negative)"
                        )
                    seq, pos, neg = [
                        torch.as_tensor(x, device=self.device, dtype=torch.long)
                        for x in batch[1:4]
                    ]
                    if (
                        seq.ndim != 2
                        or seq.shape != pos.shape
                        or seq.shape != neg.shape
                    ):
                        raise ValueError("expected matching B x L tensors")
                    if seq.shape[1] != self.net.maxlen or seq.shape[0] == 0:
                        raise ValueError("unexpected sequence length or empty batch")
                    valid = pos.ne(0)
                    ok = (
                        reserved_ok
                        and bool(((seq >= 0) & (seq <= self.net.mask_token)).all())
                        and bool((valid.sum(1) > 0).all())
                        and bool((seq[valid] == self.net.mask_token).all())
                        and bool(
                            ((pos[valid] >= 1) & (pos[valid] <= self.net.itemnum)).all()
                        )
                        and bool(
                            ((neg[valid] >= 1) & (neg[valid] <= self.net.itemnum)).all()
                        )
                        and bool((pos[valid] != neg[valid]).all())
                        and bool((neg[~valid] == 0).all())
                    )
                    if not ok:
                        raise ValueError(
                            "reserved-token, all-MASK supervision, or real-item contract failed"
                        )
                    features = self.view.log2feats(seq.detach().cpu().numpy())
                    logits = self.view.predict_score(
                        features, torch.stack((pos, neg), -1)
                    )
                    margin = (logits[..., 0] - logits[..., 1])[valid].float()
                    if not torch.isfinite(margin).all():
                        raise ValueError("nonfinite supervised logits")
                    margins.append(margin.detach().cpu())
                    checks["valid_batches"] += 1
                    checks["supervised_positions"] += int(valid.sum().item())
                except (ValueError, RuntimeError, TypeError, IndexError) as exc:
                    checks["invalid_batches"] += 1
                    checks["errors"].append(str(exc))
        finally:
            self.net.train(previous_state)
        checks["passed"] = bool(
            reserved_ok and checks["valid_batches"] and not checks["invalid_batches"]
        )
        checks["provenance_limit"] = (
            "Checks the supplied all-MASK policy; loader tensors alone cannot establish "
            "prefix provenance or verify uniform negative sampling."
        )
        self._contract = checks
        result = dict.fromkeys(
            ("mean", "p10", "neg_beats_pos_rate", "small_margin_rate")
        )
        result["count"] = 0
        low = wrong = 0
        if margins:
            values = torch.cat(margins)
            result.update(
                count=int(values.numel()),
                mean=float(values.mean()),
                p10=float(
                    values.kthvalue(max(1, int(np.ceil(0.1 * values.numel())))).values
                ),
                neg_beats_pos_rate=float((values < 0).float().mean()),
                small_margin_rate=float((values < eps).float().mean()),
            )
            low, wrong = int((values < eps).sum()), int((values < 0).sum())
        breakdown = {
            "available": False,
            "reason": "Item categories are unavailable in this benchmark.",
            "eps": float(eps),
            "low_margin_total_positions": low,
            "wrong_rank_total_positions": wrong,
            "low_margin_top_categories": {"pos_item": [], "neg_item": []},
            "wrong_rank_top_categories": {"pos_item": [], "neg_item": []},
        }
        return result, breakdown

    ### <<< Self_EvolveRec-BLOCK-END

    ### >>> Self_EvolveRec-BLOCK-START: Paired sensitivity and gated frozen-checkpoint residual patching
    def _histories(self):
        if "History" in self.user_train:
            return self.user_train["History"]
        return self.user_train

    def _serving_tensors(self, history):
        length = self.net.maxlen
        raw = torch.zeros((1, length), dtype=torch.long, device=self.device)
        retained = list(history[-length:])
        if retained:
            raw[0, -len(retained) :] = torch.tensor(retained, device=self.device)
        serving = torch.zeros_like(raw)
        serving[:, :-1] = raw[:, 1:]
        serving[:, -1] = self.net.mask_token
        positions = torch.arange(1, length + 1, device=self.device)[None, :]
        positions = positions * serving.ne(0)
        return raw, serving, positions

    def _explicit_forward(self, sequence, positions):
        # This is the supplied BERT4Rec encoder, with fixed absolute positions.
        # Interior PAD holes are excluded from attention keys in every layer.
        padding = sequence.eq(0)
        if bool(padding.all(1).any()):
            raise ValueError("diagnostic inputs must contain an attention-visible MASK")
        x = self.net.item_emb(sequence) * (self.net.hidden_units**0.5)
        x = self.net.dropout(x + self.net.pos_emb(positions))
        return self.net.norm(self.net.encoder(x, src_key_padding_mask=padding))

    @staticmethod
    def _ranking(scores, candidates):
        scores = torch.as_tensor(scores).detach().float().cpu().reshape(-1)
        candidates = torch.as_tensor(candidates).detach().long().cpu().reshape(-1)
        if scores.numel() != candidates.numel() or not torch.isfinite(scores).all():
            raise ValueError("candidate scores must have matching length and be finite")
        by_id = torch.argsort(candidates, stable=True)
        order = by_id[torch.argsort(scores[by_id], descending=True, stable=True)]
        return candidates[order].tolist()

    def _feature_scores(self, feature, candidates):
        # Preserve the documented B x L x C scoring contract even for one MASK.
        feature = feature.reshape(1, 1, -1)
        items = candidates.reshape(1, 1, -1)
        scores = self.view.predict_score(feature, items)
        if scores.shape != items.shape or not bool(torch.isfinite(scores).all()):
            raise ValueError("ModelView must return finite B x L x C logits")
        return scores.reshape(-1)

    def _score(self, sequence, positions, candidates):
        features = self._explicit_forward(sequence, positions)
        # Read the actual appended MASK slot, including after deletion holes.
        if int(sequence[0, -1]) != self.net.mask_token:
            raise ValueError("serving readout must contain the reserved MASK")
        return self._feature_scores(features[0, -1], candidates)

    def _examples(self):
        rng = random.Random(42)
        candidate_rng = np.random.default_rng(42)
        histories = self._histories()
        users = sorted(histories, key=lambda user: str(user))
        selected = rng.sample(users, min(64, len(users)))
        counts = {
            "training_users": len(users),
            "selected_users": len(selected),
            "invalid_identity": 0,
            "no_eligible_endpoint": 0,
            "repeated_target_endpoints_excluded": 0,
            "insufficient_candidate_pool": 0,
            "insufficient_visible_history": 0,
        }
        examples = []
        for user in selected:
            try:
                history = [int(item) for item in histories[user]]
            except (TypeError, ValueError):
                counts["invalid_identity"] += 1
                continue
            if any(item < 1 or item > self.net.itemnum for item in history):
                counts["invalid_identity"] += 1
                continue
            endpoints, seen = [], set()
            for endpoint, target in enumerate(history):
                if endpoint >= 4:
                    if target not in seen:
                        endpoints.append(endpoint)
                    else:
                        counts["repeated_target_endpoints_excluded"] += 1
                seen.add(target)
            if not endpoints:
                counts["no_eligible_endpoint"] += 1
                continue
            endpoint = rng.choice(endpoints)
            prefix, target = history[:endpoint], history[endpoint]
            excluded = set(prefix) | {target}
            pool = np.setdiff1d(
                np.arange(1, self.net.itemnum + 1, dtype=np.int64),
                np.fromiter(excluded, dtype=np.int64),
                assume_unique=False,
            )
            if pool.size < 99:
                counts["insufficient_candidate_pool"] += 1
                continue
            candidates = [target] + candidate_rng.choice(
                pool, 99, replace=False
            ).tolist()
            examples.append((str(user), prefix, target, candidates))
        counts["candidate_examples"] = len(examples)
        return examples, counts

    def _exported_predictor(self):
        import importlib.util
        import os
        import sys

        # Inside the existing diagnostic block: avoid dynamic-class introspection.
        path = getattr(self.view, "predictor_path", None)
        if path is None:
            path = getattr(self.net, "predictor_path", None)
        if path is None:
            # diagnosis_tools.py belongs to this trial; never search other trials.
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "working", "predict.py"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Current trial's exported predict.py is not available"
            )
        spec = importlib.util.spec_from_file_location(
            "_bert4rec_diagnostic_export", path
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the current trial's exported predictor")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not all(
            hasattr(module, name)
            for name in ("predict", "_models", "_metas", "_device")
        ):
            raise ValueError(
                "exported predictor does not expose the supplied baseline interface"
            )
        required_routes = (
            "Movies_and_TV",
            "Industrial_and_Scientific",
            "Electronics",
            "CDs_and_Vinyl",
        )
        route = self.category
        if route not in required_routes:
            # Some host adapters rebuild ModelView from a checkpoint and omit
            # Python attributes. Resolve only by exact current-trial state identity.
            import pickle

            directory = os.path.dirname(os.path.abspath(path))
            matches = []
            actual = self.net.state_dict()
            for candidate in required_routes:
                checkpoint = os.path.join(directory, f"{candidate}_model.pt")
                metadata = os.path.join(directory, f"{candidate}_meta.pkl")
                if not os.path.isfile(checkpoint) or not os.path.isfile(metadata):
                    continue
                with open(metadata, "rb") as stream:
                    config = pickle.load(stream)
                if int(config["itemnum"]) != self.net.itemnum:
                    continue
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                identical = actual.keys() == state.keys() and all(
                    actual[name].shape == state[name].shape
                    and actual[name].dtype == state[name].dtype
                    and torch.equal(actual[name].detach().cpu(), state[name])
                    for name in actual
                )
                del state
                if identical:
                    matches.append(candidate)
            if len(matches) != 1:
                raise ValueError(
                    "Category identity unavailable: require one exact current-trial checkpoint match"
                )
            route = self.category = matches[0]
        if not callable(getattr(module, "_ensure_loaded", None)):
            raise ValueError("exported predictor has no checkpoint loading entry point")
        # Exercise the real writer-generated checkpoint loader for this route.
        # This reloads the same fitted checkpoint; it never fits or fuses models.
        module.DATASETS = [route]
        module._ensure_loaded()
        if set(module._models) != {route} or route not in module._metas:
            raise ValueError("exported loader must load exactly the requested category")
        exported = module._models[route].to(self.device)
        module._device = self.device
        exported.eval()
        if int(module._metas[route]["maxlen"]) != self.net.maxlen:
            raise ValueError("exported metadata disagrees with the diagnostic model")
        actual_state = self.net.state_dict()
        loaded_state = exported.state_dict()
        if actual_state.keys() != loaded_state.keys():
            raise ValueError(
                "checkpoint architecture differs from the diagnostic model"
            )
        for name, actual in actual_state.items():
            loaded = loaded_state[name]
            if (
                actual.shape != loaded.shape
                or actual.dtype != loaded.dtype
                or not torch.equal(actual, loaded)
            ):
                raise ValueError(
                    f"checkpoint differs at {name}; diagnose the selected checkpoint"
                )
        return module, route

    def _parity(self, examples):
        result = {
            "passed": False,
            "available": False,
            "atol": 1e-5,
            "rtol": 1e-4,
            "max_absolute_score_error": None,
            "ordered_ranking_agreement": None,
            "top10_agreement": None,
            "cases": 0,
            "coverage": {
                "short": False,
                "full_length": False,
                "truncated": False,
                "batched_vs_individual": False,
            },
            "checkpoint_serialization_verified": False,
            "weight_identity": "Unverified until the exported loader reloads the current category checkpoint and every state tensor matches exactly.",
            "external_label_invariance_tested": False,
        }
        try:
            if not examples:
                raise ValueError("no eligible training-prefix examples")
            module, route = self._exported_predictor()
            result["available"] = True
            result["checkpoint_serialization_verified"] = True
            result["weight_identity"] = (
                "All state tensors from the exported category checkpoint exactly "
                "match the actual diagnostic model, including dtype and shape."
            )
            cases = [(prefix, candidates) for _, prefix, _, candidates in examples]
            # Add a short case, then real full/truncated training-prefix cases.
            cases.append((examples[0][1][:2], examples[0][3]))
            for required in (self.net.maxlen - 1, self.net.maxlen + 1):
                for history in self._histories().values():
                    if len(history) >= required and required > 0:
                        prefix = [int(item) for item in history[:required]]
                        if all(1 <= item <= self.net.itemnum for item in prefix):
                            cases.append((prefix, examples[0][3]))
                            break
            errors, ordered, overlaps, passed = [], [], [], True
            paired_inputs, paired_positions, paired_scores = [], [], []
            for history, ids in cases:
                raw, sequence, positions = self._serving_tensors(history)
                candidates = torch.tensor(ids, dtype=torch.long, device=self.device)
                direct_features = self._explicit_forward(sequence, positions)
                view_features = self.view.log2feats(sequence.detach().cpu().numpy())
                if direct_features.shape != view_features.shape:
                    raise ValueError("ModelView returned an unexpected feature shape")
                visible = sequence.ne(0)
                passed &= bool(
                    torch.allclose(
                        direct_features[visible],
                        view_features[visible],
                        atol=1e-5,
                        rtol=1e-4,
                    )
                )
                # Compare the actual ModelView readout scores as well as features.
                direct = self._feature_scores(direct_features[0, -1], candidates)
                view_scores = self._feature_scores(view_features[0, -1], candidates)
                internal = self.net.score_candidates(raw, candidates).reshape(-1)
                external = torch.tensor(
                    module.predict(0, history, ids, dataset_name=route),
                    dtype=direct.dtype,
                    device=self.device,
                ).reshape(-1)
                reference_rank = self._ranking(direct, candidates)
                for other in (view_scores, internal, external):
                    finite = bool(torch.isfinite(other).all())
                    passed &= finite and bool(
                        torch.allclose(direct, other, atol=1e-5, rtol=1e-4)
                    )
                    if not finite:
                        raise ValueError("nonfinite parity scores")
                    errors.append(float((direct - other).abs().max()))
                    rank = self._ranking(other, candidates)
                    ordered.append(float(rank == reference_rank))
                    overlaps.append(
                        len(set(rank[:10]) & set(reference_rank[:10])) / 10.0
                    )
                result["coverage"]["short"] |= len(history) < self.net.maxlen - 1
                result["coverage"]["full_length"] |= len(history) == self.net.maxlen - 1
                result["coverage"]["truncated"] |= len(history) > self.net.maxlen - 1
                if len(paired_inputs) < 8:
                    paired_inputs.append(sequence)
                    paired_positions.append(positions)
                    paired_scores.append((direct, candidates))
            batch_features = self._explicit_forward(
                torch.cat(paired_inputs), torch.cat(paired_positions)
            )
            for index, (single, candidates) in enumerate(paired_scores):
                batched = self._feature_scores(batch_features[index, -1], candidates)
                if not torch.isfinite(batched).all():
                    raise ValueError("nonfinite batched parity scores")
                passed &= bool(torch.allclose(single, batched, atol=1e-5, rtol=1e-4))
                errors.append(float((single - batched).abs().max()))
                single_rank = self._ranking(single, candidates)
                batch_rank = self._ranking(batched, candidates)
                ordered.append(float(single_rank == batch_rank))
                overlaps.append(
                    len(set(single_rank[:10]) & set(batch_rank[:10])) / 10.0
                )
            result["coverage"]["batched_vs_individual"] = len(paired_inputs) > 1
            coverage_complete = all(result["coverage"].values())
            result.update(
                passed=bool(
                    passed
                    and coverage_complete
                    and result["checkpoint_serialization_verified"]
                    and all(value == 1.0 for value in ordered)
                ),
                coverage_complete=coverage_complete,
                cases=len(cases),
                max_absolute_score_error=max(errors),
                ordered_ranking_agreement=float(np.mean(ordered)),
                top10_agreement=float(np.mean(overlaps)),
            )
            if not coverage_complete:
                result["error"] = (
                    "Required parity coverage is incomplete; sensitivity is withheld."
                )
            elif not result["passed"]:
                result["error"] = (
                    "Feature, score, or ordered-ranking parity failed at fixed tolerances."
                )
        except Exception as exc:
            result["passed"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # Release the checkpoint reload after verification; it is never fused.
            if "module" in locals():
                module._models.clear()
        return result

    def _paired_sensitivity(self, examples, counts):
        rows = []
        for user, prefix, target, ids in examples:
            _, original, positions = self._serving_tensors(prefix)
            visible = torch.nonzero(original[0, :-1].ne(0), as_tuple=False).flatten()
            k = min(2, (visible.numel() - 1) // 2)
            if k < 1:
                counts["insufficient_visible_history"] += 1
                continue
            recent, older, swapped = (
                original.clone(),
                original.clone(),
                original.clone(),
            )
            recent[0, visible[-k:]] = 0
            older[0, visible[:k]] = 0
            left, right = int(visible[-2]), int(visible[-1])
            swapped[0, left], swapped[0, right] = original[0, right], original[0, left]
            candidates = torch.tensor(ids, dtype=torch.long, device=self.device)
            measurements = {}
            for name, sequence in (
                ("original", original),
                ("recent_deleted", recent),
                ("older_deleted", older),
                ("last_two_swapped", swapped),
            ):
                scores = self._score(sequence, positions, candidates)
                ranking = self._ranking(scores, candidates)
                rank = ranking.index(target) + 1
                measurements[name] = {
                    "target_score": float(scores[0]),
                    "target_rank": float(rank),
                    "reciprocal_rank": 1.0 / rank,
                    "diagnostic_hit10": float(rank <= 10),
                    "ranking": ranking,
                }
            baseline = measurements["original"]
            row = {
                "user": user,
                "visible_history": int(visible.numel()),
                "deleted_per_variant": k,
            }
            for name, values in measurements.items():
                if name == "original":
                    row[name] = {
                        key: value for key, value in values.items() if key != "ranking"
                    }
                    continue
                row[name] = {
                    key + "_delta": values[key] - baseline[key]
                    for key in (
                        "target_score",
                        "target_rank",
                        "reciprocal_rank",
                        "diagnostic_hit10",
                    )
                }
                row[name].update(
                    top10_overlap=len(
                        set(values["ranking"][:10]) & set(baseline["ranking"][:10])
                    )
                    / 10.0,
                    ordered_ranking_agreement=float(
                        values["ranking"] == baseline["ranking"]
                    ),
                )
            rows.append(row)
        counts["paired_users"] = len(rows)
        rng = np.random.default_rng(42)
        # Reuse paired-user resamples across variants for comparable intervals.
        indices = rng.integers(0, len(rows), size=(500, len(rows))) if rows else None
        summary = {}
        for variant in ("recent_deleted", "older_deleted", "last_two_swapped"):
            keys = (
                "target_score_delta",
                "target_rank_delta",
                "reciprocal_rank_delta",
                "diagnostic_hit10_delta",
                "top10_overlap",
                "ordered_ranking_agreement",
            )
            if not rows:
                summary[variant] = {
                    key: {"mean": None, "ci95": None, "users": 0} for key in keys
                }
                continue
            values = np.asarray(
                [[row[variant][key] for key in keys] for row in rows], dtype=np.float64
            )
            boot = values[indices].mean(axis=1)
            summary[variant] = {
                key: {
                    "mean": float(values[:, index].mean()),
                    "ci95": np.percentile(boot[:, index], [2.5, 97.5]).tolist(),
                    "users": len(rows),
                }
                for index, key in enumerate(keys)
            }
        return {
            "per_category": [
                {
                    "dataset_name": self.category,
                    "counts": counts,
                    "summary": summary,
                    "paired_user_outcomes": rows,
                }
            ],
            "four_category_macro": None,
            "four_category_macro_available": False,
            "macro_unavailable_reason": "This ModelView contains one category; four category results are required.",
        }

    def run_full_diagnosis(self, train_loader):
        import time

        previous_state = self.net.training
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        metrics = {
            "status": "diagnostic_failed",
            "history_sensitivity": None,
            "residual_patching": {
                "available": False,
                "reason": "Sampler and complete exported-serving parity must pass first.",
            },
        }
        definitions = {
            "embedding_collapse_score": "(Mean ordered off-diagonal cosine + 1)/2 over at most 2048 real item embeddings sampled with seed 42. PAD/MASK excluded. 0.5 means approximately zero mean cosine; capacity or quality is not established. Null if fewer than two or nonfinite observations.",
            "embedding_sample_count": "Number of sampled real item embeddings, not number of pairs.",
            "pos_neg_margin_mean": "In-sample mean positive-minus-negative logit over verified masked supervised positions. Higher means stronger sampled-pair separation.",
            "pos_neg_margin_p10": "In-sample nearest-rank 10th percentile of those margins, in logit units.",
            "pos_neg_neg_beats_pos_rate": "Fraction of verified supervised positions with negative-minus-positive logit > 0; lower is better.",
            "pos_neg_small_margin_rate": "Fraction of verified supervised positions with positive-minus-negative logit < 0.5.",
            "pos_neg_margin_count": "Number of verified supervised positions in at most ten loader batches; zero observations yield null statistics.",
            "pos_neg_margin_category_breakdown": "Unavailable: item categories are not supplied. Category lists remain empty.",
            "sampler_contract": "Checks PAD=0, MASK=itemnum+1, real supervised IDs, and the supplied all-MASK corruption policy. Failure blocks sensitivity. Does not establish sampler provenance from tensors alone.",
            "serving_parity": "Evaluation mode, float32, atol=1e-5/rtol=1e-4. The exported loader reloads exactly the current category checkpoint; every state tensor must exactly match the actual diagnostic model. Compare explicit fixed-position and ModelView features at visible positions, their MASK logits, internal appended-MASK scoring, exported predict, and batched/individual scoring. Scores must be finite and all ordered rankings identical. Rankings use descending score then ascending item ID; top10 agreement is intersection size/10. Short, full-length, truncated, and batched coverage are all required before sensitivity. External-label mutation remains explicitly untested; no hidden labels are accessed.",
            "history_sensitivity": "Training-prefix diagnostic, not host validation. Uniformly select up to 64 users before inspecting outcomes, one endpoint/user uniformly among endpoints with >=4 preceding interactions and a target absent from the complete prefix. Negatives are 99 unique real items absent from that complete prefix and distinct from target; later events never determine eligibility. Seed 42. Complete-prefix exclusions and candidates remain fixed across all variants. Serving truncates to maxlen-1 and appends MASK at index maxlen-1. Delete earliest/latest k=min(2,floor((visible_count-1)/2)) visible items into attention-masked PAD holes; preserve surviving absolute positions and MASK index, without backfill. Swap final two visible items. Repeated historical IDs are permitted. Unknown IDs are excluded and counted, not negative labels.",
            "paired_statistics": "Variant minus original: target_score_delta in logits; target_rank_delta in one-based positions (positive means deterioration); reciprocal_rank_delta uses 1/rank; diagnostic_hit10_delta uses indicator(rank<=10). top10_overlap is intersection/10; ordered_ranking_agreement compares complete ordered candidate rankings. Equal-user category means and 500 CPU paired-user bootstrap replicates, seed 42, percentile 95% intervals. Intervals cover sampled-user uncertainty only. Raw paired outcomes are retained for a four-category aggregation; that macro is unavailable in a single-category ModelView, never computed over fewer categories.",
            "elapsed_diagnostic_seconds": "Wall seconds including warmup, loading, probes, and CPU bootstrap, CUDA synchronized at both boundaries; all diagnostic GPU work consumes search budget.",
            "availability": "Null means unavailable; counts/exclusions and parity coverage must accompany interpretation. No ratings, review text, item categories/prices/stores, or simulator traits are measured.",
        }
        try:
            self.net.eval()
            with torch.no_grad():
                metrics["embedding_collapse_score"] = (
                    self.analyze_embeddings_collapse_cosine()
                )
                metrics["embedding_sample_count"] = self._embedding_count
                margin, breakdown = self.analyze_pos_neg_margin_with_category_breakdown(
                    train_loader
                )
                metrics.update(
                    {
                        "pos_neg_margin_mean": margin["mean"],
                        "pos_neg_margin_p10": margin["p10"],
                        "pos_neg_neg_beats_pos_rate": margin["neg_beats_pos_rate"],
                        "pos_neg_small_margin_rate": margin["small_margin_rate"],
                        "pos_neg_margin_count": margin["count"],
                        "pos_neg_margin_category_breakdown": breakdown,
                        "sampler_contract": self._contract,
                    }
                )
                examples, counts = self._examples()
                metrics["prefix_counts"] = counts
                if not self._contract["passed"]:
                    metrics["status"] = "sampler_contract_failed"
                else:
                    # Explicitly require float32 instead of silently widening tolerances.
                    if self.net.item_emb.weight.dtype != torch.float32:
                        raise ValueError(
                            "serving parity requires the predeclared float32 precision"
                        )
                    parity = self._parity(examples)
                    metrics["serving_parity"] = parity
                    if not parity["passed"]:
                        metrics["status"] = (
                            "serving_parity_failed"
                            if parity["available"]
                            else "serving_parity_unavailable"
                        )
                    else:
                        metrics["history_sensitivity"] = self._paired_sensitivity(
                            examples, counts
                        )
                        metrics["status"] = (
                            "measured"
                            if counts.get("paired_users", 0)
                            else "insufficient_history"
                        )
                        try:
                            metrics["residual_patching"] = _residual_patching_pilot(
                                self, examples, seconds=30.0
                            )
                        except Exception as exc:
                            metrics["residual_patching"] = {
                                "available": False,
                                "failure_stage": "residual_pilot",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
        except Exception as exc:
            metrics["status"] = "diagnostic_failed"
            metrics["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.net.train(previous_state)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["elapsed_diagnostic_seconds"] = time.perf_counter() - start
        definitions["residual_patching"] = (
            "Frozen two-layer pre-norm BERT4Rec intervention pilot, gated by sampler "
            "and serialized-checkpoint serving parity. Public training prefixes only; "
            "fixed 100 candidates, absolute positions, padding mask, and appended MASK. "
            "Two frequency/recency-matched non-PAD replacement rules at the most "
            "recent history position: uniform donor, then closest-frequency distinct "
            "donor. Exploration selects one site and sign per rule "
            "and direction; confirmation uses disjoint users. Delta means intervention "
            "minus its unpatched destination baseline. Primary fixed_contrast holds "
            "the strongest clean negative identity constant; switching_margin permits "
            "competitor changes. tenth_negative_margin measures the diagnostic HR@10 "
            "boundary. Rank is one-based; overlap/agreement reference the clean ranking. "
            "Identity, final-history structural-zero, and final-MASK restoration checks "
            "use atol=1e-5/rtol=1e-4. Missing unrelated donors block confirmation. "
            "Random-site controls match layer, recency band, component count, and "
            "activation-change norm, verified after floating-point addition. Descriptive "
            "magnitude quartiles use exploration only and are frozen for confirmation. "
            "Every registered residual hook must execute exactly once per forward. "
            "Raw control outcomes are retained. Equal-user "
            "means and 500 paired-user bootstrap percentile intervals are descriptive. "
            "Confirmatory sign tests exclude and count exact zeros. Global Holm "
            "adjustment remains unavailable until all four categories are aggregated; "
            "the reported 16*p bound is conservative Bonferroni, not Holm. Partial "
            "or timed-out runs never yield confirmatory p-values. No repairs are "
            "trained, fused, or compared by this diagnostic."
        )
        return {"metrics": metrics, "metric_definitions": definitions}

    ### <<< Self_EvolveRec-BLOCK-END
