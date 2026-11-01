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


